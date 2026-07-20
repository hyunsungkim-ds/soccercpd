"""Run SoccerCPD over per-team (and optionally attacking/defending) streams of one match.

This is the orchestration layer above :class:`~src.soccercpd.SoccerCPD`: it splits a match's
SoccerCPD long input into streams (whole-team, or attacking/defending per team), runs SoccerCPD on each,
and rewrites the resulting segment times onto the real match clock.
"""

from datetime import datetime
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

from src.soccercpd import SoccerCPD
from src.utils import _infer_fps


def smooth_possession(data: pd.DataFrame, min_poss_sec: float = 3.0, fps: int = 25) -> pd.DataFrame:
    """Smooth short `ball_owning_team_id` spells so possession reflects sustained control.

    Expects an alive-only long stream (frame-level `ball_owning_team_id` shared across players).
    Any spell whose in-play duration is below `min_poss_sec` is absorbed into the surrounding possession.
    """
    if not min_poss_sec:
        return data

    data = data.copy()
    min_frames = int(round(min_poss_sec * fps))
    frames = data.drop_duplicates("datetime")[["datetime", "period_id", "ball_owning_team_id"]]
    frames = frames.sort_values("datetime").reset_index(drop=True)

    smoothed = {}
    for period in frames["period_id"].unique():
        owners = frames.loc[frames["period_id"] == period, ["datetime", "ball_owning_team_id"]]

        # Run-length encode the owner sequence, then flip short runs to the opponent until none remain.
        runs = []  # [owner, length]
        for o in owners["ball_owning_team_id"].to_numpy():
            if runs and runs[-1][0] == o:
                runs[-1][1] += 1
            else:
                runs.append([o, 1])

        while True:
            short = [k for k, r in enumerate(runs) if r[1] < min_frames and r[0] in ("home", "away")]
            if not short:
                break
            k = min(short, key=lambda k: runs[k][1])
            runs[k][0] = "away" if runs[k][0] == "home" else "home"
            merged = []
            for owner, length in runs:
                if merged and merged[-1][0] == owner:
                    merged[-1][1] += length
                else:
                    merged.append([owner, length])
            runs = merged

        seq = [owner for owner, length in runs for _ in range(length)]
        for dt, owner in zip(owners["datetime"].to_numpy(), seq):
            smoothed[dt] = owner

    data["ball_owning_team_id"] = data["datetime"].map(smoothed)
    return data


def retimeline_stream(stream: pd.DataFrame, fps: int = 25):
    """Rebuild a compact, gap-free timeline for a (phase-)filtered stream.

    Since attacking/defending phase filtering leaves time gaps,
    this function relays the retained frames on a dense 1/fps grid so that downstream code sees a continuous stream.
    Returns (retimelined_stream, frame_map), while the original clock is preserved in the `match_ts` column.
    """
    stream = stream.copy()
    dt = 1.0 / fps
    frames = stream.drop_duplicates("datetime")[["period_id", "player_seg", "datetime", "timestamp"]]
    frames = frames.sort_values("datetime").reset_index(drop=True)

    # Lay the retained frames end-to-end at 1/fps spacing, but restart each player_seg at a whole second.
    # This keeps player_seg boundaries on the 1s grid RoleRep resamples onto;
    # otherwise a compact 1s bin could straddle a substitution and pick up 11 players.
    new_ts = np.empty(len(frames))
    cum = 0.0
    prev_seg = None
    for idx, seg in enumerate(frames["player_seg"].to_numpy()):
        if seg != prev_seg:
            cum = float(np.ceil(cum)) if idx > 0 else 0.0
            prev_seg = seg
        new_ts[idx] = round(cum, 3)
        cum += dt

    base = datetime(2024, 1, 1, 1)
    frames["new_timestamp"] = new_ts
    frames["new_datetime"] = base + pd.to_timedelta(new_ts, unit="s")

    frame_map = frames.rename(columns={"datetime": "orig_datetime", "timestamp": "orig_timestamp"})
    key = frames[["period_id", "datetime", "new_timestamp", "new_datetime"]]
    merged = stream.merge(key, on=["period_id", "datetime"], how="left")
    merged["match_ts"] = merged["timestamp"]  # preserve real per-period match seconds before compacting
    merged["timestamp"] = merged["new_timestamp"]
    merged["datetime"] = merged["new_datetime"]
    merged = merged.drop(columns=["new_timestamp", "new_datetime"])
    return merged, frame_map


def prepare_cpd_streams(
    input_data: pd.DataFrame,
    by_possession: bool = False,
    retimeline: bool = True,
    min_poss_sec: float = 3.0,
    min_frames: int = 750,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Split the SoccerCPD long input into per-team (optionally attacking/defending) streams.

    by_possession=False: one stream per team (phase "all").
    by_possession=True: split alive frames into re-timelined attacking/defending streams per team.
    Returns {(home_away, phase): stream}.
    """
    streams = {}
    if not by_possession:
        for team in ["home", "away"]:
            streams[(team, "all")] = input_data[input_data["home_away"] == team].copy()
        return streams

    required = {"ball_state", "ball_owning_team_id"}
    if not required.issubset(input_data.columns):
        raise ValueError(f"by_possession=True requires columns {required}.")

    fps = _infer_fps(input_data)
    alive_data = input_data[input_data["ball_state"] == "alive"].copy()
    # Drop frames with no resolved owner (TRACAB leaves `ball_owning_team_id` None on some alive frames).
    # Only home/away possessions feed the attacking/defending split and smoothing.
    alive_data = alive_data[alive_data["ball_owning_team_id"].isin(["home", "away"])].copy()
    if min_poss_sec:
        alive_data = smooth_possession(alive_data, min_poss_sec=min_poss_sec, fps=fps)

    for team in ["home", "away"]:
        team_data = alive_data[alive_data["home_away"] == team]
        attack_data = team_data[team_data["ball_owning_team_id"] == team]
        defend_data = team_data[team_data["ball_owning_team_id"] != team]
        phase_data = {"attack": attack_data, "defend": defend_data}

        for phase, stream in phase_data.items():
            n_frames = stream["datetime"].nunique()
            if n_frames < min_frames:
                print(f"[prepare_cpd_streams] skip ({team}, {phase}): {n_frames} frames < {min_frames}")
                continue
            if retimeline:
                stream, _ = retimeline_stream(stream.copy(), fps=fps)
            streams[(team, phase)] = stream

    return streams


def _replace_segment_times(segs: pd.DataFrame, seg_col: str, frame_times: pd.DataFrame, real_col: str):
    """Swap a segment table's synthetic `start_dt`/`end_dt` for real match seconds `start_ts`/`end_ts`."""
    if segs is None or seg_col not in segs.columns or "start_dt" not in segs.columns:
        return segs
    agg = (
        frame_times.dropna(subset=[seg_col]).groupby(seg_col, observed=True)[real_col].agg(start_ts="min", end_ts="max")
    )
    pos = segs.columns.get_loc("start_dt")
    segs = segs.copy()
    segs["start_ts"] = segs[seg_col].map(agg["start_ts"]).round(3)
    segs["end_ts"] = segs[seg_col].map(agg["end_ts"]).round(3)
    remaining = [c for c in segs.columns if c not in ("start_dt", "end_dt", "start_ts", "end_ts")]
    return segs[remaining[:pos] + ["start_ts", "end_ts"] + remaining[pos:]]


def _fill_segment_gaps(segs: pd.DataFrame, seg_col: str, period_ends: dict, period_subs: dict):
    """Stretch segment `start_ts`/`end_ts` so they tile each period with no gaps.

    Per period, the first segment starts at 0, each end extends to the next start (snapping to a
    substitution time falling in the gap), the last segment ends at the period end.
    """
    if segs is None or "period_id" not in segs.columns or "start_ts" not in segs.columns:
        return segs
    segs = segs.sort_values(seg_col).reset_index(drop=True)
    for period, grp in segs.groupby("period_id", sort=True):
        idx = grp.index.tolist()
        subs = sorted(period_subs.get(period, []))
        segs.loc[idx[0], "start_ts"] = 0.0
        for a, b in zip(idx[:-1], idx[1:]):
            a_end, b_start = segs.loc[a, "end_ts"], segs.loc[b, "start_ts"]
            in_gap = [s for s in subs if a_end < s < b_start]
            boundary = round(in_gap[0] if in_gap else b_start, 3)
            segs.loc[a, "end_ts"] = boundary
            segs.loc[b, "start_ts"] = boundary
        if period in period_ends:
            segs.loc[idx[-1], "end_ts"] = round(period_ends[period], 3)
    return segs


def finalize_segment_times(
    cpd_results: Union[SoccerCPD, Dict[Tuple[str, str], SoccerCPD]], match_data: pd.DataFrame
) -> Union[SoccerCPD, Dict[Tuple[str, str], SoccerCPD]]:
    """Rewrite each SoccerCPD's segment times onto the real match clock; return `cpd_results` (mutated).

    For each SoccerCPD in `cpd_results`, swaps `start_dt`/`end_dt` for real `start_ts`/`end_ts`
    and tiles per-period gaps across form/role/player segments and role_assign.
    """
    for cpd in cpd_results.values() if isinstance(cpd_results, dict) else [cpd_results]:
        data = cpd.data
        real_col = "match_ts" if "match_ts" in data.columns else "timestamp"
        if real_col not in data.columns:
            continue
        frame_times = data.reset_index()

        # Period ends and per-period substitution times for this stream's team, on the same clock as
        # `real_col` (the original per-period `timestamp`).
        period_ends, period_subs = {}, {}
        md = match_data
        if md is not None and "datetime" not in md.columns:
            md = md.reset_index()
        if md is not None and "timestamp" in md.columns:
            if "home_away" in md.columns and "home_away" in frame_times.columns:
                md = md[md["home_away"] == frame_times["home_away"].iloc[0]]
            fr = md.drop_duplicates("datetime").sort_values(["period_id", "timestamp"])
            period_ends = fr.groupby("period_id")["timestamp"].max().to_dict()
            if "player_seg" in fr.columns:
                for period, grp in fr.groupby("period_id"):
                    changed = grp["player_seg"].ne(grp["player_seg"].shift())
                    period_subs[period] = grp.loc[changed, "timestamp"].tolist()[1:]  # drop period start

        # form_segs / role_segs carry seg_col as a column; player_segs carries it as the index, so
        # round-trip it through a column for the shared helpers.
        for attr, seg_col, indexed in [
            ("form_segs", "form_seg", False),
            ("role_segs", "role_seg", False),
            ("player_segs", "player_seg", True),
        ]:
            segs = getattr(cpd, attr, None)
            if segs is None or (indexed and "start_dt" not in segs.columns):
                continue
            if indexed:
                segs = segs.reset_index()
            segs = _replace_segment_times(segs, seg_col, frame_times, real_col)
            segs = _fill_segment_gaps(segs, seg_col, period_ends, period_subs)
            setattr(cpd, attr, segs.set_index(seg_col) if indexed else segs)

        # Carry the gap-filled role-segment times onto the per-player role assignment table.
        role_assign = getattr(cpd, "role_assign", None)
        if role_assign is not None and "start_dt" in role_assign.columns:
            role_assign = _replace_segment_times(role_assign, "role_seg", frame_times, real_col)
            filled = cpd.role_segs.set_index("role_seg")[["start_ts", "end_ts"]]
            role_assign["start_ts"] = role_assign["role_seg"].map(filled["start_ts"])
            role_assign["end_ts"] = role_assign["role_seg"].map(filled["end_ts"])
            cpd.role_assign = role_assign

    return cpd_results


def run_cpd(
    input_data: pd.DataFrame,
    by_possession: bool = False,
    formcpd_method: str = "gseg_avg",
    rolecpd_method: str = "gseg_avg",
    min_poss_sec: float = 3.0,
    min_frames: int = 750,
    verbose: bool = True,
    **run_kwargs,
) -> Dict[Tuple[str, str], SoccerCPD]:
    """Run the full pipeline for one match: split into streams, run SoccerCPD, finalize times.

    `by_possession` selects whole-team vs attacking/defending streams;
    `run_kwargs` pass through to `SoccerCPD.run` (e.g. `freq`, `max_sr`, `min_seg_dur`).
    Returns {(home_away, phase): SoccerCPD} with segment times finalized (`start_ts`/`end_ts`).
    """
    streams = prepare_cpd_streams(input_data, by_possession, min_poss_sec=min_poss_sec, min_frames=min_frames)

    results: Dict[Tuple[str, str], SoccerCPD] = {}
    for (team, phase), stream in streams.items():
        if verbose:
            label = f"{team.title()} team" if phase == "all" else f"{team.title()} {phase}"
            print(f"\n{'=' * 22} {label} {'=' * 22}")
        cpd = SoccerCPD(stream, formcpd_method=formcpd_method, rolecpd_method=rolecpd_method)
        cpd.run(**run_kwargs)
        results[(team, phase)] = cpd

    return finalize_segment_times(results, match_data=input_data)
