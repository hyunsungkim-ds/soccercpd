import os
from collections import Counter
from datetime import datetime, timedelta
from pprint import pprint
from typing import List, Union, Optional

import numpy as np
import pandas as pd
import rpy2.rinterface_lib.embedded as rembedded
import rpy2.robjects as robjects
import ruptures as rpt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay, distance_matrix
from sklearn.metrics import pairwise_distances
from sympy.combinatorics import Permutation
from sympy.interactive import init_printing

from src.config import *

init_printing(perm_cyclic=True, pretty_print=False)

def player_to_team(player_id) -> Optional[str]:
    """Return the team prefix ('home'/'away') of a player_id, or None for out_/goal_/NaN/non-str."""
    if isinstance(player_id, str):
        if player_id.startswith("home_"):
            return "home"
        if player_id.startswith("away_"):
            return "away"
    return None


def seconds_to_timestamp(total_seconds: float) -> str:
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{int(seconds):02d}{f'{seconds % 1:.2f}'[1:]}"


def timestamp_to_seconds(timestamp: str) -> float:
    minutes, seconds = timestamp.split(":")
    return float(minutes) * 60 + float(seconds)


def series_to_seconds(series: pd.Series) -> pd.Series:
    def _to_sec(val):
        if isinstance(val, (int, float, np.floating)) and not np.isnan(val):
            return float(val)
        return timestamp_to_seconds(val)

    return series.apply(_to_sec)


def _player_sort_key(s: str):
    team, num = s.split("_", 1)
    return (0 if team == "home" else 1, int(num))


def list_players(data: pd.DataFrame) -> List[str]:
    players = [c[:-2] for c in data.columns if c[:4] in ["home", "away"] and c.endswith("_x")]
    return sorted(players, key=_player_sort_key)


def wide_to_long(tracking: pd.DataFrame) -> pd.DataFrame:
    xy_list = []
    players = [c[:3] for c in tracking.columns if c[3:] == "_x"]

    for p in players:
        cols = ["datetime", "period_id", "timestamp", "player_seg", f"{p}_x", f"{p}_y"]
        player_xy = tracking[cols].copy().rename(columns={f"{p}_x": "x", f"{p}_y": "y"})
        player_xy["player_id"] = p
        xy_list.append(player_xy)

    return pd.concat(xy_list).set_index("datetime")


def label_player_segs(tracking: pd.DataFrame, fps: int = 25, min_gap: int = 750) -> pd.DataFrame:
    """Add per-team player-segment columns (`home_player_seg`/`away_player_seg`) to a wide tracking frame.

    A new player segment begins at each period start and at each substitution. Every player of a
    team shares the same player-segment timeline, which becomes ``player_seg`` in the SoccerCPD input.

    Roster changes (each player's first/last valid frame) that fall within ``min_gap`` frames of one
    another are snapped to a single boundary, and each player's data outside its snapped presence
    window is set to NaN. This guarantees every player segment has a *constant* roster (no 1-3s
    micro-segments from staggered substitutions, and no segment with more than the on-pitch players),
    which SoccerCPD relies on. Frame ranges use the ``frame_id`` column, not the DataFrame index.
    """
    tracking = tracking.copy()
    players = list_players(tracking)
    period_starts = sorted(int(f) for f in tracking.groupby("period_id")["frame_id"].first().values)
    frame_ids = tracking["frame_id"].to_numpy()

    for team in ["home", "away"]:
        team_players = [p for p in players if p.split("_")[0] == team]

        # First/last valid frame per player.
        records = {}
        for p in team_players:
            valid = tracking.loc[tracking[f"{p}_x"].notna(), "frame_id"]
            if not valid.empty:
                records[p] = (int(valid.iloc[0]), int(valid.iloc[-1]))

        # Candidate change frames: period starts (hard) + each player's in-frame and out-frame+1.
        raw_changes = set(period_starts)
        for in_frame, out_frame in records.values():
            raw_changes.add(in_frame)
            raw_changes.add(out_frame + 1)

        # Snap near-simultaneous changes to a single boundary; align it to a whole second (a multiple
        # of fps) so player_seg boundaries land on the 1s grid RoleRep resamples onto -- otherwise a
        # 1s bin straddling a substitution would contain both the outgoing and incoming player.
        # Period starts and the final change (match end) are kept exact so rounding never orphans
        # the trailing frames into an empty player segment.
        last_change = max(raw_changes)
        snap = {}
        anchor = None
        for f in sorted(raw_changes):
            if f in period_starts or f == last_change:
                anchor = f
            elif anchor is None or f - anchor >= min_gap:
                anchor = int(round(f / fps)) * fps
            snap[f] = anchor

        boundaries = np.array(sorted(set(snap.values())))

        # Assign a player_seg to every frame based on the snapped boundaries.
        player_seg = np.zeros(len(tracking), dtype=int)
        edges = np.append(boundaries, frame_ids.max() + 1)
        for i in range(len(edges) - 1):
            player_seg[(frame_ids >= edges[i]) & (frame_ids < edges[i + 1])] = i + 1
        tracking[f"{team}_player_seg"] = player_seg

        # Snap each player's presence window to boundaries: interpolate internal dropouts and NaN
        # out everything outside the window, so every frame of a player segment has a constant roster.
        for p, (in_frame, out_frame) in records.items():
            snap_in, snap_out = snap[in_frame], snap[out_frame + 1] - 1
            player_cols = [c for c in tracking.columns if c.rsplit("_", 1)[0] == p]
            if snap_out < snap_in:
                tracking[player_cols] = np.nan
                continue
            inside = (frame_ids >= snap_in) & (frame_ids <= snap_out)
            tracking.loc[inside, player_cols] = tracking.loc[inside, player_cols].interpolate(limit_direction="both")
            tracking.loc[~inside, player_cols] = np.nan

    return tracking


def aggregate_player_segs(data: pd.DataFrame) -> pd.DataFrame:
    if "datetime" not in data.columns:
        data = data.reset_index().rename(columns={"index": "datetime"})

    grouper = data.groupby("player_seg")
    freq = round(data["timestamp"].iloc[1] - data["timestamp"].iloc[0], 3)

    periods = grouper["period_id"].first()
    start_dts = grouper["datetime"].first().rename("start_dt")
    end_dts = grouper["datetime"].last().rename("end_dt") + timedelta(seconds=freq)
    player_segs = pd.concat([periods, start_dts, end_dts], axis=1)

    player_segs["players"] = None
    for i in player_segs.index:
        pp_data: pd.DataFrame = data[data["player_seg"] == i]
        player_segs.at[i, "players"] = pp_data.groupby("player_id")["x"].first().dropna().index.tolist()

    n_players = player_segs["players"].apply(len)
    n_players = pd.concat([player_segs["period_id"], n_players], axis=1)
    player_segs["subperiod_id"] = (n_players.diff().fillna(1) != 0).any(axis=1).astype(int).cumsum()

    return player_segs


def _infer_fps(data: pd.DataFrame) -> int:
    if "timestamp" in data.columns:
        steps = data.drop_duplicates("datetime").groupby("period_id")["timestamp"].diff()
        step = steps[steps > 0].median()
        if step and step > 0:
            return int(round(1.0 / step))
    return 25


def smooth_possession(data: pd.DataFrame, min_poss_sec: float = 3.0, fps: int = 25) -> pd.DataFrame:
    """Reassign short `ball_owning_team_id` spells to the opponent so possession reflects sustained control.

    Expects an alive-only long stream (frame-level `ball_owning_team_id` shared across players). Per
    `period_id`, a spell is a maximal run of the same owner over the alive frames; any spell whose
    in-play duration (n_frames / fps) is below `min_possession_sec` is flipped to the opponent. Flips
    are repeated (shortest-first) until no spell is below the threshold, so brief blips get absorbed
    into the surrounding possession. Only `ball_owning_team_id` is modified.
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
    """Reassign compact, contiguous `timestamp`/`datetime` to a (possession-)filtered stream.

    Filtering leaves large real-time gaps that break freq inference, duration accounting, and CPD
    frame-count gates. This rebuilds a dense per-period frame grid (spacing 1/fps) and stacks periods
    end-to-end for a monotonic `datetime`, so downstream (RoleRep resampling, `aggregate_player_segs`,
    MIN_PERIOD_DUR) behaves as on a continuous stream. All players share the same frame grid (the
    possession filter is frame-level), so pivots stay aligned; `player_seg`/`period_id`/coords are kept.

    Returns (retimelined_stream, frame_map) where frame_map maps the compact grid back to the original
    `datetime`/`timestamp` (for reporting CPD change-points on the real match clock).
    """
    stream = stream.copy()
    dt = 1.0 / fps
    frames = stream.drop_duplicates("datetime")[["period_id", "player_seg", "datetime", "timestamp"]]
    frames = frames.sort_values("datetime").reset_index(drop=True)

    # Lay the retained frames end-to-end at 1/fps spacing, but restart each player_seg (substitution
    # segment) at a whole second. This keeps player_seg boundaries on the 1s grid RoleRep resamples
    # onto -- otherwise a compact 1s bin could straddle a substitution and pick up 11 players.
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
) -> dict:
    """Split the SoccerCPD long input into per-team (and optionally per-phase) streams.

    Returns ``{(home_away, phase): stream}``. SoccerCPD is applied to each stream by the caller.
    - by_possession=False: one stream per team, phase="all" (whole-team, current behavior).
    - by_possession=True: keep alive frames, smooth short possessions (`smooth_possession`), then per
      team split into attack (`ball_owning_team_id == team`) / defend (`!= team`); each stream is
      re-timelined. Streams with fewer than `min_frames` frames are skipped (logged).
    """
    streams = {}
    if not by_possession:
        for team in ["home", "away"]:
            streams[(team, "all")] = input_data[input_data["home_away"] == team].copy()
        return streams

    required = {"ball_state", "ball_owning_team_id"}
    if not required.issubset(input_data.columns):
        raise ValueError(
            f"by_possession=True requires columns {required}. "
            f"Re-run MatchData.to_soccercpd_input(carry_possession=True)."
        )

    fps = _infer_fps(input_data)
    alive_data = input_data[input_data["ball_state"] == "alive"].copy()
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


# Apply Delaunay triangulation to the given player coordinates to obtain the role-adjacency matrix
def delaunay_adj_mat(coords):
    tri_pts = Delaunay(coords).simplices
    edges = np.concatenate((tri_pts[:, :2], tri_pts[:, 1:], tri_pts[:, ::2]), axis=0)
    adj_mat = np.zeros((coords.shape[0], coords.shape[0]))
    adj_mat[edges[:, 0], edges[:, 1]] = 1
    return np.clip(adj_mat + adj_mat.T, 0, 1)


# Hamming distance between two permutations of the same shape
def hamming_dist(perm1, perm2):
    return (perm1 != perm2).astype(int).sum()


# Manhattan distance between two matrices of the same shape
def manhattan_dist(mat1, mat2):
    return np.abs(mat1 - mat2).sum()


def most_common(player_roles: pd.DataFrame):
    try:
        counter = Counter(player_roles[player_roles.notna()])
        return counter.most_common(1)[0][0]
    except IndexError:
        return np.nan


def compute_delaunay_dists(form1: pd.Series, form2: pd.Series) -> float:
    xy_idx = [c for c in form1.index if c[0] in ["x", "y"]]
    form1_node_xy = form1[xy_idx].dropna().astype(float).values.reshape(-1, 2)
    form2_node_xy = form2[xy_idx].dropna().astype(float).values.reshape(-1, 2)

    cost_mat = distance_matrix(form1_node_xy, form2_node_xy)
    row_idx, col_idx = linear_sum_assignment(cost_mat)

    form1_adj_mat = form1["adj_mat"][row_idx][:, row_idx]
    form2_adj_mat = form2["adj_mat"][col_idx][:, col_idx]
    return np.abs(form1_adj_mat - form2_adj_mat).sum()


def compute_switch_rate(moment_role_df: pd.DataFrame) -> pd.DataFrame:
    hamming = hamming_dist(moment_role_df["role"], moment_role_df["base_role"])
    moment_role_df["switch_rate"] = hamming / len(moment_role_df)
    return moment_role_df


def seconds_to_time_str(x: float) -> str:
    minutes = int(x // 60)
    seconds = int(x % 60)
    return f"{minutes:02d}:{seconds:02d}"


def ints_to_range_str(nums: List[int]) -> str:
    if not nums:
        return ""

    ranges = []
    start = nums[0]
    end = nums[0]

    for i in range(1, len(nums)):
        if nums[i] == end + 1:
            end = nums[i]
        else:
            if end > start:
                ranges.append(f"{start}-{end}")
            else:
                ranges.extend(map(str, range(start, end + 1)))
            start = nums[i]
            end = nums[i]

    if end > start:
        ranges.append(f"{start}-{end}")
    else:
        ranges.extend(map(str, range(start, end + 1)))

    return ",".join(ranges)


def complete_perm(perm: Union[pd.Series, dict], role_set: set) -> Union[pd.Series, dict]:
    if isinstance(perm, pd.Series) and perm.isnull().sum():
        return perm.fillna(list(role_set - set(perm.dropna()))[0])
    elif isinstance(perm, dict) and (0 in perm.values() or np.nan in perm.values()):
        return {k: v if v in role_set else list(role_set - set(perm.values()))[0] for k, v in perm.items()}
    else:
        return perm


def decompose_perm_to_cycles(perm: pd.Series, labels: dict = None) -> list:
    if perm["switch_rate"] > 0.6:
        return []

    perm_list = [0] + [r for (l, r) in sorted([t for t in perm[:-1] if str(t) != "nan"])]
    if len(perm_list) < 11:
        return []

    p = Permutation(perm_list)
    perm_str = str(p)

    ret = []
    cycles_str = perm_str.split(")")
    for c in cycles_str[:-1]:
        c = c.replace("(", "")
        ret.append(c.split(" "))

    if labels is None:
        return [[int(r) for r in c] for c in ret if len(c) > 1]
    else:
        return [[labels[int(r)] for r in c] for c in ret if len(c) > 1]


def detect_change_times(
    input_seq: pd.DataFrame,
    sub_dts: pd.Series,
    mode="form",
    method="gseg_avg",
    max_pval=MAX_PVAL,
    min_pdur=MIN_PERIOD_DUR,
    min_fdist=MIN_FORM_DIST,
) -> List[datetime]:
    # if mode == "form" (FormCPD), the input is a sequence of role-adjacency matrices
    # if mode == "role" (RoleCPD), the input a sequence of role permutations

    start_time = input_seq.index[0].time()
    end_time = input_seq.index[-1].time()

    if (mode == "role") or ("gseg" in method):
        metric = manhattan_dist if mode == "form" else hamming_dist
        dists = pd.DataFrame(pairwise_distances(input_seq.drop_duplicates(), metric=metric))

        # save the input sequence and the pairwise distances so that we can use them in the R script below
        if not os.path.exists(DIR_TEMP_DATA):
            os.mkdir(DIR_TEMP_DATA)
        input_seq.to_csv(f"{DIR_TEMP_DATA}/temp_seq.csv", index=False)
        dists.to_csv(f"{DIR_TEMP_DATA}/temp_dists.csv", index=False)

        try:
            print(f"Applying g-segmentation to the sequence between {start_time} and {end_time}...")

            if mode == "form":
                gseg_type = method.split("_")[1][0]
            else:
                gseg_type = method.split("_")[1][0]

            # run the R function "gseg1_discrete" to find a change-point
            # rpackages.importr("gSeg", lib_loc=rpackages.importr("base")._libPaths()[0])
            robjects.r(
                f"""
                dir = '{DIR_TEMP_DATA}'
                seq_path = paste(dir, 'temp_seq.csv', sep='/')
                seq = read.csv(seq_path)
                dists_path = paste(dir, 'temp_dists.csv', sep='/')
                dists = read.csv(dists_path)
                n = dim(seq)[1]
                edge_mat = nnl(dists, 1)
                seq_str = do.call(paste, seq)
                ids = match(seq_str, unique(seq_str))
                output = gseg1_discrete(n, edge_mat, ids, statistics='generalized', n0=0.1*n, n1=0.9*n)
                chg_idx = output$scanZ$generalized$tauhat_{gseg_type}
                pval = output$pval.appr$generalized_{gseg_type}
                """
            )

        except rembedded.RRuntimeError:
            return []

        # check whether the detected change-point is significant, using the following three conditions
        # condition (1): The p-value of the scan statistic must be less than 0.1
        if robjects.r["pval"][0] >= max_pval:
            print("Change-point insignificant: The p-value is not small enough.\n")
            return []
        else:
            chg_idx = robjects.r["chg_idx"][0]

    elif "kernel" in method:
        print(f"Applying kernel-based CPD to the sequence between {start_time} and {end_time}...")
        kernel_type = method.split("_")[1]
        algo = rpt.Binseg(model=kernel_type).fit(input_seq.values)
        chg_idx = algo.predict(n_bkps=1)[0]

    elif "rank" in method:
        print(f"Applying rank-based CPD to the sequence between {start_time} and {end_time}...")
        algo = rpt.Binseg(model="rank").fit(input_seq.values)
        chg_idx = algo.predict(n_bkps=1)[0]

    else:
        raise ValueError("Invalid formcpd_type.")

    chg_dt = input_seq.index[chg_idx]

    # fine-tune chg_dt to the closest substitution time (if exists)
    if len(sub_dts) > 0:
        tds = np.abs(sub_dts - chg_dt.to_pydatetime())
        if tds.min().total_seconds() <= 180:
            chg_dt = sub_dts[tds.argmin()]

    # condition (2): Both of the segments must last for at least five minutes
    seq1 = input_seq[:chg_dt]
    seq2 = input_seq[chg_dt:]
    if (len(seq1) < min_pdur) or (len(seq2) < min_pdur):
        print("Change-point insignificant: One of the periods has not enough duration.\n")
        return []

    if mode == "form":
        # condition (3) for FormCPD: The respective mean role-adjacency matrices
        # from the segments before and after chg_dt are far enough from each other
        form1_adj_mat = seq1.mean(axis=0).values
        form2_adj_mat = seq2.mean(axis=0).values
        if manhattan_dist(form1_adj_mat, form2_adj_mat) < min_fdist:
            print("Change-point insignificant: The formation is not changed.\n")
            return []
        else:
            # if significant, recursively detect another change-points before and after chg_dt
            print(f"A significant fine-tuned change-point at {chg_dt.time()}.\n")
            prev_chg_dts = detect_change_times(seq1, sub_dts)
            next_chg_dts = detect_change_times(seq2, sub_dts)
            return prev_chg_dts + [chg_dt] + next_chg_dts

    elif mode == "role":
        # condition (3) for RoleCPD: The most frequent permutations differ between before and after chg_dt
        seq1_str = seq1.apply(lambda row: np.array2string(row.values), axis=1)
        seq2_str = seq2.apply(lambda row: np.array2string(row.values), axis=1)
        counter1 = Counter(seq1_str)
        counter2 = Counter(seq2_str)
        if counter1.most_common(1)[0][0] == counter2.most_common(1)[0][0]:
            print("Change-point insignificant: The most frequent permutation is not changed.\n")
            return []
        else:
            # if significant, recursively detect another change-points before and after chg_dt
            print(f"A significant fine-tuned change-point at {chg_dt.time()}.")
            print(f"- Frequent permutations before {chg_dt.time()}:")
            pprint(counter1.most_common(5))
            print(f"- Frequent permutations after {chg_dt.time()}:")
            pprint(counter2.most_common(5))
            print()
            prev_chg_dts = detect_change_times(seq1, sub_dts)
            next_chg_dts = detect_change_times(seq2, sub_dts)
            return prev_chg_dts + [chg_dt] + next_chg_dts

    else:
        raise ValueError("Invalid mode")
