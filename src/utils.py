import os
import re
from collections import Counter
from datetime import datetime, timedelta
from pprint import pprint
from typing import List, Optional, Tuple, Union

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
from tqdm import tqdm

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
    min_seg_dur=MIN_SEG_DUR,
    min_form_dist=MIN_FORM_DIST,
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
            robjects.r(f"""
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
                """)

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
    if (len(seq1) < min_seg_dur) or (len(seq2) < min_seg_dur):
        print("Change-point insignificant: One of the periods has not enough duration.\n")
        return []

    if mode == "form":
        # condition (3) for FormCPD: The respective mean role-adjacency matrices
        # from the segments before and after chg_dt are far enough from each other
        form1_adj_mat = seq1.mean(axis=0).values
        form2_adj_mat = seq2.mean(axis=0).values
        if manhattan_dist(form1_adj_mat, form2_adj_mat) < min_form_dist:
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


def find_active_players(traces: pd.DataFrame, frame: int = None, team: str = None, include_goals=False) -> dict:
    if pd.isna(frame):
        snapshot = traces.dropna(how="all", axis=1).copy()
    else:
        snapshot = traces.loc[frame:frame].dropna(how="all", axis=1).copy()

    if include_goals:
        home_players = [c[:-2] for c in snapshot.columns if re.match(r"home_.*_x", c)]
        away_players = [c[:-2] for c in snapshot.columns if re.match(r"away_.*_x", c)]
    else:
        home_players = [c[:-2] for c in snapshot.columns if re.match(r"home_\d+_x", c)]
        away_players = [c[:-2] for c in snapshot.columns if re.match(r"away_\d+_x", c)]

    if not pd.isna(frame):
        team = team or traces.at[frame, "ball_owning_home_away"]
    else:
        team = team or "home"

    if team == "home":
        players = [home_players, away_players]
    else:
        players = [away_players, home_players]

    return players


def label_frames_and_episodes(
    tracking: pd.DataFrame, events: pd.DataFrame = None, fps: float = 25.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tracking = tracking.copy().sort_values(["period_id", "timestamp"], ignore_index=True)

    if "frame_id" not in tracking.columns:
        tracking["frame_id"] = (tracking["timestamp"] * fps).round().astype(int)
        n_prev_frames = 0

        for i in tracking["period_id"].unique():
            period_tracking = tracking[tracking["period_id"] == i]
            tracking.loc[period_tracking.index, "frame_id"] += n_prev_frames
            n_prev_frames += len(period_tracking)

    if "episode_id" not in tracking.columns:
        tracking["episode_id"] = 0
        n_prev_episodes = 0

        for i in tracking["period_id"].unique():
            period_tracking = tracking[tracking["period_id"] == i].copy()
            alive_tracking = period_tracking[period_tracking["ball_state"] == "alive"].copy()

            frame_diffs = np.diff(alive_tracking["frame_id"].values, prepend=-5)
            period_episode_ids = (frame_diffs >= 5).astype(int).cumsum() + n_prev_episodes
            tracking.loc[alive_tracking.index, "episode_id"] = period_episode_ids

            n_prev_episodes = period_episode_ids.max()

    tracking = tracking.set_index("frame_id")

    if events is not None and "episode_id" not in events.columns:
        events = events.copy()
        events["episode_id"] = 0

        for i in events.index:
            frame_id = events.at[i, "frame_id"]
            if not pd.isna(frame_id):
                events.at[i, "episode_id"] = tracking.at[frame_id, "episode_id"]

    return tracking.reset_index(), events


def summarize_playing_times(tracking: pd.DataFrame) -> pd.DataFrame:
    if "frame_id" in tracking.columns:
        tracking = tracking.copy().set_index("frame_id")

    players = [c[:-2] for c in tracking.columns if c[:4] in ["home", "away"] and c.endswith("_x")]
    play_records = dict()

    for p in players:
        player_x = tracking[f"{p}_x"].dropna()
        if not player_x.empty:
            play_records[p] = {"in_frame_id": player_x.index[0], "out_frame_id": player_x.index[-1]}

    return pd.DataFrame(play_records).T


def label_phases(tracking: pd.DataFrame, keepers: List[str] = None) -> pd.DataFrame:
    has_frame_as_col = "frame_id" in tracking.columns
    if has_frame_as_col:
        tracking = tracking.copy().set_index("frame_id")

    keepers = [] if keepers is None else list(keepers)

    play_records = summarize_playing_times(tracking)
    player_in_frames = play_records["in_frame_id"].unique().tolist()
    player_out_frames = (play_records["out_frame_id"].unique() + 1).tolist()
    period_start_frames = tracking.reset_index().groupby("period_id")["frame_id"].first().values.tolist()
    phase_changes = np.sort(np.unique(player_in_frames + player_out_frames + period_start_frames))

    phases = []

    for i, start_frame in enumerate(phase_changes[:-1]):
        end_frame = phase_changes[i + 1] - 1
        alive_tracking = tracking[tracking["ball_state"] == "alive"].loc[start_frame:end_frame].copy()
        if len(alive_tracking) < 100:
            continue

        active_players = find_active_players(alive_tracking)
        home_keepers = [p for p in keepers if p in active_players[0]]
        away_keepers = [p for p in keepers if p in active_players[1]]
        home_x_cols = [f"{p}_x" for p in active_players[0]]
        away_x_cols = [f"{p}_x" for p in active_players[1]]

        # Determine which team is on the left by comparing mean x positions
        home_mean_x = alive_tracking[home_x_cols].mean().mean()
        away_mean_x = alive_tracking[away_x_cols].mean().mean()
        home_is_left = home_mean_x < away_mean_x

        if home_keepers and away_keepers:
            home_keeper = home_keepers[0]
            away_keeper = away_keepers[0]
        elif home_is_left:
            home_keeper = alive_tracking[home_x_cols].mean().idxmin()[:-2]
            away_keeper = alive_tracking[away_x_cols].mean().idxmax()[:-2]
        else:
            home_keeper = alive_tracking[home_x_cols].mean().idxmax()[:-2]
            away_keeper = alive_tracking[away_x_cols].mean().idxmin()[:-2]

        phase_dict = {
            "period_id": alive_tracking["period_id"].iloc[0],
            "start_frame_id": start_frame,
            "end_frame_id": end_frame,
            "active_players": active_players[0] + active_players[1],
            "active_keepers": [home_keeper, away_keeper],
        }
        phases.append(phase_dict)

    phases = pd.DataFrame(phases)
    phases.index.name = "phase"
    phases.index += 1

    # Update tracking phase_id to match the computed phases
    tracking["phase_id"] = 0
    for phase, row in phases.iterrows():
        mask = tracking.index.to_series().between(row["start_frame_id"], row["end_frame_id"])
        tracking.loc[mask, "phase_id"] = phase

    if has_frame_as_col:
        tracking = tracking.reset_index()

    return tracking, phases


def calculate_running_features(tracking: pd.DataFrame, fps=25, denoise: bool = True) -> pd.DataFrame:
    from scipy.signal import savgol_filter

    tracking = tracking.copy()

    if "episode_id" not in tracking.columns:
        tracking = label_frames_and_episodes(tracking)

    if "phase_id" not in tracking.columns:
        tracking, _ = label_phases(tracking)

    home_players = [c[:-2] for c in tracking.dropna(axis=1, how="all").columns if re.match(r"home_.*_x", c)]
    away_players = [c[:-2] for c in tracking.dropna(axis=1, how="all").columns if re.match(r"away_.*_x", c)]
    objects = home_players + away_players + ["ball"]
    physical_features = ["x", "y", "vx", "vy", "speed", "accel"]

    state_cols = ["frame_id", "period_id", "timestamp", "phase_id", "episode_id", "ball_state", "ball_owning_team_id"]
    feature_cols = [f"{p}_{f}" for p in objects for f in physical_features]
    if "ball_z" in tracking.columns:
        feature_cols.append("ball_z")

    if "player_id" in tracking.columns:
        state_cols.append("player_id")

    speed_threshold = 12.0  # m/s, ~ human sprint limit
    n_outliers_removed = 0

    for p in tqdm(objects, desc="Calculating running features per player"):
        new_cols = [f"{p}_{x}" for x in physical_features[2:]]
        new_features = pd.DataFrame(np.nan, index=tracking.index, columns=new_cols)

        # Drop pre-existing columns to avoid duplicate column names during concat/assign
        tracking = tracking.drop(columns=[c for c in new_cols if c in tracking.columns], errors="ignore")
        tracking = pd.concat([tracking, new_features], axis=1)

        for i in tracking["period_id"].unique():
            period_mask = tracking["period_id"] == i
            x: pd.Series = tracking.loc[period_mask, f"{p}_x"].dropna()
            y: pd.Series = tracking.loc[period_mask, f"{p}_y"].dropna()
            if x.empty:
                continue

            if denoise:
                # Detect outliers: frames where instantaneous speed exceeds threshold.
                # Skip outlier filtering for the ball, which can legitimately exceed human speed.
                dx = x.diff()
                dy = y.diff()
                inst_speed = np.sqrt(dx**2 + dy**2) * fps
                outlier_idx = pd.Index([]) if p == "ball" else x.index[inst_speed > speed_threshold]
                if len(outlier_idx) > 0:
                    tracking.loc[outlier_idx, f"{p}_x"] = np.nan
                    tracking.loc[outlier_idx, f"{p}_y"] = np.nan
                    n_outliers_removed += len(outlier_idx)

                    in_play_idx = x.index
                    x = tracking.loc[in_play_idx, f"{p}_x"].interpolate(method="linear").bfill().ffill()
                    y = tracking.loc[in_play_idx, f"{p}_y"].interpolate(method="linear").bfill().ffill()
                    tracking.loc[in_play_idx, f"{p}_x"] = x
                    tracking.loc[in_play_idx, f"{p}_y"] = y

                vx = savgol_filter(np.diff(x.values) * fps, window_length=15, polyorder=2)
                vy = savgol_filter(np.diff(y.values) * fps, window_length=15, polyorder=2)
                ax = savgol_filter(np.diff(vx) * fps, window_length=9, polyorder=2)
                ay = savgol_filter(np.diff(vy) * fps, window_length=9, polyorder=2)

            else:
                vx = np.diff(x.values) * fps
                vy = np.diff(y.values) * fps
                ax = np.diff(vx) * fps
                ay = np.diff(vy) * fps

            tracking.loc[x.index[1:], f"{p}_vx"] = vx
            tracking.loc[x.index[1:], f"{p}_vy"] = vy
            tracking.loc[x.index[1:], f"{p}_speed"] = np.sqrt(vx**2 + vy**2)
            tracking.loc[x.index[1:-1], f"{p}_accel"] = np.sqrt(ax**2 + ay**2)

            tracking.at[x.index[0], f"{p}_vx"] = tracking.at[x.index[1], f"{p}_vx"]
            tracking.at[x.index[0], f"{p}_vy"] = tracking.at[x.index[1], f"{p}_vy"]
            tracking.at[x.index[0], f"{p}_speed"] = tracking.at[x.index[1], f"{p}_speed"]
            tracking.loc[[x.index[0], x.index[-1]], f"{p}_accel"] = 0

    if denoise:
        print(f"Removed {n_outliers_removed} outlier frames")

    return tracking[state_cols + feature_cols].copy()
