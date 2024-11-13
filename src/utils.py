import os
from collections import Counter
from datetime import datetime, timedelta
from pprint import pprint
from typing import List, Union

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

from src.myconstants import *

init_printing(perm_cyclic=True, pretty_print=False)


def reshape_traces(traces: pd.DataFrame) -> pd.DataFrame:
    xy_list = []
    players = [c[:3] for c in traces.columns if c[3:] == "_x"]

    for p in players:
        cols = ["datetime", "session", "time", "player_period", f"{p}_x", f"{p}_y"]
        player_xy = traces[cols].copy().rename(columns={f"{p}_x": "x", f"{p}_y": "y"})
        player_xy["player_id"] = p
        xy_list.append(player_xy)

    return pd.concat(xy_list).set_index("datetime")


def aggregate_player_periods(data: pd.DataFrame) -> pd.DataFrame:
    if "datetime" not in data.columns:
        data = data.reset_index().rename(columns={"index": "datetime"})

    grouper = data.groupby("player_period")
    freq = round(data["time"].iloc[1] - data["time"].iloc[0], 3)

    sessions = grouper["session"].first()
    start_dts = grouper["datetime"].first().rename("start_dt")
    end_dts = grouper["datetime"].last().rename("end_dt") + timedelta(seconds=freq)
    player_periods = pd.concat([sessions, start_dts, end_dts], axis=1)

    player_periods["players"] = None
    for i in player_periods.index:
        pp_data: pd.DataFrame = data[data["player_period"] == i]
        player_periods.at[i, "players"] = pp_data.groupby("player_id")["x"].first().dropna().index.tolist()

    n_players = player_periods["players"].apply(len)
    n_players = pd.concat([player_periods["session"], n_players], axis=1)
    player_periods["subsession"] = (n_players.diff().fillna(1) != 0).any(axis=1).astype(int).cumsum()

    return player_periods


# apply Delaunay triangulation to the given player coordinates to obtain the role-adjacency matrix
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
