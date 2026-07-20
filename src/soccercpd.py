import os
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from pprint import pprint

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from tqdm import tqdm

from src.config import *
from src.rolerep import RoleRep
from src.utils import (
    aggregate_player_segs,
    complete_perm,
    compute_delaunay_dists,
    compute_switch_rate,
    decompose_perm_to_cycles,
    delaunay_adj_mat,
    detect_change_times,
    most_common,
)

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 20)


# Formation and role change-point detection (main algorithm)
class SoccerCPD:
    def __init__(self, data: pd.DataFrame, formcpd_method: str = "gseg_avg", rolecpd_method: str = "gseg_avg"):
        # formcpd_methods: ["gseg_avg", "gseg_union", "kernel_linear", "kernel_rbf", "kernel_cosine", "rank"]
        # rolecpd_methods: ["gseg_avg", "gseg_union"]

        self.data = data.set_index("datetime") if "datetime" in data.columns else data
        self.player_segs = aggregate_player_segs(self.data)

        self.formcpd_method = formcpd_method
        self.rolecpd_method = rolecpd_method

        self.role_seq = pd.DataFrame(columns=["datetime"] + HEADER_ROLE_SEQ)
        self.form_segs = None
        self.role_segs = None
        self.role_assign = None
        self.role_labels = None

    def revise_short_role_segs(self, thres_dur: float = MIN_SEG_DUR, max_sr: float = MAX_SWITCH_RATE):
        # The role assignment in short segments is unreliable,
        # so snap each player's role to the neighbouring segments' where it deviates from both,
        # then fill the remaining roles by nearest position.
        for fp in self.role_segs["form_seg"].unique():
            fp_role_segs: pd.DataFrame = self.role_segs[self.role_segs["form_seg"] == fp]

            if len(fp_role_segs) < 2:
                continue

            for i in fp_role_segs.index:
                if self.role_segs.at[i, "duration"] < thres_dur:
                    rp = self.role_segs.at[i, "role_seg"] if "role_seg" in self.role_segs.columns else i
                    rp_seq = self.role_seq[self.role_seq["role_seg"] == rp]
                    valid_seq: pd.DataFrame = rp_seq[rp_seq["switch_rate"] < max_sr]

                    if valid_seq.empty:
                        continue

                    assignment: dict = deepcopy(self.role_segs.at[i, "assignment"])

                    if i - 1 in fp_role_segs.index:
                        prev_roles = defaultdict(int, fp_role_segs.at[i - 1, "assignment"])
                        role_set = set(prev_roles.values())
                    else:
                        prev_roles = defaultdict(int)

                    if i + 1 in fp_role_segs.index:
                        next_roles = defaultdict(int, fp_role_segs.at[i + 1, "assignment"])
                        role_set = set(next_roles.values())
                    else:
                        next_roles = defaultdict(int)

                    for player_id, role in assignment.items():
                        if role not in [prev_roles[player_id], next_roles[player_id]]:
                            if prev_roles[player_id] == next_roles[player_id]:
                                assignment[player_id] = prev_roles[player_id]
                            else:
                                assignment[player_id] = 0

                    player_xy = valid_seq.groupby("player_id")[["x_norm", "y_norm"]].mean().astype(float)
                    unassigned_players = np.array([k for k, v in assignment.items() if v == 0])
                    player_xy = player_xy.loc[unassigned_players].values

                    unassigned_roles = np.array(list(role_set - set(assignment.values())))
                    role_xy_cols = [f"{x}{r}" for x in ["x", "y"] for r in unassigned_roles]
                    role_xy = self.form_segs.loc[1, role_xy_cols].values.astype(float).reshape(2, -1).T

                    cost_mat = distance_matrix(player_xy, role_xy)
                    row_idx, col_idx = linear_sum_assignment(cost_mat)
                    for j, player_id in enumerate(unassigned_players[row_idx]):
                        assignment[player_id] = unassigned_roles[col_idx[j]]

                    self.role_segs.at[i, "assignment"] = assignment

    def reassign_base_role(self, row: pd.Series) -> pd.Series:
        assignment = self.role_segs.at[row["role_seg"], "assignment"]
        row["base_role"] = assignment[row["player_id"]]
        return row

    # Refind base roles per player period and recompute the switch rate per frame for the given role_seq
    def reset_precomputed_role_seq(self):
        for i in self.player_segs.index[1:]:
            pp_role_seq: pd.DataFrame = self.role_seq[self.role_seq["player_seg"] == i]
            perms: pd.DataFrame = pp_role_seq.pivot_table("role", "datetime", "player_id", "first")
            if perms.empty:
                continue

            # role_set = set(perms.dropna().iloc[0])
            role_set = set(np.arange(10) + 1)
            perms = perms.apply(complete_perm, axis=1, args=(role_set,)).astype(int)
            perms_str = perms.apply(lambda perm: np.array2string(perm.values), axis=1)
            base_perm_list = np.fromstring(most_common(perms_str)[1:-1], dtype="float32", sep=" ")
            base_perm_dict = dict(zip(perms.columns, base_perm_list))

            self.role_seq.loc[pp_role_seq.index, "base_role"] = pp_role_seq["player_id"].map(base_perm_dict)

        self.role_seq = self.role_seq.groupby("datetime", group_keys=False).apply(compute_switch_rate)

    # Align corresponding roles from different formation segments
    def align_formations(self):
        xy_cols = [c for c in self.form_segs.columns if c[0] in ["x", "y"]]
        base_node_xy = self.form_segs[xy_cols].iloc[0].astype(float).values.reshape(-1, 2)

        for i in self.form_segs.index[1:]:
            cur_form_seg: pd.Series = self.form_segs.loc[i]
            cur_node_xy = cur_form_seg[xy_cols].dropna().astype(float).values.reshape(-1, 2)

            cost_mat = distance_matrix(base_node_xy, cur_node_xy)
            row_idx, col_idx = linear_sum_assignment(cost_mat)

            self.form_segs.at[i, "adj_mat"] = cur_form_seg["adj_mat"][col_idx][:, col_idx]
            self.form_segs.loc[i, xy_cols] = np.nan
            for r, c in zip(row_idx, col_idx):
                self.form_segs.at[i, f"x{r + 1}"] = cur_node_xy[c, 0]
                self.form_segs.at[i, f"y{r + 1}"] = cur_node_xy[c, 1]

            inverse_perm = dict(zip(col_idx + 1, row_idx + 1))
            fp_role_segs = self.role_segs[self.role_segs["form_seg"] == cur_form_seg.name]
            for j in fp_role_segs.index:
                assignment: dict = fp_role_segs.at[j, "assignment"]
                self.role_segs.at[j, "assignment"] = {k: inverse_perm[v] for k, v in assignment.items()}

            fp_role_seq = self.role_seq[self.role_seq["form_seg"] == i]
            for col in ["role", "base_role"]:
                self.role_seq.loc[fp_role_seq.index, col] = fp_role_seq[col].apply(lambda c: inverse_perm[c])

    def summarize_role_assignments(self) -> pd.DataFrame:
        grouped = self.role_seq.groupby(["player_id", "role_seg"], group_keys=False, as_index=False)
        role_summary = grouped[["player_seg", "base_role"]].first()
        role_summary = pd.merge(role_summary, self.role_segs[HEADER_ROLE_SEGS[:-1]])

        forms = self.form_segs.set_index("form_seg")
        role_summary["x"] = role_summary.apply(lambda x: forms.at[x["form_seg"], f"x{x['base_role']}"], axis=1)
        role_summary["y"] = role_summary.apply(lambda x: forms.at[x["form_seg"], f"y{x['base_role']}"], axis=1)

        # role_summary = pd.merge(role_summary, self.roster[["squad_num", "player_name"]].reset_index())
        return role_summary[HEADER_ROLE_SUMMARY[1:]].astype({"player_seg": int})

    def run(
        self,
        precomputed_path: str = None,
        freq: str = "5S",
        max_sr: float = MAX_SWITCH_RATE,
        min_seg_dur: float = MIN_SEG_DUR,
    ) -> None:
        form_segments = []
        role_segments = []

        # If self.use_precomputed == True, load and initialize the precomputed role details
        if precomputed_path is not None and os.path.exists(precomputed_path):
            self.role_seq = pd.read_csv(precomputed_path, header=0, encoding="utf-8-sig", parse_dates=["datetime"])
            self.reset_precomputed_role_seq()

        # Initialize form and role segment labels from the subperiod labels
        self.data["subperiod_id"] = self.data["player_seg"].map(self.player_segs["subperiod_id"].to_dict())
        self.data["form_seg"] = self.data["subperiod_id"]
        self.data["role_seg"] = self.data["subperiod_id"]

        role_list = []
        perm_list = []

        for i in self.player_segs["subperiod_id"].unique():
            player_segs: pd.DataFrame = self.player_segs[self.player_segs["subperiod_id"] == i]
            n_players = len(player_segs["players"].iloc[0])

            start_dt: datetime = player_segs["start_dt"].iloc[0]
            end_dt: datetime = player_segs["end_dt"].iloc[-1]
            sub_dts = pd.to_datetime(player_segs["start_dt"].values[1:])
            subperiod_data: pd.DataFrame = self.data[self.data["subperiod_id"] == i]

            print(f"\n{'-' * 24} Subperiod {i} {'-' * 24}")
            print(player_segs.drop(["players", "subperiod_id"], axis=1))

            if precomputed_path is None or self.role_seq.empty:
                print("\n* Step 1: Frame-by-frame role assignment using RoleRep")
                rolerep = RoleRep(subperiod_data)
                subperiod_role_seq = rolerep.run(freq="1S")
            else:
                print("\n* Step 1: Load the pre-computed role assignment result")
                subperiod_role_seq = self.role_seq[self.role_seq["subperiod_id"] == i]
                print(f"Subperiod role sequence loaded and filtered from '{precomputed_path}'.")

            role_list.append(subperiod_role_seq)

            # Exclude situations such as set-pieces that are irrelevant to the team formation
            valid_seq = subperiod_role_seq[subperiod_role_seq["switch_rate"] <= max_sr]

            # Check whether all the 10 outfield players are measured for some periods
            role_x = valid_seq.pivot_table("x_norm", "datetime", "role", aggfunc="first")
            role_y = valid_seq.pivot_table("y_norm", "datetime", "role", aggfunc="first")

            # Skip degenerate subperiods (e.g. very short possession-filtered streams) with no moment
            # where all roles are present -- otherwise the pivots below raise.
            if role_x.dropna().empty:
                print(f"  (skipping subperiod {i}: no frame with all roles present)")
                continue

            role_xy = np.dstack([role_x.dropna().values, role_y.dropna().values])

            # Generate the sequence of role-adjacency matrices
            adj_mats = []
            for xy in role_xy:
                adj_mats.append(delaunay_adj_mat(xy).reshape(-1))
            adj_mats = pd.DataFrame(np.stack(adj_mats, axis=0), index=role_x.dropna().index)

            if self.formcpd_method is not None:
                print("\n* Step 2: FormCPD based on role-adjacency matrices")
                form_chg_dts = detect_change_times(adj_mats, sub_dts, "form", self.formcpd_method)

                # Round down chg_dts to the nearest 5-second mark with an offset
                freq_sec = float(freq[:-1])
                offset = start_dt.second % freq_sec
                form_chg_dts_rounded = []
                for dt in form_chg_dts:
                    form_chg_dts_rounded.append(dt - timedelta(seconds=(dt.second - offset) % freq_sec))

                print("Detected formation change-points:")
                pprint(form_chg_dts_rounded)

                print("\n* Step 3: RoleCPD per formation period based on role permutations")
                form_chg_dts = [start_dt] + form_chg_dts_rounded + [end_dt]

            else:
                print("\n* Step 2: Compute the formation graph of the subperiod")
                # Assume there are no formation change throughout the subperiod
                form_chg_dts = [start_dt, end_dt]

                print("\n* Step 3: Find the most frequent role permutation per 5-minute segment")

            # Generate the sequence of role permutations
            perms = valid_seq.pivot_table("base_role", "datetime", "role", aggfunc="first")
            role_set = set(perms.dropna().iloc[0])
            perms = perms.apply(complete_perm, axis=1, args=(role_set,)).astype(int)
            perms_str = perms.apply(lambda perm: np.array2string(perm.values), axis=1)
            perm_list.append(perms_str.rename("perm").to_frame())

            for form_chg_idx in range(1, len(form_chg_dts)):
                form_seg = len(form_segments) + 1
                fp_start_dt = form_chg_dts[form_chg_idx - 1]
                fp_end_dt = form_chg_dts[form_chg_idx]

                mean_x = role_x[fp_start_dt:fp_end_dt].dropna().mean(axis=0).round(4).values
                mean_y = role_y[fp_start_dt:fp_end_dt].dropna().mean(axis=0).round(4).values
                mean_adj_mat = adj_mats[fp_start_dt:fp_end_dt].mean(axis=0).round(4).values

                # Record the details of the formation period
                form_record = {
                    "period_id": player_segs["period_id"].iloc[0],
                    "form_seg": form_seg,
                    "start_dt": fp_start_dt,
                    "end_dt": fp_end_dt,
                    "duration": (fp_end_dt - fp_start_dt).total_seconds(),
                    "adj_mat": mean_adj_mat.reshape(n_players, n_players),
                }
                for r in np.arange(n_players):
                    form_record[f"x{r + 1}"] = mean_x[r]
                    form_record[f"y{r + 1}"] = mean_y[r]

                form_segments.append(form_record)

                if self.rolecpd_method is not None:
                    # Recursive change-point detection for the permutation sequence
                    print(f"\nRoleCPD for the formation period {form_seg}:")
                    input_perms = perms[fp_start_dt:fp_end_dt]
                    input_sub_dts = np.array([dt for dt in sub_dts if (dt >= fp_start_dt) and (dt < fp_end_dt)])
                    role_chg_dts = detect_change_times(input_perms, input_sub_dts, "role", self.rolecpd_method)

                    # Round down chg_dts to the nearest 5-second mark with an offset
                    freq_sec = float(freq[:-1])
                    offset = fp_start_dt.second % freq_sec
                    role_chg_dts_rounded = []
                    for dt in role_chg_dts:
                        role_chg_dts_rounded.append(dt - timedelta(seconds=(dt.second - offset) % freq_sec))

                    print("Detected role change-points:")
                    pprint(role_chg_dts_rounded)

                    # Keep form-period ends + substitutions as boundaries; a detected change-point is
                    # dropped when it would make a segment shorter than `min_seg_dur`, merging it into a neighbour.
                    role_chg_dts = sorted(set([fp_start_dt, fp_end_dt] + input_sub_dts.tolist()))
                    for cp in sorted(pd.to_datetime(role_chg_dts_rounded)):
                        if all(abs((cp - b).total_seconds()) >= min_seg_dur for b in role_chg_dts):
                            role_chg_dts.append(cp)
                    role_chg_dts.sort()

                    for role_chg_idx in range(1, len(role_chg_dts)):
                        role_seg = len(role_segments) + 1
                        rp_start_dt = role_chg_dts[role_chg_idx - 1]
                        rp_end_dt = role_chg_dts[role_chg_idx]
                        duration = (rp_end_dt - rp_start_dt).total_seconds()

                        # Find the most frequent role assignment in the role period
                        rp_seq = valid_seq[(valid_seq["datetime"] >= rp_start_dt) & (valid_seq["datetime"] < rp_end_dt)]
                        if rp_seq.empty:
                            # No possession-valid frames in this role period (e.g. an all-high-
                            # switch-rate tail); skip it so it merges into the neighbouring role
                            # period instead of raising on the empty selection below.
                            continue
                        player_seg = rp_seq.iloc[0]["player_seg"]
                        pp_seq: pd.DataFrame = valid_seq[valid_seq["player_seg"] == player_seg]

                        temp_roles = pp_seq.pivot_table("role", "datetime", "player_id", aggfunc="first")
                        role_set = set(temp_roles.dropna().iloc[0])
                        temp_roles = temp_roles.apply(complete_perm, axis=1, args=(role_set,)).astype(int)
                        temp_roles_str = temp_roles.apply(lambda perm: np.array2string(perm.values), axis=1)

                        counter = Counter(temp_roles_str[rp_start_dt:rp_end_dt])
                        most_common_roles = np.fromstring(counter.most_common(1)[0][0][1:-1], dtype=int, sep=" ")
                        assignment = dict(zip(temp_roles.columns, most_common_roles))

                        # record the details of the role period
                        role_segments.append(
                            {
                                "period_id": player_segs["period_id"].iloc[0],
                                "form_seg": form_seg,
                                "role_seg": role_seg,
                                "start_dt": rp_start_dt,
                                "end_dt": rp_end_dt,
                                "duration": duration,
                                "assignment": assignment,
                            },
                        )

        if role_list:
            self.role_seq = pd.concat(role_list, ignore_index=True)
        else:
            return

        if self.rolecpd_method is None:
            # Find the most frequent role permutation per 5-minute segment
            perms_str = pd.concat(perm_list)
            bins = self.player_segs["start_dt"].tolist()[1:] + [self.player_segs["end_dt"].iloc[0]]
            perms_str["player_seg"] = pd.cut(perms_str.index, bins, labels=self.player_segs.index[1:])

            base_perm_list = []
            for i in self.player_segs.index[1:]:
                period_perms_str: pd.DataFrame = perms_str[perms_str["player_seg"] == i]
                if period_perms_str.empty:
                    continue

                i = self.player_segs.at[i, "period_id"]
                period_perms_str["period_id"] = i
                period_perms_str["form_seg"] = i

                period_start_dt: datetime = self.player_segs.at[i, "start_dt"]
                offset = f"{period_start_dt.minute * SCALAR_TIME + period_start_dt.second}S"
                resampler = period_perms_str.resample("5T", closed="left", offset=offset)

                base_perms = resampler.apply(most_common).reset_index()
                base_perms["end_dt"] = base_perms["datetime"].shift(-1)
                base_perms.iat[-1, -1] = self.player_segs.at[i, "end_dt"]
                base_perm_list.append(base_perms)

            role_segments = pd.concat(base_perm_list, ignore_index=True)
            role_segments.rename(columns={"datetime": "start_dt"}, inplace=True)

            perms_list: pd.Series = role_segments["perm"].apply(lambda x: np.fromstring(x[1:-1], dtype=int, sep=" "))
            role_segments["assignment"] = perms_list.apply(lambda perm: dict(zip(np.arange(10) + 1, perm)))
            role_segments["role_seg"] = role_segments.index + 1
            tds = role_segments["end_dt"] - role_segments["start_dt"]
            role_segments["duration"] = tds.apply(lambda x: x.total_seconds())
            self.role_segs = role_segments[HEADER_ROLE_SEGS]

        self.form_segs = pd.DataFrame(form_segments).set_index("form_seg")
        self.role_segs = pd.DataFrame(role_segments).set_index("role_seg")

        # Label formation and role periods to the timestamps of data and role_seq
        match_end_dt = self.player_segs["end_dt"].iloc[-1]
        self.data["form_seg"] = pd.cut(
            self.data.index,
            bins=self.form_segs["start_dt"].tolist() + [match_end_dt],
            right=False,
            labels=self.form_segs.index,
        )
        self.data["role_seg"] = pd.cut(
            self.data.index,
            bins=self.role_segs["start_dt"].tolist() + [match_end_dt],
            right=False,
            labels=self.role_segs.index,
        )
        self.role_seq["form_seg"] = pd.cut(
            self.role_seq["datetime"],
            bins=self.form_segs["start_dt"].tolist() + [match_end_dt],
            right=False,
            labels=self.form_segs.index,
        )
        self.role_seq["role_seg"] = pd.cut(
            self.role_seq["datetime"],
            bins=self.role_segs["start_dt"].tolist() + [match_end_dt],
            right=False,
            labels=self.role_segs.index,
        )

        # Reflect the instructed roles and recompute switch rates in role_seq
        self.role_seq = self.role_seq.apply(self.reassign_base_role, axis=1)
        self.role_seq = self.role_seq.groupby("datetime", group_keys=False).apply(compute_switch_rate)
        self.align_formations()
        self.revise_short_role_segs()

        self.role_seq = self.role_seq.apply(self.reassign_base_role, axis=1)
        self.role_seq.sort_values(by=["player_id", "datetime"], ignore_index=True, inplace=True)

        self.form_segs = self.form_segs.reset_index()
        self.role_segs = self.role_segs.reset_index()[HEADER_ROLE_SEGS]
        self.role_assign = self.summarize_role_assignments()
        print()
        print("-" * 73)
        print("Formation segments:")
        print(self.form_segs[["form_seg", "period_id", "start_dt", "end_dt", "duration"]])
        print()
        print("Role segments:")
        print(self.role_segs[["form_seg", "role_seg", "period_id", "start_dt", "end_dt", "duration"]])
        print()

    def label_roles(
        self,
        benchmark_roles: pd.DataFrame,
        benchmark_forms: pd.DataFrame = None,
        form_labels: dict = None,
    ):
        assert benchmark_forms is not None or form_labels is not None
        xy_cols = [c for c in self.form_segs.columns if c[0] in ["x", "y"]]

        role_labels = dict()
        self.form_segs["formation"] = np.nan
        self.role_assign["formation"] = np.nan
        self.role_assign["aligned_role"] = np.nan

        for i in self.form_segs.index:
            form_seg = self.form_segs.at[i, "form_seg"]

            if form_labels:
                form_label = form_labels[form_seg]
            elif len(self.form_segs.loc[i, xy_cols[0::2]].dropna()) < 10:
                form_label = "others"
            else:
                benchmark_forms["dist_to_sample"] = 0
                for j in benchmark_forms.index:
                    ref_form = self.form_segs.loc[i]
                    cur_form = benchmark_forms.loc[j]
                    benchmark_forms.at[j, "dist_to_sample"] = compute_delaunay_dists(ref_form, cur_form)
                form_label = benchmark_forms.groupby("label")["dist_to_sample"].mean().idxmin()

            if form_label in ROLE_TEMPLATE.index[:-1]:
                group_roles = benchmark_roles[benchmark_roles["formation"] == form_label]
                args = {"col_x": "x", "col_y": "y", "filter": False}
                role_distns: pd.Series = group_roles.groupby("aligned_role").apply(RoleRep.estimate_mvn, **args)
            else:
                args = {"col_x": "x", "col_y": "y", "filter": False}
                role_distns: pd.Series = benchmark_roles.groupby("aligned_role").apply(RoleRep.estimate_mvn, **args)

            instance_xy = self.form_segs.loc[i, xy_cols].dropna().astype(float)
            valid_roles = np.array([int(c[1:]) for c in instance_xy.index[0::2]])
            instance_xy = instance_xy.values.reshape(-1, 2)

            cost_mat: pd.DataFrame = role_distns.apply(lambda n: pd.Series(-np.log(n.pdf(instance_xy))))
            row_idx, col_idx = linear_sum_assignment(cost_mat.values)
            role_labels[form_seg] = dict(zip(valid_roles[col_idx], cost_mat.index[row_idx].values))

            fp_rs: pd.DataFrame = self.role_assign.loc[self.role_assign["form_seg"] == form_seg]
            if len(self.form_segs.loc[i, xy_cols[0::2]].dropna()) < 10:
                form_label = "others"

            self.form_segs.at[i, "formation"] = form_label
            self.role_assign.loc[fp_rs.index, "formation"] = form_label
            self.role_assign.loc[fp_rs.index, "aligned_role"] = fp_rs["base_role"].replace(role_labels[form_seg])

        self.role_labels = pd.DataFrame(role_labels).T[np.arange(10) + 1]

    def detect_switches(self, role_labels: pd.DataFrame) -> pd.DataFrame:
        self.role_seq["roleperm"] = self.role_seq.apply(lambda x: (x["base_role"], x["role"]), axis=1)
        roleperms = self.role_seq.pivot_table("roleperm", "datetime", "player_id", aggfunc="first")
        roleperms["switch_rate"] = self.role_seq.groupby("datetime")["switch_rate"].first()

        start_dts = roleperms[((roleperms.notna()) & (roleperms != roleperms.shift(1))).any(axis=1)].index
        end_dts = roleperms[((roleperms.notna()) & (roleperms != roleperms.shift(-1))).any(axis=1)].index

        switches = pd.DataFrame(np.stack([start_dts, end_dts]).T, columns=["start_dt", "end_dt"])
        switches["duration"] = (switches["end_dt"] - switches["start_dt"]).apply(lambda x: x.total_seconds() + 1)
        switches["switch_rate"] = roleperms.loc[start_dts, "switch_rate"].values

        match_times = self.role_seq.set_index("datetime")[["period_id", "timestamp", "form_seg"]].drop_duplicates()
        switches = pd.merge(match_times, switches, left_index=True, right_on="start_dt")

        switches = switches[(switches["duration"] > 1) & (switches["switch_rate"] > 0)].reset_index(drop=True).copy()
        switches["switch_roles"] = np.nan
        switches["switch_players"] = np.nan
        switches["argmax_speed"] = 0
        switches["max_speed"] = 0

        for i in tqdm(switches.index):
            start_dt = switches.at[i, "start_dt"]
            form_seg = switches.at[i, "form_seg"]
            roleperm = roleperms.loc[start_dt]
            switches.at[i, "switch_roles"] = decompose_perm_to_cycles(roleperm, role_labels.loc[form_seg])

            role_players = self.role_seq.set_index("datetime")[["base_role", "player_id"]].loc[start_dt]
            role_players = role_players.set_index("base_role")["player_id"].to_dict()
            switches.at[i, "switch_players"] = decompose_perm_to_cycles(roleperm, role_players)

        return switches

    def save_results(self, target_dir, form_summary=True, role_summary=True, role_seq=True):
        os.makedirs(target_dir, exist_ok=True)

        # Save form_segs
        if form_summary:
            self.form_segs.to_pickle(f"{target_dir}/form_summary.pkl")
            print(f"Successfully saved in '{target_dir}/form_summary.pkl'.")

        # Save role_summary
        if role_summary:
            self.role_assign.to_csv(f"{target_dir}/role_summary.csv", index=False, encoding="utf-8-sig")
            print(f"Successfully saved in '{target_dir}/role_summary.csv'.")

        # Save role_seq
        if role_seq:
            self.role_seq.to_csv(f"{target_dir}/role_seq.csv", index=False, encoding="utf-8-sig")
            print(f"Successfully saved in '{target_dir}/role_seq.csv'.")
