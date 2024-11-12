import os
from collections import Counter
from datetime import datetime, timedelta
from pprint import pprint

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from tqdm import tqdm

from src.myconstants import *
from src.rolerep import RoleRep
from src.utils import (
    aggregate_player_periods,
    complete_perm,
    compute_delaunay_dists,
    compute_switch_rate,
    decompose_perm_to_cycles,
    delaunay_adj_mat,
    detect_change_times,
    most_common,
)
from src.visualize import plot_graph, plot_timeline

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 20)


# formation and role change-point detection (main algorithm)
class SoccerCPD:
    def __init__(self, data: pd.DataFrame, formcpd_method="gseg_avg", rolecpd_method="gseg_avg"):
        # formcpd_methods: ["gseg_avg", "gseg_union", "kernel_linear", "kernel_rbf", "kernel_cosine", "rank"]
        # rolecpd_methods: ["gseg_avg", "gseg_union"]

        self.data = data.set_index("datetime") if "datetime" in data.columns else data
        self.player_periods = aggregate_player_periods(self.data)

        self.formcpd_method = formcpd_method
        self.rolecpd_method = rolecpd_method

        self.role_seq = pd.DataFrame(columns=["datetime"] + HEADER_ROLE_SEQ)
        self.form_periods = None
        self.role_periods = None
        self.role_summary = None
        self.role_labels = None

    # align corresponding roles from different formation periods
    @staticmethod
    def align_formations(role_seq: pd.DataFrame, form_periods: pd.DataFrame):
        xy_cols = [c for c in form_periods.columns if c[0] in ["x", "y"]]
        base_node_xy = form_periods[xy_cols].iloc[0].astype(float).values.reshape(-1, 2)

        for i in form_periods.index[1:]:
            cur_form_period: pd.Series = form_periods.loc[i]
            cur_node_xy = cur_form_period[xy_cols].dropna().astype(float).values.reshape(-1, 2)

            cost_mat = distance_matrix(base_node_xy, cur_node_xy)
            row_idx, col_idx = linear_sum_assignment(cost_mat)

            form_periods.at[i, "adj_mat"] = cur_form_period["adj_mat"][col_idx][:, col_idx]
            form_periods.loc[i, xy_cols] = np.nan
            for r, c in zip(row_idx, col_idx):
                form_periods.at[i, f"x{r + 1}"] = cur_node_xy[c, 0]
                form_periods.at[i, f"y{r + 1}"] = cur_node_xy[c, 1]

            inverse_perm = dict(zip(col_idx + 1, row_idx + 1))
            fp_role_seq: pd.DataFrame = role_seq[role_seq["form_period"] == i]
            for col in ["role", "base_role"]:
                role_seq.loc[fp_role_seq.index, col] = fp_role_seq[col].apply(lambda r: inverse_perm[r])

        return role_seq, form_periods

    def reassign_base_role(self, role_row):
        base_perm = self.role_periods.at[role_row["role_period"], "base_perm"]
        role_row["base_role"] = base_perm[role_row["base_role"]]
        return role_row

    # refind base roles per player period and recompute the switch rate per frame for the given role_seq
    def reset_precomputed_role_seq(self):
        for i in self.player_periods.index[1:]:
            pp_role_seq: pd.DataFrame = self.role_seq[self.role_seq["player_period"] == i]
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

    def summarize_role_assignment(self) -> pd.DataFrame:
        grouped = self.role_seq.groupby(["player_id", "role_period"], group_keys=False, as_index=False)
        role_summary = grouped[["player_period", "base_role"]].first()
        role_summary = pd.merge(role_summary, self.role_periods[HEADER_ROLE_PERIODS[:-1]])

        forms = self.form_periods.set_index("form_period")
        role_summary["x"] = role_summary.apply(lambda x: forms.at[x["form_period"], f"x{x['base_role']}"], axis=1)
        role_summary["y"] = role_summary.apply(lambda x: forms.at[x["form_period"], f"y{x['base_role']}"], axis=1)

        # role_summary = pd.merge(role_summary, self.roster[["squad_num", "player_name"]].reset_index())
        return role_summary[HEADER_ROLE_SUMMARY[1:]].astype({"player_period": int})

    def run(self, precomputed_path=None, freq="5S", max_sr=MAX_SWITCH_RATE):
        form_periods = []
        role_periods = []

        # if self.use_precomputed == True, load and initialize the precomputed role details
        if precomputed_path is not None and os.path.exists(precomputed_path):
            self.role_seq = pd.read_csv(precomputed_path, header=0, encoding="utf-8-sig", parse_dates=["datetime"])
            self.reset_precomputed_role_seq()

        # initialize formation and role period labels by the session labels
        self.data["subsession"] = self.data["player_period"].map(self.player_periods["subsession"].to_dict())
        self.data["form_period"] = self.data["subsession"]
        self.data["role_period"] = self.data["subsession"]

        role_list = []
        perm_list = []

        for i in self.player_periods["subsession"].unique():
            player_periods: pd.DataFrame = self.player_periods[self.player_periods["subsession"] == i]
            n_players = len(player_periods["players"].iloc[0])

            start_dt: datetime = player_periods["start_dt"].iloc[0]
            end_dt: datetime = player_periods["end_dt"].iloc[-1]
            sub_dts = pd.to_datetime(player_periods["start_dt"].values[1:])
            subsession_data: pd.DataFrame = self.data[self.data["subsession"] == i]

            print(f"\n{'-' * 24} Subsession {i} {'-' * 24}")
            print(player_periods.drop(["players", "subsession"], axis=1))

            if precomputed_path is None or self.role_seq.empty:
                print("\n* Step 1: Frame-by-frame role assignment using RoleRep")
                rolerep = RoleRep(subsession_data)
                session_role_seq = rolerep.run(freq="1S")
            else:
                print("\n* Step 1: Load the pre-computed role assignment result")
                session_role_seq = self.role_seq[self.role_seq["subsession"] == i]
                print(f"Session role sequence loaded and filtered from '{precomputed_path}'.")

            # exclude situations such as set-pieces that are irrelevant to the team formation
            valid_role_seq = session_role_seq[session_role_seq["switch_rate"] <= max_sr]

            # check whether all the 10 outfield players are measured for some periods
            role_x = valid_role_seq.pivot_table("x_norm", "datetime", "role", aggfunc="first")
            role_y = valid_role_seq.pivot_table("y_norm", "datetime", "role", aggfunc="first")
            role_xy = np.dstack([role_x.dropna().values, role_y.dropna().values])
            role_list.append(session_role_seq)

            # generate the sequence of role-adjacency matrices
            adj_mats = []
            for xy in role_xy:
                adj_mats.append(delaunay_adj_mat(xy).reshape(-1))
            adj_mats = pd.DataFrame(np.stack(adj_mats, axis=0), index=role_x.dropna().index)

            if self.formcpd_method is not None:
                print("\n* Step 2: FormCPD based on role-adjacency matrices")
                form_chg_dts = detect_change_times(adj_mats, sub_dts, mode="form", method=self.formcpd_method)

                # round down chg_dts to the nearest 5-second mark with an offset
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
                print("\n* Step 2: Compute the formation graph of the session")
                # assume there are no formation change throughout the session
                form_chg_dts = [start_dt, end_dt]

                print("\n* Step 3: Find the most frequent role permutation per 5-minute segment")

            # generate the sequence of role permutations
            perms = valid_role_seq.pivot_table("base_role", "datetime", "role", aggfunc="first")
            role_set = set(perms.dropna().iloc[0])
            perms = perms.apply(complete_perm, axis=1, args=(role_set,)).astype(int)
            perms_str = perms.apply(lambda perm: np.array2string(perm.values), axis=1)
            perm_list.append(perms_str.rename("perm").to_frame())

            for form_chg_idx in range(1, len(form_chg_dts)):
                form_period = len(form_periods) + 1
                form_start_dt = form_chg_dts[form_chg_idx - 1]
                form_end_dt = form_chg_dts[form_chg_idx]

                mean_x = role_x[form_start_dt:form_end_dt].dropna().mean(axis=0).round(4).values
                mean_y = role_y[form_start_dt:form_end_dt].dropna().mean(axis=0).round(4).values
                mean_adj_mat = adj_mats[form_start_dt:form_end_dt].mean(axis=0).round(4).values

                # recording the details of the formation period
                form_record = {
                    "session": player_periods["session"].iloc[0],
                    "form_period": form_period,
                    "start_dt": form_start_dt,
                    "end_dt": form_end_dt,
                    "duration": (form_end_dt - form_start_dt).total_seconds(),
                    # "node_xy": np.stack([mean_x, mean_y]).T,
                    "adj_mat": mean_adj_mat.reshape(n_players, n_players),
                }
                for r in np.arange(n_players):
                    form_record[f"x{r + 1}"] = mean_x[r]
                    form_record[f"y{r + 1}"] = mean_y[r]

                form_periods.append(form_record)

                if self.rolecpd_method is not None:
                    # recursive change-point detection for the permutation sequence
                    print(f"\nRoleCPD for the formation period {form_period}:")
                    input_perms = perms[form_start_dt:form_end_dt]
                    input_sub_dts = np.array([dt for dt in sub_dts if (dt >= form_start_dt) and (dt < form_end_dt)])
                    role_chg_dts = detect_change_times(
                        input_perms,
                        input_sub_dts,
                        mode="role",
                        method=self.rolecpd_method,
                    )

                    # round down chg_dts to the nearest 5-second mark with an offset
                    freq_sec = float(freq[:-1])
                    offset = form_start_dt.second % freq_sec
                    role_chg_dts_rounded = []
                    for dt in role_chg_dts:
                        role_chg_dts_rounded.append(dt - timedelta(seconds=(dt.second - offset) % freq_sec))

                    print("Detected role change-points:")
                    pprint(role_chg_dts_rounded)

                    role_chg_dts = [form_start_dt, form_end_dt] + input_sub_dts.tolist()
                    role_chg_dts = list(set(role_chg_dts) | set(pd.to_datetime(role_chg_dts_rounded)))
                    role_chg_dts.sort()

                    for role_chg_idx in range(1, len(role_chg_dts)):
                        role_period = len(role_periods) + 1
                        role_start_dt = role_chg_dts[role_chg_idx - 1]
                        role_end_dt = role_chg_dts[role_chg_idx]

                        # Set the instructed roles per player by the most frequent permutation in the role period
                        counter = Counter(perms_str[role_start_dt:role_end_dt])
                        base_perm_list = np.fromstring(counter.most_common(1)[0][0][1:-1], dtype=int, sep=" ")
                        base_perm_dict = dict(zip(perms.columns, base_perm_list))

                        # Recording the details of the role period
                        role_periods.append(
                            {
                                "session": player_periods["session"].iloc[0],
                                "form_period": form_period,
                                "role_period": role_period,
                                "start_dt": role_start_dt,
                                "end_dt": role_end_dt,
                                "duration": (role_end_dt - role_start_dt).total_seconds(),
                                "base_perm": base_perm_dict,
                            },
                        )

        if role_list:
            self.role_seq = pd.concat(role_list, ignore_index=True)
        else:
            return

        if self.rolecpd_method is None:
            # finding the most frequent role permutation per 5-minute segment
            perms_str = pd.concat(perm_list)
            bins = self.player_periods["start_dt"].tolist()[1:] + [self.player_periods["end_dt"].iloc[0]]
            perms_str["player_period"] = pd.cut(perms_str.index, bins, labels=self.player_periods.index[1:])

            base_perm_list = []
            for i in self.player_periods.index[1:]:
                period_perms_str = perms_str[perms_str["player_period"] == i]
                if period_perms_str.empty:
                    continue

                i = self.player_periods.at[i, "session"]
                period_perms_str["session"] = i
                period_perms_str["form_period"] = i

                period_start_dt = self.player_periods.at[i, "start_dt"]
                offset = f"{period_start_dt.minute * SCALAR_TIME + period_start_dt.second}S"
                resampler = period_perms_str.resample("5T", closed="left", offset=offset)

                base_perms = resampler.apply(most_common).reset_index()
                base_perms["end_dt"] = base_perms["datetime"].shift(-1)
                base_perms.iat[-1, -1] = self.player_periods.at[i, "end_dt"]
                base_perm_list.append(base_perms)

            role_periods = pd.concat(base_perm_list, ignore_index=True)
            role_periods.rename(columns={"datetime": "start_dt"}, inplace=True)

            perms_list = role_periods["perm"].apply(lambda x: np.fromstring(x[1:-1], dtype=int, sep=" "))
            role_periods["base_perm"] = perms_list.apply(lambda perm: dict(zip(np.arange(10) + 1, perm)))
            role_periods["role_period"] = role_periods.index + 1
            tds = role_periods["end_dt"] - role_periods["start_dt"]
            role_periods["duration"] = tds.apply(lambda x: x.total_seconds())
            self.role_periods = role_periods[HEADER_ROLE_PERIODS]

        self.form_periods = pd.DataFrame(form_periods).set_index("form_period")
        self.role_periods = pd.DataFrame(role_periods).set_index("role_period")

        # label formation and role periods to the timestamps of data and role_seq
        match_end_dt = self.player_periods["end_dt"].iloc[-1]
        self.data["form_period"] = pd.cut(
            self.data.index,
            bins=self.form_periods["start_dt"].tolist() + [match_end_dt],
            right=False,
            labels=self.form_periods.index,
        )
        self.data["role_period"] = pd.cut(
            self.data.index,
            bins=self.role_periods["start_dt"].tolist() + [match_end_dt],
            right=False,
            labels=self.role_periods.index,
        )
        self.role_seq["form_period"] = pd.cut(
            self.role_seq["datetime"],
            bins=self.form_periods["start_dt"].tolist() + [match_end_dt],
            right=False,
            labels=self.form_periods.index,
        )
        self.role_seq["role_period"] = pd.cut(
            self.role_seq["datetime"],
            bins=self.role_periods["start_dt"].tolist() + [match_end_dt],
            right=False,
            labels=self.role_periods.index,
        )

        # # reflect the instructed roles and recompute switch rates in role_seq
        self.role_seq = self.role_seq.apply(self.reassign_base_role, axis=1)
        self.role_seq = self.role_seq.groupby("datetime", group_keys=False).apply(compute_switch_rate)
        self.role_seq, self.form_periods = SoccerCPD.align_formations(self.role_seq, self.form_periods)
        self.role_seq.sort_values(by=["player_id", "datetime"], ignore_index=True, inplace=True)

        self.form_periods = self.form_periods.reset_index()
        self.role_periods = self.role_periods.reset_index()[HEADER_ROLE_PERIODS]
        self.role_summary = self.summarize_role_assignment()
        print()
        print("-" * 73)
        print("Formation Periods:")
        print(self.form_periods[HEADER_FORM_PERIODS[:-1]])
        print()
        print("Role Periods:")
        print(self.role_periods[HEADER_ROLE_PERIODS[:-1]])
        print()

    def label_roles(
        self,
        benchmark_roles: pd.DataFrame,
        benchmark_forms: pd.DataFrame = None,
        form_labels: dict = None,
    ):
        assert benchmark_forms is not None or form_labels is not None
        xy_cols = [c for c in self.form_periods.columns if c[0] in ["x", "y"]]

        role_labels = dict()
        self.form_periods["formation"] = np.nan
        self.role_summary["formation"] = np.nan
        self.role_summary["aligned_role"] = np.nan

        for i in self.form_periods.index:
            form_period = self.form_periods.at[i, "form_period"]

            if form_labels:
                form_label = form_labels[form_period]
            elif len(self.form_periods.loc[i, xy_cols[0::2]].dropna()) < 10:
                form_label = "others"
            else:
                benchmark_forms["dist_to_sample"] = 0
                for j in benchmark_forms.index:
                    ref_form = self.form_periods.loc[i]
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

            instance_xy = self.form_periods.loc[i, xy_cols].dropna().astype(float)
            valid_roles = np.array([int(c[1:]) for c in instance_xy.index[0::2]])
            instance_xy = instance_xy.values.reshape(-1, 2)

            cost_mat: pd.DataFrame = role_distns.apply(lambda n: pd.Series(-np.log(n.pdf(instance_xy))))
            row_idx, col_idx = linear_sum_assignment(cost_mat.values)
            role_labels[form_period] = dict(zip(valid_roles[col_idx], cost_mat.index[row_idx].values))

            fp_rs: pd.DataFrame = self.role_summary.loc[self.role_summary["form_period"] == form_period]
            if len(self.form_periods.loc[i, xy_cols[0::2]].dropna()) < 10:
                form_label = "others"

            self.form_periods.at[i, "formation"] = form_label
            self.role_summary.loc[fp_rs.index, "formation"] = form_label
            self.role_summary.loc[fp_rs.index, "aligned_role"] = fp_rs["base_role"].replace(role_labels[form_period])

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

        match_times = self.role_seq.set_index("datetime")[["session", "time", "form_period"]].drop_duplicates()
        switches = pd.merge(match_times, switches, left_index=True, right_on="start_dt")

        switches = switches[(switches["duration"] > 1) & (switches["switch_rate"] > 0)].reset_index(drop=True).copy()
        switches["switch_roles"] = np.nan
        switches["switch_players"] = np.nan
        switches["argmax_speed"] = 0
        switches["max_speed"] = 0

        for i in tqdm(switches.index):
            start_dt = switches.at[i, "start_dt"]
            end_dt = switches.at[i, "end_dt"]
            form_period = switches.at[i, "form_period"]
            roleperm = roleperms.loc[start_dt]
            switches.at[i, "switch_roles"] = decompose_perm_to_cycles(roleperm, role_labels.loc[form_period])

            role_players = self.role_seq.set_index("datetime")[["base_role", "player_id"]].loc[start_dt]
            role_players = role_players.set_index("base_role")["player_id"].to_dict()
            switches.at[i, "switch_players"] = decompose_perm_to_cycles(roleperm, role_players)

            # if len(switches.at[i, "switch_players"]) > 0:
            #     switch_players = np.concatenate(switches.at[i, "switch_players"])
            #     switch_speeds = self.data.loc[start_dt:end_dt, [f"{p}_speed" for p in switch_players]].max()
            #     switches.at[i, "argmax_speed"] = switch_speeds.idxmax().split("_")[0]
            #     switches.at[i, "max_speed"] = switch_speeds.max()

        return switches

    def visualize(self, roster: pd.DataFrame = None, role_labels: pd.DataFrame = None, save_dir=None):
        import matplotlib.font_manager as fm
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
        import seaborn as sns

        # This is for visualizing Korean characters. If you don't need this, remove the following four lines.
        font_path = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Light.ttc"
        fontprop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = fontprop.get_name()
        sns.set_theme(font=fontprop.get_name(), font_scale=1.5)

        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        gs = gridspec.GridSpec(2, 4, left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)

        for i, form_period in enumerate(self.form_periods["form_period"][:4]):
            fp_role_seq = self.role_seq[(self.role_seq["form_period"] == form_period) & (self.role_seq["role"].notna())]
            fp_role_labels = role_labels.loc[form_period].dropna().to_dict() if role_labels is not None else None
            fp_graph = self.form_periods.loc[i]

            plt.subplot(gs[0, i])
            plot_graph(fp_role_seq, fp_role_labels, fp_graph)

            fp_role_periods = self.role_periods[self.role_periods["form_period"] == form_period]
            start_rp = fp_role_periods["role_period"].min()
            form_label = "others" if fp_graph["formation"] == "others" else "-".join(fp_graph["formation"])

            if len(fp_role_periods) == 1:
                plt.title(f"Role Period {start_rp}: {form_label}", fontsize=18)
            else:
                end_rp = fp_role_periods["role_period"].max()
                plt.title(f"Role Periods {start_rp}-{end_rp}: {form_label}", fontsize=18)

        ax = fig.add_subplot(gs[1, :])
        plot_timeline(self.role_seq, roster, ax)
        plt.title("Role Change Timeline", fontsize=18)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/timeline.png", bbox_inches="tight")
            plt.close(fig)
            print(f"Successfully saved in '{save_dir}/timeline.png'.")

        else:
            plt.show()
            plt.close(fig)

        sns.reset_orig()
        return

    def save_results(self, target_dir, form_summary=True, role_summary=True, role_seq=True):
        os.makedirs(target_dir, exist_ok=True)

        # save form_periods
        if form_summary:
            self.form_periods.to_pickle(f"{target_dir}/form_summary.pkl")
            print(f"Successfully saved in '{target_dir}/form_summary.pkl'.")

        # save role_summary
        if role_summary:
            self.role_summary.to_csv(f"{target_dir}/role_summary.csv", index=False, encoding="utf-8-sig")
            print(f"Successfully saved in '{target_dir}/role_summary.csv'.")

        # save role_seq
        if role_seq:
            self.role_seq.to_csv(f"{target_dir}/role_seq.csv", index=False, encoding="utf-8-sig")
            print(f"Successfully saved in '{target_dir}/role_seq.csv'.")
