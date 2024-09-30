import os
from collections import Counter
from datetime import timedelta
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
    delaunay_edge_mat,
    detect_change_times,
    most_common,
)
from src.visualize import plot_graph, plot_timeline

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 20)


# formation and role change-point detection (main algorithm)
class SoccerCPD:
    def __init__(self, data: pd.DataFrame, apply_cpd=True, formcpd_method="gseg_avg", rolecpd_method="gseg_avg"):
        # formcpd_methods: ["gseg_avg", "gseg_union", "kernel_linear", "kernel_rbf", "kernel_cosine", "rank"]
        # rolecpd_methods: ["gseg_avg", "gseg_union"]

        self.data = data.set_index("datetime") if "datetime" in data.columns else data
        self.player_periods = aggregate_player_periods(self.data)

        self.apply_cpd = apply_cpd
        self.formcpd_method = formcpd_method
        self.rolecpd_method = rolecpd_method

        self.role_seq = pd.DataFrame(columns=["datetime"] + HEADER_ROLE_SEQ)
        self.form_periods = pd.DataFrame(columns=HEADER_FORM_PERIODS)
        self.role_periods = pd.DataFrame(columns=HEADER_ROLE_PERIODS)
        self.role_summary = None
        self.role_labels = None

        self.target_dir = f"{DIR_DATA}/{formcpd_method}" if apply_cpd else f"{DIR_DATA}/noncpd"

    # align corresponding roles from different formation periods
    @staticmethod
    def align_formations(role_seq: pd.DataFrame, form_periods: pd.DataFrame):
        base_form_period = form_periods.iloc[0]

        for i in form_periods.index[1:]:
            cur_form_period = form_periods.loc[i]
            cost_mat = distance_matrix(
                base_form_period["coords"],
                cur_form_period["coords"],
            )
            _, perm = linear_sum_assignment(cost_mat)
            form_periods.at[i, "coords"] = cur_form_period["coords"][perm]
            form_periods.at[i, "edge_mat"] = cur_form_period["edge_mat"][perm][:, perm]

            inverse_perm = dict(zip(np.array(perm) + 1, np.arange(10) + 1))
            fp_role_seq = role_seq[role_seq["form_period"] == i]
            for col in ["role", "base_role"]:
                role_seq.loc[fp_role_seq.index, col] = fp_role_seq[col].apply(lambda role: inverse_perm[role])

        return role_seq, form_periods

    def reassign_base_role(self, role_row):
        base_perm = self.role_periods.at[role_row["role_period"], "base_perm"]
        role_row["base_role"] = base_perm[role_row["base_role"]]
        return role_row

    # refind base roles per player period and recompute the switch rate per frame for the given role_seq
    def reset_precomputed_role_seq(self):
        for i in self.player_periods.index[1:]:
            pp_role_seq = self.role_seq[self.role_seq["player_period"] == i]
            perms = pp_role_seq.pivot_table("role", "datetime", "player_id", "first")
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

        role_summary = pd.merge(role_summary, self.form_periods[["form_period", "coords"]])
        role_summary["x"] = role_summary.apply(lambda x: x["coords"][x["base_role"] - 1, 0], axis=1)
        role_summary["y"] = role_summary.apply(lambda x: x["coords"][x["base_role"] - 1, 1], axis=1)

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
        self.data["form_period"] = self.data["session"]
        self.data["role_period"] = self.data["session"]

        role_list = []
        perm_list = []

        for session in self.data["session"].unique():
            print(f"\n{'-' * 25} Session {session} {'-' * 26}")
            player_periods = self.player_periods[self.player_periods["session"] == session]
            session_start_dt = player_periods["start_dt"].iloc[0]
            session_end_dt = player_periods["end_dt"].iloc[-1]
            session_data = self.data[self.data["session"] == session]

            grouper = session_data.dropna(subset="x").groupby("time", group_keys=False)
            if grouper["player_id"].apply(len).max() < 10:
                # if less than 10 players have been measured during the session, skip the process
                print("Not enough players to estimate a formation.")
                continue
            else:
                print(player_periods)

            if precomputed_path is None or self.role_seq.empty:
                print("\n* Step 1: Frame-by-frame role assignment using RoleRep")
                rolerep = RoleRep(session_data)
                session_role_seq = rolerep.run(freq="1S")
            else:
                print("\n* Step 1: Load the pre-computed role assignment result")
                session_role_seq = self.role_seq[self.role_seq["session"] == session]
                print(f"Session role sequence loaded and filtered from '{precomputed_path}'.")

            # exclude situations such as set-pieces that are irrelevant to the team formation
            valid_role_seq = session_role_seq[session_role_seq["switch_rate"] <= max_sr]

            # check whether all the 10 outfield players are measured for some periods
            role_x = valid_role_seq.pivot_table("x_norm", "datetime", "role", aggfunc="first")
            role_y = valid_role_seq.pivot_table("y_norm", "datetime", "role", aggfunc="first")
            role_xy = np.dstack([role_x.dropna().values, role_y.dropna().values])
            if role_xy.shape[1] < 10:
                print("Not enough players to estimate a formation.")
                continue
            else:
                role_list.append(session_role_seq)

            # generate the sequence of role-adjacency matrices
            edge_mats = []
            for xy in role_xy:
                edge_mats.append(delaunay_edge_mat(xy).reshape(-1))
            edge_mats = pd.DataFrame(np.stack(edge_mats, axis=0), index=role_x.dropna().index)

            if self.apply_cpd:
                print("\n* Step 2: FormCPD based on role-adjacency matrices")
                sub_dts = pd.to_datetime(player_periods["start_dt"].values[1:])
                form_chg_dts = detect_change_times(edge_mats, sub_dts, mode="form", method=self.formcpd_method)

                # round down chg_dts to the nearest 5-second mark with an offset
                freq_sec = float(freq[:-1])
                offset = session_start_dt.second % freq_sec
                form_chg_dts_rounded = []
                for dt in form_chg_dts:
                    form_chg_dts_rounded.append(dt - timedelta(seconds=(dt.second - offset) % freq_sec))

                print("Detected formation change-points:")
                pprint(form_chg_dts_rounded)

                print("\n* Step 3: RoleCPD per formation period based on role permutations")
                form_chg_dts = [session_start_dt] + form_chg_dts_rounded + [session_end_dt]

            else:
                print("\n* Step 2: Compute the formation graph of the session")
                # assume there are no formation change throughout the session
                form_chg_dts = [session_start_dt, session_end_dt]

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
                mean_xy = np.stack([mean_x, mean_y]).T
                mean_edge_mat = edge_mats[form_start_dt:form_end_dt].mean(axis=0).round(4).values

                # recording the details of the formation period
                form_periods.append(
                    {
                        "session": session,
                        "form_period": form_period,
                        "start_dt": form_start_dt,
                        "end_dt": form_end_dt,
                        "duration": (form_end_dt - form_start_dt).total_seconds(),
                        "coords": mean_xy,
                        "edge_mat": mean_edge_mat.reshape(10, 10),
                    }
                )

                if self.apply_cpd:
                    # recursive change-point detection for the permutation sequence
                    print(f"\nRoleCPD for the formation period {form_period}:")
                    input_perms = perms[form_start_dt:form_end_dt]
                    input_sub_dts = np.array([dt for dt in sub_dts if (dt >= form_start_dt) and (dt < form_end_dt)])
                    role_chg_dts = detect_change_times(
                        input_perms, input_sub_dts, mode="role", method=self.rolecpd_method
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
                                "session": session,
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

        if not self.apply_cpd:
            # finding the most frequent role permutation per 5-minute segment
            perms_str = pd.concat(perm_list)
            bins = self.player_periods["start_dt"].tolist()[1:] + [self.player_periods["end_dt"].iloc[0]]
            perms_str["player_period"] = pd.cut(perms_str.index, bins, labels=self.player_periods.index[1:])

            base_perm_list = []
            for i in self.player_periods.index[1:]:
                period_perms_str = perms_str[perms_str["player_period"] == i]
                if period_perms_str.empty:
                    continue

                session = self.player_periods.at[i, "session"]
                period_perms_str["session"] = session
                period_perms_str["form_period"] = session

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

        # label formation and role periods to the timestamps in role_seq
        match_end_dt = self.player_periods["end_dt"].iloc[-1]
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

        # reflect the instructed roles and recompute switch rates in role_seq
        self.role_seq = self.role_seq.apply(self.reassign_base_role, axis=1)
        self.role_seq = self.role_seq.groupby("datetime", group_keys=False).apply(compute_switch_rate)
        self.role_seq, self.form_periods = SoccerCPD.align_formations(self.role_seq, self.form_periods)
        # self.role_seq = pd.merge(self.role_seq, self.roster[["squad_num", "player_name"]].reset_index())
        self.role_seq.sort_values(by=["player_id", "datetime"], ignore_index=True, inplace=True)

        self.form_periods = self.form_periods.reset_index()[HEADER_FORM_PERIODS]
        self.role_periods = self.role_periods.reset_index()[HEADER_ROLE_PERIODS]
        self.role_summary = self.summarize_role_assignment()
        print()
        print("-" * 73)
        print("Formation Periods:")
        print(self.form_periods[HEADER_FORM_PERIODS[1:-2]])
        print()
        print("Role Periods:")
        print(self.role_periods[HEADER_ROLE_PERIODS[1:-1]])
        print()

    def label_roles(
        self,
        role_benchmarks: pd.DataFrame,
        form_benchmarks: pd.DataFrame = None,
        form_labels: dict = None,
    ):
        assert form_benchmarks is not None or form_labels is not None

        role_labels = dict()
        self.form_periods["formation"] = np.nan
        self.role_summary["aligned_role"] = np.nan

        for i in self.form_periods.index:
            fp = self.form_periods.at[i, "form_period"]

            if form_labels:
                formation = form_labels[fp]
            else:
                form_benchmarks["dist_to_sample"] = 0
                for j in form_benchmarks.index:
                    ref_form = self.form_periods.loc[i]
                    cur_form = form_benchmarks.loc[j]
                    form_benchmarks.at[j, "dist_to_sample"] = compute_delaunay_dists(ref_form, cur_form)
                formation = form_benchmarks.groupby("formation")["dist_to_sample"].mean().idxmin()

            group_role_benchmarks = role_benchmarks[role_benchmarks["formation"] == formation]
            mean_xy = group_role_benchmarks.groupby("aligned_role")[["x", "y"]].mean()

            cost_mat = distance_matrix(mean_xy.values, self.form_periods.at[0, "coords"])
            _, perm = linear_sum_assignment(cost_mat)

            self.form_periods.at[i, "formation"] = formation
            role_labels[fp] = dict(zip(perm + 1, mean_xy.index))
            fp_rs = self.role_summary.loc[self.role_summary["form_period"] == fp]
            self.role_summary.loc[fp_rs.index, "aligned_role"] = fp_rs["base_role"].replace(role_labels[fp])

        self.role_labels = pd.DataFrame(role_labels).T[np.arange(len(perm)) + 1]

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

    def visualize(self, match_id: int = None, roster: pd.DataFrame = None, role_labels=None, save=False):
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
        import seaborn as sns

        # sns.set(font="Arial", rc={"axes.unicode_minus": False}, font_scale=1.5)
        sns.set(font_scale=1.5)

        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        gs = gridspec.GridSpec(2, 4, left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)

        for idx, form_period in enumerate(self.form_periods["form_period"][:4]):
            fp_role_seq = self.role_seq[(self.role_seq["form_period"] == form_period) & (self.role_seq["role"].notna())]
            fp_formation = self.form_periods.loc[idx]
            fp_role_periods = self.role_periods[self.role_periods["form_period"] == form_period]
            fp_role_labels = role_labels.loc[form_period].to_dict() if role_labels is not None else None

            plt.subplot(gs[0, idx])
            plot_graph(fp_role_seq, fp_formation, fp_role_labels)

            start_rp = fp_role_periods["role_period"].min()
            if len(fp_role_periods) == 1:
                plt.title(f"Role Period {start_rp}", fontsize=18)
            else:
                end_rp = fp_role_periods["role_period"].max()
                plt.title(f"Role Periods {start_rp}-{end_rp}", fontsize=18)

        ax = fig.add_subplot(gs[1, :])
        plot_timeline(self.role_seq, roster, ax)
        plt.title("Role Change Timeline", fontsize=18)

        if save:
            assert match_id is not None

            report_dir = f"{self.target_dir}/viz_report"
            report_path = f"{report_dir}/{match_id}.png"
            if not os.path.exists(f"{self.target_dir}"):
                os.mkdir(f"{self.target_dir}")
            if not os.path.exists(report_dir):
                os.mkdir(report_dir)

            plt.savefig(report_path, bbox_inches="tight")
            plt.close(fig)
            print(f"'{report_path}' saving done.")

        else:
            plt.show()
            plt.close(fig)

    def save_stats(self, match_id: int, form_summary=True, role_summary=True, role_seq=True):
        if not os.path.exists(f"{self.target_dir}"):
            os.mkdir(f"{self.target_dir}")

        # save form_periods
        if form_summary:
            form_summary_dir = f"{self.target_dir}/form_summary"
            if not os.path.exists(form_summary_dir):
                os.mkdir(form_summary_dir)
            form_summary_path = f"{form_summary_dir}/{match_id}.pkl"
            self.form_periods.to_pickle(form_summary_path)
            print(f"'{form_summary_path}' saving done.")

        # save role_summary
        if role_summary:
            role_summary_dir = f"{self.target_dir}/role_summary"
            if not os.path.exists(role_summary_dir):
                os.mkdir(role_summary_dir)
            role_summary_path = f"{role_summary_dir}/{match_id}.csv"
            self.role_summary.to_csv(role_summary_path, index=False, encoding="utf-8-sig")
            print(f"'{role_summary_path}' saving done.")

        # save role_seq
        if role_seq:
            role_seq_dir = f"{self.target_dir}/role_seq"
            if not os.path.exists(role_seq_dir):
                os.mkdir(role_seq_dir)
            role_seq_path = f"{role_seq_dir}/{match_id}.csv"
            self.role_seq.to_csv(role_seq_path, index=False, encoding="utf-8-sig")
            print(f"'{role_seq_path}' saving done.")
