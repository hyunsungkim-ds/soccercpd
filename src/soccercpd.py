import os
from collections import Counter
from datetime import datetime, timedelta
from pprint import pprint

import numpy as np
import pandas as pd
import rpy2.rinterface_lib.embedded as rembedded
import rpy2.robjects as robjects
import ruptures as rpt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay, distance_matrix
from sklearn.metrics import pairwise_distances

from src.myconstants import *
from src.rolerep import RoleRep

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 20)


# formation and role change-point detection (main algorithm)
class SoccerCPD:
    def __init__(
        self,
        activity_id: int,
        team_roster: pd.DataFrame,
        team_traces: pd.DataFrame,
        apply_cpd=True,
        formcpd_type="gseg_avg",  # ["gseg_avg", "gseg_union", "kernel_linear", "kernel_rbf", "kernel_cosine", "rank"]
        rolecpd_type="gseg_avg",  # ["gseg_avg", "gseg_union"]
        max_sr=MAX_SWITCH_RATE,
        max_pval=MAX_PVAL,
        min_pdur=MIN_PERIOD_DUR,
        min_fdist=MIN_FORM_DIST,
    ):
        self.activity_id = activity_id
        self.roster = team_roster
        self.traces = SoccerCPD.reshape_traces(team_traces)
        self.phase_records = SoccerCPD.construct_phase_records(team_traces)

        self.apply_cpd = apply_cpd
        self.formcpd_type = formcpd_type
        self.rolecpd_type = rolecpd_type

        self.max_sr = max_sr
        self.max_pval = max_pval
        self.min_pdur = min_pdur
        self.min_fdist = min_fdist

        self.role_df = pd.DataFrame(columns=["datetime"] + HEADER_ROLES)
        self.form_periods = pd.DataFrame(columns=HEADER_FORM_PERIODS)
        self.role_periods = pd.DataFrame(columns=HEADER_ROLE_PERIODS)
        self.role_summary = None

        self.target_dir = f"{DIR_DATA}/{formcpd_type}" if apply_cpd else f"{DIR_DATA}/noncpd"

    @staticmethod
    def reshape_traces(team_traces: pd.DataFrame):
        trace_list = []
        players = [c[:3] for c in team_traces.columns if c[3:] == "_x"]

        for p in players:
            cols = ["datetime", "session", "time", "phase", f"{p}_x", f"{p}_y"]
            player_trace = team_traces[cols].copy().rename(columns={f"{p}_x": "x", f"{p}_y": "y"})
            player_trace["player_code"] = p
            trace_list.append(player_trace)

        return pd.concat(trace_list).set_index("datetime")

    @staticmethod
    def construct_phase_records(team_traces: pd.DataFrame):
        grouped = team_traces.groupby("phase")
        sessions = grouped["session"].first()
        start_dts = (grouped["datetime"].first() - timedelta(seconds=0.1)).rename("start_dt")
        end_dts = grouped["datetime"].last().rename("end_dt")
        return pd.concat([sessions, start_dts, end_dts], axis=1)

    # apply Delaunay triangulation to the given player coordinates to obtain the role-adjacency matrix
    @staticmethod
    def delaunay_edge_mat(coords):
        tri_pts = Delaunay(coords).simplices
        edges = np.concatenate((tri_pts[:, :2], tri_pts[:, 1:], tri_pts[:, ::2]), axis=0)
        edge_mat = np.zeros((coords.shape[0], coords.shape[0]))
        edge_mat[edges[:, 0], edges[:, 1]] = 1
        return np.clip(edge_mat + edge_mat.T, 0, 1)

    @staticmethod
    def complete_perm(perm, role_set):
        if perm.isnull().sum():
            return perm.fillna(list(role_set - set(perm.dropna()))[0])
        else:
            return perm

    @staticmethod
    def hamming(perm1, perm2):
        return (perm1 != perm2).astype(int).sum()

    @staticmethod
    def manhattan(mat1, mat2):
        return np.abs(mat1 - mat2).sum()

    # recursive change-point detection for the input sequence
    def detect_change_times(self, input_seq, sub_dts, mode="form"):
        # if mode == "form" (FormCPD), the input is a sequence of role-adjacency matrices
        # if mode == "role" (RoleCPD), the input a sequence of role permutations

        start_time = input_seq.index[0].time()
        end_time = input_seq.index[-1].time()

        if (mode == "role") or ("gseg" in self.formcpd_type):
            metric = SoccerCPD.manhattan if mode == "form" else SoccerCPD.hamming
            dists = pd.DataFrame(pairwise_distances(input_seq.drop_duplicates(), metric=metric))

            # save the input sequence and the pairwise distances so that we can use them in the R script below
            if not os.path.exists(DIR_TEMP_DATA):
                os.mkdir(DIR_TEMP_DATA)
            input_seq.to_csv(f"{DIR_TEMP_DATA}/{self.activity_id}_temp_seq.csv", index=False)
            dists.to_csv(f"{DIR_TEMP_DATA}/{self.activity_id}_temp_dists.csv", index=False)

            try:
                print(f"Applying g-segmentation to the sequence between {start_time} and {end_time}...")

                if mode == "form":
                    gseg_type = self.formcpd_type.split("_")[1][0]
                else:
                    gseg_type = self.rolecpd_type.split("_")[1][0]

                # run the R function "gseg1_discrete" to find a change-point
                # rpackages.importr("gSeg", lib_loc=rpackages.importr("base")._libPaths()[0])
                robjects.r(
                    f"""
                    dir = '{DIR_TEMP_DATA}'
                    seq_path = paste(dir, '{self.activity_id}_temp_seq.csv', sep='/')
                    seq = read.csv(seq_path)
                    dists_path = paste(dir, '{self.activity_id}_temp_dists.csv', sep='/')
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
            if robjects.r["pval"][0] >= self.max_pval:
                print("Change-point insignificant: The p-value is not small enough.\n")
                return []
            else:
                chg_idx = robjects.r["chg_idx"][0]

        elif "kernel" in self.formcpd_type:
            print(f"Applying kernel-based CPD to the sequence between {start_time} and {end_time}...")
            kernel_type = self.formcpd_type.split("_")[1]
            algo = rpt.Binseg(model=kernel_type).fit(input_seq.values)
            chg_idx = algo.predict(n_bkps=1)[0]

        elif "rank" in self.formcpd_type:
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
        if (len(seq1) < self.min_pdur) or (len(seq2) < self.min_pdur):
            print("Change-point insignificant: One of the periods has not enough duration.\n")
            return []

        if mode == "form":
            # condition (3) for FormCPD: The respective mean role-adjacency matrices
            # from the segments before and after chg_dt are far enough from each other
            form1_edge_mat = seq1.mean(axis=0).values
            form2_edge_mat = seq2.mean(axis=0).values
            if self.manhattan(form1_edge_mat, form2_edge_mat) < self.min_fdist:
                print("Change-point insignificant: The formation is not changed.\n")
                return []
            else:
                # if significant, recursively detect another change-points before and after chg_dt
                print(f"A significant fine-tuned change-point at {chg_dt.time()}.\n")
                prev_chg_dts = self.detect_change_times(seq1, sub_dts)
                next_chg_dts = self.detect_change_times(seq2, sub_dts)
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
                prev_chg_dts = self.detect_change_times(seq1, sub_dts)
                next_chg_dts = self.detect_change_times(seq2, sub_dts)
                return prev_chg_dts + [chg_dt] + next_chg_dts

        else:
            raise ValueError("Invalid mode")

    # align corresponding roles from different formation periods
    @staticmethod
    def align_formations(role_df, form_period_records):
        base_form_period_record = form_period_records.iloc[0]

        for form_period in form_period_records.index[1:]:
            cur_form_period_record = form_period_records.loc[form_period]
            cost_mat = distance_matrix(
                base_form_period_record["coords"],
                cur_form_period_record["coords"],
            )
            _, perm = linear_sum_assignment(cost_mat)
            form_period_records.at[form_period, "coords"] = cur_form_period_record["coords"][perm]
            form_period_records.at[form_period, "edge_mat"] = cur_form_period_record["edge_mat"][perm][:, perm]

            inverse_perm = dict(zip(np.array(perm) + 1, np.arange(10) + 1))
            fp_role_df = role_df[role_df["form_period"] == form_period]
            for col in ["role", "base_role"]:
                role_df.loc[fp_role_df.index, col] = fp_role_df[col].apply(lambda role: inverse_perm[role])

        return role_df, form_period_records

    @staticmethod
    def most_common(player_roles):
        try:
            counter = Counter(player_roles[player_roles.notna()])
            return counter.most_common(1)[0][0]
        except IndexError:
            return np.nan

    def reassign_base_role(self, role_row):
        base_perm = self.role_periods.at[role_row["role_period"], "base_perm"]
        role_row["base_role"] = base_perm[role_row["base_role"]]
        return role_row

    # recompute the "switch rate" of a frame by the Hamming distance
    # between the temporary roles and the instructed roles
    @staticmethod
    def recompute_switch_rate(moment_role_df):
        hamming = SoccerCPD.hamming(moment_role_df["role"], moment_role_df["base_role"])
        moment_role_df["switch_rate"] = hamming / len(moment_role_df)
        return moment_role_df

    # refind base roles per player period and recompute switch rate per frame for the precomputed role details
    def reset_precomputed_role_df(self):
        for i in self.phase_records.index[1:]:
            phase_role_df = self.role_df[self.role_df["phase"] == i]
            perms = phase_role_df.pivot_table("role", "datetime", "player_code", "first")
            if perms.empty:
                continue
            # role_set = set(perms.dropna().iloc[0])
            role_set = set(np.arange(10) + 1)
            perms = perms.apply(SoccerCPD.complete_perm, axis=1, args=(role_set,)).astype(int)
            perms_str = perms.apply(lambda perm: np.array2string(perm.values), axis=1)
            base_perm_list = np.fromstring(SoccerCPD.most_common(perms_str)[1:-1], dtype="float32", sep=" ")
            base_perm_dict = dict(zip(perms.columns, base_perm_list))
            self.role_df.loc[phase_role_df.index, "base_role"] = phase_role_df["player_code"].map(base_perm_dict)
        self.role_df = self.role_df.groupby("datetime", group_keys=False).apply(SoccerCPD.recompute_switch_rate)

    def construct_role_summary(self):
        grouped = self.role_df.groupby(["player_code", "role_period"], group_keys=False, as_index=False)
        role_summary = grouped[["phase", "base_role"]].first()
        role_summary = pd.merge(role_summary, self.role_periods[HEADER_ROLE_PERIODS[:-1]])

        role_summary = pd.merge(role_summary, self.form_periods[["form_period", "coords"]])
        role_summary["x"] = role_summary.apply(lambda x: x["coords"][x["base_role"] - 1, 0], axis=1)
        role_summary["y"] = role_summary.apply(lambda x: x["coords"][x["base_role"] - 1, 1], axis=1)

        role_summary = pd.merge(role_summary, self.roster[["squad_num", "player_name"]].reset_index())
        return role_summary[HEADER_ROLE_SUMMARY].astype({"phase": int})

    def run(self, use_precomputed=True, freq="1S"):
        role_path = f"data/{self.formcpd_type}/role_details/{self.activity_id}.csv"
        form_periods = []
        role_periods = []

        # if self.use_precomputed == True, load and initialize the precomputed role details
        if use_precomputed and os.path.exists(role_path):
            self.role_df = pd.read_csv(role_path, header=0, encoding="utf-8-sig")
            self.role_df["datetime"] = self.role_df["datetime"].apply(
                lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
            )
            self.reset_precomputed_role_df()

        # initialize formation and role period labels by the session labels
        self.traces["form_period"] = self.traces["session"]
        self.traces["role_period"] = self.traces["session"]

        role_list = []
        perm_list = []

        for session in self.traces["session"].unique():
            print(f"\n{'-' * 33} Session {session} {'-' * 34}")
            player_periods = self.phase_records[self.phase_records["session"] == session]
            session_start_dt = pd.to_datetime(player_periods["start_dt"].iloc[0])
            session_end_dt = pd.to_datetime(player_periods["end_dt"].iloc[-1])
            session_traces = self.traces[self.traces["session"] == session]

            grouper = session_traces.dropna(subset="x").groupby("time", group_keys=False)
            if grouper["player_code"].apply(len).max() < 10:
                # if less than 10 players have been measured during the session, skip the process
                print("Not enough players to estimate a formation.")
                continue
            else:
                # print(player_periods[HEADER_PLAYER_PERIODS[2:7]])
                print(player_periods)

            if use_precomputed and not self.role_df.empty:
                print("\n* Step 1: Load the pre-computed role assignment result")
                session_role_df = self.role_df[self.role_df["session"] == session]
                print(f"Session role details loaded and filtered from '{role_path}'.")
            else:
                print("\n* Step 1: Frame-by-frame role assignment using RoleRep")
                rolerep = RoleRep(session_traces)
                session_role_df = rolerep.run(freq=freq)

            # exclude situations such as set-pieces that are irrelevant to the team formation
            valid_role_df = session_role_df[session_role_df["switch_rate"] <= self.max_sr]

            # check whether all the 10 outfield players are measured for some periods
            role_x = valid_role_df.pivot_table("x_norm", "datetime", "role", aggfunc="first")
            role_y = valid_role_df.pivot_table("y_norm", "datetime", "role", aggfunc="first")
            role_coords = np.dstack([role_x.dropna().values, role_y.dropna().values])
            if role_coords.shape[1] < 10:
                print("Not enough players to estimate a formation.")
                continue
            else:
                role_list.append(session_role_df)

            # generate the sequence of role-adjacency matrices
            edge_mats = []
            for coords in role_coords:
                edge_mats.append(SoccerCPD.delaunay_edge_mat(coords).reshape(-1))
            edge_mats = pd.DataFrame(np.stack(edge_mats, axis=0), index=role_x.dropna().index)

            if self.apply_cpd:
                print("\n* Step 2: FormCPD based on role-adjacency matrices")
                sub_dts = pd.to_datetime(player_periods["start_dt"].values[1:])
                form_chg_dts = self.detect_change_times(edge_mats, sub_dts, mode="form")

                print("Detected formation change-points (rounded off to the nearest 10 second mark):")
                form_chg_dts_rounded = []
                for dt in form_chg_dts:
                    form_chg_dts_rounded.append(dt - timedelta(seconds=dt.second % 10))
                pprint(form_chg_dts_rounded)

                print("\n* Step 3: RoleCPD per formation period based on role permutations")
                form_chg_dts = [session_start_dt] + form_chg_dts_rounded + [session_end_dt]

            else:
                print("\n* Step 2: Compute the formation graph of the session")
                # assume there are no formation change throughout the session
                form_chg_dts = [session_start_dt, session_end_dt]

                print("\n* Step 3: Find the most frequent role permutation per 5-minute segment")

            # generate the sequence of role permutations
            perms = valid_role_df.pivot_table("base_role", "datetime", "role", aggfunc="first")
            role_set = set(perms.dropna().iloc[0])
            perms = perms.apply(SoccerCPD.complete_perm, axis=1, args=(role_set,)).astype(int)
            perms_str = perms.apply(lambda perm: np.array2string(perm.values), axis=1)
            perm_list.append(perms_str.rename("perm").to_frame())

            for form_chg_idx in range(1, len(form_chg_dts)):
                form_period = len(form_periods) + 1
                form_start_dt = form_chg_dts[form_chg_idx - 1]
                form_end_dt = form_chg_dts[form_chg_idx]

                mean_x = role_x[form_start_dt:form_end_dt].dropna().mean(axis=0).round().values
                mean_y = role_y[form_start_dt:form_end_dt].dropna().mean(axis=0).round().values
                mean_coords = np.stack([mean_x, mean_y]).T
                mean_edge_mat = edge_mats[form_start_dt:form_end_dt].mean(axis=0).round(3).values

                # recording the details of the formation period
                form_periods.append(
                    {
                        "activity_id": self.activity_id,
                        "session": session,
                        "form_period": form_period,
                        "start_dt": form_start_dt,
                        "end_dt": form_end_dt,
                        "duration": (form_end_dt - form_start_dt).total_seconds(),
                        "coords": mean_coords,
                        "edge_mat": mean_edge_mat.reshape(10, 10),
                    }
                )

                if self.apply_cpd:
                    # recursive change-point detection for the permutation sequence
                    print(f"\nRoleCPD for the formation period {form_period}:")
                    input_perms = perms[form_start_dt:form_end_dt]
                    input_sub_dts = np.array([dt for dt in sub_dts if (dt >= form_start_dt) and (dt < form_end_dt)])
                    role_chg_dts = self.detect_change_times(input_perms, input_sub_dts, mode="role")

                    print("Detected role change-points (rounded off to the nearset 10 second mark):")
                    role_chg_dts_rounded = []
                    for dt in role_chg_dts:
                        role_chg_dts_rounded.append(dt - timedelta(seconds=dt.second % 10))
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
                                "activity_id": self.activity_id,
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
            self.role_df = pd.concat(role_list, ignore_index=True)
        else:
            return

        if not self.apply_cpd:
            # finding the most frequent role permutation per 5-minute segment
            perms_str = pd.concat(perm_list)
            bins = self.phase_records["start_dt"].tolist()[1:] + [self.phase_records["end_dt"].iloc[0]]
            perms_str["phase"] = pd.cut(perms_str.index, bins, labels=self.phase_records.index[1:])

            base_perm_list = []
            for i in self.phase_records.index[1:]:
                period_perms_str = perms_str[perms_str["phase"] == i]
                if period_perms_str.empty:
                    continue

                session = self.phase_records.at[i, "session"]
                period_perms_str["session"] = session
                period_perms_str["form_period"] = session

                period_start_dt = self.phase_records.at[i, "start_dt"]
                offset = f"{period_start_dt.minute * SCALAR_TIME + period_start_dt.second}S"
                resampler = period_perms_str.resample("5T", closed="right", offset=offset)

                base_perms = resampler.apply(SoccerCPD.most_common).reset_index()
                base_perms["end_dt"] = base_perms["datetime"].shift(-1)
                base_perms.iat[-1, -1] = self.phase_records.at[i, "end_dt"]
                base_perm_list.append(base_perms)

            role_periods = pd.concat(base_perm_list, ignore_index=True)
            role_periods.rename(columns={"datetime": "start_dt"}, inplace=True)

            perms_list = role_periods["perm"].apply(lambda x: np.fromstring(x[1:-1], dtype=int, sep=" "))
            role_periods["base_perm"] = perms_list.apply(lambda perm: dict(zip(np.arange(10) + 1, perm)))

            role_periods["activity_id"] = self.activity_id
            role_periods["role_period"] = role_periods.index + 1
            role_periods["duration"] = (role_periods["end_dt"] - role_periods["start_dt"]).apply(
                lambda td: td.total_seconds()
            )
            self.role_periods = role_periods[HEADER_ROLE_PERIODS]

        self.form_periods = pd.DataFrame(form_periods).set_index("form_period")
        self.role_periods = pd.DataFrame(role_periods).set_index("role_period")

        # label formation and role periods to the timestamps in role_df
        match_end_dt = self.phase_records["end_dt"].iloc[-1]
        form_bins = self.form_periods["start_dt"].tolist() + [match_end_dt]
        role_bins = self.role_periods["start_dt"].tolist() + [match_end_dt]
        self.role_df["form_period"] = pd.cut(self.role_df["datetime"], bins=form_bins, labels=self.form_periods.index)
        self.role_df["role_period"] = pd.cut(self.role_df["datetime"], bins=role_bins, labels=self.role_periods.index)

        # reflect the instructed roles and recompute switch rates in role_df
        self.role_df = self.role_df.apply(self.reassign_base_role, axis=1)
        self.role_df = self.role_df.groupby("datetime", group_keys=False).apply(SoccerCPD.recompute_switch_rate)
        self.role_df, self.form_periods = SoccerCPD.align_formations(self.role_df, self.form_periods)
        self.role_df = pd.merge(self.role_df, self.roster[["squad_num", "player_name"]].reset_index())
        self.role_df.sort_values(by=["player_code", "datetime"], ignore_index=True, inplace=True)

        self.form_periods = self.form_periods.reset_index()[HEADER_FORM_PERIODS]
        self.role_periods = self.role_periods.reset_index()[HEADER_ROLE_PERIODS]
        self.role_summary = self.construct_role_summary()
        print()
        print("-" * 78)
        print("Formation Periods:")
        print(self.form_periods[HEADER_FORM_PERIODS[1:-2]])
        print()
        print("Role Periods:")
        print(self.role_periods[HEADER_ROLE_PERIODS[1:-1]])
        print()

    @staticmethod
    def seconds_to_time_str(x: float):
        minutes = int(x // 60)
        seconds = int(x % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def visualize(self, role_labels=None):
        import math

        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
        import seaborn as sns

        # sns.set(font="Arial", rc={"axes.unicode_minus": False}, font_scale=1.5)
        sns.set(font_scale=1.5)

        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        gs = gridspec.GridSpec(2, 4, left=0.05, right=0.95, wspace=0.3, hspace=0.1)

        xlim = 25
        ylim = 30

        for idx, form_period in enumerate(self.form_periods["form_period"][:4]):
            fp_role_df = self.role_df[(self.role_df["form_period"] == form_period) & (self.role_df["role"].notna())]
            role_coords = np.dot(self.form_periods.at[idx, "coords"], [[0, 1], [-1, 0]])
            edge_mat = self.form_periods.at[idx, "edge_mat"]

            plt.subplot(gs[0, idx])
            role_period_records = self.role_periods[self.role_periods["form_period"] == form_period]
            role_period_from = role_period_records["role_period"].iloc[0]
            if len(role_period_records) == 1:
                plt.title(f"Role Period {role_period_from}", fontsize=20)
            else:
                role_period_to = role_period_records["role_period"].iloc[-1]
                plt.title(f"Role Periods {role_period_from}-{role_period_to}", fontsize=20)

            plt.scatter(
                -fp_role_df["y_norm"],
                fp_role_df["x_norm"],
                c=fp_role_df["role"],
                vmin=0.5,
                vmax=10.5,
                cmap="tab10",
                alpha=0.4,
                zorder=0,
            )
            plt.scatter(
                role_coords[:, 0],
                role_coords[:, 1],
                s=1200,
                c="w",
                edgecolors="k",
                zorder=2,
            )

            for r in np.arange(10):
                role_label = role_labels[form_period][r] if role_labels is not None else r + 1
                plt.annotate(
                    role_label,
                    xy=role_coords[r],
                    ha="center",
                    va="center",
                    fontsize=15,
                    zorder=3,
                )
                for s in np.arange(10):
                    plt.plot(
                        role_coords[[r, s], 0],
                        role_coords[[r, s], 1],
                        linewidth=edge_mat[r, s] ** 2 * 4,
                        c="k",
                        zorder=1,
                    )

            plt.xlim(-xlim, xlim)
            plt.ylim(-ylim, ylim)
            plt.vlines([-xlim, xlim], ymin=-ylim, ymax=ylim, color="k")
            plt.hlines([-ylim, 0, ylim], xmin=-xlim, xmax=xlim, color="k", zorder=1)
            plt.axis("off")

        ax = fig.add_subplot(gs[1, :])
        box = ax.get_position()
        ax.set_position([box.x0 + box.width * 0.15, box.y0 + box.height * 0.03, box.width * 0.8, box.height * 0.95])
        plt.title("Timeline of Instructed Roles", fontsize=20)

        roles_reshaped = self.role_df.pivot_table("base_role", "datetime", "player_code", aggfunc="first")
        times = self.role_df[["datetime", "session", "time"]].drop_duplicates().sort_values(["session", "time"])
        roles_reshaped = pd.merge(times, roles_reshaped.reset_index())
        roles_resampled = []

        for s in self.phase_records["session"].unique():
            start_dt = self.phase_records.loc[self.phase_records["session"] == s, "start_dt"].iloc[0]
            offset = f"{start_dt.second % 5}S"

            session_roles_reshaped = roles_reshaped[roles_reshaped["session"] == s].set_index("datetime")
            session_roles_reshaped = session_roles_reshaped.resample("5S", offset=offset).first()
            session_roles_reshaped.at[session_roles_reshaped.index[0], "time"] = 0
            roles_resampled.append(session_roles_reshaped)

        self.roster["display"] = self.roster.apply(lambda x: f"{x['player_name']} ({x['squad_num']})", axis=1)
        player_dict = self.roster["display"].to_dict()
        roles_resampled = pd.concat(roles_resampled).rename(columns=player_dict)
        players = [c for c in roles_resampled.columns if c[-1] == ")"]
        sns.heatmap(roles_resampled[players].T, vmin=0.5, vmax=10.5, cmap="tab10", cbar=False)

        xticks = []
        for dt in self.role_periods["start_dt"]:
            xticks.append(roles_resampled.index.get_loc(dt))
        xticks.append(len(roles_resampled) - 1)

        session_labels = roles_resampled["session"].iloc[xticks].apply(lambda x: f"H{x}-")
        time_labels = roles_resampled["time"].iloc[xticks].apply(SoccerCPD.seconds_to_time_str)
        labels = (session_labels + time_labels).values

        ax.vlines(xticks, ymin=0, ymax=len(players), colors="k", linestyles="--")
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_xlabel("session-time")
        ax.set_ylabel("player")

        report_dir = f"{self.target_dir}/report"
        report_path = f"{report_dir}/{self.activity_id}.png"
        if not os.path.exists(f"{self.target_dir}"):
            os.mkdir(f"{self.target_dir}")
        if not os.path.exists(report_dir):
            os.mkdir(report_dir)

        plt.savefig(report_path)
        plt.close(fig)
        print(f"'{report_path}' saving done.")

    def save_stats(self, form_summary=True, role_summary=True, role_details=True):
        if not os.path.exists(f"{self.target_dir}"):
            os.mkdir(f"{self.target_dir}")

        # save form_periods
        if form_summary:
            form_summary_dir = f"{self.target_dir}/form_summary"
            if not os.path.exists(form_summary_dir):
                os.mkdir(form_summary_dir)
            form_summary_path = f"{form_summary_dir}/{self.activity_id}.pkl"
            self.form_periods.to_pickle(form_summary_path)
            print(f"'{form_summary_path}' saving done.")

        # save role_summary
        if role_summary:
            role_summary_dir = f"{self.target_dir}/role_summary"
            if not os.path.exists(role_summary_dir):
                os.mkdir(role_summary_dir)
            role_summary_path = f"{role_summary_dir}/{self.activity_id}.csv"
            self.role_summary.to_csv(role_summary_path, index=False, encoding="utf-8-sig")
            print(f"'{role_summary_path}' saving done.")

        # save role_df
        if role_details:
            role_details_dir = f"{self.target_dir}/role_details"
            if not os.path.exists(role_details_dir):
                os.mkdir(role_details_dir)
            role_details_path = f"{role_details_dir}/{self.activity_id}.csv"
            self.role_df.to_csv(role_details_path, index=False, encoding="utf-8-sig")
            print(f"'{role_details_path}' saving done.")
