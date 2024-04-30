import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy.stats import multivariate_normal

from src.myconstants import *

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 20)


# frame-by-frame role assignment proposed by Bialkowski et al. (2014)
class RoleRep:
    def __init__(self, ugp_: pd.DataFrame):
        self.ugp = ugp_
        self.fgp = None
        self.role_distns = None

    @staticmethod
    def normalize_locs(moment_fgp: pd.DataFrame) -> pd.DataFrame:
        locs = moment_fgp[["x", "y"]]
        moment_fgp[["x_norm", "y_norm"]] = locs - locs.mean()
        return moment_fgp

    @staticmethod
    def init_fgp(ugp: pd.DataFrame, freq="1S") -> pd.DataFrame:
        ugp = ugp[ugp["x"].notna()]
        fgp = []

        for i, player_id in enumerate(ugp["player_id"].unique()):
            player_ugp = ugp[ugp["player_id"] == player_id]
            resampler = player_ugp.resample(freq, closed="right", label="right")
            player_fgp = resampler[HEADER_ROLE_DETAILS[:4]].last()
            player_fgp["x"] = resampler["x"].mean()
            player_fgp["y"] = resampler["y"].mean()
            player_fgp["x_norm"] = np.nan
            player_fgp["y_norm"] = np.nan
            player_fgp["form_period"] = resampler["form_period"].last()
            player_fgp["role_period"] = resampler["role_period"].last()
            player_fgp["role"] = i + 1
            player_fgp["base_role"] = i + 1
            player_fgp["switch_rate"] = 0
            fgp.append(player_fgp[HEADER_ROLE_DETAILS])

        fgp = pd.concat(fgp).reset_index().rename(columns={"index": "datetime"})
        return fgp.groupby("datetime", group_keys=False).apply(RoleRep.normalize_locs)

    @staticmethod
    def estimate_mvn(df, col_x="x_norm", col_y="y_norm", filter=True):
        if filter:
            coords = df[df["switch_rate"] <= MAX_SWITCH_RATE][[col_x, col_y]]
        else:
            coords = df[[col_x, col_y]]

        if filter and len(coords) < 30:
            return np.nan
        else:
            return multivariate_normal(coords.mean(), coords.cov())

    @staticmethod
    def update_params(fgp: pd.DataFrame, by_player_period=False) -> pd.DataFrame:
        cols = ["player_period", "role"] if by_player_period else ["role"]
        role_distns = fgp.groupby(cols).apply(RoleRep.estimate_mvn).reset_index()
        return role_distns.dropna().rename(columns={0: "distn"})

    @staticmethod
    def align_formations(fgp: pd.DataFrame, role_distns: pd.DataFrame, label_group="session"):
        groups = fgp[label_group].unique()
        base_group = groups[role_distns.groupby(label_group)["role"].count().argmax()]
        base_role_distns = role_distns[role_distns[label_group] == base_group]

        for group in groups:
            if group == base_group:
                continue
            group_role_distns = role_distns[role_distns[label_group] == group]
            cost_mat = distance_matrix(
                group_role_distns["distn"].apply(lambda x: pd.Series(x.mean)).values,
                base_role_distns["distn"].apply(lambda x: pd.Series(x.mean)).values,
            )
            row_idx, col_idx = linear_sum_assignment(cost_mat)
            role_dict = dict(zip(group_role_distns["role"].iloc[row_idx], base_role_distns["role"].iloc[col_idx]))
            role_dict[0] = 0
            role_distns.loc[role_distns[label_group] == group, "role"] = col_idx + 1
            fgp.loc[fgp[label_group] == group, "role"] = fgp.loc[fgp[label_group] == group, "role"].apply(
                lambda role: role_dict[role]
            )
            fgp.loc[fgp[label_group] == group, "base_role"] = fgp.loc[fgp[label_group] == group, "base_role"].apply(
                lambda role: role_dict[role]
            )

        return fgp, role_distns.sort_values(by=[label_group, "role"]).reset_index(drop=True)

    def hungarian(self, moment_fgp: pd.DataFrame, role_distns: pd.DataFrame) -> float:
        cost_mat = moment_fgp[moment_fgp.columns[(len(HEADER_ROLE_DETAILS) + 1) :]].values
        row_idx, col_idx = linear_sum_assignment(cost_mat)
        base_roles = moment_fgp["base_role"].iloc[row_idx].values
        temp_roles = role_distns["role"].iloc[col_idx].values
        self.fgp.loc[moment_fgp.index, "role"] = temp_roles
        self.fgp.loc[moment_fgp.index, "switch_rate"] = (base_roles != temp_roles).sum() / len(row_idx)
        return cost_mat[row_idx, col_idx].mean()

    def run(self, freq="1S", verbose=True) -> pd.DataFrame:
        temp_fgp = self.ugp.groupby("player_period").apply(RoleRep.init_fgp, freq=freq)
        temp_fgp = temp_fgp.reset_index(drop=True).dropna()
        temp_role_distns = RoleRep.update_params(temp_fgp, by_player_period=True)
        temp_fgp = pd.merge(temp_fgp, temp_role_distns[["player_period", "role"]])
        self.fgp, _ = RoleRep.align_formations(temp_fgp, temp_role_distns, "player_period")
        self.role_distns = RoleRep.update_params(self.fgp)

        max_iter = 10
        cost_prev = float("inf")
        tol = 0.005
        self.fgp.reset_index(drop=True, inplace=True)

        for i_iter in range(max_iter):
            cost_df = pd.DataFrame(
                self.role_distns["distn"]
                .apply(lambda n: pd.Series(-np.log(n.pdf(self.fgp[["x_norm", "y_norm"]]))))
                .transpose()
                .values,
                index=self.fgp.index,
            )
            fgp_cost_df = pd.concat([self.fgp, cost_df], axis=1)
            costs = fgp_cost_df.groupby("datetime").apply(self.hungarian, self.role_distns).mean()
            cost_new = costs.mean()
            if verbose:
                print("- Cost after iteration {0}: {1:.3f}".format(i_iter + 1, cost_new))
            self.role_distns = RoleRep.update_params(self.fgp)
            if cost_new + tol > cost_prev:
                if verbose:
                    print("Iteration finished since there are no significant changes.")
                    break
            cost_prev = cost_new

        session = self.ugp["session"].iloc[0]
        self.role_distns["session"] = session

        return self.fgp


# if __name__ == '__main__':
#     rm = RecordManager()
#     activity_ids = [int(os.path.splitext(f)[0]) for f in os.listdir(DIR_UGP_DATA) if f.endswith('.ugp')]
#     activity_records = rm.activity_records[(rm.activity_records[LABEL_DATA_SAVED] == 1) &
#                                            (rm.activity_records[LABEL_STATS_SAVED] == 0)]
#     print()
#     print('Activity Records:')
#     print(activity_records)
#
#     for i in activity_records.index:
#         activity_id = activity_records.at[i, "activity_id"]
#         date = activity_records.at[i, LABEL_DATE]
#         team_name = activity_records.at[i, LABEL_TEAM_NAME]
#         print()
#         print(f'[{i}] activity_id: {activity_id}, date: {date}, team_name: {team_name}')
#
#         activity_args = rm.load_activity_data(activity_id)
#         match = Match(*activity_args)
#         if match.player_periods["players"].iloc[1:].apply(len).max() >= 10:
#             match.construct_inplay_df()
#             match.rotate_pitch()
#
#             match.player_periods["form_period"] = match.player_periods["session"]
#             match.player_periods["role_period"] = match.player_periods["session"]
#             match.ugp["form_period"] = match.ugp["session"]
#             match.ugp["role_period"] = match.ugp["session"]
#             match_role_distns = pd.DataFrame(columns=HEADER_ROLE_RECORDS)
#
#             match_fgp = pd.DataFrame(columns=["datetime"] + HEADER_FGP)
#             for j in match.ugp["role_period"].unique():
#                 form_ugp = match.ugp[match.ugp["role_period"] == j]
#                 rolerep = RoleRep(form_ugp)
#                 print(f'\nRunning RoleRep for session {j}...')
#                 rolerep.run(freq='1S')
#                 match_role_distns = match_role_distns.append(rolerep.role_distns, sort=True)
#                 match_fgp = match_fgp.append(rolerep.fgp)
#
#             match_fgp_path = f'{DIR_DATA}/fgp_avg/{activity_id}.csv'
#             match_fgp.to_csv(match_fgp_path, index=False, encoding='utf-8-sig')
#             print(f"'{match_fgp_path}' saving done.")
#
#         else:
#             print('Not enough players to estimate a formation.')
#             continue
#
#         match_role_distns["activity_id"] = activity_id
#         match_role_distns = match_role_distns[["activity_id"] + HEADER_ROLE_RECORDS]
#         print()
#         print(match_role_distns)
#
#         # rm.activity_records.at[i, LABEL_STATS_SAVED] = 1
#         # rm.save_records(VARNAME_ACTIVITY_RECORDS)
