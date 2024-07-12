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
    def __init__(self, xy: pd.DataFrame):
        self.xy = xy
        self.role_seq = None
        self.role_distns = None

    @staticmethod
    def normalize_locs(moment_roles: pd.DataFrame) -> pd.DataFrame:
        moment_xy = moment_roles[["x", "y"]]
        moment_roles[["x_norm", "y_norm"]] = moment_xy - moment_xy.mean()
        return moment_roles

    @staticmethod
    def init_role_seq(xy: pd.DataFrame, freq="1S") -> pd.DataFrame:
        xy = xy[xy["x"].notna()]
        role_seq = []

        for i, player_id in enumerate(xy["player_id"].unique()):
            player_xy = xy[xy["player_id"] == player_id]
            resampler = player_xy.resample(freq, closed="right", label="right")
            player_role_seq = resampler[HEADER_ROLE_SEQ[:4]].last()
            player_role_seq["x"] = resampler["x"].mean()
            player_role_seq["y"] = resampler["y"].mean()
            player_role_seq["x_norm"] = np.nan
            player_role_seq["y_norm"] = np.nan
            player_role_seq["form_period"] = resampler["form_period"].last()
            player_role_seq["role_period"] = resampler["role_period"].last()
            player_role_seq["role"] = i + 1
            player_role_seq["base_role"] = i + 1
            player_role_seq["switch_rate"] = 0
            role_seq.append(player_role_seq[HEADER_ROLE_SEQ])

        role_seq = pd.concat(role_seq).reset_index().rename(columns={"index": "datetime"})
        return role_seq.groupby("datetime", group_keys=False).apply(RoleRep.normalize_locs)

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
    def update_params(role_seq: pd.DataFrame, by_player_period=False) -> pd.DataFrame:
        cols = ["player_period", "role"] if by_player_period else ["role"]
        role_distns = role_seq.groupby(cols).apply(RoleRep.estimate_mvn).reset_index()
        return role_distns.dropna().rename(columns={0: "distn"})

    @staticmethod
    def align_formations(role_seq: pd.DataFrame, role_distns: pd.DataFrame, label_group="session"):
        groups = role_seq[label_group].unique()
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
            role_seq.loc[role_seq[label_group] == group, "role"] = role_seq.loc[
                role_seq[label_group] == group, "role"
            ].apply(lambda role: role_dict[role])
            role_seq.loc[role_seq[label_group] == group, "base_role"] = role_seq.loc[
                role_seq[label_group] == group, "base_role"
            ].apply(lambda role: role_dict[role])

        return role_seq, role_distns.sort_values(by=[label_group, "role"]).reset_index(drop=True)

    def hungarian(self, moment_roles: pd.DataFrame, role_distns: pd.DataFrame) -> float:
        cost_mat = moment_roles[moment_roles.columns[(len(HEADER_ROLE_SEQ) + 1) :]].values
        row_idx, col_idx = linear_sum_assignment(cost_mat)
        base_roles = moment_roles["base_role"].iloc[row_idx].values
        temp_roles = role_distns["role"].iloc[col_idx].values
        self.role_seq.loc[moment_roles.index, "role"] = temp_roles
        self.role_seq.loc[moment_roles.index, "switch_rate"] = (base_roles != temp_roles).sum() / len(row_idx)
        return cost_mat[row_idx, col_idx].mean()

    def run(self, freq="1S", verbose=True) -> pd.DataFrame:
        temp_role_seq = self.xy.groupby("player_period").apply(RoleRep.init_role_seq, freq=freq)
        temp_role_seq = temp_role_seq.reset_index(drop=True).dropna()
        temp_role_distns = RoleRep.update_params(temp_role_seq, by_player_period=True)
        temp_role_seq = pd.merge(temp_role_seq, temp_role_distns[["player_period", "role"]])
        self.role_seq, _ = RoleRep.align_formations(temp_role_seq, temp_role_distns, "player_period")
        self.role_distns = RoleRep.update_params(self.role_seq)

        max_iter = 10
        cost_prev = float("inf")
        tol = 0.005
        self.role_seq.reset_index(drop=True, inplace=True)

        for i_iter in range(max_iter):
            cost_df = pd.DataFrame(
                self.role_distns["distn"]
                .apply(lambda n: pd.Series(-np.log(n.pdf(self.role_seq[["x_norm", "y_norm"]]))))
                .transpose()
                .values,
                index=self.role_seq.index,
            )
            cost_df = pd.concat([self.role_seq, cost_df], axis=1)
            costs = cost_df.groupby("datetime").apply(self.hungarian, self.role_distns).mean()
            cost_new = costs.mean()
            if verbose:
                print("- Cost after iteration {0}: {1:.3f}".format(i_iter + 1, cost_new))
            self.role_distns = RoleRep.update_params(self.role_seq)
            if cost_new + tol > cost_prev:
                if verbose:
                    print("Iteration finished since there are no significant changes.")
                    break
            cost_prev = cost_new

        session = self.xy["session"].iloc[0]
        self.role_distns["session"] = session

        return self.role_seq
