import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Match:
    def __init__(self, data: pd.DataFrame, roles: pd.DataFrame):
        self.data = data
        self.roles = roles
        self.stats = None

        defenders = ["LCB", "CB", "RCB", "LB", "RB", "LWB", "RWB"]
        midfields = ["LDM", "CDM", "RDM", "LCM", "RCM", "CAM", "LM", "RM"]
        forwards = ["LCF", "CF", "RCF"]
        self.role_order = defenders + midfields + forwards

    @staticmethod
    def _aggregate(player_data: pd.DataFrame, td=0.1, hsr_speed=20, hsr_time=0.5) -> pd.Series:
        duration = len(player_data.dropna(subset=["x"])) * td
        distance = player_data["speed"].sum() / 3.6 * td
        hsr_data = player_data[player_data["speed"] >= hsr_speed].copy()

        if hsr_data.empty:
            return pd.Series([duration, distance, 0, 0])

        else:
            hsr_tds = hsr_data.reset_index()["datetime"].diff().apply(lambda x: x.total_seconds())
            hsr_ids = (hsr_tds >= 0.5).astype(int).cumsum()
            hsr_ids.index = hsr_data.index

            time_counts = hsr_ids.value_counts(sort=False)
            valid_hsr_ids = hsr_ids[hsr_ids.isin(time_counts[time_counts * td >= hsr_time].index)]
            hsr_count = len(valid_hsr_ids.unique())
            hsr_dist = hsr_data.loc[valid_hsr_ids.index, "speed"].sum() / 3.6 * td

            return pd.Series([duration, distance, hsr_count, hsr_dist])

    def aggregate(self, sort_by_role=True):
        summary = self.data.groupby(["player_id", "role_period"], as_index=False).apply(Match._aggregate).astype(int)
        summary.columns = ["player_id", "role_period", "duration", "distance", "hsr_count", "hsr_dist"]
        summary = pd.merge(self.roles[["player_id", "role_period", "aligned_role"]], summary)

        if sort_by_role:
            summary["role_rank"] = 0
            for i in summary.index:
                player_id = summary.at[i, "player_id"]
                starting_role = summary[summary["player_id"] == player_id].iloc[0]["aligned_role"]
                summary.at[i, "role_rank"] = self.role_order.index(starting_role)
            self.stats = summary.sort_values(["role_rank", "role_period"], ignore_index=True)

        else:
            self.stats = summary.sort_values(["player_id", "role_period"], ignore_index=True)

    def plot(self, metric="distance"):
        plt.rcParams.update({"font.size": 12})
        _, ax = plt.subplots(figsize=(10, 7))
        cmap = plt.get_cmap("tab10")

        role_labels = self.roles.pivot_table("aligned_role", "role_period", "base_role", "first")
        player_ids = self.stats["player_id"].unique()
        player_index = 0

        for player_id in player_ids:
            player_stats = self.stats[self.stats["player_id"] == player_id]
            player_index += 1
            bottom = 0

            for role_period in player_stats["role_period"]:
                rp_stats = player_stats[player_stats["role_period"] == role_period].iloc[0]
                role = rp_stats["aligned_role"]
                value = rp_stats[metric]

                rp_role_labels = role_labels.loc[role_period]
                color_index = rp_role_labels[rp_role_labels == role].index[0] - 1
                text = f"{role_period}-{role}\n{value}"

                ax.bar(player_index, value, bottom=bottom, color=cmap(color_index), label=role)
                ax.text(player_index, bottom + value / 2, text, ha="center", va="center", color="k")
                if bottom > 0:
                    ax.hlines(bottom, xmin=player_index - 0.4, xmax=player_index + 0.4, color="k", linestyle="--")

                bottom += value

        ax.set_xticks(np.arange(len(player_ids)) + 1, player_ids)
        plt.show()
