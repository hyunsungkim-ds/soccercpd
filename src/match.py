import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from src.utils import ints_to_range_str


class Match:
    def __init__(self, data: pd.DataFrame, roles: pd.DataFrame, id=0):
        self.id = id
        self.data = data
        self.roles = roles
        self.stats = None
        self.role_stats = None

        defenders = ["LCB", "CB", "RCB", "LB", "RB", "LWB", "RWB"]
        midfields = ["LDM", "CDM", "RDM", "LCM", "RCM", "CAM", "LM", "RM"]
        forwards = ["LCF", "CF", "RCF"]
        self.role_order = defenders + midfields + forwards

    @staticmethod
    def compute_player_stats(player_data: pd.DataFrame, td=0.1, hsr_speed=20, hsr_time=0.5) -> pd.Series:
        duration = len(player_data.dropna(subset=["x"])) * td
        distance = player_data["s"].sum() / 3.6 * td
        hsr_data = player_data[player_data["s"] >= hsr_speed].copy()

        if hsr_data.empty:
            return pd.Series([duration, distance, 0, 0])

        else:
            hsr_tds = hsr_data.reset_index()["datetime"].diff().apply(lambda x: x.total_seconds())
            hsr_ids = (hsr_tds >= 0.5).astype(int).cumsum()
            hsr_ids.index = hsr_data.index

            time_counts = hsr_ids.value_counts(sort=False)
            valid_hsr_ids = hsr_ids[hsr_ids.isin(time_counts[time_counts * td >= hsr_time].index)]
            hsr_count = len(valid_hsr_ids.unique())
            hsr_dist = hsr_data.loc[valid_hsr_ids.index, "s"].sum() / 3.6 * td

            return pd.Series([duration, distance, hsr_count, hsr_dist])

    def compute_stats(self, sort_by_role=True, save=False):
        stats = self.data.groupby(["player_id", "role_seg"], as_index=False).apply(Match.compute_player_stats)
        stats.columns = ["player_id", "role_seg", "duration", "distance", "hsr_count", "hsr_dist"]
        stats["distance_90min"] = stats["distance"] / stats["duration"] * 5400
        stats["hsr_dist_90min"] = stats["hsr_dist"] / stats["duration"] * 5400

        stats = stats[stats["duration"] > 0].copy().astype(int)
        stats = pd.merge(self.roles.drop("duration", axis=1), stats)

        if sort_by_role:
            stats["role_index"] = 0
            for i in stats.index:
                player_id = stats.at[i, "player_id"]
                starting_role = stats[stats["player_id"] == player_id].iloc[0]["aligned_role"]
                stats.at[i, "role_index"] = self.role_order.index(starting_role)
            self.stats = stats.sort_values(["role_index", "player_id", "role_seg"], ignore_index=True)
        else:
            self.stats = stats.sort_values(["player_id", "role_seg"], ignore_index=True)

        self.stats["color_index"] = 0
        role_labels = self.roles.pivot_table("aligned_role", "role_seg", "base_role", "first")

        for i in self.stats.index:
            role_seg = self.stats.at[i, "role_seg"]
            rp_role_labels = role_labels.loc[role_seg]
            role = self.stats.at[i, "aligned_role"]
            self.stats.at[i, "color_index"] = rp_role_labels[rp_role_labels == role].index[0] - 1

        if save:
            os.makedirs(f"results/{self.id}", exist_ok=True)
            self.stats.drop(["role_index", "color_index"], axis=1).to_csv(f"results/{self.id}/stats.csv", index=False)
            print(f"Successfully saved in 'results/{self.id}/stats.csv'.")

    def subplot_by_player(self, ax: Axes, role_labels: pd.DataFrame = None, metric="distance"):
        if role_labels is None:
            role_labels = self.roles.pivot_table("aligned_role", "role_seg", "base_role", "first")

        cmap = plt.get_cmap("tab10")
        max_value = self.stats.groupby("player_id")[metric].sum().max()

        player_ids = self.stats["player_id"].unique()
        player_index = 0

        for player_id in player_ids:
            player_stats = self.stats[self.stats["player_id"] == player_id]
            player_index += 1
            bottom = 0

            for role_seg in player_stats["role_seg"]:
                rp_stats = player_stats[player_stats["role_seg"] == role_seg].iloc[0]
                value = rp_stats[metric]
                if value == 0:
                    continue

                rp_role_labels = role_labels.loc[role_seg]
                role = rp_stats["aligned_role"]
                color_index = rp_role_labels[rp_role_labels == role].index[0] - 1
                ax.bar(player_index, value, bottom=bottom, color=cmap(color_index), label=role)

                if value > max_value / 20:
                    text = f"{role_seg}-{role}\n{value}" if value > max_value / 10 else f"{role_seg}-{role}"
                    ax.text(player_index, bottom + value / 2, text, ha="center", va="center", color="k", fontsize=11)

                if bottom > 0:
                    ax.hlines(bottom, xmin=player_index - 0.4, xmax=player_index + 0.4, color="k", linestyle="--")

                bottom += value

        title_dict = {"distance": "Total Distance", "hsr_dist": "HSR Distance", "hsr_count": "Number of HSRs"}
        title = f"{title_dict[metric[:-6]]} per 90 min." if metric.endswith("_90min") else title_dict[metric]
        ax.set_title(title, fontsize=18)
        ax.set_xticks(np.arange(len(player_ids)) + 1, player_ids)

    def plot_by_player(self, save=None):
        role_labels = self.roles.pivot_table("aligned_role", "role_seg", "base_role", "first")
        player_ids = self.stats["player_id"].unique()

        plt.rcParams.update({"font.size": 12})
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(len(player_ids) * 0.7, 10))

        self.subplot_by_player(ax1, role_labels, "distance")
        self.subplot_by_player(ax2, role_labels, "hsr_dist")

        if save:
            os.makedirs(f"results/{self.id}", exist_ok=True)
            plt.savefig(f"results/{self.id}/plot_dist.png", bbox_inches="tight")
            print(f"Successfully saved in 'results/{self.id}/plot_dist.png'.")

        plt.tight_layout()
        plt.show()
        plt.close()

    def compute_role_stats(self):
        grouped = self.stats.groupby(["aligned_role", "player_id"])
        role_stats = grouped[["duration", "distance", "hsr_dist"]].sum()
        role_stats["start_period"] = grouped["role_seg"].first()
        role_stats["total_periods"] = grouped["role_seg"].apply(lambda x: ints_to_range_str(x.values.tolist()))

        role2index = dict(zip(self.role_order, np.arange(len(self.role_order))))
        role2color = self.stats[["role_seg", "aligned_role", "color_index"]].drop_duplicates()

        role_stats = role_stats[role_stats["duration"] >= 300].reset_index()
        role_stats["distance_90min"] = (role_stats["distance"] / role_stats["duration"] * 5400).astype(int)
        role_stats["hsr_dist_90min"] = (role_stats["hsr_dist"] / role_stats["duration"] * 5400).astype(int)
        role_stats["role_index"] = role_stats["aligned_role"].map(role2index)

        self.role_stats = pd.merge(role_stats, role2color.rename(columns={"role_seg": "start_period"}))
        self.role_stats.sort_values(["role_index", "start_period"], ignore_index=True, inplace=True)

    def subplot_by_role(self, ax: Axes = None, metric="distance_90min"):
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in self.role_stats["color_index"]]
        ax.bar(self.role_stats.index, self.role_stats[metric], color=colors)

        for x in self.role_stats.index:
            y = self.role_stats.at[x, metric] / 2
            if y > 0:
                label = f"RP {self.role_stats.at[x, 'total_periods']}"
                ax.text(x, y, label, ha="center", va="center", rotation=90)

        counts = self.role_stats.groupby("aligned_role", sort=False)["player_id"].count()
        role_xticks = counts.cumsum() - counts / 2 - 0.5

        ytick_unit = 1000 if metric.startswith("distance") else 200
        ymax = math.ceil(self.role_stats[metric].max() / ytick_unit) * ytick_unit

        ax.set_xticks(self.role_stats.index, self.role_stats["player_id"])
        for role in role_xticks.index:
            ax.text(role_xticks[role], -ymax / 10, role, ha="center", va="center")

        ax.vlines(counts.cumsum().iloc[:-1] - 0.5, -1000, ymax, colors="k", linestyles="--")
        ax.set_xlim(-0.5, len(self.role_stats) - 0.5)
        ax.set_ylim(0, ymax)

        title_dict = {"distance": "Total Distance", "hsr_dist": "HSR Distance", "hsr_count": "Number of HSRs"}
        title = f"{title_dict[metric[:-6]]} per 90 min." if metric.endswith("_90min") else title_dict[metric]
        ax.set_title(title, fontsize=18)

    def plot_by_role(self, save=False):
        plt.rcParams.update({"font.size": 12})
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(len(self.role_stats) / 2, 10))

        self.subplot_by_role(ax1, "distance_90min")
        self.subplot_by_role(ax2, "hsr_dist_90min")

        if save:
            os.makedirs(f"results/{self.id}", exist_ok=True)
            plt.savefig(f"results/{self.id}/plot_dist_90min.png", bbox_inches="tight")
            print(f"Successfully saved in 'results/{self.id}/plot_dist_90min.png'.")

        plt.tight_layout()
        plt.show()
        plt.close()
