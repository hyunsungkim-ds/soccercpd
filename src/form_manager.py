import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering

from src.myconstants import *

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 20)
plt.rcParams["font.size"] = 15
plt.rcParams["font.family"] = "Arial"


class FormManager:
    def __init__(self, form_summary: pd.DataFrame, role_summary: pd.DataFrame = None):
        self.form_summary = form_summary
        self.role_summary = role_summary

    @staticmethod
    def align_group(form_summary):
        role_aligns = pd.DataFrame(np.vstack(form_summary["coords"].values), columns=["x", "y"])
        coloring_model = AgglomerativeClustering(n_clusters=10).fit(role_aligns.values)

        role_aligns["activity_id"] = 0
        role_aligns["form_period"] = 0

        base_roles_repeated = np.repeat(np.arange(10)[np.newaxis, :] + 1, form_summary.shape[0], axis=0)
        role_aligns["base_role"] = base_roles_repeated.flatten()
        role_aligns["aligned_role"] = coloring_model.labels_

        mean_coords = role_aligns.groupby("aligned_role")[["x", "y"]].mean().values

        for _ in range(3):
            for i, coords in enumerate(form_summary["coords"]):
                assign_cost_mat = distance_matrix(mean_coords, coords)
                _, perm = linear_sum_assignment(assign_cost_mat)
                role_aligns.loc[perm + 10 * i, "aligned_role"] = np.arange(10) + 1
                role_aligns.loc[perm + 10 * i, "activity_id"] = form_summary.at[i, "activity_id"]
                role_aligns.loc[perm + 10 * i, "form_period"] = form_summary.at[i, "form_period"]
            mean_coords = role_aligns.groupby("aligned_role")[["x", "y"]].mean()

        mean_coords["center_dist"] = np.linalg.norm(mean_coords, axis=1)

        mean_coords_df = mean_coords[(mean_coords["center_dist"] >= 12) & (mean_coords["x"] < 0)]
        mean_coords_dm = mean_coords[(mean_coords["center_dist"] < 12) & (mean_coords["x"] < 0)]
        mean_coords_am = mean_coords[(mean_coords["center_dist"] < 12) & (mean_coords["x"] >= 0)]
        mean_coords_fw = mean_coords[(mean_coords["center_dist"] >= 12) & (mean_coords["x"] >= 0)]

        roles_from = pd.concat(
            [
                mean_coords_df.sort_values("y", ascending=False),
                mean_coords_dm.sort_values("y", ascending=False),
                mean_coords_am.sort_values("y", ascending=False),
                mean_coords_fw.sort_values("y", ascending=False),
            ]
        ).index.tolist()
        role_dict = dict(zip(roles_from, np.arange(10) + 1))

        role_aligns["aligned_role"] = role_aligns["aligned_role"].replace(role_dict)
        return role_aligns[HEADER_ROLE_ALIGNS]

    def align(self, group_type="formation"):
        role_aligns_list = []
        for group in np.sort(self.form_summary[group_type].unique()):
            form_summary = self.form_summary[self.form_summary[group_type] == group].reset_index()
            role_aligns_list.append(FormManager.align_group(form_summary))
            print(f"Roles aligned for {group_type} '{group}'")

        role_aligns = pd.concat(role_aligns_list)[HEADER_ROLE_ALIGNS[:-2]]
        self.role_summary = pd.merge(
            self.role_summary[HEADER_ROLE_SUMMARY],
            self.form_summary[["activity_id", "form_period", "formation"]],
        )
        self.role_summary = pd.merge(self.role_summary, role_aligns).sort_values(
            ["activity_id", "role_period", "player_id"], ignore_index=True
        )

    @staticmethod
    def visualize_single_graph(coords, edge_mat, labels=None):
        plt.figure(figsize=(7, 5))
        plt.scatter(
            coords[:, 0],
            coords[:, 1],
            c=np.arange(10) + 1,
            s=1000,
            vmin=0.5,
            vmax=10.5,
            cmap="tab10",
            zorder=1,
        )

        if labels is None:
            labels = np.arange(11)
            fontsize = 20
        else:
            fontsize = 15

        for i in np.arange(10):
            plt.annotate(
                labels[i + 1],
                xy=coords[i],
                ha="center",
                va="center",
                c="w",
                fontsize=fontsize,
                fontweight="bold",
                zorder=2,
            )
            for j in np.arange(10):
                plt.plot(coords[[i, j], 0], coords[[i, j], 1], linewidth=edge_mat[i, j] ** 2 * 10, c="k", zorder=0)

        xlim = 30
        ylim = 24
        plt.xlim(-xlim, xlim)
        plt.ylim(-ylim, ylim)
        plt.vlines([-xlim, 0, xlim], ymin=-ylim, ymax=ylim, color="k", zorder=0)
        plt.hlines([-ylim, ylim], xmin=-xlim, xmax=xlim, color="k", zorder=0)
        plt.axis("off")

    def visualize_group(self, group, group_type="formation", paint=True, annotate=True):
        if self.role_summary is not None and "aligned_role" in self.role_summary.columns:
            role_summary = self.role_summary[self.role_summary[group_type] == group]
        else:
            form_summary = self.form_summary[self.form_summary[group_type] == group]
            role_summary = FormManager.align_group(form_summary.reset_index(drop=True))
        colors = role_summary["aligned_role"] if paint else "gray"

        plt.figure(figsize=(7, 5))
        plt.scatter(role_summary["x"], role_summary["y"], s=150, alpha=0.5, c=colors, cmap="tab10", zorder=0)

        if annotate:
            mean_coords = role_summary.groupby("aligned_role")[["x", "y"]].mean()
            plt.scatter(mean_coords["x"], mean_coords["y"], s=1000, c="w", edgecolors="k", zorder=1)
            for r in mean_coords.index:
                plt.annotate(r, xy=mean_coords.loc[r], ha="center", va="center", fontsize=25, zorder=2)

        xlim = 30
        ylim = 30
        plt.xlim(-xlim, xlim)
        plt.ylim(-ylim, ylim)
        plt.vlines([-xlim, 0, xlim], ymin=-ylim, ymax=ylim, color="k", zorder=0)
        plt.hlines([-ylim, ylim], xmin=-xlim, xmax=xlim, color="k", zorder=0)
        plt.axis("off")

    def visualize(self, group_type="formation", ignore_outliers=False, paint=True, annotate=True, save=False):
        counts = self.form_summary[group_type].value_counts()
        for group in np.sort(self.form_summary[group_type].unique()):
            if ignore_outliers and (group == -1 or group == "others"):
                continue
            self.visualize_group(group, group_type, paint, annotate)
            if save:
                plt.savefig(f"img/{group_type}_{group}.png", bbox_inches="tight")
            title = f"{group_type[0].upper() + group_type[1:]} {group} -  {counts[group]} periods"
            plt.title(title)
