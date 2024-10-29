from datetime import timedelta
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from src.utils import seconds_to_time_str


def plot_graph(
    role_seq: pd.DataFrame = None,
    role_labels: Dict[int, str] = None,
    form_graph: pd.Series = None,
    show_edges=True,
    annotate=True,
    xlim=30,
    ylim=35,
):
    if role_seq is not None:
        plt.scatter(
            -role_seq["y_norm"],
            role_seq["x_norm"],
            c=role_seq["role"],
            vmin=0.5,
            vmax=10.5,
            cmap="tab10",
            alpha=0.4,
            zorder=0,
        )

    if form_graph is not None:
        xy_idx = [c for c in form_graph.index if c[0] in ["x", "y"]]
        mean_xy: np.ndarray = form_graph[xy_idx].dropna().astype(float)
        valid_roles = np.array([int(c[1:]) for c in mean_xy.index[0::2]])
        mean_xy: np.ndarray = np.dot(mean_xy.values.reshape(-1, 2), [[0, 1], [-1, 0]])
        adj_mat: np.ndarray = form_graph["adj_mat"]

        plt.scatter(
            mean_xy[:, 0],
            mean_xy[:, 1],
            s=600 if role_labels is None else 1200,
            c="w",
            edgecolors="k",
            zorder=2,
        )

        for i, r in enumerate(valid_roles):
            if show_edges:
                for j in np.arange(mean_xy.shape[0]):
                    plt.plot(
                        mean_xy[[i, j], 0],
                        mean_xy[[i, j], 1],
                        linewidth=adj_mat[i, j] ** 2 * 5,
                        c="k",
                        zorder=1,
                    )

            if annotate:
                role_label = role_labels[r] if role_labels is not None else r
                plt.annotate(
                    role_label,
                    xy=mean_xy[i],
                    ha="center",
                    va="center",
                    fontsize=15,
                    zorder=3,
                )

    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)
    plt.vlines([-xlim, xlim], ymin=-ylim, ymax=ylim, color="k")
    plt.hlines([-ylim, 0, ylim], xmin=-xlim, xmax=xlim, color="k", zorder=1)
    plt.axis("off")


def plot_timeline(role_seq: pd.DataFrame, roster: pd.DataFrame = None, ax: Axes = None) -> Axes:
    if ax is None:
        ax = plt.subplot()

    box = ax.get_position()
    xmin = box.x0 + box.width * 0.1 if roster is None else box.x0 + box.width * 0.15
    ymin = box.y0 + box.height * 0.03
    xlen = box.width * 0.9 if roster is None else box.width * 0.85
    ylen = box.height * 0.95
    ax.set_position([xmin, ymin, xlen, ylen])

    roles_reshaped = role_seq.pivot_table("base_role", "datetime", "player_id", aggfunc="first")
    if roster is None:
        player_dict = {i: f"Player {i}" for i in roles_reshaped.columns}
    else:
        roster["player_label"] = roster.apply(lambda x: f"{x['player_name']} ({x['squad_num']})", axis=1)
        player_dict = roster.set_index("squad_num")["player_label"].to_dict()

    times = role_seq[["datetime", "session", "time"]].drop_duplicates().sort_values("datetime")
    roles_reshaped = pd.merge(times, roles_reshaped.rename(columns=player_dict).reset_index())
    roles_resampled = []

    session_start_dts = role_seq.groupby("session")["datetime"].min()  # - timedelta(seconds=1)
    for s, dt in session_start_dts.items():
        offset = f"{dt.second % 5}S"
        session_roles_reshaped = roles_reshaped[roles_reshaped["session"] == s].set_index("datetime")
        session_roles_resampled = session_roles_reshaped.resample("5S", offset=offset).first()
        session_roles_resampled.at[session_roles_resampled.index[0], "time"] = 0
        roles_resampled.append(session_roles_resampled)

    roles_resampled = pd.concat(roles_resampled)
    # players = np.sort([c for c in roles_resampled.columns if c not in ["session", "time"]])
    players = [c for c in roles_resampled.columns if c not in ["session", "time"]]
    sns.heatmap(roles_resampled[players].T, vmin=0.5, vmax=10.5, cmap="tab10", cbar=False)

    role_start_dts = role_seq.groupby("role_period")["datetime"].min()  # - timedelta(seconds=1)
    xticks = []
    for dt in role_start_dts:
        xticks.append(roles_resampled.index.get_loc(dt))
    xticks.append(len(roles_resampled) - 1)

    session_labels = roles_resampled["session"].iloc[xticks].apply(lambda x: f"H{x}-").values
    xticktimes = roles_resampled["time"].iloc[xticks[:-1]].values.tolist() + [roles_resampled["time"].iloc[-1] + 5]
    time_labels = np.array([seconds_to_time_str(x) for x in xticktimes])

    ax.vlines(xticks, ymin=0, ymax=len(players), colors="k", linestyles="--")
    ax.set_xticks(xticks)
    ax.set_xticklabels(session_labels + time_labels, rotation=45)
    ax.set_yticks(np.arange(len(players)) + 0.5)
    ax.set_yticklabels(players)
    ax.set_xlabel("session-time")
    ax.set_ylabel("player")

    return ax
