import os
from contextlib import contextmanager
from datetime import timedelta
from typing import Dict, Union

import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from src.utils import seconds_to_time_str

# Role-segment label prefix per possession phase (attack -> "AR1, AR2, ...", defend -> "DR1, ...").
_PHASE_PREFIX = {"attack": "A", "defend": "D"}

_RESULT_FONT_PATH = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Light.ttc"


def _save_or_show(fig, save_path: str = None):
    """Save the figure to ``save_path`` (creating parent dirs) or show it, then close it.
    Shared save/show tail for the public composite plot builders."""
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Successfully saved in '{save_path}'.")
    else:
        plt.show()
        plt.close(fig)


@contextmanager
def _set_result_font(font_path: str = _RESULT_FONT_PATH):
    """Temporarily apply a CJK-capable font + seaborn theme (e.g. for Korean roster names in a
    result figure), restoring global matplotlib/seaborn state afterwards even if drawing raises."""
    prev_family = plt.rcParams.get("font.family")
    try:
        fontprop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = fontprop.get_name()
        sns.set_theme(font=fontprop.get_name(), font_scale=1.5)
        yield
    finally:
        sns.reset_orig()
        plt.rcParams["font.family"] = prev_family


def _draw_form_graph(
    ax: Axes,
    role_seq: pd.DataFrame = None,
    role_labels: Dict[int, str] = None,
    form_graph: pd.Series = None,
    show_edges=True,
    annotate=True,
    xlim=30,
    ylim=35,
):
    """Draw a single formation as a role-adjacency graph: white role nodes (optionally labelled) over
    a faint scatter of the raw role positions, connected by adjacency-weighted edges. Element drawer:
    draws only onto ``ax``."""
    if role_seq is not None:
        ax.scatter(
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

        ax.scatter(
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
                    ax.plot(
                        mean_xy[[i, j], 0],
                        mean_xy[[i, j], 1],
                        linewidth=adj_mat[i, j] ** 2 * 5,
                        c="k",
                        zorder=1,
                    )

            if annotate:
                role_label = role_labels[r] if role_labels is not None else r
                ax.annotate(
                    role_label,
                    xy=mean_xy[i],
                    ha="center",
                    va="center",
                    fontsize=15,
                    zorder=3,
                )

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.vlines([-xlim, xlim], ymin=-ylim, ymax=ylim, color="k")
    ax.hlines([-ylim, 0, ylim], xmin=-xlim, xmax=xlim, color="k", zorder=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_role_heatmap(ax: Axes, role_seq: pd.DataFrame, roster: pd.DataFrame = None) -> Axes:
    """Draw the per-player role heatmap (players x resampled time, coloured by role) onto ``ax``,
    with dashed role-change-point verticals and period-time tick labels. Element drawer."""
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

    times = role_seq[["datetime", "period_id", "timestamp"]].drop_duplicates().sort_values("datetime")
    roles_reshaped = pd.merge(times, roles_reshaped.rename(columns=player_dict).reset_index())
    roles_resampled = []

    period_start_dts = role_seq.groupby("period_id")["datetime"].min()  # - timedelta(seconds=1)
    for s, dt in period_start_dts.items():
        offset = f"{dt.second % 5}S"
        period_roles_reshaped = roles_reshaped[roles_reshaped["period_id"] == s].set_index("datetime")
        period_roles_resampled = period_roles_reshaped.resample("5S", offset=offset).first()
        period_roles_resampled.at[period_roles_resampled.index[0], "timestamp"] = 0
        roles_resampled.append(period_roles_resampled)

    roles_resampled = pd.concat(roles_resampled)
    # players = np.sort([c for c in roles_resampled.columns if c not in ["period_id", "timestamp"]])
    players = [c for c in roles_resampled.columns if c not in ["period_id", "timestamp"]]
    sns.heatmap(roles_resampled[players].T, vmin=0.5, vmax=10.5, cmap="tab10", cbar=False, ax=ax)

    role_start_dts = role_seq.groupby("role_seg")["datetime"].min()  # - timedelta(seconds=1)
    xticks = []
    for dt in role_start_dts:
        xticks.append(roles_resampled.index.get_loc(dt))
    xticks.append(len(roles_resampled) - 1)

    period_labels = roles_resampled["period_id"].iloc[xticks].apply(lambda x: f"H{x}-").values
    xticktimes = roles_resampled["timestamp"].iloc[xticks[:-1]].values.tolist() + [
        roles_resampled["timestamp"].iloc[-1] + 5
    ]
    time_labels = np.array([seconds_to_time_str(x) for x in xticktimes])

    ax.vlines(xticks, ymin=0, ymax=len(players), colors="k", linestyles="--")
    ax.set_xticks(xticks)
    ax.set_xticklabels(period_labels + time_labels, rotation=45)
    ax.set_yticks(np.arange(len(players)) + 0.5)
    ax.set_yticklabels(players)
    ax.set_xlabel("period-time")
    ax.set_ylabel("player")

    return ax


def _squad_num(player_id) -> str:
    # player_id is either a "team_num" string (e.g. "home_7") or a bare squad number.
    return str(player_id).split("_")[-1]


def _draw_formation(
    ax: Axes,
    form_row: pd.Series,
    assignment: Dict,
    player_colors: Dict,
    title: str = None,
    title_loc: str = "top",
    xlim: int = 30,
    ylim: int = 35,
    show_edges: bool = True,
):
    """Draw a single formation snapshot: role mean positions as circles coloured by the
    assigned player, with the player's squad number written inside each circle."""
    xy_idx = [c for c in form_row.index if c[0] in ["x", "y"] and c[1:].isdigit()]
    mean_xy: pd.Series = form_row[xy_idx].dropna().astype(float)
    valid_roles = np.array([int(c[1:]) for c in mean_xy.index[0::2]])

    # Same transform as plot_form_graph: (x, y) -> (-y, x) for a vertical, upward-attacking pitch.
    plot_xy = np.dot(mean_xy.values.reshape(-1, 2), [[0, 1], [-1, 0]])
    role_to_player = {role: player_id for player_id, role in assignment.items()}

    if show_edges and isinstance(form_row.get("adj_mat"), np.ndarray):
        adj_mat: np.ndarray = form_row["adj_mat"]
        for i in range(len(valid_roles)):
            for j in range(len(valid_roles)):
                ax.plot(
                    plot_xy[[i, j], 0],
                    plot_xy[[i, j], 1],
                    linewidth=adj_mat[i, j] ** 2 * 5,
                    c="k",
                    zorder=1,
                )

    for i, r in enumerate(valid_roles):
        player_id = role_to_player.get(r)
        color = player_colors.get(player_id, "w")
        ax.scatter(plot_xy[i, 0], plot_xy[i, 1], s=550, c=[color], edgecolors="k", zorder=2)
        if player_id is not None:
            ax.annotate(
                _squad_num(player_id),
                xy=plot_xy[i],
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                zorder=3,
            )

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.vlines([-xlim, xlim], ymin=-ylim, ymax=ylim, color="k")
    ax.hlines([-ylim, 0, ylim], xmin=-xlim, xmax=xlim, color="k", zorder=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        # Place the title on the tier's outer edge so it never lands in the inner gap that the
        # timeline's change-point tick labels occupy.
        if title_loc == "top":
            ax.set_title(title, fontsize=15, fontweight="bold")
        else:
            ax.text(0.5, -0.03, title, transform=ax.transAxes, ha="center", va="top", fontsize=15, fontweight="bold")


def _segment_match_times(res) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Real match-clock start/end of every role segment, as ``(periods, start_mt, end_mt)`` (one
    entry per segment). ``*_mt`` is per-period seconds (H2 restarts at 0).

    SoccerCPD finalises ``role_segs`` with real per-period ``start_ts``/``end_ts`` columns, so use
    those directly. Fall back to deriving them from the stream's ``match_ts``/``timestamp`` via the
    (older) ``start_dt``/``end_dt`` datetimes when the ``*_ts`` columns are absent.
    """
    role_segs = res.role_segs.sort_values("role_seg")
    periods = role_segs["period_id"].astype(int).to_numpy()

    if "start_ts" in role_segs.columns and "end_ts" in role_segs.columns:
        return periods, role_segs["start_ts"].to_numpy(float), role_segs["end_ts"].to_numpy(float)

    data = res.data
    ts_col = "match_ts" if "match_ts" in data.columns else "timestamp"
    frames = (
        data.reset_index()[["datetime", "period_id", ts_col]]
        .dropna(subset=[ts_col])
        .drop_duplicates("datetime")
        .sort_values("datetime")
    )
    # Look a change-point's clock up within its own period, so a boundary datetime shared with the
    # adjacent (other-half) frame never returns that frame's reset clock.
    per_period = {}
    for pid, grp in frames.groupby("period_id"):
        per_period[int(pid)] = (grp["datetime"].values.astype("datetime64[ns]"), grp[ts_col].values.astype(float))

    def _match_ts(dt, period) -> float:
        dts, tss = per_period[int(period)]
        i = int(np.searchsorted(dts, np.datetime64(dt, "ns")))
        i = min(max(i, 0), len(dts) - 1)
        if i > 0 and abs(dts[i - 1] - np.datetime64(dt, "ns")) < abs(dts[i] - np.datetime64(dt, "ns")):
            i -= 1
        return float(tss[i])

    start_mt = np.array([_match_ts(dt, p) for dt, p in zip(role_segs["start_dt"], periods)])
    end_mt = np.array([_match_ts(dt, p) for dt, p in zip(role_segs["end_dt"], periods)])
    return periods, start_mt, end_mt


def _tier_axis(periods: np.ndarray, start_mt: np.ndarray, end_mt: np.ndarray, half_len: float, gap: float):
    """Place a tier's segments on the shared match-time axis. The first half keeps its real seconds;
    the second half is offset to start at ``half_len + gap`` (the ``gap`` is a blank that separates
    the two halves so the H1/H2 clocks read unambiguously without an H1-/H2- label prefix).

    Returns ``(cum_start, cum_end, boundaries)`` where ``cum_*`` are per-segment block edges and
    ``boundaries`` is a list of distinct ``(cum_x, period, match_ts)`` change-points (both a
    segment's start and end, deduplicated) used for the connectors' fan and the match-clock labels.
    """
    m0 = start_mt[periods == 2].min() if (periods == 2).any() else 0.0
    off = half_len + gap
    cum_start = np.where(periods == 1, start_mt, off + (start_mt - m0))
    cum_end = np.where(periods == 1, end_mt, off + (end_mt - m0))

    boundaries, seen = [], set()
    for i in range(len(periods)):
        for cx, mt in ((cum_start[i], start_mt[i]), (cum_end[i], end_mt[i])):
            key = round(float(cx), 2)
            if key not in seen:
                seen.add(key)
                boundaries.append((float(cx), int(periods[i]), float(mt)))
    boundaries.sort()
    return cum_start, cum_end, boundaries


def _draw_seg_timeline(
    ax: Axes,
    role_segs: pd.DataFrame,
    cum_start: np.ndarray,
    cum_end: np.ndarray,
    x_total: float,
    form_colors: Dict,
    label_prefix: str = "",
):
    """Draw one timeline row: role segments as match-time-proportional coloured blocks on the shared
    ``[0, x_total]`` axis. Block ``k`` spans ``[cum_start[k], cum_end[k]]``."""
    ax.set_xlim(0, x_total)
    ax.set_ylim(0, 1)
    for k, (_, seg) in enumerate(role_segs.iterrows()):
        start, end = cum_start[k], cum_end[k]
        ax.add_patch(
            plt.Rectangle(
                (start, 0),
                end - start,
                1,
                facecolor=form_colors.get(int(seg["form_seg"]), "0.85"),
                edgecolor="k",
                linewidth=1,
                zorder=1,
            )
        )
        ax.annotate(
            f"{label_prefix}R{int(seg['role_seg'])}",
            xy=((start + end) / 2, 0.5),
            ha="center",
            va="center",
            fontsize=13,
            zorder=2,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_form_timeline(
    cpds: Union[object, Dict[str, object]],
    title: str = None,
    save_path: str = None,
):
    """Formation-change timeline for one or two SoccerCPD results.

    ``cpds`` is either a single SoccerCPD result or a ``{phase: result}`` dict.
    - Full CPD (single result / single-entry dict): one tier of formation snapshots above a single
      timeline row.
    - Phase-wise CPD (``{"attack": ..., "defend": ...}``): two tiers -- attacking tactical changes on
      top, defending changes on the bottom -- with the two timeline rows meeting in the centre.
    """
    # Normalise the input into an ordered [(phase, result)] list; "attack" first so it sits on top.
    if not isinstance(cpds, dict):
        tiers = [("all", cpds)]
    else:
        order = ["attack", "defend"]
        keys = sorted(cpds.keys(), key=lambda k: (order.index(k) if k in order else len(order), k))
        tiers = [(k, cpds[k]) for k in keys]
    phase_wise = len(tiers) > 1

    # Per-player colours: tab20 in squad-number order, consistent across every tier.
    all_players = sorted(
        {p for _, res in tiers for a in res.role_segs["assignment"] for p in a},
        key=lambda p: int(_squad_num(p)),
    )
    cmap = plt.get_cmap("tab20")
    player_colors = {p: cmap(i % 20) for i, p in enumerate(all_players)}

    # Light per-formation shades for the timeline blocks (shared across tiers).
    all_form_segs = sorted({int(f) for _, res in tiers for f in res.form_segs["form_seg"]})
    shade_cmap = plt.get_cmap("Pastel1")
    form_colors = {f: shade_cmap(i % 9) for i, f in enumerate(all_form_segs)}

    # Per-segment match-clock spans, then a single shared cumulative-time axis for every tier.
    # A blank `gap` separates the first- and second-half blocks so the two clocks read unambiguously
    # without an H1-/H2- label prefix.
    seg_times = {name: _segment_match_times(res) for name, res in tiers}
    half_len = max((emt[pr == 1].max() for pr, _, emt in seg_times.values() if (pr == 1).any()), default=0.0)
    h2_span = max(
        ((emt[pr == 2].max() - smt[pr == 2].min()) for pr, smt, emt in seg_times.values() if (pr == 2).any()),
        default=0.0,
    )
    gap = 0.03 * (half_len + h2_span) if h2_span else 0.0
    x_total = half_len + gap + h2_span if h2_span else half_len
    axes = {name: _tier_axis(pr, smt, emt, half_len, gap) for name, (pr, smt, emt) in seg_times.items()}

    # --- Layout geometry (inches) ---------------------------------------------------------------
    xlim, ylim = 30, 35
    cell_w = 2.3
    cell_h = cell_w * (2 * ylim) / (2 * xlim)  # keep the pitch undistorted
    th = 0.4  # timeline-row height (only holds "R1", "R2", ... so keep it thin)
    tick_gap = 0.72  # band holding change-point tick labels + connectors
    # Phase-wise puts the defend snapshot titles below their pitches, so leave room at the bottom;
    # full CPD puts the change-point tick labels below the single timeline instead.
    pad_l = 0.7 if phase_wise else 0.3
    pad_r, pad_t = 0.3, 0.6
    pad_b = 0.4 if phase_wise else tick_gap

    ncols = max(len(res.role_segs) for _, res in tiers)
    plot_w = ncols * cell_w
    fig_w = pad_l + plot_w + pad_r

    if phase_wise:
        fig_h = pad_b + cell_h + tick_gap + 2 * th + tick_gap + cell_h + pad_t
    else:
        fig_h = pad_b + cell_h + tick_gap + th + pad_t

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)

    def _pitch_rect(col, n_segs, bottom_in):
        center = pad_l + plot_w * (col + 0.5) / n_segs
        left = center - cell_w / 2
        return [left / fig_w, bottom_in / fig_h, cell_w / fig_w, cell_h / fig_h]

    def _tl_rect(bottom_in):
        return [pad_l / fig_w, bottom_in / fig_h, plot_w / fig_w, th / fig_h]

    geom = dict(
        pad_l=pad_l,
        plot_w=plot_w,
        fig_w=fig_w,
        fig_h=fig_h,
        x_total=x_total,
        xlim=xlim,
        ylim=ylim,
        th=th,
        cell_w=cell_w,
        cell_h=cell_h,
    )

    if phase_wise:
        att_res, def_res = tiers[0][1], tiers[1][1]
        att_pitch_bottom = pad_b + cell_h + tick_gap + 2 * th + tick_gap
        att_tl_bottom = pad_b + cell_h + tick_gap + th
        def_tl_bottom = pad_b + cell_h + tick_gap
        def_pitch_bottom = pad_b

        _compose_tier(
            fig,
            att_res,
            axes[tiers[0][0]],
            att_pitch_bottom,
            att_tl_bottom,
            player_colors,
            form_colors,
            _pitch_rect,
            _tl_rect,
            geom,
            pitch_above=True,
            label_prefix=_PHASE_PREFIX.get(tiers[0][0], ""),
        )
        _compose_tier(
            fig,
            def_res,
            axes[tiers[1][0]],
            def_pitch_bottom,
            def_tl_bottom,
            player_colors,
            form_colors,
            _pitch_rect,
            _tl_rect,
            geom,
            pitch_above=False,
            label_prefix=_PHASE_PREFIX.get(tiers[1][0], ""),
        )

        # Centre each side label on its tier but biased toward the timeline (midpoint of the tier's
        # timeline edge and its far pitch edge), so the two labels stay well apart yet sit close to
        # their timeline rather than floating up at the pitch centre.
        att_label_y = (att_tl_bottom + att_pitch_bottom + cell_h) / 2
        def_label_y = (def_tl_bottom + th + def_pitch_bottom) / 2
        label_kwargs = dict(rotation=90, ha="center", va="center", fontsize=15, fontweight="bold")
        fig.text(0.012, att_label_y / fig_h, "Attacking", **label_kwargs)
        fig.text(0.012, def_label_y / fig_h, "Defending", **label_kwargs)
    else:
        name, res = tiers[0]
        pitch_bottom = pad_b + th + tick_gap
        tl_bottom = pad_b
        _compose_tier(
            fig,
            res,
            axes[name],
            pitch_bottom,
            tl_bottom,
            player_colors,
            form_colors,
            _pitch_rect,
            _tl_rect,
            geom,
            pitch_above=True,
            label_prefix=_PHASE_PREFIX.get(name, ""),
        )

    if title is not None:
        fig.suptitle(title, fontsize=18, y=1.0)

    _save_or_show(fig, save_path)


def _compose_tier(
    fig,
    res,
    tier_axis: "tuple",
    pitch_bottom: float,
    tl_bottom: float,
    player_colors: Dict,
    form_colors: Dict,
    pitch_rect_fn,
    tl_rect_fn,
    geom: Dict,
    pitch_above: bool,
    label_prefix: str = "",
):
    """Lay out one phase onto ``fig``: a snapshot per role segment, a match-time-proportional
    timeline row, dashed change-point connectors to the snapshot-column boundaries, and match-clock
    tick labels. Composer (manages several axes on ``fig``), not a single-ax element drawer."""
    from matplotlib.lines import Line2D

    role_segs = res.role_segs.sort_values("role_seg").reset_index(drop=True)
    forms = res.form_segs.set_index("form_seg")
    n_segs = len(role_segs)
    cum_start, cum_end, boundaries = tier_axis
    pad_l, plot_w, fig_w, fig_h = geom["pad_l"], geom["plot_w"], geom["fig_w"], geom["fig_h"]
    x_total, xlim, ylim, th = geom["x_total"], geom["xlim"], geom["ylim"], geom["th"]
    cell_w = geom["cell_w"]

    tl_ax = fig.add_axes(tl_rect_fn(tl_bottom))
    _draw_seg_timeline(tl_ax, role_segs, cum_start, cum_end, x_total, form_colors, label_prefix)

    for col in range(n_segs):
        seg = role_segs.iloc[col]
        form_row = forms.loc[seg["form_seg"]]
        if isinstance(form_row, pd.DataFrame):  # guard against a duplicate form_seg index
            form_row = form_row.iloc[0]

        formation = form_row["formation"] if "formation" in form_row.index else None
        if isinstance(formation, str):
            form_label = formation if formation == "others" else "-".join(formation)
            snap_title = f"{label_prefix}R{int(seg['role_seg'])}: {form_label}"
        else:
            snap_title = f"{label_prefix}R{int(seg['role_seg'])}"

        pitch_ax = fig.add_axes(pitch_rect_fn(col, n_segs, pitch_bottom))
        _draw_formation(
            pitch_ax,
            form_row,
            seg["assignment"],
            player_colors,
            title=snap_title,
            title_loc="top" if pitch_above else "bottom",
            xlim=xlim,
            ylim=ylim,
        )

    # Timeline row extent in figure fractions.
    tl_top_f = (tl_bottom + th) / fig_h
    tl_bot_f = tl_bottom / fig_h
    pitch_edge_f = pitch_bottom / fig_h if pitch_above else (pitch_bottom + geom["cell_h"]) / fig_h

    def _x_frac_time(x):  # cumulative match seconds -> figure-fraction x
        return (pad_l + (x / x_total if x_total else 0) * plot_w) / fig_w

    def _snap_edges(col):  # left/right edge of snapshot `col` in figure-fraction x (matches _pitch_rect)
        center = pad_l + plot_w * (col + 0.5) / n_segs
        return (center - cell_w / 2) / fig_w, (center + cell_w / 2) / fig_w

    # Connect each snapshot's own edges to its start/end change-points: left edge -> cum_start[col],
    # right edge -> cum_end[col]. Using the snapshot's real edges (not the evenly-spaced column
    # midlines) keeps the fan aligned when a tier's snapshots are spread out, and using per-segment
    # start/end means the H1-end and H2-start on either side of the mid-timeline gap both get a line.
    tl_edge_f = tl_top_f if pitch_above else tl_bot_f
    for col in range(n_segs):
        left_f, right_f = _snap_edges(col)
        for x_time, x_pitch in ((cum_start[col], left_f), (cum_end[col], right_f)):
            line = Line2D(
                [_x_frac_time(x_time), x_pitch],
                [tl_edge_f, pitch_edge_f],
                color="0.4",
                linewidth=1,
                linestyle="--",
                zorder=0,
            )
            line.set_transform(fig.transFigure)
            line.set_clip_on(False)
            fig.add_artist(line)

    # Match-clock tick label centred on each change-point, just outside the timeline (above for
    # attack, below for defend). Time only (no H1-/H2- prefix); the mid gap distinguishes the halves.
    for cum_x, _period, mt in boundaries:
        if pitch_above:
            y, va = tl_top_f + 0.006, "bottom"
        else:
            y, va = tl_bot_f - 0.006, "top"
        fig.text(
            _x_frac_time(cum_x),
            y,
            seconds_to_time_str(mt),
            rotation=45,
            ha="center",
            va=va,
            fontsize=12,
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.7),
        )


def plot_role_timeline(
    result,
    roster: pd.DataFrame = None,
    role_labels: pd.DataFrame = None,
    title: str = None,
    save_path: str = None,
):
    """Role-change timeline for a single SoccerCPD result: up to four formation graphs (role key) on
    top and the per-player role heatmap over match time below.

    ``result`` is a SoccerCPD result exposing ``form_segs`` (with role positions, ``adj_mat`` and a
    ``formation`` label), ``role_seq`` and ``role_segs``. ``roster`` optionally maps squad numbers to
    display names; ``role_labels`` is the (per ``form_seg``) role-label DataFrame (``cpd.role_labels``).
    Saves to ``save_path`` (full path) or shows. Companion of :func:`plot_form_timeline`.
    """
    with _set_result_font():
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        gs = gridspec.GridSpec(2, 4, left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)

        forms = result.form_segs.set_index("form_seg")
        for i, form_seg in enumerate(result.form_segs["form_seg"][:4]):
            fp_role_seq = result.role_seq[(result.role_seq["form_seg"] == form_seg) & (result.role_seq["role"].notna())]
            fp_role_labels = role_labels.loc[form_seg].dropna().to_dict() if role_labels is not None else None
            fp_graph = forms.loc[form_seg]

            ax = fig.add_subplot(gs[0, i])
            _draw_form_graph(ax, fp_role_seq, fp_role_labels, fp_graph)

            fp_role_segs = result.role_segs[result.role_segs["form_seg"] == form_seg]
            start_rp = fp_role_segs["role_seg"].min()
            form_label = "others" if fp_graph["formation"] == "others" else "-".join(fp_graph["formation"])
            if len(fp_role_segs) == 1:
                ax.set_title(f"Role Period {start_rp}: {form_label}", fontsize=18)
            else:
                end_rp = fp_role_segs["role_seg"].max()
                ax.set_title(f"Role Periods {start_rp}-{end_rp}: {form_label}", fontsize=18)

        ax = fig.add_subplot(gs[1, :])
        _draw_role_heatmap(ax, result.role_seq, roster)
        ax.set_title("Role Change Timeline", fontsize=18)

        if title is not None:
            fig.suptitle(title, fontsize=22, y=1.0)

    _save_or_show(fig, save_path)
