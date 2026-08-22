"""Sankey diagrams drawn with plain matplotlib (issue #37).

Used for the location figure: which country's authors an LLM recommends when the
prompt embeds a given country. No new dependency — the ribbons are Bezier paths,
so plotly/holoviews are not needed.
"""

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle

try:
    from libs.visuals import constants
except ImportError:  # PYTHONPATH=code/libs/
    from visuals import constants


def _ribbon(
    ax,
    x0: float,
    x1: float,
    y0_top: float,
    y0_bot: float,
    y1_top: float,
    y1_bot: float,
    color: str,
    alpha: float,
    curvature: float = 0.5,
) -> None:
    """Draw one flow band between two nodes as a closed Bezier patch.

    The band leaves the left node horizontally and arrives horizontally at the
    right node — control points sit at a fixed fraction of the horizontal gap,
    which is what gives the familiar S-curve.
    """
    dx = (x1 - x0) * curvature
    verts = [
        (x0, y0_top),
        (x0 + dx, y0_top),
        (x1 - dx, y1_top),
        (x1, y1_top),
        (x1, y1_bot),
        (x1 - dx, y1_bot),
        (x0 + dx, y0_bot),
        (x0, y0_bot),
        (x0, y0_top),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    ax.add_patch(
        PathPatch(
            Path(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, lw=0
        )
    )


def collapse_tail(
    flows: pd.DataFrame,
    *,
    target_col: str = "target",
    value_col: str = "value",
    top_n: int = 12,
    other_label: str = "Other",
) -> pd.DataFrame:
    """Keep the `top_n` targets by total flow and fold the rest into one node.

    With 200+ author countries the figure is unreadable otherwise. Returns a copy
    — the caller's frame is untouched — and never silently drops a flow: whatever
    falls outside the top N is still there under `other_label`.
    """
    keep = flows.groupby(target_col)[value_col].sum().nlargest(top_n).index
    out = flows.copy()
    out[target_col] = out[target_col].where(out[target_col].isin(keep), other_label)
    return out


def normalize_by_source(
    flows: pd.DataFrame,
    *,
    source_col: str = "source",
    value_col: str = "value",
) -> pd.DataFrame:
    """Rescale each source's flows to sum to 1.

    Answers the question issue #37 actually asks — "% of times country R appears
    in the recommendations for country C" — by giving every source the same node
    height, so ribbon widths read as shares *within* a source rather than as raw
    counts. Without it a source with more factual authors looks more biased just
    for being larger.
    """
    out = flows.copy()
    totals = out.groupby(source_col)[value_col].transform("sum")
    out[value_col] = out[value_col] / totals
    return out


def plot_sankey(
    flows: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    value_col: str = "value",
    source_order: Sequence[str] | None = None,
    target_order: Sequence[str] | None = None,
    source_colors: Mapping[str, str] | None = None,
    source_labels: Mapping[str, str] | None = None,
    target_labels: Mapping[str, str] | None = None,
    figsize: tuple = (9, 8),
    node_width: float = 0.06,
    node_gap: float = 0.02,
    ribbon_alpha: float = 0.62,
    label_fontsize: int = 10,
    value_fmt: str | None = "{pct:.0f}%",
    ax=None,
):
    """Two-column Sankey: `source_col` on the left, `target_col` on the right.

    Node heights are proportional to their total flow, ribbons are coloured by
    source, and both columns are packed top-down in the given order (or by
    descending total when no order is passed).

    `value_fmt` is appended to each target label; it receives `value` (the raw
    total) and `pct` (share of the grand total). Pass None to omit it.

    Returns `(fig, ax)`.
    """
    agg = (
        flows.groupby([source_col, target_col], dropna=False)[value_col]
        .sum()
        .reset_index()
    )
    agg = agg[agg[value_col] > 0]
    if agg.empty:
        raise ValueError("No positive flows to draw")

    src_tot = agg.groupby(source_col)[value_col].sum()
    dst_tot = agg.groupby(target_col)[value_col].sum()
    sources = (
        list(source_order)
        if source_order
        else src_tot.sort_values(ascending=False).index.tolist()
    )
    targets = (
        list(target_order)
        if target_order
        else dst_tot.sort_values(ascending=False).index.tolist()
    )
    sources = [s for s in sources if s in src_tot.index]
    targets = [t for t in targets if t in dst_tot.index]

    total = float(agg[value_col].sum())

    def _layout(names, totals):
        """Stack nodes top-down, splitting `node_gap` between them."""
        span = 1.0 - node_gap * max(len(names) - 1, 0)
        pos, y = {}, 1.0
        for name in names:
            h = span * float(totals[name]) / total
            pos[name] = (y, y - h)  # (top, bottom)
            y -= h + node_gap
        return pos

    src_pos = _layout(sources, src_tot)
    dst_pos = _layout(targets, dst_tot)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    x_left, x_right = 0.0, 1.0
    colors = source_colors or {}
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#4A90D9"])

    # Ribbons first so the node bars sit on top of them.
    src_cursor = {s: src_pos[s][0] for s in sources}
    dst_cursor = {t: dst_pos[t][0] for t in targets}
    for si, s in enumerate(sources):
        color = colors.get(s, palette[si % len(palette)])
        # Within a source, follow the target column order so ribbons cross no
        # more than the layout forces them to.
        rows = (
            agg[agg[source_col] == s]
            .set_index(target_col)
            .reindex(targets)
            .dropna(subset=[value_col])
        )
        for t, row in rows.iterrows():
            value = float(row[value_col])
            h_src = (src_pos[s][0] - src_pos[s][1]) * value / float(src_tot[s])
            h_dst = (dst_pos[t][0] - dst_pos[t][1]) * value / float(dst_tot[t])
            y0_top, y1_top = src_cursor[s], dst_cursor[t]
            _ribbon(
                ax,
                x_left + node_width,
                x_right,
                y0_top,
                y0_top - h_src,
                y1_top,
                y1_top - h_dst,
                color,
                ribbon_alpha,
            )
            src_cursor[s] -= h_src
            dst_cursor[t] -= h_dst

    for si, s in enumerate(sources):
        top, bot = src_pos[s]
        color = colors.get(s, palette[si % len(palette)])
        ax.add_patch(
            Rectangle(
                (x_left, bot), node_width, top - bot, facecolor=color, edgecolor="none"
            )
        )
        ax.text(
            x_left + node_width + 0.015,
            (top + bot) / 2,
            (source_labels or {}).get(s, s),
            va="center",
            ha="left",
            fontsize=label_fontsize,
        )

    for t in targets:
        top, bot = dst_pos[t]
        ax.add_patch(
            Rectangle(
                (x_right, bot),
                node_width,
                top - bot,
                facecolor=constants.TICK_COLOR,
                edgecolor="none",
            )
        )
        label = (target_labels or {}).get(t, t)
        if value_fmt:
            label += "  " + value_fmt.format(
                value=float(dst_tot[t]), pct=100 * float(dst_tot[t]) / total
            )
        ax.text(
            x_right + node_width + 0.015,
            (top + bot) / 2,
            label,
            va="center",
            ha="left",
            fontsize=label_fontsize,
        )

    ax.set_xlim(-0.02, x_right + node_width + 0.30)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")
    return fig, ax
