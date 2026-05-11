
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from typing import Any, Callable


Number = Union[int, float]


# -----------------------------
# Specs
# -----------------------------
@dataclass(frozen=True)
class PanelSpec:
    """One metric column."""
    metric_name: str
    label: str

    # Per-metric axis config (can be overridden by globals)
    xlim: Optional[Tuple[Number, Number]] = None
    xticks: Optional[Sequence[Number]] = None

    # For line plots: treat ylim/yticks explicitly
    ylim: Optional[Tuple[Number, Number]] = None
    yticks: Optional[Sequence[Number]] = None

    # CI rendering per metric
    draw_ci: bool = True
    ci_style: str = "horizontal"  # {"horizontal", "vertical", "none"}

    # Value label format (bar plot)
    value_fmt: str = "{:.2f}"

    

@dataclass(frozen=True)
class LayoutSpec:
    """Grid layout controls."""
    figsize: Tuple[float, float] = (18.0, 4.6)
    wspace: float = 0.28
    hspace: float = 0.30

    # Width ratios: labels area vs each metric panel
    label_ratio: float = 1.75
    panel_ratio: float = 1.0

    # Figure margins
    left: float = 0.04
    right: float = 0.99
    top: float = 0.88
    bottom: float = 0.14

    # Group separator
    draw_group_separators: bool = True
    separator_lw: float = 1.2

    # Width ratios for before and after
    label_ratio_1: float = 0.12
    label_ratio_2: float = 0.12
    panel_ratio: float = 0.30
    width_ratios: Optional[Sequence[float]] = None

@dataclass(frozen=True)
class StyleSpec:
    title_fontsize: int = 11
    label_fontsize: int = 11
    ylabel_fontsize: int = 11
    tick_fontsize: int = 10
    value_fontsize: int = 9
    ylabel_pad: float = 28

    grid_x: bool = True
    grid_y: bool = True
    grid_alpha: float = 0.35
    keep_bottom_spine: bool = True

    # Bars
    bar_height: float = 0.72

    # Lines
    line_width: float = 2.0
    point_size: float = 28.0

    # CI errorbar style (bars)
    ci_capsize: float = 2.
    ci_lw: float = 0.5
    ci_color: str = "black"

    # Slope plot
    slope_linewidth: float = 2.0
    slope_point_size: float = 40.0

    # legend
    legend_fontsize: int = 10
    legend_frameon: bool = True

    # axis cosmetics
    hide_spines_left: bool = True
    hide_spines_bottom: bool = True
    hide_spines_top: bool = True
    hide_spines_right: bool = True

    # x reference lines at x=0 and x=1
    draw_x_vlines: bool = True
    x_vlines_lw: float = 1.8
    x_vlines_alpha: float = 0.95
    x_vlines_color: str = "grey"

    # horizontal reference lines at y ticks
    draw_y_hlines: bool = True
    hline_lw: float = 1.0
    hline_alpha: float = 0.25
    hline_style: str = "--"

    # labels
    show_xticklabels_only_bottom: bool = True
    hide_y_ticks_nonfirstcol: bool = True

    # row-label placement
    row_label_x: float = -0.22
    row_label_pad: float = 28

    # numeric annotation near endpoints
    annotate_points: bool = True
    annotation_fontsize: int = 10
    annotation_dx: float = 0.03  # in x-axis units (because x is [0,1])
    annotation_dy: float = 0.0   # in y-axis units
    annotation_color: str = "black"

    # internal y labels
    show_internal_y_labels: bool = True
    internal_y_label_x: float = 0.5     # axes fraction (middle)
    internal_y_label_fontsize: int = 10
    internal_y_label_alpha: float = 0.6
    internal_y_label_zorder: int = 2
    internal_y_label_bbox_pad: float = 0.35  # white background padding
    internal_y_label_color: str = "lightgray"

    # legend frame (white background)
    legend_frame: bool = True
    legend_frame_alpha: float = 0.85

    cmap: Optional[str] = None
    tick_pad: float = 0.0
    title_pad: float = 0
    spine_lw: float = 1.0
    tick_font_color: str = "#000000"


@dataclass(frozen=True)
class CellDrawSpec:
    draw_hlines: bool = True
    draw_internal_y_labels: bool = True



CellSpecFn = Callable[[Any, int, int, PanelSpec], Optional[CellDrawSpec]]






# -----------------------------
# Validation and small helpers
# -----------------------------


def _resolve_axis_setting(global_val, local_val):
    return global_val if global_val is not None else local_val


def _ensure_metrics_present(values: pd.DataFrame, panels: Sequence[PanelSpec]) -> List[str]:
    metric_names = [p.metric_name for p in panels]
    missing = [m for m in metric_names if m not in values.columns]
    if missing:
        raise ValueError(f"Missing metrics in values.columns: {missing}")
    return metric_names


def _ensure_same_index_and_columns(
    base: pd.DataFrame, other: pd.DataFrame, *, base_name: str, other_name: str
) -> None:
    if (not base.index.equals(other.index)) or (not base.columns.equals(other.columns)):
        raise ValueError(f"{other_name} must match {base_name} in both index and columns.")


def _compute_group_bounds_from_index(idx: pd.Index) -> List[Tuple[Hashable, int, int]]:
    """
    For MultiIndex (>=2 levels), treat level 0 as group blocks, assuming rows are already ordered.
    Returns list of (group_value, start_row, end_row), inclusive.
    """
    if not isinstance(idx, pd.MultiIndex) or idx.nlevels < 2:
        return []

    gvals = idx.get_level_values(0).to_numpy()
    bounds: List[Tuple[Hashable, int, int]] = []
    if len(gvals) == 0:
        return bounds

    start = 0
    for i in range(1, len(gvals) + 1):
        if i == len(gvals) or gvals[i] != gvals[i - 1]:
            bounds.append((gvals[i - 1], start, i - 1))
            start = i
    return bounds


def _apply_index_label_maps(
    idx: pd.Index,
    maps: Optional[Mapping[int, Mapping[Hashable, str]]],
    *,
    sep: str = " | ",
) -> List[str]:
    """
    maps[level][raw] -> display
    If maps is None, uses str(raw) per component.
    """
    if maps is None:
        if isinstance(idx, pd.MultiIndex):
            return [sep.join(map(str, t)) for t in idx.to_list()]
        return [str(x) for x in idx.to_list()]

    if isinstance(idx, pd.MultiIndex):
        out: List[str] = []
        for t in idx.to_list():
            parts: List[str] = []
            for level, raw in enumerate(t):
                m = maps.get(level, None)
                parts.append(m.get(raw, str(raw)) if m else str(raw))
            out.append(sep.join(parts))
        return out

    m0 = maps.get(0, None)
    return [m0.get(x, str(x)) if m0 else str(x) for x in idx.to_list()]


def _default_row_colors(n: int) -> List[str]:
    return ["#CCCCCC"] * n


def _row_colors_from_group_palette(
    idx: pd.Index,
    group_bounds: List[Tuple[Hashable, int, int]],
    group_palette: Optional[Mapping[Hashable, Sequence]],
) -> List:
    if group_palette is None or len(group_bounds) == 0:
        return _default_row_colors(len(idx))

    colors: List = []
    for g, s, e in group_bounds:
        pal = list(group_palette.get(g, []))
        if len(pal) == 0:
            pal = ["#CCCCCC"]
        for j in range(s, e + 1):
            colors.append(pal[(j - s) % len(pal)])
    if len(colors) != len(idx):
        return _default_row_colors(len(idx))
    return colors


from typing import Callable, Any

Color = Any  # matplotlib accepts many types (hex, rgba tuple, etc.)

def _row_colors_from_spec(
    idx: pd.Index,
    *,
    row_colors: Optional[Sequence[Color]] = None,
    row_color_map: Optional[Mapping[Hashable, Color]] = None,
    row_color_func: Optional[Callable[[Hashable], Color]] = None,
    group_bounds: Optional[List[Tuple[Hashable, int, int]]] = None,
    group_palette: Optional[Mapping[Hashable, Sequence[Color]]] = None,
    default_color: Color = "#CCCCCC",
) -> List[Color]:
    """
    Generic row color resolution.

    Priority:
      1) row_colors list aligned with idx
      2) row_color_func(key)
      3) row_color_map[key]
      4) group_palette (by group bounds)
      5) default_color
    """
    n = len(idx)

    if row_colors is not None:
        if len(row_colors) != n:
            raise ValueError("row_colors must have length equal to number of rows.")
        return list(row_colors)

    if row_color_func is not None:
        out = [row_color_func(k) for k in idx.to_list()]
        return [default_color if c is None else c for c in out]

    if row_color_map is not None:
        out = [row_color_map.get(k, default_color) for k in idx.to_list()]
        return out

    if group_palette is not None and group_bounds is not None and len(group_bounds) > 0:
        return _row_colors_from_group_palette(idx, group_bounds, group_palette)

    return [default_color] * n



def _apply_axis_cosmetics(ax: plt.Axes, style: StyleSpec) -> None:
    if style.hide_spines_top:
        ax.spines["top"].set_visible(False)
    if style.hide_spines_right:
        ax.spines["right"].set_visible(False)
    if style.hide_spines_left:
        ax.spines["left"].set_visible(False)
    if style.hide_spines_bottom:
        ax.spines["bottom"].set_visible(style.keep_bottom_spine)


# -----------------------------
# Grid builder (pivot-native)
# -----------------------------
def _build_grid_from_index(
    *,
    nrows: int,
    panels: Sequence[PanelSpec],
    group_bounds: Optional[List[Tuple[Hashable, int, int]]] = None,
    row_labels: Optional[Sequence[str]] = None,
    group_label_map: Optional[Mapping[Hashable, str]] = None,
    metric_label_map: Optional[Mapping[str, str]] = None,
    layout: LayoutSpec = LayoutSpec(),
    style: StyleSpec = StyleSpec(),
    row_labels_for_first_panel: Optional[Sequence[str]] = None,
) -> Tuple[plt.Figure, Optional[plt.Axes], List[plt.Axes], np.ndarray]:
    """
    Returns
    -------
    fig, ax_lbl (or None), axes, y
    """
    if nrows <= 0:
        raise ValueError("nrows must be positive.")
    if group_bounds is None:
        group_bounds = []

    y = np.arange(nrows)[::-1]

    use_label_axis = (len(group_bounds) > 0) or (row_labels is not None)

    fig = plt.figure(figsize=layout.figsize, constrained_layout=False)

    if use_label_axis:
        ncols = 1 + len(panels)
        width_ratios = [layout.label_ratio] + [layout.panel_ratio] * len(panels) if layout.width_ratios is None else layout.width_ratios
        if len(width_ratios) != ncols:
            raise ValueError(f"width_ratios must have length {ncols}.")

        gs = GridSpec(
            1, ncols,
            figure=fig,
            width_ratios=width_ratios,
            wspace=layout.wspace
        )

        fig.subplots_adjust(left=layout.left, right=layout.right, top=layout.top, bottom=layout.bottom)

        ax_lbl = fig.add_subplot(gs[0, 0])
        axes = [fig.add_subplot(gs[0, j]) for j in range(1, ncols)]

        ax_lbl.set_xlim(0, 1)
        ax_lbl.set_ylim(-0.5, nrows - 0.5)
        ax_lbl.axis("off")

        if row_labels is not None:
            if len(row_labels) != nrows:
                raise ValueError("row_labels must have length nrows.")
            for i, lab in enumerate(row_labels):
                ax_lbl.text(
                    0.42, y[i], str(lab),
                    ha="left", va="center",
                    fontsize=style.label_fontsize,
                )

        for g, s, e in group_bounds:
            glab = group_label_map.get(g, g) if group_label_map else g
            yc = 0.5 * (y[s] + y[e])
            ax_lbl.text(
                0.02, yc, str(glab),
                ha="left", va="center",
                fontsize=style.label_fontsize,
            )

        ref_ax = ax_lbl
    else:
        ncols = len(panels)
        width_ratios = [layout.panel_ratio] * len(panels) if layout.width_ratios is None else layout.width_ratios
        if len(width_ratios) != ncols:
            raise ValueError(f"width_ratios must have length {ncols}.")

        gs = GridSpec(
            1, ncols,
            figure=fig,
            width_ratios=width_ratios,
            wspace=layout.wspace
        )
        fig.subplots_adjust(left=layout.left, right=layout.right, top=layout.top, bottom=layout.bottom)

        ax_lbl = None
        axes = [fig.add_subplot(gs[0, j]) for j in range(ncols)]
        ref_ax = axes[0]

        if row_labels_for_first_panel is not None:
            if len(row_labels_for_first_panel) != nrows:
                raise ValueError("row_labels_for_first_panel must have length nrows.")
            axes[0].set_yticks(y)
            axes[0].set_yticklabels(row_labels_for_first_panel, fontsize=style.tick_fontsize)

    for ax, p in zip(axes, panels):
        title = metric_label_map.get(p.metric_name, p.label) if metric_label_map else p.label
        ax.set_title(title, fontsize=style.title_fontsize, pad=10)

    if layout.draw_group_separators and len(group_bounds) > 1:
        for _, _, e in group_bounds[:-1]:
            y_data = 0.5 * (y[e] + y[e + 1])
            y_disp = ref_ax.transData.transform((0, y_data))[1]
            y_fig = fig.transFigure.inverted().transform((0, y_disp))[1]
            fig.add_artist(
                Line2D(
                    [layout.left, layout.right],
                    [y_fig, y_fig],
                    transform=fig.transFigure,
                    color="grey",
                    linewidth=layout.separator_lw,
                    zorder=10,
                )
            )

    return fig, ax_lbl, axes, y


# -----------------------------
# Width ratios helper
# -----------------------------

def width_ratios_for_split(k_panels: int, bars_share: float = 0.8, panel_ratio: float = 1.0) -> List[float]:
    """Return width_ratios list for GridSpec: [label_ratio] + [panel_ratio]*k with target share for panels."""
    if not (0.0 < bars_share < 1.0):
        raise ValueError("bars_share must be in (0,1)")
    label_share = 1.0 - bars_share
    label_ratio = (label_share / bars_share) * k_panels * panel_ratio
    return [label_ratio] + [panel_ratio] * k_panels






# ============================================================
# Long-form facet renderers: time series and slopegraph
# ============================================================

def _as_multiindex_from_cols(df: pd.DataFrame, cols: Sequence[str]) -> pd.MultiIndex:
    if len(cols) == 0:
        raise ValueError("left_group_cols must contain at least one column.")
    if len(cols) == 1:
        return pd.MultiIndex.from_arrays([df[cols[0]].to_numpy()], names=cols)
    return pd.MultiIndex.from_frame(df[list(cols)])


def _unique_facet_index(df_long: pd.DataFrame, left_group_cols: Sequence[str]) -> pd.MultiIndex:
    mi = _as_multiindex_from_cols(df_long, left_group_cols)
    # preserve first occurrence order
    seen = set()
    tuples = []
    for t in mi.to_list():
        if t not in seen:
            seen.add(t)
            tuples.append(t)
    return pd.MultiIndex.from_tuples(tuples, names=list(left_group_cols))


def _group_bounds_from_facet_index(facet_index: pd.MultiIndex) -> List[Tuple[Hashable, int, int]]:
    # Treat first level as group blocks if >=2 levels
    return _compute_group_bounds_from_index(facet_index)


def _facet_row_label(facet_tuple: Tuple, *, maps: Optional[Mapping[int, Mapping[Hashable, str]]] = None) -> str:
    if maps is None:
        if len(facet_tuple) == 1:
            return str(facet_tuple[0])
        return " | ".join(str(x) for x in facet_tuple)
    parts = []
    for level, raw in enumerate(facet_tuple):
        m = maps.get(level, None) if maps else None
        parts.append(m.get(raw, str(raw)) if m else str(raw))
    if len(parts) == 1:
        return parts[0]
    return " | ".join(parts)


def _apply_row_label_maps_from_cols(
    facet_index: pd.MultiIndex,
    *,
    row_label_maps: Optional[Mapping[str, Mapping[Hashable, str]]] = None,
) -> Mapping[int, Mapping[Hashable, str]]:
    """
    Convert a mapping keyed by column-name -> {raw: display}
    into the level-indexed structure used by _apply_index_label_maps.
    """
    if row_label_maps is None:
        return {}
    out: Dict[int, Mapping[Hashable, str]] = {}
    for level, name in enumerate(facet_index.names):
        if name in row_label_maps:
            out[level] = row_label_maps[name]
    return out