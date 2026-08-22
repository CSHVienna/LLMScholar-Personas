"""
grouped_metrics.py

Reusable grouped-bar metric grid plot.

Produces the figure style used in the paper (e.g. Figure 2, "Infrastructure-level
performance"): a left label column listing section/row labels, then one panel per
metric with horizontal bars (mean ± 95% CI), best-in-group values bolded, and
arrows indicating directional preference (↑/↓) on metric titles.

Public API
----------
- ``plot_grouped_metrics``           — one block of grouped metric panels.
- ``plot_grouped_metrics_labeled``   — same, with row/section label remapping.
- ``plot_grouped_metrics_stacked``   — *N* blocks stacked vertically, sharing
                                       the x-axis. Use when you want to slice
                                       the same metrics by several different
                                       grouping variables (e.g. k, model, field)
                                       in one figure.

All three share the same internal primitives (``_build_sections``,
``_compute_y_layout``, ``_label_col_geometry``, ``_draw_label_column``,
``_draw_metric_panel``) so styling and layout stay consistent.

Typical use (single block):

    from libs.visuals.grouped_metrics import plot_grouped_metrics

    plot_grouped_metrics(
        calls_df,
        group_configs=[
            {'label': 'Access', 'column': 'model_access', 'color': '#4A90D9',
             'order': ['Open', 'Proprietary']},
            {'label': 'Size',   'column': 'model_size',   'color': '#5DB85D',
             'order': ['Small', 'Medium', 'Large', 'XL']},
        ],
        metrics=['factuality', 'duplicates', 'consistency', ...],
        metric_directions={'factuality': '↑', 'duplicates': '↓'},
        save_path='out.pdf',
    )

Typical use (stacked):

    plot_grouped_metrics_stacked(
        panels=[
            {'df': df_k,     'panel_label': 'K',
             'group_configs': [{'label': 'Top-k', 'column': 'k',     'color': '#34495E', 'order': K_ORDER}]},
            {'df': df_model, 'panel_label': 'Model',
             'group_configs': [{'label': 'Model', 'column': 'model', 'color': '#34495E', 'order': MODEL_ORDER}]},
        ],
        metrics=metric_list,
        metric_directions=METRIC_DIRECTIONS,
        metric_labels=PLOT_LABELS,
        save_path='out.pdf',
    )
"""

from __future__ import annotations

import textwrap as _textwrap
from typing import Any, Mapping, Optional, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Default style constants (mirror gridcons.py) ─────────────────────────────
FIG_DPI = 600
TICK_FONT_SIZE = 8
TICK_FONT_COLOR = "#828282"
LABEL_FONT_SIZE = 11
SPINE_LW = 0.3
SECTION_GAP = 0.4  # vertical gap between sections (data units, i.e. row heights)

# ── Label-column layout constants (inches) ───────────────────────────────────
# Layout: [section label]──gap──[bracket]──gap──[row labels (right-aligned)]
_SEC_LEFT_PAD = 0.10  # left margin for section labels
_SEC_BX_GAP = 0.0  # gap between section label and bracket
_BX_ROW_GAP = 0.7  # gap between bracket end and row labels
_BRACKET_W_IN = 0.10  # horizontal bracket extent
_RIGHT_PAD = 0.10  # margin right of row labels
_PLAIN_SEC_ROW_GAP = 0.25  # gap between section label and row labels in 'plain' style


DEFAULT_DIRECTIONS = {
    "validity": "↑",
    "refusals": "↓",
    "factuality_author": "↑",
    "factuality_field": "↑",
    "factuality_seniority": "↑",
    "bias_location": None,  # see METRIC_DIRECTIONS in visuals/constants.py
    "consistency": None,
    "duplicates": "↓",
    "div_gender": None,
    "div_ethnicity": None,
    "div_location": None,
    "div_productivity_works": None,
    "div_productivity_citations": None,
    "parity_gender": "↑",
    "parity_ethnicity": "↑",
    "parity_works": "↑",
    "parity_citations": "↑",
    "popularity_works": None,
    "popularity_citations": None,
}


# ═════════════════════════════════════════════════════════════════════════════
# Statistical helpers
# ═════════════════════════════════════════════════════════════════════════════


def _shades(hex_color: str, n: int) -> list[tuple]:
    """Return n shades of ``hex_color``, light → dark."""
    base = np.array(mcolors.to_rgb(hex_color))
    white = np.ones(3)
    if n == 1:
        return [tuple(white * 0.25 + base * 0.75)]
    return [tuple(white * (1 - t) + base * t) for t in np.linspace(0.35, 1.0, n)]


# ═════════════════════════════════════════════════════════════════════════════
# Pure data + layout helpers (no plotting)
# ═════════════════════════════════════════════════════════════════════════════


def _build_sections(
    all_calls_df: pd.DataFrame,
    group_configs: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
) -> list[dict]:
    """
    Build the per-(section, row) data structure from a *pre-aggregated*
    DataFrame (the output of ``aggregate_scores`` — columns are
    ``{metric}_mean``, ``{metric}_ci``, ``{metric}_n``, plus a global ``n``).

    Each returned section is a dict ``{'label', 'color', 'rows'}`` where each
    row carries its display label and the per-metric mean/ci/n values.
    """
    sections: list[dict] = []
    for gc in group_configs:
        col = gc["column"]
        src_df = all_calls_df
        for fk, fv in gc.get("filter", {}).items():
            if fk in src_df.columns:
                src_df = src_df[src_df[fk] == fv]
        if col not in src_df.columns:
            print(f"Warning: '{col}' not found — skipping '{gc['label']}'.")
            continue
        available = set(src_df[col].dropna().unique())
        if "order" in gc:
            order = [v for v in gc["order"] if v in available]
            order += sorted([v for v in available if v not in gc["order"]], key=str)
        else:
            order = sorted(available, key=str)
        rows: list[dict] = []
        for val in order:
            grp = src_df[src_df[col] == val]
            row = {"label": str(val)}
            for m in metrics:
                m_mean = f"{m}_mean"
                tmp_mean = (
                    grp[m_mean].dropna()
                    if m_mean in grp.columns
                    else pd.Series(dtype=float)
                )
                row[m_mean] = tmp_mean.iloc[0] if not tmp_mean.empty else np.nan

                m_ci = f"{m}_ci"
                tmp_ci = (
                    grp[m_ci].dropna()
                    if m_ci in grp.columns
                    else pd.Series(dtype=float)
                )
                row[m_ci] = tmp_ci.iloc[0] if not tmp_ci.empty else np.nan

                m_n = f"{m}_n"
                cn = "n"
                row[m_n] = grp[cn].dropna().iloc[0] if cn in grp.columns else np.nan
            rows.append(row)
        if rows:
            sections.append(
                {
                    "label": gc["label"],
                    "color": gc["color"],
                    "rows": rows,
                    # Per-section shading flag. When False, all rows in the section
                    # use the base ``color`` uniformly (no light→dark gradient).
                    "shade": gc.get("shade", True),
                }
            )
    return sections


def _compute_y_layout(
    sections: list[dict],
    section_gap: float = SECTION_GAP,
    y_sep: float = 0.7,
) -> tuple[dict, list, float, list]:
    """
    Compute y-coordinates for each (section, row) cell.

    Returns
    -------
    y_lookup   : dict, (section_idx, row_idx) → y
    sec_ranges : list of (y_top, y_bot) per section
    y_max      : y of the last row
    sep_ys     : list of separator y-positions between adjacent sections
    """
    y = 0.0
    y_lookup, sec_ranges = {}, []
    for si, sec in enumerate(sections):
        if si > 0:
            y += section_gap
        y_top = y
        for ri in range(len(sec["rows"])):
            y_lookup[(si, ri)] = y
            y += y_sep
        sec_ranges.append((y_top, y - y_sep))
    y_max = y - y_sep
    sep_ys = [
        (sec_ranges[i][1] + sec_ranges[i + 1][0]) / 2 for i in range(len(sections) - 1)
    ]
    return y_lookup, sec_ranges, y_max, sep_ys


def _label_col_geometry(
    sections: list[dict],
    textwrap_width: int,
    tick_font_size: int,
    label_font_size: int,
    *,
    section_style: str = "bracket",
    section_label_width: Optional[float] = None,
    section_label_pad: Optional[float] = None,
) -> dict:
    """
    Compute label-column geometry primitives. Decoupled from the choice of
    column width so callers can override it (e.g. unify width across stacked
    rows). Pair with ``_bx_for_width`` to derive the bracket x position.

    Parameters
    ----------
    section_style : {'bracket', 'plain'}
        Controls the label-column chrome:
          - 'bracket' (default): section label on the left with a vertical
            bracket connecting to its rows.
          - 'plain':  section label left-aligned, no bracket — tighter
            horizontal footprint.
    section_label_width : float, optional
        Override (inches) for the section-label width. Use when the auto
        estimate (0.5 em per char) underestimates bold/wide-font text and
        section labels crash into the row labels.
    section_label_pad : float, optional
        Override (inches) for the gap between section label and row labels
        in 'plain' mode. Defaults to ``_PLAIN_SEC_ROW_GAP``.

    Returns
    -------
    dict with keys
      row_label_in    : longest row label width (inches)
      sec_label_in    : longest (wrapped) section-label width (inches)
      natural_w       : auto-computed label column width (inches)
      single_section  : True if section chrome (label + bracket) should be hidden
    """
    max_label_chars = max(
        (len(row["label"]) for sec in sections for row in sec["rows"]), default=10
    )
    max_sec_chars = max(
        max(
            (
                len(line)
                for line in _textwrap.fill(
                    sec["label"], width=textwrap_width, break_long_words=True
                ).split("\n")
            ),
            default=8,
        )
        for sec in sections
    )

    # /72  → points-to-inches conversion (1 in = 72 pt).
    # 0.50 → average glyph width as a fraction of font size (em).
    row_label_in = max_label_chars * tick_font_size * 0.50 / 72
    sec_label_in = (
        section_label_width
        if section_label_width is not None
        else max_sec_chars * label_font_size * 0.72 * 0.50 / 72
    )
    sec_row_gap = (
        section_label_pad if section_label_pad is not None else _PLAIN_SEC_ROW_GAP
    )

    single_section = len(sections) == 1
    if single_section:
        # Section chrome hidden either way → reclaim that horizontal space.
        natural_w = max(1.0, _SEC_LEFT_PAD + row_label_in + _RIGHT_PAD)
    elif section_style == "plain":
        # No bracket: section label sits left-aligned next to row labels.
        natural_w = max(
            1.2,
            _SEC_LEFT_PAD + sec_label_in + sec_row_gap + row_label_in + _RIGHT_PAD,
        )
    else:  # 'bracket'
        natural_w = max(
            1.8,
            _SEC_LEFT_PAD
            + _SEC_BX_GAP
            + _BX_ROW_GAP
            + _BRACKET_W_IN
            + _RIGHT_PAD
            + sec_label_in
            + row_label_in,
        )

    return {
        "row_label_in": row_label_in,
        "sec_label_in": sec_label_in,
        "natural_w": natural_w,
        "single_section": single_section,
    }


def _bx_for_width(geom: dict, width: float) -> float:
    """
    Bracket x position (normalized [0,1]) for a label column of given ``width``
    (inches). Bracket sits ``_BX_ROW_GAP`` to the left of the row labels (whose
    right edge is at x≈0.98).
    """
    row_label_left = 0.98 - geom["row_label_in"] / width
    return row_label_left - _BX_ROW_GAP / width - _BRACKET_W_IN / width


# ═════════════════════════════════════════════════════════════════════════════
# Drawing primitives (operate on already-created axes)
# ═════════════════════════════════════════════════════════════════════════════


def _draw_label_column(
    lax: plt.Axes,
    sections: list[dict],
    sec_ranges: list,
    y_lookup: dict,
    y_lo: float,
    y_hi: float,
    label_col_w: float,
    BX: Optional[float],
    *,
    geom: dict,
    tick_font_size: int,
    tick_font_color: str,
    label_font_size: int,
    spine_lw: float,
    textwrap_width: int,
    section_style: str = "bracket",
    section_label_align: Optional[str] = None,
    row_label_align: str = "right",
    row_label_pad: float = 0.05,
) -> None:
    """
    Draw section labels, brackets, and row labels onto ``lax``.

    ``section_style='plain'`` skips the bracket entirely and left-aligns the
    section label next to its rows for a tighter, no-wasted-space layout.
    ``BX`` is ignored in plain mode and may be None.

    ``section_label_align`` / ``row_label_align`` independently control the
    horizontal alignment of the two label columns. Each is either ``'left'``
    or ``'right'`` and picks an anchor + matching ha for the text. Defaults:
    section labels = ``'right'`` in bracket mode, ``'left'`` in plain mode;
    row labels = ``'right'`` in both modes.

    ``row_label_pad`` is the inch padding between the row labels' right edge
    and the first metric panel. Increase to push row labels left (toward the
    section labels); decrease for tight packing.
    """
    lax.set_xlim(0, 1)
    lax.set_ylim(y_lo, y_hi)
    lax.invert_yaxis()
    lax.axis("off")

    # Resolve mode-appropriate default for the section-label alignment.
    if section_label_align is None:
        section_label_align = "right" if section_style == "bracket" else "left"

    # Single-section figures hide section chrome (label + bracket) entirely —
    # it'd be visual noise without conveying any grouping information.
    multi_section = len(sections) > 1
    sec_label_in = geom["sec_label_in"]
    row_label_in = geom["row_label_in"]
    bracket_w_norm = _BRACKET_W_IN / label_col_w

    # Section-label anchor: x position + ha. In bracket mode we keep the
    # legacy right-anchored-at-SEC_LEFT_PAD placement (the text overflows
    # leftward into the figure margin); the 'left' alignment uses a zone
    # immediately to the right of SEC_LEFT_PAD.
    if section_label_align == "right":
        sec_anchor_x = (
            _SEC_LEFT_PAD
            if section_style == "bracket"
            else (_SEC_LEFT_PAD + sec_label_in)
        ) / label_col_w
        sec_ha = "right"
    else:  # 'left'
        sec_anchor_x = _SEC_LEFT_PAD / label_col_w
        sec_ha = "left"

    # Row-label anchor: text always fits inside [row_zone_left, row_zone_right];
    # we anchor at one end of that zone and match ha. ``row_label_pad`` is the
    # absolute (inches) right padding between row labels and the first metric
    # panel — fixed in inches rather than as a fraction of label_col_w so the
    # layout doesn't drift as the column widens.
    row_zone_right = (label_col_w - row_label_pad) / label_col_w
    row_zone_left = row_zone_right - row_label_in / label_col_w
    if row_label_align == "left":
        row_anchor_x, row_ha = row_zone_left, "left"
    else:  # 'right'
        row_anchor_x, row_ha = row_zone_right, "right"

    for si, (sec, (y_top, y_bot)) in enumerate(zip(sections, sec_ranges)):
        y_c = (y_top + y_bot) / 2

        if multi_section:
            wrapped = _textwrap.fill(
                sec["label"], width=textwrap_width, break_long_words=True
            )
            lax.text(
                sec_anchor_x,
                y_c,
                wrapped,
                ha=sec_ha,
                va="center",
                multialignment=sec_ha,
                fontsize=label_font_size * 0.72,
                fontweight="bold",
            )
            if section_style == "bracket":
                lax.plot(
                    [BX, BX], [y_top - 0.3, y_bot + 0.3], color="#444", lw=spine_lw * 3
                )
                lax.plot(
                    [BX, BX + bracket_w_norm],
                    [y_top - 0.3, y_top - 0.3],
                    color="#444",
                    lw=spine_lw * 3,
                )
                lax.plot(
                    [BX, BX + bracket_w_norm],
                    [y_bot + 0.3, y_bot + 0.3],
                    color="#444",
                    lw=spine_lw * 3,
                )

        for ri, row in enumerate(sec["rows"]):
            wrapped = _textwrap.fill(
                row["label"], width=textwrap_width, break_long_words=True
            )
            lax.text(
                row_anchor_x,
                y_lookup[(si, ri)],
                wrapped,
                ha=row_ha,
                va="center",
                multialignment=row_ha,
                fontsize=tick_font_size,
                color=tick_font_color,
            )


def _draw_metric_panel(
    ax: plt.Axes,
    metric: str,
    sections: list[dict],
    y_lookup: dict,
    y_lo: float,
    y_hi: float,
    dirs: Mapping[str, Optional[str]],
    labels: Mapping[str, str],
    panel_width: float,
    *,
    tick_font_size: int,
    tick_font_color: str,
    spine_lw: float,
    show_title: bool = True,
    show_xaxis: bool = True,
    show_mid_gridline: bool = True,
) -> None:
    """
    Draw one metric panel (bars, errors, value labels, chrome) onto ``ax``.

    Section separators are NOT drawn here — they're rendered by
    ``_draw_section_separators`` at the figure level so they can span the
    full row (including the label column and inter-panel gaps).

    ``show_xaxis`` gates the bottom spine, tick marks, and tick labels together
    — set False on non-bottom rows of a stacked figure for a clean shared-axis
    look. ``show_mid_gridline`` toggles the vertical 0.5 gridline; the x-tick
    *positions* drive the grid, so dropping the 0.5 tick drops its gridline.
    """
    direction = dirs.get(metric)
    arrow = f" {direction}" if direction else ""
    nice = labels.get(metric, metric)

    if show_title:
        ax.set_title(
            f"{nice}{arrow}", fontsize=tick_font_size, fontweight="bold", pad=4
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(y_lo, y_hi)
    ax.invert_yaxis()
    ax.set_yticks([])

    # X-tick positions also drive the vertical grid (drawn via ax.grid),
    # so dropping the 0.5 tick is the cleanest way to hide its gridline.
    if show_mid_gridline:
        ax.set_xticks([0, 0.5, 1])
        tick_labels = ["0", "", "1"]  # blank middle label avoids "1.00.0" overlap
    else:
        ax.set_xticks([0, 1])
        tick_labels = ["0", "1"]

    if show_xaxis:
        ax.set_xticklabels(tick_labels)
        ax.tick_params(
            axis="x",
            labelsize=tick_font_size,
            labelcolor=tick_font_color,
            width=spine_lw,
        )
        ax.spines["bottom"].set_linewidth(spine_lw)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.spines["bottom"].set_visible(False)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.4, alpha=0.3, zorder=0)

    # Estimate label text width in data-coord units (panel spans [0, 1] mapped
    # onto ``panel_width`` inches).
    CHAR_W_DATA = (tick_font_size - 1) / 72 * 0.65 / panel_width  # data units per char
    MARGIN = 0.97  # don't let labels overflow

    for si, sec in enumerate(sections):
        n_rows = len(sec["rows"])
        # Flat color when sec['shade'] is False, light→dark gradient otherwise.
        colors = (
            _shades(sec["color"], n_rows)
            if sec.get("shade", True)
            else [sec["color"]] * n_rows
        )
        vals = [row[f"{metric}_mean"] for row in sec["rows"]]
        cis = [row[f"{metric}_ci"] for row in sec["rows"]]
        ns = [row[f"{metric}_n"] for row in sec["rows"]]

        good = [(i, v) for i, v in enumerate(vals) if np.isfinite(v)]
        if direction == "↑" and good:
            best = max(v for _, v in good)
        elif direction == "↓" and good:
            best = min(v for _, v in good)
        else:
            best = None

        for ri, (val, ci, n, color) in enumerate(zip(vals, cis, ns, colors)):
            yv = y_lookup[(si, ri)]
            if not np.isfinite(val):
                continue
            is_best = best is not None and abs(val - best) < 1e-9
            low_n = n < 3
            ax.barh(yv, val, height=0.55, color=color, alpha=0.92, zorder=2)
            ci_val = ci if (np.isfinite(ci) and not low_n) else 0.0
            if ci_val:
                ax.errorbar(
                    val,
                    yv,
                    xerr=ci_val,
                    fmt="none",
                    color="#333",
                    lw=0.9,
                    capsize=2,
                    zorder=3,
                )

            txt = f'{val:.2f}{"*" if low_n else ""}'
            fw = "bold" if is_best else "normal"
            gap = ci_val + 0.015
            x_out = val + gap  # outside (right of bar)
            x_in = val - gap  # inside  (left of bar end)
            txt_w = len(txt) * CHAR_W_DATA

            # Place outside unless it would overflow the axis margin.
            if x_out + txt_w <= MARGIN:
                ax.text(
                    x_out,
                    yv,
                    txt,
                    ha="left",
                    va="center",
                    fontsize=tick_font_size - 1,
                    fontweight=fw,
                    zorder=4,
                    clip_on=True,
                )
            else:
                ax.text(
                    x_in,
                    yv,
                    txt,
                    ha="right",
                    va="center",
                    fontsize=tick_font_size - 1,
                    fontweight=fw,
                    zorder=4,
                    clip_on=True,
                )


def _draw_panel_label(
    plax: plt.Axes,
    text: Optional[str],
    *,
    fontsize: int,
    rotation: float,
    bold: bool,
) -> None:
    """
    Render a panel-level ylabel-style marker on a small dedicated axes.

    No-op (axes cleared) when ``text`` is falsy.
    """
    plax.set_xlim(0, 1)
    plax.set_ylim(0, 1)
    plax.axis("off")
    if text:
        plax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            rotation=rotation,
            fontsize=fontsize,
            fontweight=("bold" if bold else "normal"),
        )


def _draw_section_separators(
    fig: plt.Figure,
    axes_row: Sequence[plt.Axes],
    sep_ys: Sequence[float],
    *,
    style: str = "solid",
    color: str = "#888",
    lw: float = 0.8,
    left_pad: float = 0.0,
) -> None:
    """
    Draw horizontal separator lines between adjacent sections.

    Parameters
    ----------
    fig : Figure
    axes_row : sequence of Axes
        Axes that share the same y-data limits (label column + metric panels).
        The leftmost provides the y-data → display transform; the leftmost and
        rightmost together fix the line's x-extent in figure coordinates.
    sep_ys : sequence of float
        Y-data positions for each separator (midpoints between adjacent
        sections, from ``_compute_y_layout``).
    style : {'solid', 'dashed', 'none'}
        - 'solid':  Single line spanning ``axes_row[0].x0 + left_pad`` →
                    ``axes_row[-1].x1`` in figure coords. Renders the screenshot
                    look. Axes patches are made transparent so the line is
                    visible across the inter-panel wspace gaps.
        - 'dashed': Legacy per-axes dashed line inside each metric panel only
                    (skips the label column).
        - 'none':   No separator drawn.
    color, lw : style overrides for the separator line.
    left_pad : float, default 0.0
        Inches by which to push the line's left endpoint to the right of the
        leftmost axes' left edge. Pass ``_SEC_LEFT_PAD`` (= 0.10") to make the
        line start at the first letter of left-aligned section titles.

    Notes
    -----
    Call this *after* any final layout step (e.g. ``fig.tight_layout``) — the
    line's x-extent is captured from ``ax.get_position()`` at call time and
    will not auto-update if axes positions later change.
    """
    if not sep_ys or style == "none":
        return

    if style == "dashed":
        # Legacy: per-axes dashed line inside the metric panels (skip label column).
        for ax in axes_row[1:]:
            for sy in sep_ys:
                ax.axhline(sy, color=color, lw=lw, ls="--", zorder=1)
        return

    # 'solid' — figure-level line spanning the whole row.
    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory

    # Make axes patches transparent so the figure-level line shows through
    # each axes' background (otherwise each axes' white facecolor would
    # cover the line in its bbox).
    for ax in axes_row:
        ax.patch.set_visible(False)

    ref_ax = axes_row[0]
    trans = blended_transform_factory(fig.transFigure, ref_ax.transData)
    # left_pad is in inches → convert to figure-fraction by dividing by fig width.
    x0 = axes_row[0].get_position().x0 + (left_pad / fig.get_figwidth())
    x1 = axes_row[-1].get_position().x1

    for sy in sep_ys:
        line = Line2D(
            [x0, x1],
            [sy, sy],
            transform=trans,
            color=color,
            lw=lw,
            linestyle="-",
            zorder=2,
            clip_on=False,
        )
        fig.add_artist(line)


# ═════════════════════════════════════════════════════════════════════════════
# Public API: single-block plot
# ═════════════════════════════════════════════════════════════════════════════


def plot_grouped_metrics(
    all_calls_df: pd.DataFrame,
    group_configs: Sequence[Mapping[str, Any]],
    *,
    metrics: Optional[Sequence[str]] = None,
    metric_directions: Optional[Mapping[str, Optional[str]]] = None,
    metric_labels: Optional[Mapping[str, str]] = None,
    figsize: Optional[tuple] = None,
    save_path: Optional[str] = None,
    tick_font_size: int = TICK_FONT_SIZE,
    tick_font_color: str = TICK_FONT_COLOR,
    label_font_size: int = LABEL_FONT_SIZE,
    spine_lw: float = SPINE_LW,
    fig_dpi: int = FIG_DPI,
    panel_width: float = 1.6,
    show: bool = True,
    textwrap_width: int = 11,
    section_gap: float = SECTION_GAP,
    section_style: str = "bracket",
    section_sep_style: str = "solid",
    show_mid_gridline: bool = True,
    row_label_width: Optional[float] = None,
    section_label_width: Optional[float] = None,
    section_label_pad: Optional[float] = None,
    section_label_align: Optional[str] = None,
    row_label_align: str = "right",
    row_label_pad: float = 0.05,
    section_sep_left_pad: float = 0.0,
) -> plt.Figure:
    """
    Plot grouped horizontal-bar metric panels with section labels and 95% CIs.

    Parameters
    ----------
    all_calls_df : DataFrame
        Per-cell pre-aggregated values. Each metric ``m`` must have columns
        ``{m}_mean``, ``{m}_ci``, ``{m}_n``.
    group_configs : list of dict
        Each dict defines one section (block of rows). Keys:
          - 'label' (str): section label shown in the left column.
          - 'column' (str): DataFrame column whose values become rows.
          - 'color' (str hex): base color; rows get a gradient of shades.
          - 'order' (list, optional): preferred row order; unknown values appended.
          - 'filter' (dict, optional): equality filters applied to the DataFrame
            before grouping (e.g. {'field_en': 'Biology'}).
          - 'shade' (bool, optional, default True): when False, all rows in the
            section use the base ``color`` uniformly (no light→dark gradient).
    metrics : list of str, optional
        Metric column names to plot. Defaults to columns of ``all_calls_df``
        present in DEFAULT_DIRECTIONS.
    metric_directions : dict, optional
        Per-metric direction: '↑' (higher better), '↓' (lower better), or None.
        Best-in-group cells are bolded for directional metrics.
    figsize : tuple, optional
        (width, height) in inches; auto-sized if omitted.
    save_path : str, optional
        If given, savefig to this path (bbox_inches='tight').
    show : bool
        If True, plt.show() and plt.close() at the end.
    section_style : {'bracket', 'plain'}, default 'bracket'
        Label-column chrome. 'bracket' = section label + vertical bracket
        connecting the rows (legacy). 'plain' = left-aligned section label,
        no bracket, tighter horizontal footprint.
    section_sep_style : {'solid', 'dashed', 'none'}, default 'solid'
        Separator between adjacent sections. 'solid' = one figure-level line
        spanning the full row (label column through last panel). 'dashed' =
        legacy per-axes dashed line inside metric panels only. 'none' = off.
    show_mid_gridline : bool, default True
        If False, the vertical 0.5 gridline inside each panel is removed
        (only the 0 and 1 ticks/grid lines remain).
    row_label_width : float, optional
        Override (inches) for the label-column axes width. Note: section
        labels and row labels share one matplotlib axes, so this controls
        the full left-of-bars label area, not just the row-label text.
        Use this if section labels overlap row labels (auto-estimation can
        be inaccurate for bold fonts, wide fonts, or non-default font sizes).

        If passed *together* with ``figsize``, the label column gets exactly
        ``row_label_width`` inches and ``panel_width`` is derived to make the
        metric panels split the remaining figure width equally — useful when
        many metric panels would otherwise squish the label column.
    section_label_width : float, optional
        Override (inches) for the section-label width estimate. Only the
        section-label portion grows; the column widens to accommodate it.
        Use when section labels need more space than auto-estimated.
    section_label_pad : float, optional
        Override (inches) for the gap between section label and row labels
        in 'plain' section style. Defaults to ~0.25" (``_PLAIN_SEC_ROW_GAP``).
    row_label_pad : float, default 0.05
        Inch padding between the row labels' right edge and the first metric
        panel. Increase to push row labels left (toward the section labels);
        decrease for a tight pack against the bars.
    section_sep_left_pad : float, default 0.0
        Inches by which to push the section-separator line's left endpoint
        rightward from the label column's axes edge. Pass ``0.10`` (the value
        of ``_SEC_LEFT_PAD``) to make the line start exactly at the first
        letter of left-aligned section titles.
    section_label_align : {'left', 'right'}, optional
        Horizontal alignment of the section labels (first label column).
        Default 'right' in 'bracket' style, 'left' in 'plain' style.
    row_label_align : {'left', 'right'}, default 'right'
        Horizontal alignment of the row labels (second label column).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if metrics is None:
        metrics = [f"{m}_mean" for m in all_calls_df.columns if m in DEFAULT_DIRECTIONS]
    dirs = {**DEFAULT_DIRECTIONS, **(metric_directions or {})}
    labels = metric_labels or {}

    # 1. Build sections + layout.
    sections = _build_sections(all_calls_df, group_configs, metrics)
    if not sections or not metrics:
        raise ValueError("No valid sections or metrics.")
    y_lookup, sec_ranges, y_max, sep_ys = _compute_y_layout(sections, section_gap)
    geom = _label_col_geometry(
        sections,
        textwrap_width,
        tick_font_size,
        label_font_size,
        section_style=section_style,
        section_label_width=section_label_width,
        section_label_pad=section_label_pad,
    )
    label_col_w = row_label_width if row_label_width is not None else geom["natural_w"]
    BX = _bx_for_width(geom, label_col_w) if section_style == "bracket" else None

    # 2. Figure.
    n_m = len(metrics)
    pad = 0.4
    if figsize is None:
        # Auto-size figure from per-column widths.
        w = label_col_w + panel_width * n_m + 0.5
        h = max((y_max + 2 * pad) * 0.65, 1.2)
        figsize = (w, h)
    elif row_label_width is not None:
        # User pinned both figure width AND label column width — derive
        # panel_width from the remainder so the label column actually gets
        # the requested inches (instead of being proportionally squished).
        panel_width = max(0.3, (figsize[0] - label_col_w - 0.5) / n_m)

    fig, axes = plt.subplots(
        1,
        n_m + 1,
        figsize=figsize,
        gridspec_kw={
            "width_ratios": [label_col_w] + [panel_width] * n_m,
            "wspace": 0.12,
        },
    )
    axes = np.atleast_1d(axes)
    y_lo, y_hi = -pad, y_max + pad

    # 3. Draw.
    _draw_label_column(
        axes[0],
        sections,
        sec_ranges,
        y_lookup,
        y_lo,
        y_hi,
        label_col_w,
        BX,
        geom=geom,
        tick_font_size=tick_font_size,
        tick_font_color=tick_font_color,
        label_font_size=label_font_size,
        spine_lw=spine_lw,
        textwrap_width=textwrap_width,
        section_style=section_style,
        section_label_align=section_label_align,
        row_label_align=row_label_align,
        row_label_pad=row_label_pad,
    )
    for m, ax in zip(metrics, axes[1:]):
        _draw_metric_panel(
            ax,
            m,
            sections,
            y_lookup,
            y_lo,
            y_hi,
            dirs,
            labels,
            panel_width,
            tick_font_size=tick_font_size,
            tick_font_color=tick_font_color,
            spine_lw=spine_lw,
            show_mid_gridline=show_mid_gridline,
        )

    fig.tight_layout(pad=0.3, w_pad=0.0, h_pad=0.3)

    # 4. Section separators — call AFTER tight_layout so axes positions are final.
    _draw_section_separators(
        fig, list(axes), sep_ys, style=section_sep_style, left_pad=section_sep_left_pad
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=fig_dpi)
        print(f"Saved → {save_path}")
    if show:
        plt.show()
        plt.close()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Public API: labeled variant (preprocess + delegate)
# ═════════════════════════════════════════════════════════════════════════════


def plot_grouped_metrics_labeled(
    all_calls_df: pd.DataFrame,
    group_configs: Sequence[Mapping[str, Any]],
    *,
    row_labels: Optional[Mapping[str, str]] = None,
    **plot_kwargs,
):
    """
    Same as ``plot_grouped_metrics`` plus optional label remapping for the
    section header and the row labels.

    Parameters
    ----------
    all_calls_df, group_configs, **plot_kwargs
        Forwarded to ``plot_grouped_metrics`` after preprocessing. See its
        docstring for ``metrics``, ``metric_directions``, ``metric_labels``,
        ``figsize``, ``save_path``, ``show``, etc.
    row_labels : dict, optional
        Global mapping ``{df_value: display}`` for unique values of every
        ``gc['column']``, which appear as the row labels.

    Per-config overrides (optional keys inside each ``group_config`` dict):
        - ``label_labels``   — overrides ``section_labels`` for that section
        - ``column_labels``  — overrides ``row_labels`` for that section

    Resolution order: per-config > top-level > original value.
    Unknown values fall through unchanged.

    Implementation note
    -------------------
    Row-label remapping is done by writing a derived column to a local
    copy of the DataFrame and pointing ``gc['column']`` at it; this keeps
    the downstream function (incl. aggregation, ordering, best-in-group
    highlighting) entirely unchanged.
    """
    df, new_configs = _apply_label_remap(all_calls_df, group_configs, row_labels)
    return plot_grouped_metrics(df, new_configs, **plot_kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# Public API: NEW — stacked variant (N blocks sharing x-axis)
# ═════════════════════════════════════════════════════════════════════════════


def plot_grouped_metrics_stacked(
    panels: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
    metric_directions: Optional[Mapping[str, Optional[str]]] = None,
    metric_labels: Optional[Mapping[str, str]] = None,
    row_labels: Optional[Mapping[str, str]] = None,
    figsize: Optional[tuple] = None,
    save_path: Optional[str] = None,
    tick_font_size: int = TICK_FONT_SIZE,
    tick_font_color: str = TICK_FONT_COLOR,
    label_font_size: int = LABEL_FONT_SIZE,
    spine_lw: float = SPINE_LW,
    fig_dpi: int = FIG_DPI,
    panel_width: float = 1.6,
    show: bool = True,
    textwrap_width: int = 11,
    section_gap: float = SECTION_GAP,
    panel_label_width: float = 0.35,
    panel_label_font_size: Optional[int] = None,
    panel_label_rotation: float = 90,
    panel_label_bold: bool = True,
    hspace: float = 0.25,
    section_style: str = "bracket",
    section_sep_style: str = "solid",
    show_mid_gridline: bool = True,
    row_label_width: Optional[float] = None,
    section_label_width: Optional[float] = None,
    section_label_pad: Optional[float] = None,
    section_label_align: Optional[str] = None,
    row_label_align: str = "right",
    row_label_pad: float = 0.05,
    section_sep_left_pad: float = 0.0,
) -> plt.Figure:
    """
    Stack multiple grouped-metric blocks vertically into one figure, sharing
    the x-axis across columns.

    Re-uses ``plot_grouped_metrics``'s rendering primitives (sections, label
    column, metric bars), so styling stays identical. Use when you want to
    slice the same metrics by several different grouping variables (e.g. k,
    model, field) side-by-side in one figure — the second arg of
    ``aggregate_scores`` (and so the ``column`` of each ``group_configs``)
    is what varies between panels.

    Parameters
    ----------
    panels : list of dict
        Each dict defines one stacked block (row of metric panels). Keys:
          - 'df' (DataFrame, required): pre-aggregated table
            (e.g. ``aggregate_scores(df_valid_calls, ['k'])``).
          - 'group_configs' (list of dict, required): same shape as in
            ``plot_grouped_metrics``.
          - 'panel_label' (str, optional): row label displayed at the far
            left, analogous to a y-axis label for the entire block (e.g. ``'K'``).
    metrics : list of str
        Metric column names; consistent across all panels.
    row_labels : dict, optional
        Global ``{df_value: display}`` mapping applied per panel via the same
        remap mechanism as ``plot_grouped_metrics_labeled``. Per-panel /
        per-config overrides (``column_labels``, ``label_labels``) inside each
        ``group_configs`` entry are also honored.
    panel_label_width : float
        Width (inches) of the leftmost column reserved for panel labels.
    panel_label_font_size : int, optional
        Defaults to ``label_font_size``.
    panel_label_rotation : float
        Rotation (degrees) for panel-label text. Default 90 (vertical).
        Per-panel override: ``panel['panel_label_rotation']``.
    panel_label_bold : bool
        Whether panel labels are bold. Default True.
        Per-panel override: ``panel['panel_label_bold']``.
    hspace : float
        Inter-row spacing as a fraction of average row height
        (passed to gridspec).
    section_style : {'bracket', 'plain'}, default 'bracket'
        Label-column chrome. 'bracket' = section label + vertical bracket
        connecting the rows. 'plain' = left-aligned section label, no bracket,
        tighter horizontal footprint.
    section_sep_style : {'solid', 'dashed', 'none'}, default 'solid'
        Separator between adjacent sections within each row. 'solid' = one
        figure-level line spanning the label column through the last panel.
        'dashed' = legacy per-axes dashed line in the metric panels only.
        'none' = off.
    show_mid_gridline : bool, default True
        If False, the vertical 0.5 gridline inside each panel is removed
        (only the 0 and 1 ticks/grid lines remain).
    row_label_width : float, optional
        Override (inches) for the unified label-column axes width across all
        rows. Note: section labels and row labels share one matplotlib axes,
        so this controls the full left-of-bars label area, not just the
        row-label text. Use this if section labels overlap row labels
        (auto-estimation can be inaccurate for bold fonts, wide fonts, or
        non-default font sizes).

        If passed *together* with ``figsize``, the first two columns get
        exactly ``panel_label_width + row_label_width`` inches and
        ``panel_width`` is derived to make the metric panels split the
        remaining figure width equally.
    section_label_width : float, optional
        Override (inches) for the section-label width estimate, applied
        per-row before the unified column width is computed.
    section_label_pad : float, optional
        Override (inches) for the gap between section label and row labels
        in 'plain' section style.
    row_label_pad : float, default 0.05
        Inch padding between the row labels' right edge and the first metric
        panel. Increase to push row labels left.
    section_sep_left_pad : float, default 0.0
        Inches by which to push each row's section-separator line rightward
        from its label column's axes edge. Pass ``0.10`` to align with the
        first letter of left-aligned section titles.
    section_label_align : {'left', 'right'}, optional
        Horizontal alignment of the section labels (first label column).
        Default 'right' in 'bracket' style, 'left' in 'plain' style.
    row_label_align : {'left', 'right'}, default 'right'
        Horizontal alignment of the row labels (second label column).

    Per-section keys honored inside each ``group_configs`` entry:
        - ``shade`` (bool, default True): when False, all rows in the section
          use the base ``color`` uniformly (no light→dark gradient).

    All other parameters carry the same semantics as ``plot_grouped_metrics``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if not panels:
        raise ValueError("`panels` must be non-empty.")

    dirs = {**DEFAULT_DIRECTIONS, **(metric_directions or {})}
    labels = metric_labels or {}
    panel_label_font_size = panel_label_font_size or label_font_size

    # 1. Per-row: optional remap + build sections + layout primitives.
    rows_data: list[dict] = []
    for p in panels:
        df_p, configs_p = _apply_label_remap(p["df"], p["group_configs"], row_labels)
        sections = _build_sections(df_p, configs_p, metrics)
        if not sections:
            raise ValueError(
                f"Panel '{p.get('panel_label', '?')}' produced no valid sections. "
                f"Check group_configs and DataFrame columns."
            )
        y_lookup, sec_ranges, y_max, sep_ys = _compute_y_layout(sections, section_gap)
        geom = _label_col_geometry(
            sections,
            textwrap_width,
            tick_font_size,
            label_font_size,
            section_style=section_style,
            section_label_width=section_label_width,
            section_label_pad=section_label_pad,
        )
        rows_data.append(
            {
                "panel_label": p.get("panel_label"),
                # Per-panel overrides fall back to the function-level defaults.
                "panel_label_bold": p.get("panel_label_bold", panel_label_bold),
                "panel_label_rotation": p.get(
                    "panel_label_rotation", panel_label_rotation
                ),
                "sections": sections,
                "y_lookup": y_lookup,
                "sec_ranges": sec_ranges,
                "y_max": y_max,
                "sep_ys": sep_ys,
                "geom": geom,
            }
        )

    # 2. Unify label-column width across rows so bars line up vertically.
    label_col_w = (
        row_label_width
        if row_label_width is not None
        else max(r["geom"]["natural_w"] for r in rows_data)
    )
    for rd in rows_data:
        rd["BX"] = (
            _bx_for_width(rd["geom"], label_col_w)
            if section_style == "bracket"
            else None
        )

    # 3. Figure.
    n_m = len(metrics)
    pad = 0.4
    row_h = [max((rd["y_max"] + 2 * pad) * 0.65, 1.2) for rd in rows_data]
    if figsize is None:
        # Auto-size figure from per-column widths.
        w = panel_label_width + label_col_w + panel_width * n_m + 0.5
        figsize = (w, sum(row_h))
    elif row_label_width is not None:
        # User pinned both figure width AND label column width — derive
        # panel_width from the remainder so the first two columns actually
        # get (panel_label_width + label_col_w) inches and the metric
        # panels split the rest equally.
        panel_width = max(
            0.3, (figsize[0] - panel_label_width - label_col_w - 0.5) / n_m
        )

    fig, axes = plt.subplots(
        len(rows_data),
        n_m + 2,
        figsize=figsize,
        gridspec_kw={
            "width_ratios": [panel_label_width, label_col_w] + [panel_width] * n_m,
            "height_ratios": row_h,
            "wspace": 0.12,
            "hspace": hspace,
        },
        squeeze=False,
    )

    # 4. Draw each row. Title appears only on the top row; x-tick labels only
    #    on the bottom row (visual x-axis sharing).
    n_rows = len(rows_data)
    for r_i, rd in enumerate(rows_data):
        is_top = r_i == 0
        is_bottom = r_i == n_rows - 1
        y_lo, y_hi = -pad, rd["y_max"] + pad

        # 4a. Panel label (far-left mini axes, like a row-level ylabel).
        _draw_panel_label(
            axes[r_i, 0],
            rd["panel_label"],
            fontsize=panel_label_font_size,
            rotation=rd["panel_label_rotation"],
            bold=rd["panel_label_bold"],
        )

        # 4b. Label column.
        _draw_label_column(
            axes[r_i, 1],
            rd["sections"],
            rd["sec_ranges"],
            rd["y_lookup"],
            y_lo,
            y_hi,
            label_col_w,
            rd["BX"],
            geom=rd["geom"],
            tick_font_size=tick_font_size,
            tick_font_color=tick_font_color,
            label_font_size=label_font_size,
            spine_lw=spine_lw,
            textwrap_width=textwrap_width,
            section_style=section_style,
            section_label_align=section_label_align,
            row_label_align=row_label_align,
            row_label_pad=row_label_pad,
        )

        # 4c. Metric panels — spine/ticks/labels only on the bottom row.
        for m_i, m in enumerate(metrics):
            _draw_metric_panel(
                axes[r_i, 2 + m_i],
                m,
                rd["sections"],
                rd["y_lookup"],
                y_lo,
                y_hi,
                dirs,
                labels,
                panel_width,
                tick_font_size=tick_font_size,
                tick_font_color=tick_font_color,
                spine_lw=spine_lw,
                show_title=is_top,
                show_xaxis=is_bottom,
                show_mid_gridline=show_mid_gridline,
            )

        # 4d. Section separators (within this row, spanning label col + panels).
        _draw_section_separators(
            fig,
            list(axes[r_i, 1:]),
            rd["sep_ys"],
            style=section_sep_style,
            left_pad=section_sep_left_pad,
        )

    # NOTE: deliberately *not* calling fig.tight_layout — it fights with the
    # explicit gridspec hspace/height_ratios. ``bbox_inches='tight'`` on save
    # trims outer whitespace cleanly.
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=fig_dpi)
        print(f"Saved → {save_path}")
    if show:
        plt.show()
        plt.close()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers — label remapping (shared by labeled + stacked variants)
# ═════════════════════════════════════════════════════════════════════════════


def _apply_label_remap(
    df: pd.DataFrame,
    group_configs: Sequence[Mapping[str, Any]],
    row_labels: Optional[Mapping[str, str]] = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Apply optional row-label remapping by writing derived columns to a copy
    of ``df`` and pointing each ``gc['column']`` at the derived column.

    No-op (returns the inputs essentially unchanged, just with the helper
    keys stripped) when no mapping is provided.

    Returns
    -------
    (new_df, new_configs)
    """
    df = df.copy()
    new_configs: list[dict] = []
    for i, gc in enumerate(group_configs):
        # Strip extension keys before forwarding to plot_grouped_metrics.
        new_gc = {
            k: v for k, v in gc.items() if k not in ("label_labels", "column_labels")
        }

        col_map = _merge_label_map(row_labels, gc.get("column_labels"))
        src_col = gc.get("column")
        if col_map and src_col and src_col in df.columns:
            derived = f"__lbl__{src_col}__{i}"
            df[derived] = _remap_series(df[src_col], col_map)
            new_gc["column"] = derived
            if "order" in gc:
                new_gc["order"] = _remap_order(gc["order"], col_map)
            # NB: ``filter`` references its own column(s), unrelated to
            # gc['column'], so nothing to do for filters.
        new_configs.append(new_gc)
    return df, new_configs


def _merge_label_map(
    global_map: Optional[Mapping[str, str]],
    local_map: Optional[Mapping[str, str]],
) -> dict:
    """Merge a global mapping with a per-config one; local wins."""
    return {**(global_map or {}), **(local_map or {})}


def _remap_series(s: pd.Series, mapping: Mapping[str, str]) -> pd.Series:
    """Apply a {orig: display} mapping to a Series, preserving NaN and
    leaving unmapped values unchanged."""
    return s.map(lambda v: mapping.get(v, v) if pd.notna(v) else v)


def _remap_order(
    order: Optional[Sequence], mapping: Mapping[str, str]
) -> Optional[list]:
    """Apply mapping to an order list. Accepts either original or already-
    mapped values, so users can specify ``order`` in whichever form they
    have on hand."""
    if order is None:
        return None
    return [mapping.get(v, v) for v in order]
