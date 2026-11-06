"""
Figures of tag combinations (response format / recommendation name quality): stacked bars per model,
overall distributions, and the tags paper figure (both in one: per-model panel + overall panel,
vertical bars, A4 width). Constants: ``viscons.TAGS_*``. Saving: ``viz.figures.save_figure``.
All sizes (fonts, lines, export dpi) come from the style: call ``viz.style.apply_style()`` first.

Colour encodes validity and frequency: valid categories take shades of green, the others shades
of red, darkest = most frequent. Non-valid segments are also hatched, so validity never relies on
colour alone (green/red are not separable for red-green colour-blind readers).
"""

import math
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from responses.tag_stats import CombinationCategories
from viz import constants as viscons
from viz.figures import bars, shades, style_bar_axes
from viz.labels import model_labels, tag_label


class TagPalette:
    """Colour and hatch per display category (built from :class:`CombinationCategories`)."""

    def __init__(self, categories: CombinationCategories):
        self.categories = categories.categories
        self.validity = categories.validity
        self.colors: Dict[str, str] = {}
        for family_valid, ramp in ((True, viscons.TAGS_VALID_RAMP), (False, viscons.TAGS_INVALID_RAMP)):
            family = [c for c in self.categories if self.validity[c] == family_valid]
            self.colors.update(dict(zip(family, shades(ramp, len(family)))))

    def color(self, category: str) -> str:
        return self.colors[category]

    def hatch(self, category: str) -> str:
        return '' if self.validity[category] else viscons.TAGS_INVALID_HATCH

    def legend_handles(self, short: bool = False) -> List[Patch]:
        return [Patch(facecolor=self.color(c), hatch=self.hatch(c), edgecolor=viscons.COLOR_BAR_EDGE,
                      label=tag_label(c, short)) for c in self.categories]


# -----------------------------------------------------------------------------
# Plots for inspection (horizontal bars, one figure each)
# -----------------------------------------------------------------------------
def plot_tags_per_model(shares: pd.DataFrame, palette: TagPalette, value_label: str,
                        title: Optional[str] = None) -> plt.Figure:
    """Horizontal 100% stacked bars, one per row (model) of ``shares``, with the legend on the right."""
    _, ax = plt.subplots(figsize=(9.0, max(3.0, 0.24 * len(shares) + 1.5)))
    draw_tags_per_model(ax, shares, palette, vertical=False)
    style_bar_axes(ax, value_label, vertical=False, title=title)
    ax.legend(handles=palette.legend_handles(), loc='upper left', bbox_to_anchor=(1.06, 1.0), title='tags')
    return ax.figure


def plot_tags_overall(table: pd.DataFrame, palette: TagPalette, value: str, value_label: str,
                      title: Optional[str] = None) -> plt.Figure:
    """Horizontal bars per category; error bars if ``table`` has ``ci_low`` / ``ci_high``."""
    _, ax = plt.subplots(figsize=(9.0, max(2.5, 0.3 * len(table) + 1.2)))
    draw_tags_overall(ax, table, palette, value, vertical=False)
    style_bar_axes(ax, value_label, vertical=False, title=title)
    return ax.figure


# -----------------------------------------------------------------------------
# Paper figure (vertical bars, both panels in one)
# -----------------------------------------------------------------------------
def plot_tags_paper_figure(shares: pd.DataFrame, table: pd.DataFrame, palette: TagPalette, value: str,
                           value_label: str, tags_label: str, models_label: str = 'model',
                           annotation: Optional[str] = None, left_title: Optional[str] = None,
                           right_title: Optional[str] = None) -> plt.Figure:
    """Per-model stacked bars (left) and the overall distribution (right), both vertical, sharing the y axis.

    ``annotation`` (e.g. the sample sizes) is written inside the overall panel, so it is part of
    every export; panel titles only appear in PNG exports (see :func:`viz.figures.save_figure`).
    Tags use their short labels (``viscons.TAGS_SHORT_LABELS``). The legend sits above both panels in
    ``viscons.TAGS_PAPER_LEGEND_ROWS`` rows (columns filled top to bottom, so valid categories come first).
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(viscons.PAPER_TEXT_WIDTH_IN, viscons.TAGS_PAPER_HEIGHT_IN),
                                      sharey=True, layout='constrained',
                                      gridspec_kw={'width_ratios': viscons.TAGS_PAPER_WIDTH_RATIOS})
    draw_tags_per_model(left, shares, palette, vertical=True)
    style_bar_axes(left, value_label, vertical=True)
    draw_tags_overall(right, table, palette, value, vertical=True, short_labels=True)
    style_bar_axes(right, value_label, vertical=True, value_axis_label=False)
    right.tick_params(axis='y', labelleft=False)
    left.set_xlabel(models_label)       # what the ticks are: models / tag combinations
    right.set_xlabel(tags_label)
    fig.align_xlabels([left, right])     # both x labels at the same height

    # Annotation inside the overall panel (top right, where the bars are lowest)
    if annotation:
        right.text(0.97, 0.97, annotation, transform=right.transAxes, ha='right', va='top',
                   color=viscons.COLOR_TEXT_SECONDARY)

    # Titles (PNG only), above the valid-share numbers (pad in points)
    pad = 1.5 * mpl.rcParams['font.size']
    for ax, title in ((left, left_title), (right, right_title)):
        if title:
            ax.set_title(title, loc='left', pad=pad)

    fig.legend(handles=palette.legend_handles(short=True), loc='outside upper center',
               ncol=math.ceil(len(palette.categories) / viscons.TAGS_PAPER_LEGEND_ROWS),
               handlelength=1.5, columnspacing=1.0)
    return fig


# -----------------------------------------------------------------------------
# Drawing on an axes (orientation-independent)
# -----------------------------------------------------------------------------
def draw_tags_per_model(ax: plt.Axes, shares: pd.DataFrame, palette: TagPalette, vertical: bool) -> None:
    """100% stacked bars (rows of ``shares`` = bars, columns = categories).

    The valid share is written at a bar's end only when it is below 100% (100% is implied).
    """
    positions = np.arange(len(shares))
    start = np.zeros(len(shares))

    # 1. Segments, in category order (valid first)
    for category in shares.columns:
        bars(ax, vertical, positions, shares[category].values, start=start, width=0.75,
             color=palette.color(category), hatch=palette.hatch(category))
        start += shares[category].values

    # 2. Valid share at the end of each bar, only if it rounds below 100%
    valid_share = shares[[c for c in shares.columns if palette.validity[c]]].sum(axis=1)
    for position, share in zip(positions, valid_share):
        if round(share) >= 100:
            continue
        xy = (position, 101) if vertical else (101, position)
        ax.text(*xy, f"{share:.0f}", fontsize=viscons.VALUE_FONT_SIZE, color=viscons.COLOR_TEXT_SECONDARY,
                ha='center' if vertical else 'left', va='bottom' if vertical else 'center')

    # 3. Bar labels: short model names
    labels = list(model_labels(shares.index).values())
    if vertical:
        ax.set_xticks(positions, labels, rotation=90)
        ax.set_xlim(-0.6, len(shares) - 0.4)
        ax.set_ylim(0, 100)
    else:
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)


def draw_tags_overall(ax: plt.Axes, table: pd.DataFrame, palette: TagPalette, value: str, vertical: bool,
                      short_labels: bool = False) -> None:
    """One bar per category (value column ``value``), error bars if ``ci_low`` / ``ci_high`` exist."""
    positions = np.arange(len(table))
    has_ci = {'ci_low', 'ci_high'} <= set(table.columns)
    errors = np.vstack([table[value] - table['ci_low'], table['ci_high'] - table[value]]) if has_ci else None

    for position, (_, row) in zip(positions, table.iterrows()):
        bars(ax, vertical, [position], [row[value]], start=[0], width=0.7,
             color=palette.color(row['category']), hatch=palette.hatch(row['category']))
    if errors is not None:
        error_kw = dict(fmt='none', ecolor=viscons.COLOR_ERROR_BAR, elinewidth=0.8, capsize=1.5)
        if vertical:
            ax.errorbar(positions, table[value], yerr=errors, **error_kw)
        else:
            ax.errorbar(table[value], positions, xerr=errors, **error_kw)

    # Values at the end of the bars (after the error bar), in text ink
    ends = table['ci_high'] if has_ci else table[value]
    for position, end, v in zip(positions, ends, table[value]):
        xy = (position, end + 1) if vertical else (end, position)
        ax.text(*xy, f"{v:.1f}" if vertical else f"  {v:.1f}%", fontsize=viscons.VALUE_FONT_SIZE,
                color=viscons.COLOR_TEXT_SECONDARY,
                ha='center' if vertical else 'left', va='bottom' if vertical else 'center')

    labels = [tag_label(c, short_labels) for c in table['category']]
    if vertical:
        ax.set_xticks(positions, labels, rotation=90)
        ax.set_xlim(-0.6, len(table) - 0.4)
    else:
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, max(100.0, float(ends.max()) * 1.15) if value == 'pct' else None)
