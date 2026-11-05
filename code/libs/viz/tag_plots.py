"""
Plots of tag combinations: stacked bars per model, overall distributions, and the paper figure
(both in one: per-model panel 2/3 + overall panel 1/3, vertical bars, A4 width).
All sizes (fonts, lines, export dpi) come from the style: call ``viz.style.apply_style()`` first.

Colour encodes validity and frequency: valid categories take shades of green, the others shades
of red, darkest = most frequent. Non-valid segments are also hatched, so validity never relies on
colour alone (green/red are not separable for red-green colour-blind readers).
"""

import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.patches import Patch

from utils import constants as cons
from responses.tag_stats import CombinationCategories
from viz.labels import model_labels, tag_label

TEXT_SECONDARY = '#52514e'
VALUE_FONT_SIZE = 'x-small'      # numbers written on/next to bars, relative to the style's font size
GRID_COLOR = '#e5e4e0'
ERROR_COLOR = '#0b0b0b'


class TagPalette:
    """Colour and hatch per display category (built from :class:`CombinationCategories`)."""

    def __init__(self, categories: CombinationCategories):
        self.categories = categories.categories
        self.validity = categories.validity
        self.colors: Dict[str, str] = {}
        for family_valid, ramp in ((True, cons.VALID_RAMP), (False, cons.INVALID_RAMP)):
            family = [c for c in self.categories if self.validity[c] == family_valid]
            self.colors.update(dict(zip(family, _shades(ramp, len(family)))))

    def color(self, category: str) -> str:
        return self.colors[category]

    def hatch(self, category: str) -> str:
        return '' if self.validity[category] else cons.INVALID_HATCH

    def legend_handles(self, short: bool = False) -> List[Patch]:
        return [Patch(facecolor=self.color(c), hatch=self.hatch(c), edgecolor=cons.SEGMENT_EDGE_COLOR,
                      label=tag_label(c, short)) for c in self.categories]


# -----------------------------------------------------------------------------
# Plots for inspection (horizontal bars, one figure each)
# -----------------------------------------------------------------------------
def plot_stacked_shares(shares: pd.DataFrame, palette: TagPalette, value_label: str,
                        title: Optional[str] = None) -> plt.Figure:
    """Horizontal 100% stacked bars, one per row (model) of ``shares``, with the legend on the right."""
    _, ax = plt.subplots(figsize=(9.0, max(3.0, 0.24 * len(shares) + 1.5)))
    draw_stacked_shares(ax, shares, palette, vertical=False)
    _style_axes(ax, value_label, vertical=False, title=title)
    ax.legend(handles=palette.legend_handles(), loc='upper left', bbox_to_anchor=(1.06, 1.0), title='tags')
    return ax.figure


def plot_overall(table: pd.DataFrame, palette: TagPalette, value: str, value_label: str,
                 title: Optional[str] = None) -> plt.Figure:
    """Horizontal bars per category; error bars if ``table`` has ``ci_low`` / ``ci_high``."""
    _, ax = plt.subplots(figsize=(9.0, max(2.5, 0.3 * len(table) + 1.2)))
    draw_overall(ax, table, palette, value, vertical=False)
    _style_axes(ax, value_label, vertical=False, title=title)
    return ax.figure


# -----------------------------------------------------------------------------
# Paper figure (vertical bars, both panels in one)
# -----------------------------------------------------------------------------
def plot_paper_figure(shares: pd.DataFrame, table: pd.DataFrame, palette: TagPalette, value: str,
                      value_label: str, tags_label: str, models_label: str = 'model',
                      annotation: Optional[str] = None, left_title: Optional[str] = None,
                      right_title: Optional[str] = None) -> plt.Figure:
    """Per-model stacked bars (left) and the overall distribution (right), both vertical, sharing the y axis.

    ``annotation`` (e.g. the sample sizes) is written inside the overall panel, so it is part of
    every export; panel titles only appear in PNG exports (see :func:`save_figure`).
    Tags use their short labels (``cons.TAG_SHORT_LABELS``). The legend sits above both panels in
    ``cons.PAPER_LEGEND_ROWS`` rows (columns filled top to bottom, so valid categories come first).
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(cons.PAPER_FIGURE_WIDTH_IN, cons.PAPER_FIGURE_HEIGHT_IN),
                                      sharey=True, layout='constrained',
                                      gridspec_kw={'width_ratios': cons.PAPER_FIGURE_WIDTH_RATIOS})
    draw_stacked_shares(left, shares, palette, vertical=True)
    _style_axes(left, value_label, vertical=True)
    draw_overall(right, table, palette, value, vertical=True, short_labels=True)
    _style_axes(right, value_label, vertical=True, value_axis_label=False)
    right.tick_params(axis='y', labelleft=False)
    left.set_xlabel(models_label)       # what the ticks are: models / tag combinations
    right.set_xlabel(tags_label)
    fig.align_xlabels([left, right])     # both x labels at the same height

    # Annotation inside the overall panel (top right, where the bars are lowest)
    if annotation:
        right.text(0.97, 0.97, annotation, transform=right.transAxes, ha='right', va='top', color=TEXT_SECONDARY)

    # Titles (PNG only), above the valid-share numbers (pad in points)
    pad = 1.5 * mpl.rcParams['font.size']
    for ax, title in ((left, left_title), (right, right_title)):
        if title:
            ax.set_title(title, loc='left', pad=pad)

    fig.legend(handles=palette.legend_handles(short=True), loc='outside upper center',
               ncol=math.ceil(len(palette.categories) / cons.PAPER_LEGEND_ROWS), handlelength=1.5, columnspacing=1.0)
    return fig


def save_figure(fig: plt.Figure, name: str, plots_dir: Path | str = cons.DEFAULT_PLOTS_DIR,
                formats: List[str] = cons.FIGURE_FORMATS) -> List[Path]:
    """Save ``fig`` as <plots_dir>/<name>.<format> for every format; returns the paths.

    Rule: titles are kept only in ``cons.FORMATS_WITH_TITLES`` (PNG); e.g. PDFs for the paper have none.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    # matplotlib keeps the centre, left and right titles of an axes as three Text artists
    titles = [t for ax in fig.axes for t in (ax.title, ax._left_title, ax._right_title)]
    paths = []
    for fmt in formats:
        visible = fmt in cons.FORMATS_WITH_TITLES
        for title in titles:
            title.set_visible(visible)
        path = plots_dir / f"{name}.{fmt}"
        fig.savefig(path)  # dpi, bbox and font embedding come from the style (viz/style.py)
        paths.append(path)
    for title in titles:
        title.set_visible(True)
    return paths


# -----------------------------------------------------------------------------
# Drawing on an axes (orientation-independent)
# -----------------------------------------------------------------------------
def draw_stacked_shares(ax: plt.Axes, shares: pd.DataFrame, palette: TagPalette, vertical: bool) -> None:
    """100% stacked bars (rows of ``shares`` = bars, columns = categories).

    The valid share is written at a bar's end only when it is below 100% (100% is implied).
    """
    positions = np.arange(len(shares))
    start = np.zeros(len(shares))

    # 1. Segments, in category order (valid first)
    for category in shares.columns:
        _bars(ax, vertical, positions, shares[category].values, start=start, width=0.75,
              color=palette.color(category), hatch=palette.hatch(category))
        start += shares[category].values

    # 2. Valid share at the end of each bar, only if it rounds below 100%
    valid_share = shares[[c for c in shares.columns if palette.validity[c]]].sum(axis=1)
    for position, share in zip(positions, valid_share):
        if round(share) >= 100:
            continue
        xy = (position, 101) if vertical else (101, position)
        ax.text(*xy, f"{share:.0f}", fontsize=VALUE_FONT_SIZE, color=TEXT_SECONDARY,
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


def draw_overall(ax: plt.Axes, table: pd.DataFrame, palette: TagPalette, value: str, vertical: bool,
                 short_labels: bool = False) -> None:
    """One bar per category (value column ``value``), error bars if ``ci_low`` / ``ci_high`` exist."""
    positions = np.arange(len(table))
    has_ci = {'ci_low', 'ci_high'} <= set(table.columns)
    errors = np.vstack([table[value] - table['ci_low'], table['ci_high'] - table[value]]) if has_ci else None

    for position, (_, row) in zip(positions, table.iterrows()):
        _bars(ax, vertical, [position], [row[value]], start=[0], width=0.7,
              color=palette.color(row['category']), hatch=palette.hatch(row['category']))
    if errors is not None:
        error_kw = dict(fmt='none', ecolor=ERROR_COLOR, elinewidth=0.8, capsize=1.5)
        if vertical:
            ax.errorbar(positions, table[value], yerr=errors, **error_kw)
        else:
            ax.errorbar(table[value], positions, xerr=errors, **error_kw)

    # Values at the end of the bars (after the error bar), in text ink
    ends = table['ci_high'] if has_ci else table[value]
    for position, end, v in zip(positions, ends, table[value]):
        xy = (position, end + 1) if vertical else (end, position)
        ax.text(*xy, f"{v:.1f}" if vertical else f"  {v:.1f}%", fontsize=VALUE_FONT_SIZE, color=TEXT_SECONDARY,
                ha='center' if vertical else 'left', va='bottom' if vertical else 'center')

    labels = [tag_label(c, short_labels) for c in table['category']]
    if vertical:
        ax.set_xticks(positions, labels, rotation=90)
        ax.set_xlim(-0.6, len(table) - 0.4)
    else:
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, max(100.0, float(ends.max()) * 1.15) if value == 'pct' else None)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _bars(ax: plt.Axes, vertical: bool, positions, lengths, start, width: float, color: str, hatch: str) -> None:
    """Bars along the value axis (y if vertical, x otherwise), separated by thin surface-coloured edges."""
    style = dict(color=color, hatch=hatch, edgecolor=cons.SEGMENT_EDGE_COLOR, linewidth=0.4)
    if vertical:
        ax.bar(positions, lengths, bottom=start, width=width, **style)
    else:
        ax.barh(positions, lengths, left=start, height=width, **style)


def _shades(ramp: List[str], n: int) -> List[str]:
    """``n`` colours from a dark->light ramp (evenly spaced; a single one takes the darkest)."""
    if n == 0:
        return []
    cmap = LinearSegmentedColormap.from_list('ramp', ramp)
    return [to_hex(cmap(x)) for x in (np.linspace(0, 1, n) if n > 1 else [0.0])]


def _style_axes(ax: plt.Axes, value_label: str, vertical: bool,
                title: Optional[str] = None, value_axis_label: bool = True) -> None:
    """Grid on the value axis only and no spine on the category baseline (sizes come from the style)."""
    value_axis = 'y' if vertical else 'x'
    if value_axis_label:
        (ax.set_ylabel if vertical else ax.set_xlabel)(value_label)
    ax.grid(False)
    ax.grid(axis=value_axis, color=GRID_COLOR)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'bottom' if vertical else 'left'):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, loc='left')
