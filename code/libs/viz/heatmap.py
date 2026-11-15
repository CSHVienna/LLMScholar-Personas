"""
Shared layout of the heatmaps (RQ1, RQ2, interactions): HeatmapFrame (cells, family bands, column labels, brackets,
TECHNICAL / SOCIAL bands, axis titles, a note), one colour bar per row family, and the deviation colours (green better /
red worse / grey without a direction) of the RQ2 heatmap and the interaction cells.
"""

from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle

from metrics import summary
from utils import constants as cons
from viz import constants as viscons
from viz.figures import text_on, text_size
from viz.formatting import cell


class HeatmapFrame:
    """Shared layout of the RQ heatmaps (in cell units; inches from the measured text): a y-axis title, the family
    bands (rotated, white text), the row labels (label_w inches), the cells (a gap between the TECHNICAL and SOCIAL
    columns, a gap between row blocks, optional short residual row), then below the grid the rotated column labels,
    the subgroup brackets, the TECHNICAL / SOCIAL bands and the x-axis title. The cell width is the largest that fits
    `width` (bounded by viscons.HEATMAP_MIN_CELL_W_IN and RQ1_HEATMAP_CELL_IN[0])."""

    pad, band_w, title_w, angle = 0.06, 0.17, 0.2, 45

    def __init__(self, columns: Sequence[summary.Metric], block_sizes: Sequence[int], label_w: float,
                 residual: bool = False, right_in: float = 0.1, width: Optional[float] = None):
        self.columns = list(columns)
        self.xs, x = [], 0.0
        for i, metric in enumerate(self.columns):
            x += viscons.RQ1_HEATMAP_BAND_GAP if i and metric.band != self.columns[i - 1].band else 0
            self.xs.append(x)
            x += 1
        self.width_units = x
        self.ys, self.blocks, y = [], [], 0.0     # ys: row tops per block (whole rows); blocks: (top, bottom)
        for i, n in enumerate(block_sizes):
            y += viscons.RQ1_HEATMAP_ROW_GAP if i else 0
            self.blocks.append((y, y + n))
            self.ys.append([y + k for k in range(int(np.floor(n)))] or [y])
            y += n
        self.residual_h = viscons.HEATMAP_RESIDUAL_HEIGHT
        self.y_residual = y + viscons.RQ1_HEATMAP_ROW_GAP if residual else None
        self.height_units = self.y_residual + self.residual_h if residual else y
        self.column_h = max(rotated_height(m.short or m.key, 'small', self.angle) for m in self.columns)
        self.has_brackets = any(self.columns[i].group == self.columns[i + 1].group for i in range(len(self.columns) - 1))
        self.left_in = self.title_w + self.band_w + self.pad + label_w + self.pad
        self.bottom_in = (self.pad + self.column_h + (viscons.HEATMAP_BRACKET_IN if self.has_brackets else 0)
                          + viscons.HEATMAP_BAND_IN + viscons.HEATMAP_TITLE_GAP_IN + 0.16 + self.pad)
        width = width or viscons.HEATMAP_WIDTH_IN
        cell_w = (width - self.left_in - right_in) / self.width_units
        self.cell_w = float(np.clip(cell_w, viscons.HEATMAP_MIN_CELL_W_IN, viscons.RQ1_HEATMAP_CELL_IN[0]))
        self.cell_h = viscons.RQ1_HEATMAP_CELL_IN[1]
        self.value_size = viscons.MIN_FONT_SIZE if self.cell_w < 0.27 else 'x-small'
        fig_w = self.left_in + self.cell_w * self.width_units + right_in
        fig_h = 0.08 + self.cell_h * self.height_units + self.bottom_in
        self.fig = plt.figure(figsize=(fig_w, fig_h))
        self.ax = self.fig.add_axes([self.left_in / fig_w, self.bottom_in / fig_h, self.cell_w * self.width_units / fig_w,
                                     self.cell_h * self.height_units / fig_h])
        self.ax.set_xlim(0, self.width_units)
        self.ax.set_ylim(self.height_units, 0)
        self.ax.axis('off')
        self.ux, self.uy = 1 / self.cell_w, 1 / self.cell_h          # inches -> cell units

    # cells ------------------------------------------------------------------------------------------------------
    def cell(self, x: float, y: float, color) -> None:
        self.ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor='white', linewidth=viscons.HEATMAP_BORDER,
                                    clip_on=False))

    def value(self, x: float, y: float, text: str, color: str) -> None:
        self.ax.text(x + 0.5, y + 0.5, text, ha='center', va='center_baseline', fontsize=self.value_size, color=color)

    def value_cell(self, x: float, y: float, value: float, significant: bool, cmap, norm: Normalize) -> None:
        """An effect-size cell (RQ1, interactions): fill from cmap (negatives that round to 0 shown as 0, other
        negatives as 0 colour), the value in contrasting text, or faded (RQ1_HEATMAP_FADED_TEXT) if not significant."""
        shown = 0.0 if np.isfinite(value) and -viscons.HEATMAP_ROUNDS_TO_ZERO < value < 0 else value
        fill = cmap(norm(max(shown, 0))) if np.isfinite(shown) else 'white'
        self.cell(x, y, fill)
        self.value(x, y, cell(shown), text_on(fill) if significant else viscons.RQ1_HEATMAP_FADED_TEXT)

    def residual(self, x: float, text: str) -> None:
        """A short white box with a thin grey border (all four edges drawn: clip_on=False)."""
        y, inset = self.y_residual, 0.06
        self.ax.add_patch(Rectangle((x + inset, y), 1 - 2 * inset, self.residual_h, facecolor='white',
                                    edgecolor=viscons.HEATMAP_RESIDUAL_EDGE, linewidth=0.5, clip_on=False))
        self.ax.text(x + 0.5, y + self.residual_h / 2, text, ha='center', va='center_baseline',
                     fontsize=self.value_size, color=viscons.COLOR_TEXT_SECONDARY)

    # frame ------------------------------------------------------------------------------------------------------
    def bands(self, titles: Sequence[str], families: Sequence[str]) -> None:
        """Family bands left of the row labels: base colour, rotated white title."""
        band_x = -(self.left_in - self.title_w) * self.ux
        for title, family, (top, bottom) in zip(titles, families, self.blocks):
            self.ax.add_patch(Rectangle((band_x, top), self.band_w * self.ux, bottom - top, clip_on=False,
                                        facecolor=viscons.FAMILY_COLORS[family], edgecolor='none'))
            room = (bottom - top) * self.cell_h - 0.03                    # inches available along the band
            size = next((s for s in ('x-small', viscons.MIN_FONT_SIZE) if text_size(title.upper(), s, 'bold')[0] <= room),
                        viscons.MIN_FONT_SIZE)
            self.ax.text(band_x + self.band_w * self.ux / 2, (top + bottom) / 2, title.upper(), rotation=90,
                         ha='center', va='center', color='white', fontweight='bold', fontsize=size, clip_on=False)

    def column_labels(self) -> None:
        ax, ux, uy = self.ax, self.ux, self.uy
        y0 = self.height_units + self.pad * uy
        for metric, xx in zip(self.columns, self.xs):
            ax.text(xx + 0.5, y0, metric.short or metric.key, rotation=self.angle, ha='right', va='top',
                    fontsize='small', rotation_mode='anchor', clip_on=False)
        y = y0 + self.column_h * uy + 0.03 * uy
        if self.has_brackets:
            brackets(ax, self.columns, self.xs, y)
            y += viscons.HEATMAP_BRACKET_IN * uy
        band_h = viscons.HEATMAP_BAND_IN
        for band in dict.fromkeys(m.band for m in self.columns):
            bx = [xx for m, xx in zip(self.columns, self.xs) if m.band == band]
            ax.add_patch(Rectangle((bx[0], y), bx[-1] + 1 - bx[0], band_h * uy, facecolor=viscons.RQ1_HEATMAP_BAND,
                                   edgecolor='none', clip_on=False))
            adjusted = any(m.key.endswith(cons.ADJ_SUFFIX) for m in self.columns if m.band == band)
            title = band.upper() + (' (Adj.)' if adjusted else '')        # size-adjusted: on the band, not subgroups
            ax.text((bx[0] + bx[-1] + 1) / 2, y + band_h / 2 * uy, title, ha='center', va='center_baseline',
                    fontweight='bold', fontsize='x-small', clip_on=False)
        self._x_title_y = y + (band_h + viscons.HEATMAP_TITLE_GAP_IN) * uy

    def note(self, text: str) -> None:
        """Small secondary note just above the top right corner of the figure (e.g. what a faded value means)."""
        fig = self.fig
        fig.text(1 - 0.1 / fig.get_figwidth(), 1 - 0.02 / fig.get_figheight(), text, ha='right', va='bottom',
                 fontsize=viscons.MIN_FONT_SIZE, color=viscons.COLOR_TEXT_SECONDARY)

    def axis_titles(self, y_title: Optional[str], x_title: Optional[str]) -> None:
        if y_title:
            self.ax.text(-self.left_in * self.ux + 0.02 * self.ux, self.height_units / 2, y_title, rotation=90,
                         ha='left', va='center', fontsize='small', clip_on=False, color=viscons.COLOR_TEXT)
        if x_title:
            self.ax.text(self.width_units / 2, self._x_title_y, x_title, ha='center', va='top', fontsize='small',
                         clip_on=False, color=viscons.COLOR_TEXT)


def rotated_height(text: str, size: str, degrees: float) -> float:
    w, h = text_size(text, size)
    a = np.radians(degrees)
    return w * np.sin(a) + h * np.cos(a)


def brackets(ax: plt.Axes, columns: Sequence[summary.Metric], xs: Sequence[float], y: float) -> None:
    """A bracket under every run of >= 2 adjacent columns of the same subgroup (e.g. Factuality), its name below."""
    i = 0
    while i < len(columns):
        j = i
        while j + 1 < len(columns) and columns[j + 1].group == columns[i].group and xs[j + 1] == xs[j] + 1:
            j += 1
        if j > i:
            x0, x1 = xs[i] + 0.08, xs[j] + 0.92
            ax.plot([x0, x0, x1, x1], [y, y + 0.12, y + 0.12, y], color=viscons.COLOR_TEXT_SECONDARY, linewidth=0.6,
                    clip_on=False)
            ax.text((x0 + x1) / 2, y + 0.28, columns[i].group.removesuffix(cons.ADJ_SUFFIX).capitalize(), ha='center', va='top', fontsize='x-small',
                    color=viscons.COLOR_TEXT_SECONDARY, clip_on=False)
        i = j + 1


def family_colorbars(frame: HeatmapFrame, cmaps, norm: Normalize, rows: Sequence[summary.RowGroup], label: str) -> None:
    """One colour bar per row family, next to its rows, sharing the scale; no outline; the label next to the middle."""
    fig, ax = frame.fig, frame.ax
    fig.canvas.draw()
    to_fig = (ax.transData + fig.transFigure.inverted()).transform
    right = ax.get_position().x1 + 0.1 / fig.get_figwidth()
    for i, (group, (top, bottom)) in enumerate(zip(rows, frame.blocks)):
        ys = frame.ys[i]
        (_, y_top), (_, y_bottom) = to_fig((0, top)), to_fig((0, bottom))
        cax = fig.add_axes([right, y_bottom, 0.1 / fig.get_figwidth(), y_top - y_bottom])
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmaps[group.color_key]), cax=cax)
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=viscons.MIN_FONT_SIZE, length=1.5, width=0.4, pad=1)
        bar.locator = plt.MaxNLocator(nbins=4 if len(ys) > 1 else 2)
        bar.update_ticks()
        if i == len(rows) // 2:
            bar.set_label(label, fontsize='x-small', labelpad=2)


def deviation_color(value: float, direction: int, scale: float):
    """Fill of a deviation cell (RQ2 heatmap, interaction cells): COLOR_BETTER / COLOR_WORSE by the metric's direction,
    COLOR_NO_DIRECTION (|value|) for a metric without one; white -> 85 % of the colour as |value| -> scale."""
    if not np.isfinite(value):
        return 'white'
    end = viscons.COLOR_NO_DIRECTION if direction == summary.NEUTRAL else \
        viscons.COLOR_BETTER if value * direction > 0 else viscons.COLOR_WORSE
    strength = min(abs(value) / scale, 1.0) if scale > 0 else 0.0
    return LinearSegmentedColormap.from_list('deviation', ['white', end])(viscons.DEVIATION_MAX_STRENGTH * strength)


def deviation_legend_handles(reference: str) -> List[Patch]:
    """Legend of deviation_color: better / worse than ``reference``, |deviation| for metrics without a direction."""
    return [Patch(color=viscons.COLOR_BETTER, label=f'Better than {reference}'),
            Patch(color=viscons.COLOR_WORSE, label='Worse'),
            Patch(color=viscons.COLOR_NO_DIRECTION, label='No better direction: |deviation|')]
