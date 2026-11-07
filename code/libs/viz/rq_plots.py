"""
Figures of the research questions, built on the metric bar figure (viz/metric_plots.plot_metric_bars).

RQ1 (results of batch_rq1.py; measure = 'omega2' or 'shapley_r2'):
    plot_rq1          bars: rows = the model and the configuration variables (grouped model / persona / context, in
                      their colours), columns = metrics, bars with cluster-bootstrap CIs; bold = largest of its group
    plot_rq1_heatmap  heatmap: the same rows (persona, context, model), columns grouped TECHNICAL / SOCIAL with
                      subgroup brackets; one colour ramp per row family on a shared scale; a value whose CI includes 0
                      is faded; with Shapley R^2 a Residual row (1 - R^2): every column then sums to 1
    plot_rq1_figure   either of them (kind = 'bars' | 'heatmap'); file name: rq1_figure_name()

RQ2 (results of batch_rq2.py; value = 'deviation' from the grand mean or 'emm', the estimated marginal mean):
    plot_rq2          bars: rows = the values of every configuration variable (grouped by variable, persona orange /
                      context blue), columns = metrics; deviations are signed bars around 0
    plot_rq2_forest   forest plot: one dot + CI per value, one panel per metric; value 'contrast' (vs the reference
                      value, named in the variable's header; a subfield vs the reference of its field) or 'deviation';
                      R^2, adjusted R^2 and Delta R^2 (LLM) in each panel
    plot_rq2_heatmap  heatmap of the deviations: green = better than the grand mean, red = worse (the metric's
                      direction), grey = metrics without a better direction; colour scaled per metric; faded = CI
                      includes 0; file names: rq2_figure_name()

RQ3 (analysis/tradeoff.model_scores of the configuration table):
    plot_rq3          scatter technical vs social score per model, quadrants Q1-Q4 (relative: medians of the models;
                      absolute: half of each axis maximum); file name: rq3_figure_name()
"""

from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

from analysis.influence import MODEL_FACTOR
from metrics import summary
from utils import constants as cons
from viz import constants as viscons
from viz.metric_plots import plot_metric_bars, text_size

RQ1_MEASURES = {'omega2': 'partial $\\omega^2$', 'shapley_r2': 'Shapley $R^2$'}
_FAMILY_TITLES = {'model': 'LLM', 'persona': 'Persona', 'context': 'Context'}
RQ1_ROW_ORDER = ('persona', 'context', 'model')      # prompt composition, then the LLM (bars and heatmap)


def load_rq(folder: Path | str, rq: str = 'rq1') -> pd.DataFrame:
    """All <metric>.parquet files of an RQ output folder (rq1 / rq2)."""
    files = sorted(Path(folder).glob('*.parquet'))
    if not files:
        raise FileNotFoundError(f"No {rq.upper()} results in {folder} (run run_metrics_pipeline.sh -s {rq})")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_rq1(folder: Path | str) -> pd.DataFrame:
    return load_rq(folder, 'rq1')


def load_rq2(folder: Path | str) -> pd.DataFrame:
    return load_rq(folder, 'rq2')


def rq1_rows(variables: Sequence[str], order: Sequence[str] = RQ1_ROW_ORDER) -> List[summary.RowGroup]:
    """One block per family (in `order`), one row per variable (in the given order)."""
    families = {f: [] for f in order}
    for v in variables:
        family = 'model' if v == MODEL_FACTOR else 'persona' if v in cons.PERSONA_VARIABLES else 'context'
        families[family].append(v)
    return [summary.RowGroup(_FAMILY_TITLES[f], 'variable', vs,
                             ['Model' if v == MODEL_FACTOR else summary.PROMPT_TITLES.get(v, v) for v in vs], f)
            for f, vs in families.items() if vs]


def rq1_metric_keys(results: pd.DataFrame, metrics: Optional[str | Sequence[str]] = None) -> List[str]:
    """Metric keys with results: all raw ones (None; the size-adjusted <key>_adj only when asked for, e.g. 'all_adj'),
    a family / subgroup ('technical', 'factuality', ...) or a list of keys."""
    available = [k for k in summary.METRICS if k in set(results['metric'])]
    if metrics is None:
        return [k for k in available if not k.endswith(summary.ADJ_SUFFIX)]
    wanted = [m.key for m in summary.select_metrics(metrics)]
    return [k for k in wanted if k in available]


def rq1_bars(results: pd.DataFrame, measure: str = 'omega2', metrics: Optional[str | Sequence[str]] = None
             ) -> Tuple[pd.DataFrame, List[summary.RowGroup], List[summary.Metric]]:
    """RQ1 results as (table, rows, metrics) of plot_metric_bars; panels scaled to the data."""
    keys = rq1_metric_keys(results, metrics)
    rows = rq1_rows(list(dict.fromkeys(results['variable'])))
    title_of = {v: g.title for g in rows for v in g.levels}
    table = pd.DataFrame({'group': results['variable'].map(title_of), 'row': results['variable'],
                          'metric': results['metric'], 'mean': results[measure].clip(lower=0),
                          'ci_low': results[f"{measure}_ci_low"].clip(lower=0),
                          'ci_high': results[f"{measure}_ci_high"].clip(lower=0)})
    columns = [replace(summary.METRICS[k], direction=summary.HIGHER, limits=None) for k in keys]
    return table, rows, columns


def plot_rq1(results: pd.DataFrame, measure: str = 'omega2', metrics: Optional[Sequence[str]] = None,
             title: Optional[str] = None) -> plt.Figure:
    table, rows, columns = rq1_bars(results, measure, metrics)
    return plot_metric_bars(table, rows, columns, title=title or f"RQ1: {RQ1_MEASURES[measure]} per metric", arrows=False,
                            ci_linewidth=viscons.RQ_CI_LINEWIDTH, ci_capsize=viscons.RQ_CI_CAPSIZE,
                            tick_size=viscons.RQ_TICK_SIZE)


def plot_rq1_figure(results: pd.DataFrame, kind: str = 'heatmap', measure: str = 'shapley_r2',
                    metrics: Optional[Sequence[str]] = None) -> plt.Figure:
    """kind = 'bars' | 'heatmap'; save it as f"rq1_{kind}_{measure}"."""
    if kind == 'bars':
        return plot_rq1(results, measure, metrics)
    if kind == 'heatmap':
        return plot_rq1_heatmap(results, measure, metrics)
    raise ValueError(f"kind must be 'bars' or 'heatmap', not {kind!r}")


def plot_rq1_heatmap(results: pd.DataFrame, measure: str = 'shapley_r2', metrics: Optional[Sequence[str]] = None,
                     residual: Optional[bool] = None, vmax: Optional[float] = None,
                     title: Optional[str] = None, width: Optional[float] = None,
                     fit_row: Optional[str] = 'auto') -> plt.Figure:
    """Heatmap of RQ1 (see module docstring). fit_row, the short row under the grid: 'residual' (1 - R^2: with
    Shapley R^2 every column then sums to 1), 'r2' (the model's R^2: partial omega^2 values are not shares of a total,
    so a residual would not complete them), 'auto' (residual for Shapley R^2, R^2 for omega^2) or None. residual=
    True / False (older argument) = 'residual' / None. The layout (HeatmapFrame) fits `width` (default
    viscons.HEATMAP_WIDTH_IN, the A4 text width) by narrowing the cells."""
    if residual is not None:
        fit_row = 'residual' if residual else None
    if fit_row == 'auto':
        fit_row = 'residual' if measure == 'shapley_r2' else 'r2'
    if fit_row not in (None, 'residual', 'r2'):
        raise ValueError(f"fit_row must be 'auto', 'residual', 'r2' or None, not {fit_row!r}")
    residual = fit_row is not None
    fit_label = {'residual': 'Residual', 'r2': '$R^2$'}.get(fit_row, '')
    keys = rq1_metric_keys(results, metrics)
    columns = [summary.METRICS[k] for k in keys]
    rows = rq1_rows(list(dict.fromkeys(results['variable'])))
    value = results.pivot(index='variable', columns='metric', values=measure)
    low = results.pivot(index='variable', columns='metric', values=f"{measure}_ci_low")
    r2 = results.groupby('metric')['r2_full'].first()
    vmax = vmax or max(0.05, float(np.nanmax(value[keys].clip(lower=0).to_numpy())))
    norm = Normalize(0, vmax)
    cmaps = {f: _family_cmap(f) for f in viscons.FAMILY_COLORS}
    labels = [label for g in rows for label in g.labels]
    label_w = max(_text_size(t, 'small', 'bold')[0] for t in labels + (['Residual'] if residual else []))
    frame = HeatmapFrame(columns, [len(g.levels) for g in rows], label_w, residual=residual, right_in=0.62,
                         width=width)
    ax, value_size = frame.ax, frame.value_size
    for group, ys in zip(rows, frame.ys):
        family = group.color_key
        for variable, label, yy in zip(group.levels, group.labels, ys):
            for metric, xx in zip(columns, frame.xs):
                v = value.at[variable, metric.key] if variable in value.index and metric.key in value.columns else np.nan
                ci_low = low.at[variable, metric.key] if variable in low.index and metric.key in low.columns else np.nan
                shown = 0.0 if np.isfinite(v) and -0.005 < v < 0 else v
                color = cmaps[family](norm(max(shown, 0))) if np.isfinite(shown) else '#ffffff'
                frame.cell(xx, yy, color)
                faded = not np.isfinite(ci_low) or ci_low <= 0
                text_color = viscons.RQ1_HEATMAP_FADED_TEXT if faded else ('white' if norm(shown) > 0.55 else '#1f1f1f')
                frame.value(xx, yy, '–' if not np.isfinite(shown) else f"{shown:.2f}", text_color)
            ax.text(-frame.pad * frame.ux, yy + 0.5, label, ha='right', va='center_baseline', fontweight='bold',
                    fontsize='small', clip_on=False, color=_label_color(family))
    if residual:
        y = frame.y_residual
        for metric, xx in zip(columns, frame.xs):
            r = r2.get(metric.key, np.nan)
            shown = (1 - r) if fit_row == 'residual' else r
            frame.residual(xx, f"{shown:.2f}" if np.isfinite(r) else '–')
        ax.text(-frame.pad * frame.ux, y + frame.residual_h / 2, fit_label, ha='right', va='center_baseline',
                fontsize='small', clip_on=False, color=viscons.COLOR_TEXT_SECONDARY)
    frame.bands([g.title for g in rows], [g.color_key for g in rows])
    frame.column_labels()
    frame.axis_titles('Factor (source of variation)', 'Evaluation metric')
    _colorbars(frame, cmaps, norm, rows, RQ1_MEASURES[measure])
    if title:
        frame.fig.suptitle(title, fontsize='small', y=1.0)
    return frame.fig


def _family_cmap(family: str) -> LinearSegmentedColormap:
    """Near-white tint -> base -> dark of a row family (zero cells: almost white, the family hue just visible)."""
    return LinearSegmentedColormap.from_list(family, [viscons.FAMILY_ZERO[family], viscons.FAMILY_COLORS[family],
                                                      viscons.RQ1_HEATMAP_DARK[family]])


def _label_color(family: str) -> str:
    return viscons.FAMILY_COLORS[family] if family != 'model' else '#7a7a7a'


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
        self.column_h = max(_rotated_height(m.short or m.key, 'small', self.angle) for m in self.columns)
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
            size = next((s for s in ('x-small', viscons.MIN_FONT_SIZE) if _text_size(title.upper(), s, 'bold')[0] <= room),
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
            _brackets(ax, self.columns, self.xs, y)
            y += viscons.HEATMAP_BRACKET_IN * uy
        band_h = viscons.HEATMAP_BAND_IN
        for band in dict.fromkeys(m.band for m in self.columns):
            bx = [xx for m, xx in zip(self.columns, self.xs) if m.band == band]
            ax.add_patch(Rectangle((bx[0], y), bx[-1] + 1 - bx[0], band_h * uy, facecolor=viscons.RQ1_HEATMAP_BAND,
                                   edgecolor='none', clip_on=False))
            adjusted = any(m.key.endswith(summary.ADJ_SUFFIX) for m in self.columns if m.band == band)
            title = band.upper() + (' (Adj.)' if adjusted else '')        # size-adjusted: on the band, not subgroups
            ax.text((bx[0] + bx[-1] + 1) / 2, y + band_h / 2 * uy, title, ha='center', va='center_baseline',
                    fontweight='bold', fontsize='x-small', clip_on=False)
        self._x_title_y = y + (band_h + viscons.HEATMAP_TITLE_GAP_IN) * uy

    def axis_titles(self, y_title: Optional[str], x_title: Optional[str]) -> None:
        if y_title:
            self.ax.text(-self.left_in * self.ux + 0.02 * self.ux, self.height_units / 2, y_title, rotation=90,
                         ha='left', va='center', fontsize='small', clip_on=False, color=viscons.METRIC_ROW_LABEL_COLOR)
        if x_title:
            self.ax.text(self.width_units / 2, self._x_title_y, x_title, ha='center', va='top', fontsize='small',
                         clip_on=False, color=viscons.METRIC_ROW_LABEL_COLOR)


def _text_size(text: str, size: str, weight: str = 'normal') -> Tuple[float, float]:
    """(width, height) in inches of a text drawn at a matplotlib font size ('small', ...)."""
    return text_size(text, size, weight)


def _rotated_height(text: str, size: str, degrees: float) -> float:
    w, h = _text_size(text, size)
    a = np.radians(degrees)
    return w * np.sin(a) + h * np.cos(a)


def _brackets(ax: plt.Axes, columns: Sequence[summary.Metric], xs: Sequence[float], y: float) -> None:
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
            ax.text((x0 + x1) / 2, y + 0.28, columns[i].group.removesuffix(summary.ADJ_SUFFIX).capitalize(), ha='center', va='top', fontsize='x-small',
                    color=viscons.COLOR_TEXT_SECONDARY, clip_on=False)
        i = j + 1


def _colorbars(frame: 'HeatmapFrame', cmaps, norm: Normalize, rows: Sequence[summary.RowGroup], label: str) -> None:
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


def rq_figure_name(rq: int, kind: str, measure: str, metrics: Optional[str | Sequence[str]] = None) -> str:
    """File name of an RQ figure: rq<rq>_<measure>_<metrics>_<kind>[_adj], metrics = 'all' / a column set / keys joined
    by '-', kind = 'heatmap' | 'bar' ('bars' accepted) | 'forest'; size-adjusted metrics: _adj at the end (e.g.
    rq1_omega2_paper_bar and rq1_omega2_paper_bar_adj, rq2_deviation_all_heatmap)."""
    selection, adjusted = summary.split_adjusted(metrics)
    kind = 'bar' if kind == 'bars' else kind
    return f"rq{rq}_{measure}_{selection}_{kind}" + (summary.ADJ_SUFFIX if adjusted else '')


def rq1_figure_name(kind: str, measure: str, metrics: Optional[str | Sequence[str]] = None) -> str:
    """rq1_<measure>_<metrics>_<kind> (see rq_figure_name; e.g. rq1_omega2_technical_heatmap)."""
    return rq_figure_name(1, kind, measure, metrics)


# --- RQ3: socio-technical trade-off ------------------------------------------------------------------------------
def plot_rq3(scores: pd.DataFrame, mode: str = 'relative', pareto: bool = False,
             technical: Sequence[str] = tuple(cons.RQ3_TECHNICAL_METRICS),
             social: Sequence[str] = tuple(cons.RQ3_SOCIAL_METRICS), title: Optional[str] = None,
             grid: bool = False, labels_in_quadrant: bool = viscons.RQ3_LABELS_IN_QUADRANT) -> plt.Figure:
    """Scatter of tradeoff.model_scores: x = technical score, y = social score, one labelled point per model coloured
    by its quadrant (its name too); Pearson correlation of the two scores in the corner. mode 'relative': split at the
    medians of the models, axes zoomed on the data; 'absolute': split at half of each axis maximum, axes from 0 to the
    maximum. pareto: dotted line through the Pareto front; grid: background grid (default off: only the split lines).

    Labels: deterministic greedy placement in pixels (_place_labels): each label takes the nearest free spot around
    its point, clear of every point, every placed label and the correlation box; labels_in_quadrant keeps it inside
    its point's quadrant when it fits. A label away from its point gets a thin, gently curved leader line."""
    from scipy.stats import pearsonr
    from analysis import tradeoff
    from viz.labels import model_labels

    split_x, split_y = tradeoff.splits(scores, mode)
    quadrant = tradeoff.quadrants(scores, mode)
    colors = list(quadrant.map(viscons.RQ3_QUADRANT_COLORS))
    fig, ax = plt.subplots(figsize=viscons.RQ3_FIGURE_IN)
    for q, description in tradeoff.QUADRANTS.items():
        chosen = quadrant == q
        ax.scatter(scores.loc[chosen, 'technical'], scores.loc[chosen, 'social'], s=viscons.RQ3_MARKER_SIZE,
                   color=viscons.RQ3_QUADRANT_COLORS[q], edgecolor='white', linewidth=0.4, zorder=3,
                   label=f"{q} · {description}")
    ax.axvline(split_x, zorder=1, **viscons.RQ3_SPLIT_LINE)
    ax.axhline(split_y, zorder=1, **viscons.RQ3_SPLIT_LINE)
    if pareto:
        front = scores[tradeoff.pareto_front(scores)].sort_values('technical')
        ax.plot(front['technical'], front['social'], zorder=2, **viscons.RQ3_PARETO_LINE)

    if mode == 'absolute':
        ax.set_xlim(0, float(scores['technical_max'].iloc[0]))
        ax.set_ylim(0, float(scores['social_max'].iloc[0]))
    else:   # room around the data for the labels of the outer points (labels stay inside the axes)
        for axis, column, (low_pad, high_pad) in ((ax.set_xlim, 'technical', viscons.RQ3_PAD_X),
                                                  (ax.set_ylim, 'social', viscons.RQ3_PAD_Y)):
            low, high = scores[column].min(), scores[column].max()
            span = high - low if high > low else 1.0
            axis(low - low_pad * span, high + high_pad * span)
    corr = None
    xs, ys = scores['technical'].to_numpy(dtype=float), scores['social'].to_numpy(dtype=float)
    if len(scores) >= 3:
        r, p = pearsonr(xs, ys)
        corr = ax.text(0.012, 0.975, f"Pearson $\\rho$ = {r:.2f} ({_stars(p)})", transform=ax.transAxes, ha='left',
                       va='top', fontsize='x-small', color=viscons.METRIC_ROW_LABEL_COLOR, zorder=5,
                       bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': 1.2})

    ax.set_xlabel(f"Technical: {tradeoff.axis_label(technical)}", fontsize='small')
    ax.set_ylabel(f"Social: {tradeoff.axis_label(social)}", fontsize='small')
    if grid:                                    # default off: only the split lines (grid(False, **style) turns it on)
        ax.grid(color=viscons.COLOR_GRID, linewidth=0.5, zorder=0)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize='x-small',
              handletextpad=0.1, columnspacing=1.2, borderaxespad=0.2)
    split = 'medians of the models' if mode == 'relative' else 'half of the maximum'
    ax.set_title(title or f"RQ3: socio-technical trade-off (split: {split})", pad=16)   # PNG only (save_figure)
    fig.tight_layout()   # final layout BEFORE placing: a later re-layout would move points under fixed-size labels
    labels = model_labels(scores[cons.COL_MODEL])
    _label_points(ax, [labels[m] for m in scores[cons.COL_MODEL]], xs, ys, colors, (split_x, split_y),
                  [corr] if corr is not None else [], labels_in_quadrant)
    return fig


def plot_rq3_inset(scores: pd.DataFrame, pareto: bool = False,
                   technical: Sequence[str] = tuple(cons.RQ3_TECHNICAL_METRICS),
                   social: Sequence[str] = tuple(cons.RQ3_SOCIAL_METRICS), title: Optional[str] = None,
                   grid: bool = False, color_by: Optional[str] = None) -> plt.Figure:
    """The absolute view (axes from 0 to the maximum, quadrants split at half of each axis maximum; points without
    labels) with a zoomed inset on the models, where the labels are (placed as in plot_rq3). The inset sits in the
    empty part of the axes (viscons.RQ3_INSET_BOUNDS); a light shaded frame in the inset's colour marks the zoomed
    area, joined to the inset by two thin connectors, with the legend of the quadrants that occur under it. color_by: a model characteristic (summary.MODEL_ATTRIBUTES: 'access', 'reasoning', 'size', 'origin') for the
    points and labels of the inset, with their own legend (the overview keeps the quadrant colours); None: quadrants."""
    from scipy.stats import pearsonr
    from analysis import tradeoff
    from viz.labels import model_labels

    split_x, split_y = tradeoff.splits(scores, 'absolute')
    quadrant = tradeoff.quadrants(scores, 'absolute')
    colors = list(quadrant.map(viscons.RQ3_QUADRANT_COLORS))
    xs, ys = scores['technical'].to_numpy(dtype=float), scores['social'].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=viscons.RQ3_INSET_FIGURE_IN)

    def draw(axes, size, legend):
        for q, description in tradeoff.QUADRANTS.items():
            chosen = (quadrant == q).to_numpy()
            axes.scatter(xs[chosen], ys[chosen], s=size, color=viscons.RQ3_QUADRANT_COLORS[q], edgecolor='white',
                         linewidth=0.4, zorder=3, label=f"{q} · {description}" if legend else None)
        axes.axvline(split_x, zorder=1, **viscons.RQ3_SPLIT_LINE)
        axes.axhline(split_y, zorder=1, **viscons.RQ3_SPLIT_LINE)
        if pareto:
            front = scores[tradeoff.pareto_front(scores)].sort_values('technical')
            axes.plot(front['technical'], front['social'], zorder=2, **viscons.RQ3_PARETO_LINE)
        if grid:
            axes.grid(color=viscons.COLOR_GRID, linewidth=0.5, zorder=0)
        else:
            axes.grid(False)
        axes.set_axisbelow(True)

    draw(ax, viscons.RQ3_INSET_OVERVIEW_MARKER_SIZE, legend=True)
    ax.set_xlim(0, float(scores['technical_max'].iloc[0]))
    ax.set_ylim(0, float(scores['social_max'].iloc[0]))
    ax.set_xlabel(f"Technical: {tradeoff.axis_label(technical)}", fontsize='small')
    ax.set_ylabel(f"Social: {tradeoff.axis_label(social)}", fontsize='small')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if len(scores) >= 3:    # bottom left: the inset takes the top left
        r, p = pearsonr(xs, ys)
        ax.text(0.012, 0.02, f"Pearson $\\rho$ = {r:.2f} ({_stars(p)})", transform=ax.transAxes, ha='left',
                va='bottom', fontsize='x-small', color=viscons.METRIC_ROW_LABEL_COLOR, zorder=5,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': 1.2})
    ax.set_title(title or "RQ3: socio-technical trade-off (split: half of the maximum; inset: zoom on the models)",
                 pad=16)   # PNG only (save_figure)
    fig.tight_layout()

    inset = ax.inset_axes(viscons.RQ3_INSET_BOUNDS)
    inset_colors = colors
    if color_by:
        if color_by not in summary.MODEL_ATTRIBUTES:
            raise ValueError(f"color_by must be one of {list(summary.MODEL_ATTRIBUTES)}, got {color_by!r}")
        name, levels, level_labels = summary.MODEL_ATTRIBUTES[color_by]
        palette = dict(zip(levels, viscons.RQ3_INSET_PALETTE))
        value_of = summary.load_model_metadata().set_index(cons.COL_MODEL)[color_by]
        inset_colors = [palette.get(value_of.get(m), viscons.COLOR_TEXT_SECONDARY) for m in scores[cons.COL_MODEL]]
        draw(inset, 0, legend=False)                                   # split lines etc.; the points below
        inset.scatter(xs, ys, s=viscons.RQ3_MARKER_SIZE, color=inset_colors, edgecolor='white', linewidth=0.4, zorder=3)
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker='o', linestyle='', color=palette[v], markeredgecolor='white', markersize=4,
                          label=l) for v, l in zip(levels, level_labels) if v in palette]
        inset.legend(handles=handles, title=name, loc='lower right', fontsize='xx-small', title_fontsize='xx-small',
                     frameon=True, framealpha=0.9, edgecolor='none', handletextpad=0.1, borderpad=0.4)
    else:
        draw(inset, viscons.RQ3_MARKER_SIZE, legend=False)
    for axis, values, (low_pad, high_pad) in ((inset.set_xlim, xs, viscons.RQ3_PAD_X),
                                              (inset.set_ylim, ys, viscons.RQ3_PAD_Y)):
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        span = high - low if high > low else 1.0
        axis(low - low_pad * span, high + high_pad * span)
    inset.set_facecolor('white')                 # opaque, in front: the overview's split lines must not show through
    inset.set_zorder(5)
    inset.tick_params(labelsize='xx-small', length=2, pad=1)
    for side in inset.spines.values():           # a full frame (the style hides the top / right spines)
        side.set_visible(True)
        side.set_edgecolor(viscons.RQ3_INSET_FRAME_COLOR)
        side.set_linewidth(0.6)
    from matplotlib.patches import ConnectionPatch, Rectangle   # the zoomed area: a light shaded frame
    (x0, x1), (y0, y1) = inset.get_xlim(), inset.get_ylim()
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=viscons.RQ3_INSET_FRAME_COLOR,
                           alpha=viscons.RQ3_INSET_SHADE_ALPHA, edgecolor='none', zorder=1))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=viscons.RQ3_INSET_FRAME_COLOR,
                           linewidth=0.6, zorder=2))
    labels = model_labels(scores[cons.COL_MODEL])
    for corner_y, box_y in ((1.0, y1), (0.0, y0)):  # connectors: right corners of the inset -> left corners of the frame
        ax.add_artist(ConnectionPatch(xyA=(1.0, corner_y), coordsA=inset.transAxes, xyB=(x0, box_y),
                                      coordsB=ax.transData, color=viscons.RQ3_INSET_FRAME_COLOR, linewidth=0.5,
                                      zorder=1))
    # quadrant legend under the frame, only the quadrants that occur (the absolute split: usually one)
    present = [q for q in tradeoff.QUADRANTS if (quadrant == q).any()]
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', linestyle='', color=viscons.RQ3_QUADRANT_COLORS[q], markeredgecolor='white',
                      markersize=4.5, label=f"{q} · {tradeoff.QUADRANTS[q]}") for q in present]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=((x0 + x1) / 2, y0), bbox_transform=ax.transData,
              ncol=len(handles), frameon=False, fontsize='x-small', handletextpad=0.1, columnspacing=1.0,
              borderaxespad=0.3)
    legends = [inset.get_legend()] if inset.get_legend() is not None else []
    _label_points(inset, [labels[m] for m in scores[cons.COL_MODEL]], xs, ys, inset_colors, (split_x, split_y),
                  legends, in_quadrant=False)
    return fig


def _stars(p: float) -> str:
    return 'ns' if not np.isfinite(p) or p >= 0.05 else '***' if p < 0.001 else '**' if p < 0.01 else '*'


def _label_points(ax: plt.Axes, labels: Sequence[str], xs: np.ndarray, ys: np.ndarray, colors: Sequence[str],
                  split: Tuple[float, float], obstacles: Sequence, in_quadrant: bool) -> None:
    """Place the labels (_place_labels) and draw them, with a leader line for the ones away from their point."""
    fig = ax.figure
    fig.canvas.draw()                                         # final extents of the texts, the axes, the obstacles
    renderer = fig.canvas.get_renderer()
    points = ax.transData.transform(np.column_stack([xs, ys]))
    dot = np.sqrt(viscons.RQ3_MARKER_SIZE) / 2 * fig.dpi / 72                   # marker radius in pixels
    probes = [ax.text(0, 0, label, fontsize=viscons.RQ3_LABEL_SIZE) for label in labels]
    sizes = [(b.width, b.height) for b in (t.get_window_extent(renderer) for t in probes)]
    for t in probes:
        t.remove()
    box = ax.get_window_extent(renderer)
    blocked = [tuple(o.get_window_extent(renderer).extents) for o in obstacles]
    regions = None
    if in_quadrant:            # quadrant boxes in pixels (a point on a split line belongs to the >= side)
        mx, my = ax.transData.transform(split)
        mx, my = min(max(mx, box.x0), box.x1), min(max(my, box.y0), box.y1)   # split outside the axes (zoomed inset)
        regions = [(mx if px >= mx else box.x0, my if py >= my else box.y0,
                    box.x1 if px >= mx else mx, box.y1 if py >= my else my) for px, py in points]
    centres = _place_labels(sizes, points, tuple(box.extents), blocked, dot, regions,
                            gap=viscons.RQ3_LABEL_GAP_PX * fig.dpi / 100)
    text_xy = ax.transData.inverted().transform(centres)
    for label, color, x, y, (tx, ty), (w, h), (cx, cy), (px, py) in zip(labels, colors, xs, ys, text_xy, sizes,
                                                                        centres, points):
        near = abs(cx - px) < w / 2 + 2.5 * dot and abs(cy - py) < h / 2 + 2.5 * dot   # touching its point
        ax.annotate(label, xy=(x, y), xytext=(tx, ty), fontsize=viscons.RQ3_LABEL_SIZE, color=color, ha='center',
                    va='center', zorder=4, annotation_clip=False,
                    arrowprops=None if near else dict(arrowstyle='-', color=color, alpha=0.6,
                                                      lw=viscons.RQ3_LEADER_WIDTH, shrinkA=1,
                                                      shrinkB=dot * 72 / fig.dpi + 0.5,
                                                      connectionstyle=f"arc3,rad={viscons.RQ3_LEADER_CURVE}"))


def _place_labels(sizes, points, axes_box, obstacles, dot_r, regions=None, gap=3.5, step=4.0, max_dist=400.0):
    """Greedy, deterministic label placement in pixel space (from the Submission2027 quadrant_scatter.py).

    Each label tries rings of growing distance around its point (16 directions, right / left / up / down first) and
    takes the first spot inside its region that clears every placed label, every point and the obstacles; crowded
    points go first so they get the spots closest to them. If the rings find nothing, the region is scanned on a grid
    (nearest first); a label that fits nowhere in its region tries the whole axes, else it stays on its point.
    Returns the label centres (n, 2) in pixels."""
    points = np.asarray(points, float)
    dots = [(x - dot_r, y - dot_r, x + dot_r, y + dot_r) for x, y in points]
    placed = list(obstacles)

    def hits(box, boxes):
        x0, y0, x1, y1 = box
        return any(x0 < b[2] + gap and b[0] - gap < x1 and y0 < b[3] + gap and b[1] - gap < y1 for b in boxes)

    def inside(box, region):
        return box[0] >= region[0] and box[2] <= region[2] and box[1] >= region[1] and box[3] <= region[3]

    angles = np.deg2rad([0, 180, 90, 270, 45, 135, 315, 225, 22.5, 157.5, 337.5, 202.5, 67.5, 112.5, 292.5, 247.5])
    directions = np.column_stack([np.cos(angles), np.sin(angles)])

    def find_spot(i, region):
        (w, h), (px, py) = sizes[i], points[i]
        others = dots[:i] + dots[i + 1:]
        for dist in np.arange(0, max_dist, step):
            for ux, uy in directions:      # the box EDGE (not its centre) sits `dist` beyond the point
                cx, cy = px + ux * (w / 2 + dot_r + gap + dist), py + uy * (h / 2 + dot_r + gap + dist)
                box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                if inside(box, region) and not hits(box, placed) and not hits(box, others):
                    return (cx, cy), box
        gx = np.arange(region[0] + w / 2, region[2] - w / 2 + 1e-9, step)
        gy = np.arange(region[1] + h / 2, region[3] - h / 2 + 1e-9, step)
        if len(gx) and len(gy):
            candidates = np.array(np.meshgrid(gx, gy)).reshape(2, -1).T
            for cx, cy in candidates[np.argsort(((candidates - (px, py)) ** 2).sum(1))]:
                box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                if not hits(box, placed) and not hits(box, others):
                    return (cx, cy), box
        return None

    crowding = (((points[:, None, :] - points[None, :, :]) ** 2).sum(-1) < (8 * dot_r) ** 2).sum(1)
    centres, boxes = points.copy(), [None] * len(points)
    for i in np.argsort(-crowding, kind='stable'):
        spot = find_spot(i, regions[i]) if regions is not None else None
        if spot is None:
            spot = find_spot(i, axes_box)
        if spot is not None:
            centres[i], boxes[i] = spot
            placed.append(boxes[i])

    # untangle: two leader lines that cross swap their label spots (centre on centre) when both labels still fit
    def box_at(i, c):
        (w, h) = sizes[i]
        return (c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2)

    def crosses(p1, p2, q1, q2):
        d = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return d(p1, p2, q1) * d(p1, p2, q2) < 0 and d(q1, q2, p1) * d(q1, q2, p2) < 0

    def fits(i, box, skip):
        region = regions[i] if regions is not None else axes_box
        rest = [b for k, b in enumerate(boxes) if b is not None and k not in skip] + list(obstacles)
        return (inside(box, region) or inside(box, axes_box)) and not hits(box, rest) \
            and not hits(box, [d for k, d in enumerate(dots) if k != i])

    for _ in range(3):
        swapped = False
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if boxes[i] is None or boxes[j] is None or not crosses(points[i], centres[i], points[j], centres[j]):
                    continue
                bi, bj = box_at(i, centres[j]), box_at(j, centres[i])
                if fits(i, bi, {i, j}) and fits(j, bj, {i, j}) and not hits(bi, [bj]):
                    centres[[i, j]] = centres[[j, i]]
                    boxes[i], boxes[j] = bi, bj
                    swapped = True
        if not swapped:
            break
    return centres


def rq3_figure_name(mode: str, technical: Optional[Sequence[str]] = None, social: Optional[Sequence[str]] = None) -> str:
    """rq3_<mode> for the default metric sets, else rq3_<mode>_<technical keys>_<social keys> (keys joined by '-')."""
    technical = list(technical or cons.RQ3_TECHNICAL_METRICS)
    social = list(social or cons.RQ3_SOCIAL_METRICS)
    if technical == list(cons.RQ3_TECHNICAL_METRICS) and social == list(cons.RQ3_SOCIAL_METRICS):
        return f"rq3_{mode}"
    return f"rq3_{mode}_{'-'.join(technical)}_{'-'.join(social)}"


# --- RQ2: local influence of each value ---------------------------------------------------------------------------
RQ2_VALUES = {'deviation': 'deviation from the grand mean', 'emm': 'estimated marginal mean',
              'contrast': 'difference from the reference value'}


def _level_label(level: str) -> str:
    return summary.PROMPT_LABELS.get(level, level.capitalize() if level.islower() else level)


def rq2_rows(results: pd.DataFrame, variables: Optional[Sequence[str]] = None) -> List[summary.RowGroup]:
    """One block per configuration variable (persona first), one row per value in the design order
    (cons.VARIABLE_VALUES; subfields grouped by field)."""
    from analysis.deviations import design_order
    present = results.groupby('variable')['level'].unique()
    order = [v for v in (variables or cons.CONFIG_VARIABLES) if v in present.index]
    groups = []
    for v in order:
        levels = design_order(v, present[v])
        groups.append(summary.RowGroup(summary.PROMPT_TITLES.get(v, v), v, levels, [_level_label(x) for x in levels],
                                       summary.PERSONA if v in cons.PERSONA_VARIABLES else summary.CONTEXT))
    return groups


def rq2_bars(results: pd.DataFrame, value: str = 'deviation', metrics: Optional[str | Sequence[str]] = None,
             variables: Optional[Sequence[str]] = None
             ) -> Tuple[pd.DataFrame, List[summary.RowGroup], List[summary.Metric]]:
    """RQ2 results as (table, rows, metrics) of plot_metric_bars; deviations on data-scaled signed panels."""
    if value not in RQ2_VALUES:
        raise ValueError(f"value must be one of {list(RQ2_VALUES)}")
    keys = rq1_metric_keys(results, metrics)
    rows = rq2_rows(results, variables)
    title_of = {g.column: g.title for g in rows}
    chosen = results[results['variable'].isin(title_of)]
    table = pd.DataFrame({'group': chosen['variable'].map(title_of), 'row': chosen['level'], 'metric': chosen['metric'],
                          'mean': chosen[value], 'ci_low': chosen[f"{value}_ci_low"],
                          'ci_high': chosen[f"{value}_ci_high"]})
    columns = [replace(summary.METRICS[k], limits=None) if value == 'deviation' else summary.METRICS[k] for k in keys]
    return table, rows, columns


def plot_rq2(results: pd.DataFrame, value: str = 'deviation', metrics: Optional[str | Sequence[str]] = None,
             variables: Optional[Sequence[str]] = None, title: Optional[str] = None) -> plt.Figure:
    table, rows, columns = rq2_bars(results, value, metrics, variables)
    return plot_metric_bars(table, rows, columns, title=title or f"RQ2: {RQ2_VALUES[value]} per value",
                            ci_linewidth=viscons.RQ_CI_LINEWIDTH, ci_capsize=viscons.RQ_CI_CAPSIZE,
                            tick_size=viscons.RQ_TICK_SIZE)


def _short_signed(v: float) -> str:
    """'+.03', '−.12', '+1.20' (no leading zero: narrow cells)."""
    if not np.isfinite(v):
        return '–'
    text = f"{v:+.2f}"
    if abs(v) < 1:
        text = text.replace('0.', '.', 1)
    return '0' if text in ('+.00', '-.00') else text.replace('-', '−')


def plot_rq2_heatmap(results: pd.DataFrame, metrics: Optional[str | Sequence[str]] = None,
                     variables: Optional[Sequence[str]] = None, title: Optional[str] = None,
                     width: Optional[float] = None) -> plt.Figure:
    """Heatmap of the RQ2 deviations (see module docstring), in the RQ1 heatmap frame: family bands, then the variable
    titles (bold, right-aligned) and their values (left-aligned)."""
    from matplotlib.patches import Patch
    keys = rq1_metric_keys(results, metrics)
    columns = [summary.METRICS[k] for k in keys]
    rows = rq2_rows(results, variables)
    index = results.set_index(['variable', 'level', 'metric'])
    better = LinearSegmentedColormap.from_list('better', ['#ffffff', viscons.RQ2_HEATMAP_BETTER])
    worse = LinearSegmentedColormap.from_list('worse', ['#ffffff', viscons.RQ2_HEATMAP_WORSE])
    neutral = LinearSegmentedColormap.from_list('neutral', ['#ffffff', viscons.RQ2_HEATMAP_NEUTRAL])

    def cell(variable, level, key, field):
        k = (variable, level, key)
        return float(index.at[k, field]) if k in index.index else np.nan

    scale = {m.key: np.nanmax([abs(cell(g.column, lv, m.key, 'deviation')) for g in rows for lv in g.levels]
                              + [viscons.RQ2_HEATMAP_MIN_SCALE]) for m in columns}
    families = list(dict.fromkeys(g.color_key for g in rows))
    blocks = [[g for g in rows if g.color_key == f] for f in families]
    gap_rows = 0.35                                        # space between variables inside a family block (rows)
    title_w = max(_text_size(g.title, 'small', 'bold')[0] for g in rows)
    level_w = max(_text_size(label, 'small')[0] for g in rows for label in g.labels)
    tgap = 0.08
    frame = HeatmapFrame(columns, [sum(len(g.levels) for g in b) + gap_rows * (len(b) - 1) for b in blocks],
                         title_w + tgap + level_w, right_in=0.1, width=width)
    ax, ux = frame.ax, frame.ux
    labels_left = -(frame.pad + title_w + tgap + level_w) * ux + (title_w + tgap) * ux
    for block, ys in zip(blocks, frame.ys):
        y = ys[0]
        for k, group in enumerate(block):
            if k:
                rule = y + gap_rows / 2
                ax.plot([-(frame.pad + title_w + tgap + level_w) * ux, frame.width_units], [rule, rule],
                        color=viscons.COLOR_GRID, linewidth=0.6, clip_on=False)
                y += gap_rows
            group_ys = [y + i for i in range(len(group.levels))]
            for level, label, yy in zip(group.levels, group.labels, group_ys):
                for metric, xx in zip(columns, frame.xs):
                    v = cell(group.column, level, metric.key, 'deviation')
                    lo, hi = (cell(group.column, level, metric.key, f"deviation_{s}") for s in ('ci_low', 'ci_high'))
                    if not np.isfinite(v):
                        color, strength = '#ffffff', 0.0
                    else:
                        strength = min(abs(v) / scale[metric.key], 1.0)
                        cmap = neutral if metric.direction == summary.NEUTRAL else \
                            better if v * metric.direction > 0 else worse
                        color = cmap(0.85 * strength)
                    frame.cell(xx, yy, color)
                    faded = not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 <= hi
                    text_color = viscons.RQ1_HEATMAP_FADED_TEXT if faded else ('white' if strength > 0.6 else '#1f1f1f')
                    frame.value(xx, yy, _short_signed(v), text_color)
                ax.text(labels_left, yy + 0.5, label, ha='left', va='center_baseline', fontsize='small', clip_on=False,
                        color=viscons.METRIC_ROW_LABEL_COLOR)
            ax.text(labels_left - tgap * ux, (group_ys[0] + group_ys[-1] + 1) / 2, group.title, ha='right',
                    va='center_baseline', fontweight='bold', fontsize='small', clip_on=False,
                    color=viscons.FAMILY_COLORS[group.color_key])
            y = group_ys[-1] + 1
    frame.bands([_FAMILY_TITLES[f] for f in families], families)
    frame.column_labels()
    frame.axis_titles('Value of the prompt variable', 'Evaluation metric')
    fig = frame.fig
    handles = [Patch(color=viscons.RQ2_HEATMAP_BETTER, label='better than the grand mean'),
               Patch(color=viscons.RQ2_HEATMAP_WORSE, label='worse'),
               Patch(color=viscons.RQ2_HEATMAP_NEUTRAL, label='|deviation| (no better direction)')]
    top = 1 - 0.02 / fig.get_figheight()
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(frame.left_in / fig.get_figwidth(), top), ncol=3,
               frameon=False, fontsize='x-small', handlelength=1.0, handletextpad=0.4, columnspacing=1.2)
    fig.text(1 - 0.1 / fig.get_figwidth(), top, f"colour scaled per metric (full at ≥ {viscons.RQ2_HEATMAP_MIN_SCALE:g})"
             " · faded: 95% CI includes 0", ha='right', va='bottom', fontsize=viscons.MIN_FONT_SIZE,
             color=viscons.COLOR_TEXT_SECONDARY)
    if title:
        fig.suptitle(title, y=1 + 0.28 / fig.get_figheight(), fontsize='small')
    return fig


def rq2_figure_name(kind: str, value: str = 'deviation', metrics: Optional[str | Sequence[str]] = None) -> str:
    """rq2_<value>_<metrics>_<kind> (see rq_figure_name; e.g. rq2_deviation_all_heatmap, rq2_emm_technical_bar)."""
    return rq_figure_name(2, kind, value, metrics)


def plot_rq2_forest(results: pd.DataFrame, value: str = 'contrast', metrics: Optional[str | Sequence[str]] = 'technical',
                    variables: Optional[Sequence[str]] = None, fit: bool = True,
                    title: Optional[str] = None) -> plt.Figure:
    """Forest plot of RQ2 (see module docstring): rows = a header per variable, then its values (the reference left
    out for contrasts), persona orange / context blue; fit = R^2 box in every panel."""
    from matplotlib.lines import Line2D
    if value not in ('contrast', 'deviation'):
        raise ValueError("value must be 'contrast' or 'deviation'")
    keys = rq1_metric_keys(results, metrics)
    columns = [summary.METRICS[k] for k in keys]
    groups = rq2_rows(results, variables)
    first = results.drop_duplicates(['variable', 'parent', 'level']).set_index(['variable', 'level'])

    lines = []                                  # (kind, text, family, variable, level)
    for g in groups:
        family = g.color_key
        header = g.title
        if value == 'contrast':
            refs = first.loc[g.column, 'reference'].unique()
            header += f" [ref: {_level_label(refs[0])}]" if g.column not in influence_nested() else ' [ref: first within field]'
        lines.append(('header', header, family, g.column, None))
        for level, label in zip(g.levels, g.labels):
            reference = first.at[(g.column, level), 'reference']
            if value == 'contrast' and level == reference:
                continue
            if value == 'contrast' and g.column in influence_nested():
                label += f" (vs. {_level_label(reference)})"
            lines.append(('row', label, family, g.column, level))
    n = len(lines)
    room = viscons.RQ2_FOREST_FIT_ROWS if fit and 'r2' in results else 0.0      # rows at the bottom for the fit text
    label_w = max(text_size(text, 'small', 'bold' if kind == 'header' else 'normal')[0] for kind, text, *_ in lines)
    top_in, bottom_in = 0.24, 0.34
    height = top_in + bottom_in + viscons.RQ2_FOREST_ROW_IN * (n + room)
    width = viscons.PAPER_TEXT_WIDTH_IN
    fig, axes = plt.subplots(1, len(columns), figsize=(width, height), sharey=True, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=(label_w + 0.08) / width, right=0.99, top=1 - top_in / height, bottom=bottom_in / height,
                        wspace=0.12)
    ys = np.arange(n)
    index = results.set_index(['variable', 'level', 'metric'])
    half = viscons.RQ2_FOREST_ROW_GAP / 2                 # white gap between shaded rows
    for ax, metric in zip(axes, columns):
        extent = 0.0
        for y, (kind, text, family, variable, level) in zip(ys, lines):
            if kind != 'row':
                continue
            ax.axhspan(y - 0.5 + half, y + 0.5 - half, color=viscons.FAMILY_LIGHT[family], linewidth=0, zorder=0)
            k = (variable, level, metric.key)
            if k not in index.index:
                continue
            v, lo, hi = (float(index.at[k, c]) for c in (value, f"{value}_ci_low", f"{value}_ci_high"))
            if not np.isfinite(v):
                continue
            err = [[max(v - lo, 0)], [max(hi - v, 0)]] if np.isfinite(lo) and np.isfinite(hi) else None
            extent = max(extent, abs(v), *(abs(x) for x in (lo, hi) if np.isfinite(x)))
            color = viscons.FAMILY_COLORS[family]
            ax.errorbar([v], [y], xerr=err, fmt='o', color=color, markersize=viscons.RQ2_FOREST_MARKER_SIZE,
                        markeredgecolor='white', markeredgewidth=0.3, elinewidth=viscons.METRIC_CI_LINEWIDTH,
                        capsize=viscons.METRIC_CI_CAPSIZE, capthick=viscons.METRIC_CI_LINEWIDTH, zorder=3)
        limit = _nice_limit(extent)
        ax.set_xlim(-limit, limit)
        ax.set_xticks([-limit, 0, limit])
        ax.set_xticklabels([_tick(-limit), '0', _tick(limit)])
        ticks = ax.get_xticklabels()
        ticks[0].set_horizontalalignment('left')                # end labels inwards: no clash between panels
        ticks[-1].set_horizontalalignment('right')
        ax.axvline(0, color=viscons.RQ2_FOREST_ZERO_COLOR, linewidth=viscons.RQ2_FOREST_ZERO_WIDTH, zorder=0.5)
        ax.set_title(metric.label, fontsize='small', fontweight='bold', pad=3)
        ax.tick_params(axis='x', labelsize='x-small', colors=viscons.COLOR_TEXT_SECONDARY, length=2, width=0.5, pad=1.5)
        ax.tick_params(axis='y', length=0)
        ax.grid(False)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        stats = results.loc[results['metric'] == metric.key, ['r2', 'r2_adj', 'delta_r2_model']] \
            if 'r2' in results else pd.DataFrame()
        if fit and len(stats):
            r = stats.iloc[0]
            ax.text(0.98, 0.01, f"$R^2$ {r['r2']:.2f} (adj. {r['r2_adj']:.2f})\n$\\Delta R^2_{{LLM}}$ {r['delta_r2_model']:.2f}",
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=viscons.MIN_FONT_SIZE, linespacing=1.05,
                    color=viscons.COLOR_TEXT_SECONDARY, zorder=4,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.8})
    axes[0].set_ylim(n - 0.5 + room, -0.5)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([text for _, text, *_ in lines], fontsize='small')
    for tick, (kind, _, family, *_) in zip(axes[0].get_yticklabels(), lines):
        tick.set_color(viscons.FAMILY_COLORS[family] if kind == 'header' else viscons.METRIC_ROW_LABEL_COLOR)
        if kind == 'header':
            tick.set_fontweight('bold')
    fig.supxlabel('Difference from the reference value' if value == 'contrast' else 'Deviation from the grand mean',
                  fontsize='x-small', color=viscons.COLOR_TEXT_SECONDARY, y=0.004,
                  x=(fig.subplotpars.left + fig.subplotpars.right) / 2)     # centred over the panels, not the figure
    handles = [Line2D([], [], marker='o', linestyle='', color=viscons.FAMILY_COLORS[f], label=f.capitalize())
               for f in (summary.PERSONA, summary.CONTEXT) if any(line[2] == f for line in lines)]
    fig.legend(handles=handles, loc='lower left', ncol=len(handles), frameon=False, fontsize='x-small',
               bbox_to_anchor=(0.005, 0.0), handletextpad=0.2, borderaxespad=0.1)
    if title:
        fig.suptitle(title, x=0.02, ha='left', y=1.0, fontsize='small')
    return fig


def _nice_limit(extent: float) -> float:
    """Symmetric axis end above the data: 1, 2, 2.5 or 5 x a power of ten (e.g. 0.004 -> 0.005, 0.07 -> 0.1)."""
    if not np.isfinite(extent) or extent <= 0:
        return 0.01
    power = 10 ** np.floor(np.log10(extent * 1.05))
    for step in (1, 2, 2.5, 5, 10):
        if step * power >= extent * 1.05:
            return float(step * power)
    return float(10 * power)


def _tick(value: float) -> str:
    """Compact tick label: no trailing zeros, true minus sign (e.g. −0.05, 0.005)."""
    return f"{value:.4f}".rstrip('0').rstrip('.').replace('-', '−')


def influence_nested() -> List[str]:
    from analysis.influence import NESTED
    return list(NESTED)
