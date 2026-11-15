"""
Figures of the research questions, built on the metric bar figure (viz/metric_plots.plot_metric_bars).

RQ1 (results of batch_rq1.py; measure = 'omega2' or 'shapley_r2'):
    plot_rq1          bars: rows = the model and the configuration variables (grouped model / persona / context, in
                      their colours), columns = metrics, bars with cluster-bootstrap CIs; bold = largest of its group
    plot_rq1_heatmap  heatmap: the same rows (persona, context, model), columns grouped TECHNICAL / SOCIAL with
                      subgroup brackets; one colour ramp per row family on a shared scale; a value not significant
                      after the correction (cons.HEATMAP_CORRECTION, default BY; one-sided) is faded; with Shapley R^2 a Residual row (1 - R^2): every column then sums to 1
    file names: viz/names.rq1_figure_name()

RQ2 (results of batch_rq2.py; notebook 4: raw technical + factuality as EMM bars, size-adjusted social as EMM dots):
    plot_rq2          bars of the estimated marginal mean (EMM): rows = the values of every configuration variable
                      (grouped by variable, persona orange / context blue), columns = metrics; raw metrics from 0
    plot_rq2_forest   forest plot: one dot + CI per value, one panel per metric; value 'emm' (size-adjusted: signed
                      axis, 0 = as a random recommender), 'contrast' (vs the reference value, named in the variable's
                      header; a subfield vs the reference of its field) or 'deviation'; R^2, adjusted R^2 and
                      Delta R^2 (LLM) in each panel
    plot_rq2_heatmap  heatmap of the deviations: green = better than the grand mean, red = worse (the metric's
                      direction), grey = metrics without a better direction; colour scaled per metric; faded = not
                      significant after the correction (cons.HEATMAP_CORRECTION, default BY); file names: viz/names.rq2_figure_name()

RQ3 (analysis/tradeoff.model_scores of the configuration table):
    plot_rq3          scatter technical vs social score per model, quadrants Q1-Q4 (relative: medians of the models;
                      absolute: middle of a raw axis, 0 = random recommender on an adjusted one; axes = attainable range); file name: viz/names.rq3_figure_name()
"""

from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

from analysis import multiple_testing as mt
from analysis import tradeoff
from analysis.deviations import design_order
from analysis.influence import NESTED
from metrics import summary
from utils import constants as cons
from viz import constants as viscons
from viz.figures import ci, clean_axes, family_cmap, model_attribute_colors, text_on, text_size
from viz import formatting
from viz.formatting import cell, nice_limit, short_signed
from viz.labels import FAMILY_TITLES, MEASURE_LABELS, fade_rule, level_label, model_labels, variable_title
from viz.heatmap import HeatmapFrame, deviation_color, deviation_legend_handles, family_colorbars
from viz.metric_plots import plot_metric_bars

RQ1_MEASURES = ('omega2', 'shapley_r2')            # labels: viz.labels.MEASURE_LABELS
RQ1_ROW_ORDER = (cons.FAMILY_PERSONA, cons.FAMILY_CONTEXT, cons.FAMILY_MODEL)      # prompt composition, then the LLM (bars and heatmap)


def load_rq(folder: Path | str, rq: str = 'rq1') -> pd.DataFrame:
    """All <metric>.parquet files of an RQ output folder (rq1 / rq2)."""
    files = sorted(Path(folder).glob('*.parquet'))
    if not files:
        raise FileNotFoundError(f"No {rq.upper()} results in {folder} (run run_metrics_pipeline.sh -s {rq})")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_rq1(folder: Path | str) -> pd.DataFrame:
    """RQ1 results with p-values (one-sided Wald, > 0) and their corrections over all variables x metrics:
    p_omega2[_holm|_bh|_by], p_shapley_r2[...] (analysis/multiple_testing.py)."""
    results = load_rq(folder, 'rq1')
    for measure in RQ1_MEASURES:
        results = mt.add_p_values(results, measure, 'greater')
    return results


def load_rq2(folder: Path | str) -> pd.DataFrame:
    """RQ2 results with p-values of the deviations (two-sided Wald) and their corrections over all values x metrics:
    p_deviation[_holm|_bh|_by]."""
    return mt.add_p_values(load_rq(folder, 'rq2'), 'deviation', 'two')


def rq1_rows(variables: Sequence[str], order: Sequence[str] = RQ1_ROW_ORDER) -> List[summary.RowGroup]:
    """One block per family (in `order`), one row per variable (in the given order)."""
    families = {f: [] for f in order}
    for v in variables:
        families[cons.variable_family(v)].append(v)
    return [summary.RowGroup(FAMILY_TITLES[f], 'variable', vs, [variable_title(v) for v in vs], f)
            for f, vs in families.items() if vs]


def metric_keys(results: pd.DataFrame, metrics: Optional[str | Sequence[str]] = None) -> List[str]:
    """Metric keys with results: all raw ones (None; the size-adjusted <key>_adj only when asked for, e.g. 'all_adj'),
    a family / subgroup ('technical', 'factuality', ...) or a list of keys."""
    available = [k for k in summary.METRICS if k in set(results['metric'])]
    if metrics is None:
        return [k for k in available if not k.endswith(cons.ADJ_SUFFIX)]
    wanted = [m.key for m in summary.select_metrics(metrics)]
    keys = [k for k in wanted if k in available]
    if not keys:
        raise ValueError(f"No results for the metrics {metrics!r} ({', '.join(wanted)}); results exist for: "
                         f"{', '.join(available)}. Size-adjusted (_adj) metrics need the expected + social steps, then "
                         f"the RQ step again (or set ADJUSTED = False in the notebook)")
    return keys


def rq1_bars(results: pd.DataFrame, measure: str = 'omega2', metrics: Optional[str | Sequence[str]] = None
             ) -> Tuple[pd.DataFrame, List[summary.RowGroup], List[summary.Metric]]:
    """RQ1 results as (table, rows, metrics) of plot_metric_bars; panels scaled to the data."""
    keys = metric_keys(results, metrics)
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
    return plot_metric_bars(table, rows, columns, title=title or f"RQ1: {MEASURE_LABELS[measure]} per metric", arrows=False,
                            tick_size=viscons.RQ_TICK_SIZE)


def plot_rq1_heatmap(results: pd.DataFrame, measure: str = 'shapley_r2', metrics: Optional[Sequence[str]] = None,
                     vmax: Optional[float] = None,
                     title: Optional[str] = None, width: Optional[float] = None,
                     fit_row: Optional[str] = 'auto', correction: Optional[str] = cons.HEATMAP_CORRECTION,
                     alpha: float = cons.HEATMAP_ALPHA) -> plt.Figure:
    """Heatmap of RQ1 (see module docstring). fit_row, the short row under the grid: 'residual' (1 - R^2: with
    Shapley R^2 every column then sums to 1), 'r2' (the model's R^2: partial omega^2 values are not shares of a total,
    so a residual would not complete them), 'auto' (residual for Shapley R^2, R^2 for omega^2) or None. The layout
    (HeatmapFrame) fits `width` (default viscons.HEATMAP_WIDTH_IN, the A4 text width) by narrowing the cells. Faded = not significant: adjusted p >= alpha
    (correction over ALL rows of `results`, i.e. every variable x metric loaded; None: the 95% CI includes 0)."""
    if fit_row == 'auto':
        fit_row = 'residual' if measure == 'shapley_r2' else 'r2'
    if fit_row not in (None, 'residual', 'r2'):
        raise ValueError(f"fit_row must be 'auto', 'residual', 'r2' or None, not {fit_row!r}")
    residual = fit_row is not None
    fit_label = {'residual': 'Residual', 'r2': '$R^2$'}.get(fit_row, '')
    keys = metric_keys(results, metrics)
    columns = [summary.METRICS[k] for k in keys]
    rows = rq1_rows(list(dict.fromkeys(results['variable'])))
    value = results.pivot(index='variable', columns='metric', values=measure)
    low = results.assign(_sig=mt.significant(results, measure, correction, alpha, 'greater')).pivot(
        index='variable', columns='metric', values='_sig')                  # True = significant
    r2 = results.groupby('metric')['r2_full'].first()
    vmax = vmax or max(0.05, float(np.nanmax(value[keys].clip(lower=0).to_numpy())))
    norm = Normalize(0, vmax)
    cmaps = {f: family_cmap(f) for f in viscons.FAMILY_COLORS}
    labels = [label for g in rows for label in g.labels]
    label_w = max(text_size(t, 'small', 'bold')[0] for t in labels + (['Residual'] if residual else []))
    frame = HeatmapFrame(columns, [len(g.levels) for g in rows], label_w, residual=residual, right_in=0.62,
                         width=width)
    ax, value_size = frame.ax, frame.value_size
    for group, ys in zip(rows, frame.ys):
        family = group.color_key
        for variable, label, yy in zip(group.levels, group.labels, ys):
            for metric, xx in zip(columns, frame.xs):
                v = value.at[variable, metric.key] if variable in value.index and metric.key in value.columns else np.nan
                sig = bool(low.at[variable, metric.key]) if variable in low.index and metric.key in low.columns else False
                frame.value_cell(xx, yy, v, sig, cmaps[family], norm)
            ax.text(-frame.pad * frame.ux, yy + 0.5, label, ha='right', va='center_baseline', fontweight='bold',
                    fontsize='small', clip_on=False, color=viscons.FAMILY_TEXT[family])
    if residual:
        y = frame.y_residual
        for metric, xx in zip(columns, frame.xs):
            r = r2.get(metric.key, np.nan)
            shown = (1 - r) if fit_row == 'residual' else r
            frame.residual(xx, cell(shown))
        ax.text(-frame.pad * frame.ux, y + frame.residual_h / 2, fit_label, ha='right', va='center_baseline',
                fontsize='small', clip_on=False, color=viscons.COLOR_TEXT_SECONDARY)
    frame.bands([g.title for g in rows], [g.color_key for g in rows])
    frame.column_labels()
    frame.axis_titles('Factor (source of variation)', 'Evaluation metric')
    family_colorbars(frame, cmaps, norm, rows, MEASURE_LABELS[measure])
    frame.note(f"faded: {fade_rule(correction, alpha)}")
    if title:
        frame.fig.suptitle(title, fontsize='small', y=1 + 0.2 / frame.fig.get_figheight())
    return frame.fig


# --- RQ3: socio-technical trade-off ------------------------------------------------------------------------------
def plot_rq3(scores: pd.DataFrame, mode: str = 'relative', pareto: bool = False,
             technical: Optional[Sequence[str]] = None, social: Optional[Sequence[str]] = None,
             title: Optional[str] = None, grid: bool = False, labels_in_quadrant: bool = viscons.RQ3_LABELS_IN_QUADRANT,
             ceilings: bool = False) -> plt.Figure:
    """Scatter of tradeoff.model_scores: x = technical score, y = social score, one labelled point per model coloured
    by its quadrant (its name too); Pearson correlation of the two scores in the corner. mode 'relative': split at the
    medians of the models, axes zoomed on the data; 'absolute': split at the middle of each axis' possible range,
    axes over the attainable range of the models (tradeoff.axis_range; raw: 0 to the number of metrics, split at the
    middle; size-adjusted: from the expected values, split at 0 = random recommender). pareto: dotted line through the
    Pareto front; grid: background grid (default off: only the split lines); ceilings (absolute only): each model's
    own social maximum (it differs through n), a dotted line from the point up to a short tick.

    Labels: deterministic greedy placement in pixels (_place_labels): each label takes the nearest free spot around
    its point, clear of every point, every placed label and the correlation box; labels_in_quadrant keeps it inside
    its point's quadrant when it fits. A label away from its point gets a thin, gently curved leader line.
    technical / social: the axis metrics (None: the RQ3 defaults at call time, tradeoff.axis_metrics)."""
    technical, social = tradeoff.axis_metrics(technical, social)

    split_x, split_y = tradeoff.splits(scores, mode)
    quadrant = tradeoff.quadrants(scores, mode)
    colors = list(quadrant.map(viscons.RQ3_QUADRANT_COLORS))
    xs, ys = scores['technical'].to_numpy(dtype=float), scores['social'].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=viscons.RQ3_FIGURE_IN)
    _draw_rq3(ax, scores, quadrant, (split_x, split_y), viscons.RQ3_MARKER_SIZE, legend=True, pareto=pareto, grid=grid)
    if mode == 'absolute':
        ax.set_xlim(*tradeoff.axis_range(scores, 'technical'))
        ax.set_ylim(*tradeoff.axis_range(scores, 'social'))
        if ceilings:
            _draw_ceilings(ax, scores, colors)
    else:   # room around the data for the labels of the outer points (labels stay inside the axes)
        _pad_limits(ax, xs, ys)
    corr = _corr_box(ax, xs, ys, top=True)
    _axis_titles(ax, technical, social)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize='x-small',
              handletextpad=0.1, columnspacing=1.2, borderaxespad=0.2)
    split = 'medians of the models' if mode == 'relative' else 'middle of the range / random recommender'
    ax.set_title(title or f"RQ3: socio-technical trade-off (split: {split})", pad=16)   # PNG only (save_figure)
    fig.tight_layout()   # final layout BEFORE placing: a later re-layout would move points under fixed-size labels
    labels = model_labels(scores[cons.COL_MODEL])
    _label_points(ax, [labels[m] for m in scores[cons.COL_MODEL]], xs, ys, colors, (split_x, split_y),
                  [corr] if corr is not None else [], labels_in_quadrant)
    return fig


def plot_rq3_inset(scores: pd.DataFrame, pareto: bool = False,
                   technical: Optional[Sequence[str]] = None, social: Optional[Sequence[str]] = None,
                   title: Optional[str] = None,
                   grid: bool = False, color_by: Optional[str] = None, ceilings: bool = False) -> plt.Figure:
    """The absolute view (axes over the attainable range, quadrants split as in plot_rq3; points without
    labels) with a zoomed inset on the models, where the labels are (placed as in plot_rq3). The inset sits in the
    empty part of the axes (viscons.RQ3_INSET_BOUNDS); a light shaded frame in the inset's colour marks the zoomed
    area, joined to the inset by two thin connectors, with the legend of the quadrants that occur under it. color_by: a model characteristic (summary.MODEL_ATTRIBUTES: 'access', 'reasoning', 'size', 'origin') for the
    points and labels of the inset, with their own legend (the overview keeps the quadrant colours); None: quadrants.
    ceilings: each model's own social maximum in the overview (as plot_rq3)."""
    technical, social = tradeoff.axis_metrics(technical, social)

    split_x, split_y = tradeoff.splits(scores, 'absolute')
    quadrant = tradeoff.quadrants(scores, 'absolute')
    colors = list(quadrant.map(viscons.RQ3_QUADRANT_COLORS))
    xs, ys = scores['technical'].to_numpy(dtype=float), scores['social'].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=viscons.RQ3_INSET_FIGURE_IN)

    def draw(axes, size, legend):
        _draw_rq3(axes, scores, quadrant, (split_x, split_y), size, legend=legend, pareto=pareto, grid=grid)

    draw(ax, viscons.RQ3_INSET_OVERVIEW_MARKER_SIZE, legend=True)
    ax.set_xlim(*tradeoff.axis_range(scores, 'technical'))
    ax.set_ylim(*tradeoff.axis_range(scores, 'social'))
    if ceilings:
        _draw_ceilings(ax, scores, colors)
    _axis_titles(ax, technical, social)
    _corr_box(ax, xs, ys, top=False)                  # bottom left: the inset takes the top left
    ax.set_title(title or "RQ3: socio-technical trade-off (absolute split; inset: zoom on the models)",
                 pad=16)   # PNG only (save_figure)
    fig.tight_layout()

    inset = ax.inset_axes(viscons.RQ3_INSET_BOUNDS)
    inset_colors = colors
    if color_by:
        if color_by not in summary.MODEL_ATTRIBUTES:
            raise ValueError(f"color_by must be one of {list(summary.MODEL_ATTRIBUTES)}, got {color_by!r}")
        name, levels, level_labels = summary.MODEL_ATTRIBUTES[color_by]
        palette = model_attribute_colors(levels)
        value_of = summary.load_model_metadata().set_index(cons.COL_MODEL)[color_by]
        inset_colors = [palette.get(value_of.get(m), viscons.COLOR_TEXT_SECONDARY) for m in scores[cons.COL_MODEL]]
        draw(inset, 0, legend=False)                                   # split lines etc.; the points below
        inset.scatter(xs, ys, s=viscons.RQ3_MARKER_SIZE, color=inset_colors, edgecolor='white', linewidth=0.4, zorder=3)
        handles = [Line2D([0], [0], marker='o', linestyle='', color=palette[v], markeredgecolor='white', markersize=4,
                          label=l) for v, l in zip(levels, level_labels) if v in palette]
        inset.legend(handles=handles, title=name, loc='lower right', fontsize='xx-small', title_fontsize='xx-small',
                     frameon=True, framealpha=0.9, edgecolor='none', handletextpad=0.1, borderpad=0.4)
    else:
        draw(inset, viscons.RQ3_MARKER_SIZE, legend=False)
    _pad_limits(inset, xs, ys)
    inset.set_facecolor('white')                 # opaque, in front: the overview's split lines must not show through
    inset.set_zorder(5)
    inset.tick_params(labelsize='xx-small', length=2, pad=1)
    for side in inset.spines.values():           # a full frame (the style hides the top / right spines)
        side.set_visible(True)
        side.set_edgecolor(viscons.RQ3_INSET_FRAME_COLOR)
        side.set_linewidth(0.6)
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
    handles = [Line2D([0], [0], marker='o', linestyle='', color=viscons.RQ3_QUADRANT_COLORS[q], markeredgecolor='white',
                      markersize=4.5, label=f"{q} · {tradeoff.QUADRANTS[q]}") for q in present]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=((x0 + x1) / 2, y0), bbox_transform=ax.transData,
              ncol=len(handles), frameon=False, fontsize='x-small', handletextpad=0.1, columnspacing=1.0,
              borderaxespad=0.3)
    legends = [inset.get_legend()] if inset.get_legend() is not None else []
    _label_points(inset, [labels[m] for m in scores[cons.COL_MODEL]], xs, ys, inset_colors, (split_x, split_y),
                  legends, in_quadrant=False)
    return fig


def _draw_rq3(ax: plt.Axes, scores: pd.DataFrame, quadrant: pd.Series, splits: tuple, size: float, legend: bool,
              pareto: bool, grid: bool) -> None:
    """The RQ3 scatter: points in their quadrant colours (size 0: only the rest), split lines, Pareto front, grid
    (default off: only the split lines)."""
    xs, ys = scores['technical'].to_numpy(dtype=float), scores['social'].to_numpy(dtype=float)
    for q, description in tradeoff.QUADRANTS.items():
        chosen = (quadrant == q).to_numpy()
        ax.scatter(xs[chosen], ys[chosen], s=size, color=viscons.RQ3_QUADRANT_COLORS[q], edgecolor='white',
                   linewidth=0.4, zorder=3, label=f"{q} · {description}" if legend else None)
    ax.axvline(splits[0], zorder=1, **viscons.RQ3_SPLIT_LINE)
    ax.axhline(splits[1], zorder=1, **viscons.RQ3_SPLIT_LINE)
    if pareto:
        front = scores[tradeoff.pareto_front(scores)].sort_values('technical')
        ax.plot(front['technical'], front['social'], zorder=2, **viscons.RQ3_PARETO_LINE)
    if grid:
        ax.grid(color=viscons.COLOR_GRID, linewidth=0.5, zorder=0)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)


def _axis_titles(ax: plt.Axes, technical: Sequence[str], social: Sequence[str]) -> None:
    """Axis titles (the summed metrics) and no top / right spines."""
    ax.set_xlabel(f"Technical: {tradeoff.axis_label(technical)}", fontsize='small')
    ax.set_ylabel(f"Social: {tradeoff.axis_label(social)}", fontsize='small')
    clean_axes(ax)


def _corr_box(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, top: bool):
    """Pearson r of the two scores (with stars) in the top / bottom left corner; None with < 3 models."""
    if len(xs) < 3:
        return None
    r, p = pearsonr(xs, ys)
    return ax.text(0.012, 0.975 if top else 0.02, f"Pearson $r$ = {r:.2f} ({_stars(p)})", transform=ax.transAxes,
                   ha='left', va='top' if top else 'bottom', fontsize='x-small', color=viscons.COLOR_TEXT, zorder=5,
                   bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': 1.2})


def _pad_limits(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray) -> None:
    """Axes zoomed on the points, with room for the labels of the outer points (RQ3_PAD_X / RQ3_PAD_Y)."""
    for set_limits, values, (low_pad, high_pad) in ((ax.set_xlim, xs, viscons.RQ3_PAD_X),
                                                    (ax.set_ylim, ys, viscons.RQ3_PAD_Y)):
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        span = high - low if high > low else 1.0
        set_limits(low - low_pad * span, high + high_pad * span)


def _draw_ceilings(ax: plt.Axes, scores: pd.DataFrame, colors: Sequence[str]) -> None:
    """Each model's own social maximum: a dotted line from its point up to a short tick (call after the limits)."""
    x0, x1 = ax.get_xlim()
    half = viscons.RQ3_CEILING_TICK * (x1 - x0)
    for x, y, top, color in zip(scores['technical'], scores['social'], scores['social_max'], colors):
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(top)):
            continue
        ax.plot([x, x], [y, top], color=color, zorder=2, **viscons.RQ3_CEILING_LINE)
        ax.plot([x - half, x + half], [top, top], color=color, zorder=2, linewidth=0.8)


def _stars(p: float) -> str:
    return 'ns' if not np.isfinite(p) or p >= cons.ALPHA else '***' if p < 0.001 else '**' if p < 0.01 else '*'


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


# --- RQ2: local influence of each value ---------------------------------------------------------------------------
RQ2_VALUES = {'deviation': 'deviation from the grand mean', 'emm': 'estimated marginal mean',
              'contrast': 'difference from the reference value'}


def rq2_rows(results: pd.DataFrame, variables: Optional[Sequence[str]] = None) -> List[summary.RowGroup]:
    """One block per configuration variable (persona first), one row per value in the design order
    (cons.VARIABLE_VALUES; subfields grouped by field)."""
    present = results.groupby('variable')['level'].unique()
    order = [v for v in (variables or cons.CONFIG_VARIABLES) if v in present.index]
    groups = []
    for v in order:
        levels = design_order(v, present[v])
        groups.append(summary.RowGroup(variable_title(v), v, levels, [level_label(v, x) for x in levels],
                                       cons.variable_family(v)))
    return groups


def rq2_bars(results: pd.DataFrame, value: str = 'deviation', metrics: Optional[str | Sequence[str]] = None,
             variables: Optional[Sequence[str]] = None
             ) -> Tuple[pd.DataFrame, List[summary.RowGroup], List[summary.Metric]]:
    """RQ2 results as (table, rows, metrics) of plot_metric_bars; deviations on data-scaled signed panels."""
    if value not in RQ2_VALUES:
        raise ValueError(f"value must be one of {list(RQ2_VALUES)}")
    keys = metric_keys(results, metrics)
    rows = rq2_rows(results, variables)
    title_of = {g.column: g.title for g in rows}
    chosen = results[results['variable'].isin(title_of)]
    table = pd.DataFrame({'group': chosen['variable'].map(title_of), 'row': chosen['level'], 'metric': chosen['metric'],
                          'mean': chosen[value], 'ci_low': chosen[f"{value}_ci_low"],
                          'ci_high': chosen[f"{value}_ci_high"]})
    columns = [replace(summary.METRICS[k], limits=None) if value == 'deviation' else summary.METRICS[k] for k in keys]
    return table, rows, columns


def plot_rq2(results: pd.DataFrame, metrics: Optional[str | Sequence[str]] = None,
             variables: Optional[Sequence[str]] = None, title: Optional[str] = None) -> plt.Figure:
    """Bars of the estimated marginal mean (EMM) per value: the LEVEL of the metric (differences: plot_rq2_forest).
    Raw metrics on their scale; size-adjusted ones signed around 0 = as a random recommender of the same n."""
    table, rows, columns = rq2_bars(results, 'emm', metrics, variables)
    return plot_metric_bars(table, rows, columns, title=title or f"RQ2: {RQ2_VALUES['emm']} per value",
                            tick_size=viscons.RQ_TICK_SIZE)


def plot_rq2_heatmap(results: pd.DataFrame, metrics: Optional[str | Sequence[str]] = None,
                     variables: Optional[Sequence[str]] = None, title: Optional[str] = None,
                     width: Optional[float] = None, correction: Optional[str] = cons.HEATMAP_CORRECTION,
                     alpha: float = cons.HEATMAP_ALPHA) -> plt.Figure:
    """Heatmap of the RQ2 deviations (see module docstring), in the RQ1 heatmap frame: family bands, then the variable
    titles (bold, right-aligned) and their values (left-aligned). Faded = not significant: adjusted p >= alpha
    (two-sided; correction over ALL rows of `results`; None: the 95% CI includes 0)."""
    keys = metric_keys(results, metrics)
    columns = [summary.METRICS[k] for k in keys]
    rows = rq2_rows(results, variables)
    index = results.assign(_sig=mt.significant(results, 'deviation', correction, alpha, 'two').astype(float)) \
        .set_index(['variable', 'level', 'metric'])

    def cell(variable, level, key, field):
        k = (variable, level, key)
        return float(index.at[k, field]) if k in index.index else np.nan

    scale = {m.key: np.nanmax([abs(cell(g.column, lv, m.key, 'deviation')) for g in rows for lv in g.levels]
                              + [viscons.RQ2_HEATMAP_MIN_SCALE]) for m in columns}
    families = list(dict.fromkeys(g.color_key for g in rows))
    blocks = [[g for g in rows if g.color_key == f] for f in families]
    gap_rows = 0.35                                        # space between variables inside a family block (rows)
    title_w = max(text_size(g.title, 'small', 'bold')[0] for g in rows)
    level_w = max(text_size(label, 'small')[0] for g in rows for label in g.labels)
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
                    sig = cell(group.column, level, metric.key, '_sig') == 1
                    color = deviation_color(v, metric.direction, scale[metric.key])
                    frame.cell(xx, yy, color)
                    text_color = viscons.RQ1_HEATMAP_FADED_TEXT if not sig else text_on(color)
                    frame.value(xx, yy, short_signed(v), text_color)
                ax.text(labels_left, yy + 0.5, label, ha='left', va='center_baseline', fontsize='small', clip_on=False,
                        color=viscons.COLOR_TEXT)
            ax.text(labels_left - tgap * ux, (group_ys[0] + group_ys[-1] + 1) / 2, group.title, ha='right',
                    va='center_baseline', fontweight='bold', fontsize='small', clip_on=False,
                    color=viscons.FAMILY_COLORS[group.color_key])
            y = group_ys[-1] + 1
    frame.bands([FAMILY_TITLES[f] for f in families], families)
    frame.column_labels()
    frame.axis_titles('Value of the prompt variable', 'Evaluation metric')
    fig = frame.fig
    handles = deviation_legend_handles('the grand mean')
    top = 1 - 0.02 / fig.get_figheight()
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(frame.left_in / fig.get_figwidth(), top), ncol=3,
               frameon=False, fontsize='x-small', handlelength=1.0, handletextpad=0.4, columnspacing=1.2)
    frame.note(f"colour scaled per metric (full at ≥ {viscons.RQ2_HEATMAP_MIN_SCALE:g}) · faded: "
               f"{fade_rule(correction, alpha)}")
    if title:
        fig.suptitle(title, y=1 + 0.28 / fig.get_figheight(), fontsize='small')
    return fig


def rq2_forest_xlabel(value: str, adjusted: bool = False) -> str:
    """x label of the RQ2 forest; size-adjusted metrics: what the difference is made of (obs. = observed, exp. =
    expected under a random recommender)."""
    if value == 'emm':
        return 'Estimated marginal mean of obs. − exp. (0 = as random)' if adjusted else 'Estimated marginal mean'
    if value == 'contrast':
        return 'Difference in (obs. − exp.) from the reference value' if adjusted else \
            'Difference from the reference value'
    return 'Deviation of (obs. − exp.) from the grand mean' if adjusted else 'Deviation from the grand mean'


def plot_rq2_forest(results: pd.DataFrame, value: str = 'contrast', metrics: Optional[str | Sequence[str]] = 'technical',
                    variables: Optional[Sequence[str]] = None, fit: bool = True,
                    title: Optional[str] = None, marks: str = 'dots', nest_subfields: bool = True) -> plt.Figure:
    """Forest plot of RQ2 (see module docstring): rows = a header per variable, then its values (the reference left
    out for contrasts), persona orange / context blue; fit = R^2 box in every panel. value 'emm': size-adjusted
    metrics on a signed axis with 0 = as a random recommender; raw metrics on their own scale (no 0 line).
    marks: 'dots' (dot + CI on shaded rows) | 'bars' (bar from 0 + CI, no shading; e.g. raw EMM).
    nest_subfields: each field's subfields right under it (smaller grey label, lighter row and mark; the field's EMM
    averages them), no separate Subfield block; False = Subfield as its own block."""
    if value not in ('contrast', 'deviation', 'emm'):
        raise ValueError("value must be 'contrast', 'deviation' or 'emm'")
    if marks not in ('dots', 'bars'):
        raise ValueError("marks must be 'dots' or 'bars'")
    keys = metric_keys(results, metrics)
    columns = [summary.METRICS[k] for k in keys]
    groups = rq2_rows(results, variables)
    first = results.drop_duplicates(['variable', 'parent', 'level']).set_index(['variable', 'level'])

    nested = {}                                 # field level -> [(subfield level, label)] when nested
    subs = next((g for g in groups if g.column == cons.VAR_SUBFIELD), None)
    if nest_subfields and subs is not None and any(g.column == cons.VAR_FIELD for g in groups):
        groups = [g for g in groups if g is not subs]
        for level, label in zip(subs.levels, subs.labels):
            nested.setdefault(first.at[(cons.VAR_SUBFIELD, level), 'parent'], []).append((level, label))

    def value_line(kind, variable, level, label, family):
        reference = first.at[(variable, level), 'reference']
        if value == 'contrast' and level == reference:
            return None
        if value == 'contrast' and variable in influence_nested():
            label += f" (vs. {level_label(variable, reference)})"
        return (kind, label, family, variable, level)

    lines = []                                  # (kind 'header' | 'row' | 'sub', text, family, variable, level)
    for g in groups:
        family = g.color_key
        header = g.title
        if value == 'contrast':
            refs = first.loc[g.column, 'reference'].unique()
            header += f" [ref: {level_label(g.column, refs[0])}]" if g.column not in influence_nested() else ' [ref: first within field]'
        lines.append(('header', header, family, g.column, None))
        for level, label in zip(g.levels, g.labels):
            lines.append(value_line('row', g.column, level, label, family))
            for sub, sub_label in nested.get(level, []) if g.column == cons.VAR_FIELD else []:
                lines.append(value_line('sub', cons.VAR_SUBFIELD, sub, sub_label, subs.color_key))
    lines = [line for line in lines if line is not None]
    n = len(lines)
    label_size = {'header': 'small', 'row': 'small', 'sub': viscons.RQ2_FOREST_SUB_LABEL_SIZE}
    room = viscons.RQ2_FOREST_FIT_ROWS if fit and 'r2' in results else 0.0      # rows at the bottom for the fit text
    label_w = max(text_size(text, label_size[kind], 'bold' if kind == 'header' else 'normal')[0]
                  for kind, text, *_ in lines)
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
            if kind == 'header':
                continue
            sub = kind == 'sub'
            if marks == 'dots':
                shade = viscons.FAMILY_LIGHT[family]
                shade = _lighter(shade, viscons.RQ2_FOREST_SUB_SHADE_LIGHTEN) if sub else shade
                ax.axhspan(y - 0.5 + half, y + 0.5 - half, color=shade, linewidth=0, zorder=0)
            k = (variable, level, metric.key)
            if k not in index.index:
                continue
            v, lo, hi = (float(index.at[k, c]) for c in (value, f"{value}_ci_low", f"{value}_ci_high"))
            if not np.isfinite(v):
                continue
            extent = max(extent, abs(v), *(abs(x) for x in (lo, hi) if np.isfinite(x)))
            color = viscons.FAMILY_COLORS[family]
            color = _lighter(color, viscons.RQ2_FOREST_SUB_MARK_LIGHTEN) if sub else color
            scale = viscons.RQ2_FOREST_SUB_BAR_SCALE if sub else 1.0          # bar height, CI caps
            if marks == 'bars':
                ax.barh(y, v, height=viscons.RQ2_FOREST_BAR_HEIGHT * scale, color=color, linewidth=0, zorder=2)
                ci(ax, [v], [y], [lo], [hi], scale=scale, zorder=3)
            else:          # CI in the dot's colour, dot drawn apart (errorbar's markeredgewidth would also thin the caps)
                ci(ax, [v], [y], [lo], [hi], color=color, scale=scale, zorder=3)
                ax.plot([v], [y], 'o', color=color, markeredgewidth=0, zorder=4,
                        markersize=viscons.RQ2_FOREST_MARKER_SIZE * (viscons.RQ2_FOREST_SUB_MARK_SCALE if sub else 1))
        signed = value != 'emm' or metric.key.endswith(cons.ADJ_SUFFIX)     # 0 = reference / mean / random
        if signed:
            limit = nice_limit(extent)
            low, high = -limit, limit
        else:                                                   # raw EMM: the metric's scale
            low, high = metric.limits or (0.0, nice_limit(extent))
        ax.set_xlim(low, high)
        ax.set_xticks([low, (low + high) / 2, high])
        ax.set_xticklabels([formatting.tick(low), formatting.tick((low + high) / 2), formatting.tick(high)])
        ticks = ax.get_xticklabels()
        ticks[0].set_horizontalalignment('left')                # end labels inwards: no clash between panels
        ticks[-1].set_horizontalalignment('right')
        if signed:
            ax.axvline(0, color=viscons.RQ2_FOREST_ZERO_COLOR, linewidth=viscons.RQ2_FOREST_ZERO_WIDTH, zorder=0.5)
        ax.set_title(metric.label, fontsize='small', fontweight='bold', pad=3)
        ax.tick_params(axis='x', labelsize='x-small', colors=viscons.COLOR_TEXT_SECONDARY, length=2, width=0.5, pad=1.5)
        ax.tick_params(axis='y', length=0)
        ax.grid(False)
        clean_axes(ax, hide=('top', 'right', 'left'))
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
    label_color = {'row': viscons.COLOR_TEXT, 'sub': viscons.RQ2_FOREST_SUB_LABEL_COLOR}
    for tick, (kind, _, family, *_) in zip(axes[0].get_yticklabels(), lines):
        tick.set_fontsize(label_size[kind])
        tick.set_color(viscons.FAMILY_COLORS[family] if kind == 'header' else label_color[kind])
        if kind == 'header':
            tick.set_fontweight('bold')
    adjusted = any(m.key.endswith(cons.ADJ_SUFFIX) for m in columns)
    center = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2       # over the panels, not the figure
    fig.supxlabel(rq2_forest_xlabel(value, adjusted), x=center, fontsize='x-small',
                  color=viscons.COLOR_TEXT_SECONDARY, y=0.004)
    handles = [Line2D([], [], marker='o' if marks == 'dots' else 's', linestyle='', color=viscons.FAMILY_COLORS[f],
                      label=FAMILY_TITLES[f])
               for f in (cons.FAMILY_PERSONA, cons.FAMILY_CONTEXT) if any(line[2] == f for line in lines)]
    fig.legend(handles=handles, loc='lower left', ncol=len(handles), frameon=False, fontsize='x-small',
               bbox_to_anchor=(0.005, 0.0), handletextpad=0.2, borderaxespad=0.1)
    if title:
        fig.suptitle(title, x=0.02, ha='left', y=1.0, fontsize='small')
    return fig


def _lighter(color, amount: float) -> Tuple[float, float, float]:
    """``color`` blended ``amount`` (0-1) towards white."""
    return tuple(c + (1 - c) * amount for c in to_rgb(color))


def influence_nested() -> List[str]:
    return list(NESTED)
