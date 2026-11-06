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
    """Metric keys with results: all (None), a family / subgroup ('technical', 'factuality', ...) or a list of keys."""
    available = [k for k in summary.METRICS if k in set(results['metric'])]
    if metrics is None:
        return available
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
    return plot_metric_bars(table, rows, columns, title=title or f"RQ1: {RQ1_MEASURES[measure]} per metric", arrows=False)


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
                     title: Optional[str] = None) -> plt.Figure:
    """Heatmap of RQ1 (see module docstring). residual: default = only for Shapley R^2 (a decomposition).
    Compact layout: every margin is sized from the measured text (row labels, rotated column labels)."""
    residual = (measure == 'shapley_r2') if residual is None else residual
    keys = rq1_metric_keys(results, metrics)
    columns = [summary.METRICS[k] for k in keys]
    rows = rq1_rows(list(dict.fromkeys(results['variable'])))
    value = results.pivot(index='variable', columns='metric', values=measure)
    low = results.pivot(index='variable', columns='metric', values=f"{measure}_ci_low")
    r2 = results.groupby('metric')['r2_full'].first()
    vmax = vmax or max(0.05, float(np.nanmax(value[keys].clip(lower=0).to_numpy())))
    norm = Normalize(0, vmax)
    cmaps = {f: LinearSegmentedColormap.from_list(f, ['#ffffff', viscons.METRIC_BAR_COLORS[f], viscons.RQ1_HEATMAP_DARK[f]])
             for f in viscons.RQ1_HEATMAP_DARK}
    cell_w, cell_h = viscons.RQ1_HEATMAP_CELL_IN

    # grid in cell units: columns (a gap between bands), rows (a gap between families, the residual row below)
    xs, x = [], 0.0
    for i, metric in enumerate(columns):
        x += viscons.RQ1_HEATMAP_BAND_GAP if i and metric.band != columns[i - 1].band else 0
        xs.append(x)
        x += 1
    width_units = x
    ys, cells, y = [], [], 0.0
    for i, group in enumerate(rows):
        y += viscons.RQ1_HEATMAP_ROW_GAP if i else 0
        for variable, label in zip(group.levels, group.labels):
            ys.append(y)
            cells.append((variable, label, group.color_key))
            y += 1
    y_residual = y + viscons.RQ1_HEATMAP_ROW_GAP if residual else None
    height_units = y_residual + 1 if residual else y

    # margins (inches) from the measured text
    label_w = max(_text_size(label, 'small', 'bold')[0] for _, label, _ in cells + ([('', 'Residual', '')] if residual else []))
    column_h = max(_rotated_height(m.short or m.key, 'small', 40) for m in columns)
    has_brackets = any(columns[i].group == columns[i + 1].group for i in range(len(columns) - 1))
    pad, band_w = 0.07, 0.16
    left_in = band_w + 2 * pad + label_w
    bottom_in = pad + column_h + (0.22 if has_brackets else 0) + 0.2 + pad
    right_in = 0.7
    fig_w, fig_h = left_in + cell_w * width_units + right_in, 0.08 + cell_h * height_units + bottom_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left_in / fig_w, bottom_in / fig_h, cell_w * width_units / fig_w, cell_h * height_units / fig_h])
    ax.set_xlim(0, width_units)
    ax.set_ylim(height_units, 0)
    ax.axis('off')
    ux, uy = 1 / cell_w, 1 / cell_h                     # inches -> cell units

    for (variable, label, family), yy in zip(cells, ys):
        for metric, xx in zip(columns, xs):
            v = value.at[variable, metric.key] if variable in value.index and metric.key in value.columns else np.nan
            ci_low = low.at[variable, metric.key] if variable in low.index and metric.key in low.columns else np.nan
            shown = 0.0 if np.isfinite(v) and -0.005 < v < 0 else v
            color = cmaps[family](norm(max(shown, 0))) if np.isfinite(shown) else '#ffffff'
            ax.add_patch(Rectangle((xx, yy), 1, 1, facecolor=color, edgecolor='white', linewidth=0.6))
            faded = not np.isfinite(ci_low) or ci_low <= 0
            text_color = viscons.RQ1_HEATMAP_FADED_TEXT if faded else ('white' if norm(shown) > 0.6 else '#1f1f1f')
            ax.text(xx + 0.5, yy + 0.5, '–' if not np.isfinite(shown) else f"{shown:.2f}", ha='center', va='center',
                    fontsize='x-small', color=text_color)
        ax.text(-pad * ux, yy + 0.5, label, ha='right', va='center', fontweight='bold', fontsize='small', clip_on=False,
                color=viscons.RQ1_HEATMAP_DARK[family] if family != 'model' else '#7a7a7a')
    if residual:
        for metric, xx in zip(columns, xs):
            r = r2.get(metric.key, np.nan)
            ax.add_patch(Rectangle((xx, y_residual), 1, 1, facecolor='white', edgecolor='#d0d0d0', linewidth=0.5))
            ax.text(xx + 0.5, y_residual + 0.5, f"{1 - r:.2f}" if np.isfinite(r) else '–', ha='center', va='center',
                    fontsize='x-small', color=viscons.COLOR_TEXT_SECONDARY)
        ax.text(-pad * ux, y_residual + 0.5, 'Residual', ha='right', va='center', fontsize='small', clip_on=False,
                color=viscons.COLOR_TEXT_SECONDARY)

    # family bands, left of the row labels
    band_x = -(left_in - pad) * ux
    start = 0
    for group in rows:
        top, bottom = ys[start], ys[start + len(group.levels) - 1] + 1
        start += len(group.levels)
        ax.add_patch(Rectangle((band_x, top), band_w * ux, bottom - top, clip_on=False,
                               facecolor=viscons.METRIC_BAR_COLORS[group.color_key]))
        ax.text(band_x + band_w * ux / 2, (top + bottom) / 2, group.title.upper(), rotation=90, ha='center',
                va='center', color='white', fontweight='bold', fontsize='x-small', clip_on=False)

    # column labels, subgroup brackets, TECHNICAL / SOCIAL bands, right under the grid
    y0 = height_units + pad * uy
    for metric, xx in zip(columns, xs):
        ax.text(xx + 0.5, y0, metric.short or metric.key, rotation=40, ha='right', va='top', fontsize='small',
                rotation_mode='anchor', clip_on=False)
    y_bracket = y0 + column_h * uy + 0.05 * uy
    if has_brackets:
        _brackets(ax, columns, xs, y_bracket)
        y_bracket += 0.2 * uy
    for band in dict.fromkeys(m.band for m in columns):
        bx = [xx for m, xx in zip(columns, xs) if m.band == band]
        ax.add_patch(Rectangle((bx[0], y_bracket), bx[-1] + 1 - bx[0], 0.2 * uy, facecolor=viscons.RQ1_HEATMAP_BAND,
                               edgecolor='none', clip_on=False))
        ax.text((bx[0] + bx[-1] + 1) / 2, y_bracket + 0.1 * uy, band.upper(), ha='center', va='center',
                fontweight='bold', fontsize='x-small', clip_on=False)
    _colorbars(fig, ax, cmaps, norm, rows, ys, RQ1_MEASURES[measure])
    if title:
        ax.set_title(title, fontsize='small')
    return fig


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
            ax.text((x0 + x1) / 2, y + 0.18, columns[i].group.capitalize(), ha='center', va='top', fontsize='x-small',
                    color=viscons.COLOR_TEXT_SECONDARY, clip_on=False)
        i = j + 1


def _colorbars(fig: plt.Figure, ax: plt.Axes, cmaps, norm: Normalize, rows: Sequence[summary.RowGroup],
               ys: Sequence[float], label: str) -> None:
    """One colour bar per row family, next to its rows, sharing the scale."""
    fig.canvas.draw()
    to_fig = (ax.transData + fig.transFigure.inverted()).transform
    right = ax.get_position().x1 + 0.012
    start = 0
    for i, group in enumerate(rows):
        top, bottom = ys[start], ys[start + len(group.levels) - 1] + 1
        start += len(group.levels)
        (_, y_top), (_, y_bottom) = to_fig((0, top)), to_fig((0, bottom))
        cax = fig.add_axes([right, y_bottom, 0.012, y_top - y_bottom])
        family = group.color_key
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmaps[family]), cax=cax)
        bar.ax.tick_params(labelsize='xx-small', length=2)
        bar.outline.set_linewidth(0.4)
        if i == 0:
            bar.set_label(label, fontsize='x-small')


def rq1_figure_name(kind: str, measure: str, metrics: Optional[str | Sequence[str]] = None) -> str:
    """File name of an RQ1 figure: rq1_<metrics>_<measure>_<kind>, metrics = 'all' / a family / keys joined by '-'
    (e.g. rq1_technical_shapley_r2_heatmap, rq1_all_omega2_bars)."""
    selection = 'all' if metrics is None else metrics if isinstance(metrics, str) else '-'.join(metrics)
    return f"rq1_{selection}_{measure}_{kind}"


# --- RQ3: socio-technical trade-off ------------------------------------------------------------------------------
def plot_rq3(scores: pd.DataFrame, mode: str = 'relative', pareto: bool = False,
             technical: Sequence[str] = tuple(cons.RQ3_TECHNICAL_METRICS),
             social: Sequence[str] = tuple(cons.RQ3_SOCIAL_METRICS), title: Optional[str] = None) -> plt.Figure:
    """Scatter of tradeoff.model_scores: x = technical score, y = social score, one labelled point per model coloured
    by its quadrant. mode 'relative': split at the medians of the models, axes zoomed on the data; 'absolute': split
    at half of each axis maximum, axes from 0 to the maximum. pareto: dotted line through the Pareto front."""
    from adjustText import adjust_text
    from analysis import tradeoff
    from viz.labels import model_labels

    split_x, split_y = tradeoff.splits(scores, mode)
    quadrant = tradeoff.quadrants(scores, mode)
    fig, ax = plt.subplots(figsize=viscons.RQ3_FIGURE_IN)
    for q, description in tradeoff.QUADRANTS.items():
        chosen = quadrant == q
        ax.scatter(scores.loc[chosen, 'technical'], scores.loc[chosen, 'social'], s=viscons.RQ3_MARKER_SIZE,
                   color=viscons.RQ3_QUADRANT_COLORS[q], edgecolor='white', linewidth=0.5, zorder=3,
                   label=f"{q} · {description}")
    ax.axvline(split_x, zorder=1, **viscons.RQ3_SPLIT_LINE)
    ax.axhline(split_y, zorder=1, **viscons.RQ3_SPLIT_LINE)
    if pareto:
        front = scores[tradeoff.pareto_front(scores)].sort_values('technical')
        ax.plot(front['technical'], front['social'], zorder=2, **viscons.RQ3_PARETO_LINE)

    if mode == 'absolute':
        ax.set_xlim(0, float(scores['technical_max'].iloc[0]))
        ax.set_ylim(0, float(scores['social_max'].iloc[0]))
    else:
        for axis, column in ((ax.set_xlim, 'technical'), (ax.set_ylim, 'social')):
            low, high = scores[column].min(), scores[column].max()
            pad = 0.08 * (high - low) if high > low else 0.1
            axis(low - pad, high + pad)
    labels = model_labels(scores[cons.COL_MODEL])
    texts = [ax.text(x, y, labels[m], fontsize=viscons.VALUE_FONT_SIZE, color=viscons.COLOR_TEXT_SECONDARY, zorder=4)
             for m, x, y in scores[[cons.COL_MODEL, 'technical', 'social']].itertuples(index=False)]
    if texts:
        adjust_text(texts, x=scores['technical'].to_numpy(), y=scores['social'].to_numpy(), ax=ax,
                    arrowprops={'arrowstyle': '-', 'color': viscons.COLOR_TEXT_SECONDARY, 'linewidth': 0.4})

    n_t, n_s = len(technical), len(social)
    ax.set_xlabel(f"Technical score (max {n_t}): {tradeoff.axis_label(technical)}")
    ax.set_ylabel(f"Social score (max {n_s}): {tradeoff.axis_label(social)}")
    ax.grid(color=viscons.COLOR_GRID, linewidth=0.5, zorder=0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize='x-small',
              handletextpad=0.2, columnspacing=1.0)
    split = 'medians of the models' if mode == 'relative' else 'half of the maximum'
    ax.set_title(title or f"RQ3: socio-technical trade-off (split: {split})")
    fig.tight_layout()
    return fig


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
    return plot_metric_bars(table, rows, columns, title=title or f"RQ2: {RQ2_VALUES[value]} per value")


def _short_signed(v: float) -> str:
    """'+.03', '−.12', '+1.20' (no leading zero: narrow cells)."""
    if not np.isfinite(v):
        return '–'
    text = f"{v:+.2f}"
    if abs(v) < 1:
        text = text.replace('0.', '.', 1)
    return '0' if text in ('+.00', '-.00') else text.replace('-', '−')


def plot_rq2_heatmap(results: pd.DataFrame, metrics: Optional[str | Sequence[str]] = None,
                     variables: Optional[Sequence[str]] = None, title: Optional[str] = None) -> plt.Figure:
    """Heatmap of the RQ2 deviations (see module docstring); compact layout sized from the measured text."""
    from matplotlib.patches import Patch
    keys = rq1_metric_keys(results, metrics)
    columns = [summary.METRICS[k] for k in keys]
    rows = rq2_rows(results, variables)
    index = results.set_index(['variable', 'level', 'metric'])
    cell_w, cell_h = viscons.RQ2_HEATMAP_CELL_IN
    better = LinearSegmentedColormap.from_list('better', ['#ffffff', viscons.RQ2_HEATMAP_BETTER])
    worse = LinearSegmentedColormap.from_list('worse', ['#ffffff', viscons.RQ2_HEATMAP_WORSE])
    neutral = LinearSegmentedColormap.from_list('neutral', ['#ffffff', viscons.RQ2_HEATMAP_NEUTRAL])

    def cell(variable, level, key, field):
        k = (variable, level, key)
        return float(index.at[k, field]) if k in index.index else np.nan

    xs, x = [], 0.0
    for i, metric in enumerate(columns):
        x += viscons.RQ1_HEATMAP_BAND_GAP if i and metric.band != columns[i - 1].band else 0
        xs.append(x)
        x += 1
    width_units = x
    ys, cells, y = [], [], 0.0
    for i, group in enumerate(rows):
        y += viscons.RQ1_HEATMAP_ROW_GAP if i else 0
        for level, label in zip(group.levels, group.labels):
            ys.append(y)
            cells.append((group, level, label))
            y += 1
    height_units = y
    scale = {m.key: np.nanmax([abs(cell(g.column, lv, m.key, 'deviation')) for g, lv, _ in cells]
                              + [viscons.RQ2_HEATMAP_MIN_SCALE]) for m in columns}

    pad = 0.07
    title_w = max(_text_size(g.title, 'small', 'bold')[0] for g in rows)
    label_w = max(_text_size(label, 'small')[0] for _, _, label in cells)
    column_h = max(_rotated_height(m.short or m.key, 'small', 40) for m in columns)
    has_brackets = any(columns[i].group == columns[i + 1].group for i in range(len(columns) - 1))
    left_in = pad + title_w + pad + label_w + pad
    bottom_in = pad + column_h + (0.22 if has_brackets else 0) + 0.2 + pad
    top_in = 0.3
    fig_w, fig_h = left_in + cell_w * width_units + 0.15, top_in + cell_h * height_units + bottom_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left_in / fig_w, bottom_in / fig_h, cell_w * width_units / fig_w, cell_h * height_units / fig_h])
    ax.set_xlim(0, width_units)
    ax.set_ylim(height_units, 0)
    ax.axis('off')
    ux, uy = 1 / cell_w, 1 / cell_h

    previous = None
    for (group, level, label), yy in zip(cells, ys):
        for metric, xx in zip(columns, xs):
            v = cell(group.column, level, metric.key, 'deviation')
            lo, hi = (cell(group.column, level, metric.key, f"deviation_{s}") for s in ('ci_low', 'ci_high'))
            if not np.isfinite(v):
                color, strength = '#ffffff', 0.0
            else:
                strength = min(abs(v) / scale[metric.key], 1.0)
                cmap = neutral if metric.direction == summary.NEUTRAL else \
                    better if v * metric.direction > 0 else worse
                color = cmap(0.85 * strength)
            ax.add_patch(Rectangle((xx, yy), 1, 1, facecolor=color, edgecolor='white', linewidth=0.6))
            faded = not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 <= hi
            text_color = viscons.RQ1_HEATMAP_FADED_TEXT if faded else ('white' if strength > 0.6 else '#1f1f1f')
            ax.text(xx + 0.5, yy + 0.5, _short_signed(v), ha='center', va='center', fontsize='xx-small',
                    color=text_color)
        dark = viscons.RQ1_HEATMAP_DARK[group.color_key]
        ax.text(-pad * ux, yy + 0.5, label, ha='right', va='center', fontsize='small', clip_on=False,
                color=viscons.METRIC_ROW_LABEL_COLOR)
        if group is not previous:
            ax.text(-(left_in - pad) * ux, yy + 0.5, group.title, ha='left', va='center', fontweight='bold',
                    fontsize='small', clip_on=False, color=dark)
            previous = group
    start = 0
    for group in rows[:-1]:                               # thin rules between the variables
        start += len(group.levels)
        rule = ys[start - 1] + 1 + viscons.RQ1_HEATMAP_ROW_GAP / 2
        ax.plot([-(left_in - pad) * ux, width_units], [rule, rule], color=viscons.COLOR_GRID, linewidth=0.6,
                clip_on=False)

    y0 = height_units + pad * uy
    for metric, xx in zip(columns, xs):
        ax.text(xx + 0.5, y0, metric.short or metric.key, rotation=40, ha='right', va='top', fontsize='small',
                rotation_mode='anchor', clip_on=False)
    y_bracket = y0 + column_h * uy + 0.05 * uy
    if has_brackets:
        _brackets(ax, columns, xs, y_bracket)
        y_bracket += 0.2 * uy
    for band in dict.fromkeys(m.band for m in columns):
        bx = [xx for m, xx in zip(columns, xs) if m.band == band]
        ax.add_patch(Rectangle((bx[0], y_bracket), bx[-1] + 1 - bx[0], 0.2 * uy, facecolor=viscons.RQ1_HEATMAP_BAND,
                               edgecolor='none', clip_on=False))
        ax.text((bx[0] + bx[-1] + 1) / 2, y_bracket + 0.1 * uy, band.upper(), ha='center', va='center',
                fontweight='bold', fontsize='x-small', clip_on=False)
    handles = [Patch(color=viscons.RQ2_HEATMAP_BETTER, label='better than the grand mean'),
               Patch(color=viscons.RQ2_HEATMAP_WORSE, label='worse'),
               Patch(color=viscons.RQ2_HEATMAP_NEUTRAL, label='|deviation| (no better direction)')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(left_in / fig_w, 1.0), ncol=3, frameon=False,
               fontsize='x-small', handlelength=1.0, handletextpad=0.4, columnspacing=1.2)
    fig.text(1 - 0.15 / fig_w, 1 - 0.04 / fig_h, f"colour scaled per metric (full at ≥ {viscons.RQ2_HEATMAP_MIN_SCALE:g}) · faded: 95% CI includes 0", ha='right',
             va='top', fontsize='xx-small', color=viscons.COLOR_TEXT_SECONDARY)
    if title:
        fig.suptitle(title, y=1 + 0.25 / fig_h, fontsize='small')
    return fig


def rq2_figure_name(kind: str, value: str = 'deviation', metrics: Optional[str | Sequence[str]] = None) -> str:
    """rq2_<metrics>_<value>_<kind>, as rq1_figure_name (e.g. rq2_all_deviation_heatmap, rq2_technical_emm_bars)."""
    selection = 'all' if metrics is None else metrics if isinstance(metrics, str) else '-'.join(metrics)
    return f"rq2_{selection}_{value}_{kind}"


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
    room = 3.2 if fit and 'r2' in results else 0.0             # empty rows at the bottom for the fit box
    label_w = max(text_size(text, 'small', 'bold' if kind == 'header' else 'normal')[0] for kind, text, *_ in lines)
    height = 0.75 + viscons.RQ2_FOREST_ROW_IN * (n + room)
    width = viscons.PAPER_TEXT_WIDTH_IN
    fig, axes = plt.subplots(1, len(columns), figsize=(width, height), sharey=True, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=(label_w + 0.1) / width, right=0.99, top=1 - 0.42 / height, bottom=0.36 / height,
                        wspace=0.1)
    ys = np.arange(n)
    index = results.set_index(['variable', 'level', 'metric'])
    for ax, metric in zip(axes, columns):
        shade = False
        for y, (kind, text, family, variable, level) in zip(ys, lines):
            if kind != 'row':
                shade = False                                  # every group starts with a shaded row
                continue
            shade = not shade
            color = viscons.METRIC_BAR_COLORS[family]
            if shade:
                ax.axhspan(y - 0.5, y + 0.5, color=color, alpha=2.5 * viscons.RQ2_FOREST_SHADE_ALPHA, linewidth=0)
            k = (variable, level, metric.key)
            if k not in index.index:
                continue
            v, lo, hi = (float(index.at[k, c]) for c in (value, f"{value}_ci_low", f"{value}_ci_high"))
            if not np.isfinite(v):
                continue
            err = [[max(v - lo, 0)], [max(hi - v, 0)]] if np.isfinite(lo) and np.isfinite(hi) else None
            ax.errorbar([v], [y], xerr=err, fmt='o', color=color, markersize=viscons.RQ2_FOREST_MARKER_SIZE,
                        elinewidth=viscons.METRIC_CI_LINEWIDTH, capsize=viscons.METRIC_CI_CAPSIZE,
                        capthick=viscons.METRIC_CI_LINEWIDTH)
        ax.axvline(0, color=viscons.METRIC_GROUP_RULE_COLOR, linewidth=0.6, linestyle='--', zorder=0)
        ax.set_title(metric.label, fontsize='small', fontweight='bold')
        ax.set_xlabel('Difference from reference' if value == 'contrast' else 'Deviation from grand mean',
                      fontsize='x-small', color=viscons.COLOR_TEXT_SECONDARY)
        ax.tick_params(axis='x', labelsize='x-small', colors=viscons.COLOR_TEXT_SECONDARY)
        ax.tick_params(axis='y', length=0)
        ax.locator_params(axis='x', nbins=4)
        ax.grid(axis='x', color=viscons.COLOR_GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        stats = results.loc[results['metric'] == metric.key, ['r2', 'r2_adj', 'delta_r2_model']] \
            if 'r2' in results else pd.DataFrame()
        if fit and len(stats):
            r = stats.iloc[0]
            ax.text(0.97, 0.015, f"$R^2$ = {r['r2']:.2f}\n$R^2_{{adj}}$ = {r['r2_adj']:.2f}\n"
                                 f"$\\Delta R^2$ (LLM) = {r['delta_r2_model']:.2f}",
                    transform=ax.transAxes, ha='right', va='bottom', fontsize='xx-small',
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85, 'pad': 1.5})
    axes[0].set_ylim(n - 0.5 + room, -0.5)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([text for _, text, *_ in lines], fontsize='small')
    for tick, (kind, _, family, *_) in zip(axes[0].get_yticklabels(), lines):
        tick.set_color(viscons.RQ1_HEATMAP_DARK[family] if kind == 'header' else viscons.METRIC_BAR_COLORS[family])
        if kind == 'header':
            tick.set_fontweight('bold')
    handles = [Line2D([], [], marker='o', linestyle='', color=viscons.METRIC_BAR_COLORS[f], label=f.capitalize())
               for f in (summary.PERSONA, summary.CONTEXT) if any(line[2] == f for line in lines)]
    fig.legend(handles=handles, loc='upper right', ncol=len(handles), frameon=False, fontsize='x-small',
               bbox_to_anchor=(0.99, 1.0), handletextpad=0.2)
    if title:
        fig.suptitle(title, x=0.02, ha='left', y=1.0, fontsize='small')
    return fig


def influence_nested() -> List[str]:
    from analysis.influence import NESTED
    return list(NESTED)
