"""
GT descriptives (notebook 0_gt.ipynb; data: groundtruth/stats.py). Rows are grouped by field (field name above its
rows, 'All fields' last), seniority junior / mid / senior inside a group.

    plot_gt_population(pop, by)           scholars per field, stacked by seniority / perceived gender / ethnicity
    plot_gt_field_overlap(overlap)        heatmap: share of the row field's scholars that also belong to the column field
    plot_gt_bibliometrics(quantiles)      one panel per bibliometric: a box per field x seniority from precomputed
                                          quantiles (box q25-q75, line median, whiskers q05-q95, dot mean)
    plot_gt_bibliometrics_by(quantiles, attribute)
                                          the same per perceived gender / ethnicity: per class a dot at the median and
                                          a line q25-q75 (offset inside the row)
    plot_gt_shares_by_prominence(table, attribute, cls)
                                          share of one class (e.g. female) inside L / M / H and the top x % (vs all),
                                          one panel per prominence attribute
    plot_gt_histogram(hist, thresholds)   distribution of one gt_authors column (bars, or one line per field), with the
                                          config thresholds as dashed lines (seniority cut-offs, gender edges, ...)
    plot_gt_composition(shares)           the GT targets: class shares per field x seniority, one panel per attribute
                                          (the flows' colours: gender diverging, ethnicity / country categorical,
                                          L / M / H sequential; 'other' / 'unknown' grey)
"""

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.ticker import FuncFormatter, NullFormatter

from groundtruth import stats as gs
from viz import constants as viscons
from viz.figures import shades
from viz.flow_plots import TARGET_TITLES, class_hatch, class_label, prompt_countries, target_order_colors

METRIC_LABELS = {'academic_age': 'academic age (years)', 'n_works': '# works', 'n_citations': '# citations',
                 'n_coauthors': '# unique coauthors', 'h_index': 'h-index',
                 'citations_per_paper_per_year': 'citations per paper per year'}
LOG_METRICS = {'n_works': 1, 'n_citations': 1, 'n_coauthors': 1, 'h_index': 1,
               'citations_per_paper_per_year': 0.1}    # metric -> symlog linear threshold (0 and skewed counts)
SENIORITY_COLORS = dict(zip(gs.SENIORITY_ORDER, shades(list(viscons.FLOW_SEQUENTIAL), len(gs.SENIORITY_ORDER))))


# --- layout ------------------------------------------------------------------------------------------------------
def row_positions(cells: pd.DataFrame) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    """y of every (field, seniority) row (top to bottom, a gap with the field name above each group) and the
    (field, y) of the group titles."""
    y, positions, titles = 0.0, [], []
    for field, group in cells.groupby('field', sort=False):
        titles.append((field, y))
        y += viscons.GT_GROUP_GAP
        for _ in range(len(group)):
            positions.append(y)
            y += 1
        y += 0.25
    return np.array(positions), titles


def _row_axes(ax: plt.Axes, cells: pd.DataFrame, positions: np.ndarray, titles, labels: bool = True) -> None:
    ax.set_yticks(positions)
    ax.set_yticklabels(cells['seniority'] if labels else [])
    ax.tick_params(axis='y', length=0)
    for field, y in titles:
        if labels:
            ax.annotate(field, (0, y + 0.35), xycoords=('axes fraction', 'data'),
                        xytext=(-2, 0), textcoords='offset points', ha='right', va='center', fontsize='x-small',
                        fontweight='bold', annotation_clip=False)
    ax.set_ylim(positions.max() + 0.75, -0.25)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)


def _height(n_rows: int, n_groups: int) -> float:
    return 0.6 + viscons.GT_ROW_IN * (n_rows + n_groups * (viscons.GT_GROUP_GAP + 0.25))


# --- population --------------------------------------------------------------------------------------------------
def count_unit(n_max: float) -> Tuple[float, str]:
    """Divisor and name of the unit for counts up to n_max (scholars, thousands, millions)."""
    if n_max >= 1e6:
        return 1e6, 'millions'
    if n_max >= 1e4:
        return 1e3, 'thousands'
    return 1.0, ''


def _text_color(color: str) -> str:
    r, g, b = to_rgb(color)
    return 'white' if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else viscons.COLOR_TEXT_SECONDARY


def plot_gt_population(pop: pd.DataFrame, by: str = 'seniority', include_all: bool = False,
                       include_unknown: bool = True, normalize: bool = False, gender_labels=None,
                       title: Optional[str] = None) -> plt.Figure:
    """Scholars per field (memberships), stacked by `by` (seniority junior / mid / senior, or perceived gender /
    ethnicity in the flows' colours), values written in the segments. normalize=False: counts, unit (scholars /
    thousands / millions) chosen from the largest field; True: % of the field (each row sums to 100; without unknown,
    over the known classes)."""
    pop = pop if include_all else pop[pop['field'] != gs.ALL_FIELDS]
    if not include_unknown:
        pop = pop[pop[by] != gs.UNKNOWN]
    if by == 'seniority':
        classes = [s for s in gs.SENIORITY_ORDER if s in set(pop[by])]
        colors = SENIORITY_COLORS
    else:
        classes, colors = target_order_colors(by, pop[by].unique(), gender_labels)
    table = pop.pivot_table(index='field', columns=by, values='n', aggfunc='sum').reindex(
        index=gs.field_order(pop['field'].unique()), columns=classes).fillna(0)
    if normalize:
        table = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
        divisor, unit, decimals = 1.0, '', 0
    else:
        divisor, unit = count_unit(table.sum(axis=1).max())
        decimals = 1 if divisor > 1 else 0
    fig, ax = plt.subplots(figsize=(viscons.PAPER_TEXT_WIDTH_IN, 0.7 + 0.22 * len(table)), layout='constrained')
    left = np.zeros(len(table))
    y = np.arange(len(table))
    min_label = 0.04 * table.sum(axis=1).max() / divisor          # write a count if its segment is wide enough
    for c in table.columns:
        values = table[c].to_numpy() / divisor
        ax.barh(y, values, left=left, height=0.7, color=colors[c], label=class_label(by, c) if by != 'seniority' else c,
                **{'edgecolor': viscons.COLOR_BAR_EDGE, 'linewidth': 0.4, **class_hatch(c)})
        for yi, l, v in zip(y, left, values):
            if v >= min_label:
                ax.text(l + v / 2, yi, f"{v:,.{decimals}f}{'%' if normalize else ''}", ha='center', va='center',
                        fontsize=viscons.VALUE_FONT_SIZE, color=_text_color(colors[c]))
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(table.index)
    ax.invert_yaxis()
    if normalize:
        ax.set_xlim(0, 100)
        ax.set_xlabel('% of the field\'s scholars')
    else:
        ax.set_xlabel(f"scholars{f' ({unit})' if unit else ''}; a scholar counts in every field they belong to")
    ax.grid(axis='x', color=viscons.COLOR_GRID)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.legend(ncol=len(table.columns), loc='lower right', bbox_to_anchor=(1, 1), fontsize='x-small', frameon=False)
    name = 'seniority' if by == 'seniority' else TARGET_TITLES.get(by, by)
    ax.set_title(title or f"GT scholars per field and {name}", loc='left')
    return fig


def plot_gt_field_overlap(overlap: pd.DataFrame, title: Optional[str] = None) -> plt.Figure:
    """Row-normalised overlap: % of the row field's scholars also in the column field (diagonal: field size)."""
    sizes = np.diag(overlap.to_numpy()).astype(float)
    share = overlap.to_numpy() / sizes[:, None]
    n = len(overlap)
    fig, ax = plt.subplots(figsize=(viscons.PAPER_TEXT_WIDTH_IN / 2, 0.6 + 0.3 * n), layout='constrained')
    off = np.where(np.eye(n, dtype=bool), np.nan, share)
    vmax = float(np.nanmax(off)) if n > 1 and np.isfinite(np.nanmax(off)) and np.nanmax(off) > 0 else 1.0
    ax.imshow(off * 100, cmap=viscons.GT_HEATMAP_CMAP, vmin=0, vmax=vmax * 100, aspect='auto')
    for i in range(n):
        for j in range(n):
            text = f"{sizes[i] / 1e6:.1f}M" if i == j else f"{share[i, j]:.0%}"
            dark = i != j and share[i, j] > 0.6 * vmax
            ax.text(j, i, text, ha='center', va='center', fontsize=viscons.VALUE_FONT_SIZE,
                    color='white' if dark else viscons.COLOR_TEXT_SECONDARY, fontweight='bold' if i == j else 'normal')
    ax.set_xticks(range(n))
    ax.set_xticklabels(overlap.columns, rotation=35, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(overlap.index)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel('also in')
    ax.set_title(title or 'Field overlap (% of the row field)', loc='left')
    return fig


# --- bibliometrics -----------------------------------------------------------------------------------------------
def plot_gt_bibliometrics(quantiles: pd.DataFrame, metrics: Optional[Sequence[str]] = None, ncols: int = 3,
                          title: Optional[str] = None) -> plt.Figure:
    """One panel per metric, a box per field x seniority (precomputed approximate quantiles)."""
    metrics = list(metrics) if metrics is not None else list(dict.fromkeys(quantiles['metric'].astype(str)))
    cells = quantiles.loc[quantiles['metric'].astype(str) == metrics[0], ['field', 'seniority']].reset_index(drop=True)
    positions, titles = row_positions(cells)
    ncols = min(ncols, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    height = _height(len(cells), len(titles))
    fig, axes = plt.subplots(nrows, ncols, figsize=(viscons.PAPER_TEXT_WIDTH_IN, height * nrows), squeeze=False,
                             layout='constrained')
    for k, metric in enumerate(metrics):
        ax = axes.flat[k]
        q = quantiles[quantiles['metric'].astype(str) == metric].reset_index(drop=True)
        q = cells.merge(q, on=['field', 'seniority'], how='left')
        boxes = [{'med': r['q50'], 'q1': r['q25'], 'q3': r['q75'], 'whislo': r['q05'], 'whishi': r['q95'],
                  'mean': r['mean'], 'fliers': []} for _, r in q.iterrows()]
        artists = ax.bxp(boxes, positions=positions, orientation='horizontal', widths=0.7, showmeans=True, showfliers=False,
                         patch_artist=True, manage_ticks=False,
                         meanprops={'marker': 'o', 'markersize': 1.8, 'markerfacecolor': viscons.COLOR_ERROR_BAR,
                                    'markeredgecolor': 'none'},
                         medianprops={'color': viscons.COLOR_ERROR_BAR, 'linewidth': 0.8},
                         whiskerprops={'linewidth': 0.6}, capprops={'linewidth': 0.6})
        for patch, seniority in zip(artists['boxes'], q['seniority']):
            patch.set_facecolor(SENIORITY_COLORS.get(seniority, viscons.FLOW_OTHER_COLOR))
            patch.set_edgecolor(viscons.COLOR_ERROR_BAR)
            patch.set_linewidth(0.5)
        if metric in LOG_METRICS:
            ax.set_xscale('symlog', linthresh=LOG_METRICS[metric])
            ax.set_xlim(left=0)
        _row_axes(ax, cells, positions, titles, labels=k % ncols == 0)
        ax.grid(axis='x', color=viscons.COLOR_GRID)
        ax.set_axisbelow(True)
        ax.set_xlabel(METRIC_LABELS.get(metric, metric))
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle(title or 'Bibliometrics (box q25-q75, whiskers q05-q95, dot mean)', fontsize='small')
    return fig


def plot_gt_bibliometrics_by(quantiles: pd.DataFrame, attribute: str, metrics: Optional[Sequence[str]] = None,
                             ncols: int = 3, include_unknown: bool = False, gender_labels=None,
                             title: Optional[str] = None) -> plt.Figure:
    """Per metric panel, rows field x seniority; per perceived class a dot (median) and a line (q25-q75)."""
    q = quantiles if include_unknown else quantiles[quantiles[attribute] != gs.UNKNOWN]
    metrics = list(metrics) if metrics is not None else list(dict.fromkeys(q['metric'].astype(str)))
    cells = q[['field', 'seniority']].drop_duplicates().reset_index(drop=True)
    cells = gs._sort_cells(cells)
    positions, titles = row_positions(cells)
    order, colors = target_order_colors(attribute, q[attribute].unique(), gender_labels)
    offsets = np.linspace(-0.3, 0.3, len(order)) if len(order) > 1 else [0.0]
    ncols = min(ncols, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    height = _height(len(cells), len(titles))
    fig, axes = plt.subplots(nrows, ncols, figsize=(viscons.PAPER_TEXT_WIDTH_IN, height * nrows + 0.3), squeeze=False,
                             layout='constrained')
    y_of = {(f, s): y for (f, s), y in zip(cells.itertuples(index=False, name=None), positions)}
    for k, metric in enumerate(metrics):
        ax = axes.flat[k]
        m = q[q['metric'].astype(str) == metric]
        for c, dy in zip(order, offsets):
            g = m[m[attribute] == c]
            y = np.array([y_of[(f, s)] for f, s in zip(g['field'], g['seniority'])]) + dy
            ax.hlines(y, g['q25'], g['q75'], color=colors[c], linewidth=0.8)
            ax.plot(g['q50'], y, 'o', color=colors[c], markersize=1.8,
                    label=class_label(attribute, c) if k == 0 else None)
        if metric in LOG_METRICS:
            ax.set_xscale('symlog', linthresh=LOG_METRICS[metric])
            ax.set_xlim(left=0)
        _row_axes(ax, cells, positions, titles, labels=k % ncols == 0)
        ax.grid(axis='x', color=viscons.COLOR_GRID)
        ax.set_axisbelow(True)
        ax.set_xlabel(METRIC_LABELS.get(metric, metric))
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    fig.legend(loc='outside upper right', ncol=len(order), fontsize=viscons.MIN_FONT_SIZE, frameon=False,
               title=TARGET_TITLES.get(attribute, attribute).capitalize(), title_fontsize='x-small')
    fig.suptitle(title or f"Bibliometrics by {TARGET_TITLES.get(attribute, attribute)} (dot median, line q25-q75)",
                 fontsize='small')
    return fig


def plot_gt_shares_by_prominence(table: pd.DataFrame, attribute: str, cls: str,
                                 groups: Optional[Sequence[str]] = None, bin_labels=None,
                                 title: Optional[str] = None) -> plt.Figure:
    """Share of class `cls` (e.g. female) inside each prominence level, rows field x seniority, one panel per group:
    L / M / H (sequential colours) or all vs top x % (grey vs dark). Equal dots = the class is spread evenly."""
    present = set(table['group'])
    groups = list(groups) if groups is not None else [g for g in gs.PROMINENCE_GROUPS if g in present]
    cells = gs._sort_cells(table[['field', 'seniority']].drop_duplicates().reset_index(drop=True))
    positions, titles = row_positions(cells)
    y_of = {(f, s): y for (f, s), y in zip(cells.itertuples(index=False, name=None), positions)}
    ncols = min(5, len(groups))
    nrows = int(np.ceil(len(groups) / ncols))
    height = _height(len(cells), len(titles))
    fig, axes = plt.subplots(nrows, ncols, figsize=(viscons.PAPER_TEXT_WIDTH_IN, height * nrows + 0.2),
                             squeeze=False, layout='constrained')
    for k, group in enumerate(groups):
        ax = axes.flat[k]
        g = table[(table['group'] == group)]
        if group.startswith('top_'):
            levels, colors = ['all', 'top'], {'all': viscons.FLOW_OTHER_COLOR, 'top': viscons.FLOW_SEQUENTIAL[-1]}
        else:
            levels, colors = target_order_colors(group, g['level'].unique(), bin_labels=bin_labels)
            levels = [l for l in levels if l != gs.UNKNOWN]
        # share of cls in every (cell, level) that has members: 0 when the class is absent, NaN when the level is
        exists = g.groupby(['field', 'seniority', 'level'])['n'].sum().index
        wide = (g[g['class'] == cls].set_index(['field', 'seniority', 'level'])['share']
                .reindex(exists).fillna(0).unstack('level').reindex(columns=levels))
        for (f, s), row in wide.iterrows():
            values = [row.get(l, np.nan) * 100 for l in levels]
            ax.plot(values, [y_of[(f, s)]] * len(values), color=viscons.COLOR_GRID, linewidth=0.8, zorder=1)
        for level in levels:
            if wide[level].notna().any():
                ax.scatter(wide[level] * 100, [y_of[i] for i in wide.index], s=7, color=colors[level], zorder=2,
                           label=level, edgecolors='none')
        _row_axes(ax, cells, positions, titles, labels=k % ncols == 0)
        ax.grid(axis='x', color=viscons.COLOR_GRID)
        ax.set_axisbelow(True)
        ax.set_xlabel(f"% {class_label(attribute, cls).lower()}")
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=len(levels), fontsize=viscons.MIN_FONT_SIZE, frameon=False,
                  handletextpad=0.1, columnspacing=0.5, title=TARGET_TITLES.get(group, group.replace('_', ' ')).capitalize(),
                  title_fontsize='x-small', alignment='left')
    for ax in axes.flat[len(groups):]:
        ax.set_visible(False)
    fig.suptitle(title or f"Share {class_label(attribute, cls).lower()} ({TARGET_TITLES.get(attribute, attribute)}) "
                          "by prominence class", fontsize='small')
    return fig


# --- distributions -----------------------------------------------------------------------------------------------
def plot_gt_histogram(hist: pd.DataFrame, xlabel: str, thresholds: Optional[Dict[str, float]] = None,
                      ax: Optional[plt.Axes] = None, title: Optional[str] = None, log_y: bool = False,
                      legend: bool = True) -> plt.Figure:
    """Bars (share of scholars per bin), or one line per field when hist has a 'field' column (share within the field,
    memberships); thresholds {label: x} as dashed lines."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(viscons.PAPER_TEXT_WIDTH_IN / 2, viscons.GT_PANEL_HEIGHT_IN), layout='constrained')
    else:
        fig = ax.figure
    if 'field' in hist:
        palette = viscons.FLOW_CATEGORICAL
        for i, (field, g) in enumerate(hist.groupby('field', sort=False)):
            share = g['n'] / max(g['n'].sum(), 1)
            ax.step(g['left'], share * 100, where='post', color=palette[i % len(palette)], linewidth=0.9, label=field)
        if legend:
            ax.legend(fontsize=viscons.MIN_FONT_SIZE, frameon=False, ncol=2)
    else:
        share = hist['n'] / max(hist['n'].sum(), 1)
        ax.bar(hist['left'], share * 100, width=hist['right'] - hist['left'], align='edge',
               color=SENIORITY_COLORS[gs.SENIORITY_ORDER[-1]], edgecolor=viscons.COLOR_BAR_EDGE, linewidth=0.2)
    if log_y:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    for label, x in (thresholds or {}).items():
        ax.axvline(x, **viscons.GT_THRESHOLD_LINE)
        ax.annotate(label, (x, 1), xycoords=('data', 'axes fraction'), xytext=(2, -2), textcoords='offset points',
                    ha='left', va='top', fontsize=viscons.MIN_FONT_SIZE, color=viscons.COLOR_TEXT_SECONDARY)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('% of scholars')
    ax.grid(axis='y', color=viscons.COLOR_GRID)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, loc='left')
    return fig


# --- composition (the GT targets) --------------------------------------------------------------------------------
def composition_tables(shares: pd.DataFrame, attributes: Sequence[str], include_unknown: bool = True,
                       top_countries: int = viscons.FLOW_TOP_COUNTRIES) -> Dict[str, pd.DataFrame]:
    return {a: gs.composition(shares, a, include_unknown, top=top_countries if a == 'country' else None,
                              keep=prompt_countries() if a == 'country' else ())
            for a in attributes}


def plot_gt_composition(shares: pd.DataFrame, attributes: Sequence[str] = tuple(viscons.GT_COMPOSITION_ATTRIBUTES),
                        include_unknown: bool = True, ncols: int = viscons.GT_COMPOSITION_NCOLS,
                        gender_labels=None, bin_labels=None, title: Optional[str] = None) -> plt.Figure:
    """Stacked 100 % bars of the class shares per field x seniority, one panel per attribute (legend above it)."""
    tables = composition_tables(shares, attributes, include_unknown)
    cells = tables[attributes[0]].index.to_frame(index=False)
    positions, titles = row_positions(cells)
    ncols = min(ncols, len(attributes))
    nrows = int(np.ceil(len(attributes) / ncols))
    height = _height(len(cells), len(titles)) + 0.35
    fig, axes = plt.subplots(nrows, ncols, figsize=(viscons.PAPER_TEXT_WIDTH_IN, height * nrows), squeeze=False,
                             layout='constrained')
    for k, attribute in enumerate(attributes):
        ax = axes.flat[k]
        table = tables[attribute].reindex(pd.MultiIndex.from_frame(cells))
        order, colors = target_order_colors(attribute, table.columns, gender_labels, bin_labels)
        left = np.zeros(len(table))
        for c in order:
            values = table[c].fillna(0).to_numpy() if c in table else np.zeros(len(table))
            ax.barh(positions, values * 100, left=left * 100, height=0.8, color=colors[c], label=class_label(attribute, c),
                    **{'edgecolor': viscons.COLOR_BAR_EDGE, 'linewidth': 0.3, **class_hatch(c)})
            left += values
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        _row_axes(ax, cells, positions, titles, labels=k % ncols == 0)
        ax.set_xlabel('% of GT scholars')
        ax.legend(ncol=min(len(order), 4), loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=viscons.MIN_FONT_SIZE,
                  frameon=False, handlelength=0.8, columnspacing=0.6,
                  title=TARGET_TITLES.get(attribute, attribute).capitalize(), title_fontsize='x-small',
                  alignment='left')
    for ax in axes.flat[len(attributes):]:
        ax.set_visible(False)
    fig.suptitle(title or 'GT composition per field and seniority (the parity targets)', fontsize='small')
    return fig
