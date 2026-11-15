"""
Figures of the interactions (analysis/interactions.py; plan D.7.3), apart from the RQ figures.

    plot_interaction_heatmap  rows = the screening blocks (prompt x prompt, LLM x prompt; * = pre-specified, black, before the label), columns =
                              metrics (TECHNICAL | SOCIAL, as the RQ1 heatmap), value = partial omega^2 (or another
                              measure); faded = not flagged (correction 'by': flag_screening, 'bh': flag_screening_bh;
                              both also need omega^2 >= the minimum); bottom row: Delta R^2 of all interactions
    plot_interaction_cells    one panel per metric: rows = levels of the first factor, columns = levels of the second,
                              colour / value = deviation of the cell from the additive prediction (observed - additive;
                              coloured as the RQ2 heatmap: green better / red worse by the metric's direction, grey
                              |deviation| without one; full at the panel limit, at least the SESOI so smaller
                              deviations stay pale); faded
                              = not significant after the correction over all cells loaded (cons.HEATMAP_CORRECTION,
                              default BY; LLM cells: no CI, not faded); panel title: metric (+- colour limit);
                              shared y axis, one row of panels while panels x columns <= INTERACTION_CELLS_MAX_COLUMNS and
                              panels <= INTERACTION_CELLS_MAX_PANELS; legend above
    file names: viz/names.interaction_figure_name
"""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, to_rgba

from analysis import interactions as ix
from analysis import multiple_testing as mt
from analysis.deviations import design_order
from analysis.influence import MODEL_FACTOR
from metrics import summary
from utils import constants as cons
from viz import constants as viscons
from viz.figures import clean_axes, family_cmap, text_on, text_size
from viz.formatting import cell, signed_cell
from viz.labels import CORRECTION_LABELS, LLM, MEASURE_LABELS, fade_rule, level_label, model_labels, variable_title
from viz.heatmap import HeatmapFrame, deviation_color, deviation_legend_handles, family_colorbars
from viz.rq_plots import metric_keys

ALL_INTERACTIONS = r'All ($\Delta R^2$)'      # bottom row: Delta R^2 of all interactions together
FLAG_COLUMNS = {'by': 'flag_screening', 'bh': 'flag_screening_bh', 'prespecified': 'flag_prespecified'}


PRESPECIFIED_MARK = '*'          # before the label of a pre-specified block, in black


def block_label(block: str, prespecified: bool = False) -> str:
    a, b = block.split(':')
    return (f"{PRESPECIFIED_MARK} " if prespecified else '') + f"{variable_title(a)} × {variable_title(b)}"


def plot_interaction_heatmap(table: pd.DataFrame, metrics: Optional[str | Sequence[str]] = None,
                             measure: str = 'omega2', flag: str = 'by', vmax: Optional[float] = None,
                             width: Optional[float] = None, title: Optional[str] = None) -> plt.Figure:
    """Heatmap of the screening blocks (see module docstring). table: interactions.load() (flags included)."""
    blocks = table[table['role'].isin([ix.PRESPECIFIED, ix.SCREENING])]
    keys = metric_keys(blocks, metrics)
    columns = [summary.METRICS[k] for k in keys]
    order = list(dict.fromkeys(blocks['block']))
    groups = [('Prompt × prompt', cons.FAMILY_INTERACTION, [b for b in order if MODEL_FACTOR not in b.split(':')]),
              (f'{LLM} × prompt', cons.FAMILY_MODEL, [b for b in order if MODEL_FACTOR in b.split(':')])]
    groups = [g for g in groups if g[2]]
    pre = set(blocks.loc[blocks['role'] == ix.PRESPECIFIED, 'block'])
    value = blocks.pivot(index='block', columns='metric', values=measure)
    flagged = blocks.pivot(index='block', columns='metric', values=FLAG_COLUMNS[flag]).fillna(False).astype(bool)
    glob = table[table['role'] == ix.GLOBAL_ROLE].set_index('metric')
    vmax = vmax or max(0.02, float(np.nanmax(value[keys].clip(lower=0).to_numpy())))
    norm = Normalize(0, vmax)
    labels = [block_label(b, b in pre) for _, _, members in groups for b in members]
    label_w = max(text_size(t, 'small', 'bold')[0] for t in labels + [ALL_INTERACTIONS])
    frame = HeatmapFrame(columns, [len(g[2]) for g in groups], label_w, residual=True, right_in=0.62, width=width)
    ax = frame.ax
    rows = [summary.RowGroup(title, 'block', members, [block_label(b) for b in members], family)
            for title, family, members in groups]
    for group, ys in zip(rows, frame.ys):
        cmap = family_cmap(group.color_key)
        for block, label, yy in zip(group.levels, group.labels, ys):
            for metric, xx in zip(columns, frame.xs):
                v = value.at[block, metric.key] if metric.key in value.columns else np.nan
                on = bool(flagged.at[block, metric.key]) if metric.key in flagged.columns else False
                frame.value_cell(xx, yy, v, on, cmap, norm)
            text = ax.text(-frame.pad * frame.ux, yy + 0.5, label, ha='right', va='center_baseline',
                           fontweight='bold', fontsize='small', clip_on=False, color=viscons.FAMILY_TEXT[group.color_key])
            if block in pre:                           # pre-specified: black mark just left of the label
                ax.annotate(f"{PRESPECIFIED_MARK} ", xy=(0, 0), xycoords=text, ha='right', va='bottom',
                            fontweight='bold', fontsize='small', annotation_clip=False, color='black')
    for metric, xx in zip(columns, frame.xs):
        r = glob['delta_r2'].get(metric.key, np.nan)
        frame.residual(xx, cell(r))
    ax.text(-frame.pad * frame.ux, frame.y_residual + frame.residual_h / 2, ALL_INTERACTIONS, ha='right',
            va='center_baseline', fontsize='small', clip_on=False, color=viscons.COLOR_TEXT_SECONDARY)
    frame.bands([g.title for g in rows], [g.color_key for g in rows])
    frame.column_labels()
    frame.axis_titles('Interaction block', 'Evaluation metric')
    family_colorbars(frame, {g.color_key: family_cmap(g.color_key) for g in rows}, norm, rows,
               MEASURE_LABELS.get(measure, measure))
    if flag in CORRECTION_LABELS:              # the rule of interactions.flag()
        frame.note(f"faded: not flagged ({fade_rule(flag, cons.INTERACTION_ALPHA)} or "
                   f"{MEASURE_LABELS['omega2'].lower()} < {cons.INTERACTION_OMEGA2_MIN:g})")
    if title:
        frame.fig.suptitle(title, fontsize='small', y=1 + 0.2 / frame.fig.get_figheight())
    return frame.fig


def _levels(cells: pd.DataFrame, factor: str, column: str) -> list:
    present = list(dict.fromkeys(cells[column].astype(str)))
    if factor in cons.VARIABLE_VALUES:
        return design_order(factor, present)
    if factor in summary.MODEL_ATTRIBUTES:
        known = [str(x) for x in summary.MODEL_ATTRIBUTES[factor][1]]
        return [x for x in known if x in present] + sorted(set(present) - set(known))
    return sorted(present)


def _level_text(factor: str, level: str, models: dict) -> str:
    if factor == MODEL_FACTOR:
        return models.get(level, level)
    return level_label(factor, level)


def cells_grid(n_panels: int, n_columns: int, max_columns: int = viscons.INTERACTION_CELLS_MAX_COLUMNS,
               ncols: Optional[int] = None, max_panels: int = viscons.INTERACTION_CELLS_MAX_PANELS) -> tuple:
    """(nrows, ncols) of the panels: one row while panels x cell columns <= max_columns and panels <= max_panels,
    else balanced rows."""
    if ncols is None:
        ncols = min(max_panels, max(1, max_columns // max(n_columns, 1)))
    ncols = max(1, min(ncols, n_panels))
    nrows = int(np.ceil(n_panels / ncols))
    return nrows, int(np.ceil(n_panels / nrows))          # balanced: 7 panels at 4 per row -> 4 + 3


def plot_interaction_cells(cells: pd.DataFrame, block: str, metrics: Sequence[str], ncols: Optional[int] = None,
                           max_columns: int = viscons.INTERACTION_CELLS_MAX_COLUMNS,
                           max_panels: int = viscons.INTERACTION_CELLS_MAX_PANELS,
                           width: float = viscons.PAPER_TEXT_WIDTH_IN, title: Optional[str] = None,
                           correction: Optional[str] = cons.HEATMAP_CORRECTION,
                           alpha: float = cons.HEATMAP_ALPHA) -> plt.Figure:
    """Cell deviations of one block for the given metrics (see module docstring). cells: interactions.load_cells()
    (the correction runs over all its cells). Panels share the y axis; one row while panels x cell columns <=
    max_columns (ncols forces the panels per row). Legend: better / worse than the additive prediction (scale per
    panel in its title) + what faded values mean."""
    cells = cells.assign(_sig=mt.significant(cells, 'deviation', correction, alpha, 'two').astype(float),
                         _has_ci=np.isfinite(cells['deviation_std'].to_numpy(float)).astype(float))
    data = cells[(cells['block'] == block) & cells['metric'].isin(metrics)]
    metrics = [m for m in metrics if m in set(data['metric'])]
    if not metrics:
        raise ValueError(f"No cells of block {block!r} for the metrics {list(metrics)}")
    fa, fb = block.split(':')
    rows_lv, cols_lv = _levels(data, fa, 'level_a'), _levels(data, fb, 'level_b')
    models = model_labels(rows_lv) if fa == MODEL_FACTOR else {}
    nrows, ncols = cells_grid(len(metrics), len(cols_lv), max_columns, ncols, max_panels)
    cell_h = 0.16 if len(rows_lv) > 12 else 0.22
    panel_h = cell_h * len(rows_lv) + 0.55
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, panel_h * nrows + 0.3), squeeze=False, sharey=True,
                             layout='constrained')
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.04)
    for ax in axes.ravel()[len(metrics):]:
        ax.axis('off')
    small = len(rows_lv) <= 12 and len(cols_lv) <= 8
    for k, (ax, key) in enumerate(zip(axes.ravel(), metrics)):
        d = data[data['metric'] == key].assign(level_a=lambda f: f['level_a'].astype(str),
                                               level_b=lambda f: f['level_b'].astype(str))
        grid = d.pivot(index='level_a', columns='level_b', values='deviation').reindex(index=rows_lv, columns=cols_lv)
        sig = d.pivot(index='level_a', columns='level_b', values='_sig').reindex(index=rows_lv, columns=cols_lv)
        has_ci = d.pivot(index='level_a', columns='level_b', values='_has_ci').reindex(index=rows_lv, columns=cols_lv)
        largest = float(np.nanmax(np.abs(grid.to_numpy()))) if np.isfinite(grid.to_numpy()).any() else 0.0
        extent = max(largest, ix.sesoi(key))           # floor at the SESOI: deviations below it stay pale
        metric = summary.METRICS.get(key)
        direction = metric.direction if metric else summary.NEUTRAL
        fills = [[to_rgba(deviation_color(v, direction, extent)) for v in row] for row in grid.to_numpy(dtype=float)]
        ax.imshow(np.array(fills), aspect='auto')
        for i in range(len(rows_lv)):
            for j in range(len(cols_lv)):
                v = grid.iat[i, j]
                if not np.isfinite(v) or not small:
                    continue
                faded = has_ci.iat[i, j] == 1 and sig.iat[i, j] != 1
                color = viscons.RQ1_HEATMAP_FADED_TEXT if faded else text_on(fills[i][j])
                ax.text(j, i, signed_cell(v), ha='center', va='center',
                        fontsize=viscons.MIN_FONT_SIZE, color=color)
        ax.set_xticks(range(len(cols_lv)), [_level_text(fb, x, {}) for x in cols_lv], rotation=45, ha='right',
                      fontsize=viscons.MIN_FONT_SIZE)
        ax.set_yticks(range(len(rows_lv)), [_level_text(fa, x, models) for x in rows_lv],
                      fontsize=viscons.MIN_FONT_SIZE)
        ax.tick_params(length=0, pad=1, labelleft=k % ncols == 0)   # shared y: labels on the first panel of a row
        clean_axes(ax, hide='all')
        ax.set_title(f"{metric.label if metric else key}  (±{extent:.2f})", fontsize='x-small', pad=2)
    for ax in axes[:, 0]:
        ax.set_ylabel(variable_title(fa), fontsize='x-small')
    fig.supxlabel(variable_title(fb), fontsize='x-small')   # one x title for all panels
    _cells_legend(fig, faded=bool((data['_has_ci'] == 1).any()), correction=correction, alpha=alpha)
    if title:
        fig.suptitle(title, fontsize='small')
    return fig


def _cells_legend(fig: plt.Figure, faded: bool, correction: Optional[str], alpha: float) -> None:
    """Legend of the cell figures, as the RQ2 heatmap: observed - additive better / worse than the additive prediction
    (the metric's direction), |deviation| for metrics without one; colour full at the panel's limit (± in its title);
    what a faded value means."""
    note = 'observed − additive · colour full at ± the panel limit (in its title)'
    if faded:
        note += f" · faded: {fade_rule(correction, alpha)}"
    fig.legend(handles=deviation_legend_handles('the additive prediction'), loc='outside upper center', ncol=3,
               frameon=False, fontsize=viscons.MIN_FONT_SIZE, handlelength=1.0, handletextpad=0.4, columnspacing=1.2,
               title=note, title_fontsize=viscons.MIN_FONT_SIZE)
