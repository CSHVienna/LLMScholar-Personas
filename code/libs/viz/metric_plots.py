"""
Bar figures of the benchmark metrics: rows = models, model characteristics or prompt variables (metrics/summary.py
RowGroup), columns = metrics. One horizontal bar per row and metric, its confidence interval at the bar end, the
value next to it (bold = best within its group of rows, following the metric's direction).

    configs = summary.with_model_attributes(summary.configuration_table(metrics_dir))
    rows, metrics = summary.attribute_rows(), summary.select_metrics('all')
    fig = plot_metric_bars(summary.estimate(configs, rows, metrics), rows, metrics)
"""

from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.lines import Line2D

from metrics.summary import HIGHER, LOWER, Metric, RowGroup
from viz import constants as viscons


def format_value(value: float, decimals: int, signed: bool = False) -> str:
    """Rounded value; tiny non-zero values as '<0.01' ('±<0.01' when signed: deviations get an explicit sign)."""
    if value is None or np.isnan(value):
        return '–'
    tiny = 10 ** -decimals
    if 0 < abs(value) < tiny:
        return f"{('+' if value > 0 else '−') if signed else ''}<{tiny:.{decimals}f}"
    return f"{value:+.{decimals}f}".replace('-', '−') if signed else f"{value:.{decimals}f}"


def nice_upper(value: float) -> float:
    """Axis end for data-scaled panels (metric.limits = None): the next 0.05 step above the data (at least 0.05)."""
    if value is None or not np.isfinite(value) or value <= 0:
        return 0.05
    return float(np.ceil(value * 1.05 / 0.05) * 0.05)


def best_rows(values: pd.Series, direction: int, decimals: int = 2) -> pd.Series:
    """True for the best row(s) of a group, judged on the DISPLAYED (rounded) values: ties are all bold, and nothing
    is bold if every row ties, for NEUTRAL metrics, or for a single row."""
    shown = values.round(decimals)
    if direction not in (HIGHER, LOWER) or shown.notna().sum() < 2 or shown.nunique() < 2:
        return pd.Series(False, index=values.index)
    best = shown.max() if direction == HIGHER else shown.min()
    return (shown == best) & shown.notna()


def _layout(rows: Sequence[RowGroup]):
    """y position of every row (top = 0, growing downwards) and the y of the rules between groups."""
    y, positions, rules = 0.0, [], []
    for i, group in enumerate(rows):
        if i:
            rules.append(y - 0.5 + (viscons.METRIC_GROUP_GAP - 1) / 2 + 0.5)
            y += viscons.METRIC_GROUP_GAP
        for _ in group.levels:
            positions.append(y)
            y += 1
    return np.array(positions), rules, y


def row_colors(rows: Sequence[RowGroup]) -> List[str]:
    """One colour per row: a light -> dark ramp across the values of a model characteristic, a distinct colour per
    model family ('family:<i>'), else the flat colour of the group (models grey, persona orange, context blue)."""
    colors = []
    for group in rows:
        key, n = group.color_key, len(group.levels)
        if key in viscons.METRIC_ROW_RAMPS:
            ramp = LinearSegmentedColormap.from_list(key, list(viscons.METRIC_ROW_RAMPS[key]))
            colors += [to_hex(ramp(x)) for x in (np.linspace(0, 1, n) if n > 1 else [0.7])]
        elif key.startswith('family:'):
            palette = viscons.METRIC_FAMILY_PALETTE
            colors += [palette[int(key.split(':')[1]) % len(palette)]] * n
        else:
            colors += [viscons.METRIC_BAR_COLORS.get(key, viscons.METRIC_BAR_COLORS['model'])] * n
    return colors


def plot_metric_bars(table: pd.DataFrame, rows: Sequence[RowGroup], metrics: Sequence[Metric],
                     title: Optional[str] = None, width: float = viscons.PAPER_TEXT_WIDTH_IN,
                     label_width: Optional[float] = None, arrows: bool = True,
                     ci_linewidth: float = viscons.METRIC_CI_LINEWIDTH,
                     ci_capsize: float = viscons.METRIC_CI_CAPSIZE, tick_size: Optional[str] = None) -> plt.Figure:
    """One panel per metric; rows in the order of `rows` (estimate() output in `table`). arrows: the metric's
    direction (higher / lower is better) next to its name. Rows of the persona / context families get their labels in
    the family colour (viscons.FAMILY_COLORS), as in the RQ heatmaps."""
    positions, rules, height = _layout(rows)
    has_titles = any(g.title for g in rows)
    title_width = max((text_size(g.title, 'medium', 'bold')[0] for g in rows if g.title), default=0.0)
    labels_width = max(text_size(label, 'medium')[0] for g in rows for label in g.labels)
    # label area: [margin][group titles, right-aligned][gap][row labels, left-aligned][gap][panels]
    title_right = viscons.METRIC_LABEL_MARGIN_IN + title_width
    labels_left = title_right + (viscons.METRIC_TITLE_GAP_IN if has_titles else 0.0)
    label_width = label_width or (labels_left + labels_width + viscons.METRIC_LABEL_GAP_IN)
    fig_height = 0.45 + viscons.METRIC_ROW_HEIGHT_IN * height
    fig, axes = plt.subplots(1, len(metrics), figsize=(width, fig_height), sharey=True, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=label_width / width, right=0.985, top=1 - 0.28 / fig_height, bottom=0.25 / fig_height,
                        wspace=0.12)
    row_keys = [(g.title, level) for g in rows for level in g.levels]
    colors = row_colors(rows)
    indexed = table.set_index(['group', 'row', 'metric'])
    compact = len(metrics) > viscons.METRIC_COMPACT_FROM          # many narrow panels: smaller text
    header_size = viscons.METRIC_HEADER_FONT_SIZE_COMPACT if compact else viscons.METRIC_HEADER_FONT_SIZE
    value_size = viscons.VALUE_FONT_SIZE_COMPACT if compact else viscons.VALUE_FONT_SIZE

    for ax, metric in zip(axes, metrics):
        values = pd.DataFrame([indexed.loc[(group, row, metric.key), ['mean', 'ci_low', 'ci_high']]
                               if (group, row, metric.key) in indexed.index else
                               pd.Series({'mean': np.nan, 'ci_low': np.nan, 'ci_high': np.nan})
                               for group, row in row_keys]).reset_index(drop=True)
        groups = pd.Series([group for group, _ in row_keys])
        bold = pd.concat([best_rows(values['mean'][groups == g], metric.direction, metric.decimals)
                          for g in groups.unique()]).sort_index()
        if metric.limits:
            low_limit, high_limit = metric.limits
        elif values[['mean', 'ci_low']].min().min() < 0:      # signed values (e.g. deviations): symmetric around 0
            high_limit = nice_upper(values[['mean', 'ci_low', 'ci_high']].abs().max().max())
            low_limit = -high_limit
        else:
            low_limit, high_limit = 0.0, nice_upper(values[['mean', 'ci_high']].max().max())
        if low_limit < 0 < high_limit:
            ax.axvline(0, color=viscons.METRIC_GROUP_RULE_COLOR, linewidth=0.6, zorder=1)
        ax.barh(positions, values['mean'].fillna(0), height=0.78, color=colors, linewidth=0)
        has_ci = values[['mean', 'ci_low', 'ci_high']].notna().all(axis=1).to_numpy()
        if has_ci.any():
            v = values[has_ci]
            ax.errorbar(v['mean'], positions[has_ci], xerr=[(v['mean'] - v['ci_low']).clip(lower=0),
                                                            (v['ci_high'] - v['mean']).clip(lower=0)],
                        fmt='none', ecolor=viscons.METRIC_CI_COLOR, elinewidth=ci_linewidth,
                        capsize=ci_capsize, capthick=ci_linewidth)
        span = high_limit - low_limit
        for y, (mean, low, high), is_bold in zip(positions, values[['mean', 'ci_low', 'ci_high']].to_numpy(), bold):
            if not np.isnan(mean) and mean < 0:                            # negative bar: value on its left end
                end = np.nanmin([mean, low])
                if high_limit - end > viscons.METRIC_VALUE_INSIDE_FROM * span:   # no room on the left: right of 0
                    x, ha = np.nanmax([0.0, high]) + 0.015 * span, 'left'
                else:
                    x, ha = end - 0.015 * span, 'right'
            else:
                end = np.nanmax([mean, high]) if not np.isnan(mean) else max(low_limit, 0.0)
                if end - low_limit > viscons.METRIC_VALUE_INSIDE_FROM * span:     # no room on the right: inside the bar
                    x, ha = np.nanmin([mean, low]) - 0.015 * span, 'right'
                else:
                    x, ha = end + 0.015 * span, 'left'
            ax.text(x, y, format_value(mean, metric.decimals, signed=low_limit < 0), va='center', ha=ha,
                    fontsize=value_size, fontweight='bold' if is_bold else 'normal')
        ax.set_xlim(low_limit, high_limit)
        ax.set_xticks([low_limit, (low_limit + high_limit) / 2, high_limit])
        ax.set_xticklabels([f"{t:g}" for t in ax.get_xticks()], color=viscons.COLOR_TEXT_SECONDARY,
                           **({'fontsize': tick_size} if tick_size else {}))
        if low_limit < 0:                                     # signed panel: end labels inwards (no clash between panels)
            ticks = ax.get_xticklabels()
            ticks[0].set_horizontalalignment('left')
            ticks[-1].set_horizontalalignment('right')
        ax.grid(axis='x', color=viscons.COLOR_GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.set_title(metric.header(len(metrics)) + (viscons.METRIC_ARROWS[metric.direction] if arrows else ''),
                     fontweight='bold',
                     fontsize=header_size)

    axes[0].set_ylim(height - 0.5, -0.5)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([])
    _row_labels(fig, axes[0], rows, positions, title_right / width if has_titles else None, labels_left / width)
    _rules(fig, axes[0], rules)
    if title:
        fig.suptitle(title, y=1.0, fontsize='medium')
    return fig


def text_size(text: str, size: str, weight: str = 'normal') -> tuple:
    """(width, height) in inches of a text at a matplotlib font size (with a small margin)."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    extents = TextPath((0, 0), text, prop=FontProperties(size=size, weight=weight)).get_extents()
    return 1.12 * extents.width / 72, 1.12 * extents.height / 72


def _row_labels(fig: plt.Figure, ax: plt.Axes, rows: Sequence[RowGroup], positions: np.ndarray,
               title_right: Optional[float], labels_left: float) -> None:
    """Group titles (bold, right-aligned at title_right) and row labels (left-aligned at labels_left), x in figure
    fraction, y in data: two tight columns instead of a title at the far left and labels at the axis."""
    from matplotlib.transforms import blended_transform_factory
    transform = blended_transform_factory(fig.transFigure, ax.transData)
    start = 0
    for group in rows:
        ys = positions[start:start + len(group.levels)]
        start += len(group.levels)
        color = viscons.FAMILY_COLORS.get(group.color_key) if group.color_key in ('persona', 'context') else None
        if title_right is not None and group.title:
            ax.text(title_right, ys.mean(), group.title, transform=transform, ha='right', va='center',
                    fontweight='bold', clip_on=False, color=color or viscons.METRIC_ROW_LABEL_COLOR)
        for y, label in zip(ys, group.labels):
            ax.text(labels_left, y, label, transform=transform, ha='left', va='center', clip_on=False,
                    color=color or viscons.METRIC_ROW_LABEL_COLOR)


def _rules(fig: plt.Figure, ax: plt.Axes, rules: Sequence[float]) -> None:
    """One continuous rule per group boundary, from the left to the right edge of the figure (also across the gaps
    between the panels)."""
    from matplotlib.transforms import blended_transform_factory
    transform = blended_transform_factory(fig.transFigure, ax.transData)
    for rule in rules:
        fig.add_artist(Line2D([viscons.METRIC_RULE_INSET, 1 - viscons.METRIC_RULE_INSET], [rule, rule],
                              transform=transform, color=viscons.METRIC_GROUP_RULE_COLOR, linewidth=0.6,
                              clip_on=False))
