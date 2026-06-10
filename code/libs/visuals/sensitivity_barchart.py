"""
Sensitivity bar chart.

Horizontal grouped bar chart designed as a companion to
GroupedSensitivityHeatmap. Same color palette, same name mapping,
same DataFrame conventions. Answers the question "which metrics
are most affected?" at a glance — sorted, no rotation, clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from grouped_heatmap import PROMPT_VAR_COLORS, NAME_MAP


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

@dataclass
class SensitivityBarChart:
    """Horizontal grouped bar chart, sorted by max sensitivity descending.

    Required
    --------
    df : pd.DataFrame
        Index = metric names, columns = group names (e.g. 'context', 'persona').
    group_colors : dict[str, str]
        Maps group name -> hex color.

    Display kwargs
    --------------
    figsize, title, xlabel, ylabel : tuple, str|None
    sort_by : 'max' | 'sum' | 'mean' | <column-name>
        How to rank metrics top-to-bottom.
    bar_height : float in (0, 1)
        Total height for the group of bars at one metric.
    show_values : bool
        Annotate bar values at the bar end.
    value_fmt : str
    value_threshold : float
        Hide annotations for values below this (declutter).
    legend_loc : str or None
        Matplotlib loc string, or None to hide.
    """

    df: pd.DataFrame
    group_colors: Dict[str, str]
    name_map: Dict[str, str] = field(default_factory=dict)

    figsize: Tuple[float, float] = (8, 9)
    title: Optional[str] = None
    xlabel: Optional[str] = r'Partial $\eta^2$'
    ylabel: Optional[str] = None

    sort_by: str = 'max'
    bar_height: float = 0.72
    show_values: bool = True
    value_fmt: str = '.3f'
    value_threshold: float = 0.003

    legend_loc: Optional[str] = 'lower right'

    # margins (None = sensible defaults)
    left: Optional[float] = None
    right: Optional[float] = None
    top: Optional[float] = None
    bottom: Optional[float] = None

    # -- helpers -----------------------------------------------------------

    def _pretty(self, name: str) -> str:
        return self.name_map.get(name, name)

    def _sorted(self) -> pd.DataFrame:
        df = self.df.apply(pd.to_numeric, errors='coerce')
        if self.sort_by == 'max':
            key = df.max(axis=1)
        elif self.sort_by == 'sum':
            key = df.sum(axis=1)
        elif self.sort_by == 'mean':
            key = df.mean(axis=1)
        elif self.sort_by in df.columns:
            key = df[self.sort_by]
        else:
            raise ValueError(f"Unknown sort_by: {self.sort_by!r}")
        # ascending so that the largest ends up at the TOP of a horizontal chart
        return df.loc[key.sort_values(ascending=True).index]

    def _resolve_margins(self) -> Tuple[float, float, float, float]:
        left = self.left if self.left is not None else 0.22
        right = self.right if self.right is not None else 0.95
        top = self.top if self.top is not None else (0.93 if self.title else 0.97)
        bottom = self.bottom if self.bottom is not None else (
            0.09 if self.xlabel else 0.04
        )
        return left, right, top, bottom

    # -- main --------------------------------------------------------------

    def draw(self):
        data = self._sorted()
        groups = list(data.columns)
        n_metrics, n_groups = data.shape

        ys = np.arange(n_metrics)
        bar_one = self.bar_height / n_groups
        offsets = (np.arange(n_groups) - (n_groups - 1) / 2) * bar_one

        fig, ax = plt.subplots(figsize=self.figsize, facecolor='white')
        left, right, top, bottom = self._resolve_margins()
        plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

        vmax = float(data.max().max())
        annot_offset = vmax * 0.012

        for g, off in zip(groups, offsets):
            color = self.group_colors.get(g, '#888')
            bars = ax.barh(
                ys + off, data[g].values,
                height=bar_one * 0.92,
                color=color, edgecolor='white', linewidth=0.4,
                label=g.capitalize(),
            )
            if self.show_values:
                for bar, v in zip(bars, data[g].values):
                    if pd.isna(v) or v < self.value_threshold:
                        continue
                    ax.text(
                        v + annot_offset,
                        bar.get_y() + bar.get_height() / 2,
                        format(v, self.value_fmt),
                        ha='left', va='center',
                        fontsize=8, color='#444',
                    )

        # y-axis: metric names
        ax.set_yticks(ys)
        ax.set_yticklabels([self._pretty(m) for m in data.index], fontsize=10)
        ax.tick_params(axis='y', length=0, pad=4)
        ax.set_ylim(-0.6, n_metrics - 0.4)

        # x-axis
        ax.set_xlim(0, vmax * 1.12)  # room for annotations
        ax.tick_params(axis='x', length=0, labelsize=9, colors='#444')
        if self.xlabel:
            ax.set_xlabel(self.xlabel, fontsize=11, color='#555', labelpad=8)
        if self.ylabel:
            ax.set_ylabel(self.ylabel, fontsize=11, color='#555')

        # spines / grid
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_color('#cccccc')
        ax.spines['bottom'].set_linewidth(0.8)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#eeeeee', linewidth=0.8)

        # legend with group-colored labels
        if self.legend_loc:
            legend = ax.legend(loc=self.legend_loc, frameon=False, fontsize=10)
            for txt, g in zip(legend.get_texts(), groups):
                txt.set_color(self.group_colors.get(g, '#444'))
                txt.set_fontweight('semibold')

        if self.title:
            ax.set_title(self.title, fontsize=13, fontweight='semibold',
                         color='#222', loc='left', pad=10)

        return fig


# ---------------------------------------------------------------------------
# Demo — values taken from your image 2 so the preview shows the real figure
# ---------------------------------------------------------------------------

def _make_demo_df() -> pd.DataFrame:
    """Values transcribed from the partial η² heatmap in image 2."""
    rows = {
        'consistency':         (0.017,   0.0095),
        'div_ethnicity':       (0.0058,  0.084),
        'div_gender':          (0.021,   0.012),
        'div_location':        (0.039,   0.12),
        'duplicates':          (0.012,   0.023),
        'factuality_author':   (0.0012,  0.017),
        'factuality_field':    (0.23,    0.0022),
        'factuality_location': (0.065,   0.13),
        'factuality_seniority':(0.15,    0.003),
        'parity_ethnicity':    (0.026,   0.039),
        'parity_gender':       (0.087,   0.0064),
        'pct_high_citations':  (0.0054,  0.025),
        'pct_high_works':      (0.0051,  0.039),
        'pct_low_citations':   (0.00087, 0.03),
        'pct_low_works':       (0.0013,  0.028),
        'pct_med_citations':   (0.0019,  0.0029),
        'pct_med_works':       (0.0011,  0.0057),
        'refusals':            (0.0024,  0.0029),
        'validity':            (0.0052,  0.0021),
    }
    return pd.DataFrame.from_dict(rows, orient='index',
                                  columns=['context', 'persona'])


def main():
    df = _make_demo_df()
    chart = SensitivityBarChart(
        df=df,
        group_colors=PROMPT_VAR_COLORS,
        name_map=NAME_MAP,
        figsize=(8, 9),
        sort_by='max',
    )
    fig = chart.draw()
    fig.savefig('/home/claude/barchart_preview.png', dpi=160,
                facecolor='white')


if __name__ == '__main__':
    main()