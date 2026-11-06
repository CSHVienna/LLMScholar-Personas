"""
Helpers shared by all figures: saving (titles only in PNG), bars, colour shades and axes styling.
All sizes (fonts, lines, export dpi) come from the style: call ``viz.style.apply_style()`` first.
Family-specific plots (e.g. ``viz/tag_plots.py``) build on these.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex

from utils import constants as cons
from viz import constants as viscons


def save_figure(fig: plt.Figure, name: str, plots_dir: Path | str = cons.DEFAULT_PLOTS_DIR,
                formats: List[str] = viscons.FIGURE_FORMATS) -> List[Path]:
    """Save ``fig`` as <plots_dir>/<name>.<format> for every format; returns the paths.

    Rule: titles are kept only in ``viscons.FORMATS_WITH_TITLES`` (PNG); e.g. PDFs for the paper have none.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    # matplotlib keeps the centre, left and right titles of an axes as three Text artists
    titles = [t for ax in fig.axes for t in (ax.title, ax._left_title, ax._right_title)]
    if getattr(fig, '_suptitle', None) is not None:          # figure-level titles (suptitle) too
        titles.append(fig._suptitle)
    paths = []
    for fmt in formats:
        visible = fmt in viscons.FORMATS_WITH_TITLES
        for title in titles:
            title.set_visible(visible)
        path = plots_dir / f"{name}.{fmt}"
        fig.savefig(path)  # dpi, bbox and font embedding come from the style (viz/style.py)
        paths.append(path)
    for title in titles:
        title.set_visible(True)
    return paths


def bars(ax: plt.Axes, vertical: bool, positions, lengths, start, width: float, color: str,
         hatch: str = '') -> None:
    """Bars along the value axis (y if vertical, x otherwise), separated by thin surface-coloured edges."""
    style = dict(color=color, hatch=hatch, edgecolor=viscons.COLOR_BAR_EDGE, linewidth=0.4)
    if vertical:
        ax.bar(positions, lengths, bottom=start, width=width, **style)
    else:
        ax.barh(positions, lengths, left=start, height=width, **style)


def shades(ramp: List[str], n: int) -> List[str]:
    """``n`` colours from a dark->light ramp (evenly spaced; a single one takes the darkest)."""
    if n == 0:
        return []
    cmap = LinearSegmentedColormap.from_list('ramp', ramp)
    return [to_hex(cmap(x)) for x in (np.linspace(0, 1, n) if n > 1 else [0.0])]


def style_bar_axes(ax: plt.Axes, value_label: str, vertical: bool,
                   title: Optional[str] = None, value_axis_label: bool = True) -> None:
    """Grid on the value axis only and no spine on the category baseline (sizes come from the style)."""
    value_axis = 'y' if vertical else 'x'
    if value_axis_label:
        (ax.set_ylabel if vertical else ax.set_xlabel)(value_label)
    ax.grid(False)
    ax.grid(axis=value_axis, color=viscons.COLOR_GRID)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'bottom' if vertical else 'left'):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, loc='left')
