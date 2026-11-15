"""
Helpers shared by all figures: saving (figure titles only in PNG), bars, colour shades and axes styling.
All sizes (fonts, lines, export dpi) come from the style: call ``viz.style.apply_style()`` first.
Family-specific plots (e.g. ``viz/tag_plots.py``) build on these.
"""

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from utils import constants as cons
from viz import constants as viscons


def save_figure(fig: plt.Figure, name: str, plots_dir: Path | str = cons.DEFAULT_PLOTS_DIR,
                formats: List[str] = viscons.FIGURE_FORMATS) -> List[Path]:
    """Save ``fig`` as <plots_dir>/<name>.<format> for every format; returns the paths.

    Rule: only the paper's main-text figures (``viscons.PAPER_MAIN_FIGURES``) go to <plots_dir>/; all others go to
    <plots_dir>/si/ (``viscons.SI_SUBDIR``).
    Rule: titles are kept in ``viscons.FORMATS_WITH_TITLES`` (PNG); other formats (PDFs for the paper) drop the
    figure title (suptitle) and, for a figure with a single subplot, that subplot's title. Subplot titles of
    multi-panel figures are kept (they name the panels).
    """
    plots_dir = Path(plots_dir)
    if name not in viscons.PAPER_MAIN_FIGURES:
        plots_dir = plots_dir / viscons.SI_SUBDIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    titles = []
    if getattr(fig, '_suptitle', None) is not None:          # figure-level title (suptitle)
        titles.append(fig._suptitle)
    # panels = axes with data or a title (colorbars and empty / legend-only axes are skipped); matplotlib keeps the centre,
    # left and right titles of an axes as three Text artists
    panels = [ax for ax in fig.axes if getattr(ax, '_colorbar', None) is None
              and (ax.has_data() or any(t.get_text() for t in (ax.title, ax._left_title, ax._right_title)))]
    if len(panels) == 1:
        titles += [panels[0].title, panels[0]._left_title, panels[0]._right_title]
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


def ramp(colors: Sequence[str], n: int, single: float = 0.0) -> List[str]:
    """``n`` colours evenly spaced along the ramp through ``colors`` (first -> last); a single colour sits at
    ``single`` (0 = the first colour, 0.5 = the middle, 1 = the last). The one ramp rule of every figure."""
    if n == 0:
        return []
    cmap = LinearSegmentedColormap.from_list('ramp', list(colors))
    return [to_hex(cmap(x)) for x in (np.linspace(0, 1, n) if n > 1 else [single])]


def family_cmap(family: str) -> LinearSegmentedColormap:
    """Near-white tint -> base -> dark of a row family (heatmaps: a zero cell is almost white, the hue just visible)."""
    return LinearSegmentedColormap.from_list(family, [viscons.FAMILY_ZERO[family], viscons.FAMILY_COLORS[family],
                                                      viscons.FAMILY_DARK[family]])


def model_attribute_colors(levels) -> dict:
    """Value of a model characteristic -> colour: MODEL_ATTRIBUTE_RAMP light -> dark in the given order (one value:
    the dark end). The same values get the same colours in every figure (metric bars, RQ3 inset)."""
    levels = list(levels)
    return dict(zip(levels, ramp(viscons.MODEL_ATTRIBUTE_RAMP, len(levels), single=1.0)))


def _luminance(color) -> float:
    """WCAG relative luminance."""
    channels = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(color)]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def text_on(color) -> str:
    """Text colour for a value written on a filled cell / segment: white or viscons.COLOR_TEXT, whichever contrasts
    more with the fill (WCAG). The one rule of every heatmap and stacked bar."""
    fill = _luminance(color)
    on_white = (1.05) / (fill + 0.05)
    on_dark = (fill + 0.05) / (_luminance(viscons.COLOR_TEXT) + 0.05)
    return 'white' if on_white > on_dark else viscons.COLOR_TEXT


def text_size(text: str, size: str, weight: str = 'normal') -> tuple:
    """(width, height) in inches of a text at a matplotlib font size (with a small margin)."""
    extents = TextPath((0, 0), text, prop=FontProperties(size=size, weight=weight)).get_extents()
    return 1.12 * extents.width / 72, 1.12 * extents.height / 72


def ci(ax: plt.Axes, values, positions, low, high, vertical: bool = False, color: Optional[str] = None,
       scale: float = 1.0, **style) -> None:
    """The one confidence-interval style of every figure (viscons.CI_STYLE): a bar from low to high around each value
    (along y if vertical, else x), drawn where all three are finite. color: the bar colour (default CI_COLOR); scale:
    cap size factor (smaller marks); style: other errorbar arguments (e.g. zorder)."""
    values, positions = np.asarray(values, dtype=float), np.asarray(positions, dtype=float)
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    ok = np.isfinite(values) & np.isfinite(low) & np.isfinite(high)
    if not ok.any():
        return
    err = [np.clip(values[ok] - low[ok], 0, None), np.clip(high[ok] - values[ok], 0, None)]
    style = {**viscons.CI_STYLE, 'capsize': viscons.CI_CAPSIZE * scale, **({'ecolor': color} if color else {}), **style}
    if vertical:
        ax.errorbar(positions[ok], values[ok], yerr=err, **style)
    else:
        ax.errorbar(values[ok], positions[ok], xerr=err, **style)


def panel_grid(n_panels: int, ncols: int) -> tuple:
    """(nrows, ncols) of a grid of n panels with at most ncols per row."""
    ncols = max(1, min(ncols, n_panels))
    return int(np.ceil(n_panels / ncols)), ncols


def hide_unused(axes, n_used: int) -> None:
    """Hide the empty axes after the first n_used of a panel grid."""
    for ax in np.asarray(axes).flat[n_used:]:
        ax.set_visible(False)


def clean_axes(ax: plt.Axes, grid: Optional[str] = None, hide: Sequence[str] = ('top', 'right'),
               **grid_style) -> None:
    """The one axes clean-up of every figure: hide the given spines (hide='all': every spine) and, with grid='x' /
    'y' / 'both', a light grid on that axis behind the data (grid_style: e.g. linewidth)."""
    for side in (list(ax.spines) if hide == 'all' else hide):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=viscons.COLOR_GRID, **grid_style)
        ax.set_axisbelow(True)


def style_bar_axes(ax: plt.Axes, value_label: str, vertical: bool,
                   title: Optional[str] = None, value_axis_label: bool = True) -> None:
    """Grid on the value axis only and no spine on the category baseline (sizes come from the style)."""
    value_axis = 'y' if vertical else 'x'
    if value_axis_label:
        (ax.set_ylabel if vertical else ax.set_xlabel)(value_label)
    ax.grid(False)
    clean_axes(ax, value_axis, hide=('top', 'right', 'bottom' if vertical else 'left'))
    if title:
        ax.set_title(title, loc='left')
