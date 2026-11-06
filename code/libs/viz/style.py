"""
One plotting style for all figures (scientific-paper conventions).

* seaborn ``ticks`` style + ``paper`` context, scaled by ``viscons.PLOT_FONT_SCALE``. Figures are drawn
  at their final printed size, so fonts are set in print points: with the defaults, tick labels are
  ~6.6 pt and axis labels / legends ~7 pt (venues usually ask for >= 6-7 pt).
* A serif TrueType font (``viscons.PLOT_FONT_FAMILY``; first one installed from ``viscons.PLOT_FONTS``), also
  for math text ($N_m$), so labels and symbols match.
* Fonts embedded as TrueType (Type 42) in PDF/PS: Type 3 fonts (matplotlib's default) are rejected
  by several submission checkers (e.g. ACM, IEEE PDF eXpress).
* Recessive axes: no top/right spines, thin lines, no legend frame.

Call :func:`apply_style` once (e.g. at the top of a notebook) before plotting.
"""

import matplotlib as mpl
import seaborn as sns
from matplotlib import font_manager

from viz import constants as viscons


def apply_style(context: str = viscons.PLOT_CONTEXT, style: str = viscons.PLOT_STYLE,
                font_scale: float = viscons.PLOT_FONT_SCALE) -> str:
    """Apply the paper style to matplotlib (globally). Returns the font family used."""
    font = first_installed_font(viscons.PLOT_FONTS)
    sns.set_theme(context=context, style=style, font_scale=font_scale, rc={
        # fonts
        'font.family': viscons.PLOT_FONT_FAMILY,
        f'font.{viscons.PLOT_FONT_FAMILY}': [font],
        'mathtext.fontset': 'custom',
        'mathtext.rm': font, 'mathtext.it': f'{font}:italic', 'mathtext.bf': f'{font}:bold',
        'mathtext.sf': font, 'mathtext.cal': font, 'mathtext.tt': font,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        # lines and axes
        'axes.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2, 'ytick.major.size': 2,
        'grid.linewidth': 0.5, 'hatch.linewidth': 0.6,
        'legend.frameon': False,
        # export
        'savefig.dpi': viscons.FIGURE_DPI, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    })
    return font


def first_installed_font(candidates) -> str:
    """First font family of ``candidates`` installed as TrueType (.ttf) on this machine (matplotlib's default otherwise).

    Only TrueType files are considered: they embed cleanly as Type 42 in PDFs (OpenType/CFF ones do not).
    """
    installed = {f.name for f in font_manager.fontManager.ttflist if f.fname.lower().endswith('.ttf')}
    return next((name for name in candidates if name in installed), mpl.rcParams['font.sans-serif'][0])
