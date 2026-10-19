"""
vis.py — Shared figure style.
"""

import matplotlib as mpl
import seaborn as sns
from matplotlib import rc


FIG_DPI = 600


def sns_paper_style(font_scale=1.51):
    sns.set_context(
        "paper", font_scale=font_scale
    )  # rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5})
    rc("font", family="DejaVu Sans")
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.labelsize"] = 10   # axis label font size
