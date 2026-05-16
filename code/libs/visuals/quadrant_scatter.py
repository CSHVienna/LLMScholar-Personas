"""
quadrant_scatter.py — color-by-quadrant scatter with non-overlapping labels.
Requires: matplotlib, numpy, pandas, adjustText  (pip install adjustText)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text


# ─── config ────────────────────────────────────────────────────────────────
QUADRANT_COLORS = {
    "Q1": "#2ca02c",  # top-right    — high tech, high social  (green)
    "Q2": "#1f77b4",  # top-left     — low  tech, high social  (blue)
    "Q3": "#a5a5a5",  # bottom-left  — low  tech, low  social  (red)
    "Q4": "#ff7f0e",  # bottom-right — high tech, low  social  (orange)
}


# ─── utils ─────────────────────────────────────────────────────────────────
def assign_quadrants(df, x_col="x_technical", y_col="y_social"):
    """Median-split each axis and tag every row as Q1..Q4."""
    x_med, y_med = df[x_col].median(), df[y_col].median()
    conds = [
        (df[x_col] >= x_med) & (df[y_col] >= y_med),  # Q1
        (df[x_col] <  x_med) & (df[y_col] >= y_med),  # Q2
        (df[x_col] <  x_med) & (df[y_col] <  y_med),  # Q3
        (df[x_col] >= x_med) & (df[y_col] <  y_med),  # Q4
    ]
    labels = np.select(conds, ["Q1", "Q2", "Q3", "Q4"], default="")
    return labels, x_med, y_med


def colors_from_quadrants(quadrants, palette=QUADRANT_COLORS):
    return [palette[q] for q in quadrants]


# ─── visualization ─────────────────────────────────────────────────────────
def plot_quadrant_scatter(
    df,
    x_col="x_technical",
    y_col="y_social",
    label_col="model",
    figsize=(16, 9),
    point_size=55,
    label_fontsize=9,
    curve=0.2,            # 0 = straight, ~0.3 = noticeably curvy
    show_quadrant_legend=True,
):
    quadrants, x_med, y_med = assign_quadrants(df, x_col, y_col)
    colors = colors_from_quadrants(quadrants)

    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    # median crosshair
    ax.axvline(x_med, ls="--", lw=0.8, color="grey", alpha=0.7, zorder=1)
    ax.axhline(y_med, ls="--", lw=0.8, color="grey", alpha=0.7, zorder=1)

    # points
    ax.scatter(
        df[x_col], df[y_col],
        c=colors, s=point_size,
        edgecolor="white", linewidth=0.8, zorder=3,
    )

    # text objects (colored to match the dot)
    texts = [
        ax.text(x, y, str(lbl), fontsize=label_fontsize, color=c, zorder=4)
        for x, y, lbl, c in zip(df[x_col], df[y_col], df[label_col], colors)
    ]

    # de-overlap + curvy leader lines
    adjust_text(
        texts,
        x=df[x_col].values,
        y=df[y_col].values,
        ax=ax,
        expand=(1.3, 1.6),
        force_text=(0.6, 0.9),
        force_points=(0.4, 0.6),
        arrowprops=dict(
            arrowstyle="-",
            color="grey",
            lw=0.5,
            alpha=0.6,
            connectionstyle=f"arc3,rad={curve}",
        ),
    )

    ax.set_xlabel(
        r"Technical: validity + refusals$^c$ + duplicates$^c$ + $\sum$ factuality$_b$  "
        #r"(author, field, seniority, location)"
    )
    ax.set_ylabel(
        r"Social: $\sum$ parity$_a$" # (gender, ethnicity, publications, citations)
    )
    ax.grid(True, alpha=0.25)

    if show_quadrant_legend:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", linestyle="", color=QUADRANT_COLORS[q],
                   label=lbl, markersize=8, markeredgecolor="white")
            for q, lbl in [
                ("Q1", "Q1 · high tech / high social"),
                ("Q2", "Q2 · low tech / high social"),
                ("Q3", "Q3 · low tech / low social"),
                ("Q4", "Q4 · high tech / low social"),
            ]
        ]
        ax.legend(handles=handles, loc="best", frameon=True, framealpha=0.9)

    plt.tight_layout()
    return fig, ax


# ─── usage ─────────────────────────────────────────────────────────────────
# fig, ax = plot_quadrant_scatter(df, label_col="model")
# plt.show()