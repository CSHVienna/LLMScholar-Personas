"""
quadrant_scatter.py — color-by-quadrant scatter with non-overlapping labels.
Requires: matplotlib, numpy, pandas
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from libs.visuals.constants import QUADRANT_COLORS
except ImportError:  # PYTHONPATH=code/libs/
    from visuals.constants import QUADRANT_COLORS


# ─── utils ─────────────────────────────────────────────────────────────────
def assign_quadrants(df, x_col="x_technical", y_col="y_social", x_split=None, y_split=None):
    """Split each axis (at the median, unless a split value is given) and tag every row as Q1..Q4."""
    x_med = df[x_col].median() if x_split is None else x_split
    y_med = df[y_col].median() if y_split is None else y_split
    conds = [
        (df[x_col] >= x_med) & (df[y_col] >= y_med),  # Q1
        (df[x_col] < x_med) & (df[y_col] >= y_med),  # Q2
        (df[x_col] < x_med) & (df[y_col] < y_med),  # Q3
        (df[x_col] >= x_med) & (df[y_col] < y_med),  # Q4
    ]
    labels = np.select(conds, ["Q1", "Q2", "Q3", "Q4"], default="")
    return labels, x_med, y_med


def colors_from_quadrants(quadrants, palette=QUADRANT_COLORS):
    # A row with NaN in either axis falls through np.select to "" (no quadrant);
    # fall back to neutral grey instead of KeyError-ing on the missing key.
    return [palette.get(q, "#6B7280") for q in quadrants]



def _place_labels(sizes, pts, axes_box, obstacles, dot_r, regions=None, gap=3.5, step=5.0, max_dist=400.0):
    """Greedy, deterministic label placement in pixel space.

    Each label tries rings of growing distance around its dot (16 directions per
    ring) and takes the first spot that stays inside its region and clears every
    already-placed label, every dot and the extra `obstacles` (e.g. the legend).
    `regions` optionally gives one (x0, y0, x1, y1) box per label (e.g. its
    quadrant); a label that does not fit there falls back to the whole axes.
    Crowded points are placed first so they get the slots closest to them.
    If the rings find nothing, the region is scanned on a grid (nearest first).
    Returns label centres (n, 2) in pixels; falls back to the dot itself if no
    free spot exists at all.
    """
    pts = np.asarray(pts, float)
    dots = [(x - dot_r, y - dot_r, x + dot_r, y + dot_r) for x, y in pts]
    placed = list(obstacles)

    def hits(box, boxes):
        x0, y0, x1, y1 = box
        return any(
            x0 < b[2] + gap and b[0] - gap < x1 and y0 < b[3] + gap and b[1] - gap < y1
            for b in boxes
        )

    def inside(box, region):
        return (
            box[0] >= region[0] and box[2] <= region[2]
            and box[1] >= region[1] and box[3] <= region[3]
        )

    # right, left, up, down first (most readable), then the diagonals and in-betweens
    angles = np.deg2rad([0, 180, 90, 270, 45, 135, 315, 225, 22.5, 157.5, 337.5, 202.5, 67.5, 112.5, 292.5, 247.5])
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])

    def find_spot(i, region):
        (w, h), (px, py) = sizes[i], pts[i]
        others = dots[:i] + dots[i + 1 :]
        for dist in np.arange(0, max_dist, step):
            for ux, uy in dirs:
                # offset so the box edge (not its centre) sits `dist` beyond the dot
                cx = px + ux * (w / 2 + dot_r + gap + dist)
                cy = py + uy * (h / 2 + dot_r + gap + dist)
                box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                if inside(box, region) and not hits(box, placed) and not hits(box, others):
                    return (cx, cy), box
        # rings only sample 16 directions; before giving up, scan the whole
        # region on a fine grid, nearest candidates first
        gx = np.arange(region[0] + w / 2, region[2] - w / 2 + 1e-9, step)
        gy = np.arange(region[1] + h / 2, region[3] - h / 2 + 1e-9, step)
        if len(gx) and len(gy):
            cand = np.array(np.meshgrid(gx, gy)).reshape(2, -1).T
            cand = cand[np.argsort(((cand - (px, py)) ** 2).sum(1))]
            for cx, cy in cand:
                box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                if not hits(box, placed) and not hits(box, others):
                    return (cx, cy), box
        return None

    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    crowding = (d2 < (8 * dot_r) ** 2).sum(1)
    order = np.argsort(-crowding, kind="stable")

    centres = pts.copy()
    for i in order:
        spot = find_spot(i, regions[i]) if regions is not None else None
        if spot is None:
            spot = find_spot(i, axes_box)
        if spot is not None:
            centres[i], box = spot
            placed.append(box)
    return centres


# ─── visualization ─────────────────────────────────────────────────────────
def plot_quadrant_scatter(
    df,
    x_col="x_technical",
    y_col="y_social",
    label_col="model",
    figsize=(16, 9),
    point_size=55,
    label_fontsize=9,
    curve=0.2,  # 0 = straight, ~0.3 = noticeably curvy
    show_quadrant_legend=True,
    legend_loc="lower right",
    x_headroom=2.0,  # right padding, in multiples of the default 4% pad
    y_headroom=6.0,  # top padding — room for labels above the dense cluster
    x_footroom=2.0,  # left padding, in multiples of the default 4% pad
    y_footroom=2.5,  # bottom padding — keeps low outliers' labels off the axis
    labels_in_quadrant=False,  # keep each label inside its dot's quadrant (falls back to anywhere if it can't fit)
    split="median",  # crosshair position: "median" of the data, or "midpoint" of x_range / y_range
    x_range=None,  # (min, max) attainable x, e.g. (0, 7); axes then span the full range
    y_range=None,  # (min, max) attainable y, e.g. (0, 4)
):
    if split not in ("median", "midpoint"):
        raise ValueError(f"split must be 'median' or 'midpoint', got {split!r}")
    if split == "midpoint" and (x_range is None or y_range is None):
        raise ValueError("split='midpoint' needs both x_range and y_range")
    x_split = sum(x_range) / 2 if split == "midpoint" else None
    y_split = sum(y_range) / 2 if split == "midpoint" else None

    quadrants, x_med, y_med = assign_quadrants(df, x_col, y_col, x_split, y_split)
    colors = colors_from_quadrants(quadrants)

    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    # quadrant crosshair (median or range midpoint)
    ax.axvline(x_med, ls="--", lw=0.8, color="grey", alpha=0.7, zorder=1)
    ax.axhline(y_med, ls="--", lw=0.8, color="grey", alpha=0.7, zorder=1)

    # points
    ax.scatter(
        df[x_col],
        df[y_col],
        c=colors,
        s=point_size,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    # headroom so labels of the dense top cluster have somewhere to go
    # (labels are kept inside the axes); with a given range the axes span all of
    # it, so a midpoint crosshair sits in the visual middle
    x_lo, x_hi = x_range if x_range is not None else (df[x_col].min(), df[x_col].max())
    y_lo, y_hi = y_range if y_range is not None else (df[y_col].min(), df[y_col].max())
    x_pad, y_pad = 0.04 * (x_hi - x_lo), 0.04 * (y_hi - y_lo)
    ax.set_xlim(x_lo - x_pad * x_footroom, x_hi + x_pad * x_headroom)
    ax.set_ylim(y_lo - y_pad * y_footroom, y_hi + y_pad * y_headroom)

    # legend first, so labels can be steered away from it
    legend = None
    if show_quadrant_legend:
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=QUADRANT_COLORS[q],
                label=lbl,
                markersize=8,
                markeredgecolor="white",
            )
            for q, lbl in [
                ("Q1", "Q1 · high tech / high social"),
                ("Q2", "Q2 · low tech / high social"),
                ("Q3", "Q3 · low tech / low social"),
                ("Q4", "Q4 · high tech / low social"),
            ]
        ]
        legend = ax.legend(
            handles=handles,
            loc=legend_loc,
            frameon=True,
            framealpha=0.9,
            handlelength=0.5,  # was 2.0 — collapses the handle area to just the dot
            handletextpad=0.6,  # was 0.8 — tightens dot ↔ text gap
        )

    ax.set_xlabel(
        r"Technical: validity + refusals$^c$ + duplicates$^c$ + $\sum$ factuality$_b$  "
        # r"(author, field, seniority, location)"
    )
    ax.set_ylabel(
        r"Social: $\sum$ parity$_a$"  # (gender, ethnicity, publications, citations)
    )
    ax.grid(True, alpha=0.25)

    # finalise the axes size *before* placing labels — tight_layout afterwards
    # would rescale the data→pixel mapping and re-introduce overlaps
    plt.tight_layout()
    fig.canvas.draw()  # label placement needs final text/legend extents
    renderer = fig.canvas.get_renderer()

    xs, ys = df[x_col].values, df[y_col].values
    pts_px = ax.transData.transform(np.column_stack([xs, ys]))
    dot_radius_px = np.sqrt(point_size) / 2 * fig.dpi / 72

    # measure every label once, then place them in pixel space
    labels = [str(lbl) for lbl in df[label_col]]
    probes = [ax.text(0, 0, lbl, fontsize=label_fontsize) for lbl in labels]
    sizes = [(b.width, b.height) for b in (t.get_window_extent(renderer) for t in probes)]
    for t in probes:
        t.remove()

    ab = ax.get_window_extent(renderer)
    obstacles = [tuple(legend.get_window_extent(renderer).extents)] if legend is not None else []
    regions = None
    if labels_in_quadrant:
        # quadrant boxes in pixels, split at the median crosshair
        # (a dot on the median belongs to the >= side, as in assign_quadrants)
        mx, my = ax.transData.transform((x_med, y_med))
        regions = [
            (
                mx if px >= mx else ab.x0,
                my if py >= my else ab.y0,
                ab.x1 if px >= mx else mx,
                ab.y1 if py >= my else my,
            )
            for px, py in pts_px
        ]
    centres_px = _place_labels(
        sizes, pts_px, tuple(ab.extents), obstacles, dot_radius_px, regions=regions
    )
    centres = ax.transData.inverted().transform(centres_px)

    # labels colored to match the dot; far-away ones get a curvy leader line
    # (matplotlib clips it at the text's bounding box, so it never strikes through)
    for lbl, c, (x, y), (tx, ty), (w, _), (cpx, cpy), (ppx, ppy) in zip(
        labels, colors, zip(xs, ys), centres, sizes, centres_px, pts_px
    ):
        near = (
            abs(cpx - ppx) < w / 2 + 3 * dot_radius_px
            and abs(cpy - ppy) < 3 * dot_radius_px + label_fontsize
        )
        ax.annotate(
            lbl,
            xy=(x, y),
            xytext=(tx, ty),
            fontsize=label_fontsize,
            color=c,
            ha="center",
            va="center",
            zorder=4,
            arrowprops=None if near else dict(
                arrowstyle="-",
                color="grey",
                lw=0.5,
                alpha=0.6,
                shrinkA=2,
                shrinkB=dot_radius_px * 72 / fig.dpi + 1,
                connectionstyle=f"arc3,rad={curve}",
            ),
        )

    return fig, ax


# ─── usage ─────────────────────────────────────────────────────────────────
# fig, ax = plot_quadrant_scatter(df, label_col="model")
# plt.show()
