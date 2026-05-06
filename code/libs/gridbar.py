"""
gridbars.py

Pivot-native grid plotting for bar panels with optional confidence intervals.

Key idea:
- You pass a pivot table `values` with index (Index or MultiIndex) and columns = metric names.
- Optionally pass matching `ci_low` and `ci_high` pivot tables.

Index convention:
- If MultiIndex with >=2 levels: level 0 is treated as the "group" that forms blocks.
- Remaining levels define the within-group rows.
- If single Index: rows are models; no group column is drawn. Row labels appear on the first panel (optional).

This module is intentionally strict:
- It validates alignment of values and CI tables.
- It validates panel metric presence.
"""



from libs.visuals.grid import *
from libs.visuals import grid


# -----------------------------
# Main plotter (bar grid)
# -----------------------------
def plot_metric_grid_from_pivot(
    values: pd.DataFrame,
    *,
    panels: Sequence[PanelSpec],
    ci_low: Optional[pd.DataFrame] = None,
    ci_high: Optional[pd.DataFrame] = None,
    # ordering
    index_order: Optional[Sequence[Hashable]] = None,
    group_order: Optional[Sequence[Hashable]] = None,
    within_group_order: Optional[Mapping[Hashable, Sequence[Hashable]]] = None,
    # labeling
    index_label_maps: Optional[Mapping[int, Mapping[Hashable, str]]] = None,
    group_label_map: Optional[Mapping[Hashable, str]] = None,
    metric_label_map: Optional[Mapping[str, str]] = None,
    # colors
    row_colors: Optional[Sequence] = None,
    group_palette: Optional[Mapping[Hashable, Sequence]] = None,
    row_color_map: Optional[Mapping[Hashable, object]] = None,
    row_color_func: Optional[Callable[[Hashable], object]] = None,
    default_row_color: object = "#CCCCCC",

    # layout/style
    layout: LayoutSpec = LayoutSpec(),
    style: StyleSpec = StyleSpec(),
    # axis overrides
    xlim: Optional[Tuple[Number, Number]] = None,
    xticks: Optional[Sequence[Number]] = None,
    # highlighting
    highlight_rules: Optional[Mapping[str, str]] = None,
    # label placement for single Index
    single_index_as_ylabel: bool = True,
) -> plt.Figure:
    metric_names = grid._ensure_metrics_present(values, panels)

    v = values.loc[:, metric_names].copy()

    have_ci = (ci_low is not None) and (ci_high is not None)
    if have_ci:
        lo = ci_low.loc[:, metric_names].copy()
        hi = ci_high.loc[:, metric_names].copy()
        grid._ensure_same_index_and_columns(v, lo, base_name="values", other_name="ci_low")
        grid._ensure_same_index_and_columns(v, hi, base_name="values", other_name="ci_high")
    else:
        lo = hi = None

    # Ordering
    if index_order is not None:
        v = v.reindex(index_order)
        if have_ci:
            lo = lo.reindex(v.index)
            hi = hi.reindex(v.index)
    else:
        if isinstance(v.index, pd.MultiIndex) and v.index.nlevels >= 2 and group_order is not None:
            blocks = []
            for g in group_order:
                try:
                    block = v.xs(g, level=0, drop_level=False)
                except KeyError:
                    continue
                if within_group_order and (g in within_group_order):
                    block = block.reindex(within_group_order[g])
                blocks.append(block)
            if len(blocks) > 0:
                v = pd.concat(blocks, axis=0)
                if have_ci:
                    lo = lo.reindex(v.index)
                    hi = hi.reindex(v.index)

    idx = v.index
    nlevels = idx.nlevels if isinstance(idx, pd.MultiIndex) else 1

    group_bounds = grid._compute_group_bounds_from_index(idx)

    # Labels
    full_labels = grid._apply_index_label_maps(idx, index_label_maps)

    if nlevels >= 2:
        # within-group label: levels 1..K-1 (mapped)
        row_labels = []
        for s in full_labels:
            parts = s.split(" | ")
            row_labels.append(" | ".join(parts[1:]) if len(parts) > 1 else s)
        # group labels come from level 0
        group_lbls = idx.get_level_values(0).to_list()
    else:
        row_labels = full_labels
        group_lbls = None

    # Colors
    row_colors = grid._row_colors_from_spec(
        idx,
        row_colors=row_colors,
        row_color_map=row_color_map,
        row_color_func=row_color_func,
        group_bounds=group_bounds,
        group_palette=group_palette,
        default_color=default_row_color,
    )

    width_ratios = [layout.label_ratio] + [layout.panel_ratio] * len(panels) if layout.width_ratios is None else layout.width_ratios

    # Build figure grid
    fig, ax_lbl, axes, y = grid._build_grid_from_index(
        nrows=len(v),
        panels=panels,
        group_bounds=group_bounds,
        row_labels=row_labels if nlevels >= 2 else None,
        group_label_map=group_label_map,
        metric_label_map=metric_label_map,
        layout=layout,
        style=style,
        row_labels_for_first_panel=row_labels if (nlevels == 1 and single_index_as_ylabel) else None,
    )

    # Winner highlighting
    # Winner highlighting (ties included): (group, metric) -> set of row indices
    winners: Optional[Dict[Tuple[Hashable, str], set[int]]] = None
    if highlight_rules:
        winners = {}

        def _mark_ties(rows: Sequence[int], vals: np.ndarray, rule: str, tol: float = 1e-3) -> set[int]:
            mask = np.isfinite(vals)
            if not mask.any():
                return set()

            rows_f = np.asarray(rows)[mask]
            vals_f = vals[mask]

            best = np.nanmax(vals_f) if rule == "max" else np.nanmin(vals_f)

            tied = np.isclose(vals_f, best, rtol=0.0, atol=tol)
            return set(map(int, rows_f[tied]))

        if nlevels >= 2:
            for g, s, e in group_bounds:
                rows_g = list(range(s, e + 1))
                for p in panels:
                    rule = highlight_rules.get(p.metric_name)
                    if rule is None:
                        continue

                    vals = v[p.metric_name].iloc[rows_g].to_numpy(dtype=float)
                    tied_rows = _mark_ties(rows_g, vals, rule)

                    if tied_rows:
                        winners[(g, p.metric_name)] = tied_rows
        else:
            rows_all = list(range(len(v)))
            for p in panels:
                rule = highlight_rules.get(p.metric_name)
                if rule is None:
                    continue

                vals = v[p.metric_name].to_numpy(dtype=float)
                tied_rows = _mark_ties(rows_all, vals, rule)

                if tied_rows:
                    winners[("ALL", p.metric_name)] = tied_rows

    # Render panels
    for col_i, (ax, p) in enumerate(zip(axes, panels)):
        x = v[p.metric_name].to_numpy(dtype=float)

        # Axis config
        ax_xlim = grid._resolve_axis_setting(xlim, p.xlim)
        ax_xticks = grid._resolve_axis_setting(xticks, p.xticks)

        ax.set_ylim(-0.5, len(y) - 0.5)
        if style.hide_y_ticks_nonfirstcol and col_i > 0:
            ax.set_yticks([])
        elif nlevels >= 2:
            ax.set_yticks([])

        if ax_xlim is not None:
            ax.set_xlim(*ax_xlim)
        if ax_xticks is not None:
            ax.set_xticks(list(ax_xticks))
            

        # styling spines:
        ax.spines['bottom'].set_linewidth(style.spine_lw) 
        ax.tick_params(axis='x', width=style.spine_lw)

        ax.tick_params(axis="x", labelsize=style.tick_fontsize, labelcolor=style.tick_font_color)

        if style.grid_x:
            ax.grid(True, axis="x", linewidth=0.6, alpha=style.grid_alpha)
        if style.grid_y:
            ax.grid(True, axis="y", linewidth=0.6, alpha=style.grid_alpha)

        grid._apply_axis_cosmetics(ax, style)

        if style.draw_x_vlines:
            # only if xlim exists or axis is numeric, draw at 0 and 1
            ax.axvline(0, linewidth=style.x_vlines_lw, alpha=style.x_vlines_alpha, zorder=0, color=style.x_vlines_color)
            ax.axvline(1, linewidth=style.x_vlines_lw, alpha=style.x_vlines_alpha, zorder=0, color=style.x_vlines_color)

        if style.draw_y_hlines and (not (style.hide_y_ticks_nonfirstcol and col_i > 0)):
            yt = ax.get_yticks()
            for yy in yt:
                ax.axhline(yy, linewidth=style.hline_lw, alpha=style.hline_alpha, linestyle=style.hline_style, zorder=0)

        # Bars
        ax.barh(y, x, height=style.bar_height, color=row_colors, zorder=10)

        # CI
        if p.draw_ci and p.ci_style != "none" and (lo is not None) and (hi is not None):
            ci_l = lo[p.metric_name].to_numpy(dtype=float)
            ci_h = hi[p.metric_name].to_numpy(dtype=float)

            if p.ci_style == "horizontal":
                err = np.vstack([x - ci_l, ci_h - x])
                err = np.where(np.isfinite(err), np.maximum(err, 0.0), np.nan)
                ax.errorbar(
                    x, y, xerr=err,
                    fmt="none",
                    capsize=style.ci_capsize,
                    linewidth=style.ci_lw,
                    zorder=10,
                    color=style.ci_color,
                )
            elif p.ci_style == "vertical":
                # Not typical for barh, but supported if you use y as a numeric scale
                err = np.vstack([y - ci_l, ci_h - y])
                err = np.where(np.isfinite(err), np.maximum(err, 0.0), np.nan)
                ax.errorbar(
                    x, y, yerr=err,
                    fmt="none",
                    capsize=style.ci_capsize,
                    linewidth=style.ci_lw,
                    color=style.ci_color,
                    zorder=10,
                )
            else:
                raise ValueError(f"Invalid ci_style: {p.ci_style}")

        # Value labels (anchored at CI high if available, else at mean)
        xmin, xmax = ax.get_xlim()
        span = float(xmax) - float(xmin)

        if (lo is not None) and (hi is not None) and p.draw_ci and p.ci_style == "horizontal":
            ci_h_for_anchor = hi[p.metric_name].to_numpy(dtype=float)
        else:
            ci_h_for_anchor = None

        for i, (yi, xm) in enumerate(zip(y, x)):
            if not np.isfinite(xm):
                continue

            anchor = xm
            if ci_h_for_anchor is not None and np.isfinite(ci_h_for_anchor[i]):
                anchor = float(ci_h_for_anchor[i])

            if nlevels >= 2:
                g_i = idx[i][0]
                is_winner = (winners is not None) and (i in winners.get((g_i, p.metric_name), set()))
            else:
                is_winner = (winners is not None) and (i in winners.get(("ALL", p.metric_name), set()))

            ax.text(
                anchor + 0.012 * span,
                yi,
                p.value_fmt.format(float(xm)),
                ha="left",
                va="center",
                fontsize=style.value_fontsize,
                color="black",
                fontweight="bold" if is_winner else "normal",
                clip_on=False,
            )

        # x tick labels only on bottom row of panels (you have 1 row, so this is always bottom)
        # Kept for compatibility with your multi-row layout plans.
        if style.show_xticklabels_only_bottom:
            ax.tick_params(axis="x", labelbottom=True)
            ax.xaxis.label.set_color(style.tick_font_color)

    return fig














