"""
legacy/unused_functions.py — Small functions that nothing calls any more,
moved out of their modules. Kept for reference only; the imports they need
are those of the module named above each one.
"""


# ── from libs/visuals/grouped_metrics.py ─────────────────────────────

def plot_grouped_metrics_stacked(
    panels: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
    metric_directions: Optional[Mapping[str, Optional[str]]] = None,
    metric_labels: Optional[Mapping[str, str]] = None,
    row_labels: Optional[Mapping[str, str]] = None,
    figsize: Optional[tuple] = None,
    save_path: Optional[str] = None,
    tick_font_size: int = TICK_FONT_SIZE,
    tick_font_color: str = TICK_FONT_COLOR,
    label_font_size: int = LABEL_FONT_SIZE,
    spine_lw: float = SPINE_LW,
    fig_dpi: int = FIG_DPI,
    panel_width: float = 1.6,
    show: bool = True,
    textwrap_width: int = 11,
    section_gap: float = SECTION_GAP,
    panel_label_width: float = 0.35,
    panel_label_font_size: Optional[int] = None,
    panel_label_rotation: float = 90,
    panel_label_bold: bool = True,
    hspace: float = 0.25,
    section_style: str = "bracket",
    section_sep_style: str = "solid",
    show_mid_gridline: bool = True,
    row_label_width: Optional[float] = None,
    section_label_width: Optional[float] = None,
    section_label_pad: Optional[float] = None,
    section_label_align: Optional[str] = None,
    row_label_align: str = "right",
    row_label_pad: float = 0.05,
    section_sep_left_pad: float = 0.0,
) -> plt.Figure:
    """
    Stack multiple grouped-metric blocks vertically into one figure, sharing
    the x-axis across columns.

    Re-uses ``plot_grouped_metrics``'s rendering primitives (sections, label
    column, metric bars), so styling stays identical. Use when you want to
    slice the same metrics by several different grouping variables (e.g. k,
    model, field) side-by-side in one figure — the second arg of
    ``aggregate_scores`` (and so the ``column`` of each ``group_configs``)
    is what varies between panels.

    Parameters
    ----------
    panels : list of dict
        Each dict defines one stacked block (row of metric panels). Keys:
          - 'df' (DataFrame, required): pre-aggregated table
            (e.g. ``aggregate_scores(df_valid_calls, ['k'])``).
          - 'group_configs' (list of dict, required): same shape as in
            ``plot_grouped_metrics``.
          - 'panel_label' (str, optional): row label displayed at the far
            left, analogous to a y-axis label for the entire block (e.g. ``'K'``).
    metrics : list of str
        Metric column names; consistent across all panels.
    row_labels : dict, optional
        Global ``{df_value: display}`` mapping applied per panel via the same
        remap mechanism as ``plot_grouped_metrics_labeled``. Per-panel /
        per-config overrides (``column_labels``, ``label_labels``) inside each
        ``group_configs`` entry are also honored.
    panel_label_width : float
        Width (inches) of the leftmost column reserved for panel labels.
    panel_label_font_size : int, optional
        Defaults to ``label_font_size``.
    panel_label_rotation : float
        Rotation (degrees) for panel-label text. Default 90 (vertical).
        Per-panel override: ``panel['panel_label_rotation']``.
    panel_label_bold : bool
        Whether panel labels are bold. Default True.
        Per-panel override: ``panel['panel_label_bold']``.
    hspace : float
        Inter-row spacing as a fraction of average row height
        (passed to gridspec).
    section_style : {'bracket', 'plain'}, default 'bracket'
        Label-column chrome. 'bracket' = section label + vertical bracket
        connecting the rows. 'plain' = left-aligned section label, no bracket,
        tighter horizontal footprint.
    section_sep_style : {'solid', 'dashed', 'none'}, default 'solid'
        Separator between adjacent sections within each row. 'solid' = one
        figure-level line spanning the label column through the last panel.
        'dashed' = legacy per-axes dashed line in the metric panels only.
        'none' = off.
    show_mid_gridline : bool, default True
        If False, the vertical 0.5 gridline inside each panel is removed
        (only the 0 and 1 ticks/grid lines remain).
    row_label_width : float, optional
        Override (inches) for the unified label-column axes width across all
        rows. Note: section labels and row labels share one matplotlib axes,
        so this controls the full left-of-bars label area, not just the
        row-label text. Use this if section labels overlap row labels
        (auto-estimation can be inaccurate for bold fonts, wide fonts, or
        non-default font sizes).

        If passed *together* with ``figsize``, the first two columns get
        exactly ``panel_label_width + row_label_width`` inches and
        ``panel_width`` is derived to make the metric panels split the
        remaining figure width equally.
    section_label_width : float, optional
        Override (inches) for the section-label width estimate, applied
        per-row before the unified column width is computed.
    section_label_pad : float, optional
        Override (inches) for the gap between section label and row labels
        in 'plain' section style.
    row_label_pad : float, default 0.05
        Inch padding between the row labels' right edge and the first metric
        panel. Increase to push row labels left.
    section_sep_left_pad : float, default 0.0
        Inches by which to push each row's section-separator line rightward
        from its label column's axes edge. Pass ``0.10`` to align with the
        first letter of left-aligned section titles.
    section_label_align : {'left', 'right'}, optional
        Horizontal alignment of the section labels (first label column).
        Default 'right' in 'bracket' style, 'left' in 'plain' style.
    row_label_align : {'left', 'right'}, default 'right'
        Horizontal alignment of the row labels (second label column).

    Per-section keys honored inside each ``group_configs`` entry:
        - ``shade`` (bool, default True): when False, all rows in the section
          use the base ``color`` uniformly (no light→dark gradient).

    All other parameters carry the same semantics as ``plot_grouped_metrics``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if not panels:
        raise ValueError("`panels` must be non-empty.")

    dirs = {**DEFAULT_DIRECTIONS, **(metric_directions or {})}
    labels = metric_labels or {}
    panel_label_font_size = panel_label_font_size or label_font_size

    # 1. Per-row: optional remap + build sections + layout primitives.
    rows_data: list[dict] = []
    for p in panels:
        df_p, configs_p = _apply_label_remap(p["df"], p["group_configs"], row_labels)
        sections = _build_sections(df_p, configs_p, metrics)
        if not sections:
            raise ValueError(
                f"Panel '{p.get('panel_label', '?')}' produced no valid sections. "
                f"Check group_configs and DataFrame columns."
            )
        y_lookup, sec_ranges, y_max, sep_ys = _compute_y_layout(sections, section_gap)
        geom = _label_col_geometry(
            sections,
            textwrap_width,
            tick_font_size,
            label_font_size,
            section_style=section_style,
            section_label_width=section_label_width,
            section_label_pad=section_label_pad,
        )
        rows_data.append(
            {
                "panel_label": p.get("panel_label"),
                # Per-panel overrides fall back to the function-level defaults.
                "panel_label_bold": p.get("panel_label_bold", panel_label_bold),
                "panel_label_rotation": p.get(
                    "panel_label_rotation", panel_label_rotation
                ),
                "sections": sections,
                "y_lookup": y_lookup,
                "sec_ranges": sec_ranges,
                "y_max": y_max,
                "sep_ys": sep_ys,
                "geom": geom,
            }
        )

    # 2. Unify label-column width across rows so bars line up vertically.
    label_col_w = (
        row_label_width
        if row_label_width is not None
        else max(r["geom"]["natural_w"] for r in rows_data)
    )
    for rd in rows_data:
        rd["BX"] = (
            _bx_for_width(rd["geom"], label_col_w)
            if section_style == "bracket"
            else None
        )

    # 3. Figure.
    n_m = len(metrics)
    pad = 0.4
    row_h = [max((rd["y_max"] + 2 * pad) * 0.65, 1.2) for rd in rows_data]
    if figsize is None:
        # Auto-size figure from per-column widths.
        w = panel_label_width + label_col_w + panel_width * n_m + 0.5
        figsize = (w, sum(row_h))
    elif row_label_width is not None:
        # User pinned both figure width AND label column width — derive
        # panel_width from the remainder so the first two columns actually
        # get (panel_label_width + label_col_w) inches and the metric
        # panels split the rest equally.
        panel_width = max(
            0.3, (figsize[0] - panel_label_width - label_col_w - 0.5) / n_m
        )

    fig, axes = plt.subplots(
        len(rows_data),
        n_m + 2,
        figsize=figsize,
        gridspec_kw={
            "width_ratios": [panel_label_width, label_col_w] + [panel_width] * n_m,
            "height_ratios": row_h,
            "wspace": 0.12,
            "hspace": hspace,
        },
        squeeze=False,
    )

    # 4. Draw each row. Title appears only on the top row; x-tick labels only
    #    on the bottom row (visual x-axis sharing).
    n_rows = len(rows_data)
    for r_i, rd in enumerate(rows_data):
        is_top = r_i == 0
        is_bottom = r_i == n_rows - 1
        y_lo, y_hi = -pad, rd["y_max"] + pad

        # 4a. Panel label (far-left mini axes, like a row-level ylabel).
        _draw_panel_label(
            axes[r_i, 0],
            rd["panel_label"],
            fontsize=panel_label_font_size,
            rotation=rd["panel_label_rotation"],
            bold=rd["panel_label_bold"],
        )

        # 4b. Label column.
        _draw_label_column(
            axes[r_i, 1],
            rd["sections"],
            rd["sec_ranges"],
            rd["y_lookup"],
            y_lo,
            y_hi,
            label_col_w,
            rd["BX"],
            geom=rd["geom"],
            tick_font_size=tick_font_size,
            tick_font_color=tick_font_color,
            label_font_size=label_font_size,
            spine_lw=spine_lw,
            textwrap_width=textwrap_width,
            section_style=section_style,
            section_label_align=section_label_align,
            row_label_align=row_label_align,
            row_label_pad=row_label_pad,
        )

        # 4c. Metric panels — spine/ticks/labels only on the bottom row.
        for m_i, m in enumerate(metrics):
            _draw_metric_panel(
                axes[r_i, 2 + m_i],
                m,
                rd["sections"],
                rd["y_lookup"],
                y_lo,
                y_hi,
                dirs,
                labels,
                panel_width,
                tick_font_size=tick_font_size,
                tick_font_color=tick_font_color,
                spine_lw=spine_lw,
                show_title=is_top,
                show_xaxis=is_bottom,
                show_mid_gridline=show_mid_gridline,
            )

        # 4d. Section separators (within this row, spanning label col + panels).
        _draw_section_separators(
            fig,
            list(axes[r_i, 1:]),
            rd["sep_ys"],
            style=section_sep_style,
            left_pad=section_sep_left_pad,
        )

    # NOTE: deliberately *not* calling fig.tight_layout — it fights with the
    # explicit gridspec hspace/height_ratios. ``bbox_inches='tight'`` on save
    # trims outer whitespace cleanly.
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=fig_dpi)
        print(f"Saved → {save_path}")
    if show:
        plt.show()
        plt.close()
    return fig


# ── from libs/visuals/sensitivity.py ─────────────────────────────────

def plot_heterogeneity_grid(model, spec, moderators=None,
                            figsize_per_panel=(5.5, None)):
    """One panel per moderator. Y-axis: each prompt_level (categorical dummy
    or numeric var). X-axis: marginal effect at each moderator level."""
    moderators = moderators or spec.llm_attrs
    n = len(moderators)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]),
                              sharey=False)
    if n == 1:
        axes = [axes]

    for ax, mod in zip(axes, moderators):
        # Collect marginal effects across all prompt vars
        me = pd.concat([marginal_effects(model, spec, pv, mod)
                        for pv in spec.prompt_vars], ignore_index=True)
        if me.empty:
            continue

        me['kind'] = me['prompt_var'].apply(
            lambda v: 'persona' if v in spec.persona_vars else 'context'
        )

        # Order y-axis: persona first, then context, sorted within each
        me_sorted = me.sort_values(['kind', 'prompt_var', 'prompt_level'])
        y_labels = me_sorted['prompt_level'].unique().tolist()
        y_map = {pl: i for i, pl in enumerate(y_labels)}

        mod_levels = ['reference'] + sorted(
            [l for l in me['mod_level'].unique() if l != 'reference']
        )
        markers = ['o', 's', '^', 'D', 'v', 'P'][:len(mod_levels)]
        jitter = np.linspace(-0.25, 0.25, len(mod_levels))

        for marker, lvl, dy in zip(markers, mod_levels, jitter):
            sub = me[me['mod_level'] == lvl]
            for _, row in sub.iterrows():
                c = PALETTE[row['kind']]
                y = y_map[row['prompt_level']] + dy
                ax.errorbar(row['estimate'], y,
                            xerr=[[row['estimate'] - row['ci_low']],
                                  [row['ci_high'] - row['estimate']]],
                            fmt=marker, color=c, ecolor=c,
                            capsize=2, markersize=5, alpha=0.85)

        ax.axvline(0, color='gray', ls='--', lw=0.5)
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel('Effect on metric')
        ax.set_title(f'Moderated by {mod}')

        handles = [Line2D([0], [0], marker=m, color='gray', lw=0, label=l)
                   for m, l in zip(markers, mod_levels)]
        ax.legend(handles=handles, frameon=False, fontsize=7, loc='best')

    fig.suptitle('Heterogeneity of prompt effects across LLM attributes', y=1.01)
    fig.tight_layout()
    return fig


# ── from libs/visuals/sensitivity.py ─────────────────────────────────

def plot_robustness_comparison(m2, m2_by_attr: dict, spec: ModelSpec):
    """Scatter m2 interaction coefficients against m2_by_attr equivalents.
    Points near the y=x line → models agree → robust to multicollinearity."""
    c2 = m2.coefs()
    rows = []
    for attr, model in m2_by_attr.items():
        cs = model.coefs()
        # Match interaction terms across the two
        for term in cs['term']:
            if attr in term and ':' in term:
                if term in c2['term'].values:
                    full_est = c2.loc[c2['term'] == term, 'estimate'].iloc[0]
                    marg_est = cs.loc[cs['term'] == term, 'estimate'].iloc[0]
                    pv = term.split(':')[0]
                    rows.append({
                        'term': term, 'attr': attr, 'kind': _kind(pv, spec),
                        'full_model': full_est, 'marginal_model': marg_est,
                    })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 6))
    for kind, color in PALETTE.items():
        sub = df[df['kind'] == kind]
        ax.scatter(sub['marginal_model'], sub['full_model'],
                   color=color, label=kind.capitalize(), alpha=0.7, s=40)

    lim = max(abs(df[['full_model', 'marginal_model']].values).max() * 1.1, 0.01)
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.5)
    ax.axhline(0, color='gray', lw=0.3); ax.axvline(0, color='gray', lw=0.3)
    ax.set_xlabel('Interaction coefficient (single-moderator model)')
    ax.set_ylabel('Interaction coefficient (full m2)')
    ax.set_title('Robustness: do single- and full-moderator models agree?')
    ax.legend(frameon=False)
    return fig


# ── from libs/prompt/generation.py ───────────────────────────────────

def iter_locations(locs: List[str]) -> Iterator[str]:
    """Yield each location string."""
    for loc in locs:
        yield loc


# ── from libs/utils/ios.py ───────────────────────────────────────────

def read_list_from_file_llm_prompt(
    file_path: Path | str, line_separator: str = PROMPT_LINE_SEPARATOR
) -> List[List[str]]:
    content = []
    with open(file_path, "r") as f:
        item = []
        for line in f.read().splitlines():

            if line_separator == line:
                content.append(item)
                item = []
            else:
                item.append(line.strip())

        return content


# ── from libs/factuality/manual_validation.py ────────────────────────

def norm_eq(a: Any, b: Any) -> bool:
    return norm(a) == norm(b) if a and b else False


# ── from libs/utils/logger.py ────────────────────────────────────────

def colored_msg(msg, color=Bcolor.ENDC):
    print_string = color + u'{}' + Bcolor.ENDC
    print(print_string.format(msg))
    return
