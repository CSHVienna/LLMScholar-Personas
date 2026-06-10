"""
Grouped sensitivity heatmap.

A single conceptual matrix split into 2x2 blocks (row-group x col-group).
Rows are partitioned into 'persona' (orange) and 'context' (blue); each
row group is rendered with its own sequential colormap built from the
group's identity color. Both colormaps share `vmin`/`vmax` so values
remain numerically comparable.

Columns are partitioned into 'technical' / 'social' and labelled in
neutral grey above each block.
"""

from __future__ import annotations

import colorsys
import stat
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from libs.sensitivity.base import SensitivityModel


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

def _hls_adjust(hex_color: str, lightness: float,
                sat_scale: float = 1.0) -> Tuple[float, float, float]:
    r, g, b = to_rgb(hex_color)
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, lightness, min(1.0, s * sat_scale))


def sequential_cmap_from_color(base: str, name: str) -> LinearSegmentedColormap:
    light = _hls_adjust(base, lightness=0.96, sat_scale=0.45)
    dark  = _hls_adjust(base, lightness=0.28, sat_scale=1.15)
    return LinearSegmentedColormap.from_list(name, [light, base, dark], N=256)

def _get_cbar_label(label):
        if label == 'omega2':
            return r'$\omega^2$'
        elif label == 'eta2':
            return r'$\eta^2$'
        elif label == 'partial_eta2':
            return r'Partial $\eta^2$'
        else:
            return label
        

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

@dataclass
class GroupedSensitivityHeatmap:
    """
    Two-tier grouped heatmap with per-row-group colormaps.

    Required
    --------
    df : pd.DataFrame
    row_groups, col_groups : dict[str, list[str]]
    row_group_colors : dict[str, str]
    name_map : dict[str, str]   (optional)

    Display kwargs
    --------------
    figsize : tuple
    title, xlabel, ylabel : str or None
        Pass None to hide.
    show_colorbar_labels, show_col_group_labels,
    show_row_group_bands, show_colorbars : bool
    left, right, top, bottom : float in [0, 1] or None
        Figure margins. None auto-picks values that depend on which
        elements are shown.
    """
    ols_results: Dict[str, SensitivityModel]
    df: pd.DataFrame
    row_groups: Dict[str, List[str]]
    col_groups: Dict[str, List[str]]
    row_group_colors: Dict[str, str]
    name_map: Dict[str, str] = field(default_factory=dict)

    # Prefix-based x-tick grouping. Maps {raw_column_prefix: bracket_label},
    # e.g. {'factuality_': 'Factuality', 'div_': 'Diversity'}. For any
    # column whose name starts with one of these prefixes the prefix is
    # stripped, the bracket label is drawn below the tick labels spanning
    # the run of matching columns, and the displayed tick label is looked
    # up in name_map by full-name first then by suffix (so name_map can
    # mix full names for ungrouped cols / row labels and suffix defaults
    # for grouped cols). When this dict is empty the behaviour is
    # identical to the original: full column names go straight through
    # name_map and no brackets are drawn.
    tick_prefix_groups: Dict[str, str] = field(default_factory=dict)

    # significance (optional): a same-shape DataFrame of p-values aligned to df
    pvalue_df: Optional[pd.DataFrame] = None

    # text content (None hides)
    figsize: Tuple[float, float] = (13.5, 6.0)
    title: Optional[str] = 'Prompt-variable sensitivity by metric'
    xlabel: Optional[str] = 'Evaluation metric'
    ylabel: Optional[str] = 'Prompt variable'
    colorbar_label: Optional[str] = None

    # toggles
    show_colorbar_labels: bool = True
    show_col_group_labels: bool = True
    show_row_group_bands: bool = True
    show_colorbars: bool = True
    show_residual_row: bool = True   # uncoloured 1 - R^2 band below the LLM row

    # layout (None -> auto)
    left: Optional[float] = None
    right: Optional[float] = None
    top: Optional[float] = None
    bottom: Optional[float] = None
    wspace: float = 0.06
    hspace: float = 0.06

    # annotation
    annot_threshold: float = 0.02
    annot_fmt: str = '.2f'

    # significance encoding (only used if pvalue_df is provided)
    #   Marking *significant* cells:
    #     'stars'     -> append *, **, *** based on star_thresholds
    #     'bold'      -> bold annotation if p < significance_threshold
    #     'italic'    -> italic annotation if p < significance_threshold
    #   Marking *non-significant* cells (inversion — useful when almost
    #   everything is significant, so the exception is the news):
    #     'dim'       -> mute the text color for non-significant cells
    #     'underline' -> draw an underline beneath non-significant cells
    #   None  -> no encoding
    significance_style: Optional[str] = None
    significance_threshold: float = 0.05
    star_thresholds: Tuple[float, float, float] = (0.05, 0.01, 0.001)
    dim_color: str = "#cecece"         # muted grey for 'dim' style (light bg)
    dim_color_on_dark: str = '#d0d0d0' # muted near-white for 'dim' on dark bg
    underline_linewidth: float = 0.55  # used for 'underline' style

    # annotation text color, switched by cell luminance for legibility
    text_color_dark: str = '#2b2b2b'      # used on light cells (default)
    text_color_light: str = 'white'       # used on dark cells
    text_luminance_threshold: float = 0.50  # below this, cell is "dark"

    # advanced geometry – usually no need to touch
    tick_label_room_y: float = 0.065  # horizontal room for y-tick text
    tick_label_room_x: float = 0.11  # vertical room for x-tick text (rotated)
    band_width: float = 0.022         # used for both row band & col box thickness
    band_gap: float = 0.006           # gap between band/box and tick labels
    ylabel_room: float = 0.025
    xlabel_room: float = 0.07        # vertical room for xlabel + gap to box
    edge_pad: float = 0.012
    cbar_gap: float = 0.022
    cbar_width: float = 0.014
    cbar_label_room: float = 0.028     # horizontal room for cbar TICK NUMBERS
                                       # (smaller = title sits closer to bars)
    cbar_title_pad: float = 0.028      # extra room for the shared colorbar label
    col_box_color: str = '#eaecef'    # light grey fill for col-group boxes
    col_box_text_color: str = '#555'

    # Residual row (uncoloured 1 - R^2 band). Drawn below the LLM block;
    # NOT part of `_data`, so it never affects vmin/vmax. Cells have a faint
    # border and the value as text only — no fill.
    residual_label: str = 'Residual'      # y-tick label for the band
    residual_row_height: float = 0.030    # band height in figure coords
    residual_gap: float = 0.010           # gap between LLM block and the band
    residual_edge_color: str = '#d9d9d9'  # very light cell border
    residual_edge_width: float = 0.6
    residual_text_color: str = '#777'     # muted, since cells are uncoloured
    residual_label_color: str = '#6e6e6e' # y-tick label color (matches LLM grey)
    residual_fmt: str = '.2f'

    # Prefix-group bracket tier (only used when tick_prefix_groups is set)
    prefix_group_room: float = 0.034     # vertical room for the bracket tier
    prefix_group_color: str = '#8a8a8a'  # bracket line + text color
    prefix_group_linewidth: float = 0.7
    prefix_group_fontsize: float = 9.0
    prefix_group_tick_height: float = 0.005   # end-tick height (fig coords)
    prefix_group_inset: float = 0.003         # bracket inset from cell edges

    # -- setup -------------------------------------------------------------

    def __post_init__(self):
        # Drop any rows/cols listed in row_groups / col_groups that aren't
        # actually in the DataFrame, emitting a single warning per kind.
        # This keeps `ROW_GROUPS` / `COL_GROUPS` reusable across data
        # slices that don't contain every metric.
        self.row_groups = GroupedSensitivityHeatmap._filter_groups(
            self.row_groups, self.df.index, kind='row',
        )
        self.col_groups = GroupedSensitivityHeatmap._filter_groups(
            self.col_groups, self.df.columns, kind='column',
        )

        rows = [r for g in self.row_groups.values() for r in g]
        cols = [c for g in self.col_groups.values() for c in g]

        if not rows:
            print(
                "No rows remain after filtering — nothing to plot.",
            )
        if not cols:
            print(
                "No columns remain after filtering — nothing to plot.",
            )

        self._data = (self.df.loc[rows, cols]
                      .apply(pd.to_numeric, errors='coerce'))

        # p-values: same row/col restriction and numeric coercion, or None
        if self.pvalue_df is not None:
            self._pvals = (self.pvalue_df.reindex(index=rows, columns=cols)
                           .apply(pd.to_numeric, errors='coerce'))
        else:
            self._pvals = None

        # collected during _annotate_cell; processed after fig.canvas.draw()
        self._underline_pending: list = []

        self.cmaps = {
            g: sequential_cmap_from_color(c, f'{g}_cmap')
            for g, c in self.row_group_colors.items()
        }

    # -- constructors ------------------------------------------------------
    
    @classmethod
    def from_long(cls, 
                  ols_results: Dict[str, SensitivityModel],
                  anova_df: pd.DataFrame, *,
                  row_col: str = 'prompt_var',
                  col_col: str = 'metric',
                  value_col: str = 'partial_eta2',
                  pvalue_col: Optional[str] = 'p_value',
                  **kwargs) -> 'GroupedSensitivityHeatmap':
        """
        Build the heatmap from a long-format ANOVA results DataFrame.

        The input is the kind of frame statsmodels (or pingouin) returns —
        one row per (prompt_var, metric) combination, with effect-size and
        p-value columns.

        Parameters
        ----------
        ols_results : dict[str, SensitivityModel]
        anova_df : pd.DataFrame
            Long-format results. Must contain `row_col`, `col_col`,
            `value_col`, and (optionally) `pvalue_col`.
        row_col, col_col : str
            Column names that become the heatmap's index and columns.
        value_col : str
            Column whose values fill the cells. Default 'partial_eta2'.
        pvalue_col : str or None
            Column with p-values for significance encoding. Pass None to
            skip p-value handling.
        **kwargs
            Forwarded to GroupedSensitivityHeatmap (row_groups, col_groups,
            row_group_colors, name_map, significance_style, etc.).
        """
        value_df = anova_df.pivot(index=row_col, columns=col_col,
                                  values=value_col)
        pvalue_df = (anova_df.pivot(index=row_col, columns=col_col,
                                    values=pvalue_col)
                     if pvalue_col is not None else None)

        colorbar_label = kwargs.get('colorbar_label', _get_cbar_label(value_col))
        kwargs['colorbar_label'] = colorbar_label

        return cls(ols_results=ols_results, df=value_df, pvalue_df=pvalue_df, **kwargs)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _filter_groups(groups: Dict[str, List[str]],
                       available: Iterable[str], *,
                       kind: str) -> Dict[str, List[str]]:
        """Return a copy of `groups` with entries missing from `available`
        removed. Groups that become empty are dropped entirely.

        A single `UserWarning` is emitted listing every dropped entry,
        annotated with the group it came from, so the user can fix their
        config or data without seeing N separate warnings.
        """
        available_set = set(available)
        filtered: Dict[str, List[str]] = {}
        missing: List[Tuple[str, str]] = []  # (group, name)
        emptied: List[str] = []

        for group, names in groups.items():
            kept = [n for n in names if n in available_set]
            missing.extend((group, n) for n in names if n not in available_set)
            if kept:
                filtered[group] = kept
            elif names:  # had members, but none survived
                emptied.append(group)

        if missing:
            formatted = ', '.join(f'{n!r} ({g})' for g, n in missing)
            warnings.warn(
                f"Skipping {len(missing)} {kind}(s) missing from "
                f"DataFrame: {formatted}",
                stacklevel=3,
            )
        if emptied:
            warnings.warn(
                f"Dropping empty {kind}-group(s) after filtering: "
                f"{', '.join(emptied)}",
                stacklevel=3,
            )
        return filtered

    def _pretty(self, name: str) -> str:
        return self.name_map.get(name, name)

    def _split_tick(self, col: str) -> Tuple[str, Optional[str]]:
        """Split a raw column name into (tick_label, prefix_group_label).

        Lookup order for the tick label is:
            1. `name_map[col]`         — full-name override (most specific)
            2. `name_map[suffix]`      — suffix default (most reusable)
            3. raw suffix              — fallback

        Returning the prefix-group label lets the caller decide whether
        to draw a bracket; the column is ungrouped (None) if no prefix
        in `tick_prefix_groups` matches.

        Longest-prefix-match is used so e.g. 'fact_' and 'factuality_'
        can coexist without ambiguity.
        """
        for prefix in sorted(self.tick_prefix_groups, key=len, reverse=True):
            if col.startswith(prefix):
                suffix = col[len(prefix):]
                if col in self.name_map:
                    label = self.name_map[col]
                else:
                    label = self.name_map.get(suffix, suffix)
                return label, self.tick_prefix_groups[prefix]
        return self._pretty(col), None

    def _find_prefix_runs(self, cg: str) -> List[Tuple[int, int, str]]:
        """Find contiguous runs of columns in col-group `cg` that share a
        prefix-group label. Returns (j_start, j_end, label) tuples with
        inclusive ends, in column order.
        """
        cols = self.col_groups[cg]
        runs: List[Tuple[int, int, str]] = []
        cur_label: Optional[str] = None
        cur_start = 0
        for j, c in enumerate(cols):
            _, label = self._split_tick(c)
            if label == cur_label:
                continue
            if cur_label is not None:
                runs.append((cur_start, j - 1, cur_label))
            cur_label = label
            cur_start = j
        if cur_label is not None:
            runs.append((cur_start, len(cols) - 1, cur_label))
        return runs

    def _block(self, row_g: str, col_g: str) -> pd.DataFrame:
        return self._data.loc[self.row_groups[row_g], self.col_groups[col_g]]

    def _pvalue_block(self, row_g: str, col_g: str) -> Optional[pd.DataFrame]:
        if self._pvals is None:
            return None
        return self._pvals.loc[self.row_groups[row_g], self.col_groups[col_g]]

    def _residual_for_metric(self, metric: str) -> Optional[float]:
        """Return 1 - R^2 for `metric`, or None if unavailable.

        R^2 is read from the fitted model:
            ols_results[metric]['m1'].r2_values['r2']
        Any missing key / attribute yields None so the cell is left blank
        rather than raising — keeps the band robust to partial results.
        """
        try:
            r2 = self.ols_results[metric]['m1'].r2_values['r2']
        except (KeyError, AttributeError, TypeError):
            return None
        if r2 is None or pd.isna(r2):
            return None
        return 1.0 - float(r2)

    def _residual_block(self, col_g: str) -> List[Optional[float]]:
        """Ordered list of residuals for the columns of col-group `col_g`,
        in the same order the heatmap blocks render them."""
        return [self._residual_for_metric(c) for c in self.col_groups[col_g]]

    def _star_suffix(self, p: float) -> str:
        """Stars by descending strictness: ***  **  *."""
        t = sorted(self.star_thresholds)  # ascending [strict, ..., lax]
        if p < t[0]:
            return '***'
        if p < t[1]:
            return '**'
        if p < t[2]:
            return '*'
        return ''

    def _cell_is_dark(self, value: float, cmap, vmin: float, vmax: float) -> bool:
        """Return True if a cell's background is dark enough that dark text
        would be hard to read on it. Uses ITU-R BT.709 relative luminance."""
        if pd.isna(value) or vmax <= vmin:
            return False
        normalized = float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))
        r, g, b, _ = cmap(normalized)
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return luminance < self.text_luminance_threshold

    def _annotate_cell(self, ax, i: int, j: int,
                       value: float, pvalue: Optional[float], *,
                       cmap, vmin: float, vmax: float) -> None:
        """Draw one annotation at cell (i, j), with significance styling."""
        if pd.isna(value) or abs(value) < self.annot_threshold:
            return

        is_dark = self._cell_is_dark(value, cmap, vmin, vmax)
        base_color = self.text_color_light if is_dark else self.text_color_dark
        dim_color = self.dim_color_on_dark if is_dark else self.dim_color

        text = format(value, self.annot_fmt)
        kwargs = dict(ha='center', va='center',
                      fontsize=7.5, color=base_color)

        style = self.significance_style
        has_p = pvalue is not None and not pd.isna(pvalue)
        is_sig = has_p and pvalue < self.significance_threshold
        is_ns = has_p and pvalue >= self.significance_threshold
        needs_underline = False

        if style == 'stars' and has_p:
            text += self._star_suffix(pvalue)
        elif style == 'bold' and is_sig:
            kwargs['fontweight'] = 'bold'
        elif style == 'italic' and is_sig:
            kwargs['fontstyle'] = 'italic'
        
        elif style == 'dim' and is_ns:
            kwargs['color'] = dim_color
        elif style == 'underline' and is_ns:
            needs_underline = True

        text_obj = ax.text(j + 0.5, i + 0.5, text, **kwargs)
        if needs_underline:
            self._underline_pending.append((ax, text_obj))

    def _draw_pending_underlines(self) -> None:
        """Add underlines below queued text annotations.

        Must be called AFTER fig.canvas.draw() so text bounding boxes
        have settled — that's how we know where each annotation lives.
        """
        for ax, text_obj in self._underline_pending:
            bbox = text_obj.get_window_extent()
            inv = ax.transData.inverted()
            # bbox.y0 = visual bottom in display coords; seaborn inverts
            # the y-axis, so visually-lower maps to LARGER data y.
            xd_left, yd_bottom = inv.transform((bbox.x0, bbox.y0))
            xd_right, _        = inv.transform((bbox.x1, bbox.y0))
            underline_y = yd_bottom + 0.04   # data units, visually below text
            ax.plot([xd_left, xd_right], [underline_y, underline_y],
                    color=text_obj.get_color(),
                    lw=self.underline_linewidth,
                    solid_capstyle='butt',
                    zorder=text_obj.get_zorder())

    def _resolve_margins(self) -> Tuple[float, float, float, float]:
        """Pick sensible defaults based on what's actually shown."""
        # left: edge + ylabel + band + gap + y-tick labels
        if self.left is not None:
            left = self.left
        else:
            left = self.edge_pad + self.tick_label_room_y
            if self.show_row_group_bands:
                left += self.band_width + self.band_gap
            if self.ylabel:
                left += self.ylabel_room

        # right: 1 - (edge + cbar tick numbers + cbar + gap [+ shared label])
        if self.right is not None:
            right = self.right
        elif self.show_colorbars:
            right = 1.0 - self.edge_pad - self.cbar_label_room \
                    - self.cbar_width - self.cbar_gap
            if self.colorbar_label:
                right -= self.cbar_title_pad
        else:
            right = 1.0 - self.edge_pad

        # top: edge + title only (col-group labels moved to bottom)
        if self.top is not None:
            top = self.top
        else:
            top = 1.0 - self.edge_pad
            if self.title:
                top -= 0.07

        # bottom: residual band + x-tick labels + (prefix bracket) + col box + xlabel + edge
        if self.bottom is not None:
            bottom = self.bottom
        else:
            bottom = self._residual_y0()
            if self._residual_active:
                bottom += self.residual_row_height + self.residual_gap

        return left, right, top, bottom

    # -- axis cosmetics ----------------------------------------------------

    @staticmethod
    def _strip(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    def _style_xticks(self, ax, labels):
        ax.set_xticklabels(labels, rotation=42, ha='center',
                           fontsize=9, color='#333')
        ax.tick_params(axis='x', length=0, pad=2)

    def _style_yticks(self, ax, labels, color):
        ax.set_yticklabels(labels, rotation=0, fontsize=10)
        for t in ax.get_yticklabels():
            t.set_color(color)
            t.set_fontweight('semibold')
        ax.tick_params(axis='y', length=0, pad=4)

    # -- block drawing -----------------------------------------------------

    def _draw_block(self, ax, block, pvalue_block, cmap, vmin, vmax, *,
                    show_xticks, show_yticks, row_group):
        sns.heatmap(
            block, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
            annot=False, cbar=False,
            linewidths=0.6, linecolor='white', square=False,
        )

        # Per-cell annotation so significance styling can vary by cell.
        n_rows, n_cols = block.shape
        for i in range(n_rows):
            for j in range(n_cols):
                v = block.iat[i, j]
                p = pvalue_block.iat[i, j] if pvalue_block is not None else None
                self._annotate_cell(ax, i, j, v, p,
                                    cmap=cmap, vmin=vmin, vmax=vmax)

        if show_xticks:
            tick_labels = [self._split_tick(c)[0] for c in block.columns]
            self._style_xticks(ax, tick_labels)
        else:
            ax.set_xticks([])
        if show_yticks:
            self._style_yticks(ax, [self._pretty(r) for r in block.index],
                               color=self.row_group_colors[row_group])
        else:
            ax.set_yticks([])
        ax.set_xlabel(''); ax.set_ylabel('')
        for sp in ax.spines.values():
            sp.set_visible(False)

    # -- group decorations -------------------------------------------------

    def _add_row_band(self, fig, ax, color, label):
        bb = ax.get_position()
        band_x = bb.x0 - self.tick_label_room_y - self.band_width
        band = fig.add_axes([band_x, bb.y0,
                             self.band_width, bb.height])
        band.set_facecolor(color)
        self._strip(band)
        band.text(0.5, 0.5, label.upper(), ha='center', va='center',
                  color='white', fontweight='bold', fontsize=10,
                  rotation=90, transform=band.transAxes)

    # -- geometry properties ----------------------------------------------

    @property
    def _col_box_height(self) -> float:
        """Col-box vertical thickness, scaled so it matches the row-band
        horizontal thickness in inches (not in figure coordinates)."""
        return self.band_width * (self.figsize[0] / self.figsize[1])

    def _col_box_y(self) -> float:
        """Y position (figure coords) for the col-group boxes."""
        y = self.edge_pad
        if self.xlabel:
            y += self.xlabel_room
        return y

    def _prefix_group_y_bottom(self) -> float:
        """Y position (figure coords) of the bottom of the bracket tier."""
        y = self._col_box_y()
        if self.show_col_group_labels:
            y += self._col_box_height + self.band_gap
        return y

    @property
    def _residual_active(self) -> bool:
        """Residual band is drawn only when toggled on AND we actually have
        models to read R^2 from. Keeps the demo / no-model paths unchanged."""
        return self.show_residual_row and bool(self.ols_results)

    def _residual_y0(self) -> float:
        """Bottom edge (figure coords) of the residual band.

        This is exactly the stack of decorations below the band — bracket
        tier + x-tick label room — so the band sits directly on top of the
        x-tick labels, mirroring how a normal bottom block row would.
        """
        y = self._prefix_group_y_bottom()
        if self.tick_prefix_groups:
            y += self.prefix_group_room
        y += self.tick_label_room_x
        return y

    def _xlabel_y(self) -> float:
        return self.edge_pad + 0.015

    def _add_col_box(self, fig, ax, label):
        """Light-grey rectangle below the x-tick labels, mirroring row bands.
        Height is scaled by figure aspect ratio so the box appears equally
        thick (in inches) as the row bands."""
        bb = ax.get_position()
        box = fig.add_axes([bb.x0, self._col_box_y(),
                            bb.width, self._col_box_height])
        box.set_facecolor(self.col_box_color)
        self._strip(box)
        box.text(0.5, 0.5, label.upper(), ha='center', va='center',
                 color=self.col_box_text_color, fontweight='bold',
                 fontsize=9.5, transform=box.transAxes)

    def _add_residual_band(self, fig, ax, cg, show_ylabel):
        """Draw the uncoloured residual strip under col-group `cg`.

        One cell per metric in the group: faint border, NO fill (so it
        never enters vmin/vmax), and the `1 - R^2` value printed as text.
        This band also carries the x-tick labels, since it is now the
        lowest row of cells in the figure.
        """
        from matplotlib.patches import Rectangle

        cols = self.col_groups[cg]
        n = len(cols)
        bb = ax.get_position()
        band = fig.add_axes([bb.x0, self._residual_y0(),
                             bb.width, self.residual_row_height])
        band.set_xlim(0, n)
        band.set_ylim(0, 1)

        for j, val in enumerate(self._residual_block(cg)):
            band.add_patch(Rectangle(
                (j, 0), 1, 1, facecolor='none',
                edgecolor=self.residual_edge_color,
                linewidth=self.residual_edge_width,
                clip_on=False,
            ))
            if val is not None and not pd.isna(val):
                band.text(j + 0.5, 0.5, format(val, self.residual_fmt),
                          ha='center', va='center', fontsize=7.5,
                          color=self.residual_text_color)

        # x-tick labels live on the band (it is the lowest cell row now)
        tick_labels = [self._split_tick(c)[0] for c in cols]
        band.set_xticks([j + 0.5 for j in range(n)])
        self._style_xticks(band, tick_labels)

        # y-tick label only on the leftmost col-group
        if show_ylabel:
            band.set_yticks([0.5])
            band.set_yticklabels([self.residual_label], rotation=0,
                                 fontsize=10)
            for t in band.get_yticklabels():
                t.set_color(self.residual_label_color)
                # t.set_fontweight('semibold')
            band.tick_params(axis='y', length=0, pad=4)
        else:
            band.set_yticks([])

        for sp in band.spines.values():
            sp.set_visible(False)

    def _add_prefix_group_brackets(self, fig, ax, cg):
        """For each contiguous prefix-group run inside col-group `cg`,
        draw `├──── label ────┤` spanning the run's columns, below the
        x-tick labels and above the col-group box.

        Positions are in figure coords. The bottom-row block axis `ax`
        anchors x positions; the cell width inside the block is
        `ax.width / N`, so each run spans an integer slice of that.
        """
        from matplotlib.lines import Line2D

        runs = self._find_prefix_runs(cg)
        if not runs:
            return

        bb = ax.get_position()
        n = len(self.col_groups[cg])
        cell_w = bb.width / n

        y0 = self._prefix_group_y_bottom()
        y_line = y0 + 0.78 * self.prefix_group_room  # bracket line near top
        y_text = y0 + 0.30 * self.prefix_group_room  # text below the line
        half_tick = self.prefix_group_tick_height / 2
        inset = self.prefix_group_inset
        lw = self.prefix_group_linewidth
        color = self.prefix_group_color

        for j_start, j_end, label in runs:
            x_left  = bb.x0 + j_start * cell_w + inset
            x_right = bb.x0 + (j_end + 1) * cell_w - inset
            x_mid   = (x_left + x_right) / 2

            # ├ end tick, ──── horizontal, ┤ end tick
            for x in (x_left, x_right):
                fig.add_artist(Line2D(
                    [x, x], [y_line - half_tick, y_line + half_tick],
                    color=color, linewidth=lw,
                    transform=fig.transFigure, clip_on=False,
                ))
            fig.add_artist(Line2D(
                [x_left, x_right], [y_line, y_line],
                color=color, linewidth=lw,
                transform=fig.transFigure, clip_on=False,
            ))
            fig.text(x_mid, y_text, label,
                     ha='center', va='center',
                     fontsize=self.prefix_group_fontsize,
                     color=color)

    def _add_colorbar(self, fig, ax_ref, cmap, vmin, vmax, right,
                      label, label_color):
        bb = ax_ref.get_position()
        cax = fig.add_axes([right + self.cbar_gap, bb.y0,
                            self.cbar_width, bb.height])
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        cb = fig.colorbar(sm, cax=cax)
        cb.outline.set_visible(False)
        cb.locator = MaxNLocator(nbins=5)
        cb.update_ticks()
        cb.ax.tick_params(length=0, labelsize=8, colors='#444')
        if self.show_colorbar_labels:
            cax.set_title(label.upper(), fontsize=9, fontweight='bold',
                          color=label_color, pad=6)

    # -- main --------------------------------------------------------------

    def draw(self):
        vmin = float(self._data.min().min())
        vmax = float(self._data.max().max())
        row_keys, col_keys = list(self.row_groups), list(self.col_groups)
        widths = [len(self.col_groups[g]) for g in col_keys]
        heights = [len(self.row_groups[g]) for g in row_keys]

        left, right, top, bottom = self._resolve_margins()

        fig = plt.figure(figsize=self.figsize, facecolor='white')
        gs = GridSpec(
            len(heights), len(widths),
            width_ratios=widths, height_ratios=heights,
            wspace=self.wspace, hspace=self.hspace,
            left=left, right=right, top=top, bottom=bottom,
            figure=fig,
        )

        # blocks
        block_axes: Dict[Tuple[str, str], plt.Axes] = {}
        last_row = len(row_keys) - 1
        resid_active = self._residual_active
        for i, rg in enumerate(row_keys):
            for j, cg in enumerate(col_keys):
                ax = fig.add_subplot(gs[i, j])
                block_axes[(rg, cg)] = ax
                self._draw_block(
                    ax, self._block(rg, cg), self._pvalue_block(rg, cg),
                    self.cmaps[rg], vmin, vmax,
                    # when the residual band is shown it owns the x-ticks
                    show_xticks=(i == last_row and not resid_active),
                    show_yticks=(j == 0),
                    row_group=rg,
                )

        fig.canvas.draw()  # finalize positions

        # underlines need text bboxes, which are only valid after the
        # canvas has been drawn at least once.
        self._draw_pending_underlines()

        # row decorations
        for rg in row_keys:
            if self.show_row_group_bands:
                self._add_row_band(
                    fig, block_axes[(rg, col_keys[0])],
                    self.row_group_colors[rg], rg,
                )
            if self.show_colorbars:
                self._add_colorbar(
                    fig, block_axes[(rg, col_keys[-1])],
                    self.cmaps[rg], vmin, vmax, right,
                    label=rg, label_color=self.row_group_colors[rg],
                )

        # residual band (uncoloured 1 - R^2 strip, directly under the
        # bottom block row; owns the x-ticks when present)
        if resid_active:
            for j, cg in enumerate(col_keys):
                self._add_residual_band(
                    fig, block_axes[(row_keys[-1], cg)], cg,
                    show_ylabel=(j == 0),
                )

        # column decorations (below the last row)
        if self.show_col_group_labels:
            for cg in col_keys:
                self._add_col_box(fig, block_axes[(row_keys[-1], cg)], cg)

        # prefix-group brackets (between tick labels and col-group boxes)
        if self.tick_prefix_groups:
            for cg in col_keys:
                self._add_prefix_group_brackets(
                    fig, block_axes[(row_keys[-1], cg)], cg,
                )

        # shared colorbar label: one rotated text spanning both colorbars
        if self.colorbar_label and self.show_colorbars:
            label_x = right + self.cbar_gap + self.cbar_width \
                      + self.cbar_label_room + 0.006
            label_y = (bottom + top) / 2
            fig.text(label_x, label_y, self.colorbar_label,
                     ha='center', va='center', rotation=90,
                     fontsize=11, color='#555')

        # figure-level labels
        mid_x = (left + right) / 2
        mid_y = (bottom + top) / 2
        if self.xlabel:
            fig.text(mid_x, self._xlabel_y(), self.xlabel,
                     ha='center', fontsize=11, color='#555')
        if self.ylabel:
            fig.text(self.edge_pad + 0.010, mid_y, self.ylabel,
                     va='center', rotation=90, fontsize=11, color='#555')
        if self.title:
            fig.suptitle(self.title, fontsize=14, fontweight='semibold',
                         y=1.0 - self.edge_pad - 0.005, color='#222')

        return fig


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

# Row groups (prompt variables, organised by family)
ROW_GROUPS: Dict[str, List[str]] = {
    'persona': ['language', 'location', 'role'],
    'context': ['k', 'field', 'subfield', 'target'],
    'llm':     ['model'],
}

# Column groups (metrics, partitioned into technical / social)
COL_GROUPS: Dict[str, List[str]] = {
    'technical': [
        'validity', 'refusals', 'consistency', 'duplicates',
        'factuality_author', 'factuality_field',
        'factuality_seniority', 'factuality_location',
    ],
    'social': [
        'parity_eth', 'parity_gender', 'parity_pub', 'parity_cit',
        'div_gen', 'div_eth', 'div_loc',
        'pct_pub_l', 'pct_pub_m', 'pct_pub_h',
        'pct_cit_l', 'pct_cit_m', 'pct_cit_h',
        'popularity_pub', 'popularity_cit',
    ],
}

PROMPT_VAR_COLORS = {
    'persona': '#E08E45',
    'context': '#5C8BB2',
    'llm':     '#6e6e6e',
}

# name_map lookup order for a grouped column 'factuality_field' is:
#   1. name_map['factuality_field']  — full-name override (most specific)
#   2. name_map['field']             — suffix default (most reusable)
#   3. 'field'                       — raw suffix fallback
#
# For row names and ungrouped columns, only step 1 (full name) is used.
# Use full-name keys to disambiguate suffix/row collisions — here both
# row variable 'field' and the suffix of 'factuality_field' would map
# to 'Field' without the explicit overrides for grouped columns.
NAME_MAP = {
    # row labels
    'language': 'Language', 'location': 'Location', 'role': 'Role',
    'k': 'k', 'field': 'Field', 'subfield': 'Subfield',
    'target': 'Target', 'model': 'Model',
    # ungrouped technical metrics
    'validity': 'Validity', 'refusals': 'Refusals',
    'consistency': 'Consistency', 'duplicates': 'Duplicates',
    # full-name overrides for grouped columns whose suffix collides with
    # a row name ('field', 'location'): force lowercase tick labels
    'factuality_field':    'field',
    'factuality_location': 'location',
    # suffix defaults (no collision → bare key is enough)
    'author': 'author', 'seniority': 'seniority',
    'eth': 'eth.', 'gender': 'gender', 'pub': 'pub.', 'cit': 'cit.',
    'gen': 'gen.', 'loc': 'loc.',
    'l_pub': 'L pub.', 'm_pub': 'M pub.', 'h_pub': 'H pub.',
    'l_cit': 'L cit.', 'm_cit': 'M cit.', 'h_cit': 'H cit.',
}

TICK_PREFIX_GROUPS = {
    'factuality_': 'Factuality',
    'parity_':     'Parity',
    'div_':        'Diversity',
    'pct_works_':        '% Publications',
    'pct_citations_':      '% Citations',
    'popularity_': 'Popularity',
}


def _make_demo_anova_df(seed: int = 7) -> pd.DataFrame:
    """Long-format ANOVA-style placeholder: one row per (prompt_var, metric).

    Columns match the user's actual schema:
        metric, prompt_var, sum_sq, df, F, p_value, partial_eta2, prompt_type

    Replace the call to this in main() with your real ANOVA DataFrame.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for group_name, vars_ in ROW_GROUPS.items():
        for var in vars_:
            for metric in sum(COL_GROUPS.values(), []):
                eta2 = float(rng.beta(0.6, 8.0))
                # p shrinks fast with effect size – mimicking large-N ANOVA.
                p = float(rng.uniform(0, 1)) * (1 - eta2) ** 8
                rows.append({
                    'metric': metric,
                    'prompt_var': var,
                    'omega2': eta2,
                    'eta2': eta2,
                    'partial_eta2': eta2,
                    'p_value': p,
                    'prompt_type': group_name,
                })
    return pd.DataFrame(rows)


def main():
    anova_df = _make_demo_anova_df()
    plot = GroupedSensitivityHeatmap.from_long(
        ols_results,
        anova_df,
        row_col='prompt_var',
        col_col='metric',
        value_col='partial_eta2',
        pvalue_col='p_value',
        row_groups=ROW_GROUPS,
        col_groups=COL_GROUPS,
        row_group_colors=PROMPT_VAR_COLORS,
        name_map=NAME_MAP,
        tick_prefix_groups=TICK_PREFIX_GROUPS,
        figsize=(14, 5.5),
        title=None,
        show_colorbar_labels=False,
        significance_style='bold',   # 'stars' | 'bold' | 'italic' | None
        colorbar_label=r'Partial $\eta^2$',
    )
    fig = plot.draw()
    fig.savefig('/home/claude/heatmap_preview.png', dpi=160,
                facecolor='white')


if __name__ == '__main__':
    main()