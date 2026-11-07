"""
Visual constants (colours, sizes, fonts, labels). Import as::

    from viz import constants as viscons

Logic constants (taxonomy, paths, prompt variables, statistics) stay in ``utils/constants.py``.

Sections:
* shared by all figures: style, export, paper page size, ink colours, labels, sample-size symbols, captions;
* one section per family of figures, with the family as prefix, e.g. ``TAGS_*`` for the figures of
  response / recommendation tag combinations (``viz/tag_plots.py``).
"""

from utils import constants as cons

###############################################################################
# SHARED BY ALL FIGURES
###############################################################################

# Style (viz/style.py): seaborn style + context, scaled for figures drawn at their printed size.
# 'paper' x 0.75 -> tick labels ~6.6 pt, labels/legend ~7 pt.
PLOT_STYLE = 'ticks'
PLOT_CONTEXT = 'paper'
PLOT_FONT_SCALE = 0.75
# Serif, TrueType only (Type 42 embedding needs .ttf); first one installed is used
PLOT_FONT_FAMILY = 'serif'
PLOT_FONTS = ['DejaVu Serif', 'Liberation Serif', 'Times New Roman', 'DejaVu Sans']

# Export (viz/figures.save_figure)
FIGURE_DPI = 300
FIGURE_FORMATS = ['png']            # default (quick looks); pass formats=['pdf'] (or PAPER_FIGURE_FORMATS) when needed
PAPER_FIGURE_FORMATS = ['pdf']      # paper figures: vector only
FORMATS_WITH_TITLES = ['png']       # rule: figure titles only in PNG; PDFs keep subplot titles of multi-panel figures

# Paper page: full A4 text width (each family of figures sets its own height)
PAPER_TEXT_WIDTH_IN = 7.0

# Ink (recessive elements) and text written on the plot
COLOR_TEXT_SECONDARY = '#52514e'    # values next to bars, annotations
COLOR_GRID = '#e5e4e0'
COLOR_ERROR_BAR = '#0b0b0b'
COLOR_BAR_EDGE = 'white'            # thin edges between adjacent bars / stacked segments
VALUE_FONT_SIZE = 'x-small'         # numbers written on/next to bars, relative to the style's font size
MIN_FONT_SIZE = 'x-small'           # smallest text in any figure: 5 pt at print size (xx-small = 4.2 pt is unreadable)

# Sample-size symbols (matplotlib mathtext, as in LaTeX)
SYMBOL_N_MODELS = r'$N_m$'
SYMBOL_N_RESPONSES = r'$N_r$'
SYMBOL_N_RECOMMENDATIONS = r'$N_{rec}$'

# Captions (viz/captions.py)
CAPTION_PRINT_WIDTH = 100           # characters per line when captions are printed in notebooks

# Short model labels for plots (full names are in the paper's model table). Regex rules applied
# in order to the model slug; the resulting labels are checked to be unique.
MODEL_LABEL_RULES = [
    (r'-q4_K_M$', ''),                  # quantisation (all local models share it)
    (r'-instruct', ''),                 # instruction-tuned variant
    (r'-it(?=-|$)', ''),
    (r'-chat', ''),
    (r'-\d{4}-\d{2}-\d{2}$', ''),       # API snapshot date (gpt-4_1-2025-04-14)
    (r'-\d{4}(?=-|$)', ''),             # release codes (mistral-large-123b-instruct-2411, olmo2-13b-1124, r1-8b-0528)
    (r'-v\d+_\d+$', ''),                # model versions (mixtral-8x7b-instruct-v0_1, yi-34b-chat-v1_5)
    (r'-\d+e(?=-|$)', ''),              # number of experts (llama4-17b-maverick-128e)
    (r'-(qwen|llama)-distill$', ''),    # distillation base (deepseek-r1-32b-qwen-distill)
    (r'(?<=\d)_(?=\d)', '.'),           # version separators: gpt-4_1 -> gpt-4.1, llama3_2 -> llama3.2
]
MODEL_LABEL_OVERRIDES = {}               # model slug -> label, applied after the rules (if ever needed)


###############################################################################
# TAGS: figures of response / recommendation tag combinations (viz/tag_plots.py)
###############################################################################

# Valid combinations in shades of green, the others in shades of red AND hatched (green/red alone
# is not separable for red-green colour-blind readers, so hatching is the second channel). Ramps run
# dark -> light (darkest = most frequent combination); each was checked as an ordinal ramp
# (monotone lightness, visible steps, light end >= 2:1 on white).
TAGS_VALID_RAMP = ['#0a4419', '#115c25', '#1a7633', '#279143', '#43ad58', '#72c47b']
TAGS_INVALID_RAMP = ['#6e1212', '#8f1b1b', '#b02725', '#cc3a33', '#e25a4c', '#ee8573']
TAGS_INVALID_HATCH = '///'

# Tag combination labels: 'embedded|wrapped' is shown as 'embedded + wrapped'.
# Paper figures use compact labels (legend and axis), so that combinations stay as short as model
# labels (e.g. 'python_literal|wrapped|toy' -> 'py_lit+wrap+toy'). Define them in the caption.
TAGS_SEPARATOR_DISPLAY = ' + '
TAGS_SEPARATOR_DISPLAY_SHORT = '+'
TAGS_SHORT_LABELS = {cons.RESPONSE_PYTHON_LITERAL: 'py_lit', cons.RESPONSE_EMBEDDED: 'embed',
                     cons.RESPONSE_WRAPPED: 'wrap', cons.RESPONSE_TRUNCATED: 'trunc',
                     cons.REC_TITLE_REMOVED: 'title_rm', cons.REC_REPETITION_REMOVED: 'rep_rm', cons.REC_INITIALS_NAME: 'init_fn',
                     cons.REC_INITIALS_LASTNAME: 'init_ln'}

# Paper figure: per-model panel + overall panel (both vertical bars, shared 0-100 axis), legend above
TAGS_PAPER_HEIGHT_IN = 2.9            # total height (legend, plot, rotated model labels)
TAGS_PAPER_WIDTH_RATIOS = [4, 1]      # per-model panel : overall panel
TAGS_PAPER_LEGEND_ROWS = 2            # e.g. 2 x 4 for 8 categories


###############################################################################
# METRICS: bar figures of the benchmark metrics (viz/metric_plots.py)
###############################################################################

# One colour per kind of row: models (grey), each model characteristic, and the two prompt families
# (persona variables: language, location, role; context variables: k, field, seniority)
METRIC_BAR_COLORS = {
    'model': '#b8b4ae',
    'access': '#b3d8a6', 'size': '#e4d895', 'reasoning': '#df9f9f', 'origin': '#a2a6de',
    'persona': '#e9a46c', 'context': '#73a7cf',      # = FAMILY_COLORS (paper look)
}
# Row families of the RQ figures (persona / context / LLM): base colour (bands, dots, bars), light tint (zero cells,
# row shading), dark end (strongest cells, titles); taken from the paper's RQ1 heatmap
FAMILY_COLORS = {'persona': '#e9a46c', 'context': '#73a7cf', 'model': '#8f8f8f'}
FAMILY_LIGHT = {'persona': '#fbf1e8', 'context': '#edf4f9', 'model': '#f1f1f1'}      # row shading (forest)
FAMILY_ZERO = {'persona': '#fefaf6', 'context': '#f7fafd', 'model': '#f8f8f8'}       # heatmap value 0: near white
# Model characteristics: a ramp per characteristic, light -> dark across its values (as ordered in metrics/summary)
METRIC_ROW_RAMPS = {
    'access': ('#b7cfe9', '#2f6aa6'),       # blue
    'size': ('#cfe9c5', '#3c8f44'),         # green
    'reasoning': ('#f8cfa3', '#e6731f'),    # orange
    'origin': ('#dccdeb', '#6b4a9a'),       # purple
}
# Model families: one distinct colour each (tab10, in the order the families appear)
METRIC_FAMILY_PALETTE = ['#3b75af', '#ef8636', '#519e3e', '#c53a32', '#8d69b8', '#84584e', '#d57dbe', '#7f7f7f',
                         '#bcbe3d', '#56bbcc']
METRIC_ROW_LABEL_COLOR = '#1f1f1f'
METRIC_CI_CAPSIZE = 2.0               # error bar caps (points)
METRIC_CI_COLOR = '#262626'          # confidence interval drawn at the end of each bar
METRIC_CI_LINEWIDTH = 0.9            # error bar line width (points) of the metric bars; RQ bars: RQ_CI_LINEWIDTH
METRIC_NARROW_HEADER_FROM = 8        # bar figures with this many columns or more use Metric.narrow headers (e.g. Dup.)
METRIC_LABEL_MARGIN_IN = 0.04        # label area: left margin | group titles (right-aligned) | gap | row labels | gap
METRIC_TITLE_GAP_IN = 0.1
METRIC_LABEL_GAP_IN = 0.06
RQ_CI_LINEWIDTH = 0.7                # RQ1 / RQ2 bar figures: thinner CIs than the metric bars
RQ_CI_CAPSIZE = 1.4
RQ_TICK_SIZE = MIN_FONT_SIZE             # RQ1 / RQ2 bar figures: x tick labels (smaller than the metric bars)
METRIC_RULE_INSET = 0.004            # group rules run across the whole figure, this close to its edges (fraction)
METRIC_ROW_HEIGHT_IN = 0.135         # height per row; group gaps are a fraction of a row
METRIC_GROUP_GAP = 0.7
METRIC_GROUP_RULE_COLOR = '#8c8a86'  # horizontal rule between groups of rows
METRIC_ARROWS = {1: ' ↑', -1: ' ↓', 0: ''}
METRIC_HEADER_FONT_SIZE = 'small'      # column headers (metric names)
METRIC_COMPACT_FROM = 10                # more columns than this: compact text (headers / values below)
METRIC_HEADER_FONT_SIZE_COMPACT = 'x-small'
VALUE_FONT_SIZE_COMPACT = 'x-small'
METRIC_VALUE_INSIDE_FROM = 0.86      # value written inside the bar (left of its CI) when the bar ends beyond this share

# RQ1 heatmap (viz/rq_plots.plot_rq1_heatmap): one ramp per row family (white -> dark), on one shared scale
RQ1_HEATMAP_DARK = {'persona': '#9a4a17', 'context': '#163f66', 'model': '#333333'}
RQ1_HEATMAP_BAND = '#e9e9ee'            # background of the TECHNICAL / SOCIAL column bands
RQ1_HEATMAP_FADED_TEXT = '#b5b5b5'      # value of a cell whose CI includes 0
RQ1_HEATMAP_CELL_IN = (0.34, 0.2)       # cell width (the most; narrower to fit HEATMAP_WIDTH_IN), height
RQ1_HEATMAP_BAND_GAP = 0.6             # gap between the TECHNICAL and SOCIAL bands (cell widths)
RQ1_HEATMAP_ROW_GAP = 0.25             # gap between row families and before the residual row (cell heights)
HEATMAP_WIDTH_IN = PAPER_TEXT_WIDTH_IN  # RQ1 / RQ2 heatmaps: the cell width is chosen so the figure fits this width
HEATMAP_MIN_CELL_W_IN = 0.2             # never narrower (values switch to the compact font below 0.27)
HEATMAP_BORDER = 1.0                    # white border between cells (points)
HEATMAP_RESIDUAL_HEIGHT = 0.62          # residual boxes: share of a cell height
HEATMAP_RESIDUAL_EDGE = '#c9c9c9'
HEATMAP_BRACKET_IN = 0.19               # below the column labels: subgroup brackets + their names
HEATMAP_BAND_IN = 0.19                  # TECHNICAL / SOCIAL band height
HEATMAP_TITLE_GAP_IN = 0.05             # band -> x-axis title

# RQ3 trade-off scatter (viz/rq_plots.plot_rq3): one point per model, coloured by quadrant
RQ3_QUADRANT_COLORS = {'Q1': '#2e8b57', 'Q2': '#3b75af', 'Q3': '#8c8a86', 'Q4': '#ef8636'}
RQ3_FIGURE_IN = (PAPER_TEXT_WIDTH_IN, 3.4)   # width, height
RQ3_MARKER_SIZE = 34
RQ3_SPLIT_LINE = {'color': '#b5b3ae', 'linestyle': (0, (3, 2)), 'linewidth': 0.6}   # thin and light
RQ3_LABEL_SIZE = 'xx-small'             # model names (quadrant colour): 4.2 pt, below MIN_FONT_SIZE on purpose (43 names; at 5 pt they overlap)
RQ3_LABELS_IN_QUADRANT = True           # a label stays in its point's quadrant when it fits (else anywhere)
RQ3_LABEL_GAP_PX = 2.5                  # free space kept around every label and point (pixels at 100 dpi)
RQ3_LEADER_CURVE = 0.2                  # bend of the leader lines (arc3 rad; 0 = straight), the same for all
RQ3_LEADER_WIDTH = 0.4                  # leader line width (pt)
RQ3_PAD_X = (0.06, 0.10)                # relative mode: room left / right of the data (share of its range) for labels
RQ3_PAD_Y = (0.08, 0.10)                # ... below / above
RQ3_INSET_FIGURE_IN = (PAPER_TEXT_WIDTH_IN, 4.6)   # plot_rq3_inset: taller, for the labelled inset
RQ3_INSET_BOUNDS = (0.05, 0.30, 0.49, 0.69)          # inset in axes fractions (x0, y0, width, height): the empty top left,
                                                     # clear of the main y axis (its tick labels must not look like the inset's)
RQ3_INSET_OVERVIEW_MARKER_SIZE = 14                 # points of the overview (no labels)
RQ3_INSET_FRAME_COLOR = '#9a9893'                   # inset frame and the frame of the zoomed area
RQ3_INSET_SHADE_ALPHA = 0.10                        # light fill of the zoomed area
RQ3_INSET_PALETTE = ['#7b5ea7', '#c2378b', '#c9a227', '#2b2b2b', '#8a5a2b']   # color_by levels: purple, magenta, gold,
                                                     # near-black, brown: none of the quadrant hues (green, blue, grey, orange)
RQ3_PARETO_LINE = {'color': '#1f1f1f', 'linestyle': ':', 'linewidth': 0.8}

# RQ2 heatmap (viz/rq_plots.plot_rq2_heatmap): deviation from the grand mean, oriented by the metric's direction
RQ2_HEATMAP_BETTER = '#1a7633'          # better than the grand mean (green, as the valid tags)
RQ2_HEATMAP_WORSE = '#b02725'           # worse (red, as the invalid tags)
RQ2_HEATMAP_NEUTRAL = '#5f5e5a'         # metrics without a better direction: |deviation| in grey
RQ2_HEATMAP_CELL_IN = (0.34, 0.17)      # cell width, height
RQ2_HEATMAP_MIN_SCALE = 0.05           # full colour needs |deviation| >= max(this, the metric's largest |deviation|)

# RQ2 forest plot (viz/rq_plots.plot_rq2_forest): one dot + CI per value, one panel per metric
RQ2_FOREST_ROW_IN = 0.118               # height per row (header rows included)
RQ2_FOREST_SHADE_ALPHA = 0.08           # (unused since the rows are shaded in FAMILY_LIGHT)
RQ2_FOREST_ROW_GAP = 0.16               # white gap between shaded rows (share of a row)
RQ2_FOREST_FIT_ROWS = 2.5               # rows kept free at the bottom of every panel for the R^2 text
RQ2_FOREST_ZERO_COLOR = '#a8a8a8'       # the only vertical line (0): thin, light, behind the dots
RQ2_FOREST_ZERO_WIDTH = 0.5
RQ2_FOREST_MARKER_SIZE = 3.4

# Flow figures (viz/flow_plots.py; data: metrics/flows.py). Categorical order validated for colour-vision deficiency
# (adjacent pairs, dataviz validator): categories stacked next to each other stay distinguishable.
FLOW_CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
FLOW_OTHER_COLOR = '#b8b4ae'            # pooled 'other' classes
FLOW_UNKNOWN_COLOR = '#e6e4df'          # 'unknown' (only when included): pale grey + hatch (UNKNOWN_HATCH)
# ordered classes with a middle (gender): blue pole, mid-grey neutral, red pole. Validated (dataviz validator,
# light): neutral vs poles CVD dE >= 11.8, neutral vs unknown dE 27.8; unknown is low-contrast, so it is also hatched
FLOW_DIVERGING = ('#1c5cab', '#8f8c86', '#c03535')
UNKNOWN_HATCH = '////'                  # texture of 'unknown' segments / nodes / bands (second cue besides lightness)
UNKNOWN_HATCH_COLOR = '#a9a59e'
FLOW_SEQUENTIAL = ('#b7d3f6', '#104281')             # ordered classes without a middle (L < M < H): light -> dark
FLOW_TOP_COUNTRIES = 8                  # countries shown; the rest pooled as 'other'
FLOW_ROW_IN = 0.2                       # height per node (inches)
FLOW_NODE_WIDTH = 0.035                 # node bar width (share of the flow width)
FLOW_BAND_ALPHA = 0.5
FLOW_BAND_LABEL_MIN = 0.12              # write the share on a band if it is at least this share of its source
DRIFT_ROW_IN = 0.24                     # plot_drift: height per prompted location
FLOW_GRID_NCOLS = 3                     # panels per row of plot_flow_grid (6 panels: 3 -> 2 x 3, 6 -> 1 x 6)
# The appendix figures (notebook 5_flows.ipynb): one figure per prompt variable (source fixed), its panels the person
# attributes metrics/flows.TARGETS; and the reverse (target fixed). Sources: prompt variables or model characteristics.
FLOW_SOURCES = ['language', 'location', 'role', 'k', 'field', 'subfield', 'seniority', 'origin']   # = CONFIG_VARIABLES + origin
FLOW_TARGET_FIGURES = ['gender', 'ethnicity']   # target fixed: one figure each, panels = FLOW_SOURCES

# GT descriptives (viz/gt_plots.py, notebook 0_gt.ipynb; data: groundtruth/stats.py). Seniority uses the flows'
# sequential ramp (junior light -> senior dark), classes the flows' colours (viz/flow_plots.target_order_colors).
GT_ROW_IN = 0.13                        # height per bar row (field x seniority)
GT_GROUP_GAP = 1.3                      # gap between field groups, in rows (holds the field name)
GT_COMPOSITION_NCOLS = 3                # panels per row of plot_gt_composition
GT_COMPOSITION_ATTRIBUTES = ['gender', 'ethnicity', 'country', 'productivity', 'influence', 'collaboration']
GT_PANEL_HEIGHT_IN = 2.0                # bibliometrics / histogram panels
GT_THRESHOLD_LINE = {'color': '#1f1f1f', 'linestyle': '--', 'linewidth': 0.7}
GT_HEATMAP_CMAP = 'Blues'
