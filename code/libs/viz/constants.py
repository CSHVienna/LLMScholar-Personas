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
FORMATS_WITH_TITLES = ['png']       # rule: titles only in PNG (quick looks); PDFs (paper) have none

# Paper page: full A4 text width (each family of figures sets its own height)
PAPER_TEXT_WIDTH_IN = 7.0

# Ink (recessive elements) and text written on the plot
COLOR_TEXT_SECONDARY = '#52514e'    # values next to bars, annotations
COLOR_GRID = '#e5e4e0'
COLOR_ERROR_BAR = '#0b0b0b'
COLOR_BAR_EDGE = 'white'            # thin edges between adjacent bars / stacked segments
VALUE_FONT_SIZE = 'x-small'         # numbers written on/next to bars, relative to the style's font size

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
# (persona variables: language, location, role; context variables: k, field, target)
METRIC_BAR_COLORS = {
    'model': '#b8b4ae',
    'access': '#b3d8a6', 'size': '#e4d895', 'reasoning': '#df9f9f', 'origin': '#a2a6de',
    'persona': '#f2b98d', 'context': '#92bfdc',
}
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
METRIC_CI_CAPSIZE = 1.6               # error bar caps (points)
METRIC_CI_COLOR = '#262626'          # confidence interval drawn at the end of each bar
METRIC_CI_LINEWIDTH = 0.9            # thin: a line, not a block
METRIC_ROW_HEIGHT_IN = 0.135         # height per row; group gaps are a fraction of a row
METRIC_GROUP_GAP = 0.7
METRIC_GROUP_RULE_COLOR = '#8c8a86'  # horizontal rule between groups of rows
METRIC_ARROWS = {1: ' ↑', -1: ' ↓', 0: ''}
METRIC_HEADER_FONT_SIZE = 'small'      # column headers (metric names)
METRIC_COMPACT_FROM = 10                # more columns than this: compact text (headers / values below)
METRIC_HEADER_FONT_SIZE_COMPACT = 'x-small'
VALUE_FONT_SIZE_COMPACT = 'xx-small'
METRIC_VALUE_INSIDE_FROM = 0.86      # value written inside the bar (left of its CI) when the bar ends beyond this share

# RQ1 heatmap (viz/rq_plots.plot_rq1_heatmap): one ramp per row family (white -> dark), on one shared scale
RQ1_HEATMAP_DARK = {'persona': '#8a3f0a', 'context': '#0f4c81', 'model': '#3d3d3d'}
RQ1_HEATMAP_BAND = '#e9e9ee'            # background of the TECHNICAL / SOCIAL column bands
RQ1_HEATMAP_FADED_TEXT = '#b5b5b5'      # value of a cell whose CI includes 0
RQ1_HEATMAP_CELL_IN = (0.34, 0.2)       # cell width, height
RQ1_HEATMAP_BAND_GAP = 0.6             # gap between the TECHNICAL and SOCIAL bands (cell widths)
RQ1_HEATMAP_ROW_GAP = 0.25             # gap between row families and before the residual row (cell heights)

# RQ3 trade-off scatter (viz/rq_plots.plot_rq3): one point per model, coloured by quadrant
RQ3_QUADRANT_COLORS = {'Q1': '#2e8b57', 'Q2': '#3b75af', 'Q3': '#8c8a86', 'Q4': '#ef8636'}
RQ3_FIGURE_IN = (PAPER_TEXT_WIDTH_IN, 3.4)   # width, height
RQ3_MARKER_SIZE = 34
RQ3_SPLIT_LINE = {'color': '#8c8a86', 'linestyle': '--', 'linewidth': 0.8}
RQ3_PARETO_LINE = {'color': '#1f1f1f', 'linestyle': ':', 'linewidth': 0.8}

# RQ2 heatmap (viz/rq_plots.plot_rq2_heatmap): deviation from the grand mean, oriented by the metric's direction
RQ2_HEATMAP_BETTER = '#1a7633'          # better than the grand mean (green, as the valid tags)
RQ2_HEATMAP_WORSE = '#b02725'           # worse (red, as the invalid tags)
RQ2_HEATMAP_NEUTRAL = '#5f5e5a'         # metrics without a better direction: |deviation| in grey
RQ2_HEATMAP_CELL_IN = (0.34, 0.17)      # cell width, height
RQ2_HEATMAP_MIN_SCALE = 0.05           # full colour needs |deviation| >= max(this, the metric's largest |deviation|)

# RQ2 forest plot (viz/rq_plots.plot_rq2_forest): one dot + CI per value, one panel per metric
RQ2_FOREST_ROW_IN = 0.15                # height per row (header rows included)
RQ2_FOREST_SHADE_ALPHA = 0.08           # row shading in the family colour (persona / context)
RQ2_FOREST_MARKER_SIZE = 3.2
