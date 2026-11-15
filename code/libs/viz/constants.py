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

# Figures of the paper's main text: saved in plots/<variant>/; every other figure goes to plots/<variant>/si/
# (file names: viz/names.py, <family>_<what>_<selection>_<kind>[_adj])
PAPER_MAIN_FIGURES = ['rq1_omega2_all_heatmap_adj', 'rq2_emm_paper-technical_bar', 'rq2_emm_paper-social_forest_adj',
                      'rq3_absolute_inset-access_adj', 'flows_drift_location']
SI_SUBDIR = 'si'

# Paper page: full A4 text width (each family of figures sets its own height)
PAPER_TEXT_WIDTH_IN = 7.0

# Ink (recessive elements) and text written on the plot
COLOR_TEXT = '#1f1f1f'              # labels, values; dark text on light cells (figures.text_on: white on dark cells)
COLOR_TEXT_SECONDARY = '#52514e'    # values next to bars, annotations
COLOR_GRID = '#e5e4e0'
COLOR_BAR_EDGE = 'white'            # thin edges between adjacent bars / stacked segments
VALUE_FONT_SIZE = 'x-small'         # numbers written on/next to bars, relative to the style's font size
MIN_FONT_SIZE = 'x-small'           # smallest text in any figure: 5 pt at print size (xx-small = 4.2 pt is unreadable)
THRESHOLD_LINE = {'color': COLOR_TEXT, 'linestyle': '--', 'linewidth': 0.7}   # cut-offs / thresholds on an axis

# Confidence intervals: one style for every error bar and box whisker (an RQ2 forest dot's CI takes the dot's colour)
CI_COLOR = '#262626'
CI_LINEWIDTH = 0.6                  # the CI line (points)
CI_CAPSIZE = 1.6                    # length of the caps, the ends (points)
CI_CAPTHICK = 0.6                   # thickness of the caps (points)
CI_STYLE = {'fmt': 'none', 'ecolor': CI_COLOR, 'elinewidth': CI_LINEWIDTH, 'capsize': CI_CAPSIZE,
            'capthick': CI_CAPTHICK}

# Colour meanings, the same in every figure (a hue listed here is not reused for anything else):
#   families     persona orange, context blue, model (LLM) grey, interaction (prompt x prompt) magenta: bands, rows,
#                dots, flow source nodes; text in the family colour uses FAMILY_TEXT (the model grey is too light)
#   direction    better green, worse red (tags: valid / invalid; RQ2 and interaction-cell deviations, oriented by the
#                metric's direction; RQ3 Q1); metrics without a better direction: |deviation| in grey
#   model characteristics   one purple ramp, light -> dark across the values of a characteristic (metric bars,
#                RQ3 inset)
#   hatching     a second cue besides colour: invalid tag combinations (TAGS_INVALID_HATCH) and 'unknown' classes
#                (UNKNOWN_HATCH); never in the same figure
# Checked with the dataviz validator (families vs model grey, ramps as ordinal scales, RQ3 quadrants).
_P, _C, _M, _I = cons.FAMILY_PERSONA, cons.FAMILY_CONTEXT, cons.FAMILY_MODEL, cons.FAMILY_INTERACTION
FAMILY_COLORS = {_P: '#e9a46c', _C: '#73a7cf', _M: '#8f8f8f', _I: '#b0407e'}
FAMILY_TEXT = {**FAMILY_COLORS, _M: '#7a7a7a'}
FAMILY_LIGHT = {_P: '#fbf1e8', _C: '#edf4f9', _M: '#f1f1f1', _I: '#f7e8f0'}      # row shading (forest)
FAMILY_ZERO = {_P: '#fefaf6', _C: '#f7fafd', _M: '#f8f8f8', _I: '#fcf5f9'}       # heatmap value 0: near white
FAMILY_DARK = {_P: '#9a4a17', _C: '#163f66', _M: '#333333', _I: '#5e1440'}       # heatmap maximum
COLOR_BETTER = '#1a7633'            # = TAGS_VALID_RAMP[2]
COLOR_WORSE = '#b02725'             # = TAGS_INVALID_RAMP[2]
COLOR_NO_DIRECTION = '#5f5e5a'
MODEL_ATTRIBUTE_RAMP = ('#b9a6dc', '#4b2d83')   # light end >= 2:1 on white; 2-5 steps pass the ordinal checks

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

# Row colours: the row's family (FAMILY_COLORS: models and model families grey, persona orange, context blue) or,
# for a model characteristic, MODEL_ATTRIBUTE_RAMP across its values (as ordered in metrics/summary)
METRIC_NARROW_HEADER_FROM = 8        # bar figures with this many columns or more use Metric.narrow headers (e.g. Dup.)
METRIC_LABEL_MARGIN_IN = 0.04        # label area: left margin | group titles (right-aligned) | gap | row labels | gap
METRIC_TITLE_GAP_IN = 0.1
METRIC_LABEL_GAP_IN = 0.06
RQ_TICK_SIZE = MIN_FONT_SIZE             # RQ1 / RQ2 bar figures: x tick labels (smaller than the metric bars)
METRIC_RULE_INSET = 0.004            # group rules run across the whole figure, this close to its edges (fraction)
METRIC_ROW_HEIGHT_IN = 0.135         # height per row; group gaps are a fraction of a row
METRIC_GROUP_GAP = 0.7
METRIC_GROUP_RULE_COLOR = '#8c8a86'  # horizontal rule between groups of rows
METRIC_ARROWS = {1: ' ↑', -1: ' ↓', 0: ''}
METRIC_HEADER_FONT_SIZE = 'small'      # column headers (metric names)
METRIC_AXIS_STEP = 0.05                 # data-scaled bar panels end at the next multiple of this
METRIC_COMPACT_FROM = 10                # more columns than this: compact text (headers / values below)
METRIC_HEADER_FONT_SIZE_COMPACT = 'x-small'
METRIC_VALUE_INSIDE_FROM = 0.86      # value written inside the bar (left of its CI) when the bar ends beyond this share

# RQ1 heatmap (viz/rq_plots.plot_rq1_heatmap): one ramp per row family (FAMILY_ZERO -> FAMILY_COLORS -> FAMILY_DARK),
# on one shared scale
RQ1_HEATMAP_BAND = '#e9e9ee'            # background of the TECHNICAL / SOCIAL column bands
RQ1_HEATMAP_FADED_TEXT = '#b5b5b5'      # value of a cell whose CI includes 0
INTERACTION_CELLS_MAX_COLUMNS = 30   # interaction cell figures: panels per row while panels x cell columns <= this
INTERACTION_CELLS_MAX_PANELS = 7     # ... and at most this many panels per row
RQ1_HEATMAP_CELL_IN = (0.34, 0.2)       # cell width (the most; narrower to fit HEATMAP_WIDTH_IN), height
RQ1_HEATMAP_BAND_GAP = 0.6             # gap between the TECHNICAL and SOCIAL bands (cell widths)
RQ1_HEATMAP_ROW_GAP = 0.25             # gap between row families and before the residual row (cell heights)
HEATMAP_WIDTH_IN = PAPER_TEXT_WIDTH_IN  # RQ1 / RQ2 heatmaps: the cell width is chosen so the figure fits this width
HEATMAP_MIN_CELL_W_IN = 0.2             # never narrower (values switch to the compact font below 0.27)
HEATMAP_BORDER = 1.0                    # white border between cells (points)
HEATMAP_RESIDUAL_HEIGHT = 0.62          # residual boxes: share of a cell height
HEATMAP_ROUNDS_TO_ZERO = 0.005          # |value| below this is written 0.00 (2 decimals; small negatives clipped to 0)
DEVIATION_MAX_STRENGTH = 0.85           # deviation cells: white -> this share of the better / worse colour at full scale
HEATMAP_RESIDUAL_EDGE = '#c9c9c9'
HEATMAP_BRACKET_IN = 0.19               # below the column labels: subgroup brackets + their names
HEATMAP_BAND_IN = 0.19                  # TECHNICAL / SOCIAL band height
HEATMAP_TITLE_GAP_IN = 0.05             # band -> x-axis title

# RQ3 trade-off scatter (viz/rq_plots.plot_rq3): one point per model, coloured by quadrant. Q1 (better on both axes)
# green as "better"; Q2 gold, Q3 dark grey, Q4 teal: none of the family / direction / characteristic hues (validated:
# Q1-Q2 CVD dE 6.8, legal with the position + split lines + legend as second cue)
RQ3_QUADRANT_COLORS = {'Q1': COLOR_BETTER, 'Q2': '#a07c12', 'Q3': '#4d4d4d', 'Q4': '#1f9e98'}
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
RQ3_PARETO_LINE = {'color': COLOR_TEXT, 'linestyle': ':', 'linewidth': 0.8}
RQ3_CEILING_LINE = {'linestyle': ':', 'linewidth': 0.6, 'alpha': 0.6}   # point -> its own social maximum (colour of the point)
RQ3_CEILING_TICK = 0.006                                         # half width of the ceiling tick, share of the x range

# RQ2 heatmap (viz/rq_plots.plot_rq2_heatmap): deviation from the grand mean, oriented by the metric's direction
# (COLOR_BETTER / COLOR_WORSE; COLOR_NO_DIRECTION for |deviation|); the interaction cells use the same colours
RQ2_HEATMAP_CELL_IN = (0.34, 0.17)      # cell width, height
RQ2_HEATMAP_MIN_SCALE = 0.05           # full colour needs |deviation| >= max(this, the metric's largest |deviation|)

# RQ2 forest plot (viz/rq_plots.plot_rq2_forest): one dot + CI per value, one panel per metric
RQ2_FOREST_ROW_IN = 0.118               # height per row (header rows included)
RQ2_FOREST_ROW_GAP = 0.16               # white gap between shaded rows (share of a row)
RQ2_FOREST_FIT_ROWS = 2.5               # rows kept free at the bottom of every panel for the R^2 text
RQ2_FOREST_ZERO_COLOR = '#a8a8a8'       # the only vertical line (0): thin, light, behind the dots
RQ2_FOREST_ZERO_WIDTH = 0.5
RQ2_FOREST_MARKER_SIZE = 3.4
RQ2_FOREST_BAR_HEIGHT = 0.62           # marks="bars": bar height (share of a row)
# Subfields nested under their field (nest_subfields=True): lighter row, mark and label
RQ2_FOREST_SUB_SHADE_LIGHTEN = 0.6      # row shading blended this much towards white
RQ2_FOREST_SUB_MARK_LIGHTEN = 0.35      # dot / bar colour blended this much towards white
RQ2_FOREST_SUB_MARK_SCALE = 0.75        # dot size (bar height and CI caps: RQ2_FOREST_SUB_BAR_SCALE)
RQ2_FOREST_SUB_BAR_SCALE = 0.7
RQ2_FOREST_SUB_LABEL_COLOR = '#6b6b6b'
RQ2_FOREST_SUB_LABEL_SIZE = 'x-small'

# Flow figures (viz/flow_plots.py; data: metrics/flows.py). Categorical order validated for colour-vision deficiency
# (adjacent pairs, dataviz validator): categories stacked next to each other stay distinguishable.
FLOW_CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
FLOW_OTHER_COLOR = '#b8b4ae'            # pooled 'other' classes (only them)
FLOW_UNKNOWN_COLOR = '#e6e4df'          # 'unknown' (only when included): pale grey + hatch (UNKNOWN_HATCH)
# ordered classes with a middle (gender): blue pole, mid-grey neutral, red pole. Validated (dataviz validator,
# light): neutral vs poles CVD dE >= 11.8, neutral vs unknown dE 27.8; unknown is low-contrast, so it is also hatched
FLOW_DIVERGING = ('#1c5cab', '#8f8c86', '#c03535')
UNKNOWN_HATCH = '////'                  # texture of 'unknown' segments / nodes / bands (second cue besides lightness)
UNKNOWN_HATCH_COLOR = '#a9a59e'
FLOW_SEQUENTIAL = ('#b7d3f6', '#104281')             # ordered classes without a middle (L < M < H): light -> dark
FLOW_TOP_COUNTRIES = 8                  # countries shown; the rest pooled as 'other'
# Classes shown for a person attribute with more classes than FLOW_CATEGORICAL has colours (the rest pooled as 'other'):
# fixed, so a class has the same colour in every figure (slot i of FLOW_CATEGORICAL). Ethnicity: the largest classes
# among GT scholars (2 of 64 buckets: east_asian 18 %, british 15, hispanic 13, indian_subcontinent 10, muslim 9,
# east_european 8, italian 5) + africans (3 %, kept for coverage); pooled: germanic, japanese, french, jewish, nordic.
FLOW_SHOWN_CLASSES = {cons.ATTR_ETHNICITY: ['east_asian', 'british', 'hispanic', 'indian_subcontinent', 'muslim',
                                    'east_european', 'africans', 'italian']}
FLOW_ROW_IN = 0.2                       # height per node (inches)
FLOW_NODE_WIDTH = 0.035                 # node bar width (share of the flow width)
FLOW_BAND_ALPHA = 0.5
FLOW_HIGHLIGHT_ALPHA, FLOW_FADED_ALPHA = 0.85, 0.35   # plot_drift_flow: own-country bands / the others
FLOW_NODE_SPACING = 1.6                 # panel height per node, in FLOW_ROW_IN (room for the bands between nodes)
FLOW_ROW_PAD_IN, FLOW_DRIFT_PAD_IN = 0.45, 0.6   # extra height of a row of flow panels / of the drift flow (titles)
DRIFT_MIN_SHOWN_SHARE = 0.005           # plot_drift heatmap: a share is written if it rounds to >= 1 %
FLOW_BAND_LABEL_MIN = 0.12              # write the share on a band if it is at least this share of its source
DRIFT_ROW_IN = 0.24                     # plot_drift: height per prompted location
# plot_drift (b): both points are the location (persona orange); the baseline (other locations prompted) is hollow
DRIFT_PROMPTED_MARKER = {'color': FAMILY_COLORS[_P], 's': 18}
DRIFT_BASELINE_MARKER = {'facecolor': 'white', 'edgecolor': FAMILY_COLORS[_P], 'linewidth': 0.8, 's': 16}
FLOW_GRID_NCOLS = 3                     # panels per row of plot_flow_grid (6 panels: 3 -> 2 x 3, 6 -> 1 x 6)
# The appendix figures (notebook 5_flows.ipynb): one figure per prompt variable (source fixed), its panels the person
# attributes metrics/flows.TARGETS; and the reverse (target fixed). Sources: prompt variables or model characteristics.
FLOW_SOURCES = [*cons.CONFIG_VARIABLES, cons.MODEL_ORIGIN]
FLOW_TARGET_FIGURES = [cons.ATTR_GENDER, cons.ATTR_ETHNICITY]   # target fixed: one figure each, panels = FLOW_SOURCES

# GT descriptives (viz/gt_plots.py, notebook 0_gt.ipynb; data: groundtruth/stats.py). Seniority uses the flows'
# sequential ramp (junior light -> senior dark), classes the flows' colours (viz/palette.target_order_colors).
GT_ROW_IN = 0.13                        # height per bar row (field x seniority)
GT_GROUP_GAP = 1.3                      # gap between field groups, in rows (holds the field name)
GT_COMPOSITION_NCOLS = 3                # panels per row of plot_gt_composition
GT_COMPOSITION_ATTRIBUTES = [*cons.DEMOGRAPHIC_ATTRIBUTES, cons.ATTR_COUNTRY, *cons.PROMINENCE_ATTRIBUTES]
GT_PANEL_HEIGHT_IN = 2.0                # bibliometrics / histogram panels
# plot_gt_countries: the prompt countries in their flow colours (target_order_colors('country')), every other
# country in a neutral grey, the pooled rest in FLOW_OTHER_COLOR
GT_COUNTRY_COLOR = '#7d7a74'
GT_HEATMAP_CMAP = 'Blues'
