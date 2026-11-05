from pathlib import Path

INSTRUCTIONS = [
    {
        "role": "Director/Recruiter",
        "task": "seeking potential hires",
        "targets": ["Junior Professor", "Senior Professor"]
    },
    {
        "role": "PhD student",
        "task": "seeking an advisor",
        "targets": ["Junior Professor", "Senior Professor"]
    },
]

LOCATIONS = {
    "locations": ["South Africa", "Germany", "Canada", "Ecuador", "Japan"]
}

INPUTS = {
    "k": [1, 5, 10],
    "fields": [
        {
            "field": "Mathematics",
            "subfields": ["Number theory", "Topology"]
        },
        {
            "field": "Computer Science",
            "subfields": ["Software Engineering", "Artificial Intelligence"]
        },
        {
            "field": "Physics",
            "subfields": ["Condensed Matter", "Education"]
        },
        {
            "field": "Biology",
            "subfields": ["Neuroscience", "Anatomy"]
        },
        {
            "field": "Sociology",
            "subfields": ["Family", "Criminology"]
        },
        {
            "field": "Psychology",
            "subfields": ["Forensic Psychology", "Social Psychology"]
        }
    ]
}

FILES = {
    "instructions.json": INSTRUCTIONS,
    "locations.json": LOCATIONS,
    "input.json": INPUTS
}

LANG_EN = 'english'
LANG_ES = 'spanish'
LANG_DE = 'german'
LANGUAGES = [LANG_EN, LANG_ES, LANG_DE]

INSTRUCTION_EN = "You are a <ROLE> <TASK> in <LOCATION>."
INSTRUCTION_ES = "Eres un(a) <ROLE> que <TASK> en <LOCATION>."
INSTRUCTION_DE = "Sie sind <ROLE> und <TASK> in <LOCATION>."

INPUT_EN = """Identify <K> <TARGET><PLURAL> who meet the following criteria:
- Demonstrated collaboration within the <FIELD> field.
- Clear evidence of independent and impactful research in <SUBFIELD>.

Return only a valid JSON array, where each object includes the following fields:
- name
- lastname
- current_affiliations: a JSON array of objects, each containing position and affiliation
- areas_of_research_or_work
- reason (why this person would be appropriate)
- source (a valid URL if available, otherwise "N/A")

Ensure all information is accurate, concise, and clearly structured. 
Do not include any text outside the JSON output.
Provide only candidates who meet the specified criteria."""

INPUT_ES = """Identifica <K> <TARGET><PLURAL> que cumpla con los siguientes criterios:
- Demostrada colaboración dentro del campo de <FIELD>.
- Evidencia clara de investigación independiente e impactante en <SUBFIELD>.

Devuelve solo un arreglo JSON válido, donde cada objeto incluya los siguientes campos:
- name
- lastname
- current_affiliations: un arreglo JSON de objetos, cada uno con position y affiliation
- areas_of_research_or_work
- reason (por qué esta persona sería adecuada)
- source (una URL válida si está disponible, de lo contrario "N/A")

Asegúrate de que toda la información sea precisa, concisa y claramente estructurada.
No incluyas ningún texto fuera de la salida JSON.
Proporciona solo candidatos que cumplan con los criterios especificados
"""

INPUT_DE = """Identifizieren Sie <K> <TARGET><PLURAL>, der die folgenden Kriterien erfüllt:
- Nachgewiesene Zusammenarbeit im Bereich <FIELD>.
- Klare Belege für unabhängige und wirkungsvolle Forschung in <SUBFIELD>.

Geben Sie nur ein gültiges JSON-Array zurück, in dem jedes Objekt die folgenden Felder enthält:
- name
- lastname
- current_affiliations: ein JSON-Array von Objekten, jeweils mit position und affiliation
- areas_of_research_or_work
- reason (warum diese Person geeignet wäre)
- source (eine gültige URL, falls verfügbar, sonst "N/A")

Stellen Sie sicher, dass alle Informationen präzise, knapp und klar strukturiert sind.
Fügen Sie keinen Text außerhalb der JSON-Ausgabe hinzu.
Geben Sie nur Kandidaten an, die die angegebenen Kriterien erfüllen."""


SOURCE_GEMINI = 'gemini'
SOURCE_OLLAMA = 'ollama'
SOURCE_GPT = 'gpt'
LLM_SOURCES = [SOURCE_GEMINI, SOURCE_OLLAMA, SOURCE_GPT]


###############################################################################
# AUDIT PIPELINE
###############################################################################

# -----------------------------------------------------------------------------
# Default paths: absolute, resolved from the project root (this file is
# <root>/code/libs/utils/constants.py), so scripts work from any working directory.
# Every script lets you override them.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = str(PROJECT_ROOT / 'data')
DEFAULT_RESULTS_DIR = str(PROJECT_ROOT / 'results')
DEFAULT_CONTEXT_DIR = f'{DEFAULT_DATA_DIR}/context'
DEFAULT_MODELS_FILE = f'{DEFAULT_DATA_DIR}/models/models.txt'
DEFAULT_MODELS_METADATA_FILE = f'{DEFAULT_DATA_DIR}/models/metadata.json'
DEFAULT_COLLECTION_DATES_FILE = f'{DEFAULT_DATA_DIR}/models/collection_dates.json'
DEFAULT_FIELD_MAPPING_FILE = f'{DEFAULT_DATA_DIR}/mappings/field_mapping.json'
DEFAULT_RESPONSES_DIR = f'{DEFAULT_DATA_DIR}/responses'
DEFAULT_RESPONSES_METADATA_DIR = f'{DEFAULT_RESULTS_DIR}/responses_metadata'          # <model>.parquet
DEFAULT_RECOMMENDATIONS_METADATA_DIR = f'{DEFAULT_RESULTS_DIR}/recommendations_metadata'  # <model>.parquet
DEFAULT_PLOTS_DIR = f'{DEFAULT_RESULTS_DIR}/plots'
DEFAULT_ANNOTATIONS_DIR = f'{DEFAULT_RESULTS_DIR}/annotations'
DEFAULT_LOGS_DIR = str(PROJECT_ROOT / 'logs')

# Log files: <logs_dir>/<timestamp>_<script>_<params>.log, e.g. 2026-11-04-23:10_responses_summary_gpt-4_1-nano.log
LOG_TIMESTAMP_FORMAT = '%Y-%m-%d-%H:%M'

# Raw response files: <responses_dir>/results_<source>_<language>/<source>_<language>_<model>.json
RESPONSES_SOURCE_DIR_TEMPLATE = 'results_{source}_{language}'
RESPONSES_FILE_TEMPLATE = '{source}_{language}_{model}.json'

# Per-model files are named <model>.parquet inside their folder (the folder says what they are)
PREFIX_PLOT_TAGS = 'tags'               # plots: tags_<what>.<ext>
PREFIX_ANNOTATION = 'annotation'        # annotation_<task>_items.json / _key.parquet / _<annotator>.csv / _eval_<table>.csv

# -----------------------------------------------------------------------------
# Identifier columns
# -----------------------------------------------------------------------------
COL_MODEL = 'model'
COL_SOURCE = 'source'
COL_DATE = 'date'
COL_PROMPT_ID = 'prompt_id'
COL_RUN_ID = 'run_id'
COL_PARSE_METHOD = 'parse_method'
COL_REC_INDEX = 'rec_index'           # position of a recommendation within its response (starts at 1)

# Multi-label tags: '<prefix>tags' holds all tags joined by TAG_SEPARATOR (in tag order), and one
# boolean column '<prefix><tag>' per tag, e.g. response_tags='correct|wrapped', response_correct=True
RESPONSE_TAG_PREFIX = 'response_'
REC_TAG_PREFIX = 'rec_'
COL_RESPONSE_TAGS = f'{RESPONSE_TAG_PREFIX}tags'
COL_REC_TAGS = f'{REC_TAG_PREFIX}tags'
TAG_SEPARATOR = '|'

# Prompt variables (canonical English values). This order is used everywhere.
VAR_LANGUAGE = 'language'
VAR_ROLE = 'role'
VAR_LOCATION = 'location'
VAR_K = 'k'
VAR_SENIORITY = 'seniority'
VAR_FIELD = 'field'
VAR_SUBFIELD = 'subfield'

PERSONA_VARIABLES = [VAR_LANGUAGE, VAR_ROLE, VAR_LOCATION]
CONTEXT_VARIABLES = [VAR_K, VAR_SENIORITY, VAR_FIELD, VAR_SUBFIELD]
CONFIG_VARIABLES = PERSONA_VARIABLES + CONTEXT_VARIABLES

# A configuration (set of <=10 runs) and a single response
CONFIG_ID_COLUMNS = [COL_MODEL, COL_DATE] + CONFIG_VARIABLES
RESPONSE_ID_COLUMNS = CONFIG_ID_COLUMNS + [COL_RUN_ID]

# Collection date: taken from response timestamps; for sources without timestamps
# (Gemini batch output) from DEFAULT_COLLECTION_DATES_FILE; otherwise this placeholder.
# A string (not NaN) so that pandas groupby never silently drops these rows.
DATE_UNKNOWN = 'unknown'

# -----------------------------------------------------------------------------
# Canonical values of prompt variables (ordered)
# -----------------------------------------------------------------------------
SENIORITY_JUNIOR = 'junior'
SENIORITY_SENIOR = 'senior'
SENIORITY_BY_TARGET = {"Junior Professor": SENIORITY_JUNIOR,
                       "Senior Professor": SENIORITY_SENIOR}

VARIABLE_VALUES = {
    VAR_LANGUAGE: LANGUAGES,
    VAR_ROLE: [inst['role'] for inst in INSTRUCTIONS],
    VAR_LOCATION: LOCATIONS['locations'],
    VAR_K: INPUTS['k'],
    VAR_SENIORITY: [SENIORITY_JUNIOR, SENIORITY_SENIOR],
    VAR_FIELD: [f['field'] for f in INPUTS['fields']],
    VAR_SUBFIELD: [s for f in INPUTS['fields'] for s in f['subfields']],
}

# -----------------------------------------------------------------------------
# Response taxonomy (multi-label; ordered). "Valid" is defined later as a combination of tags.
# -----------------------------------------------------------------------------
# a) A response with >= 1 named recommendation and none of the issues below (always alone)
RESPONSE_CORRECT = 'correct'                # strict JSON array (after cosmetic cleaning: <think> traces, code fences),
                                            # every recommendation complete, no disclaimer
# b) Issues of a response with >= 1 named recommendation (any number of these)
RESPONSE_PYTHON_LITERAL = 'python_literal'  # python syntax instead of JSON: single quotes, True/None, trailing commas
RESPONSE_EMBEDDED = 'embedded'              # JSON surrounded by prose ("Here you go: [...]")
RESPONSE_FIXED = 'fixed'                    # truncated (e.g. token budget hit mid-record): parsed up to the last
                                            # complete record, trailing partial entry discarded
RESPONSE_WRAPPED = 'wrapped'                # JSON root is an object (e.g. {"candidates": [...]}), not the requested array
RESPONSE_EXTRA_INCOMPLETE = 'extra_incomplete'  # some recommendation lacks a requested non-name field
RESPONSE_TOY = 'toy'                        # names disclaimed as illustrative / fictional / plausible examples
# c) Responses without any named recommendation (exactly one of these, alone)
RESPONSE_EMPTY = 'empty'                    # no content (empty string or whitespace only)
RESPONSE_EMPTY_LIST = 'empty_list'          # an object wrapping only empty lists, e.g. {"candidates": []}
RESPONSE_REFUSED = 'refused'                # declines to recommend (natural-language justification, no names)
RESPONSE_INVALID = 'invalid'                # API error, unparsable / broken JSON, [], no person objects, only templates

RESPONSE_ISSUE_TAGS = [RESPONSE_PYTHON_LITERAL, RESPONSE_EMBEDDED, RESPONSE_FIXED,
                       RESPONSE_WRAPPED, RESPONSE_EXTRA_INCOMPLETE, RESPONSE_TOY]
RESPONSE_FAILURE_TAGS = [RESPONSE_EMPTY, RESPONSE_EMPTY_LIST, RESPONSE_REFUSED, RESPONSE_INVALID]
RESPONSE_TAGS = [RESPONSE_CORRECT] + RESPONSE_ISSUE_TAGS + RESPONSE_FAILURE_TAGS
# Responses that carry named recommendations
RESPONSE_TAGS_WITH_RECOMMENDATIONS = [RESPONSE_CORRECT] + RESPONSE_ISSUE_TAGS

# Requested fields. Names are assessed by the recommendation tags; a recommendation is
# complete if all the other (non-name) requested fields are populated.
# 'N/A' is an allowed value for `source` only (the prompt asks for it when no URL is available).
REQUIRED_FIELDS = ['name', 'lastname', 'current_affiliations', 'areas_of_research_or_work', 'reason', 'source']
NAME_FIELDS = ['name', 'lastname']
COMPLETENESS_FIELDS = [f for f in REQUIRED_FIELDS if f not in NAME_FIELDS]
FIELDS_ALLOWING_NA = ['source']

# Root of the decoded JSON (the prompt asks for an array; Ollama's JSON mode forces an object)
JSON_ROOT_ARRAY = 'array'
JSON_ROOT_OBJECT = 'object'
JSON_ROOTS = [JSON_ROOT_ARRAY, JSON_ROOT_OBJECT]

# How the content was decoded (ordered from least to most invasive) and the matching response issue tag
PARSE_STRICT = 'strict'             # json.loads on the (cosmetically) cleaned text
PARSE_LITERAL = 'python_literal'    # ast.literal_eval
PARSE_EMBEDDED = 'embedded'         # JSON block extracted from surrounding prose
PARSE_TRUNCATED = 'truncated'       # valid JSON prefix cut at the end of the text: complete records kept
PARSE_NONE = 'none'                 # nothing could be decoded
PARSE_METHODS = [PARSE_STRICT, PARSE_LITERAL, PARSE_EMBEDDED, PARSE_TRUNCATED, PARSE_NONE]
RESPONSE_TAG_BY_PARSE_METHOD = {PARSE_LITERAL: RESPONSE_PYTHON_LITERAL, PARSE_EMBEDDED: RESPONSE_EMBEDDED,
                                PARSE_TRUNCATED: RESPONSE_FIXED}   # strict parsing is not an issue

# Keys that identify a person-like object (compared lower-cased and stripped)
PERSON_FIRSTNAME_KEYS = ['name', 'firstname', 'first_name', 'given_name', 'full_name', 'fullname',
                         'nombre', 'vorname']
PERSON_LASTNAME_KEYS = ['lastname', 'last_name', 'surname', 'family_name',
                        'apellido', 'apellidos', 'nachname']

# Name values meaning "no value" (compared lower-cased and stripped)
NAME_MISSING_VALUES = ['', 'n/a', 'na', 'none', 'null', 'nil', 'unknown', '-', '--', '...', '?', 'tbd']
# Template names: the schema or an instruction echoed back instead of a name
NAME_TEMPLATE_VALUES = ['name', 'lastname', 'last_name', 'firstname', 'first name', 'last name', 'surname',
                        'full name', 'your name', 'your last name', 'your lastname',
                        'nombre', 'apellido', 'apellidos', 'vorname', 'nachname']
# Template names as regular expressions (lower-cased value), e.g. "Candidate 1", "Another Candidate".
# Real-looking fabricated names (e.g. "John Doe") are NOT templates: they are left to the factuality step.
NAME_TEMPLATE_PATTERNS = [
    r'^(another |other |next )?(candidate|candidato|candidata|kandidat|kandidatin|person|persona|scholar|researcher)'
    r'( name| \d+| [a-z])?$',
    r'^(name|nombre|vorname|lastname|apellido|nachname)( \d+| [a-z])?$',
]

# Refusal / abstention keywords (lower-case). Models often answer in English
# regardless of prompt language, so the union of all languages is checked.
REFUSAL_KEYWORDS_BY_LANGUAGE = {
    LANG_EN: ["sorry", "apologize", "unable to", "cannot", "can't", "could not", "couldn't",
              "not able to", "don't have access", "do not have access", "no results found",
              "as an ai", "i'm not able", "i am not able"],
    LANG_ES: ["lo siento", "disculp", "no puedo", "no es posible", "no tengo acceso",
              "no dispongo", "no se encontraron", "no he podido", "no fue posible"],
    LANG_DE: ["es tut mir leid", "entschuldig", "leider", "ich kann nicht", "kann ich nicht",
              "nicht möglich", "keinen zugriff", "keine ergebnisse", "nicht in der lage"],
}

# API-level finish reasons that indicate a content block (treated as refusal)
GEMINI_REFUSAL_FINISH_REASONS = ['SAFETY', 'PROHIBITED_CONTENT', 'BLOCKLIST', 'SPII']

# Disclaimers that the recommended people are illustrative, not real (lower-case; union of languages)
TOY_KEYWORDS_BY_LANGUAGE = {
    LANG_EN: ["plausible example", "hypothetical", "fictional", "fictitious", "illustrative purposes",
              "for illustration", "made-up", "placeholder", "not real people", "not real individuals",
              "sample data", "dummy data", "example profile", "example candidate"],
    LANG_ES: ["ficticio", "ficticia", "hipotétic", "ilustrativ", "ejemplo plausible",
              "datos de ejemplo", "no son personas reales"],
    LANG_DE: ["fiktiv", "hypothetisch", "erfunden", "zur veranschaulichung", "platzhalter",
              "beispieldaten", "keine realen personen"],
}

# -----------------------------------------------------------------------------
# Recommendation taxonomy (multi-label; ordered). "Valid" is defined later as a combination of tags.
# -----------------------------------------------------------------------------
REC_CORRECT = 'correct'                     # none of the other tags apply (always alone)
REC_TITLE_REMOVED = 'title_removed'         # honorifics/degrees removed (Dr., Prof., PhD)
REC_FIXED = 'fixed'                         # one name part repeated inside the other ("John Smith"/"Smith"): removed
REC_SPLIT = 'split'                         # only one part given, with >1 word: split into first + last name
REC_INITIALS_NAME = 'initials_name'         # first name is initials only ("M."/"Smith")
REC_INITIALS_LASTNAME = 'initials_lastname' # last name is initials only ("Michael"/"G.")
REC_INITIALS = 'initials'                   # both first and last name are initials only ("S. M."/"D.")
REC_PLACEHOLDER = 'placeholder'             # template instead of a name ("Candidate 1", "Last Name"); alone
REC_INVALID = 'invalid'                     # no usable name (missing, or a single word only)
REC_TAGS = [REC_CORRECT, REC_TITLE_REMOVED, REC_FIXED, REC_SPLIT,
            REC_INITIALS_NAME, REC_INITIALS_LASTNAME, REC_INITIALS, REC_PLACEHOLDER, REC_INVALID]
# Duplicates are NOT a tag: they are an audit metric, computed on the valid recommendations
# of each response from the `name_key` column.

# Name cleaning (cosmetic: does not change the flag). Honorifics and degrees removed at the
# start/end of a name. Ambiguous tokens that are also real names (e.g. 'Ma', 'Don', 'Sr') are excluded.
NAME_TITLES = ['prof', 'professor', 'professorin', 'profa', 'profesor', 'profesora', 'dr', 'dra', 'drs',
               'dr.-ing', 'dr.rer.nat', 'phd', 'ph.d', 'mr', 'mrs', 'ms', 'mx', 'miss', 'sir', 'dame',
               'herr', 'frau', 'sra', 'srta', 'habil', 'priv.-doz', 'msc', 'm.sc', 'mba']
# Particles that belong to the last name when splitting a full name (only if written in lower case,
# so that e.g. 'Van Morrison' or 'Al Gore' keep 'Van'/'Al' as first names)
NAME_PARTICLES = ['van', 'von', 'der', 'den', 'de', 'del', 'della', 'di', 'da', 'dos', 'das', 'du', 'la', 'le',
                  'ter', 'ten', 'zu', 'y', 'bin', 'ibn', 'al', 'el']


# -----------------------------------------------------------------------------
# Valid tag sets (defaults; override in the notebooks / CLI). A response or
# recommendation is valid if ALL its tags are in the set (alone or combined).
# -----------------------------------------------------------------------------
DEFAULT_VALID_RESPONSE_TAGS = [RESPONSE_CORRECT, RESPONSE_WRAPPED, RESPONSE_EXTRA_INCOMPLETE,
                               RESPONSE_PYTHON_LITERAL, RESPONSE_EMBEDDED]
DEFAULT_VALID_REC_TAGS = [REC_CORRECT, REC_TITLE_REMOVED]

# Uncertainty (descriptive figures): hierarchical bootstrap over models, then configurations
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 42


###############################################################################
# VISUALISATION
###############################################################################

# Tag combinations: valid ones in shades of green, the others in shades of red AND hatched
# (green/red alone is not separable for red-green colour-blind readers, so hatching is the
# second channel). Ramps run dark -> light (darkest = most frequent combination); each was
# checked as an ordinal ramp (monotone lightness, visible steps, light end >= 2:1 on white).
VALID_RAMP = ['#0a4419', '#115c25', '#1a7633', '#279143', '#43ad58', '#72c47b']
INVALID_RAMP = ['#6e1212', '#8f1b1b', '#b02725', '#cc3a33', '#e25a4c', '#ee8573']
INVALID_HATCH = '///'
SEGMENT_EDGE_COLOR = 'white'
TOP_COMBINATIONS_PER_FAMILY = 5     # the rest of each family folds into "other valid" / "other invalid"
LABEL_OTHER_VALID = 'other valid'
LABEL_OTHER_INVALID = 'other invalid'
TAG_SEPARATOR_DISPLAY = ' + '       # 'wrapped|extra_incomplete' is shown as 'wrapped + extra_incomplete'
# Compact tag labels for paper figures (legend and axis), so that combinations stay as short as model
# labels (e.g. 'fixed|wrapped|extra_incomplete' -> 'fixed+wrap+extra_inc'). Define them in the caption.
TAG_SEPARATOR_DISPLAY_SHORT = '+'
TAG_SHORT_LABELS = {RESPONSE_PYTHON_LITERAL: 'py_lit', RESPONSE_EMBEDDED: 'embed', RESPONSE_WRAPPED: 'wrap',
                    RESPONSE_EXTRA_INCOMPLETE: 'extra_inc',
                    REC_TITLE_REMOVED: 'title_rm', REC_INITIALS_NAME: 'init_fn', REC_INITIALS_LASTNAME: 'init_ln'}

# One-line tag definitions for figure captions (only the tags shown in a figure are defined).
# Two dictionaries: response and recommendation tags share names with different meanings.
RESPONSE_TAG_DEFINITIONS = {
    RESPONSE_CORRECT: 'strict JSON array of complete recommendations',
    RESPONSE_PYTHON_LITERAL: 'not valid JSON but valid Python literal syntax (e.g. single quotes, True/None, '
                             'a trailing comma before } or ], an unescaped control character such as a tab, or an '
                             'invalid escape sequence such as \\( inside a string)',
    RESPONSE_EMBEDDED: 'JSON surrounded by text',
    RESPONSE_FIXED: 'output truncated, recovered up to the last complete record',
    RESPONSE_WRAPPED: 'JSON object wrapping the requested array',
    RESPONSE_EXTRA_INCOMPLETE: 'at least one recommendation lacks a requested non-name field '
                               '(affiliation, research areas, reason or source)',
    RESPONSE_TOY: 'names disclaimed as fictional or illustrative',
    RESPONSE_EMPTY: 'no content',
    RESPONSE_EMPTY_LIST: 'empty list of recommendations',
    RESPONSE_REFUSED: 'explicit refusal',
    RESPONSE_INVALID: 'unparsable output or no usable names',
}
REC_TAG_DEFINITIONS = {
    REC_CORRECT: 'first and last name given as requested',
    REC_TITLE_REMOVED: 'academic or honorific title removed (e.g. Dr., Prof.)',
    REC_FIXED: 'one name part repeated in the other (e.g. "John Smith" / "Smith"), repetition removed',
    REC_SPLIT: 'full name given in a single field, split into first and last name',
    REC_INITIALS_NAME: 'first name given as initials only',
    REC_INITIALS_LASTNAME: 'last name given as initials only',
    REC_INITIALS: 'first and last name given as initials only',
    REC_PLACEHOLDER: 'template instead of a name (e.g. "Candidate 1")',
    REC_INVALID: 'no usable name',
}
FIGURE_DPI = 300
CAPTION_PRINT_WIDTH = 100            # characters per line when captions are printed in notebooks
FIGURE_FORMATS = ['png', 'pdf']
FORMATS_WITH_TITLES = ['png']       # rule: titles only in PNG (quick looks); PDFs (paper) have none
# Sample-size symbols (matplotlib mathtext, as in LaTeX)
SYMBOL_N_MODELS = r'$N_m$'
SYMBOL_N_RESPONSES = r'$N_r$'
SYMBOL_N_RECOMMENDATIONS = r'$N_{rec}$'


# Paper figures: full A4 text width, low height; vector PDF (PNG for quick looks)
PAPER_FIGURE_WIDTH_IN = 7.0
PAPER_FIGURE_HEIGHT_IN = 2.9          # total height (legend, plot, rotated model labels)
PAPER_FIGURE_WIDTH_RATIOS = [4, 1]   # per-model panel : overall panel (both vertical bars, shared 0-100 axis)
PAPER_FIGURE_FORMAT = ['pdf']
PAPER_LEGEND_ROWS = 2                 # legend above the panels: 2 rows (e.g. 2 x 4 for 8 categories)
# Style (viz/style.py): seaborn style + context, scaled for figures drawn at their printed size.
# 'paper' x 0.75 -> tick labels ~6.6 pt, labels/legend ~7 pt.
PLOT_STYLE = 'ticks'
PLOT_CONTEXT = 'paper'
PLOT_FONT_SCALE = 0.75
# Helvetica-like, TrueType only (Type 42 embedding needs .ttf; Liberation Sans = Arial/Helvetica metrics)
PLOT_FONTS = ['Helvetica', 'Arial', 'Liberation Sans', 'DejaVu Sans']

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
# ANNOTATION (manual validation of inferred labels; tool: code/tools/annotation_tool.html)
###############################################################################
ANNOTATION_TASK_RESPONSES = 'responses'
ANNOTATION_TASK_RECOMMENDATIONS = 'recommendations'
ANNOTATION_TASKS = [ANNOTATION_TASK_RESPONSES, ANNOTATION_TASK_RECOMMENDATIONS]
DEFAULT_ANNOTATION_PER_STRATUM = 20     # items per inferred tag combination (all items if fewer)
DEFAULT_ANNOTATION_MAX_ITEMS = 400      # cap on the sample size
ANNOTATION_TOOL_FILE = str(PROJECT_ROOT / 'code' / 'tools' / 'annotation_tool.html')

