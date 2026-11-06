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
DEFAULT_TMP_DIR = str(PROJECT_ROOT / 'tmp')           # spill / scratch space (git-ignored), never outside the project
DEFAULT_DUCKDB_TEMP_DIR = f'{DEFAULT_TMP_DIR}/duckdb'
DEFAULT_DUCKDB_THREADS = 12          # bounded defaults (shared server: 64 GB / 14 cores allowed)
DEFAULT_DUCKDB_MEMORY = '40GB'

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
                                            # no disclaimer (completeness is assessed per recommendation)
# b) Issues of a response with >= 1 named recommendation (any number of these)
RESPONSE_PYTHON_LITERAL = 'python_literal'  # python syntax instead of JSON: single quotes, True/None, trailing commas
RESPONSE_EMBEDDED = 'embedded'              # JSON surrounded by prose ("Here you go: [...]")
RESPONSE_TRUNCATED = 'truncated'            # truncated (e.g. token budget hit mid-record): parsed up to the last
                                            # complete record, trailing partial entry discarded
RESPONSE_WRAPPED = 'wrapped'                # JSON root is an object (e.g. {"candidates": [...]}), not the requested array
RESPONSE_TOY = 'toy'                        # names disclaimed as illustrative / fictional / plausible examples
# c) Responses without any named recommendation (exactly one of these, alone)
RESPONSE_EMPTY = 'empty'                    # no content (empty string or whitespace only)
RESPONSE_EMPTY_LIST = 'empty_list'          # an object wrapping only empty lists, e.g. {"candidates": []}
RESPONSE_REFUSED = 'refused'                # declines to recommend (natural-language justification, no names)
RESPONSE_INVALID = 'invalid'                # API error, unparsable / broken JSON, [], no person objects, only templates

RESPONSE_ISSUE_TAGS = [RESPONSE_PYTHON_LITERAL, RESPONSE_EMBEDDED, RESPONSE_TRUNCATED,
                       RESPONSE_WRAPPED, RESPONSE_TOY]
RESPONSE_FAILURE_TAGS = [RESPONSE_EMPTY, RESPONSE_EMPTY_LIST, RESPONSE_REFUSED, RESPONSE_INVALID]
RESPONSE_TAGS = [RESPONSE_CORRECT] + RESPONSE_ISSUE_TAGS + RESPONSE_FAILURE_TAGS
# Responses that carry named recommendations
RESPONSE_TAGS_WITH_RECOMMENDATIONS = [RESPONSE_CORRECT] + RESPONSE_ISSUE_TAGS

# Requested fields. Names are assessed by the recommendation tags; a recommendation is
# complete if all the other (non-name) requested keys are present in its object (whatever their value,
# also empty). Completeness is a per-recommendation attribute (missing_fields / is_complete), not a tag.
REQUIRED_FIELDS = ['name', 'lastname', 'current_affiliations', 'areas_of_research_or_work', 'reason', 'source']
NAME_FIELDS = ['name', 'lastname']
COMPLETENESS_FIELDS = [f for f in REQUIRED_FIELDS if f not in NAME_FIELDS]

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
                                PARSE_TRUNCATED: RESPONSE_TRUNCATED}   # strict parsing is not an issue

# Keys that identify a person-like object (compared lower-cased and stripped)
PERSON_FIRSTNAME_KEYS = ['name', 'firstname', 'first_name', 'given_name', 'full_name', 'fullname',
                         'nombre', 'vorname']
PERSON_LASTNAME_KEYS = ['lastname', 'last_name', 'surname', 'family_name',
                        'apellido', 'apellidos', 'nachname']

# Name values meaning "no value" (compared lower-cased and stripped)
NAME_MISSING_VALUES = ['', 'n/a', 'n.a.', 'none', 'null', 'nil', 'unknown', '-', '--', '...', '?', 'tbd']
# ... and compared as written: 'NA' means "not available", but 'Na' is a name (Li Na; the Korean surname Na)
NAME_MISSING_VALUES_EXACT = ['NA', 'na']
# Template names: the schema or an instruction echoed back instead of a name
NAME_TEMPLATE_VALUES = ['name', 'lastname', 'last_name', 'firstname', 'first name', 'last name', 'surname',
                        'full name', 'your name', 'your last name', 'your lastname',
                        'nombre', 'apellido', 'apellidos', 'vorname', 'nachname']
# Template names as regular expressions (lower-cased value), e.g. "Candidate 1", "Another Candidate".
# Real-looking fabricated names (e.g. "John Doe") are NOT templates: they are left to author matching (B.6) and the factuality metric.
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
REC_REPETITION_REMOVED = 'repetition_removed'  # one name part repeated inside the other ("John Smith"/"Smith"): removed
REC_SPLIT = 'split'                         # only one part given, with >1 word: split into first + last name
REC_INITIALS_NAME = 'initials_name'         # first name is initials only ("M."/"Smith")
REC_INITIALS_LASTNAME = 'initials_lastname' # last name is initials only ("Michael"/"G.")
REC_INITIALS = 'initials'                   # both first and last name are initials only ("S. M."/"D.")
REC_PLACEHOLDER = 'placeholder'             # template instead of a name ("Candidate 1", "Last Name"); alone
REC_INVALID = 'invalid'                     # no usable name (missing, or a single word only)
REC_TAGS = [REC_CORRECT, REC_TITLE_REMOVED, REC_REPETITION_REMOVED, REC_SPLIT,
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
DEFAULT_VALID_RESPONSE_TAGS = [RESPONSE_CORRECT, RESPONSE_WRAPPED,
                               RESPONSE_PYTHON_LITERAL, RESPONSE_EMBEDDED]
DEFAULT_VALID_REC_TAGS = [REC_CORRECT, REC_TITLE_REMOVED]

# Uncertainty (descriptive figures): hierarchical bootstrap over models, then configurations
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 42


###############################################################################
# TAG COMBINATIONS (shared by the tag statistics, figures, captions and annotation).
# Visual constants (colours, sizes, fonts, labels) are in viz/constants.py.
###############################################################################
TOP_COMBINATIONS_PER_FAMILY = 5     # the rest of each family folds into "other valid" / "other invalid"
LABEL_OTHER_VALID = 'other valid'
LABEL_OTHER_INVALID = 'other invalid'

# One-line tag definitions for figure captions (only the tags shown in a figure are defined).
# Two dictionaries: response and recommendation tags share names with different meanings.
RESPONSE_TAG_DEFINITIONS = {
    RESPONSE_CORRECT: 'strict JSON array of recommendations, no other issue',
    RESPONSE_PYTHON_LITERAL: 'not valid JSON but valid Python literal syntax (e.g. single quotes, True/None, '
                             'a trailing comma before } or ], an unescaped control character such as a tab, or an '
                             'invalid escape sequence such as \\( inside a string)',
    RESPONSE_EMBEDDED: 'JSON surrounded by text',
    RESPONSE_TRUNCATED: 'output truncated, recovered up to the last complete record',
    RESPONSE_WRAPPED: 'JSON object wrapping the requested array',
    RESPONSE_TOY: 'names disclaimed as fictional or illustrative',
    RESPONSE_EMPTY: 'no content',
    RESPONSE_EMPTY_LIST: 'empty list of recommendations',
    RESPONSE_REFUSED: 'explicit refusal',
    RESPONSE_INVALID: 'unparsable output or no usable names',
}
REC_TAG_DEFINITIONS = {
    REC_CORRECT: 'first and last name given as requested',
    REC_TITLE_REMOVED: 'academic or honorific title removed (e.g. Dr., Prof.)',
    REC_REPETITION_REMOVED: 'one name part repeated in the other (e.g. "John Smith" / "Smith"), repetition removed',
    REC_SPLIT: 'full name given in a single field, split into first and last name',
    REC_INITIALS_NAME: 'first name given as initials only',
    REC_INITIALS_LASTNAME: 'last name given as initials only',
    REC_INITIALS: 'first and last name given as initials only',
    REC_PLACEHOLDER: 'template instead of a name (e.g. "Candidate 1")',
    REC_INVALID: 'no usable name',
}


###############################################################################
# ANNOTATION (manual validation of inferred labels; tool: code/tools/annotation_tool.html)
###############################################################################
ANNOTATION_TASK_RESPONSES = 'responses'
ANNOTATION_TASK_RECOMMENDATIONS = 'recommendations'
ANNOTATION_TASK_MATCHING = 'matching'   # verification: recommended name -> OA person (annotation/verification.py)
ANNOTATION_TASK_AUTHORS = 'authors'     # verification: GT field, seniority, country of an OA profile
ANNOTATION_TASK_DEMOGRAPHICS = 'demographics'   # blind: perceived gender + ethnicity from the name (annotation/demographics.py)
ANNOTATION_TAG_TASKS = [ANNOTATION_TASK_RESPONSES, ANNOTATION_TASK_RECOMMENDATIONS]   # blind tag tasks (tasks.py)
ANNOTATION_TASKS = ANNOTATION_TAG_TASKS + [ANNOTATION_TASK_MATCHING, ANNOTATION_TASK_AUTHORS,
                                           ANNOTATION_TASK_DEMOGRAPHICS]
# Renamed tags: old name in saved annotations -> current name (applied by the tool and the evaluation)
ANNOTATION_TAG_ALIASES = {ANNOTATION_TASK_RESPONSES: {'fixed': RESPONSE_TRUNCATED},
                          ANNOTATION_TASK_RECOMMENDATIONS: {'fixed': REC_REPETITION_REMOVED}}
DEFAULT_ANNOTATION_PER_STRATUM = 20     # items per inferred tag combination (all items if fewer)
DEFAULT_ANNOTATION_MAX_ITEMS = 400      # cap on the sample size
ANNOTATION_TOOL_FILE = str(PROJECT_ROOT / 'code' / 'tools' / 'annotation_tool.html')



###############################################################################
# GROUND TRUTH (step A; OpenAlex, read-only). Code in libs/groundtruth/ and libs/demographics/.
###############################################################################
DEFAULT_OPENALEX_DB = f'{DEFAULT_DATA_DIR}/openalex_20260923.duckdb'
DEFAULT_GT_DIR = f'{DEFAULT_RESULTS_DIR}/groundtruth'
DEFAULT_GT_AUTHORSHIPS_DIR = f'{DEFAULT_GT_DIR}/authorships'      # bucket=<b>/*.parquet (hive)
DEFAULT_GT_AUTHORS_DIR = f'{DEFAULT_GT_DIR}/authors'              # bucket-<bb>.parquet
DEFAULT_GT_PAIRS_DIR = f'{DEFAULT_GT_DIR}/coauthor_pairs'         # chunk=<c>/bucket=<b>/*.parquet (A.0b, intermediate)
DEFAULT_GT_EDGES_DIR = f'{DEFAULT_GT_DIR}/coauthor_edges'         # bucket-<bb>.parquet (A.1a, the edge list)
DEFAULT_GT_NAMES_DIR = f'{DEFAULT_GT_DIR}/names'                  # bucket-<bb>.parquet
DEFAULT_GT_DEMOGRAPHICS_DIR = f'{DEFAULT_GT_DIR}/demographics'    # bucket-<bb>.parquet (+ keys/, ethnicity/)

# Authors are split into buckets (author id % n) so that every step runs one bucket per process (GNU parallel)
DEFAULT_GT_N_BUCKETS = 64
DEFAULT_GT_WORKS_CHUNK_SIZE = 2_000_000    # works rows per A.0 chunk (~240 resumable chunks)
DEFAULT_GT_DEMOGRAPHICS_CHUNK = 20_000     # names per raceBERT checkpoint (part file)
DEFAULT_GT_NAME_SHARDS = 64                # distinct raceBERT names split by crc32(name) % n (one job per shard)

# Works counted (works x authorships): no paratext, no retracted works, publication year in [min, max]
GT_MIN_YEAR = 1900
GT_REFERENCE_YEAR = 2025                 # academic_age = GT_REFERENCE_YEAR - first_pub_year; also the max year
DEFAULT_GT_COAUTHOR_MAX_TEAM = 100       # coauthor pairs only from works with <= this many authors (5000-author works -> 25M pairs each)
DEFAULT_GT_MIN_FIELD_SHARE = 0.0         # share of an author's works (primary topic) in a field to belong to it (0 = any work)

# Seniority from academic age: junior <= JUNIOR_MAX, senior >= SENIOR_MIN, mid-career in between (own class);
# sensitivity: a single cut-off (junior <= cut-off < senior)
SENIORITY_MID = 'mid'
SENIORITY_CLASSES = [SENIORITY_JUNIOR, SENIORITY_MID, SENIORITY_SENIOR]
DEFAULT_SENIORITY_JUNIOR_MAX = 10
DEFAULT_SENIORITY_SENIOR_MIN = 20
DEFAULT_SENIORITY_SINGLE_CUTOFF = 15

# Prominence classes: rank-based bins within field x seniority (percent rank; edges between L/M and M/H)
BIN_LOW, BIN_MEDIUM, BIN_HIGH = 'L', 'M', 'H'
BIN_CLASSES = [BIN_LOW, BIN_MEDIUM, BIN_HIGH]
DEFAULT_GT_BIN_EDGES = [1 / 3, 2 / 3]   # percent-rank cut points: [0, e1) -> L, [e1, e2) -> M, [e2, 1] -> H
DEFAULT_GT_BIN_LABELS = list(BIN_CLASSES)   # one label per interval; any number of classes (e.g. 5: 0.2 0.4 0.6 0.8)
GT_BIN_METRICS = {'productivity': 'n_works', 'influence': 'n_citations', 'collaboration': 'n_coauthors'}
DEFAULT_GT_TOP_PERCENTILE = 5.0          # popularity bias: top x % within field x seniority, by each GT_TOP_METRICS metric
GT_TOP_METRICS = {'productivity': 'n_works', 'influence': 'n_citations'}   # -> is_top_<name>, shares top_<name>

# Perceived gender (nomquamgender, on the first name) and ethnicity (raceBERT, on the full name)
GENDER_FEMALE, GENDER_MALE, GENDER_NEUTRAL, GENDER_UNKNOWN = 'female', 'male', 'neutral', 'unknown'
GENDER_CLASSES = [GENDER_FEMALE, GENDER_MALE, GENDER_UNKNOWN]
# Perceived gender labels from p_female (A.4): cut points e_1 < ... < e_k in (0, 1) and k + 1 labels for the
# intervals [0, e_1), [e_1, e_2), ..., [e_k, 1]; 'unknown' without p_female. Any number of classes, e.g. 5:
# edges [0.1, 0.4, 0.6, 0.9], labels [male, mostly male, neutral, mostly female, female]
DEFAULT_GENDER_EDGES = [0.1, 0.9]
DEFAULT_GENDER_LABELS = [GENDER_MALE, GENDER_NEUTRAL, GENDER_FEMALE]
DEFAULT_GENDER_UNCERTAINTY = 0.1         # nomquamgender default: classify only if uncertainty <= threshold
ETHNICITY_ASIAN, ETHNICITY_BLACK, ETHNICITY_HISPANIC, ETHNICITY_WHITE = 'asian', 'black', 'hispanic', 'white'
ETHNICITY_OTHER, ETHNICITY_UNKNOWN = 'other', 'unknown'
ETHNICITY_CLASSES = [ETHNICITY_ASIAN, ETHNICITY_BLACK, ETHNICITY_HISPANIC, ETHNICITY_WHITE, ETHNICITY_OTHER,
                     ETHNICITY_UNKNOWN]
RACEBERT_LABELS = {'api': ETHNICITY_ASIAN, 'nh_black': ETHNICITY_BLACK, 'hispanic': ETHNICITY_HISPANIC,
                   'nh_white': ETHNICITY_WHITE, 'aian': ETHNICITY_OTHER}   # AIAN merged into Other
DEFAULT_ETHNICITY_MIN_PROB = 0.5         # label = argmax class if its probability >= this, else unknown
# Demographics annotation (perceived, from the name): the classes the annotator picks from (+ 'unknown' = cannot tell).
# Ethnicity is annotated in finer, origin-based classes (disjoint, defined by the language / region a name comes from)
# and mapped onto the raceBERT classes for the evaluation: 'main' follows the US census (OMB 1997) convention raceBERT
# was trained on (Florida voters); the other mappings are robustness checks, never chosen by fit.
ANNOTATION_GENDER_CLASSES = [GENDER_MALE, GENDER_FEMALE, GENDER_NEUTRAL, GENDER_UNKNOWN]
ANN_EUROPEAN, ANN_HISPANIC, ANN_PORTUGUESE_BRAZILIAN, ANN_MENA = 'european', 'hispanic', 'portuguese_brazilian', 'mena'
ANN_EAST_ASIAN, ANN_SOUTH_ASIAN, ANN_BLACK, ANN_AIAN = 'east_asian', 'south_asian', 'black', 'aian'
ANNOTATION_ETHNICITY_CLASSES = [ANN_EUROPEAN, ANN_HISPANIC, ANN_PORTUGUESE_BRAZILIAN, ANN_MENA, ANN_EAST_ASIAN,
                                ANN_SOUTH_ASIAN, ANN_BLACK, ANN_AIAN, ETHNICITY_UNKNOWN]
ANNOTATION_MAPPING_MAIN = 'main'
_ETHNICITY_MAIN = {ANN_EUROPEAN: ETHNICITY_WHITE, ANN_HISPANIC: ETHNICITY_HISPANIC, ANN_PORTUGUESE_BRAZILIAN: ETHNICITY_WHITE,
                   ANN_MENA: ETHNICITY_WHITE, ANN_EAST_ASIAN: ETHNICITY_ASIAN, ANN_SOUTH_ASIAN: ETHNICITY_ASIAN,
                   ANN_BLACK: ETHNICITY_BLACK, ANN_AIAN: ETHNICITY_OTHER, ETHNICITY_UNKNOWN: ETHNICITY_UNKNOWN}
ANNOTATION_ETHNICITY_MAPPINGS = {
    ANNOTATION_MAPPING_MAIN: _ETHNICITY_MAIN,
    'portuguese_brazilian_hispanic': {**_ETHNICITY_MAIN, ANN_PORTUGUESE_BRAZILIAN: ETHNICITY_HISPANIC},
    'mena_asian': {**_ETHNICITY_MAIN, ANN_MENA: ETHNICITY_ASIAN},
    'mena_other': {**_ETHNICITY_MAIN, ANN_MENA: ETHNICITY_OTHER},
}
ANNOTATION_MAPPING_FINE = 'fine'      # no mapping: the annotated fine classes vs the inferred ones (where each group lands)
DEFAULT_RACEBERT_BATCH_SIZE = 256

# GT descriptives (groundtruth/stats.py, notebook 0_gt.ipynb): DuckDB aggregations over the A.1 / A.4 outputs with
# small, bounded resources (interactive use on the shared server; tens of millions of rows, never loaded into pandas)
GT_STATS_DUCKDB_THREADS = 4
GT_STATS_DUCKDB_MEMORY = '16GB'
GT_STATS_METRICS = ['academic_age', 'n_works', 'n_citations', 'n_coauthors', 'h_index', 'citations_per_paper_per_year']
GT_STATS_QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

# xpac = OpenAlex "expansion pack" (works added in 2025 from DataCite / repositories: datasets, repository copies,
# theses; poorer metadata). Excluded by default, as in OpenAlex itself.
DEFAULT_GT_INCLUDE_XPAC = False

# OpenAlex "authors" that are organisations (e.g. 'USDOE, Washington, DC (United States)', 'Cisco Team'):
# display name matching this pattern (case-insensitive) AND no ORCID -> is_person = False (excluded from the
# target population and the demographics). Words that are also common surnames (Press, Service, Corp) are left out.
ORGANIZATION_NAME_PATTERN = (
    r"(?i)\b(univ|universit[a-z]*|universidad|universität|department|dept|institut[a-z]*|inc|ltd|llc|gmbh|"
    r"corporation|company|society|association|committee|council|consortium|collaboration|group|team|project|"
    r"network|office|agency|ministry|government|laborator[a-z]*|lab|labs|cent(er|re)|foundation|hospital|clinic|"
    r"school|college|academy|faculty|library|libraries|museum|media|publish[a-z]*|editorial[s]?|staff|anonymous|"
    r"unknown|authors|commission|bureau|organi[sz]ation|programme|survey|federal|national|international|"
    r"investigators|et al)\b|\(|\)|@|\d|&")

# Coauthorship network (connectedness) among given authors: a filter of the edge list (DEFAULT_GT_EDGES_DIR)
DEFAULT_GT_COAUTHORSHIP_DIR = f'{DEFAULT_GT_DIR}/coauthorship'


###############################################################################
# AUTHOR MATCHING (step B.6; code in libs/matching/; the factuality METRIC is computed from it). B.6a candidates (heavy, once) -> B.6b decision (thresholds).
###############################################################################
DEFAULT_MATCHING_DIR = f'{DEFAULT_RESULTS_DIR}/matching'
DEFAULT_MATCH_CANDIDATES_DIR = f'{DEFAULT_MATCHING_DIR}/candidates'       # B.6a stages, shared by every variant
DEFAULT_MATCH_AUTHORS_DIR = f'{DEFAULT_MATCHING_DIR}/authors/main'          # B.6b <model>.parquet; variants: authors/<variant>/

# Name-match rules, strictest first: a recommendation uses only its strictest rule with any match
MATCH_EXACT_DISPLAY = 'exact_display'          # all name tokens = the display name's (any order)
MATCH_EXACT_ALTERNATIVE = 'exact_alternative'  # all name tokens = an alternative name's (any order)
MATCH_SHORT = 'short'                          # first + last token
MATCH_INITIALS = 'initials'                    # recommended first name is initials only: initial + last name
MATCH_FUZZY = 'fuzzy'                          # Jaro-Winkler on full names (as written or flipped), same surname token
MATCH_RULES = [MATCH_EXACT_DISPLAY, MATCH_EXACT_ALTERNATIVE, MATCH_SHORT, MATCH_INITIALS, MATCH_FUZZY]

DEFAULT_MATCH_N_SHARDS = 64          # fuzzy rule: recommended surname tokens split by crc32(token) % n (one job per shard)
DEFAULT_MATCH_JW_FLOOR = 0.90        # B.6a stores fuzzy pairs with jw_full >= floor (any B.6b threshold must be >= floor)
# B.6b fuzzy match: jw_full >= JW_FULL and jw_first >= JW_FIRST and jw_last >= JW_LAST, i.e. BOTH name parts close
# (jw_full alone accepts any 'Alexander X' for 'Neville Alexander'). PROVISIONAL: revise on the full data + annotations.
DEFAULT_MATCH_JW_FULL = DEFAULT_MATCH_JW_FLOOR
DEFAULT_MATCH_JW_FIRST = 0.90
DEFAULT_MATCH_JW_LAST = 0.90
MATCH_SENIORITY_THREE = 'three'      # junior <= DEFAULT_SENIORITY_JUNIOR_MAX, senior >= DEFAULT_SENIORITY_SENIOR_MIN
MATCH_SENIORITY_SINGLE = 'single'    # junior <= DEFAULT_SENIORITY_SINGLE_CUTOFF < senior

# Benchmark metrics (plan §2.3; code in libs/metrics/): <dir>/config/<model>.parquet, <dir>/response/<model>.parquet
DEFAULT_METRICS_DIR = f'{DEFAULT_RESULTS_DIR}/metrics/main'          # robustness checks: metrics/robustness/<check>/

# Ways to choose the metric COLUMNS of a figure (metrics/summary.select_metrics; notebooks: 'technical', ...).
# Keys are metric keys of metrics/summary.METRICS; any of these names, a subgroup or a list of keys can be passed.
_SOCIAL_ATTRIBUTES = ['gender', 'ethnicity', 'location', 'productivity', 'influence', 'collaboration']
METRIC_COLUMN_SETS = {
    'technical': ['refusals', 'validity', 'behaviour_consistency', 'recommendation_consistency', 'duplicates'],
    'factuality': ['factuality_name', 'factuality_name_field', 'factuality_name_seniority', 'factuality_author'],
    'parity': [f"parity_{a}" for a in _SOCIAL_ATTRIBUTES],
    'diversity': [f"diversity_{a}" for a in _SOCIAL_ATTRIBUTES],
    'popularity': ['popularity_productivity', 'popularity_influence', 'popularity_location'],
    'cohesion': ['connectedness', 'similarity'],
    'consistency': ['behaviour_consistency', 'recommendation_consistency'],
}
METRIC_COLUMN_SETS['technical_factuality'] = METRIC_COLUMN_SETS['technical'] + METRIC_COLUMN_SETS['factuality']
METRIC_COLUMN_SETS['popularity_cohesion'] = METRIC_COLUMN_SETS['popularity'] + METRIC_COLUMN_SETS['cohesion']
METRIC_COLUMN_SETS['social'] = (METRIC_COLUMN_SETS['parity'] + METRIC_COLUMN_SETS['diversity']
                                + METRIC_COLUMN_SETS['popularity'] + METRIC_COLUMN_SETS['cohesion'])
METRIC_COLUMN_SETS['all'] = METRIC_COLUMN_SETS['technical_factuality'] + METRIC_COLUMN_SETS['social']
# The columns of the paper's figures (one per metric family; the full sets go to the appendix)
METRIC_COLUMN_SETS['paper'] = ['refusals', 'validity', 'recommendation_consistency', 'duplicates', 'factuality_author',
                               'parity_gender', 'diversity_gender', 'popularity_influence', 'similarity']

# RQ3: socio-technical trade-off (analysis/tradeoff.py). Per model, the sum of the metrics of each axis; metrics where
# lower is better (refusals, duplicates) enter as complements (1 - x), so larger is better on both axes.
RQ3_TECHNICAL_METRICS = ['validity', 'refusals', 'duplicates', 'factuality_name', 'factuality_name_field',
                         'factuality_name_seniority', 'factuality_author']            # max 7
RQ3_MODES = ['relative', 'absolute']   # quadrant split: per-axis medians of the models | half of each axis maximum
RQ3_SOCIAL_METRICS = list(METRIC_COLUMN_SETS['parity'])                                    # all 6 parities: max 6

# RQ2 contrasts (analysis/deviations.py): reference value per variable; not listed = the first value of
# VARIABLE_VALUES. A nested variable (subfield) uses, within each parent, the listed value if it belongs to the
# parent, else the parent's first value.
RQ2_REFERENCE_LEVELS = {VAR_LOCATION: 'Germany', VAR_FIELD: 'Physics'}
