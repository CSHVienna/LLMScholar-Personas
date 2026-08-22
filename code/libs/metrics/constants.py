CALL_KEYS = [
    "model",
    "role",
    "task",
    "location",
    "k",
    "target",
    "field",
    "subfield",
    "language",
    "run_id",
]
PROMPT_KEYS = [c for c in CALL_KEYS if c != "run_id"]

MODEL_ARCHITECTURE_COLS = ["model_access", "model_size", "model_class"]
MODEL_EXTRA_COLS = ["model_family", "model_short_name"]

VALID_FLAGS = {"cleaned", "unchanged"}  # 'valid records' = cleaned + unchanged only
FACTUAL_AUTHOR_COL = "author_found"  # SS-match OR OA-match (any source)

REFUSED_FLAG = "refused"
ETHNICITY_ORDER = ["Asian", "Black", "White", "Hispanic", "American Indian"]
GENDER_ORDER = ["Female", "Male", "Neutral"]

NEEDED_COLS = [
    "model",
    "role",
    "task",
    "location",
    "k",
    "target",
    "field",
    "subfield",
    "language",
    "run_id",
] + [
    "valid_flag",
    "name",
    "lastname",
    "author_status",
    "oa_id",
    "perceived_ethnicity",
    "gt_gender",
    "location_oa_iso",
    "field_status",
    "seniority_status",
    "location_status",
    "oa_works_count",
    "oa_cited_by_count",
    # Structural metrics (Eqs. 6-8): oa_career_age feeds the similarity features;
    # researcher_id identifies authors matched only in Semantic Scholar (no
    # oa_id) so they can be counted as exclusions instead of vanishing.
    "oa_career_age",
    "researcher_id",
]

#######################################################################################################################
# EVALUATION METRICS
#######################################################################################################################

ALL_METRICS = [
    # All-responses denominator
    "validity",
    "refusals",
    # Structural (factual-records denominator) — paper Eqs. 6-8
    "connectedness",
    "similarity",
    # Valid-responses denominator
    "consistency",
    "duplicates",
    "factuality_author",
    # Factual-records denominator
    "factuality_field",
    "factuality_seniority",
    # Bias (social representation, factual-records denominator)
    "bias_location",
    # Diversity
    "div_ethnicity",
    "div_gender",
    "div_location",
    "div_productivity_works",
    "div_productivity_citations",
    # Parity
    "parity_ethnicity",
    "parity_gender",
    "parity_works",
    "parity_citations",
    # Productivity
    'pct_works_low', 
    'pct_works_med', 
    'pct_works_high', 
    'pct_citations_low', 
    'pct_citations_med', 
    'pct_citations_high',
    
    # popularity
    "popularity_works",
    "popularity_citations",
]

PREFIX_GROUPS_METRICS = {
        'factuality_': 'Factuality',
        'bias_':       'Bias',
        'parity_':     'Parity',
        'div_':        'Diversity',
        'pct_works_':        'Publications tertile',
        'pct_citations_':        'Citations tertile',
        'popularity_': 'Popularity'
    }


# Bernoulli (binary 0/1) metrics — paper uses Wilson score CI here.
BINARY_METRICS = {"validity", "refusals"}

PRODUCTIVITY_OA_FIELDS_MAP = {
    "oa_works_count": "works",
    "oa_cited_by_count": "citations",
}
PRODUCTIVITY_TIER_LABELS = ["low", "med", "high"]
PRODUCTIVITY_METRIC_COLS = [
    'parity_works', 
    'parity_citations',
    'popularity_works', 
    'popularity_citations',
    'pct_works_low',  
    'pct_works_med',  
    'pct_works_high',  
    'div_productivity_works',
    'pct_citations_low', 
    'pct_citations_med', 
    'pct_citations_high', 
    'div_productivity_citations',
]

FACTUALITY_METRICS   = ['factuality_author',
                        'factuality_field',
                        'factuality_seniority']

# Location is a *choice*, not a factual error (reviewer's comment, issue #36):
# whether the LLM places an author in the country embedded in the prompt is read
# as social representation, so it groups with the social metrics below. The value
# is unchanged — still the share of location_match among evaluable records.
BIAS_METRICS = ['bias_location']

PARITY_METRICS = ['parity_gender', 
                  'parity_ethnicity', 
                  'parity_works', 
                  'parity_citations']

DIVERSITY_METRICS = ['div_gender', 
                     'div_ethnicity', 
                     'div_location', 
                     'div_productivity_works', 
                     'div_productivity_citations']

POPULARITY_METRICS = ['popularity_works', 
                      'popularity_citations']

PROMINENCE_METRICS = ['pct_works_low', 
                      'pct_works_med', 
                      'pct_works_high', 
                      'pct_citations_low', 
                      'pct_citations_med', 
                      'pct_citations_high']

#######################################################################################################################
# STRUCTURAL METRICS (paper Eqs. 6-8): connectedness + scholarly similarity
#######################################################################################################################
# Both are computed over U-hat_i, the set of unique *factual* authors of a
# response, and both need OpenAlex-side data (the coauthorship graph and the
# author feature vectors). Authors matched only in Semantic Scholar have no
# oa_id, hence no node and no features: they are excluded and counted in the
# exclusion columns below (~12% of factual recommendations).

CONNECTEDNESS_METRIC = "connectedness"
SIMILARITY_METRIC = "similarity"
STRUCTURAL_METRICS = [CONNECTEDNESS_METRIC, SIMILARITY_METRIC]

# Per-response count of U-hat_i members dropped for lack of graph node / features.
STRUCTURAL_EXCLUSION_COLS = {
    CONNECTEDNESS_METRIC: "n_excluded_connectedness",
    SIMILARITY_METRIC: "n_excluded_similarity",
}

# Per-response n that actually entered each formula — the denominator of Eqs. 6-8.
# Published alongside the exclusions because `n_authors_found` is NOT a valid
# denominator for these two metrics: it collapses every author without an oa_id
# into a single entry (drop_duplicates on ['_cid','author_id']), whereas U-hat_i
# here distinguishes them via a composite uid. Subtracting one from the other
# goes negative on ~8% of responses; use these columns instead.
STRUCTURAL_USED_COLS = {
    CONNECTEDNESS_METRIC: "n_used_connectedness",
    SIMILARITY_METRIC: "n_used_similarity",
}

# Feature vector for scholarly similarity. h-index / i10-index are NOT usable:
# factuality_openalex.py leaves them None (absent from the DuckDB snapshot), so
# they are 0% covered. citations_per_work stands in for the missing h-index.
SIMILARITY_FEATURE_COLS = [
    "works_count",         # productivity: raw publication volume
    "cited_by_count",      # impact: raw accumulated citations
    "citations_per_work",  # impact per unit of output (h-index surrogate)
    "career_age",          # career stage: span of active years
    "works_per_year",      # productivity intensity over the career
]

# Minimum share of variance the retained PCA components must explain (Eq. 8).
SIMILARITY_PCA_VARIANCE = 0.90
# Fixed for reproducibility (PCA with svd_solver='full' is deterministic anyway).
SIMILARITY_RANDOM_STATE = 0

TECHNICAL_METRICS = ['validity','refusals','consistency','duplicates'] + FACTUALITY_METRICS
SOCIAL_METRICS = BIAS_METRICS + PARITY_METRICS + DIVERSITY_METRICS + PROMINENCE_METRICS + POPULARITY_METRICS

TECHNICAL_METRICS_NORM = ['validity','refusals_c','duplicates_c'] + FACTUALITY_METRICS

# ── Backward compatibility ────────────────────────────────────────────────────
# Artefacts produced before issue #36 (the 425 MB valid_requests_metadata.csv and
# every effect_sizes.parquet under results/sensitivity_analysis/) still carry the
# old metric name. Apply this on load — as a column rename for wide frames, or on
# the `metric` column for long-format ANOVA output — instead of regenerating them.
# Idempotent: names already canonical are left untouched.
METRIC_RENAME_MAP = {'factuality_location': 'bias_location'}

PERSONA_VARIABLES = ['language_en', 'location_en', 'role_en']
CONTEXT_VARIABLES = ['k', 'field_en', 'subfield_en', 'target_en']

MAIN_CONTEXT_VARIABLES = CONTEXT_VARIABLES.copy()
MAIN_CONTEXT_VARIABLES.remove('subfield_en') 

PROMPT_VAR_GROUPS = {'persona': PERSONA_VARIABLES, 
                    'context': CONTEXT_VARIABLES,
                    'llm':['model'],
                    }

EVALUATION_METRIC_GROUPS = {'technical': TECHNICAL_METRICS, 
                            'social': SOCIAL_METRICS}

PROMPT_TYPE_MAP = {
    "language_en": "persona",
    "role_en": "persona",
    "location_en": "persona",
    
    "field_en": "context",
    "subfield_en": "context",
    "k": "context",
    "target_en": "context",
}


NESTED_METRIC_PAIRS = {
    'consistency': 'validity',
    'duplicates': 'validity',
    'factuality_author': 'validity',

    'factuality_field': 'factuality_author',
    'factuality_seniority': 'factuality_author',
    'bias_location': 'factuality_author',

    'div_gender': 'factuality_author',
    'div_ethnicity': 'factuality_author',
    'div_location': 'factuality_author',
    'div_productivity_works': 'factuality_author',
    'div_productivity_citations': 'factuality_author',
    
    'parity_gender': 'factuality_author',
    'parity_ethnicity': 'factuality_author',
    'parity_works': 'factuality_author',
    'parity_citations': 'factuality_author',
    
    'popularity_works': 'factuality_author',
    'popularity_citations': 'factuality_author',

    'connectedness': 'factuality_author',
    'similarity': 'factuality_author',

    'pct_works_low': 'factuality_author',
    'pct_works_med': 'factuality_author',
    'pct_works_high': 'factuality_author',
    'pct_citations_low': 'factuality_author',
    'pct_citations_med': 'factuality_author',
    'pct_citations_high': 'factuality_author',
    # add other nested pairs here
}

#######################################################################################################################
# NORMALIZATION (TO ENGLISH)
#######################################################################################################################

# ── Normalize field (population) to canonical English ─────────────────────
FIELD_NORM_MAP = {
    "Biología": "Biology",
    "Biologie": "Biology",
    "Física": "Physics",
    "Physik": "Physics",
    "Ciencias de la computación": "Computer Science",
    "Informatik": "Computer Science",
    "Sociología": "Sociology",
    "Soziologie": "Sociology",
    "Psicología": "Psychology",
    "Psychologie": "Psychology",
    "Matemáticas": "Mathematics",
    "Mathematik": "Mathematics",
}
# ── Normalize subfield to canonical English names ────────────────
SUBFIELD_NORM_MAP = {
    "Anatomía": "Anatomy",
    "Anatomie": "Anatomy",
    "Inteligencia artificial": "Artificial Intelligence",
    "Künstliche Intelligenz": "Artificial Intelligence",
    "Kondensierte Materie": "Condensed Matter",
    "Materia condensada": "Condensed Matter",
    "Criminología": "Criminology",
    "Kriminologie": "Criminology",
    "Bildung": "Education",
    "Educación": "Education",
    "Familia": "Family",
    "Familie": "Family",
    "Forensische Psychologie": "Forensic Psychology",
    "Psicología forense": "Forensic Psychology",
    "Neurociencia": "Neuroscience",
    "Neurowissenschaften": "Neuroscience",
    "Teoría de números": "Number theory",
    "Zahlentheorie": "Number theory",
    "Psicología social": "Social Psychology",
    "Sozialpsychologie": "Social Psychology",
    "Ingeniería de software": "Software Engineering",
    "Softwareentwicklung": "Software Engineering",
    "Topologie": "Topology",
    "Topología": "Topology",
}
# ── Normalize language to canonical English names ────────────────
LANGUAGE_NORM_MAP = {
    "english": "English",
    "german": "German",
    "spanish": "Spanish",
    "English": "English",
    "German": "German",
    "Spanish": "Spanish",
}
# ── Normalize location to canonical English ─────────────────────
LOCATION_NORM_MAP = {
    "Germany": "Germany",
    "Deutschland": "Germany",
    "Alemania": "Germany",
    "Canada": "Canada",
    "Canadá": "Canada",
    "Kanada": "Canada",
    "Japan": "Japan",
    "Japón": "Japan",
    "Japon": "Japan",
    "South Africa": "South Africa",
    "Sudáfrica": "South Africa",
    "Südafrika": "South Africa",
    "Sudafrica": "South Africa",
    "Ecuador": "Ecuador",
}


BENCHMARK_DEMOGRAPHIC_ATTRIBUTES = [
    "gender",
    "ethnicity",
    "prominence_pub",
    "prominence_cit",
]

BENCHMARK_MODEL_GROUPS = ["model_access", "model_size", "model_class"]
BENCHMARK_MODEL_GROUPS_LABEL_MAP = {
    "model_access": "Access",
    "model_size": "Size",
    "model_class": "Reasoning",
}

# metric -> status column emitted by the factuality pipeline, consumed by
# aggregators.aggregate_factuality_task. Replaces the pre-refactor
# BENCHMARK_FACTUALITY_FIELD_METRICS_MAP, which was deleted with
# constants_old.py in 789bbb0 and pointed at a retired column schema
# (`fact_author_field`, `fact_epoch_requested`, ...).
# Status values are {<prefix>_match | <prefix>_mismatch | <prefix>_unknown |
# not_applicable}; only the first two are evaluable — same rule as step 6 of
# scripts/metrics/build_valid_calls.py.
FACTUALITY_TASK_STATUS_COLS = {
    "factuality_field": "field_status",
    "factuality_seniority": "seniority_status",
    "bias_location": "location_status",
}

BENCHMARK_PER_ATTEMPT_COLS = BENCHMARK_MODEL_GROUPS + [
    "model",
    "grounded",
    "temperature",
    "date",
    "time",
    "task_name",
    "task_param",
    "task_attempt",
]
BENCHMARK_PER_REQUEST_COLS = BENCHMARK_MODEL_GROUPS + [
    "model",
    "grounded",
    "temperature",
    "date",
    "time",
    "task_name",
    "task_param",
]

# ── Sub-population dimensions ──────────────────────────────────────────────────
# Consumed by aggregators.aggregate_*_by_subpop to compute social metrics per
# subgroup; for each subpop value the GT is filtered to that subgroup (e.g.
# parity_gender in Japan is compared against the female/male distribution of
# authors in Japan, not the global GT).

# Canonical order of the 6 fields in the experiment — used to sort plot axes
# and table columns.
FIELD_ORDER = [
    "Biology",
    "Computer Science",
    "Mathematics",
    "Physics",
    "Psychology",
    "Sociology",
]
# ISO-2 codes of the 5 countries in the experiment (Ecuador, Japan, Germany,
# Canada, South Africa) — same format as oa_country_code in factuality_full.csv.
LOCATION_ORDER = ["EC", "JP", "DE", "CA", "ZA"]
# Canonical order of the persona-prompting languages — always English, Spanish,
# German in plots/tables. Lower-case because that's how they live in the CSV.
LANGUAGE_ORDER = ["english", "spanish", "german"]
LANGUAGE_LABELS = {"english": "English", "spanish": "Spanish", "german": "German"}

# Individual dimensions available for sub-populating.
BENCHMARK_SUBPOPULATION_DIMS = ["field", "location"]
# Combinations iterated by the notebook: field only, location only, and crossed
# field × location.
BENCHMARK_SUBPOPULATION_COMBOS = [["field"], ["location"], ["field", "location"]]


BENCHMARK_MODEL_GROUP_LABEL_MAP = {
    "open": "Open",
    "proprietary": "Proprietary",
    "S": "Small",
    "M": "Medium",
    "L": "Large",
    "XL": "Extra Large",
    "non-reasoning": "Disabled",
    "reasoning": "Enabled",
}
# ── Normalize task to canonical English ─────────────────────
TASK_NORM_MAP = {
    "buscando posibles contrataciones": "seeking potential hires",
    "buscando un(a) asesor(a)": "seeking an advisor",
    "potenzielle Einstellungen suchen": "seeking potential hires",
    "einen Betreuer(in) suchen": "seeking an advisor",
}
# ── Normalize target (seniority) to canonical English ─────────────────────
TARGET_NORM_MAP = {
    "Profesor(a) Sénior": "Senior Professor",
    "Seniorprofessor(in)": "Senior Professor",
    "Profesor(a) Júnior": "Junior Professor",
    "Juniorprofessor(in)": "Junior Professor",
}
# ── Normalize role to canonical English ─────────────────────
ROLE_NORM_MAP = {
    "PhD student": "PhD student",
    "Estudiante de doctorado": "PhD student",
    "Doktorand(in)": "PhD student",
    "Director/Recruiter": "Director/Recruiter",
    "Director(a)/Reclutador(a)": "Director/Recruiter",
    "Direktor(in)/Rekrutierende(r)": "Director/Recruiter",
}


#######################################################################################################################
# FIGURE / PLOT SETTINGS
#######################################################################################################################

FIG_DPI = 600
FONT_SCALE = 1.55


#######################################################################################################################
# FACTUALITY STATUS FLAGS
#######################################################################################################################
# Values emitted by the per-step factuality scripts in the `location_status`,
# `seniority_status`, `field_status` columns. Kept here so downstream notebooks
# and metrics aggregators reference a single source of truth.

FACTUALITY_AUTHOR_HALLUCINATED = "hallucinated"
FACTUALITY_STATUS_NOT_APPLICABLE = "not_applicable"


def factuality_status_flags(prefix: str) -> dict:
    """Return the {MATCH,MISMATCH,UNKNOWN,NOT_APPLICABLE} flags used by a
    per-attribute factuality script (e.g. prefix='field' → 'field_match',
    'field_mismatch', 'field_unknown', 'not_applicable').
    """
    return {
        "MATCH": f"{prefix}_match",
        "MISMATCH": f"{prefix}_mismatch",
        "UNKNOWN": f"{prefix}_unknown",
        "NOT_APPLICABLE": FACTUALITY_STATUS_NOT_APPLICABLE,
    }


# Legacy aliases (location is the original consumer).
FACTUALITY_LOCATION_STATUS = factuality_status_flags("location")
FACTUALITY_STATUS_MATCH = FACTUALITY_LOCATION_STATUS["MATCH"]
FACTUALITY_STATUS_MISMATCH = FACTUALITY_LOCATION_STATUS["MISMATCH"]
FACTUALITY_STATUS_UNKNOWN = FACTUALITY_LOCATION_STATUS["UNKNOWN"]


#######################################################################################################################
# SENIORITY
#######################################################################################################################
# Short-form (used in tables/plots) of the canonical Senior/Junior Professor
# values produced by TARGET_NORM_MAP.

SENIORITY_SHORT_MAP = {
    "Senior Professor": "Senior",
    "Junior Professor": "Junior",
}

# Same idea as SENIORITY_SHORT_MAP but also accepts the ES/DE variants of the
# LLM `target` column (used by factuality_seniority.py).
LLM_TARGET_TO_BUCKET = {
    "Senior Professor": "Senior",
    "Profesor(a) Sénior": "Senior",
    "Seniorprofessor(in)": "Senior",
    "Junior Professor": "Junior",
    "Profesor(a) Júnior": "Junior",
    "Juniorprofessor(in)": "Junior",
}


#######################################################################################################################
# COUNTRY ISO-2 CODES
#######################################################################################################################
# Maps every observed `location` value from the LLM (EN/ES/DE variants) to its
# ISO alpha-2 code so downstream code can compare against oa_country_code.

LLM_COUNTRY_TO_ISO = {
    "Ecuador": "EC",
    "Japan": "JP",
    "Japón": "JP",
    "Germany": "DE",
    "Alemania": "DE",
    "Deutschland": "DE",
    "Canada": "CA",
    "Canadá": "CA",
    "Kanada": "CA",
    "South Africa": "ZA",
    "Sudáfrica": "ZA",
    "Südafrika": "ZA",
}
