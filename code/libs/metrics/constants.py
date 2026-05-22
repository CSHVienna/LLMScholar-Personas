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
]

#######################################################################################################################
# EVALUATION METRICS
#######################################################################################################################

ALL_METRICS = [
    # All-responses denominator
    "validity",
    "refusals",
    # Valid-responses denominator
    "consistency",
    "duplicates",
    "factuality_author",
    # Factual-records denominator
    "factuality_field",
    "factuality_seniority",
    "factuality_location",
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
    "pct_low_works",
    "pct_med_works",
    "pct_high_works",
    "pct_low_citations",
    "pct_med_citations",
    "pct_high_citations",
    # popularity
    "popularity_works",
    "popularity_citations",
]

# Bernoulli (binary 0/1) metrics — paper uses Wilson score CI here.
BINARY_METRICS = {"validity", "refusals"}

PRODUCTIVITY_OA_FIELDS_MAP = {
    "oa_works_count": "works",
    "oa_cited_by_count": "citations",
}
PRODUCTIVITY_TIER_LABELS = ["low", "med", "high"]
PRODUCTIVITY_METRIC_COLS = [
    "parity_works",
    "parity_citations",
    "popularity_works",
    "popularity_citations",
    "pct_low_works",
    "pct_med_works",
    "pct_high_works",
    "div_productivity_works",
    "pct_low_citations",
    "pct_med_citations",
    "pct_high_citations",
    "div_productivity_citations",
]

FACTUALITY_METRICS = [
    "factuality_author",
    "factuality_field",
    "factuality_seniority",
    "factuality_location",
]
PARITY_METRICS = [
    "parity_ethnicity",
    "parity_gender",
    "parity_works",
    "parity_citations",
]

TECHNICAL_METRICS = ["validity", "refusals_c", "duplicates_c"] + FACTUALITY_METRICS
SOCIAL_METRICS = PARITY_METRICS

PERSONA_VARIABLES = ["language_en", "location_en", "role_en"]
CONTEXT_VARIABLES = ["k", "field_en", "target_en"]


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

# ── Sub-population dimensions (PLAN.md Tarea 3) ────────────────────────────────
# Consumido por aggregators.aggregate_*_by_subpop para calcular social metrics
# por subgrupo; para cada subpop value el GT se filtra a ese subgrupo (e.g.
# parity_gender en Japón se compara contra la distribución female/male de
# autores en Japón, no el GT global).

# Orden canónico de los 6 fields del experimento — usado para ordenar ejes
# de plots y columnas de tablas.
FIELD_ORDER = [
    "Biology",
    "Computer Science",
    "Mathematics",
    "Physics",
    "Psychology",
    "Sociology",
]
# Códigos ISO-2 de los 5 países del experimento (Ecuador, Japón, Alemania,
# Canadá, Sudáfrica) — mismo formato que oa_country_code en factuality_full.csv.
LOCATION_ORDER = ["EC", "JP", "DE", "CA", "ZA"]
# Orden canónico de los idiomas del persona prompting — siempre English,
# Spanish, German en plots/tablas. Lower-case porque así viven en los CSV.
LANGUAGE_ORDER = ["english", "spanish", "german"]
LANGUAGE_LABELS = {"english": "English", "spanish": "Spanish", "german": "German"}

# Dimensiones individuales por las que se puede sub-poblacionar.
BENCHMARK_SUBPOPULATION_DIMS = ["field", "location"]
# Combinaciones a iterar en el notebook: por field solo, por location solo,
# y cruzado field × location.
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
