# Re-exported so callers can `from libs.visuals.constants import FIG_DPI`.
try:
    from libs.metrics.constants import ALL_METRICS, FIG_DPI  # noqa: F401
except ImportError:  # PYTHONPATH=code/libs/
    from metrics.constants import ALL_METRICS, FIG_DPI  # noqa: F401

PROMPT_VAR_COLORS = {"context": "#7ab0d3", "persona": "#efaf76"}

# Long-form ethnicity labels used in plots (BERT/ethnicolr cascade output,
# kept distinct from the short labels in libs.metrics.constants.ETHNICITY_ORDER
# which feed metric aggregation).
ETHNICITY_PLOT_ORDER = [
    "White",
    "Asian",
    "Hispanic or Latino",
    "Black or African American",
    "Unknown",
]
ETHNICITY_PLOT_COLORS = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#aaaaaa"]
ETHNICITY_PLOT_COLOR_MAP = dict(zip(ETHNICITY_PLOT_ORDER, ETHNICITY_PLOT_COLORS))

# Paper-figure dimensions used by the analysis notebooks.
FIG_WIDTH_PAPER = 15
FIG_HEIGHT_PAPER = 0.8

# Layout constants (same values as the defaults in grouped_metrics.py).
TICK_FONT_SIZE = 8
LABEL_FONT_SIZE = 11
SPINE_LW = 0.3
SECTION_GAP = 0.4

# Quadrant scatter colors (moved from quadrant_scatter.py).
QUADRANT_COLORS = {
    "Q1": "#2ca02c",  # top-right    — high tech, high social  (green)
    "Q2": "#1f77b4",  # top-left     — low  tech, high social  (blue)
    "Q3": "#a5a5a5",  # bottom-left  — low  tech, low  social  (gray)
    "Q4": "#ff7f0e",  # bottom-right — high tech, low  social  (orange)
}

PROMPT_VAR_COLORS = {
    "context": "#7ab0d3",
    "persona": "#efaf76",
    "llm": "#9a9a9a",
    "model": "#9a9a9a",
}

from libs.metrics.constants import ETHNICITY_MAP, GENDER_MAP  # noqa: E402,F401

MODEL_FAMILY_COLORS = {
    # ─────────────────────────
    # BLUES
    # ─────────────────────────
    "gpt-oss": "#1E3A8A",
    "gpt-4": "#2563EB",
    "gemini": "#7DD3FC",
    "deepseek": "#06B6D4",
    # ─────────────────────────
    # REDS / ORANGES
    # ─────────────────────────
    "gemma": "#F9A8D4",
    "mistral": "#DC2626",
    "mixtral": "#F97316",
    "llama": "#78350F",
    # ─────────────────────────
    # GREENS / YELLOWS
    # ─────────────────────────
    "phi": "#86EFAC",
    "falcon": "#10B981",
    "qwen": "#84A532",
    "qwq": "#166534",
    # ─────────────────────────
    # BLACKS / GRAYS
    # ─────────────────────────
    "olmo": "#000000",
    "smollm": "#374151",
    "dolphin": "#94A3B8",
    "yi": "#D1D5DB",
}
MODEL_FAMILY_ORDER = [
    "gpt-oss",
    "gpt-4",
    "gemini",
    "deepseek",
    "gemma",
    "mistral",
    "mixtral",
    "llama",
    "phi",
    "falcon",
    "qwen",
    "qwq",
    "olmo",
    "smollm",
    "dolphin",
    "yi",
]

TICK_COLOR = "#828282"

PLOT_LABELS = {
    "refusals": "Refusals",
    "validity": "Validity",
    "consistency": "Consistency",
    "duplicates": "Duplicates",

    "factuality_name": "Fact. $_{name}$",
    "factuality_field": "Fact. $_{field}$",
    "factuality_seniority": "Fact. $_{seniority}$",
    "factuality_author": "Fact. $_{author}$",

    "bias_location": "Bias $_{loc.}$",
    
    "div_gender": "Div. $_{gen.}$",
    "div_ethnicity": "Div. $_{eth.}$",
    "div_location": "Div. $_{loc.}$",
    "div_productivity_works": "Div. $_{pub.}$",
    "div_productivity_citations": "Div. $_{cit.}$",
    
    "parity_gender": "Par. $_{gender}$",
    "parity_ethnicity": "Par. $_{eth.}$",
    "parity_works": "Par. $_{pub.}$",
    "parity_citations": "Par. $_{cit.}$",
    
    "popularity_works": "Pop. $_{pub.}$",
    "popularity_citations": "Pop. $_{cit.}$",
    
    "pct_citations_high": "% H cit.",
    "pct_citations_med": "% M cit.",
    "pct_citations_low": "% L cit.",
    "pct_works_high": "% H pub.",
    "pct_works_med": "% M pub.",
    "pct_works_low": "% L pub.",

    "connectedness": "Connect.",
    "similarity": "Sim.",
}

PLOT_LABELS |= {
    "role_en": "Role",
    "location_en": "Location",
    "language_en": "Language",
    "k": "k",
    "field_en": "Field",
    "subfield_en": "Subfield",
    "target_en": "Seniority",
    "model": "Model",
}

PLOT_METRICS = [
    # Display order: the 5_validity block (refusals, validity, consistency,
    # duplicates), then the factuality cascade (name → field → seniority →
    # author). Entries not in ALL_METRICS are dropped below.
    "refusals",
    "validity",
    "consistency",
    "duplicates",
    "factuality_name",
    "factuality_field",
    "factuality_seniority",
    "factuality_author",
    # Diversity block (order: gender, ethnicity, publications, citations)
    "div_gender",
    "div_ethnicity",
    "div_location",
    "div_productivity_works",  # publications
    "div_productivity_citations",  # citations
    # Parity block — SAME order as diversity
    "parity_gender",  # vs Semantic Scholar GT distribution
    "parity_ethnicity",  # vs Semantic Scholar GT distribution
    "parity_works",  # publications, vs uniform 1/3 tier
    "parity_citations",  # citations, vs uniform 1/3 tier
    # Bias block (social representation, see BIAS_METRICS)
    "bias_location",
    # Popularity (fraction in top productivity tier within field)
    "popularity_works",
    "popularity_citations",
    # Structural block (paper Eqs. 6-8)
    "connectedness",
    "similarity",
]
PLOT_METRICS = [m for m in PLOT_METRICS if m in ALL_METRICS]

PLOT_TECHNICAL_METRICS = [
    m
    for m in PLOT_METRICS
    if m in ["refusals", "validity", "consistency", "duplicates"]
    or m.startswith("factuality")
]
PLOT_SOCIAL_METRICS = [
    m
    for m in PLOT_METRICS
    if m.startswith(("bias_", "div_", "parity_", "popularity_"))
    or m in ("connectedness", "similarity")
]
METRIC_TYPES = {"technical": PLOT_TECHNICAL_METRICS, "social": PLOT_SOCIAL_METRICS}

METRIC_DIRECTIONS = {
    "refusals": "↓",
    "validity": "↑",
    "consistency": None,
    "duplicates": "↓",

    "factuality_name": "↑",
    "factuality_field": "↑",
    "factuality_seniority": "↑",
    "factuality_author": "↑",
    
    "div_gender": None,
    "div_ethnicity": None,
    "div_location": None,
    "div_productivity_works": None,
    "div_productivity_citations": None,
    
    "parity_gender": "↑",
    "parity_ethnicity": "↑",
    "parity_works": "↑",
    "parity_citations": "↑",
    
    "bias_location": None,
    "popularity_works": None,
    "popularity_citations": None,
    
    "connectedness": None,
    "similarity": None,
}

LANGUAGE_ORDER = ["English", "Spanish", "German"]
LANGUAGE_MAP = {"Spanish": "es", "German": "de", "English": "en"}
LANGUAGE_COLORS = {"English": "#4A90D9", "Spanish": "#E8A838", "German": "#5DB85D"}

LOCATION_ORDER = ["Ecuador", "Germany", "Japan", "Canada", "South Africa"]
LOCATION_MAP = {
    "Ecuador": "ECU",
    "Germany": "DEU",
    "Japan": "JPN",
    "Canada": "CAN",
    "South Africa": "ZAF",
}
LOCATION_COLORS = {
    "Ecuador": "#E8A838",
    "Germany": "#5DB85D",
    "Japan": "#C0392B",
    "Canada": "#4A90D9",
    "South Africa": "#8E44AD",
}

FIELD_ORDER = [
    "Biology",
    "Computer Science",
    "Mathematics",
    "Physics",
    "Psychology",
    "Sociology",
]
FIELD_MAP = {
    "Sociology": "Soc.",
    "Psychology": "Psyc.",
    "Mathematics": "Math.",
    "Biology": "Bio.",
    "Computer Science": "CS",
    "Physics": "Phys.",
}
FIELD_COLORS = {
    "Biology": "#27AE60",
    "Computer Science": "#2980B9",
    "Mathematics": "#8E44AD",
    "Physics": "#E67E22",
    "Psychology": "#C0392B",
    "Sociology": "#16A085",
}

SUBFIELD_ORDER = [
    "Number theory",
    "Topology",
    "Artificial Intelligence",
    "Software Engineering",
    "Condensed Matter",
    "Education",
    "Anatomy",
    "Neuroscience",
    "Family",
    "Criminology",
    "Forensic Psychology",
    "Social Psychology"]

SUBFIELD_MAP = {
    "Number theory": "Numb.:MATH",
    "Topology": "Topo.:MATH",
    "Artificial Intelligence": "AI:CS",
    "Software Engineering": "SE:CS",
    "Condensed Matter": "CM:PHY",
    "Education": "EDU:PHY",
    "Anatomy": "Anat.:BIO",
    "Neuroscience": "Neuro:BIO",
    "Family": "Family:SOC",
    "Criminology": "Crim.:SOC",
    "Forensic Psychology": "Foren.:PSY",
    "Social Psychology": "Social:PSY",
}
SUBFIELD_COLORS = {
    "Number theory": "#8E44AD",
    "Topology": "#8E44AD",
    "Artificial Intelligence": "#2980B9",
    "Software Engineering": "#2980B9",
    "Condensed Matter": "#E67E22",
    "Education": "#E67E22",
    "Anatomy": "#27AE60",
    "Neuroscience": "#27AE60",
    "Family": "#16A085",
    "Criminology": "#16A085",
    "Forensic Psychology": "#C0392B",
    "Social Psychology": "#C0392B",
}

SENIORITY_ORDER = ["Junior Professor", "Senior Professor"]
SENIORITY_MAP = {"Junior Professor": "Junior", "Senior Professor": "Senior"}
SENIORITY_COLORS = {"Junior Professor": "#E8703A", "Senior Professor": "#C0392B"}

ROLE_ORDER = ["PhD student", "Director/Recruiter"]
ROLE_MAP = {"PhD student": "Student", "Director/Recruiter": "Recruiter"}
ROLE_COLORS = {"PhD student": "#2980B9", "Director/Recruiter": "#E67E22"}

TASK_ORDER = ["seeking an advisor", "seeking potential hires"]
TASK_COLORS = {"seeking an advisor": "#8E44AD", "seeking potential hires": "#2980B9"}
TASK_MAP = {
    "seeking an advisor": "Advisor S.",
    "seeking potential hires": "Candidate S.",
}
TASK_MAP_LONG = {
    "seeking an advisor": "Advisor Search",
    "seeking potential hires": "Candidate Search",
}

SIZE_ORDER = ["T", "S", "M", "L", "XL"]
SIZE_MAP = {"T": "Tiny", "S": "Small", "M": "Medium", "L": "Large", "XL": "XL"}
SIZE_MAP_REV = {v: k for k, v in SIZE_MAP.items()}
SIZE_COLORS = {
    "T": "#2ECC71",
    "S": "#3498DB",
    "M": "#9B59B6",
    "L": "#E67E22",
    "XL": "#E74C3C",
}

ACCESS_ORDER = ["open-weight", "proprietary"]
ACCESS_MAP = {"open-weight": "Open", "proprietary": "Proprietary"}

REASONING_ORDER = [True, False]

ORIGIN_ORDER = ['USA','FRA','CHN','ARE']

MODEL_ARCHITECTURE_MAP = {
    "model_access": "Access",
    "model_size": "Size",
    "model_class": "Reasoning",
    "model_origin": "Origin",
}
MODEL_ARCHITECTURE_GROUPS = {
    "model_access": ACCESS_ORDER,
    "model_size": SIZE_ORDER,
    "model_class": REASONING_ORDER,
    "model_origin": ORIGIN_ORDER,
}


K_ORDER = [1, 5, 10]
K_COLORS = {1: "#2980B9", 5: "#E67E22", 10: "#8E44AD"}

ORDER_MAP = {
    "language_en": LANGUAGE_ORDER,
    "role_en": ROLE_ORDER,
    "location_en": LOCATION_ORDER,
    "field_en": FIELD_ORDER,
    "target_en": SENIORITY_ORDER,
    "k": K_ORDER,
}


# ISO-2 → display name for the countries that show up as recommendation targets
# in the location Sankey (issue #37). Only the ones that reach the figure need an
# entry; anything else falls back to its ISO code.
COUNTRY_ISO_NAMES = {
    "US": "United States",
    "JP": "Japan",
    "DE": "Germany",
    "ES": "Spain",
    "GB": "United Kingdom",
    "CA": "Canada",
    "ZA": "South Africa",
    "CH": "Switzerland",
    "MX": "Mexico",
    "AU": "Australia",
    "BR": "Brazil",
    "CO": "Colombia",
    "FR": "France",
    "CL": "Chile",
    "EC": "Ecuador",
    "IT": "Italy",
    "NL": "Netherlands",
    "CN": "China",
    "AT": "Austria",
    "SE": "Sweden",
    "AR":"Argentina"
}

METRICS_NAME_MAP = {
    "refusals": "Refusals",
    "validity": "Validity",
    "consistency": "Consistency",
    "duplicates": "Duplicates",
    
    "factuality_name": "Name factuality",
    "factuality_field": "Field factuality",
    "factuality_seniority": "Seniority factuality",
    "factuality_author": "Author factuality",

    "div_gender": "Gender diversity",
    "div_ethnicity": "Ethinicity diversity",
    "div_location": "Location diversity",
    "div_productivity_citations": "Citations diversity",
    "div_productivity_works": "Works diversity",

    "parity_gender": "Gender parity",
    "parity_ethnicity": "Ethinicity parity",
    "parity_works": "Works parity",
    "parity_citations": "Citations parity",

    "pct_works_high": "Works upper tertile",
    "pct_works_med": "Works middle tertile",
    "pct_works_low": "Works lower tertile",
    "pct_citations_high": "Citations upper tertile",
    "pct_citations_med": "Citations middle tertile",
    "pct_citations_low": "Citations lower tertile",

    "popularity_works": "Popularity-based works",
    "popularity_citations": "Popularity-based citations",
    
    "bias_location": "Location bias",
    
    "connectedness": "Connectedness",
    "similarity": "Similarity",
}
