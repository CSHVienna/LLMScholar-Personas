
CALL_KEYS   = ['model', 'role', 'task', 'location', 'k', 'target', 'field', 'subfield', 'language', 'run_id']
PROMPT_KEYS = [c for c in CALL_KEYS if c != 'run_id']

MODEL_ARCHITECTURE_COLS = ['model_access','model_size','model_class']
MODEL_EXTRA_COLS = ['model_family', 'model_short_name']

VALID_FLAGS = {'cleaned', 'unchanged'}  # 'valid records' = cleaned + unchanged only
FACTUAL_AUTHOR_COL = 'author_found'     # SS-match OR OA-match (any source)

REFUSED_FLAG = 'refused'
ETHNICITY_ORDER = ['Asian', 'Black', 'White', 'Hispanic', 'American Indian']
GENDER_ORDER = ['Female', 'Male', 'Neutral']

NEEDED_COLS = (
    ['model', 'role', 'task', 'location', 'k', 'target', 'field', 'subfield', 'language', 'run_id']
    + ['valid_flag', 'name', 'lastname', 'author_status', 'oa_id',
       'perceived_ethnicity', 'gt_gender', 'location_oa_iso',
       'field_status', 'seniority_status', 'location_status',
       'oa_works_count', 'oa_cited_by_count']
)

#######################################################################################################################
# EVALUATION METRICS
#######################################################################################################################

ALL_METRICS = [
    # All-responses denominator
    'validity', 'refusals',
    
    # Valid-responses denominator
    'consistency', 
    'duplicates', 
    'factuality_author', 
    
    # Factual-records denominator
    'factuality_field', 'factuality_seniority', 'factuality_location',

    # Diversity
    'div_ethnicity', 'div_gender', 'div_location','div_productivity_works','div_productivity_citations',

    # Parity
    'parity_ethnicity', 'parity_gender','parity_works', 'parity_citations',

    # Productivity
    'pct_works_low', 'pct_works_med', 'pct_works_high', 
    'pct_citations_low', 'pct_citations_med', 'pct_citations_high',
    
    # popularity
    'popularity_works', 'popularity_citations',
]

PREFIX_GROUPS_METRICS = {
        'factuality_': 'Factuality',
        'parity_':     'Parity',
        'div_':        'Diversity',
        'pct_works_':        'Publications tertile',
        'pct_citations_':        'Citations tertile',
        'popularity_': 'Popularity'
    }


# Bernoulli (binary 0/1) metrics — paper uses Wilson score CI here.
BINARY_METRICS = {'validity', 'refusals'}

PRODUCTIVITY_OA_FIELDS_MAP = {'oa_works_count': 'works', 
                              'oa_cited_by_count': 'citations'}
PRODUCTIVITY_TIER_LABELS = ['low', 'med', 'high']
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

FACTUALITY_METRICS   = ['factuality_author', 'factuality_field', 'factuality_seniority', 'factuality_location']
PARITY_METRICS = ['parity_gender', 'parity_ethnicity', 'parity_works', 'parity_citations']
DIVERSITY_METRICS = ['div_gender', 'div_ethnicity', 'div_location'] # 'div_productivity_works', 'div_productivity_citations']
POPULARITY_METRICS = ['pct_works_low', 'pct_works_med', 'pct_works_high', 'pct_citations_low', 'pct_citations_med', 'pct_citations_high', 'popularity_works', 'popularity_citations']

TECHNICAL_METRICS = ['validity','refusals','consistency','duplicates'] + FACTUALITY_METRICS
SOCIAL_METRICS = PARITY_METRICS + DIVERSITY_METRICS + POPULARITY_METRICS

TECHNICAL_METRICS_NORM = ['validity','refusals_c','duplicates_c'] + FACTUALITY_METRICS

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
    'factuality_location': 'factuality_author',
    
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
    'Biología': 'Biology',   
    'Biologie': 'Biology',
    'Física':   'Physics',   
    'Physik':   'Physics',
    'Ciencias de la computación': 'Computer Science', 
    'Informatik': 'Computer Science',
    'Sociología': 'Sociology',  
    'Soziologie': 'Sociology',
    'Psicología': 'Psychology', 
    'Psychologie': 'Psychology',
    'Matemáticas': 'Mathematics', 
    'Mathematik': 'Mathematics',
}
# ── Normalize subfield to canonical English names ────────────────
SUBFIELD_NORM_MAP = {'Anatomía':'Anatomy',
                     'Anatomie':'Anatomy',

                     'Inteligencia artificial':'Artificial Intelligence',
                     'Künstliche Intelligenz':'Artificial Intelligence',

                     'Kondensierte Materie': 'Condensed Matter',
                     'Materia condensada': 'Condensed Matter',

                     'Criminología': 'Criminology',
                     'Kriminologie': 'Criminology',
                     
                     'Bildung':'Education',
                     'Educación':'Education',
                     
                     'Familia':'Family',
                     'Familie':'Family',
                     
                     'Forensische Psychologie':'Forensic Psychology',
                     'Psicología forense':'Forensic Psychology',
                     
                     'Neurociencia':'Neuroscience',
                     'Neurowissenschaften':'Neuroscience',

                     'Teoría de números':'Number theory',
                     'Zahlentheorie':'Number theory',

                     'Psicología social':'Social Psychology',
                     'Sozialpsychologie':'Social Psychology',

                     'Ingeniería de software':'Software Engineering',
                     'Softwareentwicklung':'Software Engineering',
                     
                     'Topologie':'Topology',
                     'Topología':'Topology',
                     }
# ── Normalize language to canonical English names ────────────────
LANGUAGE_NORM_MAP = {
    'english': 'English', 
    'german': 'German', 
    'spanish': 'Spanish',
    'English': 'English', 
    'German': 'German', 
    'Spanish': 'Spanish',
}
# ── Normalize location to canonical English ─────────────────────
LOCATION_NORM_MAP = {
    'Germany': 'Germany', 
    'Deutschland': 'Germany', 
    'Alemania': 'Germany',
    'Canada': 'Canada', 
    'Canadá': 'Canada', 
    'Kanada': 'Canada',
    'Japan': 'Japan', 
    'Japón': 'Japan', 
    'Japon': 'Japan',
    'South Africa': 'South Africa', 
    'Sudáfrica': 'South Africa',
    'Südafrika': 'South Africa', 
    'Sudafrica': 'South Africa',
    'Ecuador': 'Ecuador',
}
# ── Normalize task to canonical English ─────────────────────
TASK_NORM_MAP = {
    'buscando posibles contrataciones': 'seeking potential hires',
    'buscando un(a) asesor(a)':         'seeking an advisor',
    'potenzielle Einstellungen suchen': 'seeking potential hires',
    'einen Betreuer(in) suchen':        'seeking an advisor',
}
# ── Normalize target (seniority) to canonical English ─────────────────────
TARGET_NORM_MAP = {
    'Profesor(a) Sénior':  'Senior Professor', 'Seniorprofessor(in)': 'Senior Professor',
    'Profesor(a) Júnior':  'Junior Professor', 'Juniorprofessor(in)': 'Junior Professor',
}
# ── Normalize role to canonical English ─────────────────────
ROLE_NORM_MAP = {
    'PhD student':                   'PhD student',
    'Estudiante de doctorado':       'PhD student',
    'Doktorand(in)':                 'PhD student',
    'Director/Recruiter':            'Director/Recruiter',
    'Director(a)/Reclutador(a)':     'Director/Recruiter',
    'Direktor(in)/Rekrutierende(r)': 'Director/Recruiter',
}