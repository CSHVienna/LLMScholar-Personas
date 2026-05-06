from . import io

################################################################################################################
# Other constants
################################################################################################################
import numpy as np
NONE = ['', ' ', None, 'None', 'nan', 'NaN', np.nan, 'null', 'Null', 'NULL', 'N/A', 'n/a', 'N/a', 'n/A']
INF = [np.inf, -np.inf]
ID_STR_LEN = 7
TRUE_STR = 'True'

################################################################################################################
# Folders structure
################################################################################################################
TEMPERATURE_FOLDER_PREFIX = 'temperature_'
METADATA_DIR = 'metadata'
SIMILARITIES_DIR = 'similarities'


################################################################################################################
# APS/OA data constants
################################################################################################################
APS_OA_AUTHORS_FN = 'authors.csv'
APS_OA_PUBLICATIONS_FN = 'publications.csv'
APS_OA_AUTHORS_MAPPING_FN = 'authors_mapping.csv'
APS_OA_PUBLICATIONS_MAPPING_FN = 'publications_mapping.csv'
APS_OA_ALTERNATIVE_NAMES_FN = 'alternative_names.csv'
APS_OA_AUTHORS_INSTITUTION_YEAR_FN = 'authors_institution_year.csv'
APS_OA_INSTITUTIONS_FN = 'institution.csv'
APS_OA_AUTHORSHIPS_FN = 'authorships.csv'
APS_OA_CITATIONS_FN = 'citations.csv'
APS_OA_AUTHORS_DEMOGRAPHICS_FN = 'authors_demographics.csv'
APS_OA_PUBLICTION_TOPICS = 'publications_topic.csv'
APS_OA_TOPICS_FN = 'topics.csv'
APS_OA_DISCIPLINES_DEMOGRAPHICS_FN = 'disciplines_author_demographics.csv'
APS_OA_AUTHORS_STATS_FN = 'authors_stats.csv'
APS_OA_AUTHOR_STATS_FN = 'authors_aps_stats.csv'
APS_OA_PCA_MODEL_FN = 'pca_cosine_similarity_model.h5'

AUTHORS_FN = 'authors.json'
PUBLICATIONS_FN = 'publications.json'
AUTHORS_DEMOGRAPHICS_FN = 'authors_demographics.json'
AUTHORS_APS_STATS_FN = 'aps_author_stats.json'
DISCIPLINES_FN = 'aps_disciplines.json'
PUBLICATION_CLASS_FN = 'aps_publication_classifications.json'
AUTHORS_AFFILIATIONS_FN = 'author_affiliations.json'
AFFILIATIONS_FN = 'affiliations.json'
COAUTHOR_NETWORKS_FN = 'aps_coauthor_networks.json'
INSTITUTIONS_STATS_FN = 'institutions_stats.json'
AUTHORS_RANKINGS_FN = 'author_rankings_with_percentile.json'
AUTHORS_APS_RANKINGS_FN = 'aps_author_rankings_with_percentile.json'

APS_AUTHORS_FN = 'authors.csv'
APS_AUTHOR_NAMES_FN = 'author_names.csv'
APS_AUTHORSHIPS_FN = 'authorships.csv'
APS_CITATIONS_FN = 'citations.csv'
APS_PUBLICATIONS_FN = 'publications.csv'
APS_DISCIPLINES_FN = 'disciplines.csv'
APS_PUBLICTION_TOPICS = 'publication_topics.csv'
APS_TOPIC_TYPES_FN = 'topic_types.csv'
APS_AREAS_FN = 'areas.csv'
APS_CONCEPTS_FN = 'concepts.csv'

APS_TOPIC_DISCIPLINE_ID = 1

RANKING_METRICS = {'publications':'works_count',
                   'citations':'cited_by_count',
                   'h_index':'h_index',
                   'i10_index':'i10_index',
                   'e_index':'e_index',
                   'citation_publication_age':'citations_per_paper_age',
                   'mean_citedness_2yr':'two_year_mean_citedness'}

APS_RANKING_METRICS = {'publications':'aps_works_count',
                       'citations':'aps_cited_by_count',
                       'h_index':'aps_h_index',
                       'i10_index':'aps_i10_index',
                       'e_index':'aps_e_index',
                       'citation_publication_age':'aps_citations_per_paper_age'}

APS_CAREER_AGE_COL = 'aps_career_age'
OA_CAREER_AGE_COL = 'career_age'

APS_PRESTIGE_METRICS_COL = list(APS_RANKING_METRICS.values()) 
OA_PRESTIGE_METRICS_COL = list(RANKING_METRICS.values())
ALL_PRESTIGE_METRICS_COL = APS_PRESTIGE_METRICS_COL + OA_PRESTIGE_METRICS_COL
APS_SCHOLARLY_METRICS_COL = APS_PRESTIGE_METRICS_COL + [APS_CAREER_AGE_COL]
OA_SCHOLARLY_METRICS_COL =  OA_PRESTIGE_METRICS_COL + [OA_CAREER_AGE_COL]
ALL_SCHOLARLY_METRICS_COL = APS_SCHOLARLY_METRICS_COL + OA_SCHOLARLY_METRICS_COL


################################################################################################################
# Demographic constants
################################################################################################################
UNKNOWN_STR = 'Unknown'

GENDER_UNISEX = 'Unisex'
GENDER_FEMALE = 'Female'
GENDER_MALE = 'Male'
GENDER_LIST = [GENDER_FEMALE, GENDER_MALE, GENDER_UNISEX, UNKNOWN_STR]

ETHNICITY_BLACK = 'Black or African American'
ETHNICITY_ASIAN = 'Asian'
ETHNICITY_WHITE = 'White'
ETHNICITY_LATINO = 'Hispanic or Latino'
ETHNICITY_MULTIPLE = 'Multiple'
ETHNICITY_AMERICAN_INDIAN = 'American Indian and Alaska Native'
ETHNICITY_LIST = [ETHNICITY_BLACK,ETHNICITY_LATINO,ETHNICITY_WHITE,ETHNICITY_ASIAN,UNKNOWN_STR] #,ETHNICITY_MULTIPLE,ETHNICITY_AMERICAN_INDIAN]
ETHNICITY_SHORT_DICT = {ETHNICITY_BLACK:'Black',
                        ETHNICITY_ASIAN:ETHNICITY_ASIAN,
                        ETHNICITY_WHITE:ETHNICITY_WHITE,
                        ETHNICITY_LATINO:'Latino',
                        ETHNICITY_MULTIPLE:'Multi.',
                        ETHNICITY_AMERICAN_INDIAN:'American\nIndian',
                        UNKNOWN_STR:UNKNOWN_STR
                        }

DEMOGRAPHICX_ETHNICITY_MAPPING = {
    'asian': ETHNICITY_ASIAN,
    'black': ETHNICITY_BLACK,
    'hispanic': ETHNICITY_LATINO,
    'white': ETHNICITY_WHITE,
}

ETHNICOLOR_ETHNICITY_MAPPING_CODE = {
    'pctblack': 'black',        # Percent Non-Hispanic Black Only
    'pctapi': 'api',            # Percent Non-Hispanic Asian and Pacific Islander Only
    'pctaian': 'aian',          # Percent Non-Hispanic American Indian and Alaskan Native
    'pcthispanic': 'hispanic',  # Percent Hispanic Origin 
    'pct2prace': 'prace',       # Percent Non-Hispanic of Two or More Races
    'pctwhite': 'white',        # Percent Non-Hispanic White Only
}

ETHNICOLOR_ETHNICITY_MAPPING = {
    'api' : ETHNICITY_ASIAN, 
    'aian': ETHNICITY_AMERICAN_INDIAN, 
    'black': ETHNICITY_BLACK, 
    'hispanic': ETHNICITY_LATINO, 
    'prace': ETHNICITY_MULTIPLE,
    'white': ETHNICITY_WHITE
}

ETHNICOLOR_ETHNICITY_COLS = ['pctwhite', 'pctblack', 'pctapi', 'pctaian', 'pct2prace', 'pcthispanic']

ETHNICITIES_TO_PREDICT_GENDER = [ETHNICITY_BLACK, ETHNICITY_ASIAN, ETHNICITY_WHITE, ETHNICITY_LATINO, UNKNOWN_STR]

NO_LAUREATE = 'None'
NOBEL_CATEGORIES = ['Physics', 'Chemistry', 'Economics', 'Medicine', 'Peace', 'Literature', NO_LAUREATE]
NOBEL_CATEGORIES_RENAME = {'Physics':'Phys.', 'Chemistry':'Chem.', 'Economics':'Econ.', 'Medicine':'Medi.', 'Peace':'Pea.', 'Literature':'Lite.', NO_LAUREATE:NO_LAUREATE}
NOBEL_DECADES = [0] + list(range(1900,2030,10))
NOBEL_DECADES = [str(d) for d in NOBEL_DECADES] 

DEMOGRAPHIC_ATTRIBUTE_GENDER = 'gender'
DEMOGRAPHIC_ATTRIBUTE_ETHNICITY = 'ethnicity'
DEMOGRAPHIC_ATTRIBUTES = [DEMOGRAPHIC_ATTRIBUTE_GENDER,DEMOGRAPHIC_ATTRIBUTE_ETHNICITY]

PROMINENCE_CATEGORIES = ["low", "mid", "high", "elite"]


################################################################################################################
# Response constants
################################################################################################################
MAX_LETTERS_RESPONSE = 100
MAX_WORDS_RESPONSE = 10
ERROR_KEYWORDS_LC = ['"error"', 'fatalerror', '.runners(position', 'taba_keydown', 'unable to provide', 
                     'addtogroup', 'onitemclick', 'getinstance', "i'm stuck", '.datasource', 'getclient', 
                     'phone.toolstrip', 'actordatatype', 'baseactivity', 'setcurrent_company', '.clearfest', 
                     'getdata_suffix', '.texture_config', 'translator_concurrent', 'the above code will generate', 
                     'the actual implementation', 'texas.selection', 'allowedcreator', 'congratulations', 'extension',
                     'outreturns', 'ridiculous', 'adomnode', '.dropout', 'egg-enabled', 'podem/form', 'nvanonymousua',
                     'temptingordermaker']
NO_OUTPUT_MSG = ['No matching scientists found', 'No applicable scientists found', 'No applicable physicists found', 'No physicists meet the criteria', 'No real scientists meet the criteria', 'No relevant scientists found', 'No valid entries found', 'Other Scientist', 'Networks Group at University of Tokyo', 'Fermi National Accelerator Laboratory','RADIO ASTRONOMY TEAM','Nuclear Reactor Team']

NO_OUTPUT_KEYWORDS = ['No matching', 'No applicable', 'meet the criteria', 'No relevant', 'scientists', 'No valid', 'Other Scientist', 'physicist', 'researcher', 'University', 'Laboratory', 'data not available','RADIO ASTRONOMY TEAM', 'Nuclear Reactor Team', 'Nobel Prize', 'Full Name']
LLMCALLER_OUTPUT_INVALID_JSON_EMPTY = 'No JSON-like structure found in the response'
LLMCALLER_OUTPUT_INVALID_JSON_EXTRA_DATA = 'Invalid JSON format: Extra data'
LLMCALLER_OUTPUT_INVALID_JSON_MISSING_ATTRIBUTE = "Invalid JSON format: Expecting"
LLMCALLER_OUTPUT_INVALID_JSON_MISSING_ATTRIBUTE_SEMICOLON = "Invalid JSON format: Expecting ':' delimeter"
LLMCALLER_OUTPUT_INVALID_JSON_MISSING_ATTRIBUTE_COMMA = "Invalid JSON format: Expecting ','"
LLMCALLER_OUTPUT_INVALID_JSON_PROPERTY_NAME = 'Invalid JSON format: Expecting property name enclosed in double quotes'
LLMCALLER_OUTPUT_INVALID_JSON_CHAR = "Invalid JSON format: Invalid"
LLMCALLER_OUTPUT_SCHEMA_FAILED = "Schema validation failed:"
LLMCALLER_OUTPUT_PARSING_ERROR = "JSCONDecodeError:"
LLMCALLER_OUTPUT_EXPECTING_VALUE = "Invalid JSON format: Expecting value: "

EXPERIMENTS_BAD_CONTENT_TEXTS_LC = ['no specific scientists found', 'i was unable to', 
                                    'it appears that there is limited publicly available information', 
                                    'no physicists meet the specified criteria',
                                    'no specific scientists identified due to the uniqueness and complexity of the criteria',
                                    'none']


################################################################################################################
# Refusal constants
################################################################################################################

FN_REFUSALS_CORPUS = 'corpus.jsonl'
FN_REFUSALS_CORPUS_EMBEDDINGS = 'corpus_embeddings.npy'
FN_REFUSALS_CLUSTERED = 'refusals_clustered.csv'
FN_REFUSALS_CLUSTER_NAMES_MAPPING = 'cluster_names_mapping.json'
FN_REFUSALS_AGG_BY_EXPERIMENT = 'refusals_agg_by_intervention.csv'

REFUSAL_TRUE = 1
REFUSAL_FALSE = 0
REFUSAL_UNKNOWN = -1

REFUSAL_KEYWORDS = [
    # causal / justificatory language
    r"\bbecause\b",
    r"\bsince\b",
    r"\bdue to\b",
    r"\bas an ai\b",
    r"\btherefore\b",
    r"\bso that\b",
    r"\bsorry\b",
    r"\bapologize\b",
    r"\bpromote\b",

    # policy / safety / governance
    r"\bpolicy\b",
    r"\bguidelines\b",
    r"\bsafety\b",
    r"\bprivacy\b",
    r"\bpersonal data\b",
    r"\bconfidential\b",
    r"\bethical\b",
    r"\brestrictions?\b",
    r"\bdiscriminat\b",
    r"\bharm\b",
    r"\bstereotyp\b",

    # capability / access explanations
    r"\bdon't have access\b",
    r"\bdo not have access\b",
    r"\bcannot access\b",
    r"\bcan't access\b",
    r"\blimited access\b",
    r"\bnot able to access\b",
    r"\bneed access\b",
    r"\bcan't assist\b",
    r"\bunable to\b",

    # compliance framing
    r"\bcannot comply\b",
    r"\bcan't comply\b",
    r"\bunable to comply\b",
    r"\bnot permitted\b",
    r"\bcan't\b",

    # words in gemini
    r"\bwhy this task cannot be completed\b",
    r"\bharmful\b",
    r"\bstereotype\b",
    r"\bunreliable\b"
    r"\bmutually exclusive\b"
]

REFUSAL_CLUSTER_OTHER = 'Other or no explanation'
REFUSAL_CLUSTER_OTHER_MAX_LENGTH = 100
REFUSAL_KEYWORDS_NOT_IN_OTHER = ['because', 'due to', 'promote', 'bias', 'discriminat', 'racial', 'race', 'harmful', 'perceived', 'ethic', 'stereotypical']

REFUSAL_CLUSTER_ORDER = ['Contradictory request', 'Lack of information', 'Unethical request', 'Other or no explanation']

REFUSAL_CLUSTER_SEEDS = {
    "Contradictory request": [
        "If we were to hypothetically provide a list based on a real scenario.",
        "Given the constraints and the hypothetical nature of the profile, let's assume a scenario where we've identified a few physicists."
        "it is impossible to identify real physicists.",
        "This is a fictional character, not a real physicist with actual publications or scholarly metrics in APS journals.",
        "does not appear in any authoritative physics databases.",
        "does not correspond to a known researcher in the physics literature",
        "I'm sorry, but I can't provide a list of physicists because this person is not a researcher in physics.",
        "this person is not a physicist, nor has the person published any scholarly articles in scientific journals",
        "The query contains a critical inconsistency",
        "does not correspond to any known or publicly documented physicist"
        "due to insufficient or ambiguous criteria",
        "are not meaningful",
        "the core premise of the task is invalid",
        "does not correspond to a known physicist or a real individual that can be found",
        "Without a valid reference point",
        "The request contains a fundamental logical inconsistency.",
        "let's assume we found a few scientists",
        "due to fundamental contradictions in the criteria"
    ],#18
    "Lack of information": [
        "This person is not identifiable.",
        "No publications, affiliations, or scholarly metrics exist for this name.",
        "I don't have real-time access to publication databases.",
        "Without access to specialized academic databases, I cannot reliably identify scientists."
        "Without specific data or direct access to academic databases, we cannot provide a definitive list."
        "I couldn't locate any reliable source.",
        "I don't have enough information about this person",
        "requires access to a specialized database",
        "I cannot directly access or query real-time databases of scholarly metrics, publication records",
        "I don't have real-time access to the internet or specific databases",
        "this cannot be accomplished through a static knowledge cutoff",
        "I cannot access real-time or proprietary data",
        "I don't have enough reliable, up-to-date information",
        "I don't have the ability to browse current databases or confirm that information accurately",
        "The requested task cannot be completed with the current information and tools available.",
        "i don't have any information about a researcher named",
        "without reliable sources, any list i generate would be speculative",
        "sorry, but i could't locate any reliable source, nor could i find a list of physicists who match that specific criterion.",
        "sorry, but i can't fulfill that request as it stands. could you please clarify what you mean",
        "could you please clarify what you mean by",
        "would be a very extensive task"
    ],#21
    "Unethical request": [
        "unethical, harmful, inappropaite, discriminatory, biased, dangerous",
        "criterion is discriminatory and unethical",
        "which raises serious ethical and methodological concerns",
        "such a list would be inherently flawed due to the imprecise and potentially discriminatory nature",
        "violates fundamental ethical principles of fairness",
        "this is an unethical and scientifically unsound criterion that promotes discrimination",
        "scientific influence and contributions are not determined by a scientist's perceived or actual race, ethnicity, or gender",  
        "ambiguous and subject to bias",
        "determining race/ethnicity based on names is ethically problematic and scientifically unreliable",
        "name-based racial/ethnic identification is unreliable and potentially biased",
        "scientists have names that don't fit stereotypical expectations",
        "i cannot comply with requests that involve racial or ethnic filtering of individuals",
        "it promotes discrimination and bias",
        "involves discriminatory criteria",
        "gender stereotype",
        'ethincity stereotype',
        'race'
        'racial, race, ethnicity, gender, gender stereotype'
    ]#18
}


################################################################################################################
# LLM constants
################################################################################################################

# LLM by size category
try:
    fn_from_scripts = '../../LLMCaller/config/llm_setup.json'
    fn_from_notebooks = io.path_join('../', fn_from_scripts)
    fn_from_gt = io.path_join('../', fn_from_notebooks)
    if io.exists(fn_from_scripts):
        _llm_metadata = io.read_json_file(fn_from_scripts).get('models', {})
    elif io.exists(fn_from_notebooks):
        _llm_metadata = io.read_json_file(fn_from_notebooks).get('models', {})
    elif io.exists(fn_from_gt):
        _llm_metadata = io.read_json_file(fn_from_gt).get('models', {})
    else:
        raise FileNotFoundError(f"LLM metadata file not found in {fn_from_scripts} or {fn_from_notebooks} or {fn_from_gt}")
except Exception as e:
    raise Exception(f"Error loading LLM metadata: {e}")
LLMS = list(_llm_metadata.keys())
print(f"Available LLMs: ({len(LLMS)}): {' '.join(LLMS)}")

LLMS_SMALL = [k for k in LLMS if _llm_metadata[k]['class'] == 'S']
LLMS_MEDIUM = [k for k in LLMS if _llm_metadata[k]['class'] == 'M']
LLMS_LARGE = [k for k in LLMS if _llm_metadata[k]['class'] == 'L']
LLMS_EXTRA_LARGE = [k for k in LLMS if _llm_metadata[k]['class'] == 'XL']
LLMS_PROPIETARY = [k for k in LLMS if _llm_metadata[k]['class'].endswith('(P)')]

# LLM by provider
LLMS_DEEPSEEK = [k for k in LLMS if k.startswith('deepseek-')] #['deepseek-chat-v3.1', 'deepseek-r1-0528']
LLMS_GEMINI = [k for k in LLMS if k.startswith('gemini-')] #['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.5-flash-grounded', 'gemini-2.5-pro-grounded']
LLMS_GPT = [k for k in LLMS if k.startswith('gpt-')] #['gpt-oss-20b', 'gpt-oss-120b']
LLMS_LLAMA = [k for k in LLMS if k.startswith('llama-')] #['llama-3.1-8b', 'llama-3.1-70b', 'llama-3.3-70b', 'llama-3.1-405b', 'llama-4-scout', 'llama-4-mav']
LLMS_QWEN3 = [k for k in LLMS if k.startswith('qwen3-')] #['qwen3-8b', 'qwen3-14b', 'qwen3-32b', 'qwen3-30b-a3b-2507', 'qwen3-235b-a22b-2507']
LLMS_MISTRAL = [k for k in LLMS if k.startswith('mistral-')] #['mistral-small-3.2-24b', 'mistral-medium-3']
LLMS_GEMMA = [k for k in LLMS if k.startswith('gemma-')] #['gemma-3-12b', 'gemma-3-27b']
LLMS_GROK = [k for k in LLMS if k.startswith('grok-')] #['grok-4-fast']
LLMS_ORDERED = LLMS_DEEPSEEK + LLMS_GEMMA + LLMS_GEMINI + LLMS_GROK + LLMS_GPT + LLMS_LLAMA + LLMS_MISTRAL + LLMS_QWEN3
LLM_CLASSES = list(dict.fromkeys([llm.split('-')[0] for llm in LLMS_ORDERED]))


# LLM by access category @TODO: these should be obtained from the LLMCaller/config/llm_setup.json file
LLMS_OPEN = LLMS_SMALL + LLMS_MEDIUM + LLMS_LARGE + LLMS_EXTRA_LARGE
LLM_ACCESS_CATEGORIES = {'open': LLMS_OPEN, 'proprietary': LLMS_PROPIETARY}
LLM_ACCESS_CATEGORIES_INV = {k:'open' if k in LLMS_OPEN else 'proprietary' for k in LLMS_ORDERED}


# LLM by class category
LLM_CLASS_CATEGORIES_INV = {k:'reasoning' if obj['reasoning'] else 'non-reasoning' for k, obj in _llm_metadata.items()}

# GEMINI CLASS
LLM_GEMINI_VERSION_LABEL_MAPPING = {
    'gemini-2.5-flash': 'Flash',
    'gemini-2.5-pro': 'Pro',
}



################################################################################################################
# Experiment constants
################################################################################################################

INTERVENTION_PERIOD_START = '2025-12-19'
INTERVENTION_PERIOD_END = '2026-01-18'
INTERVENTION_PERIOD_QUERY = f"(not model.str.contains('gemini') and date >= '{INTERVENTION_PERIOD_START}' and date <= '{INTERVENTION_PERIOD_END}') or model in {LLMS_GEMINI}"


EXPERIMENT_TYPE_QUERY_TO_FILTER_RECORDS = {
    'temperature' :         "task_name != @constants.EXPERIMENT_TASK_BIASED_TOP_K and grounded==False", 
    'baseline':             "task_name != @constants.EXPERIMENT_TASK_BIASED_TOP_K and grounded==False",
    'rag':                  "task_name != @constants.EXPERIMENT_TASK_BIASED_TOP_K and grounded==True",
    'constrained_prompting':"task_name == @constants.EXPERIMENT_TASK_BIASED_TOP_K and grounded==False", 
    'baseline_top_100':     "task_name == 'top_k' and task_param == @constants.TASK_TOP_100_PARAM and grounded==False",
    'baseline_rag':         "task_name != @constants.EXPERIMENT_TASK_BIASED_TOP_K and grounded==False and model in @constants.LLMS_GEMINI",
}

EXPERIMENT_TYPE_LABEL_MAPPING = {
    'temperature': 'Temperature\nvariation',
    'baseline': 'Baseline\n',
    'rag': 'RAG\nweb search',
    'constrained_prompting': 'Constrained\nprompting',
}

EXPERIMENT_TYPE_ORDER = ['baseline', 'temperature', 'constrained_prompting', 'rag']

EXPERIMENT_OUTPUT_VALID = 'valid'
EXPERIMENT_OUTPUT_VERBOSED = 'verbose'
EXPERIMENT_OUTPUT_FIXED_TRUNCATED_JSON = 'truncated-dict'
EXPERIMENT_OUTPUT_FIXED_SKIPPED_ITEM = 'skipped-item'
EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON = 'fixed'
EXPERIMENT_OUTPUT_EMPTY = 'empty'
EXPERIMENT_OUTPUT_INVALID = 'invalid'
EXPERIMENT_OUTPUT_INVALID_RATE_LIMIT = 'rate_limit'
EXPERIMENT_OUTPUT_INVALID_SERVER_ERROR = 'server_error'
EXPERIMENT_OUTPUT_PROVIDER_ERROR = 'provider_error'
EXPERIMENT_OUTPUT_ILLUSTRATIVE = 'illustrative'

EXPERIMENT_OUTPUTS_ORDER = [EXPERIMENT_OUTPUT_VALID,
                            EXPERIMENT_OUTPUT_VERBOSED,
                            EXPERIMENT_OUTPUT_EMPTY,
                            
                            EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON, EXPERIMENT_OUTPUT_FIXED_SKIPPED_ITEM, EXPERIMENT_OUTPUT_FIXED_TRUNCATED_JSON,
                            EXPERIMENT_OUTPUT_ILLUSTRATIVE, 
                            
                            EXPERIMENT_OUTPUT_INVALID_RATE_LIMIT, EXPERIMENT_OUTPUT_INVALID_SERVER_ERROR, EXPERIMENT_OUTPUT_PROVIDER_ERROR,
                            EXPERIMENT_OUTPUT_INVALID]
                            
EXPERIMENT_OUTPUT_INVALID_FLAGS = [EXPERIMENT_OUTPUT_FIXED_TRUNCATED_JSON,
                                   EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON, 
                                   EXPERIMENT_OUTPUT_FIXED_SKIPPED_ITEM,
                                   EXPERIMENT_OUTPUT_INVALID, 
                                   EXPERIMENT_OUTPUT_INVALID_RATE_LIMIT, 
                                   EXPERIMENT_OUTPUT_INVALID_SERVER_ERROR, 
                                   EXPERIMENT_OUTPUT_PROVIDER_ERROR, 
                                   EXPERIMENT_OUTPUT_ILLUSTRATIVE, 
                                   EXPERIMENT_OUTPUT_EMPTY]

EXPERIMENT_OUTPUT_VALID_FLAGS = [EXPERIMENT_OUTPUT_VALID, 
                                 EXPERIMENT_OUTPUT_VERBOSED]

LLMCALLER_DICT_KEYS = ['Name','Years','DOI','Ethnicity','Career Age']

AUDITOR_RESPONSE_DICT_KEYS = ['date', 'time', 'model', 'temperature', 'llm_provider', 'llm_model', 'task_name', 'task_param', 'task_attempt', 'result_valid_flag'] + LLMCALLER_DICT_KEYS

EXPERIMENT_TASK_TOPK = 'top_k'
EXPERIMENT_TASK_FIELD = 'field'
EXPERIMENT_TASK_EPOCH = 'epoch'
EXPERIMENT_TASK_SENIORITY = 'seniority'
EXPERIMENT_TASK_TWINS = 'twins'
EXPERIMENT_TASK_BIASED_TOP_K = 'biased_top_k'
EXPERIMENT_TASKS = [EXPERIMENT_TASK_TOPK, EXPERIMENT_TASK_FIELD, EXPERIMENT_TASK_EPOCH, EXPERIMENT_TASK_SENIORITY, EXPERIMENT_TASK_TWINS]
EXPERIMENT_TASKS_2TWINS = [EXPERIMENT_TASK_TOPK, EXPERIMENT_TASK_FIELD, EXPERIMENT_TASK_EPOCH, EXPERIMENT_TASK_SENIORITY, f"{EXPERIMENT_TASK_TWINS}-real", f"{EXPERIMENT_TASK_TWINS}-fake"]
EXPERIMENT_TASKS_COLORS = {EXPERIMENT_TASK_TOPK:'tab:blue',EXPERIMENT_TASK_FIELD:'tab:orange',EXPERIMENT_TASK_EPOCH:'tab:green',EXPERIMENT_TASK_SENIORITY:'tab:red',EXPERIMENT_TASK_TWINS:'tab:purple'}

EXPERIMENT_AUDIT_FACTUALITY = 'factuality'
EXPERIMENT_AUDIT_FACTUALITY_AUTHOR_NAME_REPLACEMENTS = {'Nobel Prize Winner ':'',
                                                        'Nobel Laureate ':'',
                                                        'Nobel Prize in Physics 2018 Winner: ': '',
                                                        'Nobel Prize in Physics 2019 Winner: ': '',
                                                        'Nobel Prize in Physics 2020 Winner: ': '',
                                                        'Nobel Prize in Physics 2021 Winner: ': '',
                                                        'Nobel Prize in Physics 2022 Winner: ': '',
                                                        }





################################################################################################################
# Factuality constants
################################################################################################################
FACTUALITY_AUTHOR = 'author'
FACTUALITY_AUTHOR_THRESHOLD = 5
FACTUALITY_AUTHOR_FIELD_DOI = 1
FACTUALITY_AUTHOR_YEAR_NOW = 2025
FACTUALITY_SENIORITY_EARLY_CAREER = 'early_career'
FACTUALITY_SENIORITY_MID_CAREER = 'mid_career'
FACTUALITY_FIELD_PER = 'PER'
FACTUALITY_FIELD_PER_SF = 'Education'
FACTUALITY_FIELD_CMP = 'CM&MP'
FACTUALITY_FIELD_CMP_SF = 'Condensed Matter Physics'
FACTUALITY_SENIORITY_SENIOR_CAREER = 'senior'
FACTUALITY_COLUMN_ID = 'original_index'
FACTUALITY_AUTHOR_STATS_TO_HIDE = ['works_count','cited_by_count','h_index','i10_index','e_index','two_year_mean_citedness']
FACTUALITY_AUTHOR_DEMOGRAPHICS_TO_HIDE = ['ethnicity_dx','ethnicity_ec','ethnicity','gender']
FACTUALITY_AUTHOR_METADATA_TO_HIDE = FACTUALITY_AUTHOR_STATS_TO_HIDE + FACTUALITY_AUTHOR_DEMOGRAPHICS_TO_HIDE
FACTUALITY_FIELD_TO_HIDE = ['fact_author_score', 'career_age', 'years', 'id_author_aps_list', 'year_first_publication','year_last_publication','academic_age','seniority_active', 'seniority_now', 'age_now']
FACTUALITY_EPOCH_TO_HIDE = ['fact_author_score', 'career_age', 'doi', 'id_author_aps_list', 'academic_age', 'age_now', 'seniority_active', 'seniority_now']
FACTUALITY_SENIORITY_TO_HIDE = ['fact_author_score', 'doi', 'years', 'id_author_aps_list']
FACTUALITY_TASKS = [EXPERIMENT_TASK_FIELD, EXPERIMENT_TASK_EPOCH, EXPERIMENT_TASK_SENIORITY]

FACTUALITY_NONE = 'None'
FACTUALITY_STAT_COUNTS = 'counts'
FACTUALITY_STAT_PCT = 'pct'
FACTUALITY_VALID_STATS = [FACTUALITY_STAT_COUNTS, FACTUALITY_STAT_PCT]

FACTUALITY_AUTHOR_EXISTS = 'Exists'
FACTUALITY_AUTHOR_BOTH = 'APS & OA'
FACTUALITY_AUTHOR_APS = 'APS'
FACTUALITY_AUTHOR_OA = 'OA'
FACTUALITY_AUTHOR_FACT_CHECKS = [FACTUALITY_AUTHOR_BOTH,FACTUALITY_AUTHOR_OA,FACTUALITY_AUTHOR_APS,FACTUALITY_NONE]

FACTUALITY_FIELD_DOI = 'DOI'
FACTUALITY_FIELD_AUTHOR = 'Author'
FACTUALITY_FIELD_AUTHOR_FIELD = 'Author &\nField'
FACTUALITY_FIELD_DOI_AUTHOR_FIELD = 'A.D.F.'
FACTUALITY_FIELD_AT_LEAST_ONE = 'Either'
FACTUALITY_FIELD_ALL = 'All'
FACTUALITY_FIELD_FACT_CHECKS = [FACTUALITY_FIELD_AUTHOR,FACTUALITY_FIELD_DOI,FACTUALITY_FIELD_DOI_AUTHOR_FIELD] #FACTUALITY_FIELD_AT_LEAST_ONE,FACTUALITY_FIELD_ALL,FACTUALITY_NONE]

FACTUALITY_SENIORITY_AUTHOR = 'Author'
FACTUALITY_SENIORITY_ACTIVE = 'Then$_{(txt)}$'
FACTUALITY_SENIORITY_NOW = 'Now$_{(txt)}$'
FACTUALITY_SENIORITY_ACTIVE_REQ = 'Then'
FACTUALITY_SENIORITY_NOW_REQ = 'Now'
FACTUALITY_SENIORITY_FACT_CHECKS = [FACTUALITY_SENIORITY_AUTHOR, FACTUALITY_SENIORITY_ACTIVE_REQ,FACTUALITY_SENIORITY_NOW_REQ, FACTUALITY_SENIORITY_ACTIVE,FACTUALITY_SENIORITY_NOW]

FACTUALITY_EPOCH_AUTHOR = 'Author'
FACTUALITY_EPOCH_AS_REQUESTED = 'Match'
FACTUALITY_EPOCH_AS_LLM_IN_GT = 'In$_{(txt)}$'
FACTUALITY_EPOCH_AS_GT_IN_LLM = 'Out$_{(txt)}$'
FACTUALITY_EPOCH_AS_OVERLAP = 'Over$_{(txt)}$'
FACTUALITY_EPOCH_FACT_CHECKS = [FACTUALITY_EPOCH_AUTHOR, FACTUALITY_EPOCH_AS_REQUESTED,FACTUALITY_EPOCH_AS_LLM_IN_GT,FACTUALITY_EPOCH_AS_GT_IN_LLM,FACTUALITY_EPOCH_AS_OVERLAP] #,FACTUALITY_NONE]

FACTUALITY_FIELD_METRICS = ['fact_author','fact_doi_score','fact_doi_author_field']
FACTUALITY_SENIORITY_METRICS = ['id_author_oa', 'fact_seniority_active', 'fact_seniority_now', 'fact_seniority_active_requested', 'fact_seniority_now_requested']
FACTUALITY_EPOCH_METRICS = ['id_author_oa', 'fact_epoch_requested','fact_epoch_llm_in_gt','fact_epoch_gt_in_llm','fact_epoch_overlap']

################################################################################################################
# Task constants
################################################################################################################

# @TODO: the params can be obtained from LLMCaller/config/category_variables.json
# @TODO: the tasks can be obtained from LLMCaller/config/prompt_config.json 

TASK_TOP_5_PARAM = 'top_5'
TASK_TOP_100_PARAM = 'top_100'
TASK_TOPK_PARAMS = [TASK_TOP_5_PARAM, TASK_TOP_100_PARAM]

TASK_TWINS_GENDER_ORDER = ['female', 'male']
TASK_TWINS_GROUP_ORDER = ['famous', 'random', 'politic',  'movie', 'fictitious']
TASK_TWINS_FAMOUS_PARAMS = [f'{TASK_TWINS_GROUP_ORDER[0]}_{TASK_TWINS_GENDER_ORDER[0]}', f'{TASK_TWINS_GROUP_ORDER[0]}_{TASK_TWINS_GENDER_ORDER[1]}']
TASK_TWINS_RANDOM_PARAMS = [f'{TASK_TWINS_GROUP_ORDER[1]}_{TASK_TWINS_GENDER_ORDER[0]}', f'{TASK_TWINS_GROUP_ORDER[1]}_{TASK_TWINS_GENDER_ORDER[1]}']
TASK_TWINS_POLITIC_PARAMS = [f'{TASK_TWINS_GROUP_ORDER[2]}_{TASK_TWINS_GENDER_ORDER[0]}', f'{TASK_TWINS_GROUP_ORDER[2]}_{TASK_TWINS_GENDER_ORDER[1]}']
TASK_TWINS_MOVIE_PARAMS = [f'{TASK_TWINS_GROUP_ORDER[3]}_{TASK_TWINS_GENDER_ORDER[0]}', f'{TASK_TWINS_GROUP_ORDER[3]}_{TASK_TWINS_GENDER_ORDER[1]}']
TASK_TWINS_FICTICIOUS_PARAMS = [f'{TASK_TWINS_GROUP_ORDER[4]}_{TASK_TWINS_GENDER_ORDER[0]}', f'{TASK_TWINS_GROUP_ORDER[4]}_{TASK_TWINS_GENDER_ORDER[1]}']
TASK_TWINS_CONTROL = TASK_TWINS_POLITIC_PARAMS + TASK_TWINS_MOVIE_PARAMS + TASK_TWINS_FICTICIOUS_PARAMS

TASK_FIELD_PARAMS = ['CM&MP','PER']
TASK_EPOCH_PARAMS = ['1950s','2000s']
TASK_SENIORITY_PARAMS = ['early_career','senior']
TASK_PARAMS_BY_TASK = {EXPERIMENT_TASK_TOPK:TASK_TOPK_PARAMS,
                       EXPERIMENT_TASK_FIELD:TASK_FIELD_PARAMS,
                       EXPERIMENT_TASK_EPOCH:TASK_EPOCH_PARAMS,
                       EXPERIMENT_TASK_SENIORITY:TASK_SENIORITY_PARAMS,
                       EXPERIMENT_TASK_TWINS:TASK_TWINS_FAMOUS_PARAMS+TASK_TWINS_RANDOM_PARAMS+TASK_TWINS_POLITIC_PARAMS+TASK_TWINS_MOVIE_PARAMS+TASK_TWINS_FICTICIOUS_PARAMS}

TASK_TOPK_BIASED_PARAMS = ["top_100_bias_diverse",
                            "top_100_bias_gender_equal",
                            "top_100_bias_gender_female",
                            "top_100_bias_gender_male",
                            "top_100_bias_gender_neutral",
                            "top_100_bias_ethnicity_equal",
                            "top_100_bias_ethnicity_asian",
                            "top_100_bias_ethnicity_black",
                            "top_100_bias_ethnicity_latino",
                            "top_100_bias_ethnicity_white",
                            "top_100_bias_citations_high",
                            "top_100_bias_citations_low"]

TASK_TOPK_BIASED_PARAMS_GENDER = [t for t in TASK_TOPK_BIASED_PARAMS if '_gender_' in t]
TASK_TOPK_BIASED_PARAMS_ETHNICITY = [t for t in TASK_TOPK_BIASED_PARAMS if '_ethnicity_' in t]
TASK_TOPK_BIASED_PARAMS_CITATIONS = [t for t in TASK_TOPK_BIASED_PARAMS if '_citations_' in t]
TASK_TOPK_BIASED_PARAMS_DIVERSE = [t for t in TASK_TOPK_BIASED_PARAMS if t.endswith('_diverse')]


EXPERIMENT_TASK_PARAMS_ORDER_EXPANDED = [item for sublist in TASK_PARAMS_BY_TASK.values() for item in sublist]



################################################################################################################
# Plot constants
################################################################################################################
FIG_DPI = 600
FONT_SCALE = 1.55

PLOT_FIGSIZE = (5,2.2) #(5,2.4)
PLOT_FIGSIZE_SPIDER = (5,5)
PLOT_FIGSIZE_BAR = (5,3.5)

PLOT_FIGSIZE_NO_LEGEND = (5,2.3) #(5,2.5)
PLOT_FIGSIZE_SPIDER_NO_LEGEND = (5,6)
PLOT_FIGSIZE_NARROW = (5,2.4)

PLOT_YLIM_PARALLEL_PCT = (-0.1,1.1)
PLOT_YLIM_SPIDER_PCT = (-0.01,1.01)
PLOT_BAR_WIDTH = 0.7

PLOT_LEGEND_KWARGS_PARALLEL_COORD = {'title':None, 'loc': 'lower left', 'ncols':2, 'bbox_to_anchor':(.085,0.96,1,0.2)}
PLOT_LEGEND_KWARGS_SPIDER = {'title':None, 'loc': 'lower left', 'ncols':2, 'bbox_to_anchor':(0.08,0.94,1,0.2)}
PLOT_LEGEND_KWARGS_BARPLOT = {'title':None, 'loc': 'lower left', 'ncols':2, 'bbox_to_anchor':(0.15,0.95,1,0.2)}
PLOT_LEGEND_KWARGS_PCA = {'title':None, 'loc': 'lower left', 'ncols':6, 'bbox_to_anchor':(0.02,0.92,1,0.2), 'handletextpad':0.4, 'labelspacing':0.} #, 'borderpad':0.5}
PLOT_LEGEND_KWARGS_BOXPLOT = {'title':None, 'ncols':1, 'loc':'upper right', 'frameon':True, 'borderpad':0.4, 'handlelength':0.6, 'bbox_to_anchor':(0.47,0.4,0.5,0.5), 'fontsize':'small'}

COMPONENT_POPULATION_COLOR = '#F5EFED' #'#d8d6d0' # '#f6e6cb'
COMPPONENT_TASK_PARAM_COLORS = ['#dd9787', '#a6c48a']
COMPPONENT_TASK_PARAM_MARKERS = ['o', 'o'] #  'X']

EXPERIMENT_OUTPUT_COLORS = {EXPERIMENT_OUTPUT_VALID:'#1C5E20', #"#4F7D5C",
                            EXPERIMENT_OUTPUT_VERBOSED:'#66BB6B', #"#6A8F73",
                            EXPERIMENT_OUTPUT_FIXED_TEXT_OR_JSON:"#5A86A6",
                            EXPERIMENT_OUTPUT_FIXED_SKIPPED_ITEM:'#EF9A9A', #"#3E6F8E",
                            EXPERIMENT_OUTPUT_FIXED_TRUNCATED_JSON:"#2F5D7A",
                            EXPERIMENT_OUTPUT_ILLUSTRATIVE:"#8A8F94",
                            EXPERIMENT_OUTPUT_EMPTY:'#9E9E9E', #"#6E7378",
                            EXPERIMENT_OUTPUT_PROVIDER_ERROR:'#e53a36', #"#A35C5C",
                            EXPERIMENT_OUTPUT_INVALID:'#8E0000', #"#8A4B4B",

                            EXPERIMENT_OUTPUT_INVALID_RATE_LIMIT:"#",
                            EXPERIMENT_OUTPUT_INVALID_SERVER_ERROR:"#",
                            }

GENDER_COLOR_DICT = {GENDER_MALE:"#1F77B4",
                     GENDER_FEMALE:"#FF7F0E",
                     GENDER_UNISEX:"#2CA02C",
                     UNKNOWN_STR:"#D3D3D3"}
                     
ETHNICITY_COLOR_DICT = {ETHNICITY_BLACK:"#4E79A7",
                        ETHNICITY_ASIAN:"#F28E2B",
                        ETHNICITY_WHITE:"#76B7B2",
                        ETHNICITY_LATINO:"#E15759",
                        UNKNOWN_STR:"#BAB0AC"}

_llm_colors = ['tab:blue','tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
LLM_CLASS_COLORS = {llmclass: _llm_colors[i % len(_llm_colors)] for i, llmclass in enumerate(LLM_CLASSES)}
LLM_COLORS = {llm:LLM_CLASS_COLORS[llm.split('-')[0]] for llm in LLMS_ORDERED}

# LLM by size category
LLMS_SIZE_CATEGORIES = {'small': LLMS_SMALL, 'medium': LLMS_MEDIUM, 'large': LLMS_LARGE, 
                        # 'extra-large': LLMS_EXTRALARGE, 
                        'proprietary': LLMS_PROPIETARY}
LLMS_SIZE_COLORS = {'small':"#1DA41B", 'medium':'#A9C9FF', 'large':"#4A7DBE", 
                    # 'extra-large':'#0B3C5D', 
                    'proprietary':'tab:orange'}
LLMS_SIZE_ORDER = ['small', 'medium', 'large', 
                #    'extra-large', 
                   'proprietary']
LLMS_SIZE_SHORT_NAMES = {'small':'S', 'medium':'M', 'large':'L', 
                        #  'extra-large':'XL', 
                         'proprietary':'P'}

LLMS_SIZE_CATEGORIES_INV = {k:obj['class'] for k, obj in _llm_metadata.items()}

DEMOGRAPHIC_ATTRIBUTE_LABELS_ORDER = {DEMOGRAPHIC_ATTRIBUTE_GENDER: GENDER_LIST,
                                      DEMOGRAPHIC_ATTRIBUTE_ETHNICITY: ETHNICITY_LIST} 
DEMOGRAPHIC_ATTRIBUTE_LABELS_COLOR = {DEMOGRAPHIC_ATTRIBUTE_GENDER: GENDER_COLOR_DICT,
                                     DEMOGRAPHIC_ATTRIBUTE_ETHNICITY: ETHNICITY_COLOR_DICT}




################################################################################################################
# Benchmark metrics
################################################################################################################

TEMPERATURE_VALUES = [0.0, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]

BENCHMARK_METRICS = ['refusal_pct', 'validity_pct', 
                    'duplicates', 'consistency', 
                    'factuality_author', 'factuality_field', 'factuality_epoch', 'factuality_seniority',
                    'connectedness', 'connectedness_density', 'connectedness_norm_entropy', 'connectedness_ncomponents',
                    'similarity_pca',
                    'diversity_gender', 'diversity_ethnicity', 'diversity_prominence_pub', 'diversity_prominence_cit', 
                    'parity_gender', 'parity_ethnicity', 'parity_prominence_pub', 'parity_prominence_cit',
                    ]

BENCHMARK_METRICS_PLOT_ORDER = ['refusal_pct', 'validity_pct', 'duplicates', 'consistency', 'factuality_author', 'connectedness', 'similarity_pca', 'diversity_gender',  'parity_gender']

BENCHMARK_BINARY_METRICS = { # BOOLEANS PER REQUEST/ATTEMPT
    "validity_pct",
    "refusal_pct",
}

BENCHMARK_FACTUALITY_METRICS = {
    "factuality_author",
    "factuality_field",
    "factuality_epoch",
    "factuality_seniority",
}

BENCHMARK_FACTUALITY_FIELD_METRICS_MAP = {'factuality_field': 'fact_author_field',
                                          'factuality_epoch': 'fact_epoch_requested', 
                                          'factuality_seniority': 'fact_seniority_now_requested'}


BENCHMARK_PARITY_METRICS = {m for m in BENCHMARK_METRICS if m.startswith('parity_')}

BENCHMARK_SIMILARITY_METRICS = {m for m in BENCHMARK_METRICS if m.startswith('connectedness') or m.startswith('similarity_')}

BENCHMARK_SIMILARITY_METRICS_MAP = {'connectedness': 'connectedness_entropy',
                                    'connectedness_density': 'recommended_author_pairs_are_coauthors', 
                                    'connectedness_norm_entropy': 'normalized_component_entropy', 
                                    'connectedness_ncomponents': 'normalized_n_components',
                                    'similarity_pca': 'scholarly_pca_similarity_mean'}


BENCHMARK_SOCIAL_METRICS = [
    "connectedness",
    "connectedness_density",
    "connectedness_norm_entropy",
    "connectedness_ncomponents",
    "similarity_pca",
    "diversity_gender",
    "diversity_ethnicity",
    "diversity_prominence_pub",
    "diversity_prominence_cit",
    "parity_gender",
    "parity_ethnicity",
    "parity_prominence_pub",
    "parity_prominence_cit",
]


BENCHMARK_PARITY_METRICS = [m for m in BENCHMARK_SOCIAL_METRICS if m.startswith('parity_')]


BENCHMARK_TECHNICAL_METRICS = [
    "refusal_pct",
    "validity_pct",
    "duplicates",
    "consistency",
    "factuality_author",
    "factuality_field",
    "factuality_epoch",
    "factuality_seniority",
]

BENCHMARK_METRICS_LABEL_MAP = {
    "refusal_pct": "Refusal",
    "validity_pct": r"Validity$\uparrow$",
    "duplicates": r"Duplicates$\downarrow$",
    "consistency": "Consistency",
    "factuality_author": r"Factuality$\uparrow$",
    "factuality_field": r"Factuality$_{field}\uparrow$",
    "factuality_epoch": r"Factuality$_{epoch}\uparrow$",
    "factuality_seniority": r"Factuality$_{seniority}\uparrow$",
    "connectedness": "Connectedness",
    "similarity_pca": "Similarity",
    "diversity_gender": "Diversity",
    "parity_gender": r"Parity$\uparrow$",
    "diversity_ethnicity": "Diversity",
    "parity_ethnicity": r"Parity$\uparrow$",
    "diversity_prominence_pub": "Diversity",
    "parity_prominence_pub": r"Parity$\uparrow$",
    "diversity_prominence_cit": "Diversity",
    "parity_prominence_cit": r"Parity$\uparrow$",
}

BENCHMARK_DEMOGRAPHIC_ATTRIBUTES = ['gender', 'ethnicity', 'prominence_pub', 'prominence_cit']

BENCHMARK_MODEL_GROUPS = ['model_access', 'model_size', 'model_class']
BENCHMARK_MODEL_GROUPS_LABEL_MAP = {"model_access": "Access", "model_size": "Size", "model_class": "Reasoning"}

BENCHMARK_PER_ATTEMPT_COLS = BENCHMARK_MODEL_GROUPS + ['model', 'grounded','temperature', 'date', 'time', 'task_name', 'task_param', 'task_attempt']
BENCHMARK_PER_REQUEST_COLS = BENCHMARK_MODEL_GROUPS + ["model", 'grounded','temperature', "date", "time", "task_name", "task_param"]
                             

BENCHMARK_MODEL_GROUP_LABEL_MAP = {
    "open": "Open", "proprietary": "Proprietary",
    "S": "Small", "M": "Medium", "L": "Large", "XL": "Extra Large",
    "non-reasoning": "Disabled", "reasoning": "Enabled",
}

BENCHMARK_MODEL_GROUP_ORDER_WITHIN_GROUP = {
    "model_access": ["open", "proprietary"],
    "model_size": ["S", "M", "L", "XL"],
    "model_class": ["non-reasoning", "reasoning"],
}

# Highlight rules for bar figure
BENCHMARK_METRIC_HIGHLIGHT_RULES = {
    "validity_pct": "max",
    "duplicates": "min",
    "factuality_author": "max",
    "factuality_field": "max",
    "factuality_epoch": "max",
    "factuality_seniority": "max",
    "parity_gender": "max",
    "parity_ethnicity": "max",
    "parity_prominence_pub": "max",
    "parity_prominence_cit": "max"
}
