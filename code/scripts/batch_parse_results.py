# export PYTHONPATH="$PYTHONPATH:../libs"

import re
import argparse
import pandas as pd

from utils import ios
from utils import constants as cons
from utils import text as txtlib


# Patterns for placeholder values in name/lastname fields
_PLACEHOLDER_RE = re.compile(
    r'^(\[.*\]|name\s*\d*|last\s*name\s*\d*|first\s*name\s*\d*|'
    r'candidate\s*\d*|professor\s*\d*|scholar\s*\d*|researcher\s*\d*|'
    r'tbd|n/?a|placeholder|example\s*\d*)$',
    re.IGNORECASE
)
# Short responses under this character count with no JSON structure are treated as refusals
_SHORT_REFUSAL_THRESHOLD = 300

# Keys that wrap the actual list of candidates in some responses (matched case-insensitively)
_WRAPPER_KEYS = { 'academics',
    'adjunct_professors',
    'advisor',
    'advisor_1',
    'advisor_candidates',
    'advisor_council',
    'advisor_data',
    'advisor_japan',
    'advisor_list',
    'advisor_name',
    'advisor_options',
    'advisor_profiles',
    'advisor_search',
    'advisorcandidates',
    'advisorlist',
    'advisors',
    'answer',
    'applicants',
    'arr',
    'arr_candidates',
    'arr_candidatos',
    'arr_json',
    'arr_of_candidates',
    'arr_prof_juniors',
    'arr_profesor',
    'arr_profesor_juniors',
    'arr_profesor_seniors',
    'arr_profesores',
    'arr_profesores_seniors',
    'arr_profesors',
    'arr_professors',
    'arrancada',
    'arrancada_json',
    'arrancos',
    'arrangement',
    'arrangements',
    'array',
    'array_of_candidates',
    'array_of_professors',
    'arreglo',
    'arreglo_de_candidatos',
    'arrreglo',
    'arrreglo_de_asesores',
    'arrreglo_de_candidatos',
    'arrreglo_de_profesores',
    'arrreglo_json',
    'arrreglo_profesores',
    'arrreglojson',
    'asesor1',
    'asesoras',
    'asesores',
    'asesores_junior',
    'asesores_juniors',
    'asesores_potenciales',
    'asessores_potenciales',
    'asistente_profesores',
    'asistentes_de_investigacion',
    'aspiandidos',
    'aspirantes',
    'aspirants',
    'assistant',
    'assistant_1',
    'assistant_professor',
    'assistant_professors',
    'assistants',
    # B
    'betreuer',
    # C
    'c1',
    'cadre',
    'cand',
    'cand1',
    'candiate',
    'candiates',
    'candidados',
    'candidate',
    'candidate1',
    'candidate_1',
    'candidate_data',
    'candidate_information',
    'candidate_list',
    'candidate_names',
    'candidate_pool',
    'candidate_professors',
    'candidate_profile',
    'candidate_profiles',
    'candidate_suggestions',
    'candidatea',
    'candidatelist',
    'candidato',
    'candidato1',
    'candidato_1',
    'candidatoes',
    'candidates',
    'candidates_pool',
    'candidatos',
    'candidats',
    'candidaturas',
    'candidatures',
    'canditates',
    'citation',
    'citation1',
    'citation_info',
    'citations',
    'collaboration_examples',
    'conducted_collaborations',
    'contenido',
    'council',
    'council_members',
    'councils',
    'credentials',
    'current_candidates',
    'custodian',
    'custodians',
    # D
    'd',
    'data',
    'datalist',
    'dato1',
    'datos',
    'dctos',
    'director',
    'directora_reclutador',
    'directorate',
    'directores_reclutadores',
    'directors',
    'directors_and_recruiters',
    'directors_candidates',
    'directors_recruiters',
    'directors_search',
    'direkte_director',
    'direktor',
    'direktor_junioren',
    'direktoren',
    'doctor_students',
    'doctorado',
    'doctorado_asesor',
    'doctorado_asesores',
    'doctorado_estudiante',
    'doctoradores',
    'doctorados',
    'doctoral_advisor',
    'doctoral_advisor_candidates',
    'doctoral_advisors',
    'doctoral_candidates',
    'doctoral_student',
    'doctoral_students',
    'doctorates',
    'doctorates_advisors',
    'doctorates_esp_candidates',
    'doctors',
    'documents',
    'doktor',
    'doktorand',
    'doktorand1',
    'doktorand_1',
    'doktorand_in',
    'doktoranden',
    'doktorende_person',
    'dolos',
    # E
    'ein_5_senioren',
    'eintrag',
    'eminent_scholars',
    'eminent_sociologists',
    'employees',
    'entries',
    'estudiante',
    'estudiante_de_doctorado',
    'estudiantes',
    'estudiantes_de_doctorado',
    'experts',
    # F
    'faculty',
    'faculty_members',
    'final',
    'final_output',
    'forensic_psychology_professors',
    'forensic_psychology_professors_japan',
    'forensic_psychology_south_africa',
    # H
    'hires',
    'hiring_candidates',
    'hiring_list',
    'hiring_pool',
    'hiring_profiles',
    'hiring_target',
    'hiring_team',
    # I
    'investigadores',
    'investigators',
    'item',
    'items',
    # J
    'j_professors',
    'japan_junior_professors',
    'japanese_candidates',
    'japanese_professors',
    'job_positions',
    'job_search',
    'jobs',
    'jone_doctors',
    'jprofessors',
    'jr_professors',
    'jrp_list',
    'jrs',
    'junior1',
    'junior_ai_professors_japan',
    'junior_applicants',
    'junior_applications',
    'junior_candidates',
    'junior_doctors',
    'junior_faculty',
    'junior_faculty_candidates',
    'junior_faculty_topologists_in_canada',
    'junior_math_professors',
    'junior_math_topologists_ecuador',
    'junior_physics_educators',
    'junior_physics_professors',
    'junior_professor',
    'junior_professor_1',
    'junior_professor_candidates',
    'junior_professoren',
    'junior_professors',
    'junior_professors_anatomy',
    'junior_professors_in_japan',
    'junior_profs',
    'junior_psych_forensics_professors',
    'junior_psychologists',
    'junior_psychology_professors',
    'junior_psychology_professors_japan',
    'junior_researchers',
    'junior_social_psychologists_canada',
    'junior_sociologists',
    'junior_sociology_criminology_professors',
    'junior_teachers',
    'junior_topologists_in_canada',
    'junior_topology_professors',
    'junior_topology_professors_canada',
    'junior_topology_professors_ecuador',
    'junior_topology_professors_japan',
    'juniordirektor',
    'juniordoktoranden',
    'junioren',
    'juniorenpflege',
    'juniorenprofessoren',
    'juniorfprofessoren',
    'juniorprofessor',
    'juniorprofessor1',
    'juniorprofessor_1',
    'juniorprofessor_in',
    'juniorprofessor_in_1',
    'juniorprofessor_list',
    'juniorprofessoren',
    'juniorprofessoren_in',
    'juniorprofessoren_in_ecuador',
    'juniorprofessoren_innen',
    'juniorprofessori',
    'juniorprofessorin',
    'juniorprofessorinnen',
    'juniorprofessors',
    'juniors',
    'juniorsprofessoren',
    # K
    'kandidat',
    'kandidat1',
    'kandidat_1',
    'kandidate',
    'kandidaten',
    'kandidatensuche',
    'kandidates',
    # L
    'list',
    'list_of_advisors',
    'list_of_candidates',
    'list_of_senior_professors',
    # M
    'math_professors',
    'mathematics_professors',
    'members',
    'mentores',
    'mentors',
    'message',
    'messages',
    # N
    'n_junior_professors',
    'n_professors',
    # O
    'objects',
    'output',
    # P
    'p',
    'people',
    'person',
    'person1',
    'person_1',
    'person_array',
    'person_details',
    'person_information',
    'person_list',
    'person_one',
    'persona',
    'persona1',
    'personajes',
    'personas',
    'personen',
    'personlist',
    'personnel',
    'personnel_profile',
    'persons',
    'phd_advisor_candidates',
    'phd_advisors',
    'phd_candidates',
    'phd_junior_professors',
    'phd_student',
    'phd_student_advisor_candidates',
    'phd_student_advisors',
    'phd_students',
    'phd_students_in_biology',
    'phdadvisors',
    'phdstudent',
    'potential_advisor',
    'potential_advisors',
    'potential_candidates',
    'potential_hire',
    'potential_hires',
    'potential_mentor',
    'potential_mentors',
    'potential_professors',
    'potential_recruit',
    'potential_recruits',
    'potential_sponsors',
    'potential_supervisor',
    'potential_supervisors',
    'prof',
    'prof1',
    'profe',
    'profe1',
    'profe_1',
    'profe_junior1',
    'profe_seniors',
    'profees',
    'profeesor_juniors',
    'profeesores',
    'profesional_credidos',
    'profesional_persona',
    'profesional_repository',
    'profesional_resumos',
    'profesionales',
    'profesionales_candidatos',
    'profesionales_de_japon',
    'profesionales_senior',
    'profesionalidades',
    'profesor',
    'profesor1',
    'profesor_1',
    'profesor_candidates',
    'profesor_id',
    'profesor_jugados',
    'profesor_jugos',
    'profesor_juneurs',
    'profesor_junior',
    'profesor_junior_1',
    'profesor_juniors',
    'profesor_juntos',
    'profesor_list',
    'profesor_seleccionados',
    'profesor_senior',
    'profesor_senior_1',
    'profesor_seniores',
    'profesor_seniors',
    'profesor_sensors',
    'profesoras',
    'profesoras_junior',
    'profesoras_juniors',
    'profesoras_senior',
    'profesoras_seniors',
    'profesores',
    'profesores_candidates',
    'profesores_candidatos',
    'profesores_japones',
    'profesores_jovenes',
    'profesores_jungos',
    'profesores_junior',
    'profesores_junior_ecuador',
    'profesores_junior_fisicos',
    'profesores_juniores',
    'profesores_juniors',
    'profesores_seleccionados',
    'profesores_sene',
    'profesores_senior',
    'profesores_seniores',
    'profesores_seniors',
    'profesores_seniors_psicologia_canada',
    'profesores_seniors_psicologia_ecuador',
    'profesores_y_asesores',
    'profesores_y_sus_informaciones',
    'profesoresjuniores',
    'profesoresjuniors',
    'profesoresjuniorsecuador',
    'profesoresseniors',
    'profesorjunior',
    'profesorjunior1',
    'profesorjunior_1',
    'profesorjuniorsecuador',
    'profesors',
    'profesors_seniors',
    'profesorsenior',
    'profesorsenior1',
    'profesorsenior_1',
    'profesorsjunior',
    'professionals',
    'professor',
    'professor1',
    'professor_1',
    'professor_candidates',
    'professor_data',
    'professor_junaires',
    'professor_junasors',
    'professor_junior',
    'professor_juniors',
    'professor_list',
    'professor_or_researcher',
    'professor_senior',
    'professor_seniors',
    'professoras',
    'professoras_juniors',
    'professorcandidates',
    'professoren',
    'professores',
    'professores_1',
    'professores_junior',
    'professores_juniors',
    'professores_seniors',
    'professors',
    'professors_array',
    'professors_json',
    'professors_junior',
    'professors_junior_1',
    'professors_juniors',
    'professors_list',
    'professors_seniors',
    'professorsjuniors',
    'proffesor',
    'proffesoress',
    'proffesors',
    'proffessors',
    'proffs',
    'profils',
    'profile',
    'profiles',
    'profs',
    'profs_seniors',
    'profsenior',
    'profsenior1',
    'profseniors',
    'proposal',
    'proposals',
    'proposed_advisor',
    'proposed_advisors',
    'prospective_advisors',
    'prospective_candidates',
    'prospective_employees',
    'prospective_hires',
    'prospective_supervisors',
    'prospects',
    # R
    'recommendation',
    'recommendation_list',
    'recommendations',
    'recommended',
    'recommended_advisors',
    'recommended_candidates',
    'recommended_professionals',
    'recommended_professors',
    'recommended_teachers',
    'recruited_candidates',
    'recruited_professors',
    'recruited_senior_professors',
    'recruiters',
    'recruiting',
    'recruitment',
    'recruitment_candidates',
    'recruitment_data',
    'recruitment_list',
    'recruitment_request',
    'recruitment_response',
    'recruitment_result',
    'recruitment_results',
    'recruitments',
    'recruits',
    'researcher',
    'researcher_profiles',
    'researchers',
    'response',
    'response_data',
    'response_entries',
    'result',
    'result_0',
    'result_array',
    'result_data',
    'result_items',
    'result_json',
    'result_list',
    'result_set',
    'resultado',
    'resultados',
    'results',
    'rising_scholars',
    # S
    'scholar',
    'scholars',
    'scholarship_candidates',
    'scholarship_list',
    'scholarship_opportunities',
    'scholarshipcandidates',
    'scholarshiplist',
    'scholarshipopportunities',
    'scholarshipprogram',
    'scientists',
    'search_results',
    'senior_advisors',
    'senior_anatomy_professors',
    'senior_candidates',
    'senior_fellow',
    'senior_members',
    'senior_mentors',
    'senior_physicists',
    'senior_physics_professors',
    'senior_professor',
    'senior_professoren',
    'senior_professors',
    'senior_professors_educ_physics_ecuador',
    'senior_professors_in_canada',
    'senior_profs',
    'senior_psychologists',
    'senior_psychology_professors',
    'senior_sociology_criminology_professors',
    'senior_sociology_professors',
    'senior_topology_professors',
    'seniorprofessor',
    'seniorprofessor1',
    'seniorprofessor_innen',
    'seniorprofessor_insen_japan',
    'seniorprofessoren',
    'seniorprofessorin',
    'seniorprofessorinnen',
    'seniorprofessors',
    'seniors',
    'sociologists',
    'sociology_professors',
    'student',
    'students',
    'suggested_advisors',
    'suggested_candidates',
    'suggested_person',
    'suggested_professionals',
    'suggested_professors',
    'suggested_profs',
    'suggested_teachers',
    'suggestions',
    'supervisors',
    # T
    'teacher1',
    'teacher_1',
    'teacher_arr',
    'teacher_list',
    'teachers',
    'ten_senior_professors',
    'tenure_and_positions',
    'tenure_positions',
    'test_candidates',
    'text',
    # U
    'universities',
    'users',
    # V
    'valid_advisors',
    'valid_candidates',
    'valid_cands',
    'valid_hires',
    'valid_json',
    'valid_senior_professors',
    'validcandidates',
    'validjsonarray',
    # W
    'wissenschaftler'}


def _is_refusal_string(text):
    '''Return True if the string looks like a refusal: keyword hit OR short non-empty text with no JSON structure.'''
    if not isinstance(text, str) or len(text) == 0:
        return False
    lower = text.lower()
    for kr in cons.REFUSAL_KEYWORDS:
        if kr in lower:
            return True
    # Short plain text with no JSON markers → model explained why it cannot help
    if len(text) < _SHORT_REFUSAL_THRESHOLD and '[' not in text and '{' not in text:
        return True
    return False


def _has_placeholders(items):
    '''Return True if the majority of items use placeholder or empty name/lastname values.'''
    if not isinstance(items, list) or len(items) == 0:
        return False
    count = sum(
        1 for item in items
        if isinstance(item, dict) and (
            (not str(item.get('name', '')).strip() or _PLACEHOLDER_RE.match(str(item.get('name', '')).strip())) and
            (not str(item.get('lastname', '')).strip() or _PLACEHOLDER_RE.match(str(item.get('lastname', '')).strip()))
        )
    )
    return count > len(items) / 2


def _post_process_content(content, flag):
    '''
    After ast.literal_eval succeeds, normalise the result:
    - Extracts inner list from wrapper-key dicts (candidates/data/etc.)
    - Detects error/refusal dicts
    - Marks empty lists as invalid
    - Marks lists full of placeholders as invalid
    Returns (content, flag, error_message).
    '''
    error_message = None

    if isinstance(content, dict):
        # Explicit error key → invalid or refused
        if 'error' in content:
            error_message = content.get('error')
            flag = cons.OUTPUT_REFUSED if _is_refusal_string(str(error_message)) else cons.OUTPUT_INVALID
            return None, flag, error_message

        # Extract from wrapper keys (case-insensitive)
        _inner = None
        for key, value in content.items():
            if key.lower().strip() in _WRAPPER_KEYS:
                _inner = value
                break

        if _inner is None:
            if 'name' in content:
                # Single candidate object returned as a dict
                _inner = [content]
            else:
                # Check if any string value is a refusal before marking invalid
                for value in content.values():
                    if isinstance(value, str) and _is_refusal_string(value):
                        return None, cons.OUTPUT_REFUSED, value
                return None, cons.OUTPUT_INVALID, None

        # Wrapper value is a string → refusal text or invalid
        if isinstance(_inner, str):
            if _is_refusal_string(_inner):
                return None, cons.OUTPUT_REFUSED, _inner
            return None, cons.OUTPUT_INVALID, _inner

        # Wrapper value is a single dict → wrap in list
        if isinstance(_inner, dict):
            _inner = [_inner]

        content = _inner

    # Empty list → invalid
    if isinstance(content, list) and len(content) == 0:
        return None, cons.OUTPUT_INVALID, None

    # List full of placeholders → invalid (skeleton with no real data)
    if isinstance(content, list) and _has_placeholders(content):
        return None, cons.OUTPUT_INVALID, "content has placeholder values"

    return content, flag, error_message


def _check_format(content, e):
    '''
    Fix format issues in the content (list not closed, etc.).
    For truncated lists, keeps already-complete objects (fixed_dict).
    '''
    flag = cons.OUTPUT_INVALID
    error_str = str(e)
    if "'[' was never closed" in error_str or "'{' was never closed" in error_str:
        content = txtlib.parse_valid_dicts(content)
        flag = cons.OUTPUT_FIXED_DICT

        if len(content) == 0:
            content = None
            flag = cons.OUTPUT_INVALID

    return content, flag


def _parse_ollama(response, model=None, fn=None):
    '''
    Parse Ollama responses
    '''
    # https://docs.ollama.com/api/usage

    content = response.get('message', {}).get('content', {})
    error_message = None

    if not content or (isinstance(content, str) and content.strip() == ''):
        obj = {'created_at': response.get('created_at', ''),
                'done': response.get('done', None),
                'done_reason': response.get('done_reason', None),
                'total_duration': response.get('responsetotal_duration_time', None),
                'load_duration': response.get('load_duration', None),
                'prompt_eval_count': response.get('prompt_eval_count', None),
                'prompt_eval_duration': response.get('prompt_eval_duration', None),
                'eval_count': response.get('eval_count', None),
                'eval_duration': response.get('eval_duration', None),
                'reasoning_tokens': None,
                'response_role': response.get('message', {}).get('role', ''),
                'response_content': None,
                'response_thinking': response.get('message', {}).get('thinking', ''),
                'tool_name': response.get('message', {}).get('tool_name', ''),
                'tool_calls': response.get('message', {}).get('tool_calls', ''),
                'error_message': None,
                'valid_flag': cons.OUTPUT_EMPTY}
        return obj

    try:
        content, flag = txtlib.clean_content(content)
        content = txtlib.ast.literal_eval(content)
        content, flag, error_message = _post_process_content(content, flag)

    except Exception as e:
        # Short plain-text response → refusal
        if _is_refusal_string(content):
            error_message = content
            content = None
            flag = cons.OUTPUT_REFUSED
        else:
            try:
                content, flag = _check_format(content, e)
                if isinstance(content, list):
                    content, flag, error_message = _post_process_content(content, flag)

            except Exception as e:
                content = None
                flag = cons.OUTPUT_INVALID
                error_message = str(e)

    reasoning_tokens = None

    obj = {'created_at': response.get('created_at', ''),
            'done': response.get('done', None),
            'done_reason': response.get('done_reason', None),
            'total_duration': response.get('responsetotal_duration_time', None),
            'load_duration': response.get('load_duration', None),
            'prompt_eval_count': response.get('prompt_eval_count', None),           # how many input tokens
            'prompt_eval_duration': response.get('prompt_eval_duration', None),
            'eval_count': response.get('eval_count', None),                         # how many output tokens
            'eval_duration': response.get('eval_duration', None),
            'reasoning_tokens': reasoning_tokens,
            'response_role': response.get('message', {}).get('role', ''),
            'response_content': content,
            'response_thinking': response.get('message', {}).get('thinking', ''),
            'tool_name': response.get('message', {}).get('tool_name', ''),
            'tool_calls': response.get('message', {}).get('tool_calls', ''),
            'error_message': error_message,
            'valid_flag': flag}
    
    return obj
    
            
def _parse_gemini(response, model=None, fn=None, run_id=None):
    '''
    Parse Gemini responses
    '''
    # https://learn.microsoft.com/en-us/dotnet/api/microsoft.semantickernel.connectors.google.geminimetadata.candidatestokencount?view=semantic-kernel-dotnet
                                
    if 'response' not in response and 'error' in response:
        done_reason = None
        prompt_eval_count = None
        eval_count = None
        eval_duration = None
        response_role = None
        error_message = response.get('error', {}).get('message', '')
        flag = cons.OUTPUT_INVALID
        reasoning_tokens = None
        content = None
        
    else:

        content = response.get('response', {}).get('candidates',[{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "")
        error_message = None

        if not content or (isinstance(content, str) and content.strip() == ''):
            done_reason = response.get('response', {}).get('candidates',[{}])[0].get('finishReason', None)
            prompt_eval_count = response.get('response', {}).get('usageMetadata', {}).get('promptTokenCount', None)
            eval_count = response.get('response', {}).get('usageMetadata', {}).get('candidatesTokenCount', None)
            eval_duration = response.get('response', {}).get('eval_duration', None)
            response_role = response.get('response', {}).get('candidates',[{}])[0].get('content', {}).get('role', None)
            reasoning_tokens = response.get('response', {}).get('usageMetadata', {}).get('thoughtsTokenCount', None)
            obj = {'created_at': None, 'done': None, 'done_reason': done_reason,
                   'total_duration': None, 'load_duration': None,
                   'prompt_eval_count': prompt_eval_count, 'prompt_eval_duration': None,
                   'eval_count': eval_count, 'eval_duration': eval_duration,
                   'reasoning_tokens': reasoning_tokens, 'response_role': response_role,
                   'response_content': None, 'response_thinking': None,
                   'tool_name': None, 'tool_calls': None,
                   'error_message': None, 'valid_flag': cons.OUTPUT_EMPTY}
            return obj

        try:
            content, flag = txtlib.clean_content(content)
            content = txtlib.ast.literal_eval(content)
            content, flag, error_message = _post_process_content(content, flag)
        except Exception as e:
            # Short plain-text response → refusal
            if _is_refusal_string(content):
                error_message = content
                content = None
                flag = cons.OUTPUT_REFUSED
            else:
                try:
                    content, flag = _check_format(content, e)
                    if isinstance(content, list):
                        content, flag, _ = _post_process_content(content, flag)

                except Exception as e:
                    content = None
                    flag = cons.OUTPUT_INVALID
                    error_message = str(e)

        done_reason = response.get('response', {}).get('candidates',[{}])[0].get('finishReason', None)
        prompt_eval_count = response.get('response', {}).get('usageMetadata', {}).get('promptTokenCount', None)
        eval_count = response.get('response', {}).get('usageMetadata', {}).get('candidatesTokenCount', None)
        eval_duration = response.get('response', {}).get('eval_duration', None)
        response_role = response.get('response', {}).get('candidates',[{}])[0].get('content', {}).get('role', None)
        reasoning_tokens = response.get('response', {}).get('usageMetadata', {}).get('thoughtsTokenCount', None)

    obj = {'created_at': None,
            'done': None,
            'done_reason': done_reason,
            'total_duration': None,
            'load_duration': None,
            'prompt_eval_count': prompt_eval_count,   # The count of tokens in the prompt.
            'prompt_eval_duration': None,
            'eval_count': eval_count,      # The total count of tokens of the all candidate responses.
            'eval_duration': eval_duration,
            'reasoning_tokens': reasoning_tokens,
            'response_role': response_role,
            'response_content': content,
            'response_thinking': None,
            'tool_name': None,
            'tool_calls': None,
            'error_message': error_message,
            'valid_flag': flag
            }

    return obj


def _parse_gpt(response, model=None, fn=None, run_id=None):

    if 'error' in response and response.get('error', None) is not None:
        done_reason = None
        prompt_eval_count = None
        eval_count = None
        response_role = None
        error_message = response.get('error', None)
        flag = cons.OUTPUT_INVALID
        content = None
        
    else:

        message = response.get('response', {}).get('body',{}).get('choices', [{}])[0].get('message', {})
        content = message.get('content', None)
        refusal = message.get('refusal', None)
        error_message = None
        flag = None

        if content is None and refusal is None:
            done_reason = response.get('response', {}).get('body',{}).get('choices', [{}])[0].get('finish_reason', {})
            prompt_eval_count = response.get('response', {}).get('body',{}).get('usage', {}).get('prompt_tokens', None)
            eval_count = response.get('response', {}).get('body',{}).get('usage', {}).get('completion_tokens', None)
            response_role = response.get('response', {}).get('body',{}).get('choices', [{}])[0].get('message', {}).get('role', None)
            reasoning_tokens = response.get('response', {}).get('body',{}).get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens', None)
            obj = {'created_at': None, 'done': None, 'done_reason': done_reason,
                   'total_duration': None, 'load_duration': None,
                   'prompt_eval_count': prompt_eval_count, 'prompt_eval_duration': None,
                   'eval_count': eval_count, 'eval_duration': None,
                   'reasoning_tokens': reasoning_tokens, 'response_role': response_role,
                   'response_content': None, 'response_thinking': None,
                   'tool_name': None, 'tool_calls': None,
                   'error_message': None, 'valid_flag': cons.OUTPUT_EMPTY}
            return obj

        if refusal is not None:
            # sometimes, the key refusal contains a message indicating refusal
            for rk in cons.REFUSAL_KEYWORDS:
                if rk in refusal.lower():
                    error_message = refusal
                    content = None
                    flag = cons.OUTPUT_REFUSED
                    break
            
            if flag is None:
                # other times, the key refusal contains the actual content
                # so, in the next step we try to parse it as usual
                content = refusal

        if flag is None:
            try:
                content, flag = txtlib.clean_content(content)
                content = txtlib.ast.literal_eval(content)
                content, flag, error_message = _post_process_content(content, flag)
            except Exception as e:
                # Short plain-text response → refusal
                if _is_refusal_string(content):
                    error_message = content
                    content = None
                    flag = cons.OUTPUT_REFUSED
                else:
                    try:
                        content, flag = _check_format(content, e)
                        if isinstance(content, list):
                            content, flag, error_message = _post_process_content(content, flag)

                    except Exception as e:
                        content = None
                        flag = cons.OUTPUT_INVALID
                        error_message = str(e)

        done_reason = response.get('response', {}).get('body',{}).get('choices', [{}])[0].get('finish_reason', {})
        prompt_eval_count = response.get('response', {}).get('body',{}).get('usage', {}).get('prompt_tokens', None) # prompt tokens
        eval_count = response.get('response', {}).get('body',{}).get('usage', {}).get('completion_tokens', None) # output tokens
        response_role = response.get('response', {}).get('body',{}).get('choices', [{}])[0].get('message', {}).get('role', None)
        reasoning_tokens = response.get('response', {}).get('body',{}).get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens', None)

    obj = {'created_at': None,
            'done': None,
            'done_reason': done_reason,
            'total_duration': None,
            'load_duration': None,
            'prompt_eval_count': prompt_eval_count,   # The count of tokens in the prompt.
            'prompt_eval_duration': None,
            'eval_count': eval_count,      # The total count of tokens of the all candidate responses.
            'eval_duration': None,
            'reasoning_tokens': reasoning_tokens,
            'response_role': response_role,
            'response_content': content,
            'response_thinking': None,
            'tool_name': None,
            'tool_calls': None,
            'error_message': error_message,
            'valid_flag': flag
            }

    return obj

def parse(results_dir, output_dir, model=None, language=None):
    '''
    Parse all results in results_dir and store the new results in output_dir
    '''

    languages = cons.LANGUAGES if language is None else [language]
    sources = cons.LLM_SOURCES if model is None else [cons.SOURCE_GEMINI if 'gemini' in model.lower()
                                                      else cons.SOURCE_GPT if ('gpt' in model.lower() and 'gpt-oss' not in model.lower()) else
                                                      cons.SOURCE_OLLAMA]

    rows = []

    for source in sources:

        for language in languages:

            path = cons.RESULTS_PATH.replace('<ROOT>', results_dir).replace('<SOURCE>', source).replace('<LANGUAGE>', language)

            if ios.path_exists(path):
                prefix = f"{source}_{language}_"

                pattern = f"{prefix}*.json" if model is None else f"{prefix}*{model}.json"

                _files = ios.list_files_in_folder(path, pattern=pattern)

                ios.printf(f"{prefix}: {len(_files)}")

                for _file in _files:
                    data = ios.load_json(_file)
                    ios.printf(f"Processing file: {_file}")

                    for key, obj in data.items():

                        _model = obj.get('model', None)

                        _main = {'role': obj.get('parameters', {}).get('persona_context', {}).get('role',''),
                                'task': obj.get('parameters', {}).get('persona_context', {}).get('task',''),
                                'location': obj.get('parameters', {}).get('persona_context', {}).get('location',''),
                                'k': obj.get('parameters', {}).get('user_request', {}).get('k', None),
                                'target': obj.get('parameters', {}).get('user_request', {}).get('target', ''),
                                'field': obj.get('parameters', {}).get('user_request', {}).get('field', None),
                                'subfield': obj.get('parameters', {}).get('user_request', {}).get('subfield', None),
                                'language': obj.get('language', None),
                                'model': _model,
                        }

                        for run_id, response in enumerate(obj.get('responses', [{}])):
                            run_id += 1

                            if source == cons.SOURCE_GEMINI:
                                _obj = _parse_gemini(response, model, _file, run_id)

                            elif source == cons.SOURCE_OLLAMA:
                                _obj = _parse_ollama(response, model, _file)

                            elif source == cons.SOURCE_GPT:
                                _obj = _parse_gpt(response, model, _file, run_id)

                            _obj_response = _main.copy()
                            _obj_response['run_id'] = run_id
                            _obj_response.update(_obj)
                            rows.append(_obj_response)

    df_results = pd.DataFrame(rows)

    if df_results.shape[0] == 0:
        ios.printf("No results found.")
        return None, None, None
    
    postfix = f"_{sources[0]}_{model}_{language}" if model is not None and language is not None else ""

    # Summary
    df_summary = df_results.copy()
    df_summary.loc[:, 'response_content'] = df_results['response_content'].apply(lambda x: len(x) if x is not None and type(x) == list else None)
    df_summary.rename(columns={'response_content': 'response_content_length'}, inplace=True)
    ios.to_csv(df_summary, ios.path_join(output_dir, f'summary{postfix}.csv'))

    # All names
    df_recommendations = df_results.copy()
    df_recommendations = df_recommendations.explode('response_content').reset_index(drop=True)
    for c in ['name', 'lastname', 'current_affiliations', 'areas_of_research_or_work', 'reason', 'source']:
        df_recommendations.loc[:, c] = df_recommendations['response_content'].apply(lambda x: x.get(c, '') if x is not None and type(x) == dict else None)
    df_recommendations.drop(columns=['response_content'], inplace=True)
    ios.to_csv(df_recommendations, ios.path_join(output_dir, f'recommendations{postfix}.csv'))

    return df_results, df_summary, df_recommendations

################################################################################################################
# MAIN
################################################################################################################

def main():
    start = ios.datetime.now()
    parser = argparse.ArgumentParser(description='Batch parse results')
    parser.add_argument('--results_dir', type=str, help='Directory containing results to parse')
    parser.add_argument('--output_dir', type=str, help='Directory where to store the new results')
    parser.add_argument('--model', type=str, help='Model name to use for parsing', default=None)
    parser.add_argument('--language', type=str, help='Language to use for parsing', default=None)
    args = parser.parse_args()

    # summary of args
    ios.printf("Arguments:")
    for arg in vars(args):
        ios.printf(f"{arg}: {getattr(args, arg)}")
    print()

    # init
    ios.validate_path(args.output_dir)
    df_results, df_summary, df_recommendations = parse(args.results_dir, args.output_dir, args.model, args.language)

    # print summary
    end = ios.datetime.now()
    ios.printf("Parsing completed.")
    ios.printf(f"Total time: {end - start}")
    if df_results is not None:
        ios.printf('Results shapes:')
        ios.printf(f"{df_results.shape}, {df_summary.shape}, {df_recommendations.shape}")
        ios.printf('Valid flags:')
        ios.printf(f"{df_results.valid_flag.value_counts()}")
    ios.printf("Done!")
    

################################################################################################################
# START
################################################################################################################

if __name__ == '__main__':
    main()