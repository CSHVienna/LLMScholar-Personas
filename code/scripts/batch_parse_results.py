# export PYTHONPATH="$PYTHONPATH:../libs"

import argparse
import pandas as pd

from utils import ios
from utils import constants as cons
from utils import text as txtlib


def _check_format(content, e):
    '''
    Fix format issues in the content (dictionary not closed, list not closed, etc)
    '''
    flag = cons.OUTPUT_INVALID
    if "'[' was never closed" in str(e):
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

    try:
        content, flag = txtlib.clean_content(content)
        content = txtlib.ast.literal_eval(content)
        error_message = None

        if 'error' in content:
            error_message = content.get('error', None)
            content = None
            flag = cons.OUTPUT_INVALID
        else:
            _content = None
            # candidates, students, profesors, data, juniorprofessors
            for key_candidate in ['candidates', 'students', 'profesors', 'data', 'juniorprofessors', 'text']:
                if key_candidate in content:
                    _content = content.get(key_candidate, [{}])
                    break

            if _content is None:
                if 'name' in content:
                    _content = [content]

            content = _content
            
    except Exception as e:

        try:
            content, flag = _check_format(content, e)
                    
        except Exception as e:
            ios.printf(f"\n====================\n{model} {fn} {e} {response.get('created_at', '')} \n >>>{content}<<<\n====================\n")
            content = None
            flag = cons.OUTPUT_INVALID
            error_message = str(e)

    obj = {'created_at': response.get('created_at', ''),
            'done': response.get('done', None),
            'done_reason': response.get('done_reason', None),
            'total_duration': response.get('responsetotal_duration_time', None),
            'load_duration': response.get('load_duration', None),
            'prompt_eval_count': response.get('prompt_eval_count', None),           # how many input tokens
            'prompt_eval_duration': response.get('prompt_eval_duration', None),
            'eval_count': response.get('eval_count', None),                         # how many output tokens
            'eval_duration': response.get('eval_duration', None),
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
        content = None
        
    else:

        content = response.get('response', {}).get('candidates',[{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "")
        error_message = None

        try:
            content, flag = txtlib.clean_content(content)
            content = txtlib.ast.literal_eval(content)
            error_message = None
        except Exception as e:
            try:
                content, flag = _check_format(content, e)

            except Exception as e:
                ios.printf(f"\n====================\n{model} {fn} {e} -{response.get('response', {}).get('responseId','')}- {response.get('key', '')} {run_id} \n >>>{content}<<<\n====================\n")
                content = None
                flag = cons.OUTPUT_INVALID
                error_message = str(e)

        done_reason = response.get('response', {}).get('candidates',[{}])[0].get('finishReason', None)
        prompt_eval_count = response.get('response', {}).get('usageMetadata', {}).get('promptTokenCount', None)
        eval_count = response.get('response', {}).get('usageMetadata', {}).get('candidatesTokenCount', None)
        eval_duration = response.get('response', {}).get('eval_duration', None)
        response_role = response.get('response', {}).get('candidates',[{}])[0].get('content', {}).get('role', None)
        
    obj = {'created_at': None,
            'done': None,
            'done_reason': done_reason,
            'total_duration': None,
            'load_duration': None,
            'prompt_eval_count': prompt_eval_count,   # The count of tokens in the prompt.
            'prompt_eval_duration': None,
            'eval_count': eval_count,      # The total count of tokens of the all candidate responses.
            'eval_duration': eval_duration,
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
    sources = cons.LLM_SOURCES if model is None else [cons.SOURCE_GEMINI if 'gemini' in model.lower() else cons.SOURCE_OLLAMA]

    df_results = pd.DataFrame()
    df_summary = pd.DataFrame()
    df_recommendations = pd.DataFrame()

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

                            _obj_response = _main.copy()
                            _obj_response['run_id'] = run_id
                            _obj_response.update(_obj)       
                            df_results = pd.concat([df_results, pd.DataFrame([_obj_response])], ignore_index=True)

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