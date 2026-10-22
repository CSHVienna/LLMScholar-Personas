# cd scripts
# export PYTHONPATH="${PYTHONPATH}:../" 
# python gpt_gemini_01_create_batch_files.py ...

import argparse
import datetime
import json
import os
import sys

from libs.utils.logger import debug_msg, DebugLevel
from libs.utils.utils import (add_temp_data,
                   check_prompt_already_processed,
                   create_folder,
                   get_duration_string,
                   load_all_responses_from_file,
                   load_json,
                   read_and_delete_temp_files,
                   save_all_responses_to_json_file)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m', '--model',
                    help='Run only a specific model.')
parser.add_argument('-r', '--repetitions', type=int, default=10,
                    help='Number of repetitions.')
parser.add_argument('-l', '--language', choices=["english", "german", "spanish"], default="english",
                    help='Specify the language used for the prompts.')
args = parser.parse_args()

NR_REPETITIONS = args.repetitions
LANGUAGE = args.language

# only use one model at once
MODEL_LIST = [
    "gpt-4.1-nano-2025-04-14",
    # "gpt-4.1-mini-2025-04-14",
    # "gpt-4.1-2025-04-14",
    # "gemini-2.5-flash-lite",
    # "gemini-2.5-flash",
    # "gemini-2.5-pro",
]

MODEL_FAMILY = MODEL_LIST[0].split("-")[0]
assert MODEL_FAMILY == "gpt" or MODEL_FAMILY == "gemini"
for model in MODEL_LIST:
    assert model.startswith(MODEL_FAMILY)

DIRECTORY = os.path.dirname(os.path.realpath(__file__))
RESULTS_DIRECTORY = os.path.join(DIRECTORY, f"results_{MODEL_FAMILY}_{LANGUAGE}")
RESULTS_FILENAME = f"{MODEL_FAMILY}_{{language}}_{{model}}.json"
BATCH_FILE_FILENAME = f"_requests_batch_file__{MODEL_FAMILY}__{{language}}__{{model}}__repetition_{{repetition:02d}}.jsonl"
LINE_LIMIT = None


def prepare_gpt_request(model, prompt_id, prompt_content):
    messages = [{"role": "user", "content": prompt_content}]
    debug_msg("request:  " + str(messages), DebugLevel.DEBUG)
    json_line = {
        "custom_id": prompt_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
    }
    return json_line


def prepare_gemini_request(prompt_id, prompt_content):
    messages = [{"parts": [{"text": prompt_content}]}]
    debug_msg("request:  " + str(messages), DebugLevel.DEBUG)

    json_line = {
        "key": prompt_id,
        "request": {
            "contents": messages,
            "generation_config": {
                "responseMimeType": "application/json",
            }
        }
    }
    return json_line


def process_prompt(prompt_item, model):
    prompt_content = prompt_item["prompt"].strip()
    debug_msg("Next prompt to process: " + str(prompt_item["id"]), DebugLevel.LOG)

    if MODEL_FAMILY == "gpt":
        json_line = prepare_gpt_request(model, str(prompt_item["id"]), prompt_content)
    if MODEL_FAMILY == "gemini":
        json_line = prepare_gemini_request(str(prompt_item["id"]), prompt_content)

    debug_msg("Prompt %s processed." % (prompt_item["id"]), DebugLevel.LOG)
    return json_line


def run_repetition(language,
                   model,
                   all_params_and_prompts,
                   repetition,
                   all_responses,
                   results_directory=RESULTS_DIRECTORY):

    batch_file_filename = BATCH_FILE_FILENAME.format(
        language=language,
        model=model,
        repetition=repetition,
    )

    if os.path.exists(os.path.join(results_directory, batch_file_filename)):
        debug_msg("Batch file %s already exists for this cycle. Return..." % (batch_file_filename), DebugLevel.LOG)
        return

    total_nr_prompts = len(all_params_and_prompts)
    debug_msg("%d prompts to process..." % total_nr_prompts, DebugLevel.LOG)
    prompt_ids_unprocessed = set()
    prompt_items_unprocessed = []

    all_temp_data = read_and_delete_temp_files(language, model, results_directory)
    if len(all_temp_data):
        all_responses = add_temp_data(all_responses, all_temp_data, language, model)
        save_all_responses_to_json_file(all_responses, language, model, results_directory, RESULTS_FILENAME)

    for item in all_params_and_prompts:
        prompt_id = str(item["id"])
        if not prompt_id in all_responses:
            all_responses[prompt_id] = item
            all_responses[prompt_id]["language"] = language
            all_responses[prompt_id]["model"] = model
            all_responses[prompt_id]["responses"] = []

        if not check_prompt_already_processed(prompt_id, repetition, all_responses):
            prompt_ids_unprocessed.add(prompt_id)
            prompt_items_unprocessed.append(item)
    if len(prompt_ids_unprocessed) < len(all_params_and_prompts):
        debug_msg("%d prompts already processed before." % (len(all_params_and_prompts) - len(prompt_ids_unprocessed)), DebugLevel.LOG)
        debug_msg("%d prompts left to process..." % len(prompt_ids_unprocessed), DebugLevel.LOG)

    save_all_responses_to_json_file(all_responses, language, model, results_directory, RESULTS_FILENAME)
    if len(prompt_ids_unprocessed) == 0: return

    json_lines = list()
    for prompt_item in prompt_items_unprocessed:
        json_lines.append(process_prompt(prompt_item, model))

    if LINE_LIMIT is None:
        with open(os.path.join(results_directory, batch_file_filename), 'w', encoding='utf8') as file:
            for json_line in json_lines:
                file.write(json.dumps(json_line, ensure_ascii=False) + '\n')
        debug_msg("Created batch file %s for %d prompts." % (batch_file_filename, len(prompt_items_unprocessed)), DebugLevel.LOG)
    else:
        i = 0
        while True:
            if LINE_LIMIT*i >= len(json_lines): break
            with open(os.path.join(results_directory, batch_file_filename + "_PART_%02d" % (i+1)), 'w', encoding='utf8') as file:
                for json_line in json_lines[LINE_LIMIT*i:LINE_LIMIT*(i+1)]:
                    file.write(json.dumps(json_line, ensure_ascii=False) + '\n')
            i += 1
            debug_msg("Created batch file %s for %d prompts." % (batch_file_filename + "_PART_%02d" % (i+1), len(json_lines[LINE_LIMIT*i:LINE_LIMIT*(i+1)])), DebugLevel.LOG)
    return


def run_repetitions(language,
                    model,
                    all_params_and_prompts,
                    nr_repetitions=NR_REPETITIONS,
                    results_directory=RESULTS_DIRECTORY):

    debug_msg("%d repetitions to process..." % nr_repetitions, DebugLevel.LOG)
    all_responses = load_all_responses_from_file(language, model, results_directory, RESULTS_FILENAME)

    for repetition in range(1, nr_repetitions+1):
        debug_msg("Next repetition to process: %s [from %d repetitions (%d%%)]" % (repetition, nr_repetitions, int(100.0 * repetition / nr_repetitions)), DebugLevel.LOG)
        run_repetition(language,
                      model,
                      all_params_and_prompts,
                      repetition,
                      all_responses,
                      results_directory)
    debug_msg("Processing %d repetitions with model %s done." % (nr_repetitions, model), DebugLevel.LOG)
    return


def main():
    all_params_and_prompts = load_json(f"data/context/{LANGUAGE}/all_params_and_prompts.json")
    results_directory = RESULTS_DIRECTORY
    create_folder(results_directory)

    for model in MODEL_LIST:
        debug_msg("Processing model %s" % str(model), level=DebugLevel.LOG)
        run_repetitions(
            LANGUAGE,
            model,
            all_params_and_prompts,
            nr_repetitions=NR_REPETITIONS,
            results_directory=results_directory
        )


if __name__ == "__main__":

    start_time = datetime.datetime.now()
    debug_msg("Started %s" % str(sys.argv[0]), DebugLevel.DEBUG)

    main()

    end_time = datetime.datetime.now()
    debug_msg("Finished %s\tDuration: %s" % (str(sys.argv[0]), get_duration_string(start_time, end_time)), DebugLevel.DEBUG)
