import argparse
import datetime
from dotenv import load_dotenv
from google import genai
import json
from logger import debug_msg, DebugLevel
from openai import OpenAI
import os
import sys
from utils import (add_temp_data,
                   get_duration_string,
                   load_all_responses_from_file,
                   read_and_delete_temp_files,
                   save_all_responses_to_json_file,
                   save_response_to_temp_file)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-mf', '--model_family', choices=["gpt", "gemini"], default="gpt",
                    help='Run a specific model family.')
args = parser.parse_args()

MODEL_FAMILY = args.model_family
DIRECTORY = os.path.dirname(os.path.realpath(__file__))
RESULTS_DIRECTORY = os.path.join(DIRECTORY, f"results_{MODEL_FAMILY}_{{language}}")
RESULTS_FILENAME = f"{MODEL_FAMILY}_{{language}}_{{model}}.json"
BATCH_FILE_FILENAME = f"_requests_batch_file__{MODEL_FAMILY}__{{language}}__{{model}}__repetition_{{repetition:02d}}.jsonl"


def save_json_to_file(filename, json_content):
    with open(filename, "w", encoding='utf8') as outfile:
        json.dump(json_content, outfile, ensure_ascii=False)


def retrieve_gpt_results():
    client = OpenAI(api_key=OPENAI_API_KEY)
    batch_filename_list_avoid_duplicate_processing = set()
    for batch in client.batches.list(limit=100).data:
        if batch.metadata is None: continue
        batch_file_filename = batch.metadata["description"]
        if batch_file_filename in batch_filename_list_avoid_duplicate_processing:
            continue
        batch_filename_list_avoid_duplicate_processing.add(batch_file_filename)
        if batch.status != "completed":
            if datetime.datetime.fromtimestamp(float(batch.created_at)) > (datetime.datetime.now() - datetime.timedelta(days=2)):
                debug_msg("Batch: %s; status: %s" % (batch_file_filename, str(batch.status)), level=DebugLevel.LOG)
                if batch.status == "failed":
                    debug_msg("  errors: %s" % (str(batch.errors)), level=DebugLevel.LOG)
            continue

        try:
            _, _, language, model, repetition = batch_file_filename.split("__")
            repetition = int(repetition.split("_")[1].split(".")[0])
        except:
            continue

        results_directory = RESULTS_DIRECTORY.format(language=language)

        if not os.path.exists(os.path.join(results_directory, batch_file_filename)):
            continue

        lock_file = os.path.join(results_directory, f"_{repetition}_BATCH.LOCK")
        if not os.path.exists(lock_file):
            continue

        debug_msg("Batch: %s; status: %s" % (batch_file_filename, str(batch.status)), level=DebugLevel.LOG)
        batch_response = client.files.content(batch.output_file_id)
        batch_outfile_filename = batch_file_filename.replace("_requests_", "_responses_")
        with open(os.path.join(results_directory, batch_outfile_filename), 'w') as file:
            file.write(batch_response.text)

        with open(os.path.join(results_directory, batch_outfile_filename), 'r') as file:
            for line in file:
                json_object = json.loads(line.strip())
                prompt_id = json_object["custom_id"]
                save_response_to_temp_file(prompt_id, json_object, language, model, results_directory)

        all_responses = load_all_responses_from_file(language, model, results_directory, RESULTS_FILENAME)
        all_temp_data = read_and_delete_temp_files(language, model, results_directory)
        if len(all_temp_data):
            all_responses = add_temp_data(all_responses, all_temp_data, language, model)
            save_all_responses_to_json_file(all_responses, language, model, results_directory, RESULTS_FILENAME)

        index = 0
        while True:
            index += 1
            if not os.path.exists(os.path.join(results_directory, batch_file_filename.replace(".jsonl", "_DONE.%d.jsonl" % index))):
                os.rename(os.path.join(results_directory, batch_file_filename),
                          os.path.join(results_directory, batch_file_filename.replace(".jsonl", "_DONE.%d.jsonl" % index)))
                os.rename(os.path.join(results_directory, batch_outfile_filename),
                          os.path.join(results_directory, batch_outfile_filename.replace(".jsonl", "_DONE.%d.jsonl" % index)))
                break

        if not os.path.exists(lock_file):
            debug_msg("Warning: Batch file %s does not exist!" % (batch_file_filename), DebugLevel.WARNING)
        else:
            os.remove(lock_file)

        debug_msg("Batch %s successfully processed." % batch_file_filename, level=DebugLevel.LOG)
    return


def retrieve_gemini_results():
    client = genai.Client(api_key=GEMINI_API_KEY)
    batch_filename_list_avoid_duplicate_processing = set()

    for batch in client.batches.list(config={"page_size": 100}):
        if batch.display_name is None: continue
        batch_file_filename = batch.display_name
        if batch_file_filename in batch_filename_list_avoid_duplicate_processing:
            continue
        batch_filename_list_avoid_duplicate_processing.add(batch_file_filename)

        if batch.state.name != "JOB_STATE_SUCCEEDED":
            debug_msg("Batch: %s; state: %s" % (batch_file_filename, str(batch.state.name)), level=DebugLevel.LOG)
            if batch.state.name == "JOB_STATE_FAILED":
                debug_msg("  errors: %s" % (str(batch.error)), level=DebugLevel.LOG)
            continue

        try:
            _, _, language, model, repetition = batch_file_filename.split("__")
            repetition = int(repetition.split("_")[1].split(".")[0])
        except:
            continue

        results_directory = RESULTS_DIRECTORY.format(language=language)
        if not os.path.exists(os.path.join(results_directory, batch_file_filename)):
            continue

        lock_file = os.path.join(results_directory, f"_{repetition}_BATCH.LOCK")
        if not os.path.exists(lock_file):
            continue

        debug_msg("Batch: %s; state: %s" % (batch_file_filename, str(batch.state)), level=DebugLevel.LOG)
        if not batch.dest or not batch.dest.file_name:
            debug_msg("Not a batch from a file. Skip...")
            continue
        result_file_name = batch.dest.file_name
        batch_response = client.files.download(file=result_file_name)
        batch_outfile_filename = batch_file_filename.replace("_requests_", "_responses_")
        with open(os.path.join(results_directory, batch_outfile_filename), 'w', encoding="utf-8") as file:
            file.write(batch_response.decode("utf-8"))

        with open(os.path.join(results_directory, batch_outfile_filename), 'r') as file:
            for line in file:
                json_object = json.loads(line.strip())
                prompt_id = json_object["key"]
                save_response_to_temp_file(prompt_id, json_object, language, model, results_directory)

        client.batches.delete(name=batch.name)
        all_responses = load_all_responses_from_file(language, model, results_directory, RESULTS_FILENAME)
        all_temp_data = read_and_delete_temp_files(language, model, results_directory)
        if len(all_temp_data):
            all_responses = add_temp_data(all_responses, all_temp_data, language, model)
            save_all_responses_to_json_file(all_responses, language, model, results_directory, RESULTS_FILENAME)

        index = 0
        while True:
            index += 1
            if not os.path.exists(os.path.join(results_directory, batch_file_filename.replace(".jsonl", "_DONE.%d.jsonl" % index))):
                os.rename(os.path.join(results_directory, batch_file_filename),
                          os.path.join(results_directory, batch_file_filename.replace(".jsonl", "_DONE.%d.jsonl" % index)))
                os.rename(os.path.join(results_directory, batch_outfile_filename),
                          os.path.join(results_directory, batch_outfile_filename.replace(".jsonl", "_DONE.%d.jsonl" % index)))
                break

        if not os.path.exists(lock_file):
            debug_msg("Warning: Batch file %s does not exist!" % (batch_file_filename), DebugLevel.WARNING)
        else:
            os.remove(lock_file)

        debug_msg("Batch %s successfully processed." % batch_file_filename, level=DebugLevel.LOG)
    return


def main():
    if MODEL_FAMILY == "gpt":
        retrieve_gpt_results()
    elif MODEL_FAMILY == "gemini":
        retrieve_gemini_results()


if __name__ == "__main__":

    start_time = datetime.datetime.now()
    debug_msg("Started %s" % str(sys.argv[0]), DebugLevel.DEBUG)

    main()

    end_time = datetime.datetime.now()
    debug_msg("Finished %s\tDuration: %s" % (str(sys.argv[0]), get_duration_string(start_time, end_time)), DebugLevel.DEBUG)
