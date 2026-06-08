import argparse
import glob
import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
from logger import debug_msg, DebugLevel
from openai import OpenAI
import os
import sys
from utils import get_duration_string
from gpt_gemini_01_create_batch_files import MODEL_LIST

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-mf', '--model_family', choices=["gpt", "gemini"], default="gpt",
                    help='Run a specific model family.')
parser.add_argument('-r', '--repetitions', type=int, default=10,
                    help='Number of repetitions.')
parser.add_argument('-l', '--language', choices=["english", "german", "spanish"], default="english",
                    help='Specify the language used for the prompts.')
args = parser.parse_args()

MODEL_FAMILY = args.model_family
NR_REPETITIONS = args.repetitions
LANGUAGE = args.language
DIRECTORY = os.path.dirname(os.path.realpath(__file__))
RESULTS_DIRECTORY = os.path.join(DIRECTORY, f"results_{MODEL_FAMILY}_{LANGUAGE}")
RESULTS_FILENAME = f"{MODEL_FAMILY}_{{language}}_{{model}}.json"
BATCH_FILE_FILENAME = f"_requests_batch_file__{MODEL_FAMILY}__{{language}}__{{model}}__repetition_{{repetition:02d}}.jsonl"


def run_repetition(language,
                   model,
                   repetition,
                   results_directory=RESULTS_DIRECTORY):
    batch_file_filename = BATCH_FILE_FILENAME.format(
        language=language,
        model=model,
        repetition=repetition,
    )

    batch_file_path_list = sorted(glob.glob(os.path.join(results_directory, batch_file_filename + "*")))
    if len(batch_file_path_list) == 0:
        return

    os.rename(batch_file_path_list[0], os.path.join(results_directory, batch_file_filename))
    lock_file = os.path.join(results_directory, f"_{repetition}_BATCH.LOCK")
    if os.path.exists(lock_file):
        debug_msg("Batch file %s and lock file found. Batch already created!" % (batch_file_filename), DebugLevel.LOG)
        return
    with open(lock_file, 'w') as fp:
        pass

    debug_msg("Batch file %s found. Going to create batch." % (batch_file_filename), DebugLevel.LOG)

    if MODEL_FAMILY == "gpt":
        client = OpenAI(api_key=OPENAI_API_KEY)

        batch_input_file = client.files.create(
            file=open(os.path.join(results_directory, batch_file_filename), "rb"),
            purpose="batch"
        )
        batch_response = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": batch_file_filename
            }
        )

    elif MODEL_FAMILY == "gemini":
        client = genai.Client(api_key=GEMINI_API_KEY)
        uploaded_file = client.files.upload(
            file=os.path.join(results_directory, batch_file_filename),
            config=types.UploadFileConfig(display_name=batch_file_filename, mime_type='application/json')
        )

        file_batch_job = client.batches.create(
            model=model,
            src=uploaded_file.name,
            config={
                'display_name': batch_file_filename,
            },
        )

    debug_msg("Batch created.", DebugLevel.LOG)
    return


def run_repetitions(language,
                    model,
                    nr_repetitions=NR_REPETITIONS,
                    results_directory=RESULTS_DIRECTORY):
    for repetition in range(1, nr_repetitions+1):
        run_repetition(language,
                       model,
                       repetition,
                       results_directory)
    return


def main():
    results_directory = RESULTS_DIRECTORY
    for model in MODEL_LIST:
        run_repetitions(
            LANGUAGE,
            model,
            nr_repetitions=NR_REPETITIONS,
            results_directory=results_directory
        )


if __name__ == "__main__":

    start_time = datetime.datetime.now()
    debug_msg("Started %s" % str(sys.argv[0]), DebugLevel.DEBUG)

    main()

    end_time = datetime.datetime.now()
    debug_msg("Finished %s\tDuration: %s" % (str(sys.argv[0]), get_duration_string(start_time, end_time)), DebugLevel.DEBUG)
