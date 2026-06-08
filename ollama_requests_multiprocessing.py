import argparse
import datetime
import multiprocessing
import http.client
from itertools import repeat
import logger
from logger import debug_msg, DebugLevel
from ollama._types import ResponseError
from ollama import Client
import os
import random
import socket
import sys
import threading
from utils import (add_temp_data,
                   check_prompt_already_processed,
                   create_folder,
                   get_duration_string,
                   load_all_responses_from_file,
                   load_json,
                   read_and_delete_temp_files,
                   save_all_responses_to_json_file,
                   save_response_to_temp_file)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-pi', '--parallel_index', type=int,
                    help='Parallel (split model list): index of process.')
parser.add_argument('-pt', '--parallel_total', type=int,
                    help='Parallel (split model list): total amount of processes.')
parser.add_argument('-m', '--model',
                    help='Run only a specific model.')
parser.add_argument('-ms', '--model_size', default="small",
                    choices=["small", "large", "all"],
                    help='Run only models of a specific size.')
parser.add_argument('-r', '--repetitions', type=int, default=10,
                    help='Number of repetitions.')
parser.add_argument('-s', '--nr_servers', type=int, default=1,
                    help='Number of ollama servers to run.')
parser.add_argument('-mp', '--max_processes', type=int, default=4,
                    help='Maximum number of processes for parallelization.')
parser.add_argument('-nt', '--num_thread', type=int, default=None,
                    help='Number of threads for each ollama server.')
parser.add_argument('-fp', '--first_port', type=int, default=11434,
                    help='Port of the first server.')
parser.add_argument('-cw', '--context_window', type=int, default=None,
                    help='Specify the context window. Per default, ollama sets it to 2048.')
parser.add_argument('-mot', '--max_output_tokens', type=int, default=None,
                    help='Specify the max. number of output tokens (ollama parameter: num_predict). Per default, ollama sets it to -1 (infinite).')
parser.add_argument('-l', '--language', choices=["english", "german", "spanish"], default="english",
                    help='Specify the language used for the prompts.')
parser.add_argument('-ls', '--logger_suffix', type=str, default=None,
                    help='Optional suffix for the logger filename (no suffix if not provided).')
parser.add_argument('-mpo', '--model_pull_only', action="store_true",
                    help='Pulling models only without processing tasks (default: false).')
args = parser.parse_args()

if (args.parallel_index is not None and args.parallel_total is None) or (args.parallel_index is None and args.parallel_total is not None):
    parser.error("parallel_index and parallel_total must both be specified!")
if args.parallel_index is not None and args.parallel_total is not None:
    if args.parallel_index <= 0:
        parser.error("parallel_index must be positive!")
    if args.parallel_index > args.parallel_total:
        parser.error("parallel_index can't be greater than parallel_total!")

NR_SERVERS = args.nr_servers
MAX_PROCESSES = args.max_processes
NUM_THREAD = args.num_thread
FIRST_PORT = args.first_port
CONTEXT_WINDOW = args.context_window
MAX_OUTPUT_TOKENS = args.max_output_tokens
NR_REPETITIONS = args.repetitions
LANGUAGE = args.language
HOST = "localhost"
RETRY_INTERVAL = 60
MODEL_PULL_ONLY = args.model_pull_only

model_list_small = [
    "deepseek-r1:8b-0528-qwen3-q4_K_M",
    # "deepseek-r1:32b-qwen-distill-q4_K_M",
    # "deepseek-r1:70b-llama-distill-q4_K_M",
    # "dolphin3:8b-llama3.1-q4_K_M",
    # "dolphin-mixtral:8x7b-v2.7-q4_K_M",
    # "dolphin-phi:2.7b-v2.6-q4_K_M",
    # "falcon3:7b-instruct-q4_K_M",
    # "falcon3:10b-instruct-q4_K_M",
    # "gemma3:4b-it-q4_K_M",
    # "gemma3:27b-it-q4_K_M",
    # "gemma3n:e4b-it-q4_K_M",
    # "gpt-oss:20b",
    # "llama3.2:3b-instruct-q4_K_M",
    # "llama3.3:70b-instruct-q4_K_M",
    # "mistral:7b-instruct-v0.3-q4_K_M",
    # "mistral-nemo:12b-instruct-2407-q4_K_M",
    # "mistral-small:22b-instruct-2409-q4_K_M",
    # "mistral-small3.2:24b-instruct-2506-q4_K_M",
    # "mixtral:8x7b-instruct-v0.1-q4_K_M",
    # "olmo2:7b-1124-instruct-q4_K_M",
    # "olmo2:13b-1124-instruct-q4_K_M",
    # "phi4-mini:3.8b-q4_K_M",
    # "phi4:14b-q4_K_M",
    # "phi4-reasoning:14b-q4_K_M",
    # "qwen3:8b-q4_K_M",
    # "qwen3:32b-q4_K_M",
    # "qwq:32b-q4_K_M",
    # "smollm2:1.7b-instruct-q4_K_M",
    # "yi:9b-chat-v1.5-q4_K_M",
    # "yi:34b-chat-v1.5-q4_K_M",
]

model_list_large = [
    "dolphin-mixtral:8x22b-v2.9-q4_K_M",
    "gpt-oss:120b",
    "llama4:17b-maverick-128e-instruct-q4_K_M",
    "llama4:17b-scout-16e-instruct-q4_K_M",
    "mistral-large:123b-instruct-2411-q4_K_M",
    "mixtral:8x22b-instruct-v0.1-q4_K_M",
    "qwen3:235b-a22b-instruct-2507-q4_K_M",
]

if args.model_size == "small":
    MODEL_LIST = model_list_small
elif args.model_size == "large":
    MODEL_LIST = model_list_large
elif args.model_size == "all":
    MODEL_LIST = model_list_small + model_list_large

if args.model is not None:
    MODEL_LIST = [args.model]

logger_path = ((logger.ROTATE_LOGGER_PATH
               .replace(".log", "")
               .lower()
               .replace(".","_"))
               + f"_{LANGUAGE}.log")
if args.logger_suffix:
    name, ext = os.path.splitext(logger_path)
    logger_path = f"{name}_{args.logger_suffix}{ext}"
logger.ROTATE_LOGGER_PATH = logger_path
if args.parallel_index is not None and args.parallel_total is not None:
    logger.ROTATE_LOGGER_PATH = logger.ROTATE_LOGGER_PATH.replace(".log", "_" + str(args.parallel_index) + "_" + str(args.parallel_total) + ".log")

DIRECTORY = os.path.dirname(os.path.realpath(__file__))
RESULTS_DIRECTORY = os.path.join(DIRECTORY, f"results_ollama_{LANGUAGE}")
RESULTS_FILENAME = "ollama_{language}_{model}.json"


def is_port_alive(host, port, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError:
        return False

    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", "/api/version")
        response = conn.getresponse()
        conn.close()

        if 200 <= response.status < 300:
            return True
        else:
            return False
    except Exception as e:
        return False


def update_port_status(port_status, base_port, idx, status, source="watcher"):
    if port_status[idx] != status:
        debug_msg(f"[{source.upper()}] changing port {base_port + idx} to {status}", DebugLevel.LOG)
        port_status[idx] = status


def port_watcher(port_status, port_load, base_port, num_servers, lock, stop_evt, interval=5):
    debug_msg("Starting port watcher thread ...", DebugLevel.LOG)
    while not stop_evt.is_set():
        for i in range(num_servers):
            port = base_port + i
            alive = is_port_alive(HOST, port)
            with lock:
                update_port_status(port_status, base_port, i, alive, source="watcher")
        if random.random() < 0.2:
            debug_msg(f"[WATCHER] STATUS: {list(port_status)} | LOAD: {list(port_load)}",
                      DebugLevel.DEBUG)
            temp_status_dict = {base_port + i: v for i, v in enumerate(port_status)}
            true_ports = [p for p, v in temp_status_dict.items() if v]
            debug_msg(f"[WATCHER] ACTIVE PORTS ({len(true_ports)}/{num_servers} [{100.0 * len(true_ports) / num_servers:.2f}%]): {true_ports}", DebugLevel.DEBUG)
        stop_evt.wait(interval)


def send_ollama_request(client, model, prompt):
    messages = [{"role": "user", "content": prompt}]
    options = dict()
    if MAX_OUTPUT_TOKENS is not None:
        options["num_predict"] = MAX_OUTPUT_TOKENS
    if NUM_THREAD is not None:
        options["num_thread"] = NUM_THREAD
    if CONTEXT_WINDOW is not None:
        options["num_ctx"] = CONTEXT_WINDOW

    response = client.chat(
        model=model,
        messages=messages,
        format="json",
        options=options,
    )

    debug_msg("request:  " + str(messages), DebugLevel.DEBUG)
    response = dict(response)
    response["message"] = dict(response["message"])

    debug_msg("response: " + str(response).strip(), DebugLevel.DEBUG)
    return response


def init_pool(shared_port_status, shared_port_load, shared_lock, shared_next_index):
    global PORT_STATUS, PORT_LOAD, LOCK, NEXT_PORT_INDEX
    PORT_STATUS     = shared_port_status
    PORT_LOAD       = shared_port_load
    LOCK            = shared_lock
    NEXT_PORT_INDEX = shared_next_index


def seed_port_status(port_status, base_port, num_servers):
    any_alive = False
    for i in range(num_servers):
        port = base_port + i
        alive = is_port_alive(HOST, port)
        port_status[i] = alive
        any_alive = any_alive or alive
    return any_alive


def process_prompt(prompt_item, language, model, results_directory):
    prompt_content = prompt_item["prompt"].strip()
    with LOCK:
        alive_ports = [(i, load) for i, load in enumerate(PORT_LOAD) if PORT_STATUS[i]]
        if not alive_ports:
            raise RuntimeError("No alive ports available!")
        idx, _ = min(alive_ports, key=lambda x: x[1])
        PORT_LOAD[idx] += 1
    port = FIRST_PORT + idx

    debug_msg(f"Next prompt to process: {prompt_item['id']} (@ port {port})", DebugLevel.LOG)

    try:
        client = Client(host="http://" + HOST + ":" + str(port), timeout=600)
        response = send_ollama_request(
            client,
            model,
            prompt_content,
        )
        save_response_to_temp_file(prompt_item["id"], response, language, model, results_directory)
        debug_msg(f"Prompt {prompt_item['id']} processed (@ port {port}).", DebugLevel.LOG)

    except Exception as e:
        debug_msg(f"[WORKER {os.getpid()}] ERROR (@ port {port}, prompt_id: {prompt_item['id']}): {repr(e)}", DebugLevel.ERROR)
        with LOCK:
            update_port_status(PORT_STATUS, FIRST_PORT, idx, False, source="worker")
        raise

    finally:
        with LOCK:
            PORT_LOAD[idx] -= 1

    return


def run_repetition(language, model, all_params_and_prompts, repetition, all_responses, results_directory=RESULTS_DIRECTORY):
    total_nr_prompts = len(all_params_and_prompts)

    while True:
        all_temp_data = read_and_delete_temp_files(language, model, results_directory)
        if len(all_temp_data):
            all_responses = add_temp_data(all_responses, all_temp_data, language, model)
            save_all_responses_to_json_file(all_responses, language, model, results_directory, RESULTS_FILENAME)

        prompt_ids_unprocessed = set()
        prompt_items_unprocessed = []

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

        if len(prompt_ids_unprocessed) == 0:
            break

        debug_msg("%d prompts to process..." % total_nr_prompts, DebugLevel.LOG)
        if len(prompt_ids_unprocessed) < len(all_params_and_prompts):
            debug_msg("%d prompts already processed before." % (len(all_params_and_prompts) - len(prompt_ids_unprocessed)), DebugLevel.LOG)
            debug_msg("%d prompts left to process..." % len(prompt_ids_unprocessed), DebugLevel.LOG)

        save_all_responses_to_json_file(all_responses, language, model, results_directory, RESULTS_FILENAME)

        with multiprocessing.Manager() as manager:
            port_status = manager.list([False] * NR_SERVERS)
            port_load = manager.list([0] * NR_SERVERS)
            seed_port_status(port_status, FIRST_PORT, NR_SERVERS)
            stop_evt = threading.Event()
            mp_lock = multiprocessing.Lock()
            next_port = manager.Value("i", 0)
            pool = multiprocessing.Pool(
                initializer=init_pool,
                initargs=(port_status, port_load, mp_lock, next_port),
                processes=MAX_PROCESSES,
                maxtasksperchild=50,
            )
            watcher = threading.Thread(
                target=port_watcher,
                args=(port_status, port_load, FIRST_PORT, NR_SERVERS, mp_lock, stop_evt),
                daemon=False,
            )
            watcher.start()

            try:
                pool.starmap(
                    process_prompt,
                    zip(
                        prompt_items_unprocessed,
                        repeat(language),
                        repeat(model),
                        repeat(results_directory)
                    )
                )
            except Exception as e:
                debug_msg("Aborting run due to worker exception: " + repr(e), DebugLevel.ERROR)
                pool.terminate()
                pool.join()
            else:
                pool.close()
                pool.join()
            finally:
                stop_evt.set()
                watcher.join()

    debug_msg("Processing %d prompts done." % total_nr_prompts, DebugLevel.LOG)
    return


def run_repetitions(language, model, all_params_and_prompts, nr_repetitions=NR_REPETITIONS, results_directory=RESULTS_DIRECTORY):
    debug_msg("%d repetitions to process in %s..." % (nr_repetitions, language), DebugLevel.LOG)
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
    debug_msg("args: %s" % str(args), DebugLevel.LOG)
    model_list = MODEL_LIST
    debug_msg(f"Number of models to process: {len(model_list)}", DebugLevel.LOG)
    if args.parallel_index is not None and args.parallel_total is not None:
        n = args.parallel_total
        idx = args.parallel_index-1
        k, m = divmod(len(model_list), n)
        model_list = [model_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)][idx]
        debug_msg(f"Number of models to process in this process: {len(model_list)}", DebugLevel.LOG)
    debug_msg(f"Models: {str(model_list)}", DebugLevel.LOG)
    for i in range(NR_SERVERS):
        port = FIRST_PORT + i
        client = Client(host="http://" + HOST + ":" + str(port))
        found_server = True
        for model in model_list:
            try:
                client.show(model)
            except ResponseError as e:
                debug_msg("Model %s not in list. Pulling..." % str(model), level=DebugLevel.LOG)
                client.pull(model)
                debug_msg("Pulling model %s done." % str(model), level=DebugLevel.LOG)
            except ConnectionError as e:
                debug_msg(f"Could not connect to ollama server (port {port})", level=DebugLevel.DEBUG)
                found_server = False
        if found_server: break
    if not found_server: return
    if MODEL_PULL_ONLY:
        return
    all_params_and_prompts = load_json(f"data/context/{LANGUAGE}/all_params_and_prompts.json")
    results_directory = RESULTS_DIRECTORY
    create_folder(results_directory)
    for model in model_list:
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

    multiprocessing.set_start_method("spawn", force=True)

    main()

    end_time = datetime.datetime.now()
    debug_msg("Finished %s\tDuration: %s" % (str(sys.argv[0]), get_duration_string(start_time, end_time)), DebugLevel.DEBUG)
