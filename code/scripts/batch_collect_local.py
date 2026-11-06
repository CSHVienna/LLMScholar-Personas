#!/usr/bin/env python3
"""
Collect (local): send the prompts to Ollama servers and store the answers in the response files
(results_ollama_<language>/ollama_<language>_<model>.json).

Servers: --nr_servers servers on ports --first_port, --first_port + 1, ... of --host (started beforehand,
e.g. ``OLLAMA_HOST=127.0.0.1:11435 ollama serve``). Missing models are pulled first (--pull_only: only that).
Per model x language and run 1..--repetitions, the prompts without that run are sent by --max_processes
workers to the least-loaded alive server; each answer is saved at once (temp file, merged into the response
file), so the script can be stopped and started again. Prompts that fail are retried up to --max_attempts
times per run (waiting --retry_wait seconds); a model with prompts still missing is reported as FAILED.
Models of other APIs are skipped. One model after the other; for several models at once use GNU parallel.

Usage (from any folder):
    python batch_collect_local.py --models gemma3-4b-it-q4_K_M
    python batch_collect_local.py --models gemma3-4b-it-q4_K_M --nr_servers 4 --max_processes 8
    python batch_collect_local.py --models gemma3-4b-it-q4_K_M --languages english --repetitions 1 \
        --limit_prompts 3 --responses_dir tmp/smoke/responses          # smoke test
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
import time

from llm import models as lm
from llm import ollama as ol
from llm.store import ResponseStore, load_prompts
from utils import batch
from utils import constants as cons
from utils import ios
from utils import logs


def run_language(model: lm.ModelSpec, language: str, args: argparse.Namespace, settings: dict) -> str:
    prompts = load_prompts(args.context_dir, language, args.limit_prompts)
    store = ResponseStore(args.responses_dir, model, language).load()
    store.add_prompts(prompts)
    recovered = store.merge_temp()                    # answers of an interrupted run
    store.save()
    if recovered:
        ios.printf(f"[{model.slug}|{language}] {recovered} answers recovered from {store.temp_dir}")

    sent = 0
    for run in range(1, args.repetitions + 1):
        for attempt in range(1, args.max_attempts + 1):
            todo = store.missing(prompts, run)
            if not todo:
                break
            if attempt > 1:
                ios.printf(f"[{model.slug}|{language}] run {run}: {len(todo)} prompts left, retry {attempt} "
                           f"in {args.retry_wait}s")
                time.sleep(args.retry_wait)
            ios.printf(f"[{model.slug}|{language}] run {run}/{args.repetitions}: {len(todo)} prompts")
            failed = ol.run_pool(todo, {**settings, 'language': language}, args.nr_servers, args.max_processes)
            sent += store.merge_temp()
            if failed:
                ios.printf(f"[{model.slug}|{language}] run {run}: {len(failed)} failed, e.g. "
                           f"{next(iter(failed.items()))}")
        left = store.missing(prompts, run)
        if left:
            raise RuntimeError(f"[{model.slug}|{language}] run {run}: {len(left)} prompts still without an answer "
                               f"after {args.max_attempts} attempts (e.g. ids {[p['id'] for p in left[:5]]})")
    return f"{language}: {sent} answers"


def run_model(slug: str, args: argparse.Namespace) -> str:
    model = lm.spec(lm.load_metadata(args.models_metadata_file), slug)
    if model.api not in cons.LOCAL_APIS:
        return f"[{slug}] skipped: api={model.api} (see batch_collect_cloud_*.py)"

    ports = ol.alive_ports(args.host, args.first_port, args.nr_servers)
    if not ports:
        raise RuntimeError(f"No Ollama server on {args.host}:{args.first_port}..{args.first_port + args.nr_servers - 1}")
    ios.printf(f"[{slug}] servers alive: {ports}")
    pulled = ol.pull_missing(args.host, ports[0], [model.api_name])
    if args.pull_only:
        return f"[{slug}] pull only: {'pulled' if pulled else 'already there'}"

    settings = {'responses_dir': args.responses_dir, 'model': model, 'host': args.host, 'first_port': args.first_port,
                'timeout': args.timeout, 'log_file': logs.current_log_file(),
                'options': ol.chat_options(args.max_output_tokens, args.num_thread, args.context_window)}
    report = [run_language(model, language, args, settings) for language in args.languages]
    return f"[{slug}] " + ' | '.join(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    batch.add_collect_arguments(parser)
    parser.add_argument('--host', default=cons.DEFAULT_OLLAMA_HOST)
    parser.add_argument('--first_port', type=int, default=cons.DEFAULT_OLLAMA_FIRST_PORT, help='Port of the first server')
    parser.add_argument('--nr_servers', type=int, default=1, help='Servers on consecutive ports')
    parser.add_argument('--max_processes', type=int, default=4, help='Worker processes sending prompts')
    parser.add_argument('--timeout', type=float, default=600, help='Seconds per request')
    parser.add_argument('--num_thread', type=int, default=None, help='Ollama num_thread (server default if not set)')
    parser.add_argument('--context_window', type=int, default=None, help='Ollama num_ctx (server default if not set)')
    parser.add_argument('--max_output_tokens', type=int, default=None, help='Ollama num_predict (default: unlimited)')
    parser.add_argument('--max_attempts', type=int, default=3, help='Rounds per run for prompts that failed')
    parser.add_argument('--retry_wait', type=int, default=60, help='Seconds between rounds')
    parser.add_argument('--pull_only', action='store_true', help='Only pull missing models')
    args = parser.parse_args()
    batch.run_per_model('collect_local', run_model, args)


if __name__ == '__main__':
    main()
