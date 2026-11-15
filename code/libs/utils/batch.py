"""
Shared plumbing for the per-model batch scripts: common CLI arguments, logging,
and the per-model job loop.

Parallelism is left to GNU parallel (one process per model, ``--models {}``);
within a process the models are processed one after the other, and a failing
model never stops the others.
"""

import argparse
import json
import os
import zlib
import traceback
from pathlib import Path
from typing import Callable, Dict, List

from utils import constants as cons
from utils import ios
from utils import logs

PARAMS_FILE = '_params.json'      # run parameters of an output folder (check_params)

Job = Callable[[str, argparse.Namespace], str]


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by every script that processes models one by one."""
    parser.add_argument('--models_file', default=cons.DEFAULT_MODELS_FILE, help='Text file with one model slug per line')
    parser.add_argument('--models', nargs='+', default=None, help='Subset of models (default: all in --models_file)')
    parser.add_argument('--languages', nargs='+', default=cons.LANGUAGES, choices=cons.LANGUAGES)
    parser.add_argument('--overwrite', action='store_true', help='Recompute existing output files')
    logs.add_log_arguments(parser)


def add_raw_response_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments of scripts that read the raw response files."""
    parser.add_argument('--responses_dir', default=cons.DEFAULT_RESPONSES_DIR, help='Folder with results_<source>_<language>/')
    parser.add_argument('--context_dir', default=cons.DEFAULT_CONTEXT_DIR, help='Folder with <language>/*.json prompt parameters')
    parser.add_argument('--collection_dates_file', default=cons.DEFAULT_COLLECTION_DATES_FILE,
                        help='Dates per model/language for sources without timestamps (Gemini)')
    parser.add_argument('--limit_prompts', type=int, default=None, help='Only the first N prompts per file (smoke tests)')


def add_collect_arguments(parser: argparse.ArgumentParser, prompts: bool = True) -> None:
    """Arguments of the scripts that collect the raw responses (batch_collect_*.py)."""
    add_model_arguments(parser)
    parser.add_argument('--models_metadata_file', default=cons.DEFAULT_MODELS_METADATA_FILE,
                        help='Models metadata with api and api_name per model slug')
    parser.add_argument('--responses_dir', default=cons.DEFAULT_RESPONSES_DIR, help='Folder with results_<api>_<language>/')
    if prompts:
        parser.add_argument('--context_dir', default=cons.DEFAULT_CONTEXT_DIR,
                            help='Folder with <language>/all_params_and_prompts.json')
        parser.add_argument('--repetitions', type=int, default=cons.DEFAULT_REPETITIONS, help='Runs per prompt')
        parser.add_argument('--limit_prompts', type=int, default=None, help='Only the first N prompts (smoke tests)')


def model_file(folder: Path | str, model: str, extension: str = 'parquet') -> Path:
    """Per-model file: <folder>/<model>.<extension>"""
    return Path(folder) / f"{model}.{extension}"


def read_models(models_file: Path | str) -> List[str]:
    """Read the model list (one model slug per line, blank lines ignored)."""
    return [line.strip() for line in ios.read_text(models_file).splitlines() if line.strip()]


def run_per_model(script: str, job: Job, args: argparse.Namespace) -> None:
    """Set up logging, run ``job(model, args)`` for every model and log one report line per model.

    The process exits with code 1 if any model failed (so GNU parallel's ``--joblog`` records it).
    """
    logs.setup_logging(script, args)
    models = args.models or read_models(args.models_file)
    ios.printf(f"Models: {len(models)} | languages: {args.languages}")
    run_jobs(models, job, args)


def run_jobs(items: List, job: Callable, args: argparse.Namespace) -> None:
    """Run ``job(item, args)`` for every item (logging must be set up); a failing item never stops the others.

    Logs one report line per item; exits with code 1 if any item failed (for GNU parallel's ``--joblog``).
    """
    start = ios.datetime.now()
    failed = []
    for item in items:
        try:
            ios.printf(job(item, args))
        except Exception:
            failed.append(item)
            logs.get_logger().error(f"FAILED [{item}]\n{traceback.format_exc()}")

    ios.printf(f"Done in {ios.datetime.now() - start} | failed: {failed or 'none'}")
    if failed:
        raise SystemExit(1)


def check_params(output_dir: Path | str, params: Dict, overwrite: bool = False) -> None:
    """Record the run parameters once (atomically); refuse to mix chunks written with different parameters.
    overwrite: replace the stored parameters (the caller recomputes every output)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / PARAMS_FILE
    if overwrite:   # atomic replace; parallel jobs write the same content
        tmp = output_dir / f".{PARAMS_FILE}.{os.getpid()}"
        tmp.write_text(json.dumps(params, indent=1))
        os.replace(tmp, path)
        return
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)   # first process wins (GNU parallel safe)
        with os.fdopen(fd, 'w') as f:
            json.dump(params, f, indent=1)
        return
    except FileExistsError:
        pass
    stored = json.loads(path.read_text() or '{}')
    if stored and stored != params:
        raise ValueError(f"{path} was written with other parameters: {stored} != {params}. "
                         f"Use another --output_dir or --overwrite (recomputes every output).")


# --- per-model metric scripts (batch_metrics, batch_social_metrics, batch_flows) ----------------------------------
def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    """The three per-model inputs: response metadata (B.1-B.2), recommendation metadata (B.3), author matching (B.6b)."""
    parser.add_argument('--responses_metadata_dir', default=cons.DEFAULT_RESPONSES_METADATA_DIR, help='Output of B.1-B.2')
    parser.add_argument('--recommendations_metadata_dir', default=cons.DEFAULT_RECOMMENDATIONS_METADATA_DIR,
                        help='Output of B.3')
    parser.add_argument('--matching_dir', default=cons.DEFAULT_MATCH_AUTHORS_DIR, help='Output of B.6b (a variant folder)')


def load_model_inputs(model: str, args: argparse.Namespace, columns: Dict[str, List[str]],
                      filter_recs: bool = False) -> tuple:
    """(responses, recommendations, matches) of one model, with the given columns (keys 'responses',
    'recommendations', 'matching'); only the languages of args.languages (responses; recommendations too if
    filter_recs). Raises if an input file is missing."""
    inputs = {'responses': model_file(args.responses_metadata_dir, model),
              'recommendations': model_file(args.recommendations_metadata_dir, model),
              'matching': model_file(args.matching_dir, model)}
    missing = [f"{name}: {path}" for name, path in inputs.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"[{model}] missing inputs: {missing}")
    responses, recs, matches = (ios.load_parquet(inputs[name], columns=columns[name])
                                for name in ('responses', 'recommendations', 'matching'))
    if args.languages != cons.LANGUAGES:
        responses = responses[responses[cons.VAR_LANGUAGE].isin(args.languages)]
        if filter_recs:
            recs = recs[recs[cons.VAR_LANGUAGE].isin(args.languages)]
    return responses, recs, matches


# Options the metrics runner passes to every per-model metric script (run_metrics_pipeline.sh -a / -x, e.g.
# "--keep_duplicates"); a script without one accepts and ignores it (hidden from --help)
RUNNER_FLAGS = ('--keep_duplicates', '--consistency_all_runs', '--order_sensitive_keys', '--include_unknown')
RUNNER_OPTIONS = {'--parity_target': {}, '--top_countries': {}, '--valid_response_tags': {'nargs': '*'},
                  '--valid_rec_tags': {'nargs': '*'}}


def accept_runner_options(parser: argparse.ArgumentParser) -> None:
    """Add the runner's options this parser does not have, as hidden no-ops. Call after the script's own options."""
    known = parser._option_string_actions
    for flag in RUNNER_FLAGS:
        if flag not in known:
            parser.add_argument(flag, action='store_true', help=argparse.SUPPRESS)
    for option, kwargs in RUNNER_OPTIONS.items():
        if option not in known:
            parser.add_argument(option, default=None, help=argparse.SUPPRESS, **kwargs)


# --- shards -------------------------------------------------------------------------------------------------------
def crc32_shard(text: str, n_shards: int) -> int:
    """Deterministic shard of a string: crc32 % n_shards (the same in every process and Python version)."""
    return zlib.crc32(text.encode('utf-8')) % n_shards


def shard_file(folder: Path | str, shard: int) -> Path:
    """<folder>/shard-<ss>.parquet."""
    return Path(folder) / f"shard-{shard:02d}.parquet"
