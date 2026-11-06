"""
Shared plumbing for the per-model batch scripts: common CLI arguments, logging,
and the per-model job loop.

Parallelism is left to GNU parallel (one process per model, ``--models {}``);
within a process the models are processed one after the other, and a failing
model never stops the others.
"""

import argparse
import traceback
from pathlib import Path
from typing import Callable, List

from utils import constants as cons
from utils import ios
from utils import logs

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
