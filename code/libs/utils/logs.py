"""
Logging for batch scripts: every run writes to the console and to
``<log_dir>/<timestamp>_<script>_<params>.log``, e.g.
``2026-11-04-23:10_responses_summary_gpt-4_1-nano-2025-04-14.log``.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils import constants as cons

LOGGER_NAME = 'llmscholar'
LOG_FORMAT = '[%(asctime)s] %(levelname)s %(message)s'
LOG_DATE_FORMAT = '%H:%M:%S'


def add_log_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--log_dir', default=cons.DEFAULT_LOGS_DIR, help='Folder for the log file of this run')


def setup_logging(script: str, args: argparse.Namespace, params: Optional[str] = None) -> Path:
    """Log to the console and to a new file named after the script and its parameters. Returns the file path.

    ``params`` defaults to :func:`params_tag` of ``args``.
    """
    timestamp = datetime.now().strftime(cons.LOG_TIMESTAMP_FORMAT)
    log_file = Path(args.log_dir) / f"{timestamp}_{script}_{params or params_tag(args)}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = get_logger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding='utf-8')):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False

    logger.info(f"Log file: {log_file}")
    logger.info(f"Command: {' '.join(sys.argv)}")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    return log_file


def params_tag(args: argparse.Namespace) -> str:
    """Short, file-name-safe summary of the run parameters: models, languages, limits."""
    parts = []
    models = getattr(args, 'models', None)
    parts.append('all' if not models else models[0] if len(models) == 1 else f"{len(models)}models")

    languages = getattr(args, 'languages', None)
    if languages and set(languages) != set(cons.LANGUAGES):
        parts.append('-'.join(languages))

    limit = getattr(args, 'limit_prompts', None)
    if limit is not None:
        parts.append(f"limit{limit}")
    return '_'.join(parts).replace('/', '-')


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)
