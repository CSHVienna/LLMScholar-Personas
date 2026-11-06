#!/usr/bin/env python3
"""
Collect (cloud), step 1 of 3: write the batch request files of the GPT / Gemini models.

For every model x language, one requests file per run 1..--repetitions with the prompts that still need
that run (resumable: prompts already answered, or waiting in an open batch, are not asked again).
Also adds a record per prompt to the response file (results_<api>_<language>/<api>_<language>_<model>.json).
Nothing is sent: batch_collect_cloud_submit.py does that. Models of other APIs are skipped.

Writes results_<api>_<language>/_batches/<api>__<language>__<model>__run<NN>.requests.jsonl
(default responses folder: data/responses/).

Usage (from any folder):
    python batch_collect_cloud_prepare.py --models gpt-4_1-nano-2025-04-14
    python batch_collect_cloud_prepare.py --models gemini-2_5-flash --languages english --repetitions 1 \
        --limit_prompts 3 --responses_dir tmp/smoke/responses          # smoke test
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse

from llm import cloud
from llm import models as lm
from llm.store import ResponseStore, load_prompts
from utils import batch
from utils import constants as cons


def run_model(slug: str, args: argparse.Namespace) -> str:
    model = lm.spec(lm.load_metadata(args.models_metadata_file), slug)
    if model.api not in cons.CLOUD_APIS:
        return f"[{slug}] skipped: api={model.api} (see batch_collect_local.py)"
    report = []
    for language in args.languages:
        prompts = load_prompts(args.context_dir, language, args.limit_prompts)
        store = ResponseStore(args.responses_dir, model, language).load()
        written = cloud.prepare_batches(store, prompts, args.repetitions, overwrite=args.overwrite)
        report.append(f"{language}: " + (', '.join(f"run{r:02d}={n}" for r, n in written.items()) or 'nothing new'))
    return f"[{slug}] requests written: " + ' | '.join(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    batch.add_collect_arguments(parser)
    args = parser.parse_args()
    batch.run_per_model('collect_cloud_prepare', run_model, args)


if __name__ == '__main__':
    main()
