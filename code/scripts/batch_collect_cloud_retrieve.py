#!/usr/bin/env python3
"""
Collect (cloud), step 3 of 3: check the submitted batch jobs and merge the finished ones into the
response files (results_<api>_<language>/<api>_<language>_<model>.json).

Per submitted batch: running -> reported and kept; done -> output downloaded to
``<batch>.responses.jsonl``, answers appended to the response file, batch files moved to ``_batches/done/``
(Gemini jobs are then deleted); failed / cancelled / expired without output -> moved to ``_batches/failed/``
(the next prepare queues those prompts again). Run it until nothing is open, then prepare again in case
some prompts are still missing (failed requests are not in the output). Models of other APIs are skipped.

Usage (from any folder):
    python batch_collect_cloud_retrieve.py --models gpt-4_1-nano-2025-04-14
    python batch_collect_cloud_retrieve.py --models gemini-2_5-flash --languages english --responses_dir tmp/smoke/responses
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
from collections import Counter

from llm import cloud
from llm import models as lm
from llm.store import ResponseStore
from utils import batch
from utils import constants as cons
from utils import ios
from utils import logs
from utils.config import read_api_key


def run_model(slug: str, args: argparse.Namespace) -> str:
    model = lm.spec(lm.load_metadata(args.models_metadata_file), slug)
    if model.api not in cons.CLOUD_APIS:
        return f"[{slug}] skipped: api={model.api} (see batch_collect_local.py)"
    module, client, counts, merged = cloud.provider(model.api), None, Counter(), 0
    for language in args.languages:
        store = ResponseStore(args.responses_dir, model, language)
        for files in cloud.open_batches(store):
            if not files.submitted.exists():
                counts['not submitted'] += 1
                continue
            client = client or module.make_client(read_api_key(model.api))
            job_id = cloud.read_submitted(files)['job_id']
            state, details = module.batch_status(client, job_id)
            ios.printf(f"[{slug}] {files.name}: {state} ({details})")
            counts[state] += 1
            if state == 'failed':
                cloud.archive(files, cons.BATCHES_FAILED_DIR)
            elif state == 'done':
                n = cloud.merge_output(store, files, module.download_batch(client, job_id))
                merged += n
                ios.printf(f"[{slug}] {files.name}: {n} answers merged into {store.path}")
                try:
                    module.finish_batch(client, job_id)
                except Exception as error:      # the answers are safe; only the clean-up failed
                    logs.get_logger().warning(f"[{slug}] could not delete job {job_id}: {error!r}")
    return f"[{slug}] batches: {dict(counts) or 'none open'} | answers merged: {merged}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    batch.add_collect_arguments(parser, prompts=False)
    args = parser.parse_args()
    batch.run_per_model('collect_cloud_retrieve', run_model, args)


if __name__ == '__main__':
    main()
