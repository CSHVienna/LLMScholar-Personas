#!/usr/bin/env python3
"""
Collect (cloud), step 2 of 3: upload the request files of batch_collect_cloud_prepare.py and start the
batch jobs (OpenAI / Gemini Batch API; results within 24h).

Every open requests file without a ``.submitted.json`` is submitted once; the job id is written to
``<batch>.submitted.json``. Running it again only submits what is new. Models of other APIs are skipped.
API keys: environment variables OPENAI_API_KEY / GEMINI_API_KEY (see README.md, Setup).

Usage (from any folder):
    python batch_collect_cloud_submit.py --models gpt-4_1-nano-2025-04-14
    python batch_collect_cloud_submit.py --models gemini-2_5-flash --languages english --responses_dir tmp/smoke/responses
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
from datetime import datetime, timezone

from llm import cloud
from llm import models as lm
from llm.store import ResponseStore
from utils import batch
from utils import constants as cons
from utils import ios
from utils.config import read_api_key


def run_model(slug: str, args: argparse.Namespace) -> str:
    model = lm.spec(lm.load_metadata(args.models_metadata_file), slug)
    if model.api not in cons.CLOUD_APIS:
        return f"[{slug}] skipped: api={model.api} (see batch_collect_local.py)"
    module, client, submitted, kept = cloud.provider(model.api), None, [], 0
    for language in args.languages:
        for files in cloud.open_batches(ResponseStore(args.responses_dir, model, language)):
            if files.submitted.exists():
                kept += 1
                continue
            client = client or module.make_client(read_api_key(model.api))
            info = module.submit_batch(client, files.requests, model.api_name, description=files.name)
            ios.save_json({**info, 'api': model.api, 'api_name': model.api_name, 'language': language,
                           'run': files.run, 'submitted_at': datetime.now(timezone.utc).isoformat()}, files.submitted)
            ios.printf(f"[{slug}] submitted {files.name}: {info['job_id']}")
            submitted.append(files.name)
    return f"[{slug}] submitted: {len(submitted)} | already submitted: {kept}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    batch.add_collect_arguments(parser, prompts=False)
    args = parser.parse_args()
    batch.run_per_model('collect_cloud_submit', run_model, args)


if __name__ == '__main__':
    main()
