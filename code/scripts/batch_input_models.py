#!/usr/bin/env python3
"""
Inputs, step 3: the model list (one slug per line) from data/models/metadata.json.

The metadata is the source of truth: each key is a model slug, with ``api`` (gpt | gemini | ollama) and
``api_name`` (the name the API expects; the key must be ``model_slug(api_name)``). The script checks this
first and stops on any problem.

An existing list with the same models is kept as it is (its order too); a different one is only replaced
with --overwrite. Without --apis the list has every model (the audit reads it); with --apis only those,
e.g. a list for the Ollama runs.

Usage (from any folder):
    python batch_input_models.py                                      # check / write data/models/models.txt
    python batch_input_models.py --apis ollama --output tmp/models_ollama.txt
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
from pathlib import Path

from llm import models as lm
from utils import batch
from utils import constants as cons
from utils import ios


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the model list from the models metadata.")
    parser.add_argument('--models_metadata_file', default=cons.DEFAULT_MODELS_METADATA_FILE)
    parser.add_argument('--output', default=cons.DEFAULT_MODELS_FILE, help='Model list to write')
    parser.add_argument('--apis', nargs='+', choices=cons.LLM_SOURCES, default=None, help='Only models of these APIs')
    parser.add_argument('--overwrite', action='store_true', help='Replace a list with other models')
    args = parser.parse_args()

    metadata = lm.load_metadata(args.models_metadata_file)
    problems = lm.validate_metadata(metadata)
    if problems:
        raise SystemExit("Fix the models metadata first:\n  " + "\n  ".join(problems))

    slugs = lm.slugs(metadata, args.apis)
    output = Path(args.output)
    counts = {api: sum(metadata[s][cons.META_API] == api for s in slugs) for api in cons.LLM_SOURCES}
    ios.printf(f"{len(slugs)} models in {args.models_metadata_file}: {counts}")

    if output.exists():
        current = batch.read_models(output)
        if sorted(current) == slugs:
            ios.printf(f"Kept {output} (same models)")
            return
        added, removed = sorted(set(slugs) - set(current)), sorted(set(current) - set(slugs))
        ios.printf(f"{output} differs: added {added}, removed {removed}")
        if not args.overwrite:
            raise SystemExit("Not replaced (use --overwrite)")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(slugs) + '\n', encoding='utf-8')
    ios.printf(f"Wrote {output}")


if __name__ == '__main__':
    main()
