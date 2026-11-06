#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inputs, step 2: all prompts of each language, from the parameters of batch_input_params.py.

Writes per language, in <output_dir>/<language>/ (default data/context/):
    all_params_and_prompts.json   [{id, parameters: {persona_context, user_request}, prompt}]  (read by batch_collect_*)
    all_prompts.txt               the prompts only, readable

``id`` is the position of the combination (0, 1, ...) and is the prompt id of the response files.
Existing files are kept unless --overwrite; a kept JSON that differs from the parameters is reported.

Usage (from any folder):
    python batch_input_prompts.py                         # all languages
    python batch_input_prompts.py -l german -c 5          # one language, show combination 5
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse

from prompt import generation as gen
from utils import constants as cons
from utils import ios


def run(language: str, output_dir: str, combination_id: int | None, overwrite: bool) -> None:
    output_path = ios.path(output_dir) / language
    records = gen.read_prompt_records(output_path, language)
    print(f"Total combinations ({language}): {len(records)} (ids 0 to {len(records) - 1})")

    json_file = output_path / cons.PROMPTS_FILE
    if overwrite or not json_file.exists():
        ios.save_json(records, json_file)
        print(f"Wrote {json_file}")
    elif ios.load_json(json_file) != records:
        print(f"WARNING: {json_file} differs from the parameter files (kept; use --overwrite to replace it). "
              f"Prompt ids of new responses would not match the old ones.")
    else:
        print(f"Kept {json_file} (up to date)")

    text_file = output_path / cons.PROMPTS_TEXT_FILE
    if overwrite or not text_file.exists():
        ios.write_list_to_file([record['prompt'] for record in records], text_file)
        print(f"Wrote {text_file}")

    if combination_id is not None and 0 <= combination_id < len(records):
        record = records[combination_id]
        print(f"\n##### Selected combination ID: {combination_id}\n")
        print("##### Parameters ##### \n")
        print(record['parameters'])
        print("\n##### Full Prompt ##### \n")
        print(record['prompt'])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate all prompts of each language.")
    parser.add_argument("-l", "--languages", nargs='+', choices=cons.LANGUAGES, default=cons.LANGUAGES)
    parser.add_argument("-o", "--output-dir", default=cons.DEFAULT_CONTEXT_DIR, help="Folder with <language>/ parameters.")
    parser.add_argument("-c", "--combination_id", type=int, default=None, help="Show this combination.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing prompt files.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    for language in args.languages:
        run(language, args.output_dir, args.combination_id, args.overwrite)
