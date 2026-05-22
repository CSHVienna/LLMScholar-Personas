#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# export PYTHONPATH="$PYTHONPATH:../../libs" # <-- run this in terminal before executing the script (adjust path as needed)

import argparse

from prompt import generation as gen
from utils import constants as cons
from utils import ios

DEFAULT_OUTPUT_DIR = "../../data/context/"


def run(combination_id: int, language: str, output_dir: str) -> None:

    # Prepare output directory
    output_path = ios.path(output_dir) / language
    ios.validate_path(output_path)

    # read prompt data
    instructions = gen.read_instructions(
        output_path / "instructions.json"
    )  # roles, tasks, targets.
    locations = gen.read_locations(output_path / "locations.json")  # countries.
    inputs = gen.read_inputs(output_path / "input.json")  # k, fields, and subfields.
    combos = list(gen.combine_all(instructions, locations, inputs))

    # All combinations
    print(f"Total combinations ({language}): {len(combos)}")
    print(f"Try indexes from 0 to {len(combos)-1}\n")
    all_prompts = []
    for obj in combos:
        persona_context = obj["persona_context"]
        user_request = obj["user_request"]
        instructions, input = gen.create_prompt(
            persona_context, user_request, language=language
        )
        prompt = instructions + "\n\n" + input
        all_prompts.append(
            {"instructions": instructions, "input": input, "prompt": prompt}
        )

    fn = output_path / "all_prompts.txt"
    if not ios.path_exists(fn):
        ios.write_list_to_file([obj["prompt"] for obj in all_prompts], fn)

    if combination_id >= 0 and combination_id < len(all_prompts):
        selected_prompt = all_prompts[combination_id]
        print(f"\n##### Selected combination ID: {combination_id}\n")
        print("##### Instructions ##### \n")
        print(selected_prompt["instructions"])
        print("\n##### Input ##### \n")
        print(selected_prompt["input"])
        print("\n##### Full Prompt ##### \n")
        print(selected_prompt["prompt"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate JSON values to a target language."
    )

    parser.add_argument(
        "-c", "--combination_id", type=int, default=0, help="Combination ID."
    )

    parser.add_argument(
        "-l",
        "--language",
        type=str,
        choices=cons.LANGUAGES,
        default=cons.LANG_EN,
        help="Target language code/name.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    return args


if __name__ == "__main__":
    # Read arguments
    args = build_parser()

    # Run
    run(args.combination_id, args.language, args.output_dir)
