#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# export PYTHONPATH="$PYTHONPATH:../../libs" # <-- run this in terminal before executing the script (adjust path as needed)

import argparse
import json

from llm import openai as llm_openai
from prompt import generation as gen
from utils import constants as cons
from utils import ios
from utils.config import load_config

DEFAULT_OUTPUT_DIR = "../../data/context/"


def translate(obj, lang, api_key):
    instructions, input = gen.create_translate_param_prompt(obj, lang)
    raw = llm_openai.prompt_gpt(api_key, instructions, input)

    try:
        return raw, json.loads(raw.output_text)
    except json.JSONDecodeError as ex:
        if (
            "Extra data" in ex.msg
            or ex.msg == "Extra data: line 1 column 104 (char 103)"
            or ex.msg == "Extra data: line 1 column 106 (char 105)"
        ):
            return raw, json.loads(raw.output_text[:-1])

        raise ValueError("LLM did not return valid JSON.")


def run(api_key, language: str, output_dir: str) -> None:
    # Prepare output directory
    output_path = ios.path(output_dir)
    ios.validate_path(output_path)

    for fname, data in cons.FILES.items():
        path = output_path / language
        ios.validate_path(path)
        with open(path / fname, "w", encoding="utf-8") as f:

            if language != cons.LANG_EN:
                raw, data_translated = translate(data, language, api_key)
                data = data_translated["data"]

            json.dump(data, f, ensure_ascii=False, indent=2)
    print("Summary:\n", list(path.glob("*.json")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate JSON values to a target language."
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

    # config
    cfg = load_config("../../../config.ini")
    api_key = ios.read_text(cfg["OPENAI_API_DIR"]).strip()

    # Run
    run(api_key, args.language, args.output_dir)
