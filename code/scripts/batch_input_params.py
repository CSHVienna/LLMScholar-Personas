#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inputs, step 1: the prompt parameters of a language (instructions, locations, inputs).

English is written from ``utils/constants.py``; other languages are translated with OpenAI
(key: environment variable OPENAI_API_KEY).

Writes <output_dir>/<language>/{instructions,locations,input}.json (default data/context/).

Usage (from any folder):
    python batch_input_params.py -l english
    python batch_input_params.py -l german
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
import json

from llm import openai as llm_openai
from prompt import generation as gen
from utils import constants as cons
from utils import ios
from utils.config import read_api_key


def translate(obj, lang, api_key):
    instructions, input = gen.create_translate_param_prompt(obj, lang)
    raw = llm_openai.prompt_gpt(api_key, instructions, input)

    try:
        return raw, json.loads(raw.output_text)
    except json.JSONDecodeError as ex:
        if 'Extra data' in ex.msg:
            return raw, json.loads(raw.output_text[:-1])
        raise ValueError("LLM did not return valid JSON.")


def run(language: str, output_dir: str) -> None:
    path = ios.path(output_dir) / language
    path.mkdir(parents=True, exist_ok=True)
    api_key = read_api_key('openai') if language != cons.LANG_EN else None

    for fname, data in cons.FILES.items():
        if language != cons.LANG_EN:
            raw, data_translated = translate(data, language, api_key)
            data = data_translated['data']
        with open(path / fname, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    print('Summary:\n', sorted(path.glob("*.json")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write the prompt parameters of a language (translated if not English).")
    parser.add_argument("-l", "--language", choices=cons.LANGUAGES, default=cons.LANG_EN, help="Target language.")
    parser.add_argument("-o", "--output-dir", default=cons.DEFAULT_CONTEXT_DIR, help="Output directory.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    run(args.language, args.output_dir)
