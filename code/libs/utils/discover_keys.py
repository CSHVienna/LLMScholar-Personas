"""
Scans all LLM response JSON files and collects the top-level keys used when
the model returns a dict instead of a bare list.  These keys are the "wrapper"
keys that batch_parse_results.py needs in _WRAPPER_KEYS.

Usage:
  python discover_keys.py --results_dir ../../results [--model gpt-4.1-2025-04-14] [--language english]
"""

import argparse
import ast
import re

try:
    from prompt import constants as cons
    from utils import ios
    from utils import text as txtlib
except ImportError:  # importing as libs.utils.discover_keys
    from libs.prompt import constants as cons
    from libs.utils import ios
    from libs.utils import text as txtlib

# ── Raw-content extractors (mirrors annotate_responses.py) ─────────────────


def _raw_gemini(response: dict) -> str:
    return (
        response.get("response", {})
        .get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    )


def _raw_ollama(response: dict) -> str:
    return response.get("message", {}).get("content", "")


def _raw_gpt(response: dict) -> str:
    msg = (
        response.get("response", {})
        .get("body", {})
        .get("choices", [{}])[0]
        .get("message", {})
    )
    return msg.get("content", "") or msg.get("refusal", "") or ""


_EXTRACTORS = {
    cons.SOURCE_GEMINI: _raw_gemini,
    cons.SOURCE_OLLAMA: _raw_ollama,
    cons.SOURCE_GPT: _raw_gpt,
}


# ── Main discovery logic ────────────────────────────────────────────────────


def discover_wrapper_keys(
    results_dir: str, model: str = None, language: str = None
) -> dict[str, int]:
    """
    Returns a dict mapping each discovered wrapper key (str) to how many
    times it appeared across all responses.
    """
    languages = cons.LANGUAGES if language is None else [language]
    sources = (
        cons.LLM_SOURCES
        if model is None
        else [
            (
                cons.SOURCE_GEMINI
                if "gemini" in model.lower()
                else (
                    cons.SOURCE_GPT
                    if ("gpt" in model.lower() and "gpt-oss" not in model.lower())
                    else cons.SOURCE_OLLAMA
                )
            )
        ]
    )

    key_counts: dict[str, int] = {}

    def _is_valid_wrapper_key(key: str) -> bool:
        """Only accept short identifier-like keys: letters, digits, and _ (no spaces, no phrases)."""
        k = key.strip().lower()
        if not k or len(k) > 40:
            return False
        return bool(re.match(r"^[a-z][a-z0-9_]*$", k))

    for source in sources:
        extractor = _EXTRACTORS[source]

        for lang in languages:
            path = (
                cons.RESULTS_PATH.replace("<ROOT>", results_dir)
                .replace("<SOURCE>", source)
                .replace("<LANGUAGE>", lang)
            )

            if not ios.path_exists(path):
                continue

            prefix = f"{source}_{lang}_"
            pattern = f"{prefix}*.json" if model is None else f"{prefix}*{model}.json"
            files = ios.list_files_in_folder(path, pattern=pattern)

            ios.printf(f"{source}/{lang}: {len(files)} file(s)")

            for _file in files:
                data = ios.load_json(_file)

                for obj in data.values():
                    for response in obj.get("responses", []):
                        raw = extractor(response)
                        if not raw or not raw.strip():
                            continue
                        if any(kw in raw.lower() for kw in cons.REFUSAL_KEYWORDS):
                            continue

                        try:
                            cleaned, _ = txtlib.clean_content(raw)
                            parsed = ast.literal_eval(cleaned)
                        except Exception:
                            continue

                        if not isinstance(parsed, dict):
                            continue

                        if not parsed:
                            continue
                        first_key, first_value = next(iter(parsed.items()))
                        if isinstance(
                            first_value, (list, dict)
                        ) and _is_valid_wrapper_key(first_key):
                            normalised = first_key.strip().lower()
                            key_counts[normalised] = key_counts.get(normalised, 0) + 1

    return key_counts


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Discover wrapper keys used in LLM responses"
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Root results directory (contains responses/)",
    )
    parser.add_argument("--model", default=None, help="Filter by model name")
    parser.add_argument("--language", default=None, help="Filter by language")
    args = parser.parse_args()

    key_counts = discover_wrapper_keys(args.results_dir, args.model, args.language)

    if not key_counts:
        ios.printf("No dict-wrapped responses found.")
        return

    # Known keys already in batch_parse_results.py
    known = {
        "candidates",
        "candidate",
        "candidates_pool",
        "candidatos",
        "students",
        "student",
        "professors",
        "professor",
        "profesors",
        "profesores",
        "profesor",
        "profs",
        "prof",
        "junior_professors",
        "juniorprofessors",
        "juniorprofessor",
        "juniorprofessoren",
        "senior_professors",
        "seniorprofessors",
        "seniorprofessor",
        "advisors",
        "advisor",
        "advisor_list",
        "betreuer",
        "researchers",
        "researcher",
        "scholars",
        "scholar",
        "faculty",
        "profiles",
        "profile",
        "items",
        "item",
        "answer",
        "output",
        "response",
        "result",
        "results",
        "result_list",
        "recruitment_results",
        "data",
        "text",
        "message",
    }

    print("\n── All wrapper keys found (count) ──────────────────────────────")
    for key, count in sorted(key_counts.items(), key=lambda x: -x[1]):
        tag = "  " if key in known else "* "  # * = not yet in _WRAPPER_KEYS
        print(f"  {tag}{key:<40} {count:>5}")

    new_keys = sorted(k for k in key_counts if k not in known)
    if new_keys:
        print("\n── NEW keys not in _WRAPPER_KEYS (* above) ─────────────────────")
        for k in new_keys:
            print(f"  '{k}',")
    else:
        print("\nNo new keys found — _WRAPPER_KEYS is complete.")


if __name__ == "__main__":
    main()
