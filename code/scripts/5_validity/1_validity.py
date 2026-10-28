"""
1_validity.py — Funnel step 3.1: validity, refusal and duplicates of every call.

Reads the LLM responses as parsed from the JSON files by
scripts/3_prompting/batch_parse_results.py (recommendations.csv: one row per
recommended author, with the response's valid_flag and its run_id 1–10).

Per call:
  validity    True if the response is a populated JSON (valid_flag cleaned or
              unchanged)
  refusal     True if the LLM declined and said why (valid_flag refused). Not
              the same as invalid: an invalid response is malformed or empty.
  duplicates  1 − unique authors / authors in the response (paper Eq. 3),
              authors compared by "name lastname" normalized (common.author_key).
              Only for valid calls.
  n_total_recommendations, n_unique_authors   the two counts behind duplicates

Output: ../results/5_validity/validity/<model>.csv — call columns + validity, refusal, duplicates
(+ the two counts), and ../results/5_validity/recommendations_valid.csv — the rows of
recommendations.csv whose call is valid, each author once per call (repeated names
left out after duplicates is measured) and only complete names (has_full_name: a
single first or last name is left out), the input of scripts/6_factuality/. This
is the only place both filters are applied; every later step reads this file.

Usage (from code/, with PYTHONPATH=.):
  python scripts/5_validity/1_validity.py
"""

import argparse

import pandas as pd

from libs.pipeline_common import (
    CALL, PARAMS, RECOMMENDATIONS, RECOMMENDATIONS_VALID, author_key, has_full_name, load, logger,
    write_step,
)

from libs.metrics.constants import REFUSED_FLAG, VALID_FLAGS  # noqa: E402


def write_valid_recommendations(recommendations_path: str, output_path: str) -> None:
    """What passes to 6_factuality/: every column of the recommendations of valid
    calls, each author once per call (a name the LLM repeated in the same answer
    keeps its first row; duplicates was measured before, on all of them), and only
    complete names (has_full_name).
    valid_flag is the same for every row of a call, so filtering rows is
    filtering calls. Read as text: the values are copied as they are."""
    df = pd.read_csv(recommendations_path, dtype=str)
    valid = df[df["valid_flag"].isin(VALID_FLAGS)].copy()
    valid["_author"] = author_key(valid)
    unique = valid.drop_duplicates(["model", "created_at", "run_id", *PARAMS, "_author"])
    complete = unique[has_full_name(unique)]
    complete.drop(columns="_author").to_csv(output_path, index=False)
    logger.info("Valid calls: %d of %d recommendations, %d repeated names and %d incomplete "
                "names left out → %d → %s", len(valid), len(df), len(valid) - len(unique),
                len(unique) - len(complete), len(complete), output_path)


def run(recommendations_path: str, valid_output: str) -> None:
    df = load(recommendations_path, ["valid_flag", "name", "lastname"])
    df["author"] = author_key(df)

    calls = df.groupby(CALL, dropna=False).agg(
        valid_flag=("valid_flag", "first"),   # the same for every row of a call
        n_authors=("author", "count"),
        n_unique=("author", "nunique"),
    ).reset_index()

    calls["validity"] = calls["valid_flag"].isin(VALID_FLAGS)
    calls["refusal"] = calls["valid_flag"] == REFUSED_FLAG
    calls["duplicates"] = (1 - (calls["n_unique"] / calls["n_authors"])).where(
        calls["validity"] & (calls["n_authors"] > 0)
    )

    calls = calls.rename(columns={"n_authors": "n_total_recommendations", "n_unique": "n_unique_authors"})
    write_step(calls[CALL + ["validity", "refusal", "duplicates",
                             "n_total_recommendations", "n_unique_authors"]], step="validity")
    write_valid_recommendations(recommendations_path, valid_output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recommendations", default=RECOMMENDATIONS)
    parser.add_argument("--valid_output", default=RECOMMENDATIONS_VALID)
    args = parser.parse_args()
    run(args.recommendations, args.valid_output)


if __name__ == "__main__":
    main()
