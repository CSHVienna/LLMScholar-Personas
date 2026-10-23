"""
1_validity.py — Funnel step 3.1: validity, refusal and duplicates of every call.

Reads the LLM responses as parsed from the JSON files by
scripts/1_prompting/batch_parse_results.py (recommendations.csv: one row per
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

Output: ../results/pipeline/validity/<model>.csv — call columns + validity, refusal, duplicates
(+ the two counts), and ../results/summary/recommendations_valid.csv — the rows of
recommendations.csv whose call is valid, the input of scripts/4_factuality/.

Usage (from code/, with PYTHONPATH=.):
  python scripts/3_validity/1_validity.py
"""

import argparse

import pandas as pd

from libs.pipeline_common import (
    CALL, RECOMMENDATIONS, RECOMMENDATIONS_VALID, author_key, load, logger, write_step,
)

from libs.metrics.constants import REFUSED_FLAG, VALID_FLAGS  # noqa: E402


def write_valid_recommendations(recommendations_path: str, output_path: str) -> None:
    """Every column of the recommendations of valid calls. valid_flag is the same
    for every row of a call, so filtering rows is filtering calls. Read as text
    and in chunks: the values are copied as they are, without loading 5 GB."""
    n_all = n_valid = 0
    chunks = pd.read_csv(recommendations_path, dtype=str, chunksize=500_000)
    for i, chunk in enumerate(chunks):
        valid = chunk[chunk["valid_flag"].isin(VALID_FLAGS)]
        valid.to_csv(output_path, index=False, mode="w" if i == 0 else "a", header=i == 0)
        n_all, n_valid = n_all + len(chunk), n_valid + len(valid)
    logger.info("Valid calls: %d of %d recommendations → %s", n_valid, n_all, output_path)


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
