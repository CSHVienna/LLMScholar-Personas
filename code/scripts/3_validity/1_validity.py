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
(+ the two counts).

Usage (from code/, with PYTHONPATH=.):
  python scripts/3_validity/1_validity.py
"""

import argparse

from libs.pipeline_common import CALL, RECOMMENDATIONS, author_key, load, write_step

from libs.metrics.constants import REFUSED_FLAG, VALID_FLAGS  # noqa: E402


def run(recommendations_path: str) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recommendations", default=RECOMMENDATIONS)
    args = parser.parse_args()
    run(args.recommendations)


if __name__ == "__main__":
    main()
