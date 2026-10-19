"""
1_validity.py — Step 1: validity, refusal and duplicates of every call.

Reads the LLM responses as parsed from the JSON files by
scripts/prompting/batch_parse_results.py (recommendations.csv: one row per
recommended author, with the response's valid_flag and its run_id 1–10).

Per call:
  validity    True if the response is a populated JSON (valid_flag cleaned or
              unchanged)
  refusal     True if the LLM declined and said why (valid_flag refused). Not
              the same as invalid: an invalid response is malformed or empty.
  duplicates  1 − unique authors / authors in the response (paper Eq. 3),
              authors compared by "name lastname". Only for valid calls.

Output: 1_validity/<model>.csv — call columns + validity, refusal, duplicates.

Usage (from code/, with PYTHONPATH=.):
  python scripts/pipeline/1_validity.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CALL, RECOMMENDATIONS, author_key, load, write_step  # noqa: E402

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
    calls["duplicates"] = (1 - calls["n_unique"] / calls["n_authors"]).where(
        calls["validity"] & (calls["n_authors"] > 0)
    )

    write_step(calls[CALL + ["validity", "refusal", "duplicates"]], step=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recommendations", default=RECOMMENDATIONS)
    args = parser.parse_args()
    run(args.recommendations)


if __name__ == "__main__":
    main()
