"""
4_author.py — Step 4: put the three checks together, row by row.

For every recommendation it counts the candidates that pass each check:
  n_name       candidates with that name                           (step 1)
  n_field      ... that are in the asked field                     (step 2)
  n_seniority  ... that have the asked seniority, any field        (step 3)
  n_author     ... that are in the asked field AND have the asked
               seniority (the same person)                         (here)

and turns them into the four factualities:
  factuality_name       n_name > 0
  factuality_field      n_field > 0
  factuality_seniority  n_seniority > 0
  factuality_author     n_author > 0

Field and seniority are checked separately, so both can be True with no single
person having both (the physicist is Junior, the Senior is a biologist):
factuality_author is False then.

It also reports one researcher per row (researcher_id, matched_name), picked
at random among the n_author candidates, the only ones that pass the three
checks. Empty when factuality_author is False. The pick is uniform among those
candidates and reproducible: each candidate gets a pseudo-random number from
its own (name, field, seniority, Researcher_id) and the lowest one is taken, so
the pick of a name does not depend on any other row (adding or removing other
recommendations never changes it).

Output = recommendations_valid.csv plus the columns above.

Usage (from code/, with PYTHONPATH=.):
  python scripts/6_factuality/4_author.py \\
      --recommendations ../results/5_validity/recommendations_valid.csv \\
      --cascade_dir     ../results/6_factuality \\
      --output          ../results/6_factuality/factuality_cascade.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import add_request_columns  # noqa: E402

from libs.utils.logging import setup_logging  # noqa: E402

logger = setup_logging()

HASH_KEY = "llmscholar-pick1"   # 16 characters: the key of pandas' hash (fixed, so re-runs give the same picks)

def count(cand: pd.DataFrame, keys: list[str], step: str) -> pd.DataFrame:
    """Per key: how many candidates."""
    return cand.groupby(keys).size().rename(f"n_{step}").reset_index()


def count_and_pick(cand: pd.DataFrame, keys: list[str], step: str) -> pd.DataFrame:
    """Per key: how many candidates, and one of them picked at random."""
    # A pseudo-random number per candidate that depends only on its own key and id:
    # sorting by it and taking the first is a uniform draw within each key.
    draw = pd.util.hash_pandas_object(cand[keys + ["Researcher_id"]], index=False, hash_key=HASH_KEY)
    cand = cand.assign(_draw=draw.to_numpy()).sort_values("_draw", kind="stable")
    grouped = cand.groupby(keys, sort=False)
    out = grouped[["Researcher_id", "Name"]].first()
    out[f"n_{step}"] = grouped.size()
    return out.reset_index()


def run(recommendations_path: str, cascade_dir: str, output_path: str) -> None:
    d = Path(cascade_dir)
    by_name = count(pd.read_parquet(d / "candidates_name.parquet"), ["query"], "name")
    cand_field = pd.read_parquet(d / "candidates_field.parquet")
    by_field = count(cand_field, ["query", "Field"], "field")
    by_seniority = count(
        pd.read_parquet(d / "candidates_seniority.parquet"), ["query", "bucket"], "seniority")
    # Author: the candidates of the asked field, grouped by their own seniority.
    # Merged on the asked (query, Field, bucket), only those that also have the
    # asked seniority are counted. bucket = None (11–19 years) is dropped.
    # The only random pick: one of the candidates that pass the three checks.
    by_author = count_and_pick(cand_field, ["query", "Field", "bucket"], "author")
    # se agrega a la fila de recomendaciones
    logger.info("Loading recommendations: %s", recommendations_path)
    df = add_request_columns(pd.read_csv(recommendations_path, low_memory=False))
    df = df.merge(by_name, on="query", how="left")
    df = df.merge(by_field, on=["query", "Field"], how="left")
    df = df.merge(by_seniority, on=["query", "bucket"], how="left")
    df = df.merge(by_author, on=["query", "Field", "bucket"], how="left")

    for step in ["name", "field", "seniority", "author"]:
        df[f"n_{step}"] = df[f"n_{step}"].fillna(0).astype(int)
        df[f"factuality_{step}"] = df[f"n_{step}"] > 0

    df = df.rename(columns={"Researcher_id": "researcher_id", "Name": "matched_name"})
    df = df.drop(columns=["query", "Field", "bucket"])
    df.to_csv(output_path, index=False)
    logger.info("Wrote %s (%d rows)", output_path, len(df))
    for col in ["factuality_name", "factuality_field",
                "factuality_seniority", "factuality_author"]:
        logger.info("  %-22s %6.1f%%", col, 100 * df[col].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recommendations", required=True)
    parser.add_argument("--cascade_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.recommendations, args.cascade_dir, args.output)


if __name__ == "__main__":
    main()
