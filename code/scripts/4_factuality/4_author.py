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
at random: among the n_author if the author is correct, otherwise among those
of field, then seniority, then name, the first that has any. Empty if the name
was not found. The pick uses a fixed seed, so re-running gives the same
researchers.

Output = recommendations.csv plus the columns above.

Usage (from code/, with PYTHONPATH=.):
  python scripts/4_factuality/4_author.py \\
      --recommendations ../results/summary/recommendations.csv \\
      --cascade_dir     ../results/summary/cascade \\
      --output          ../results/summary/cascade/factuality_cascade.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import add_request_columns  # noqa: E402

from libs.utils.logging import setup_logging  # noqa: E402

logger = setup_logging()

SEED = 42

# se cuenta y se elige uno al azar
def count_and_pick(cand: pd.DataFrame, keys: list[str], step: str) -> pd.DataFrame:
    """Per key: how many candidates, and one of them picked at random."""
    cand = cand.sample(frac=1, random_state=SEED)  # shuffle, then take the first
    grouped = cand.groupby(keys, sort=False)
    out = grouped[["Researcher_id", "Name"]].first().add_suffix(f"_{step}")
    out[f"n_{step}"] = grouped.size()
    return out.reset_index()


def run(recommendations_path: str, cascade_dir: str, output_path: str) -> None:
    d = Path(cascade_dir)
    by_name = count_and_pick(
        pd.read_parquet(d / "candidates_name.parquet"), ["query"], "name")
    cand_field = pd.read_parquet(d / "candidates_field.parquet")
    by_field = count_and_pick(cand_field, ["query", "Field"], "field")
    by_seniority = count_and_pick(
        pd.read_parquet(d / "candidates_seniority.parquet"), ["query", "bucket"], "seniority")
    # Author: the candidates of the asked field, grouped by their own seniority.
    # Merged on the asked (query, Field, bucket), only those that also have the
    # asked seniority are counted. bucket = None (11–19 years) is dropped.
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

    # The researcher we report: random pick among those that pass everything,
    # otherwise among those of field, then seniority, then name.
    df["researcher_id"] = (
        df["Researcher_id_author"]
        .fillna(df["Researcher_id_field"])
        .fillna(df["Researcher_id_seniority"])
        .fillna(df["Researcher_id_name"])
    )
    df["matched_name"] = (
        df["Name_author"].fillna(df["Name_field"])
        .fillna(df["Name_seniority"]).fillna(df["Name_name"])
    )

    df = df.drop(columns=[
        "query", "Field", "bucket",
        *[f"{col}_{step}" for col in ["Researcher_id", "Name"]
          for step in ["name", "field", "seniority", "author"]],
    ])
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
