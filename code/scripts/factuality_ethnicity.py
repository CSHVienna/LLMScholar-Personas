"""
factuality_ethnicity.py — Step 5 of the factuality pipeline.

Joins the pre-computed `perceived_ethnicity` column from
`recommendations_with_ethnicity.csv` (produced previously by the BERT+LSTM
cascade) onto the output of factuality_location.py.

We do NOT re-run the cascade — it's an expensive operation that already ran
once over the full recommendations dataset. The lookup is built as
(name, lastname) → perceived_ethnicity, which is valid because the cascade
output only depends on the author's name.

Output columns added:
  perceived_ethnicity   {Asian | White | Black or African American
                         | Hispanic or Latino | Unknown}

Usage (from code/scripts/):
  python factuality_ethnicity.py \\
      --input             ../../results/summary/factuality_location.csv \\
      --ethnicity_lookup  ../../results/summary/recommendations_with_ethnicity.csv \\
      --output            ../../results/summary/factuality_ethnicity.csv
"""

import argparse
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LOOKUP_COLS = ["name", "lastname", "perceived_ethnicity"]


def _build_lookup(lookup_path: str) -> dict[tuple[str, str], str]:
    """Build a (name, lastname) → perceived_ethnicity dict from the lookup CSV."""
    logger.info("Loading ethnicity lookup: %s", lookup_path)
    df = pd.read_csv(lookup_path, low_memory=False, usecols=LOOKUP_COLS)
    logger.info("Lookup rows: %d", len(df))

    df = df.dropna(subset=["name", "lastname", "perceived_ethnicity"])
    df["name"]     = df["name"].astype(str).str.strip()
    df["lastname"] = df["lastname"].astype(str).str.strip()

    df = df.drop_duplicates(subset=["name", "lastname"], keep="first")
    logger.info("Unique (name, lastname) pairs: %d", len(df))

    return dict(zip(zip(df["name"], df["lastname"]), df["perceived_ethnicity"]))


def run(input_path: str, lookup_path: str, output_path: str) -> None:
    lookup = _build_lookup(lookup_path)

    logger.info("Loading: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    logger.info("Rows: %d", len(df))

    names     = df["name"].fillna("").astype(str).str.strip()
    lastnames = df["lastname"].fillna("").astype(str).str.strip()

    df["perceived_ethnicity"] = [
        lookup.get((n, l), "Unknown") for n, l in zip(names, lastnames)
    ]

    n_unknown = (df["perceived_ethnicity"] == "Unknown").sum()
    logger.info("Lookup hits: %d / %d  (Unknown: %d)", len(df) - n_unknown, len(df), n_unknown)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows → %s", len(df), output_path)

    n = len(df)
    logger.info("Perceived ethnicity distribution:")
    for cat, count in df["perceived_ethnicity"].value_counts().items():
        logger.info("  %-28s %6d  (%.1f%%)", cat, count, 100 * count / n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 5: join pre-computed perceived_ethnicity from recommendations_with_ethnicity.csv"
    )
    parser.add_argument("--input",            required=True, help="Path to factuality_location.csv (output of step 4)")
    parser.add_argument("--ethnicity_lookup", required=True, help="Path to recommendations_with_ethnicity.csv")
    parser.add_argument("--output",           required=True, help="Output CSV path (final factuality dataset)")
    args = parser.parse_args()

    run(args.input, args.ethnicity_lookup, args.output)


if __name__ == "__main__":
    main()
