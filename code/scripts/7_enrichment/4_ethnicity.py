"""
4_ethnicity.py — Step 4 of run_enrichment.py: perceived ethnicity of the picked researcher.

Takes `perceived_ethnicity` from the ground-truth ethnicity table
(results/1_ground_truth/ethnicity/researcher_ethnicity_lookup.csv: one row per Semantic Scholar
researcher, inferred from their name by scripts/ethnicity/) by the researcher_id
the factuality cascade picked, the same source as the classes and the parity
reference of 8_metrics/2_socials.py. Only the recommendations with
factuality_author = True get it, like the rest of the enrichment; the others
keep it empty.

Output columns added:
  perceived_ethnicity   {Asian | White | Black or African American
                         | Hispanic or Latino | Unknown}, empty when
                         factuality_author = False

Usage (from code/, with PYTHONPATH=.):
  python scripts/7_enrichment/4_ethnicity.py \\
      --input             ../results/7_enrichment/factuality_affiliation.csv \\
      --ethnicity_lookup  ../results/1_ground_truth/ethnicity/researcher_ethnicity_lookup.csv \\
      --output            ../results/7_enrichment/factuality_full.csv
"""

import argparse

import pandas as pd

from libs.utils.cli import add_io_args
from libs.utils.ios import read_input_csv, write_output_csv
from libs.utils.logging import log_value_counts, setup_logging

logger = setup_logging()

LOOKUP_COLS = ["Researcher_id", "perceived_ethnicity"]


def _build_lookup(lookup_path: str) -> pd.Series:
    """researcher_id → perceived_ethnicity, from the ground-truth ethnicity table."""
    logger.info("Loading ethnicity lookup: %s", lookup_path)
    df = pd.read_csv(lookup_path, usecols=LOOKUP_COLS)
    df["Researcher_id"] = df["Researcher_id"].astype("int64")
    logger.info("Lookup rows: %d", len(df))
    return df.drop_duplicates("Researcher_id").set_index("Researcher_id")["perceived_ethnicity"]


def run(input_path: str, lookup_path: str, output_path: str) -> None:
    lookup = _build_lookup(lookup_path)
    df = read_input_csv(input_path, logger=logger)

    author = df["factuality_author"].astype(str) == "True"
    ids = pd.to_numeric(df.loc[author, "researcher_id"], errors="coerce").astype("Int64")

    df["perceived_ethnicity"] = None
    df.loc[author, "perceived_ethnicity"] = ids.map(lookup).to_numpy()

    missing = df.loc[author, "perceived_ethnicity"].isna().sum()
    logger.info(
        "Ethnicity of the picked researcher: %d / %d with factuality_author = True "
        "(not in the table: %d; Unknown in the table: %d)",
        author.sum() - missing, author.sum(), missing,
        (df.loc[author, "perceived_ethnicity"] == "Unknown").sum(),
    )

    write_output_csv(df, output_path, logger=logger)
    log_value_counts(
        df, "perceived_ethnicity",
        title="Perceived ethnicity distribution", width=28, logger=logger,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4 of run_enrichment.py: perceived_ethnicity of the picked researcher, from the ground-truth ethnicity table"
    )
    add_io_args(
        parser,
        input_help="Path to factuality_affiliation.csv (output of 3_affiliation.py)",
        output_help="Output CSV path (final factuality dataset)",
    )
    parser.add_argument(
        "--ethnicity_lookup",
        required=True,
        help="Path to researcher_ethnicity_lookup.csv",
    )
    args = parser.parse_args()

    run(args.input, args.ethnicity_lookup, args.output)


if __name__ == "__main__":
    main()
