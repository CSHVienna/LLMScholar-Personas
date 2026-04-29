"""
apply_enrichment.py — Merge oa_enrichment.csv (oa_id → oa_country_code) into
factuality_author.csv, filling the oa_country_code column for every row whose
oa_id appears in the enrichment lookup.

Usage (from code/scripts/):
  python apply_enrichment.py \\
      --input      ../../results/summary/factuality_author.csv \\
      --enrichment ../../results/summary/oa_enrichment.csv \\
      --output     ../../results/summary/factuality_author_enriched.csv
"""

import argparse
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run(input_path: str, enrichment_path: str, output_path: str) -> None:
    logger.info("Loading enrichment: %s", enrichment_path)
    enr = pd.read_csv(enrichment_path, low_memory=False, usecols=["oa_id", "oa_country_code"])
    enr = enr.dropna(subset=["oa_id"]).drop_duplicates(subset=["oa_id"], keep="first")
    lookup = dict(zip(enr["oa_id"], enr["oa_country_code"]))
    n_with_country = sum(1 for v in lookup.values() if pd.notna(v))
    logger.info("Enrichment lookup: %d oa_ids  (with country: %d)", len(lookup), n_with_country)

    logger.info("Loading: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    logger.info("Rows: %d", len(df))

    before = df["oa_country_code"].notna().sum() if "oa_country_code" in df.columns else 0

    # For every row with an oa_id, prefer the enrichment country if the row
    # currently lacks one. Don't overwrite already-populated values.
    if "oa_country_code" not in df.columns:
        df["oa_country_code"] = None

    mask = df["oa_id"].notna() & df["oa_country_code"].isna()
    df.loc[mask, "oa_country_code"] = df.loc[mask, "oa_id"].map(lookup)

    after = df["oa_country_code"].notna().sum()
    logger.info("oa_country_code populated: before=%d  after=%d  (+%d)",
                before, after, after - before)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows → %s", len(df), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge oa_enrichment.csv country values into factuality_author.csv"
    )
    parser.add_argument("--input",      required=True, help="Path to factuality_author.csv")
    parser.add_argument("--enrichment", required=True, help="Path to oa_enrichment.csv")
    parser.add_argument("--output",     required=True, help="Output CSV path")
    args = parser.parse_args()

    run(args.input, args.enrichment, args.output)


if __name__ == "__main__":
    main()
