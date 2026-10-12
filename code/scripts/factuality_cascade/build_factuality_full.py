"""
build_factuality_full.py — Step 5: the cascade in the format the metrics expect.

scripts/metrics/build_valid_calls.py reads a factuality_full.csv with status
columns (author_status, field_status, ...). This script builds that file from
the cascade, so the metrics and notebooks run on it unchanged.

What it does:
  1. Reads factuality_cascade.csv (step 4).
  2. Copies the OpenAlex, location, affiliation and ethnicity columns from the
     OLD factuality_full.csv. They only depend on the recommended name, which
     is the same, so they are not recomputed. Both files have one row per
     recommendation in the same order; the script checks that name and
     lastname agree row by row before copying.
  3. Turns the cascade's True/False into the old status columns:
       author_status     found | hallucinated          (factuality_name)
       field_status      field_match | field_mismatch
                         | not_applicable when the name was not found
       seniority_status  seniority_match | seniority_mismatch
                         | not_applicable when the name was not found
     so field and seniority are both measured over found names.
  4. Recomputes location_status and affiliation_status from their stored
     columns: same rules as factuality_location / factuality_affiliation, but
     not_applicable now follows factuality_name instead of the old
     author_status.
  5. Adds gt_field, gt_gender, gt_career_age, gt_citations for the researcher
     the cascade picked (looked up by researcher_id in the parquet), and
     seniority_bucket / seniority_llm_bucket (actual vs asked seniority), which
     the seniority Sankey reads.
  6. Adds tiebreak, which the Sankeys in metrics_pipeline filter on:
       unique   the name had a single candidate (nothing was chosen for us)
       cascade  several candidates; filtered by field/seniority, then random
       ""       name not found

The old factuality_full.csv is only read, never written.

Usage (from code/, with PYTHONPATH=.):
  python scripts/factuality_cascade/build_factuality_full.py \\
      --cascade ../results/summary/cascade/factuality_cascade.csv \\
      --old     ../results/summary/factuality_full.csv \\
      --output  ../results/summary/cascade/factuality_full.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import seniority_bucket  # noqa: E402

from libs.metrics.constants import LLM_TARGET_TO_BUCKET  # noqa: E402
from libs.utils.config import config_default  # noqa: E402
from libs.utils.logging import setup_logging  # noqa: E402

logger = setup_logging()

REFERENCE_YEAR = 2025  # same as factuality_author_jw
AFFILIATION_THRESHOLD = 85  # same as factuality_affiliation.TOKEN_SET_RATIO_THRESHOLD

# Columns copied as they are from the old factuality_full.csv.
OLD_COLS = [
    "oa_status", "oa_id", "oa_display_name", "oa_works_count",
    "oa_cited_by_count", "oa_h_index", "oa_i10_index", "oa_first_pub_year",
    "oa_last_pub_year", "oa_career_age", "oa_country_code",
    "oa_last_institution", "oa_match_score",
    "location_llm_country", "location_llm_iso", "location_oa_iso",
    "location_oa_institution",
    "affiliation_llm", "affiliation_oa_all", "affiliation_best_match_score",
    "affiliation_best_match_oa",
    "perceived_ethnicity",
]


def status(ok: pd.Series, applicable: pd.Series, prefix: str) -> np.ndarray:
    """match / mismatch where applicable, not_applicable elsewhere."""
    return np.where(
        ~applicable, "not_applicable",
        np.where(ok, f"{prefix}_match", f"{prefix}_mismatch"),
    )


def status_with_unknown(ok, known, applicable, prefix) -> np.ndarray:
    """Like status(), plus <prefix>_unknown when the data to decide is missing."""
    return np.where(
        ~applicable, "not_applicable",
        np.where(~known, f"{prefix}_unknown",
                 np.where(ok, f"{prefix}_match", f"{prefix}_mismatch")),
    )


def run(cascade_path: str, old_path: str, parquet_path: str, output_path: str) -> None:
    logger.info("Loading cascade: %s", cascade_path)
    df = pd.read_csv(cascade_path, low_memory=False)

    logger.info("Loading old columns: %s", old_path)
    old = pd.read_csv(old_path, usecols=["name", "lastname"] + OLD_COLS,
                      low_memory=False)

    # Same rows in the same order? Otherwise copying by position is wrong.
    same_rows = (
        len(df) == len(old)
        and df["name"].astype(str).equals(old["name"].astype(str))
        and df["lastname"].astype(str).equals(old["lastname"].astype(str))
    )
    if not same_rows:
        raise SystemExit("cascade and old factuality_full do not have the same rows")
    for col in OLD_COLS:
        df[col] = old[col].to_numpy()
    del old

    name_ok = df["factuality_name"]
    field_ok = df["factuality_field"]

    # Cascade → status columns.
    df["author_status"] = np.where(name_ok, "found", "hallucinated")
    df["field_status"] = status(field_ok, name_ok, "field")
    df["seniority_status"] = status(df["factuality_seniority"], name_ok, "seniority")

    # Location and affiliation: same rules as before, not_applicable from the cascade.
    df["location_status"] = status_with_unknown(
        ok=df["location_llm_iso"] == df["location_oa_iso"],
        known=df["location_llm_iso"].notna() & df["location_oa_iso"].notna(),
        applicable=name_ok,
        prefix="location",
    )
    df["affiliation_status"] = status_with_unknown(
        ok=df["affiliation_best_match_score"] >= AFFILIATION_THRESHOLD,
        known=df["affiliation_llm"].notna() & df["affiliation_oa_all"].notna(),
        applicable=name_ok,
        prefix="affiliation",
    )

    # gt_* of the researcher the cascade picked.
    logger.info("Loading researchers: %s", parquet_path)
    ref = pd.read_parquet(
        parquet_path,
        columns=["Researcher_id", "Field", "Combined_gender", "First_year", "Citations"],
    )
    ref = ref.drop_duplicates("Researcher_id")
    gt = pd.DataFrame({
        "researcher_id": ref["Researcher_id"],
        "gt_field": ref["Field"],
        "gt_gender": ref["Combined_gender"],
        "gt_career_age": (REFERENCE_YEAR - ref["First_year"]).clip(lower=0),
        "gt_citations": ref["Citations"],
    })
    df = df.merge(gt, on="researcher_id", how="left")
    df["seniority_bucket"] = seniority_bucket(df["gt_career_age"])
    df["seniority_llm_bucket"] = df["target"].map(LLM_TARGET_TO_BUCKET)
    df["tiebreak"] = np.where(
        ~name_ok, "", np.where(df["n_name"] == 1, "unique", "cascade")
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %s (%d rows)", output_path, len(df))
    for col in ["author_status", "field_status", "seniority_status",
                "location_status", "affiliation_status"]:
        logger.info("%s:\n%s", col, df[col].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cascade", required=True)
    parser.add_argument("--old", required=True)
    parser.add_argument("--parquet", default=config_default("ss_parquet"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).resolve() == Path(args.old).resolve():
        parser.error("--output cannot be the old factuality_full.csv")
    run(args.cascade, args.old, args.parquet, args.output)


if __name__ == "__main__":
    main()
