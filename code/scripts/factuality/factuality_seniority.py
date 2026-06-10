"""
factuality_seniority.py — Step 3 of the factuality pipeline.

Reads the output of factuality_field_check.py and decides whether the seniority
the LLM assigned to each author matches reality.

Career age = current_year − First_year (precomputed upstream as gt_career_age).

Buckets:
  career_age <= 10 → Junior  (early-career)
  career_age >= 20 → Senior
  11–19            → unclassifiable (seniority_unknown)

LLM target mapping (EN / ES / DE):
  Junior Professor  → Junior
  Senior Professor  → Senior

Output columns added:
  seniority_career_age   career age used for bucketing (current_year − First_year)
  seniority_age_source   {gt | none}
  seniority_bucket       {Junior | Senior | None}  (None = 11–19 years)
  seniority_llm_bucket   {Junior | Senior | None}
  seniority_status       {seniority_match | seniority_mismatch
                          | seniority_unknown | not_applicable}

Usage (from code/, with PYTHONPATH=.):
  python scripts/factuality/factuality_seniority.py \\
      --input  ../results/summary/factuality_field.csv \\
      --output ../results/summary/factuality_seniority.csv
"""

import argparse

import pandas as pd

from libs.metrics.constants import (
    FACTUALITY_AUTHOR_HALLUCINATED as AUTHOR_HALLUCINATED,
    LLM_TARGET_TO_BUCKET,
    factuality_status_flags,
)
from libs.utils.cli import add_io_args
from libs.utils.ios import read_input_csv, write_output_csv
from libs.utils.logging import log_value_counts, setup_logging

logger = setup_logging()

# ── Constants ──────────────────────────────────────────────────────────────────

JUNIOR_MAX = 10  # career_age <= 10 → Junior
SENIOR_MIN = 20  # career_age >= 20 → Senior  (11–19 → unclassifiable)

_STATUS = factuality_status_flags("seniority")
STATUS_MATCH = _STATUS["MATCH"]
STATUS_MISMATCH = _STATUS["MISMATCH"]
STATUS_UNKNOWN = _STATUS["UNKNOWN"]
STATUS_NOT_APPLICABLE = _STATUS["NOT_APPLICABLE"]

SOURCE_GT = "gt"
SOURCE_NONE = "none"

# factuality_status values (factuality_field pipeline)
FIELD_NOT_FOUND = "not_found"


def _is_not_found(row: pd.Series) -> bool:
    """True when the author was not matched in any ground-truth source."""
    return (
        row.get("author_status") == AUTHOR_HALLUCINATED
        or row.get("factuality_status") == FIELD_NOT_FOUND
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _resolve_career_age(row: pd.Series) -> tuple[float | None, str]:
    """Return (career_age, source). Uses gt_career_age (current_year − First_year)."""
    val = row.get("gt_career_age")
    if pd.notna(val):
        return float(val), SOURCE_GT
    return None, SOURCE_NONE


def _bucket(career_age: float) -> str | None:
    if career_age <= JUNIOR_MAX:
        return "Junior"
    if career_age >= SENIOR_MIN:
        return "Senior"
    return None  # 11–19 years: unclassifiable


def classify_row(row: pd.Series) -> dict:
    if _is_not_found(row):
        return {
            "seniority_career_age": None,
            "seniority_age_source": SOURCE_NONE,
            "seniority_bucket": None,
            "seniority_llm_bucket": LLM_TARGET_TO_BUCKET.get(
                str(row.get("target") or "").strip()
            ),
            "seniority_status": STATUS_NOT_APPLICABLE,
        }

    llm_bucket = LLM_TARGET_TO_BUCKET.get(str(row.get("target") or "").strip())
    age, src = _resolve_career_age(row)

    if age is None or llm_bucket is None:
        return {
            "seniority_career_age": age,
            "seniority_age_source": src,
            "seniority_bucket": None,
            "seniority_llm_bucket": llm_bucket,
            "seniority_status": STATUS_UNKNOWN,
        }

    actual_bucket = _bucket(age)
    if actual_bucket is None:
        status = STATUS_UNKNOWN  # 11–19 years: not Junior nor Senior
    else:
        status = STATUS_MATCH if actual_bucket == llm_bucket else STATUS_MISMATCH
    return {
        "seniority_career_age": age,
        "seniority_age_source": src,
        "seniority_bucket": actual_bucket,
        "seniority_llm_bucket": llm_bucket,
        "seniority_status": status,
    }


# ── Main ───────────────────────────────────────────────────────────────────────


def run(input_path: str, output_path: str) -> None:
    df = read_input_csv(input_path, logger=logger)

    records = [classify_row(row) for _, row in df.iterrows()]
    for col in [
        "seniority_career_age",
        "seniority_age_source",
        "seniority_bucket",
        "seniority_llm_bucket",
        "seniority_status",
    ]:
        df[col] = [r[col] for r in records]

    write_output_csv(df, output_path, logger=logger)
    log_value_counts(
        df, "seniority_status",
        title="Seniority status distribution", width=25, logger=logger,
    )
    found = df[~df.apply(_is_not_found, axis=1)]
    log_value_counts(
        found, "seniority_bucket",
        title="GT bucket distribution (found authors only)", width=10, logger=logger,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: verify the LLM-assigned seniority matches the author's actual career stage"
    )
    add_io_args(
        parser,
        input_help="Path to factuality_field.csv (output of factuality_field_check.py)",
    )
    args = parser.parse_args()

    run(args.input, args.output)


if __name__ == "__main__":
    main()
