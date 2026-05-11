"""
factuality_seniority.py — Step 3 of the factuality pipeline.

Reads the output of factuality_field_check.py and decides whether the seniority
the LLM assigned to each author matches reality.

Career age = 2025 − First_year (precomputed upstream as gt_career_age).

Buckets:
  career_age <= 10 → Junior  (early-career)
  career_age >= 20 → Senior
  11–19            → unclassifiable (seniority_unknown)

LLM target mapping (EN / ES / DE):
  Junior Professor  → Junior
  Senior Professor  → Senior

Output columns added:
  seniority_career_age   career age used for bucketing (2025 − First_year)
  seniority_age_source   {gt | none}
  seniority_bucket       {Junior | Senior | None}  (None = 11–19 years)
  seniority_llm_bucket   {Junior | Senior | None}
  seniority_status       {seniority_match | seniority_mismatch
                          | seniority_unknown | not_applicable}

Usage (from code/scripts/):
  python factuality_seniority.py \\
      --input  ../../../results/summary/factuality_field.csv \\
      --output ../../../results/summary/factuality_seniority.csv
"""

import argparse
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

JUNIOR_MAX  = 10   # career_age <= 10 → Junior
SENIOR_MIN  = 20   # career_age >= 20 → Senior  (11–19 → unclassifiable)

STATUS_MATCH          = "seniority_match"
STATUS_MISMATCH       = "seniority_mismatch"
STATUS_UNKNOWN        = "seniority_unknown"
STATUS_NOT_APPLICABLE = "not_applicable"

SOURCE_GT   = "gt"
SOURCE_NONE = "none"

# author_status values (factuality_author_jw pipeline)
AUTHOR_HALLUCINATED = "hallucinated"
# factuality_status values (factuality_field pipeline)
FIELD_NOT_FOUND = "not_found"


def _is_not_found(row: pd.Series) -> bool:
    """True when the author was not matched in any ground-truth source."""
    return (
        row.get("author_status") == AUTHOR_HALLUCINATED
        or row.get("factuality_status") == FIELD_NOT_FOUND
    )

# Maps every observed `target` value (EN/ES/DE) → canonical bucket
LLM_TARGET_TO_BUCKET = {
    "Senior Professor":    "Senior",
    "Profesor(a) Sénior":  "Senior",
    "Seniorprofessor(in)": "Senior",
    "Junior Professor":    "Junior",
    "Profesor(a) Júnior":  "Junior",
    "Juniorprofessor(in)": "Junior",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_career_age(row: pd.Series) -> tuple[float | None, str]:
    """Return (career_age, source). Uses gt_career_age (2025 − First_year)."""
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
            "seniority_bucket":     None,
            "seniority_llm_bucket": LLM_TARGET_TO_BUCKET.get(str(row.get("target") or "").strip()),
            "seniority_status":     STATUS_NOT_APPLICABLE,
        }

    llm_bucket = LLM_TARGET_TO_BUCKET.get(str(row.get("target") or "").strip())
    age, src   = _resolve_career_age(row)

    if age is None or llm_bucket is None:
        return {
            "seniority_career_age": age,
            "seniority_age_source": src,
            "seniority_bucket":     None,
            "seniority_llm_bucket": llm_bucket,
            "seniority_status":     STATUS_UNKNOWN,
        }

    actual_bucket = _bucket(age)
    if actual_bucket is None:
        status = STATUS_UNKNOWN  # 11–19 years: not Junior nor Senior
    else:
        status = STATUS_MATCH if actual_bucket == llm_bucket else STATUS_MISMATCH
    return {
        "seniority_career_age": age,
        "seniority_age_source": src,
        "seniority_bucket":     actual_bucket,
        "seniority_llm_bucket": llm_bucket,
        "seniority_status":     status,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str) -> None:
    logger.info("Loading: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    logger.info("Rows: %d", len(df))

    records = [classify_row(row) for _, row in df.iterrows()]
    for col in ["seniority_career_age", "seniority_age_source",
                "seniority_bucket", "seniority_llm_bucket", "seniority_status"]:
        df[col] = [r[col] for r in records]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows → %s", len(df), output_path)

    n = len(df)
    logger.info("Seniority status distribution:")
    for status, count in df["seniority_status"].value_counts().items():
        logger.info("  %-25s %6d  (%.1f%%)", status, count, 100 * count / n)
    logger.info("GT bucket distribution (found authors only):")
    found = df[~df.apply(_is_not_found, axis=1)]
    for bucket, count in found["seniority_bucket"].value_counts().items():
        logger.info("  %-10s %6d  (%.1f%%)", bucket, count, 100 * count / len(found))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: verify the LLM-assigned seniority matches the author's actual career stage"
    )
    parser.add_argument("--input",  required=True, help="Path to factuality_field.csv (output of factuality_field_check.py)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    run(args.input, args.output)


if __name__ == "__main__":
    main()
