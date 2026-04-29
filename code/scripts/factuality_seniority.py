"""
factuality_seniority.py — Step 3 of the factuality pipeline.

Reads the output of factuality_field_check.py and decides whether the seniority
the LLM assigned to each author matches reality.

Career age comes from `last_pub_year - first_pub_year`. We prefer the OpenAlex
value (oa_career_age, set in step 1) and fall back to Semantic Scholar GT
(gt_career_age) when OpenAlex didn't resolve the author.

Threshold:
  career_age < 15  → Junior
  career_age >= 15 → Senior

The LLM `target` column comes in EN/ES/DE — both buckets in three languages
collapse to the same Junior/Senior label.

Output columns added:
  seniority_career_age   number used for bucketing
  seniority_age_source   {openalex | gt | none}
  seniority_bucket       {Junior | Senior | None}
  seniority_llm_bucket   {Junior | Senior | None}
  seniority_status       {seniority_match | seniority_mismatch
                          | seniority_unknown | not_applicable}

Usage (from code/scripts/):
  python factuality_seniority.py \\
      --input  ../../results/summary/factuality_field.csv \\
      --output ../../results/summary/factuality_seniority.csv
"""

import argparse
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

SENIOR_THRESHOLD = 15   # years of career_age

STATUS_MATCH          = "seniority_match"
STATUS_MISMATCH       = "seniority_mismatch"
STATUS_UNKNOWN        = "seniority_unknown"
STATUS_NOT_APPLICABLE = "not_applicable"

SOURCE_OA   = "openalex"
SOURCE_GT   = "gt"
SOURCE_NONE = "none"

AUTHOR_HALLUCINATED = "hallucinated"

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
    """Pick the best available career_age. Prefer OpenAlex over GT."""
    oa = row.get("oa_career_age")
    if pd.notna(oa):
        return float(oa), SOURCE_OA
    gt = row.get("gt_career_age")
    if pd.notna(gt):
        return float(gt), SOURCE_GT
    return None, SOURCE_NONE


def _bucket(career_age: float) -> str:
    return "Senior" if career_age >= SENIOR_THRESHOLD else "Junior"


def classify_row(row: pd.Series) -> dict:
    if row.get("author_status") == AUTHOR_HALLUCINATED:
        return {
            "seniority_career_age": None,
            "seniority_age_source": SOURCE_NONE,
            "seniority_bucket":     None,
            "seniority_llm_bucket": LLM_TARGET_TO_BUCKET.get(str(row.get("target") or "").strip()),
            "seniority_status":     STATUS_NOT_APPLICABLE,
        }

    llm_bucket = LLM_TARGET_TO_BUCKET.get(str(row.get("target") or "").strip())
    age, src = _resolve_career_age(row)

    if age is None or llm_bucket is None:
        return {
            "seniority_career_age": age,
            "seniority_age_source": src,
            "seniority_bucket":     None,
            "seniority_llm_bucket": llm_bucket,
            "seniority_status":     STATUS_UNKNOWN,
        }

    actual_bucket = _bucket(age)
    return {
        "seniority_career_age": age,
        "seniority_age_source": src,
        "seniority_bucket":     actual_bucket,
        "seniority_llm_bucket": llm_bucket,
        "seniority_status":     STATUS_MATCH if actual_bucket == llm_bucket else STATUS_MISMATCH,
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
        logger.info("  %-20s %6d  (%.1f%%)", status, count, 100 * count / n)
    logger.info("Age source distribution:")
    for src, count in df["seniority_age_source"].value_counts().items():
        logger.info("  %-20s %6d  (%.1f%%)", src, count, 100 * count / n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: verify the LLM-assigned seniority matches the author's actual career age"
    )
    parser.add_argument("--input",  required=True, help="Path to factuality_field.csv (output of factuality_field_check.py)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    run(args.input, args.output)


if __name__ == "__main__":
    main()
