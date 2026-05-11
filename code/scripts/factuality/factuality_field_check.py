"""
factuality_field_check.py — Step 2 of the factuality pipeline.

Reads the output of factuality_author.py and decides whether each verified
author actually belongs to the field the LLM was asked about.

Decision per row:
  author_status = hallucinated      → field_status = not_applicable
  gt_field present                  → compare LLM `field` vs `gt_field`
  otherwise                         → field_status = field_unknown
                                       (author only in OpenAlex; concepts not
                                        materialized in this run)

Output columns added to factuality_author.csv:
  field_status        {field_match | field_mismatch | field_unknown | not_applicable}
  field_check_source  {gt | openalex_concepts | none}
  field_evidence      string compared against (gt_field or top concepts)

Usage (from code/scripts/):
  python factuality_field_check.py \\
      --input  ../../../results/summary/factuality_author.csv \\
      --output ../../../results/summary/factuality_field.csv
"""

import argparse
import logging
import os
import re

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

STATUS_MATCH          = "field_match"
STATUS_MISMATCH       = "field_mismatch"
STATUS_UNKNOWN        = "field_unknown"
STATUS_NOT_APPLICABLE = "not_applicable"

CHECK_GT       = "gt"
CHECK_CONCEPTS = "openalex_concepts"
CHECK_NONE     = "none"

AUTHOR_HALLUCINATED = "hallucinated"

# Translates LLM field values (ES / DE) → canonical English before comparison
FIELD_TRANSLATION = {
    "Biología":                   "Biology",
    "Física":                     "Physics",
    "Ciencias de la computación": "Computer Science",
    "Sociología":                 "Sociology",
    "Psicología":                 "Psychology",
    "Matemáticas":                "Mathematics",
    "Biologie":                   "Biology",
    "Physik":                     "Physics",
    "Informatik":                 "Computer Science",
    "Soziologie":                 "Sociology",
    "Psychologie":                "Psychology",
    "Mathematik":                 "Mathematics",
}


# ── Normalisation ──────────────────────────────────────────────────────────────

def _norm_field(s: str) -> str:
    """Translate ES/DE→EN, then lowercase, replace _ with space, collapse whitespace."""
    if not isinstance(s, str):
        return ""
    s = FIELD_TRANSLATION.get(s.strip(), s)
    s = s.lower().replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


# ── Per-row decision ───────────────────────────────────────────────────────────

def _check_gt(llm_field: str, gt_field: str) -> tuple[str, str]:
    """Compare LLM-requested field against the GT field that matched."""
    if not gt_field:
        return STATUS_UNKNOWN, ""
    match = _norm_field(llm_field) == _norm_field(gt_field)
    return (STATUS_MATCH if match else STATUS_MISMATCH), gt_field


def classify_row(row: pd.Series) -> tuple[str, str, str]:
    """Return (field_status, field_check_source, field_evidence)."""
    if row.get("author_status") == AUTHOR_HALLUCINATED:
        return STATUS_NOT_APPLICABLE, CHECK_NONE, ""

    llm_field = str(row.get("field") or "").strip()
    if not llm_field:
        return STATUS_UNKNOWN, CHECK_NONE, ""

    gt_field = str(row.get("gt_field") or "").strip() if pd.notna(row.get("gt_field")) else ""
    if gt_field:
        status, evidence = _check_gt(llm_field, gt_field)
        return status, CHECK_GT, evidence

    return STATUS_UNKNOWN, CHECK_NONE, ""


# ── Main ───────────────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str) -> None:
    logger.info("Loading: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    logger.info("Rows: %d", len(df))

    statuses, sources, evidences = [], [], []
    for _, row in df.iterrows():
        status, src, evidence = classify_row(row)
        statuses.append(status)
        sources.append(src)
        evidences.append(evidence)

    df["field_status"]       = statuses
    df["field_check_source"] = sources
    df["field_evidence"]     = evidences

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows → %s", len(df), output_path)

    n = len(df)
    logger.info("Field status distribution:")
    for status, count in df["field_status"].value_counts().items():
        logger.info("  %-18s %6d  (%.1f%%)", status, count, 100 * count / n)
    logger.info("Check source distribution:")
    for src, count in df["field_check_source"].value_counts().items():
        logger.info("  %-18s %6d  (%.1f%%)", src, count, 100 * count / n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: verify the LLM-recommended author belongs to the requested field"
    )
    parser.add_argument("--input",  required=True, help="Path to factuality_author.csv (output of factuality_author.py)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    run(args.input, args.output)


if __name__ == "__main__":
    main()
