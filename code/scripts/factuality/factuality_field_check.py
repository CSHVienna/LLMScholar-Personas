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

Usage (from code/, with PYTHONPATH=.):
  python scripts/factuality/factuality_field_check.py \\
      --input  ../results/summary/factuality_author_jw.csv \\
      --output ../results/summary/factuality_field.csv
"""

import argparse
import re

import pandas as pd

from libs.metrics.constants import (
    FACTUALITY_AUTHOR_HALLUCINATED as AUTHOR_HALLUCINATED,
    factuality_status_flags,
)
from libs.utils.cli import add_io_args
from libs.utils.ios import read_input_csv, write_output_csv
from libs.utils.logging import log_value_counts, setup_logging

logger = setup_logging()

# ── Constants ──────────────────────────────────────────────────────────────────

_STATUS = factuality_status_flags("field")
STATUS_MATCH = _STATUS["MATCH"]
STATUS_MISMATCH = _STATUS["MISMATCH"]
STATUS_UNKNOWN = _STATUS["UNKNOWN"]
STATUS_NOT_APPLICABLE = _STATUS["NOT_APPLICABLE"]

CHECK_GT = "gt"
CHECK_CONCEPTS = "openalex_concepts"
CHECK_NONE = "none"

# Translates LLM field values (ES / DE) → canonical English before comparison
FIELD_TRANSLATION = {
    "Biología": "Biology",
    "Física": "Physics",
    "Ciencias de la computación": "Computer Science",
    "Sociología": "Sociology",
    "Psicología": "Psychology",
    "Matemáticas": "Mathematics",
    "Biologie": "Biology",
    "Physik": "Physics",
    "Informatik": "Computer Science",
    "Soziologie": "Sociology",
    "Psychologie": "Psychology",
    "Mathematik": "Mathematics",
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

    gt_field = (
        str(row.get("gt_field") or "").strip() if pd.notna(row.get("gt_field")) else ""
    )
    if gt_field:
        status, evidence = _check_gt(llm_field, gt_field)
        return status, CHECK_GT, evidence

    return STATUS_UNKNOWN, CHECK_NONE, ""


# ── Main ───────────────────────────────────────────────────────────────────────


def run(input_path: str, output_path: str) -> None:
    df = read_input_csv(input_path, logger=logger)

    statuses, sources, evidences = [], [], []
    for _, row in df.iterrows():
        status, src, evidence = classify_row(row)
        statuses.append(status)
        sources.append(src)
        evidences.append(evidence)

    df["field_status"] = statuses
    df["field_check_source"] = sources
    df["field_evidence"] = evidences

    write_output_csv(df, output_path, logger=logger)
    log_value_counts(df, "field_status", title="Field status distribution", logger=logger)
    log_value_counts(df, "field_check_source", title="Check source distribution", logger=logger)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: verify the LLM-recommended author belongs to the requested field"
    )
    add_io_args(
        parser,
        input_help="Path to factuality_author.csv (output of factuality_author.py)",
    )
    args = parser.parse_args()

    run(args.input, args.output)


if __name__ == "__main__":
    main()
