"""
factuality_seniority_oa.py — Junior/Senior verdict computed from OpenAlex,
parallel to the Semantic Scholar one already in factuality_full.csv.

Why a second column instead of replacing the first: the two sources disagree
about the same authors, and neither is a drop-in for the other, so they stay
side by side and never mix.

    career age = reference_year - first publication year

is the same formula in both; only the source of "first publication year"
changes. Semantic Scholar's `First_year` is the first year *inside its own
6-field corpus*, which for the matched population runs 17.7 years later on
average (median 14, later in 78.6% of cases) than OpenAlex's, because SS sees
~2 papers over ~10 years of a ~38-year career. That truncation pushes authors
toward Junior:

    le pides Junior -> acierta 21.0% segun SS, 3.8% segun OpenAlex
    le pides Senior -> acierta 86.2% segun SS, 97.9% segun OpenAlex

The conclusion survives either source — asked for juniors, the models return
seniors — which is the point of publishing both. The magnitudes differ 5x, and
the SS figure is the biased one.

Coverage is the trade-off: `researcher_id` is present for 100% of factual
recommendations, `oa_first_pub_year` for 81%.

Output: <results_dir>/factualities/tables/seniority_oa.csv, one row per
recommendation, joinable back on the call keys plus oa_id.

Usage (from code/, with PYTHONPATH=.):
  python scripts/legacy/factuality_seniority_oa.py
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

from libs.metrics.constants import CALL_KEYS, LLM_TARGET_TO_BUCKET, REFERENCE_YEAR
from libs.utils.config import get_results_path
from libs.utils.logging import setup_logging, log_value_counts

warnings.filterwarnings("ignore")
logger = setup_logging()

# Same thresholds as factuality_seniority.py, imported rather than restated so
# the two columns can never drift apart on the bucket boundaries.
from scripts.legacy.factuality_seniority import (  # noqa: E402
    JUNIOR_MAX,
    SENIOR_MIN,
    SOURCE_NONE,
    STATUS_MATCH,
    STATUS_MISMATCH,
    STATUS_NOT_APPLICABLE,
    STATUS_UNKNOWN,
)

SOURCE_OA = "openalex"

# Fixed for reproducibility, matching factuality_author_jw.REFERENCE_YEAR: a
# moving current year would shuffle authors across the Junior/Senior boundary
# every time the pipeline runs.

NEEDED = CALL_KEYS + [
    "valid_flag",
    "target",
    "author_status",
    "oa_id",
    "oa_first_pub_year",
]

OUT_COLS = [
    "seniority_oa_career_age",
    "seniority_oa_age_source",
    "seniority_oa_bucket",
    "seniority_oa_llm_bucket",
    "seniority_oa_status",
]


def _bucket(career_age: float) -> str | None:
    if career_age <= JUNIOR_MAX:
        return "Junior"
    if career_age >= SENIOR_MIN:
        return "Senior"
    return None  # 11-19 years: unclassifiable, same as the SS column


def classify(df: pd.DataFrame) -> pd.DataFrame:
    first = pd.to_numeric(df["oa_first_pub_year"], errors="coerce")
    # OpenAlex publication years carry disambiguation errors: this column runs
    # from 1476 to 2026 in the snapshot. Outside a plausible window the age is
    # unknown, not extreme.
    first = first.where((first >= 1900) & (first <= REFERENCE_YEAR))
    age = (REFERENCE_YEAR - first).where(first.notna())

    llm_bucket = df["target"].astype(str).str.strip().map(LLM_TARGET_TO_BUCKET)
    actual = age.map(lambda a: _bucket(a) if pd.notna(a) else None)

    out = pd.DataFrame(index=df.index)
    out["seniority_oa_career_age"] = age
    out["seniority_oa_age_source"] = age.notna().map({True: SOURCE_OA, False: SOURCE_NONE})
    out["seniority_oa_bucket"] = actual
    out["seniority_oa_llm_bucket"] = llm_bucket

    status = pd.Series(STATUS_UNKNOWN, index=df.index, dtype=object)
    # An author nobody matched has no seniority to check, exactly as the SS
    # column treats them.
    status[df["author_status"].ne("found") & df["oa_id"].isna()] = STATUS_NOT_APPLICABLE
    comparable = actual.notna() & llm_bucket.notna()
    status[comparable & (actual == llm_bucket)] = STATUS_MATCH
    status[comparable & (actual != llm_bucket)] = STATUS_MISMATCH
    out["seniority_oa_status"] = status
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=None)
    parser.add_argument("--input", default=None, help="factuality_full.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = get_results_path(path=args.results)
    src = Path(args.input) if args.input else results / "summary" / "factuality_full.csv"
    dst = (
        Path(args.output)
        if args.output
        else results / "factualities" / "tables" / "seniority_oa.csv"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Reading %s", src)
    df = pd.read_csv(src, usecols=NEEDED, low_memory=False)
    logger.info("Rows: %d", len(df))

    out = pd.concat([df[CALL_KEYS + ["oa_id"]], classify(df)], axis=1)
    log_value_counts(out, "seniority_oa_status", logger=logger)
    log_value_counts(out, "seniority_oa_bucket", logger=logger)

    out.to_csv(dst, index=False)
    logger.info("Wrote %s (%d rows)", dst, len(out))


if __name__ == "__main__":
    main()
