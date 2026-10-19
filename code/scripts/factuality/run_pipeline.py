"""
run_pipeline.py — Runs the factuality cascade, from the recommendations to
cascade/factuality_full.csv.

  1. 1_name.py               every researcher whose name matches → candidates_name.parquet
  2. 2_field.py              ... in the asked field               → candidates_field.parquet
  3. 3_seniority.py          ... with the asked seniority         → candidates_seniority.parquet
  4. 4_author.py             counts per recommendation + one researcher at random
                                                                   → factuality_cascade.csv
  5. build_full() (here)     → factuality_full.csv, the format the metrics read:
     - copies the OpenAlex, location, affiliation and ethnicity columns from
       ../results/summary/factuality_full.csv (written by scripts/enrichment/;
       same rows in the same order, checked on name and lastname; only read)
     - turns the True/False of the cascade into the old status columns:
         author_status     found | hallucinated
         field_status      field_match | field_mismatch | not_applicable (name not found)
         seniority_status  seniority_match | seniority_mismatch | not_applicable (name not found)
     - recomputes location_status and affiliation_status, not_applicable when
       the name was not found
     - adds gt_field, gt_gender, gt_career_age, gt_citations of the picked
       researcher, seniority_bucket (actual) / seniority_llm_bucket (asked)
     - adds tiebreak: unique (one candidate) | cascade (several) | "" (not found)

Usage (from code/, with PYTHONPATH=.):
  python scripts/factuality/run_pipeline.py
  python scripts/factuality/run_pipeline.py --skip_name   # reuse candidates_name.parquet (the slow step)
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import seniority_bucket  # noqa: E402

from libs.metrics.constants import LLM_TARGET_TO_BUCKET, REFERENCE_YEAR  # noqa: E402
from libs.utils.config import config_default  # noqa: E402
from libs.utils.logging import setup_logging  # noqa: E402

logger = setup_logging()
HERE = Path(__file__).resolve().parent


def run_step(script: str, *args: str) -> None:
    logger.info("── %s", script)
    result = subprocess.run([sys.executable, str(HERE / script), *args])
    if result.returncode != 0:
        sys.exit(result.returncode)


# ── Step 5 ────────────────────────────────────────────────────────────────────

AFFILIATION_THRESHOLD = 85  # same as enrichment/affiliation.py

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


def build_full(cascade_path: str, old_path: str, parquet_path: str, output_path: str) -> None:
    """Step 5: the cascade in the format the metrics expect (see the docstring above)."""
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recommendations",
                        default="../results/summary/recommendations.csv")
    parser.add_argument("--parquet", default=config_default("ss_parquet"))
    parser.add_argument("--out_dir", default="../results/summary/cascade")
    parser.add_argument("--old_full", default="../results/summary/factuality_full.csv",
                        help="Output of scripts/enrichment/; only read, for its OpenAlex columns")
    parser.add_argument("--skip_name", action="store_true",
                        help="Reuse an existing candidates_name.parquet (the slow step)")
    args = parser.parse_args()

    rec = args.recommendations
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "factuality_full.csv").resolve() == Path(args.old_full).resolve():
        parser.error("--out_dir would overwrite the enrichment factuality_full.csv")

    if not args.skip_name:
        run_step("1_name.py", "--recommendations", rec,
                 "--parquet", args.parquet,
                 "--output", str(out / "candidates_name.parquet"))
    run_step("2_field.py", "--recommendations", rec,
             "--input", str(out / "candidates_name.parquet"),
             "--output", str(out / "candidates_field.parquet"))
    run_step("3_seniority.py", "--recommendations", rec,
             "--input", str(out / "candidates_name.parquet"),
             "--output", str(out / "candidates_seniority.parquet"))
    run_step("4_author.py", "--recommendations", rec,
             "--cascade_dir", str(out),
             "--output", str(out / "factuality_cascade.csv"))
    logger.info("── step 5: factuality_full.csv")
    build_full(str(out / "factuality_cascade.csv"), args.old_full, args.parquet,
               str(out / "factuality_full.csv"))


if __name__ == "__main__":
    main()
