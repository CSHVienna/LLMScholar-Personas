"""
run_pipeline.py — Runs the factuality cascade on Semantic Scholar, step 3 of the
funnel (after validity and consistency; before OpenAlex).

  Input: ../results/summary/recommendations_valid.csv, the recommendations of the
  valid calls, written by scripts/3_validity/1_validity.py (run it first).

  1. 1_name.py         every researcher whose name matches → candidates_name.parquet
  2. 2_field.py        ... in the asked field               → candidates_field.parquet
  3. 3_seniority.py    ... with the asked seniority         → candidates_seniority.parquet
  4. 4_author.py       counts per recommendation + one researcher at random
                                                             → factuality_cascade.csv
  5. add_ground_truth() (here) adds, from the researcher picked (only when
     factuality_author = True; empty otherwise):
       author_status     found | hallucinated (the name exists or not)
       field_status      field_match | field_mismatch | not_applicable (name not found)
       seniority_status  seniority_match | seniority_mismatch | not_applicable
       gt_field, gt_gender, gt_career_age, gt_citations, seniority_bucket
       (actual) / seniority_llm_bucket (asked), tiebreak: unique (one
       candidate passes the three checks) | cascade (several, one at random) |
       "" (factuality_author = False)
                                                             → factuality_ss.csv
     and keeps the rows with factuality_author = True and a complete name
     ("name lastname" with 2 words or more)                 → factual_authors.csv
     Only those go on to OpenAlex (scripts/5_openalex_cascade/, scripts/6_enrichment/).
  6. 5_ss_metadata.py  papers, citations and h-index of the picked researcher, from
                       the Semantic Scholar API (ss_paper_count, ss_citation_count,
                       ss_h_index, ss_status), added to both files

Usage (from code/, with PYTHONPATH=.):
  python scripts/4_factuality/run_pipeline.py
  python scripts/4_factuality/run_pipeline.py --skip_name   # reuse candidates_name.parquet (the slow step)
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
from libs.pipeline_common import RECOMMENDATIONS_VALID  # noqa: E402
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


def status(ok: pd.Series, applicable: pd.Series, prefix: str) -> np.ndarray:
    """match / mismatch where applicable, not_applicable elsewhere."""
    return np.where(
        ~applicable, "not_applicable",
        np.where(ok, f"{prefix}_match", f"{prefix}_mismatch"),
    )


def add_ground_truth(cascade_path: Path, parquet_path: str, output_path: Path,
                     factual_path: Path) -> None:
    """Step 5: status columns and gt_* of the picked researcher (see the docstring above)."""
    df = pd.read_csv(cascade_path, low_memory=False)
    name_ok = df["factuality_name"]
    df["author_status"] = np.where(name_ok, "found", "hallucinated")
    df["field_status"] = status(df["factuality_field"], name_ok, "field")
    df["seniority_status"] = status(df["factuality_seniority"], name_ok, "seniority")

    ref = pd.read_parquet(
        parquet_path,
        columns=["Researcher_id", "Field", "Combined_gender", "First_year", "Citations"],
    ).drop_duplicates("Researcher_id")
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
        ~df["factuality_author"], "", np.where(df["n_author"] == 1, "unique", "cascade")
    )
    df.to_csv(output_path, index=False)
    logger.info("Wrote %s (%d rows)", output_path, len(df))

    full_name = (df["name"].fillna("").astype(str) + " "
                 + df["lastname"].fillna("").astype(str)).str.split().str.len() >= 2
    factual = df[df["factuality_author"] & full_name]
    factual.to_csv(factual_path, index=False)
    logger.info("Wrote %s: %d recommendations with factuality_author = True (%.1f%%)",
                factual_path, len(factual), 100 * len(factual) / len(df))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recommendations", default=RECOMMENDATIONS_VALID,
                        help="Recommendations of the valid calls (scripts/3_validity/1_validity.py)")
    parser.add_argument("--parquet", default=config_default("ss_parquet"))
    parser.add_argument("--out_dir", default="../results/summary/cascade")
    parser.add_argument("--skip_name", action="store_true",
                        help="Reuse an existing candidates_name.parquet (the slow step)")
    args = parser.parse_args()

    rec = args.recommendations
    if not Path(rec).exists():
        sys.exit(f"{rec} not found: run scripts/3_validity/1_validity.py first")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

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
    logger.info("── step 5: ground truth of the picked researcher")
    add_ground_truth(out / "factuality_cascade.csv", args.parquet,
                     out / "factuality_ss.csv", out / "factual_authors.csv")
    run_step("5_ss_metadata.py", "--cascade_dir", str(out))


if __name__ == "__main__":
    main()
