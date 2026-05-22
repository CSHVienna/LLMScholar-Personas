"""
run_factuality_pipeline.py — Runs the full factuality pipeline in order.

Steps:
  0.  factuality_author_jw.py    recommendations  → factuality_author_jw.csv
                                  (global JW matching: author_status, gt_career_age)
  0.5 factuality_openalex.py     author_jw        → factuality_oa.csv
                                  (OpenAlex enrichment: oa_country_code, oa_last_institution, …)
  1.  factuality_field_check.py  oa               → factuality_field.csv
                                  (field check: field_status)
  2.  factuality_seniority.py    field            → factuality_seniority.csv
  3.  factuality_location.py     seniority        → factuality_location.csv
  3.5 factuality_affiliation.py  location         → factuality_affiliation.csv
                                  (LLM current_affiliations vs OA institution history)
  4.  factuality_ethnicity.py    affiliation      → factuality_full.csv  (final)

Usage (from code/scripts/factuality/):
  python run_factuality_pipeline.py
  python run_factuality_pipeline.py \\
      --results  ../../../results/results/summary_v2 \\
      --parquet  /data/datasets/LLMScholar-Personas/data/semantic_scholar_data/clean/Researchers_Deduplicated_Genderize_Namsor.parquet \\
      --duckdb   /data/datasets/LLMScholar-Personas/data/openalex_latest.duckdb \\
      [--skip_jw] [--skip_oa] [--skip_field]
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent

DEFAULT_PARQUET = (
    "/data/datasets/LLMScholar-Personas/data/semantic_scholar_data/clean"
    "/Researchers_Deduplicated_Genderize_Namsor.parquet"
)
DEFAULT_DUCKDB = "/data/datasets/LLMScholar-Personas/data/openalex_latest.duckdb"


def run_step(name: str, cmd: list[str]) -> None:
    logger.info("=" * 60)
    logger.info("STEP: %s", name)
    logger.info("=" * 60)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=HERE)
    elapsed = time.time() - t0
    if result.returncode != 0:
        logger.error("FAILED: %s  (%.0fs)", name, elapsed)
        sys.exit(result.returncode)
    logger.info("DONE:  %s  (%.0fs)", name, elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full factuality pipeline")
    parser.add_argument(
        "--results",
        default="../../../results/results/summary_v2",
        help="Path to results/summary_v2 directory",
    )
    parser.add_argument(
        "--parquet",
        default=DEFAULT_PARQUET,
        help="Path to Researchers_Deduplicated parquet (for JW matching)",
    )
    parser.add_argument(
        "--duckdb",
        default=DEFAULT_DUCKDB,
        help="Path to openalex_latest.duckdb (for OpenAlex pre-lookup)",
    )
    parser.add_argument(
        "--skip_jw",
        action="store_true",
        help="Skip step 0 (JW matching) and use existing factuality_author_jw.csv",
    )
    parser.add_argument(
        "--skip_oa",
        action="store_true",
        help="Skip step 0.5 (OpenAlex) and use existing factuality_oa.csv",
    )
    parser.add_argument(
        "--skip_field",
        action="store_true",
        help="Skip step 1 (field check) and use existing factuality_field.csv",
    )
    args = parser.parse_args()

    r = args.results

    oa_cmd = [
        sys.executable,
        "factuality_openalex.py",
        "--input",
        f"{r}/factuality_author_jw.csv",
        "--output",
        f"{r}/factuality_oa.csv",
        "--db_path",
        args.duckdb,
        "--cache",
        f"{r}/.oa_cache.pkl",
    ]

    steps = [
        (
            "0/4    factuality_author_jw  (global JW matching)",
            [
                sys.executable,
                "factuality_author_jw.py",
                "--recommendations",
                f"{r}/recommendations.csv",
                "--parquet",
                args.parquet,
                "--output",
                f"{r}/factuality_author_jw.csv",
            ],
            args.skip_jw,
        ),
        (
            "0.5/4  factuality_openalex  (OpenAlex enrichment)",
            oa_cmd,
            args.skip_oa,
        ),
        (
            "1/4    factuality_field_check  (field match vs gt_field)",
            [
                sys.executable,
                "factuality_field_check.py",
                "--input",
                f"{r}/factuality_oa.csv",
                "--output",
                f"{r}/factuality_field.csv",
            ],
            args.skip_field,
        ),
        (
            "2/4    factuality_seniority",
            [
                sys.executable,
                "factuality_seniority.py",
                "--input",
                f"{r}/factuality_field.csv",
                "--output",
                f"{r}/factuality_seniority.csv",
            ],
            False,
        ),
        (
            "3/5    factuality_location",
            [
                sys.executable,
                "factuality_location.py",
                "--input",
                f"{r}/factuality_seniority.csv",
                "--output",
                f"{r}/factuality_location.csv",
            ],
            False,
        ),
        # PLAN.md Task 2 — new affiliation factuality step,
        # inserted between location and ethnicity (Finding 4).
        (
            "3.5/5  factuality_affiliation",
            [
                sys.executable,
                "factuality_affiliation.py",
                "--input",
                f"{r}/factuality_location.csv",
                "--output",
                f"{r}/factuality_affiliation.csv",
            ],
            False,
        ),
        (
            # PLAN.md Task 2 — the ethnicity step now reads from the affiliation
            # CSV (previously it read from factuality_location.csv).
            "4/5    factuality_ethnicity → factuality_full",
            [
                sys.executable,
                "factuality_ethnicity.py",
                "--input",
                f"{r}/factuality_affiliation.csv",
                "--ethnicity_lookup",
                f"{r}/recommendations_with_ethnicity.csv",
                "--output",
                f"{r}/factuality_full.csv",
            ],
            False,
        ),
    ]

    t_total = time.time()
    for name, cmd, skip in steps:
        if skip:
            logger.info("SKIP:  %s", name)
            continue
        run_step(name, cmd)

    logger.info("=" * 60)
    logger.info("Pipeline complete  (%.0fs total)", time.time() - t_total)
    logger.info("Output: %s/factuality_full.csv", r)


if __name__ == "__main__":
    main()
