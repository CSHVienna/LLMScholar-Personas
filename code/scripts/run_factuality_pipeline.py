"""
run_factuality_pipeline.py — Runs the full factuality pipeline in order.

Steps:
  0. factuality_author_jw.py  recommendations → factuality_author_jw.csv
                               (global JW matching: author_status, gt_career_age)
  1. factuality_field.py      author_jw       → factuality_field.csv
                               (field-specific matching: factuality_status)
  2. factuality_seniority.py  field           → factuality_seniority.csv
  3. factuality_location.py   seniority       → factuality_location.csv
  4. factuality_ethnicity.py  location        → factuality_full.csv  (final)

Usage (from code/scripts/):
  python run_factuality_pipeline.py
  python run_factuality_pipeline.py \\
      --results  ../../results/summary \\
      --data_dir /data/datasets/LLMScholar-Personas/data/semantic_scholar_data/all \\
      --parquet  /data/datasets/LLMScholar-Personas/data/semantic_scholar_data/clean/Researchers_Deduplicated_Genderize_Namsor.parquet \\
      [--skip_jw]   # skip step 0 if factuality_author_jw.csv already exists
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
    parser.add_argument("--results",  default="../../results/summary",
                        help="Path to results/summary directory")
    parser.add_argument("--data_dir", default="/data/datasets/LLMScholar-Personas/data/semantic_scholar_data/all",
                        help="(unused) kept for backwards compatibility")
    parser.add_argument("--parquet",  default=DEFAULT_PARQUET,
                        help="Path to Researchers_Deduplicated parquet (for JW matching)")
    parser.add_argument("--skip_jw",    action="store_true",
                        help="Skip step 0 (JW matching) and use existing factuality_author_jw.csv")
    parser.add_argument("--skip_field", action="store_true",
                        help="Skip step 1 (field check) and use existing factuality_field.csv")
    args = parser.parse_args()

    r = args.results

    steps = [
        (
            "0/4  factuality_author_jw  (global matching)",
            [sys.executable, "factuality_author_jw.py",
             "--recommendations", f"{r}/recommendations.csv",
             "--parquet",         args.parquet,
             "--output",          f"{r}/factuality_author_jw.csv"],
            args.skip_jw,
        ),
        (
            "1/4  factuality_field_check  (field match vs gt_field)",
            [sys.executable, "factuality_field_check.py",
             "--input",  f"{r}/factuality_author_jw.csv",
             "--output", f"{r}/factuality_field.csv"],
            args.skip_field,
        ),
        (
            "2/4  factuality_seniority",
            [sys.executable, "factuality_seniority.py",
             "--input",  f"{r}/factuality_field.csv",
             "--output", f"{r}/factuality_seniority.csv"],
            False,
        ),
        (
            "3/4  factuality_location",
            [sys.executable, "factuality_location.py",
             "--input",  f"{r}/factuality_seniority.csv",
             "--output", f"{r}/factuality_location.csv"],
            False,
        ),
        (
            "4/4  factuality_ethnicity → factuality_full",
            [sys.executable, "factuality_ethnicity.py",
             "--input",            f"{r}/factuality_location.csv",
             "--ethnicity_lookup", f"{r}/recommendations_with_ethnicity.csv",
             "--output",           f"{r}/factuality_full.csv"],
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
