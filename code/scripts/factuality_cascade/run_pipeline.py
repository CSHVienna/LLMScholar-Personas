"""
run_pipeline.py — Runs the factuality_cascade steps in order.

  1. factuality_name.py       recommendations   → candidates_name.parquet
  2. factuality_field.py      candidates_name   → candidates_field.parquet
  3. factuality_seniority.py  candidates_name   → candidates_seniority.parquet
  4. factuality_author.py     all of the above  → factuality_cascade.csv
  5. build_factuality_full.py  cascade + old factuality_full.csv
                               → cascade/factuality_full.csv (format of the metrics)

Usage (from code/, with PYTHONPATH=.):
  python scripts/factuality_cascade/run_pipeline.py
  python scripts/factuality_cascade/run_pipeline.py \\
      --recommendations ../results/summary/recommendations.csv \\
      --out_dir         ../results/summary/cascade \\
      --old_full        ../results/summary/factuality_full.csv \\
      [--skip_name]
"""

import argparse
import subprocess
import sys
from pathlib import Path

from libs.utils.config import config_default
from libs.utils.logging import setup_logging

logger = setup_logging()
HERE = Path(__file__).resolve().parent


def run_step(script: str, *args: str) -> None:
    logger.info("── %s", script)
    result = subprocess.run([sys.executable, str(HERE / script), *args])
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recommendations",
                        default="../results/summary/recommendations.csv")
    parser.add_argument("--parquet", default=config_default("ss_parquet"))
    parser.add_argument("--out_dir", default="../results/summary/cascade")
    parser.add_argument("--old_full", default="../results/summary/factuality_full.csv",
                        help="Old pipeline output; only read, for the OpenAlex columns")
    parser.add_argument("--skip_name", action="store_true",
                        help="Reuse an existing candidates_name.parquet (the slow step)")
    args = parser.parse_args()

    rec = args.recommendations
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_name:
        run_step("factuality_name.py", "--recommendations", rec,
                 "--parquet", args.parquet,
                 "--output", str(out / "candidates_name.parquet"))
    run_step("factuality_field.py", "--recommendations", rec,
             "--input", str(out / "candidates_name.parquet"),
             "--output", str(out / "candidates_field.parquet"))
    run_step("factuality_seniority.py", "--recommendations", rec,
             "--input", str(out / "candidates_name.parquet"),
             "--output", str(out / "candidates_seniority.parquet"))
    run_step("factuality_author.py", "--recommendations", rec,
             "--cascade_dir", str(out),
             "--output", str(out / "factuality_cascade.csv"))
    run_step("build_factuality_full.py",
             "--cascade", str(out / "factuality_cascade.csv"),
             "--old", args.old_full,
             "--parquet", args.parquet,
             "--output", str(out / "factuality_full.csv"))


if __name__ == "__main__":
    main()
