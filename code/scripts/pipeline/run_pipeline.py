"""
run_pipeline.py — Runs the pipeline steps in order.

  1. 1_validity.py      recommendations.csv  → 1_validity/<model>.csv
  2. 2_consistency.py   step 1               → 2_consistency/<model>.csv
  3. 3_factuality.py    step 1 + cascade     → 3_factuality/<model>.csv
  4. 4_author_stats.py  step 3 + OpenAlex    → h/i10/e-index per author ([data].oa_author_stats)
                        (~95 min, but only when the authors of step 3 changed)
  5. 5_socials.py       steps 3 + 4          → 5_socials/<model>.csv
  6. 6_calls_table.py   steps 1, 2, 3, 5     → 6_calls.csv (one table for the notebooks)

Outputs go to ../results/pipeline/ (see common.py).

Usage (from code/, with PYTHONPATH=.):
  python scripts/pipeline/run_pipeline.py
  python scripts/pipeline/run_pipeline.py --skip_name     # reuse the cascade's name matches
  python scripts/pipeline/run_pipeline.py --from_step 5   # re-run from step 5
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = {1: "1_validity.py", 2: "2_consistency.py", 3: "3_factuality.py",
         4: "4_author_stats.py", 5: "5_socials.py", 6: "6_calls_table.py"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from_step", type=int, default=1, choices=list(STEPS))
    parser.add_argument("--skip_name", action="store_true",
                        help="Step 3: reuse the cascade's name matches (its slow step)")
    args = parser.parse_args()

    for step, script in STEPS.items():
        if step < args.from_step:
            continue
        extra = ["--skip_name"] if step == 3 and args.skip_name else []
        print(f"── step {step}: {script}", flush=True)
        if subprocess.run([sys.executable, str(HERE / script), *extra]).returncode != 0:
            sys.exit(f"step {step} failed")


if __name__ == "__main__":
    main()
