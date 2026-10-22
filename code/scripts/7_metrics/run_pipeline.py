"""
run_pipeline.py — Runs the metrics, the last step of the funnel (after
scripts/3_validity/, 4_factuality/, 5_openalex_cascade/ and 6_enrichment/).

  1. 1_factuality.py    validity + summary/factuality_full.csv → factuality/<model>.csv
  2. 2_author_stats.py  step 1 + OpenAlex   → h/i10/e-index per author ([data].oa_author_stats)
                        (~95 min, but only when step 1 brings new authors)
  3. 3_socials.py       steps 1 + 2         → socials/<model>.csv (factuality_author = True)
  4. 4_calls_table.py   validity, consistency, steps 1 and 3 → calls.csv (one table
                        for the notebooks)

Outputs go to ../results/pipeline/ (see libs/pipeline_common.py).

Usage (from code/, with PYTHONPATH=.):
  python scripts/7_metrics/run_pipeline.py
  python scripts/7_metrics/run_pipeline.py --from_step 3   # re-run from step 3
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = {1: "1_factuality.py", 2: "2_author_stats.py", 3: "3_socials.py", 4: "4_calls_table.py"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from_step", type=int, default=1, choices=list(STEPS))
    args = parser.parse_args()

    for step, script in STEPS.items():
        if step < args.from_step:
            continue
        print(f"── step {step}: {script}", flush=True)
        if subprocess.run([sys.executable, str(HERE / script)]).returncode != 0:
            sys.exit(f"step {step} failed")


if __name__ == "__main__":
    main()
