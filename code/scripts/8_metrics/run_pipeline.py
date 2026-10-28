"""
run_pipeline.py — Step 8: the metrics (after 5_validity/, 6_factuality/ and
7_enrichment/; the graph and the PCA come from 2_graph/).

  1. 1_factuality.py    validity + summary/factuality_full.csv → factuality/<model>.csv
  2. 2_socials.py       step 1 + the ground-truth graph and PCA → socials/<model>.csv
                        (factuality_author = True)
  3. 3_calls_table.py   validity, consistency, steps 1 and 2 → calls.csv (one table
                        for the notebooks)

Outputs go to ../results/8_metrics/ (see libs/pipeline_common.py).

Usage (from code/, with PYTHONPATH=.):
  python scripts/8_metrics/run_pipeline.py
  python scripts/8_metrics/run_pipeline.py --from_step 2   # re-run from step 2
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = {1: "1_factuality.py", 2: "2_socials.py", 3: "3_calls_table.py"}


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
