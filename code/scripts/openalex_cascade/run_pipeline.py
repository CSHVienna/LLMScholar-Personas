"""
run_pipeline.py — Runs the OpenAlex cascade: choose the OpenAlex author of each
recommendation the same way scripts/factuality/ chooses the Semantic Scholar one.

  1. 1_candidates.py     every OpenAlex author with the recommended name
  2. 2_author_fields.py  the field (topics and concepts) and subfields of each of them
  3. 3_pick.py           name → field → seniority → subfield, then the last tie-break,
                         once per variant: picks_random.csv, picks_most_cited.csv

scripts/enrichment/ uses the picks.

Usage (from code/, with PYTHONPATH=.):
  python scripts/openalex_cascade/run_pipeline.py
  python scripts/openalex_cascade/run_pipeline.py --from_step 3
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = ["random", "most_cited"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default="../results/summary/openalex_cascade")
    parser.add_argument("--from_step", type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()
    out = Path(args.out_dir)
    cand, fields = out / "candidates.parquet", out / "author_fields.parquet"

    steps = [
        (1, ["1_candidates.py", "--output", cand]),
        (2, ["2_author_fields.py", "--candidates", cand, "--output", fields]),
        # Step 3 once per variant: the last tie-break, at random or the most cited.
        *[(3, ["3_pick.py", "--candidates", cand, "--author_fields", fields,
               "--output", out / f"picks_{v}.csv", "--tiebreak", v]) for v in VARIANTS],
    ]
    for step, (script, *extra) in steps:
        if step < args.from_step:
            continue
        print(f"── step {step}: {script} {' '.join(map(str, extra[-2:]))}", flush=True)
        if subprocess.run([sys.executable, str(HERE / script), *map(str, extra)]).returncode != 0:
            sys.exit(f"step {step} failed")


if __name__ == "__main__":
    main()
