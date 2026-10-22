"""
run_pipeline.py — Runs the OpenAlex cascade, step 5 of the funnel: only for the
recommendations confirmed in Semantic Scholar (factuality_author = True,
summary/cascade/factual_authors.csv, written by scripts/4_factuality/), choose
their OpenAlex author the same way scripts/4_factuality/ chooses the Semantic
Scholar one.

  1. 1_candidates.py     every OpenAlex author with the recommended name
  2. 2_author_fields.py  the field (topics and concepts) and subfields of each of them
  3. 3_pick.py           name → field → seniority → subfield, then one at random
                         → picks.csv

scripts/6_enrichment/ uses the picks.

Usage (from code/, with PYTHONPATH=.):
  python scripts/5_openalex_cascade/run_pipeline.py
  python scripts/5_openalex_cascade/run_pipeline.py --from_step 3
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recommendations", default="../results/summary/cascade/factual_authors.csv",
                        help="Only these are looked up (factuality_author = True)")
    parser.add_argument("--out_dir", default="../results/summary/openalex_cascade")
    parser.add_argument("--from_step", type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()
    out = Path(args.out_dir)
    cand, fields = out / "candidates.parquet", out / "author_fields.parquet"

    steps = [
        (1, ["1_candidates.py", "--recommendations", args.recommendations, "--output", cand]),
        (2, ["2_author_fields.py", "--candidates", cand, "--output", fields]),
        (3, ["3_pick.py", "--recommendations", args.recommendations,
             "--candidates", cand, "--author_fields", fields, "--output", out / "picks.csv"]),
    ]
    for step, (script, *extra) in steps:
        if step < args.from_step:
            continue
        print(f"── step {step}: {script}", flush=True)
        if subprocess.run([sys.executable, str(HERE / script), *map(str, extra)]).returncode != 0:
            sys.exit(f"step {step} failed")


if __name__ == "__main__":
    main()
