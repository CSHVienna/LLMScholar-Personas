"""
run_all.py — Runs everything after the LLM answers are parsed, in order.

  1. openalex_cascade/  every OpenAlex author with each recommended name, their
                        field, and the pick per recommendation, for each variant
                        (random, most_cited)
  2. enrichment/        OpenAlex, location, affiliation and ethnicity data of the
                        picked author, per variant
  3. factuality/        the factuality cascade on Semantic Scholar, then its
                        output with the OpenAlex columns of each variant
  4. pipeline/          validity, consistency, factuality, author stats, socials
                        and one table with every metric of every call (6_calls.csv,
                        with an oa_variant column)
  5. plot_data/         data for specific figures (per variant where it depends on
                        OpenAlex)

Each stage has its own run script; this one only calls them in order.

Usage (from code/, with PYTHONPATH=.):
  python scripts/run_all.py
  python scripts/run_all.py --skip_name        # reuse the Semantic Scholar name matches (slow)
  python scripts/run_all.py --from pipeline    # start at a given stage
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ["openalex_cascade", "enrichment", "factuality", "pipeline", "plot_data"]
VARIANTS = ["random", "most_cited"]


def run(script: str, *args: str) -> None:
    print(f"── {script} {' '.join(args)}", flush=True)
    if subprocess.run([sys.executable, str(HERE / script), *args]).returncode != 0:
        sys.exit(f"{script} failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", choices=STAGES, default=STAGES[0])
    parser.add_argument("--skip_name", action="store_true",
                        help="Reuse the Semantic Scholar name matches of a previous run")
    args = parser.parse_args()
    todo = STAGES[STAGES.index(args.start):]

    if "openalex_cascade" in todo:
        run("openalex_cascade/run_pipeline.py")
    if "enrichment" in todo:
        run("enrichment/run_enrichment.py")
    if "factuality" in todo:
        run("factuality/run_pipeline.py", *(["--skip_name"] if args.skip_name else []))
    if "pipeline" in todo:
        run("pipeline/run_pipeline.py")
    if "plot_data" in todo:
        for v in VARIANTS:
            run("plot_data/build_sankey_flows.py", "--variant", v)
            run("plot_data/build_location_flows.py", "--variant", v)
        run("plot_data/build_gt_field_baseline.py")
        run("plot_data/build_ethnicity_distributions.py")


if __name__ == "__main__":
    main()
