"""
run_all.py — Runs everything after the LLM answers are parsed (scripts/1_prompting/,
2_llmcaller/), in the order of the funnel: each stage only passes on what
survived the previous one.

  3_validity/          validity, refusal, duplicates of every call; consistency
                       of the valid ones
  4_factuality/        the cascade on Semantic Scholar, only for the valid calls
                       (name → field → seniority → author)
  5_openalex_cascade/  the OpenAlex author, only for factuality_author = True
  6_enrichment/        their country, institution, works and citations from
                       OpenAlex; perceived ethnicity for all
  7_metrics/           factuality and bias_location per recommendation, author
                       stats, socials (only factuality_author = True) and one
                       table with every metric of every call (calls.csv)
  8_plot_data/         data for specific figures

Each folder has its own scripts; this one only calls them in order.

Usage (from code/, with PYTHONPATH=.):
  python scripts/run_all.py
  python scripts/run_all.py --skip_name             # reuse the Semantic Scholar name matches (slow)
  python scripts/run_all.py --from 7_metrics        # start at a given folder
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ["3_validity", "4_factuality", "5_openalex_cascade", "6_enrichment", "7_metrics", "8_plot_data"]


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

    if "3_validity" in todo:
        run("3_validity/1_validity.py")
        run("3_validity/2_consistency.py")
    if "4_factuality" in todo:
        run("4_factuality/run_pipeline.py", *(["--skip_name"] if args.skip_name else []))
    if "5_openalex_cascade" in todo:
        run("5_openalex_cascade/run_pipeline.py")
    if "6_enrichment" in todo:
        run("6_enrichment/run_enrichment.py")
    if "7_metrics" in todo:
        run("7_metrics/run_pipeline.py")
    if "8_plot_data" in todo:
        run("8_plot_data/build_sankey_flows.py")
        run("8_plot_data/build_location_flows.py")
        run("8_plot_data/build_gt_field_baseline.py")
        run("8_plot_data/build_ethnicity_distributions.py")


if __name__ == "__main__":
    main()
