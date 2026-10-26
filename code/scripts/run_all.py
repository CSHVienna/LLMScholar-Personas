"""
run_all.py — Runs everything, in order.

  1_ground_truth/  every Semantic Scholar researcher with their OpenAlex author
                   (through their papers, Jaro-Winkler; the cascade by name
                   otherwise) and its metadata → ground_truth.parquet
  2_graph/         author stats, coauthorship graph and similarity PCA over the
                   whole ground truth
  (3_prompting/ and 4_llmcaller/ build the prompts and get the LLM answers; they
   are run by hand, see scripts/README.md)
  5_validity/      validity, refusal, duplicates of every call; consistency of
                   the valid ones
  6_factuality/    the cascade on Semantic Scholar, only for the valid calls
                   (name → field → seniority → author); the picked researcher
                   brings their ground-truth data
  7_enrichment/    the LLM's answer vs that data: country, affiliation; ethnicity
  8_metrics/       factuality and bias_location per recommendation, socials (only
                   factuality_author = True) and one table with every metric of
                   every call (calls.csv)
  9_plot_data/     data for specific figures

Steps 1 and 2 do not depend on the LLM answers: once built, start at 5_validity.

Usage (from code/, with PYTHONPATH=.):
  python scripts/run_all.py
  python scripts/run_all.py --from 5_validity      # the ground truth and the graph already built
  python scripts/run_all.py --skip_name            # reuse the Semantic Scholar name matches (slow)
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ["1_ground_truth", "2_graph", "5_validity", "6_factuality", "7_enrichment", "8_metrics", "9_plot_data"]


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

    if "1_ground_truth" in todo:
        run("1_ground_truth/run_pipeline.py")
    if "2_graph" in todo:
        run("2_graph/run_pipeline.py")
    if "5_validity" in todo:
        run("5_validity/1_validity.py")
        run("5_validity/2_consistency.py")
    if "6_factuality" in todo:
        run("6_factuality/run_pipeline.py", *(["--skip_name"] if args.skip_name else []))
    if "7_enrichment" in todo:
        run("7_enrichment/run_enrichment.py")
    if "8_metrics" in todo:
        run("8_metrics/run_pipeline.py")
    if "9_plot_data" in todo:
        run("9_plot_data/build_sankey_flows.py")
        run("9_plot_data/build_location_flows.py")
        run("9_plot_data/build_gt_field_baseline.py")
        run("9_plot_data/build_ethnicity_distributions.py")


if __name__ == "__main__":
    main()
