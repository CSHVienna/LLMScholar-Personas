"""
run_enrichment.py — Step 7: compares what the LLM said about each recommended
author with the data of the researcher picked in Semantic Scholar and their
OpenAlex author (both from the ground truth, carried by 6_factuality/), and adds
the perceived ethnicity of the picked researcher (ground truth). Only the recommendations with
factuality_author = True have an OpenAlex author; the others get
oa_status = not_searched and location_status / affiliation_status = not_applicable.

Input: summary/cascade/factuality_ss.csv (6_factuality/). Each step reads the
previous one's output:

  1_openalex.py     → factuality_oa.csv           the OpenAlex columns of the picked researcher
  2_location.py     → factuality_location.csv     the prompt's country vs the OpenAlex country
  3_affiliation.py  → factuality_affiliation.csv  the LLM's affiliations vs the OpenAlex institutions
  4_ethnicity.py    → factuality_full.csv (final) perceived ethnicity of the picked researcher

8_metrics/ and 9_plot_data/ read summary/factuality_full.csv.

Usage (from code/, with PYTHONPATH=.):
  python scripts/7_enrichment/run_enrichment.py
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from libs.utils.config import config_default
from libs.utils.logging import setup_logging

logger = setup_logging()
HERE = Path(__file__).resolve().parent


def run_step(name: str, cmd: list) -> None:
    logger.info("── %s", name)
    t0 = time.time()
    if subprocess.run([str(c) for c in cmd]).returncode != 0:
        sys.exit(f"{name} failed")
    logger.info("done in %.0fs", time.time() - t0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="../results/6_factuality/factuality_ss.csv",
                        help="Output of scripts/6_factuality/ (all the valid recommendations)")
    parser.add_argument("--results", default="../results/7_enrichment",
                        help="Outputs go here")
    parser.add_argument("--ethnicity_lookup", default="../results/1_ground_truth/ethnicity/researcher_ethnicity_lookup.csv",
                        help="Ethnicity of every Semantic Scholar researcher (ground truth)")
    parser.add_argument("--duckdb", default=config_default("oa_duckdb"))
    args = parser.parse_args()

    r, py = args.results, sys.executable
    run_step("1/4 openalex", [py, HERE / "1_openalex.py", "--input", args.input,
                              "--output", f"{r}/factuality_oa.csv", "--db_path", args.duckdb])
    run_step("2/4 location", [py, HERE / "2_location.py", "--input", f"{r}/factuality_oa.csv",
                              "--output", f"{r}/factuality_location.csv"])
    run_step("3/4 affiliation", [py, HERE / "3_affiliation.py", "--input", f"{r}/factuality_location.csv",
                                 "--output", f"{r}/factuality_affiliation.csv"])
    run_step("4/4 ethnicity", [py, HERE / "4_ethnicity.py", "--input", f"{r}/factuality_affiliation.csv",
                               "--ethnicity_lookup", args.ethnicity_lookup,
                               "--output", f"{r}/factuality_full.csv"])
    logger.info("Output: %s/factuality_full.csv", r)


if __name__ == "__main__":
    main()
