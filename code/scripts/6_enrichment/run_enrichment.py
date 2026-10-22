"""
run_enrichment.py — Step 6 of the funnel (after scripts/5_openalex_cascade/): adds
OpenAlex, location and affiliation data to the recommendations confirmed in
Semantic Scholar (factuality_author = True), and the perceived ethnicity to all.

Input: summary/cascade/factuality_ss.csv (scripts/4_factuality/, all the valid
recommendations). Rows with factuality_author = False keep their Semantic Scholar
columns and get oa_status = not_searched, location_status and
affiliation_status = not_applicable. Each step reads the previous one's output:

  1_openalex.py     cascade/factuality_ss.csv → factuality_oa.csv
                    the OpenAlex author picked by the cascade (picks.csv): oa_id,
                    country, institution, works, citations
  2_location.py     → factuality_location.csv
                    the prompt's country vs the OpenAlex country
  3_affiliation.py  → factuality_affiliation.csv
                    the LLM's affiliations vs the OpenAlex institutions
  4_ethnicity.py    → factuality_full.csv  (final)
                    perceived ethnicity of the recommended name

scripts/7_metrics/ and scripts/8_plot_data/ read summary/factuality_full.csv.

Usage (from code/, with PYTHONPATH=.):
  python scripts/6_enrichment/run_enrichment.py
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
    parser.add_argument("--input", default="../results/summary/cascade/factuality_ss.csv",
                        help="Output of scripts/4_factuality/ (all the valid recommendations)")
    parser.add_argument("--results", default="../results/summary",
                        help="Folder with recommendations_with_ethnicity.csv; outputs go here too")
    parser.add_argument("--cascade", default="../results/summary/openalex_cascade",
                        help="Output folder of scripts/5_openalex_cascade/")
    parser.add_argument("--duckdb", default=config_default("oa_duckdb"))
    args = parser.parse_args()

    r, c, py = args.results, Path(args.cascade), sys.executable
    run_step("1/4 openalex", [py, HERE / "1_openalex.py", "--input", args.input,
                              "--picks", c / "picks.csv", "--candidates", c / "candidates.parquet",
                              "--output", f"{r}/factuality_oa.csv", "--db_path", args.duckdb])
    run_step("2/4 location", [py, HERE / "2_location.py", "--input", f"{r}/factuality_oa.csv",
                              "--output", f"{r}/factuality_location.csv"])
    run_step("3/4 affiliation", [py, HERE / "3_affiliation.py", "--input", f"{r}/factuality_location.csv",
                                 "--output", f"{r}/factuality_affiliation.csv"])
    run_step("4/4 ethnicity", [py, HERE / "4_ethnicity.py", "--input", f"{r}/factuality_affiliation.csv",
                               "--ethnicity_lookup", f"{r}/recommendations_with_ethnicity.csv",
                               "--output", f"{r}/factuality_full.csv"])
    logger.info("Output: %s/factuality_full.csv", r)


if __name__ == "__main__":
    main()
