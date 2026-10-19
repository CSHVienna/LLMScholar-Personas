"""
run_enrichment.py — Adds OpenAlex, location, affiliation and ethnicity data to
every recommendation.

Steps (each one reads the previous one's output):
  1. openalex.py     recommendations.csv  → factuality_oa.csv
                     the OpenAlex author with that name: oa_id, country,
                     institution, works, citations, ...
  2. location.py     factuality_oa.csv    → factuality_location.csv
                     the LLM's country vs the OpenAlex country
  3. affiliation.py  factuality_location  → factuality_affiliation.csv
                     the LLM's affiliations vs the OpenAlex institutions
  4. ethnicity.py    factuality_affiliation → factuality_full.csv  (final)
                     perceived ethnicity of the recommended name

The factuality cascade (scripts/factuality/) reads these columns from
factuality_full.csv.

Usage (from code/, with PYTHONPATH=.):
  python scripts/enrichment/run_enrichment.py
  python scripts/enrichment/run_enrichment.py --skip_oa   # reuse factuality_oa.csv
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


def run_step(name: str, cmd: list[str]) -> None:
    logger.info("── %s", name)
    t0 = time.time()
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f"{name} failed")
    logger.info("done in %.0fs", time.time() - t0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="../results/summary",
                        help="Folder with recommendations.csv; outputs go here too")
    parser.add_argument("--duckdb", default=config_default("oa_duckdb"))
    parser.add_argument("--skip_oa", action="store_true",
                        help="Reuse the existing factuality_oa.csv (the slow step)")
    args = parser.parse_args()

    r = args.results
    py = sys.executable

    if not args.skip_oa:
        run_step("1/4 openalex", [py, HERE / "openalex.py", "--input", f"{r}/recommendations.csv",
                                  "--output", f"{r}/factuality_oa.csv", "--db_path", args.duckdb,
                                  "--cache", f"{r}/.oa_cache_random.pkl"])  # new cache: the old one holds most-cited picks
    run_step("2/4 location", [py, HERE / "location.py", "--input", f"{r}/factuality_oa.csv",
                              "--output", f"{r}/factuality_location.csv"])
    run_step("3/4 affiliation", [py, HERE / "affiliation.py", "--input", f"{r}/factuality_location.csv",
                                 "--output", f"{r}/factuality_affiliation.csv"])
    run_step("4/4 ethnicity", [py, HERE / "ethnicity.py", "--input", f"{r}/factuality_affiliation.csv",
                               "--ethnicity_lookup", f"{r}/recommendations_with_ethnicity.csv",
                               "--output", f"{r}/factuality_full.csv"])
    logger.info("Output: %s/factuality_full.csv", r)


if __name__ == "__main__":
    main()
