"""
run_enrichment.py — Adds OpenAlex, location, affiliation and ethnicity data to
every recommendation, once per OpenAlex variant.

A variant is how scripts/openalex_cascade/ broke the last tie among the
OpenAlex authors that pass the same checks ("random" or "most_cited"). For each
variant, each step reads the previous one's output:

  1. openalex.py     recommendations.csv     → factuality_oa_<variant>.csv
                     the OpenAlex author picked by the cascade: oa_id, country,
                     institution, works, citations, oa_n_* counts
  2. location.py     → factuality_location_<variant>.csv
                     the prompt's country vs the OpenAlex country
  3. affiliation.py  → factuality_affiliation_<variant>.csv
                     the LLM's affiliations vs the OpenAlex institutions
  4. ethnicity.py    → factuality_full_<variant>.csv  (final)
                     perceived ethnicity of the recommended name

The factuality cascade (scripts/factuality/) reads factuality_full_<variant>.csv.

Usage (from code/, with PYTHONPATH=.):
  python scripts/enrichment/run_enrichment.py
  python scripts/enrichment/run_enrichment.py --variants random
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
VARIANTS = ["random", "most_cited"]


def run_step(name: str, cmd: list) -> None:
    logger.info("── %s", name)
    t0 = time.time()
    if subprocess.run([str(c) for c in cmd]).returncode != 0:
        sys.exit(f"{name} failed")
    logger.info("done in %.0fs", time.time() - t0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="../results/summary",
                        help="Folder with recommendations.csv; outputs go here too")
    parser.add_argument("--cascade", default="../results/summary/openalex_cascade",
                        help="Output folder of scripts/openalex_cascade/")
    parser.add_argument("--duckdb", default=config_default("oa_duckdb"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    args = parser.parse_args()

    r, c, py = args.results, Path(args.cascade), sys.executable
    for v in args.variants:
        run_step(f"{v} 1/4 openalex", [py, HERE / "openalex.py", "--input", f"{r}/recommendations.csv",
                                       "--picks", c / f"picks_{v}.csv", "--candidates", c / "candidates.parquet",
                                       "--output", f"{r}/factuality_oa_{v}.csv", "--db_path", args.duckdb])
        run_step(f"{v} 2/4 location", [py, HERE / "location.py", "--input", f"{r}/factuality_oa_{v}.csv",
                                       "--output", f"{r}/factuality_location_{v}.csv"])
        run_step(f"{v} 3/4 affiliation", [py, HERE / "affiliation.py", "--input", f"{r}/factuality_location_{v}.csv",
                                          "--output", f"{r}/factuality_affiliation_{v}.csv"])
        run_step(f"{v} 4/4 ethnicity", [py, HERE / "ethnicity.py", "--input", f"{r}/factuality_affiliation_{v}.csv",
                                        "--ethnicity_lookup", f"{r}/recommendations_with_ethnicity.csv",
                                        "--output", f"{r}/factuality_full_{v}.csv"])
        logger.info("Output: %s/factuality_full_%s.csv", r, v)


if __name__ == "__main__":
    main()
