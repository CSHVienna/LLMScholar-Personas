"""
1_openalex.py — Step 1 of run_enrichment.py: the OpenAlex data of the
researcher picked for each recommendation with factuality_author = True.

Nothing is looked up here: the OpenAlex author and its metadata come with the
researcher from the ground truth (scripts/1_ground_truth/, carried by
scripts/6_factuality/ as the oa_* columns). This step names them as the next
steps read them and leaves the rows without factuality_author empty:

  oa_status (found | not_found | not_searched), oa_match (jw | cascade),
  oa_id, oa_display_name, oa_works_count, oa_cited_by_count, oa_h_index,
  oa_i10_index, oa_first_pub_year, oa_last_pub_year, oa_career_age,
  oa_country_code, oa_last_institution, oa_country_source (own_papers | profile)

It also makes sure the per-year chunks of country and institution per OpenAlex
author exist (built once, reused), which 3_affiliation.py reads for the
history of institutions.

Usage (from code/, with PYTHONPATH=.):
  python scripts/7_enrichment/1_openalex.py \\
      --input  ../results/summary/cascade/factuality_ss.csv \\
      --output ../results/summary/factuality_oa.csv
"""

import argparse
import os
import time

import duckdb
import numpy as np

from libs.openalex_common import WORKS_AGG_DIR, WORKS_TMP_DIR  # noqa: E402
from libs.utils.config import config_default  # noqa: E402
from libs.utils.ios import read_input_csv, write_output_csv  # noqa: E402
from libs.utils.logging import log_value_counts, setup_logging  # noqa: E402

logger = setup_logging()



def _open_duckdb(db_path: str):
    con = duckdb.connect(db_path, read_only=True)
    cores = os.cpu_count() or 8
    try:
        con.execute(f"SET threads = {cores}")
        # Cap DuckDB at 64GB so the system keeps comfortable headroom (Python +
        # OS + buff/cache need their share of the 125GB box). With 96GB cap the
        # works-agg run drove RSS past the cap to 100GB and ate system available
        # memory down to ~12GB, risking OOM. Spill goes to /data (1.8TB free)
        # which can absorb the extra GB of intermediate happily.
        con.execute("SET memory_limit = '64GB'")
        con.execute("SET preserve_insertion_order = false")
        con.execute("SET enable_progress_bar = false")
        os.makedirs(WORKS_TMP_DIR, exist_ok=True)
        con.execute(f"SET temp_directory = '{WORKS_TMP_DIR}'")
    except Exception as exc:
        logger.debug("DuckDB tuning failed (non-fatal): %s", exc)
    return con




# (label, SQL condition) per range. Ranges are sized to about 30M papers, the
# most that fit in memory on this machine; 2025 is split in three.
YEAR_CHUNKS: list[tuple[str, str]] = [
    ("2024", "publication_year = 2024"),  # 30M ✓
    ("2025_a", "publication_year = 2025 AND hash(w.id) % 3 = 0"),  # ~15M
    ("2025_b", "publication_year = 2025 AND hash(w.id) % 3 = 1"),  # ~15M
    ("2025_c", "publication_year = 2025 AND hash(w.id) % 3 = 2"),  # ~15M
    ("2026_plus", "publication_year BETWEEN 2026 AND 2999"),  # ~12M
    ("2023", "publication_year = 2023"),  # 22M
    ("2022", "publication_year = 2022"),  # 17M
    ("2021", "publication_year = 2021"),  # 17M
    ("2020", "publication_year = 2020"),  # 16M
    ("2019", "publication_year = 2019"),  # 17M
    ("2018", "publication_year = 2018"),  # 15M
    ("2017", "publication_year = 2017"),  # 15M
    ("2015_2016", "publication_year BETWEEN 2015 AND 2016"),  # 32M (edge)
    ("2013_2014", "publication_year BETWEEN 2013 AND 2014"),  # 27M
    ("2011_2012", "publication_year BETWEEN 2011 AND 2012"),  # 23M
    ("2008_2010", "publication_year BETWEEN 2008 AND 2010"),  # ~28M
    ("2004_2007", "publication_year BETWEEN 2004 AND 2007"),  # ~25M
    ("2000_2003", "publication_year BETWEEN 2000 AND 2003"),  # ~17M
    ("1990_1999", "publication_year BETWEEN 1990 AND 1999"),  # legacy
    ("pre1990", "publication_year BETWEEN 1500 AND 1989"),  # legacy
]


# Order of "most recent" for an author's papers: year, then exact date, then
# the paper's id, then the institution's name — so ties are always broken the
# same way.
LATEST = ("{'year': w.publication_year, 'date': w.publication_date, "
          "'work': w.id, 'inst': inst.display_name}")


# The author of an authorship as the OpenAlex URL. Snapshots up to 2026-03 store
# au.author.id as that URL; from 2026-09 it is the number. typeof() is resolved
# once per query, not per row.
AUTHOR_URL = (
    "CASE WHEN typeof(au.author.id) = 'VARCHAR' THEN au.author.id::VARCHAR "
    "ELSE 'https://openalex.org/A' || au.author.id::VARCHAR END"
)


def _chunk_path(label: str) -> str:
    return os.path.join(WORKS_AGG_DIR, f"chunk_{label}.parquet")


def _build_works_agg_chunk(db_path: str, label: str, predicate: str) -> None:
    """Aggregate a single chunk (defined by predicate) into its parquet.
    Idempotent — skips if the parquet already exists and is non-empty."""
    out = _chunk_path(label)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        logger.info(
            "DuckDB chunk %s: reusing %s (%.2f GB)",
            label,
            out,
            os.path.getsize(out) / 1e9,
        )
        return

    # If a zero-byte stub from a killed previous run is here, nuke it.
    if os.path.exists(out) and os.path.getsize(out) == 0:
        os.remove(out)

    os.makedirs(WORKS_AGG_DIR, exist_ok=True)
    logger.info("DuckDB chunk %s: building %s (predicate: %s) …", label, out, predicate)
    con = _open_duckdb(db_path)
    try:
        t0 = time.time()
        con.execute(
            f"""
            COPY (
              -- The institution of the author's most recent paper in the range.
              -- LATEST (see above) breaks ties (same year: two papers, or a paper
              -- with two institutions) the same way every time, and country and
              -- institution come from the same row.
              SELECT oa_id, last.country, last.inst_name,
                     last_key.year AS last_year, last_key.date AS last_date,
                     last_key.work AS last_work
              FROM (
                SELECT {AUTHOR_URL} AS oa_id,
                       ARG_MAX({{'country': inst.country_code, 'inst_name': inst.display_name}}, {LATEST}) AS last,
                       MAX({LATEST}) AS last_key
                FROM works AS w,
                     UNNEST(w.authorships)   AS t1(au),
                     UNNEST(au.institutions) AS t2(inst)
                WHERE inst.country_code IS NOT NULL
                  AND ({predicate})
                GROUP BY {AUTHOR_URL}
              )
            ) TO '{out}' (FORMAT PARQUET)
        """
        )
        size_gb = os.path.getsize(out) / 1e9
        logger.info(
            "DuckDB chunk %s: built (%.2f GB, %.0fs)", label, size_gb, time.time() - t0
        )
    finally:
        try:
            con.close()
        except Exception:
            pass


def _build_all_chunks(db_path: str) -> None:
    """Build any chunks that aren't already on disk. Cleans DuckDB temp spill
    between chunks so /data doesn't accumulate residue from a failed chunk
    into the next."""
    for label, predicate in YEAR_CHUNKS:
        _build_works_agg_chunk(db_path, label, predicate)
        try:
            for fn in os.listdir(WORKS_TMP_DIR):
                if fn.startswith("duckdb_temp_storage_"):
                    os.remove(os.path.join(WORKS_TMP_DIR, fn))
        except OSError:
            pass


def run(input_path: str, output_path: str, db_path: str) -> None:
    df = read_input_csv(input_path, logger=logger)
    # The funnel: only the recommendations confirmed in Semantic Scholar carry an OpenAlex author.
    searched = df["factuality_author"].astype(bool).to_numpy()
    df = df.rename(columns={"oa_first_year": "oa_first_pub_year", "oa_last_year": "oa_last_pub_year",
                            "oa_institution": "oa_last_institution"})
    oa_cols = [c for c in df.columns if c.startswith("oa_")]
    df.loc[~searched, oa_cols] = None
    found = df["oa_id"].notna().to_numpy()
    df["oa_status"] = np.where(found, "found", np.where(searched, "not_found", "not_searched"))
    df["oa_career_age"] = df["oa_last_pub_year"] - df["oa_first_pub_year"]
    df["oa_match_score"] = np.where(found, 1.0, np.nan)

    _build_all_chunks(db_path)   # for 3_affiliation.py (reused when already there)

    write_output_csv(df, output_path, logger=logger)
    log_value_counts(df, "oa_status", title="OpenAlex status distribution", width=12, logger=logger)
    log_value_counts(df[searched], "oa_match", title="OpenAlex author from", width=12, logger=logger)
    logger.info("  oa_country_code populated: %.1f%%", 100 * df["oa_country_code"].notna().mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="cascade/factuality_ss.csv of scripts/6_factuality/")
    parser.add_argument("--output", required=True)
    parser.add_argument("--db_path", default=config_default("oa_duckdb"))
    args = parser.parse_args()
    run(args.input, args.output, args.db_path)


if __name__ == "__main__":
    main()
