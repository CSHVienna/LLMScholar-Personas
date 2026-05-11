"""
enrich_from_works.py — Derive country per author from the local OpenAlex
DuckDB snapshot's `works` table, since the `authors` table has
`last_known_institution` empty for every row.

Strategy:
  1. Read factuality_author.csv and collect every oa_id we need to enrich
     (authors that resolved via openalex_duckdb).
  2. Register the id list as a DuckDB DataFrame (`our_authors`) so we can
     filter inside the UNNEST without blowing up the SQL string.
  3. Run a single streamed scan of `works` that, for each oa_id, picks the
     country_code of the most-recent paper (arg_max by publication_year).
  4. Save oa_id → oa_country_code lookup CSV.

DuckDB tuning: max threads, max memory, no progress bar, no insertion-order
preservation. The `IN (SELECT …)` against the registered DataFrame triggers
pushdown so only relevant authorships materialise.

Usage (from code/scripts/):
  # Calibration on a 5M-row sample of works
  python enrich_from_works.py \\
      --input    ../../../results/summary/factuality_author.csv \\
      --db_path  ../../../data/data/openalex_latest.duckdb \\
      --output   ../../../results/summary/oa_enrichment.csv \\
      --benchmark

  # Full scan
  python enrich_from_works.py \\
      --input    ../../../results/summary/factuality_author.csv \\
      --db_path  ../../../data/data/openalex_latest.duckdb \\
      --output   ../../../results/summary/oa_enrichment.csv
"""

import argparse
import logging
import os
import time

import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_oa_ids(input_path: str) -> list[str]:
    """Return the unique non-null oa_id values to enrich."""
    logger.info("Loading factuality_author input: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False, usecols=["oa_id"])
    ids = df["oa_id"].dropna().unique().tolist()
    logger.info("Unique oa_ids to enrich: %d", len(ids))
    return ids


def _open_connection(db_path: str) -> duckdb.DuckDBPyConnection:
    logger.info("Opening DuckDB (read_only=True): %s", db_path)
    con = duckdb.connect(db_path, read_only=True)

    cores = os.cpu_count() or 8
    con.execute(f"SET threads = {cores}")
    con.execute("SET memory_limit = '96GB'")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET enable_progress_bar = false")
    # Spill location: /data/asanchez sits on a 1.9 TB disk; /tmp is only 63 GB.
    tmp_dir = "/data/asanchez/duckdb_enrich"
    os.makedirs(tmp_dir, exist_ok=True)
    con.execute(f"SET temp_directory = '{tmp_dir}'")
    return con


def _enrichment_query(works_source: str) -> str:
    """SQL: stream (oa_id, country, year) tuples for ALL authors with a known
    country. The `oa_id` filter is applied in Python on each batch — DuckDB's
    SEMI JOIN with `IN (SELECT …)` is materialised AFTER the UNNEST and blew
    memory on the full table.
    """
    return f"""
    SELECT
      au.author.id        AS oa_id,
      inst.country_code   AS country,
      w.publication_year  AS year
    FROM {works_source} AS w,
         UNNEST(w.authorships) AS t1(au),
         UNNEST(au.institutions) AS t2(inst)
    WHERE inst.country_code IS NOT NULL
    """


def run(input_path: str, db_path: str, output_path: str, benchmark: bool) -> None:
    oa_ids = _load_oa_ids(input_path)
    if not oa_ids:
        logger.warning("Nothing to enrich — exiting.")
        return

    con = _open_connection(db_path)

    works_source = "(SELECT * FROM works LIMIT 5000000)" if benchmark else "works"
    logger.info("Mode: %s   works source: %s", "BENCHMARK (5M)" if benchmark else "FULL", works_source)

    # ── Stream the result; filter by oa_id in Python with a hash set ───────────
    oa_id_set = set(oa_ids)
    logger.info("Executing streaming query (filter by oa_id done in Python)…")
    t0 = time.time()
    reader = con.execute(_enrichment_query(works_source)).fetch_record_batch(rows_per_batch=1_000_000)

    best: dict[str, tuple[int, str]] = {}  # oa_id → (year, country)
    n_rows = 0
    n_kept = 0
    n_batches = 0
    for batch in reader:
        df = batch.to_pandas()
        n_rows += len(df)
        n_batches += 1
        df = df[df["oa_id"].isin(oa_id_set)]
        n_kept += len(df)
        if not df.empty:
            df = df.sort_values("year", ascending=False).drop_duplicates(subset=["oa_id"], keep="first")
            for oa_id, country, year in zip(df["oa_id"], df["country"], df["year"]):
                prev = best.get(oa_id)
                if prev is None or year > prev[0]:
                    best[oa_id] = (year, country)
        if n_batches % 20 == 0:
            logger.info("  %d batches  scanned: %d  kept: %d  authors: %d  elapsed: %.1fs",
                        n_batches, n_rows, n_kept, len(best), time.time() - t0)

    elapsed = time.time() - t0
    logger.info("Streaming done: %d rows  %d batches  %d authors with country  %.1fs",
                n_rows, n_batches, len(best), elapsed)
    con.close()

    enriched = pd.DataFrame(
        [(oa, country) for oa, (_, country) in best.items()],
        columns=["oa_id", "oa_country_code"],
    )

    out = (
        pd.DataFrame({"oa_id": oa_ids})
          .merge(enriched, on="oa_id", how="left")
    )
    n_country = out["oa_country_code"].notna().sum()
    logger.info("Final lookup: %d rows   with country: %d", len(out), n_country)

    if benchmark:
        full_extrap = elapsed * 492 / 5
        logger.info("BENCHMARK SUMMARY:")
        logger.info("  scan time over 5M works: %.1fs", elapsed)
        logger.info("  extrapolated full run:   %.1f min (%.2f hours)",
                    full_extrap / 60, full_extrap / 3600)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info("Saved %d rows → %s", len(out), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive country per author from the local DuckDB snapshot's works table"
    )
    parser.add_argument("--input",   required=True,
                        help="Path to factuality_author.csv (we read oa_id column)")
    parser.add_argument("--db_path", required=True,
                        help="Path to openalex_latest.duckdb (read_only=True)")
    parser.add_argument("--output",  required=True,
                        help="Output CSV path: oa_id, oa_country_code")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run on LIMIT 5_000_000 works for ETA calibration")
    args = parser.parse_args()

    run(args.input, args.db_path, args.output, args.benchmark)


if __name__ == "__main__":
    main()
