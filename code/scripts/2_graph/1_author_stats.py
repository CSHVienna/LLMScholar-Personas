"""
1_author_stats.py — Step 2.1: OpenAlex per-author scholarly stats of every
author of the ground truth (1_ground_truth/), for the similarity PCA (paper Eq. 8).

Three of the four stats OpenAlex already publishes, and we read rather than
recompute them:

    h_index, i10_index, 2yr_mean_citedness   <- authors.summary_stats

(The 2026-09 snapshot has that column; earlier ones did not, and their stats
came from the raw author files, which are no longer on this machine.)

The e-index is the exception: OpenAlex does not publish it, so it needs the
per-paper citation counts and a full pass over oa.works.

Stages, each checkpointed to its own file and skipped when that file exists
(unless the ground truth has new authors):

  1. authors.parquet       the author population: every OpenAlex id of
                           ground_truth.parquet. Checked on every run; when it has
                           authors that are not in it yet, it is replaced and the
                           other stages are redone.
  2. summary_stats.parquet h_index / i10_index / 2yr_mean_citedness (minutes)
  3. author_work_citations.parquet  one row per (author, paper, citations) (the long one)
  4. eindex.parquet        h/i10/e aggregated per author

Stage 3 is the expensive one and is persisted before stage 4 aggregates it, so a
failure in the cheap step never costs the scan again.

Usage (from code/, with PYTHONPATH=.):
  python scripts/2_graph/1_author_stats.py
  python scripts/2_graph/1_author_stats.py --rebuild summary_stats
"""

import argparse
import hashlib
import json
import shutil
import time
import warnings
from pathlib import Path

import duckdb

from libs.metrics.coauthorship import chunk_bounds, works_author_key
from libs.utils.config import config_default
from libs.utils.logging import setup_logging

warnings.filterwarnings("ignore")
logger = setup_logging()

STAGES = ("authors", "summary_stats", "author_work_citations", "eindex")

N_CHUNKS = 32


def _connect(memory_limit: str, temp_dir: Path) -> duckdb.DuckDBPyConnection:
    """A connection that spills to real disk.

    `temp_dir` must not live under /tmp on this machine: /tmp is a RAM-backed
    tmpfs, so spilling there consumes more memory instead of relieving it, which
    is how a 90-minute pass gets OOM-killed at the very end.
    """
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=false")
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{temp_dir}'")
    return con


def stage_authors(con, ground_truth: str, out: Path) -> bool:
    """Every OpenAlex id of the ground truth (1_ground_truth/) — the same authors
    2_model.py builds the graph and the PCA from. Returns True when some of
    them are not in the list already on disk (their stats must be computed)."""
    new = out.with_name("authors_new.parquet")
    con.execute(
        f"""
        COPY (
          SELECT DISTINCT
                 regexp_extract(CAST(oa_id AS VARCHAR), '([^/]+)$', 1) AS author_id,
                 CAST(oa_id AS VARCHAR) AS id
          FROM read_parquet('{ground_truth}')
          WHERE oa_id IS NOT NULL AND CAST(oa_id AS VARCHAR) <> ''
        ) TO '{new}' (FORMAT parquet)
        """
    )
    n_new = con.execute(f"SELECT count(*) FROM read_parquet('{new}')").fetchone()[0]
    missing = n_new
    if out.exists():
        missing = con.execute(
            f"""SELECT count(*) FROM read_parquet('{new}')
                WHERE author_id NOT IN (SELECT author_id FROM read_parquet('{out}'))"""
        ).fetchone()[0]
    logger.info("Authors in the ground truth: %d, without stats yet: %d", n_new, missing)
    if missing:
        new.replace(out)
    else:
        new.unlink()
    return missing > 0


def stage_summary_stats(con, duckdb_path: Path, authors: Path, out: Path) -> None:
    """h_index, i10_index and 2yr_mean_citedness as OpenAlex publishes them
    (`summary_stats`), read from the authors table of the snapshot. Snapshots
    before 2026-09 do not have that column: their stats came from the raw
    author files, which are no longer on this machine."""
    con.execute(f"ATTACH '{duckdb_path}' AS oa (READ_ONLY)")
    cols = [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_catalog = 'oa' AND table_name = 'authors'").fetchall()]
    if "summary_stats" not in cols:
        raise SystemExit(f"{duckdb_path} has no authors.summary_stats: use the 2026-09 snapshot or later")
    con.execute(
        f"""
        COPY (
          SELECT u.id,
                 CAST(a.updated_date AS DATE)            AS updated_date,
                 a.works_count::BIGINT                   AS works_count,
                 a.cited_by_count::BIGINT                AS cited_by_count,
                 a.summary_stats.h_index::BIGINT         AS h_index,
                 a.summary_stats.i10_index::BIGINT       AS i10_index,
                 a.summary_stats."2yr_mean_citedness"    AS two_year_mean_citedness,
                 a.counts_by_year
          FROM read_parquet('{authors}') u
          JOIN oa.authors a ON a.id = CAST(substr(u.author_id, 2) AS BIGINT)
        ) TO '{out}' (FORMAT parquet)
        """
    )
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    logger.info("Summary stats: %d authors", n)


def stage_author_work_citations(con, duckdb_path: Path, authors: Path, out: Path) -> None:
    """One row per (author, paper) with the paper's citation count, from a pass
    over oa.works split into N_CHUNKS ranges of works.id (one pass over 490M
    works does not fit in memory). Uncited papers are kept (c = 0)."""
    con.execute(f"ATTACH '{duckdb_path}' AS oa (READ_ONLY)")
    con.execute(f"""CREATE TEMP TABLE u AS
                    SELECT id AS author_url, CAST(substr(author_id, 2) AS BIGINT) AS author_num
                    FROM read_parquet('{authors}')""")
    key = works_author_key(con)          # the snapshot refers to authors by URL or by number

    # Every finished chunk is saved, so a run that stops restarts from the first
    # missing one. The folder is named after the author list (another list
    # never reuses it) and also keeps the chunk boundaries, so resumed chunks
    # line up. It is deleted at the end.
    ids = sorted(r[0] for r in con.execute("SELECT author_url FROM u").fetchall())
    parts = out.with_name(f"{out.stem}_parts_{hashlib.md5(''.join(ids).encode()).hexdigest()[:12]}")
    parts.mkdir(exist_ok=True)
    bounds_file = parts / "bounds.json"
    if bounds_file.exists():
        bounds = json.loads(bounds_file.read_text())
    else:
        bounds = chunk_bounds(con, N_CHUNKS)
        bounds_file.write_text(json.dumps(bounds))
    for i, (start, end) in enumerate(zip(bounds, bounds[1:])):
        part = parts / f"part_{i:02d}.parquet"
        if part.exists():                      # resume after a crash
            continue
        # DISTINCT: an author listed twice in one paper counts the paper once.
        con.execute(
            f"""
            COPY (
              SELECT DISTINCT u.author_url, w.id AS work_id, w.cited_by_count::BIGINT AS c
              FROM oa.works w, UNNEST(w.authorships) AS t(a)
              JOIN u ON u.{key} = a.author.id
              WHERE w.id >= {start} AND w.id < {end}
            ) TO '{part}' (FORMAT parquet)
            """
        )
        logger.info("Works chunk %d/%d done", i + 1, len(bounds) - 1)
    con.execute(f"COPY (SELECT * FROM read_parquet('{parts}/*.parquet')) TO '{out}' (FORMAT parquet)")
    shutil.rmtree(parts)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    logger.info("Author-paper rows: %d", n)


def stage_eindex(con, pairs: Path, out: Path) -> None:
    """Aggregate the pairs into h/i10/e per author.

    h_index and i10_index are recomputed here even though summary_stats already
    supplies them: comparing the two validates the works pass, and with it the
    e-index that comes out of the same citation counts. They agreed on 98.90% of
    authors exactly (99.74% within +/-1, r = 0.99988) when this was last run.

    e_index divides by the CITED papers, per the reference's compute_e_index,
    which runs over a citation table where uncited papers have no rows.
    """
    con.execute(
        f"""
        COPY (
          WITH r AS (
            SELECT author_url, c,
                   row_number() OVER (PARTITION BY author_url ORDER BY c DESC) rn,
                   sum(c)   OVER (PARTITION BY author_url) c_total,
                   count(*) OVER (PARTITION BY author_url) n_papers,
                   count(*) FILTER (c > 0) OVER (PARTITION BY author_url) n_cited
            FROM read_parquet('{pairs}')
          )
          SELECT author_url,
                 max(n_papers) AS n_papers,
                 max(n_cited)  AS n_cited,
                 max(c_total)  AS c_total,
                 coalesce(max(CASE WHEN c >= rn THEN rn END), 0) AS h_index_calc,
                 count(*) FILTER (c >= 10)                       AS i10_index_calc,
                 CASE WHEN max(n_cited) = 0 THEN 0.0 ELSE
                   -(1.0/max(n_cited)) * sum(CASE WHEN c > 0
                     THEN c * ln(c::DOUBLE / c_total) ELSE 0 END) END AS e_index
          FROM r GROUP BY author_url
        ) TO '{out}' (FORMAT parquet)
        """
    )
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    logger.info("E-index: %d authors", n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground_truth", default="../results/1_ground_truth/ground_truth.parquet")
    parser.add_argument("--duckdb", default=config_default("oa_duckdb"))
    parser.add_argument("--out_dir", default="../results/2_graph/author_stats")
    parser.add_argument("--temp_dir", default=config_default("oa_works_tmp_dir"))
    parser.add_argument("--memory_limit", default="32GB")
    parser.add_argument(
        "--rebuild", nargs="*", choices=STAGES, default=[],
        help="Stages to recompute even if their output exists.",
    )
    args = parser.parse_args()

    if not args.out_dir:
        parser.error("Missing --out_dir and no oa_author_stats in [data] of config.ini.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {s: out_dir / f"{s}.parquet" for s in STAGES}
    temp_dir = Path(args.temp_dir or out_dir / ".tmp")

    # The author list is cheap: always check it. If the ground truth has authors
    # without stats, every other stage is redone for the new list.
    con = _connect(args.memory_limit, temp_dir)
    try:
        changed = stage_authors(con, args.ground_truth, paths["authors"])
    finally:
        con.close()
    plan = [s for s in STAGES[1:] if changed or s in args.rebuild or not paths[s].exists()]
    logger.info("Stages to run: %s", plan or "none (all outputs present)")

    for stage in plan:
        con = _connect(args.memory_limit, temp_dir)
        try:
            t0 = time.time()
            logger.info("=== %s ===", stage)
            if stage == "summary_stats":
                stage_summary_stats(con, Path(args.duckdb), paths["authors"], paths[stage])
            elif stage == "author_work_citations":
                stage_author_work_citations(con, Path(args.duckdb), paths["authors"], paths[stage])
            else:
                stage_eindex(con, paths["author_work_citations"], paths[stage])
            logger.info("%s done (%.0fs) -> %s", stage, time.time() - t0, paths[stage])
        finally:
            con.close()

    logger.info("Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
