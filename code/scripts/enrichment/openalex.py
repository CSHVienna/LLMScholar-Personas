"""
openalex.py — Step 1 of run_enrichment.py: find each recommended author in OpenAlex.

For each row in the input CSV resolves the recommended author in OpenAlex
(local DuckDB snapshot only) and adds 12 oa_* columns. All input columns are
preserved.

Pipeline (each step only handles items not already resolved):
  1. Persistent cache (--cache)  — skip names from previous runs.
  2. DuckDB authors EXACT        — single scan, normalized match against
                                    display_name AND display_name_alternatives;
                                    one author per name, at random (seeded). Fast.
  3. DuckDB works    (--db_path) — for resolved oa_ids, derive country and
                                    institution from the most-recent paper
                                    (the `authors.last_known_institution` field
                                    in the snapshot is empty, so we use
                                    publications instead).

Output columns added:
  oa_status, oa_id, oa_display_name, oa_works_count, oa_cited_by_count,
  oa_h_index, oa_i10_index, oa_first_pub_year, oa_last_pub_year, oa_career_age,
  oa_country_code, oa_last_institution

Usage (from code/, with PYTHONPATH=.):
  python scripts/enrichment/openalex.py \\
      --input    ../results/summary/recommendations.csv \\
      --output   ../results/summary/factuality_oa.csv \\
      --db_path  <path_to_oa_duckdb> \\
      --cache    ../results/summary/.oa_cache_random.pkl \\
      [--skip_works]

Defaults for paths come from [data] in config.ini.
"""

import argparse
import os
import re
import time
import unicodedata

import duckdb
import pandas as pd

from libs.utils.config import config_default
from libs.utils.ios import (
    load_pickle,
    read_input_csv,
    save_pickle,
    write_output_csv,
)
from libs.utils.logging import log_value_counts, setup_logging

logger = setup_logging()


# ── Constants ──────────────────────────────────────────────────────────────────

WORKS_TMP_DIR = config_default("oa_works_tmp_dir") or "."

STATUS_FOUND = "found"
STATUS_NOT_FOUND = "not_found"

OA_COLS: list[str] = [
    "oa_status",
    "oa_id",
    "oa_display_name",
    "oa_works_count",
    "oa_cited_by_count",
    "oa_h_index",
    "oa_i10_index",
    "oa_first_pub_year",
    "oa_last_pub_year",
    "oa_career_age",
    "oa_country_code",
    "oa_last_institution",
    "oa_match_score",  # 1.0 = exact normalized match; <1.0 = JW fuzzy match
]


# ── Name normalization (must match SQL normalization) ──────────────────────────

_NON_LETTER_RE = re.compile(r"[^a-z\s]")


def normalize_name(name: str) -> str:
    """Lowercase, strip accents (NFD), keep only letters and single spaces."""
    if not isinstance(name, str) or not name.strip():
        return ""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = _NON_LETTER_RE.sub(" ", name.lower())
    return " ".join(name.split())


# ── Career age helper ──────────────────────────────────────────────────────────


def _career_age(
    counts_by_year: list[dict],
) -> tuple[int | None, int | None, int | None]:
    years = [int(e["year"]) for e in counts_by_year if e.get("works_count", 0) > 0]
    if not years:
        return None, None, None
    return min(years), max(years), max(years) - min(years)


def _empty_record() -> dict:
    return {col: (STATUS_NOT_FOUND if col == "oa_status" else None) for col in OA_COLS}


# ── DuckDB connection helper ───────────────────────────────────────────────────


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


# ── Step 2: DuckDB authors (name → oa_id + metadata) ───────────────────────────

_AUTHORS_QUERY = r"""
WITH
  q AS (SELECT norm FROM query_names),
  expanded AS (
    -- one row per (author × candidate name): display_name + each alternative
    SELECT
      a.id,
      a.display_name,
      a.works_count,
      a.cited_by_count,
      a.counts_by_year,
      cand_name
    FROM authors AS a,
         UNNEST(list_concat([a.display_name],
                            COALESCE(a.display_name_alternatives, []))) AS t(cand_name)
  ),
  normalized AS (
    SELECT id, display_name, works_count, cited_by_count, counts_by_year,
           trim(regexp_replace(
             regexp_replace(lower(strip_accents(cand_name)), '[^a-z ]+', ' ', 'g'),
             ' +', ' ', 'g'
           )) AS norm
    FROM expanded
  ),
  filtered AS (
    SELECT n.* FROM normalized n
    JOIN q ON n.norm = q.norm
  ),
  ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (
             PARTITION BY norm
             -- one of the authors with that name, at random (as the
             -- factuality cascade does). Hashing id + name with a fixed
             -- seed ('42') makes the pick reproducible.
             ORDER BY hash(id::VARCHAR || norm || '42')
           ) AS rn
    FROM filtered
  )
SELECT id, display_name, works_count, cited_by_count, counts_by_year, norm
FROM ranked
WHERE rn = 1
"""


def resolve_via_authors(db_path: str, norm_names: list[str]) -> dict[str, dict | None]:
    """Resolve normalized names → oa_id + metadata via the local DuckDB authors
    table (single scan, matches display_name and alternatives)."""
    if not os.path.exists(db_path):
        logger.warning("DuckDB path not found: %s — skipping authors lookup.", db_path)
        return {}

    unique = sorted({n for n in norm_names if n})
    if not unique:
        return {}

    logger.info("DuckDB authors: opening %s", db_path)
    con = _open_duckdb(db_path)
    try:
        con.register("query_names", pd.DataFrame({"norm": unique}))
        logger.info("DuckDB authors: scanning (%d unique queries) …", len(unique))
        t0 = time.time()
        rows = con.execute(_AUTHORS_QUERY).fetchall()
        logger.info(
            "DuckDB authors: %d rows returned (%.0fs)", len(rows), time.time() - t0
        )
    except Exception as exc:
        logger.warning("DuckDB authors query failed: %s", exc)
        con.close()
        return {}
    finally:
        try:
            con.close()
        except Exception:
            pass

    out: dict[str, dict | None] = {}
    for oa_id, display_name, works_count, cited_by_count, counts_by_year, norm in rows:
        if norm in out:
            continue
        first_year, last_year, age = _career_age(counts_by_year or [])
        out[norm] = {
            "oa_status": STATUS_FOUND,
            "oa_id": f"https://openalex.org/A{oa_id}",   # authors.id is the number
            "oa_display_name": display_name,
            "oa_works_count": works_count,
            "oa_cited_by_count": cited_by_count,
            "oa_h_index": None,  # not in snapshot
            "oa_i10_index": None,
            "oa_first_pub_year": first_year,
            "oa_last_pub_year": last_year,
            "oa_career_age": age,
            "oa_country_code": None,  # filled later by works scan
            "oa_last_institution": None,  # filled later by works scan
            "oa_match_score": 1.0,  # exact normalized match
        }
    logger.info("DuckDB authors: resolved %d / %d names", len(out), len(unique))
    return out


# ── Step 3: DuckDB works (oa_id → most-recent country + institution) ──────────

# Country and institution of each author: those of their most recent paper.
# One query over all papers does not fit in memory, so papers are aggregated by
# publication year ranges (YEAR_CHUNKS), one Parquet per range, kept on disk
# and reused; a last small query keeps, per author, the most recent range.

WORKS_AGG_DIR = os.path.join(WORKS_TMP_DIR, "oa_works_agg_chunks")

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
              SELECT
                {AUTHOR_URL}                                       AS oa_id,
                ARG_MAX(inst.country_code, w.publication_year)     AS country,
                ARG_MAX(inst.display_name, w.publication_year)     AS inst_name,
                MAX(w.publication_year)                            AS last_year
              FROM works AS w,
                   UNNEST(w.authorships)   AS t1(au),
                   UNNEST(au.institutions) AS t2(inst)
              WHERE inst.country_code IS NOT NULL
                AND ({predicate})
              GROUP BY {AUTHOR_URL}
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


def resolve_country_via_works(
    db_path: str, oa_ids: list[str]
) -> dict[str, tuple[str | None, str | None]]:
    """Build (or reuse) year-range chunk Parquets, then filter+merge across
    chunks to derive (country, institution_display_name) per oa_id from each
    author's most-recent paper.

    Returns {oa_id: (country, institution_display_name)}.
    """
    if not os.path.exists(db_path):
        logger.warning("DuckDB path not found: %s — skipping works lookup.", db_path)
        return {}
    if not oa_ids:
        return {}

    try:
        _build_all_chunks(db_path)
    except Exception as exc:
        logger.warning("DuckDB chunk build failed: %s", exc)
        return {}

    chunk_glob = os.path.join(WORKS_AGG_DIR, "chunk_*.parquet")
    logger.info(
        "DuckDB chunks: filtering+merging across %s for %d oa_ids …",
        chunk_glob,
        len(set(oa_ids)),
    )
    t0 = time.time()
    con = _open_duckdb(db_path)
    try:
        con.register("query_oa_ids", pd.DataFrame({"id": list(set(oa_ids))}))
        # Per author, the most recent of the ranges (chunks and ids are both URLs).
        rows = con.execute(
            f"""
            SELECT
              p.oa_id,
              ARG_MAX(p.country,   p.last_year) AS country,
              ARG_MAX(p.inst_name, p.last_year) AS inst_name
            FROM read_parquet('{chunk_glob}') p
            JOIN query_oa_ids q ON p.oa_id = q.id
            GROUP BY p.oa_id
        """
        ).fetchall()
        logger.info(
            "DuckDB chunks: %d oa_ids resolved (%.0fs)", len(rows), time.time() - t0
        )
    except Exception as exc:
        logger.warning("DuckDB chunks filter/merge failed: %s", exc)
        return {}
    finally:
        try:
            con.close()
        except Exception:
            pass

    return {oa_id: (country, inst_name) for oa_id, country, inst_name in rows}


# ── Persistent cache ───────────────────────────────────────────────────────────


def load_cache(path: str | None) -> dict[str, dict | None]:
    return load_pickle(path, logger=logger) or {}


def save_cache(cache: dict[str, dict | None], path: str | None) -> None:
    save_pickle(cache, path, logger=logger)


# ── Main pipeline ──────────────────────────────────────────────────────────────


def run(
    input_path: str,
    output_path: str,
    db_path: str | None = None,
    cache_path: str | None = None,
    skip_works: bool = False,
) -> None:
    df = read_input_csv(input_path, logger=logger)

    norm_series = (
        (
            df["name"].fillna("").astype(str)
            + " "
            + df["lastname"].fillna("").astype(str)
        )
        .str.strip()
        .map(normalize_name)
    )
    unique_norms = sorted({n for n in norm_series.unique() if n})
    logger.info("Unique non-empty author names: %d", len(unique_norms))

    # 1. persistent cache
    cache = load_cache(cache_path)
    pending = [n for n in unique_norms if n not in cache]
    logger.info(
        "Cache: %d hit / %d total (pending: %d)",
        len(unique_norms) - len(pending),
        len(unique_norms),
        len(pending),
    )

    # 2. DuckDB authors — exact match (fast)
    if pending and db_path:
        cache.update(resolve_via_authors(db_path, pending))
        save_cache(cache, cache_path)  # checkpoint: safe to Ctrl+C from here

    # No fuzzy matching: in a manual check of 20 requests it was right 10% of
    # the time, against 78% for exact matches.

    # mark unresolved names as not_found (so re-runs don't retry)
    for n in unique_norms:
        if n not in cache:
            cache[n] = _empty_record()

    # 3. DuckDB works → fresh country + institution per oa_id (skip already-tried)
    if not skip_works and db_path:
        oa_ids = [
            rec["oa_id"]
            for rec in cache.values()
            if rec and rec.get("oa_id") and not rec.get("_works_tried")
        ]
        if oa_ids:
            country_map = resolve_country_via_works(db_path, oa_ids)
            updated = 0
            for rec in cache.values():
                if not rec:
                    continue
                oa_id = rec.get("oa_id")
                if not oa_id or rec.get("_works_tried"):
                    continue
                if oa_id in country_map:
                    country, inst_name = country_map[oa_id]
                    if country is not None and rec.get("oa_country_code") is None:
                        rec["oa_country_code"] = country
                    if inst_name is not None and rec.get("oa_last_institution") is None:
                        rec["oa_last_institution"] = inst_name
                    updated += 1
                rec["_works_tried"] = True
            logger.info(
                "Works enrichment: filled country/institution for %d oa_ids", updated
            )
        else:
            logger.info(
                "Works enrichment: nothing to fetch (all oa_ids already attempted)"
            )

    # persist cache
    save_cache(cache, cache_path)

    # 4. assemble output
    records = [cache.get(n) or _empty_record() for n in norm_series]
    for col in OA_COLS:
        df[col] = [r[col] for r in records]

    write_output_csv(df, output_path, logger=logger)
    log_value_counts(
        df, "oa_status", title="OpenAlex status distribution", width=12, logger=logger
    )
    n = len(df)
    has_country = df["oa_country_code"].notna().sum()
    has_inst = df["oa_last_institution"].notna().sum()
    logger.info(
        "  oa_country_code     populated: %d (%.1f%%)",
        has_country,
        100 * has_country / n,
    )
    logger.info(
        "  oa_last_institution populated: %d (%.1f%%)", has_inst, 100 * has_inst / n
    )


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAlex enrichment for the factuality pipeline (DuckDB only)"
    )
    parser.add_argument(
        "--input", required=True, help="Input CSV (e.g. recommendations.csv)"
    )
    parser.add_argument(
        "--output", required=True, help="Output CSV (input columns + oa_* columns)"
    )
    parser.add_argument(
        "--db_path",
        default=None,
        help="Path to openalex_latest.duckdb (used for both authors and works lookups)",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Path to persistent pickle cache (resume across runs)",
    )
    parser.add_argument(
        "--skip_works",
        action="store_true",
        help="Skip works streaming (country/institution lookup) — faster but no location data",
    )
    args = parser.parse_args()

    run(
        args.input,
        args.output,
        db_path=args.db_path,
        cache_path=args.cache,
        skip_works=args.skip_works,
    )


if __name__ == "__main__":
    main()
