"""
factuality_openalex.py — Step 0.5 of the factuality pipeline.

For each row in the input CSV resolves the recommended author in OpenAlex
(local DuckDB snapshot only) and adds 12 oa_* columns. All input columns are
preserved.

Pipeline (each step only handles items not already resolved):
  1. Persistent cache (--cache)  — skip names from previous runs.
  2. DuckDB authors EXACT        — single scan, normalized match against
                                    display_name AND display_name_alternatives;
                                    best author per name by citations. Fast.
  3. DuckDB authors JW           — for names not resolved by exact match,
                                    Jaro-Winkler similarity (≥0.85) against
                                    candidates pre-blocked by the first 2
                                    chars of the last name. Catches LLM
                                    name variants ("J Smith" vs "John Smith").
  4. DuckDB works    (--db_path) — for resolved oa_ids, derive country and
                                    institution from the most-recent paper
                                    (the `authors.last_known_institution` field
                                    in the snapshot is empty, so we use
                                    publications instead).

Output columns added:
  oa_status, oa_id, oa_display_name, oa_works_count, oa_cited_by_count,
  oa_h_index, oa_i10_index, oa_first_pub_year, oa_last_pub_year, oa_career_age,
  oa_country_code, oa_last_institution

Usage (from code/scripts/factuality/):
  python factuality_openalex.py \\
      --input  ../../../results/summary/factuality_author_jw.csv \\
      --output ../../../results/summary/factuality_oa.csv \\
      --db_path /data/datasets/LLMScholar-Personas/data/openalex_latest.duckdb \\
      --cache  ../../../results/summary/.oa_cache.pkl \\
      [--skip_works]
"""

import argparse
import logging
import os
import pickle
import re
import time
import unicodedata

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

WORKS_TMP_DIR    = "/data/asanchez/duckdb_enrich"
JW_THRESHOLD          = 0.85   # JW threshold on full normalized name
JW_FIRST_THRESHOLD    = 0.80   # JW threshold on first-name token (same person check)
# These thresholds are combined with EXACT last-name token match in _AUTHORS_JW_QUERY.
# The actual JW score is preserved in the `oa_match_score` output column so downstream
# notebooks can filter further (e.g. >=0.95 for stricter match).

STATUS_FOUND     = "found"
STATUS_NOT_FOUND = "not_found"

OA_COLS: list[str] = [
    "oa_status", "oa_id", "oa_display_name",
    "oa_works_count", "oa_cited_by_count",
    "oa_h_index", "oa_i10_index",
    "oa_first_pub_year", "oa_last_pub_year", "oa_career_age",
    "oa_country_code", "oa_last_institution",
    "oa_match_score",   # 1.0 = exact normalized match; <1.0 = JW fuzzy match
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

def _career_age(counts_by_year: list[dict]) -> tuple[int | None, int | None, int | None]:
    years = [int(e["year"]) for e in counts_by_year if e.get("works_count", 0) > 0]
    if not years:
        return None, None, None
    return min(years), max(years), max(years) - min(years)


def _empty_record() -> dict:
    return {col: (STATUS_NOT_FOUND if col == "oa_status" else None) for col in OA_COLS}


# ── DuckDB connection helper ───────────────────────────────────────────────────

def _open_duckdb(db_path: str):
    import duckdb  # noqa: F401  (import error handled by callers)
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
             ORDER BY cited_by_count DESC NULLS LAST,
                      works_count    DESC NULLS LAST
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
    try:
        import duckdb  # noqa: F401
    except ImportError:
        logger.warning("duckdb not installed — skipping local DB lookup.")
        return {}
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
        logger.info("DuckDB authors: %d rows returned (%.0fs)", len(rows), time.time() - t0)
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
        counts = counts_by_year or []
        first_year, last_year, age = _career_age(
            [{"year": e["year"], "works_count": e.get("works_count", 0)} for e in counts]
        )
        out[norm] = {
            "oa_status":           STATUS_FOUND,
            "oa_id":               oa_id,
            "oa_display_name":     display_name,
            "oa_works_count":      works_count,
            "oa_cited_by_count":   cited_by_count,
            "oa_h_index":          None,   # not in snapshot
            "oa_i10_index":        None,
            "oa_first_pub_year":   first_year,
            "oa_last_pub_year":    last_year,
            "oa_career_age":       age,
            "oa_country_code":     None,   # filled later by works scan
            "oa_last_institution": None,   # filled later by works scan
            "oa_match_score":      1.0,    # exact normalized match
        }
    logger.info("DuckDB authors: resolved %d / %d names", len(out), len(unique))
    return out


# ── Step 2.5: DuckDB authors JW (fuzzy fallback for unresolved names) ──────────

# For names that didn't match exactly, apply three combined filters:
#   1. EXACT last-name token match  — eliminates "Moller"/"Moll", "Kaunda"/"Kaundu" etc.
#   2. JW on first-name token >= 0.80 — eliminates "Aris"/"Isaac", "Cornelia"/"Cornelius",
#                                        "George"/"Gerrie", "Sharon"/"Stan" etc. that share
#                                        last name but are clearly different people.
#   3. JW on full normalized name >= 0.85 — overall similarity check.
#
# Common-sense passes: "Gareth F S Whitehead" / "Gareth Whitehead" (last=whitehead exact,
# first=gareth/gareth = 1.0, full JW high — accepted). "Cornelia Moller" / "Cornelius Moller"
# (last=moller exact, full JW high BUT first=cornelia/cornelius JW = 0.79 — rejected).
_AUTHORS_JW_QUERY = r"""
WITH q AS (
  SELECT norm,
         regexp_extract(norm, '[^ ]+$') AS last_token,
         split_part(norm, ' ', 1)       AS first_token
  FROM query_names
  WHERE length(norm) > 0
),
expanded AS (
  SELECT a.id,
         a.display_name,
         a.works_count,
         a.cited_by_count,
         a.counts_by_year,
         trim(regexp_replace(
           regexp_replace(lower(strip_accents(cand_name)), '[^a-z ]+', ' ', 'g'),
           ' +', ' ', 'g'
         )) AS cand_norm
  FROM authors AS a,
       UNNEST(list_concat([a.display_name],
                          COALESCE(a.display_name_alternatives, []))) AS t(cand_name)
),
indexed AS (
  SELECT *,
         regexp_extract(cand_norm, '[^ ]+$') AS last_token,
         split_part(cand_norm, ' ', 1)       AS first_token
  FROM expanded
  WHERE length(cand_norm) > 0
),
joined AS (
  SELECT idx.id,
         idx.display_name,
         idx.works_count,
         idx.cited_by_count,
         idx.counts_by_year,
         q.norm AS query_norm,
         jaro_winkler_similarity(idx.cand_norm,   q.norm)        AS jw_score,
         jaro_winkler_similarity(idx.first_token, q.first_token) AS jw_first
  FROM indexed idx
  JOIN q ON idx.last_token = q.last_token        -- EXACT last-name match
        AND length(q.last_token) >= 2
),
filtered AS (
  SELECT * FROM joined
  WHERE jw_score >= ?           -- full-name JW threshold
    AND jw_first >= ?           -- first-name JW threshold
),
ranked AS (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY query_norm
           ORDER BY jw_score        DESC,
                    jw_first        DESC,
                    cited_by_count  DESC NULLS LAST,
                    works_count     DESC NULLS LAST
         ) AS rn
  FROM filtered
)
SELECT id, display_name, works_count, cited_by_count, counts_by_year, query_norm, jw_score
FROM ranked
WHERE rn = 1
"""


def resolve_via_jw(db_path: str, norm_names: list[str]) -> dict[str, dict | None]:
    """Stage B: Jaro-Winkler fuzzy match against OpenAlex authors for names that
    exact match missed. Pre-blocks candidates by 2-char last-name prefix to
    keep the JW computation tractable over 113M authors."""
    try:
        import duckdb  # noqa: F401
    except ImportError:
        logger.warning("duckdb not installed — skipping JW lookup.")
        return {}
    if not os.path.exists(db_path):
        logger.warning("DuckDB path not found: %s — skipping JW lookup.", db_path)
        return {}

    unique = sorted({n for n in norm_names if n})
    if not unique:
        return {}

    logger.info("DuckDB JW: opening %s", db_path)
    con = _open_duckdb(db_path)
    try:
        con.register("query_names", pd.DataFrame({"norm": unique}))
        logger.info("DuckDB JW: scanning + JW for %d unresolved names "
                    "(full>=%.2f, first>=%.2f, last=exact) …",
                    len(unique), JW_THRESHOLD, JW_FIRST_THRESHOLD)
        t0 = time.time()
        rows = con.execute(_AUTHORS_JW_QUERY, [JW_THRESHOLD, JW_FIRST_THRESHOLD]).fetchall()
        logger.info("DuckDB JW: %d rows returned (%.0fs)", len(rows), time.time() - t0)
    except Exception as exc:
        logger.warning("DuckDB JW query failed: %s", exc)
        con.close()
        return {}
    finally:
        try:
            con.close()
        except Exception:
            pass

    out: dict[str, dict | None] = {}
    for oa_id, display_name, works_count, cited_by_count, counts_by_year, query_norm, jw_score in rows:
        if query_norm in out:
            continue
        counts = counts_by_year or []
        first_year, last_year, age = _career_age(
            [{"year": e["year"], "works_count": e.get("works_count", 0)} for e in counts]
        )
        out[query_norm] = {
            "oa_status":           STATUS_FOUND,
            "oa_id":               oa_id,
            "oa_display_name":     display_name,
            "oa_works_count":      works_count,
            "oa_cited_by_count":   cited_by_count,
            "oa_h_index":          None,
            "oa_i10_index":        None,
            "oa_first_pub_year":   first_year,
            "oa_last_pub_year":    last_year,
            "oa_career_age":       age,
            "oa_country_code":     None,
            "oa_last_institution": None,
            "oa_match_score":      float(jw_score),   # JW score for downstream filtering
        }
    logger.info("DuckDB JW: resolved %d / %d names (avg score in matches: %.3f)",
                len(out), len(unique),
                (sum(r[6] for r in rows if r[5] in out) / max(len(out), 1)))
    return out


# ── Step 3: DuckDB works (oa_id → most-recent country + institution) ──────────

# Year-chunked aggregation approach.
#
# History — what doesn't work:
#   1. Streaming via fetch_record_batch + Python-side oa_id filter:
#      DuckDB never yields the first Arrow batch — it buffers the whole
#      UNNEST(authorships) × UNNEST(institutions) intermediate to disk first
#      (~3.7B exploded rows from 492M papers). Verified with minimal test.
#   2. Single-shot server-side GROUP BY oa_id:
#      OOM-killed after 45 min, 414 GB spilled, with memory_limit=64GB.
#      The intermediate UNNEST × UNNEST is too large for this box even with
#      hash-agg's native spilling.
#
# Current approach — break the scan into year-range chunks. publication_year
# is a top-level BIGINT column (predicate-pushable), so each chunk only scans
# its own slice of works. Each chunk produces an aggregated Parquet with one
# row per (oa_id, year_range): country and inst_name from that range's most-
# recent paper. Chunks are sized to keep per-chunk exploded output ~10× smaller
# than the full table, well within DuckDB's hash-agg comfort zone.
#
# Final step: a small JOIN+GROUP BY across chunk Parquets keeps the most-recent
# year per oa_id, filtered to the 606K oa_ids of interest.
#
# Note on institution names: works.authorships[].institutions[].display_name
# is already inline in the struct, so we use it directly instead of looking up
# inst_id against the institutions table.

WORKS_AGG_DIR = "/data/asanchez/duckdb_enrich/oa_works_agg_chunks"

# Chunks are tuples of (label, sql_predicate). Predicate plugs into the
# `WHERE inst.country_code IS NOT NULL AND <predicate>` clause. Label is used
# for the chunk Parquet filename and progress logs.
#
# Sizing rationale: chunk 2024-2024 (30M papers) succeeded at 48 GB RSS under
# a 64 GB cap. Chunk 2025-2999 (57M papers) blew past the cap and OOM-killed.
# Empirical safe ceiling is ~30M papers per chunk on this box. Big single
# years (2025 = 45M) get split via `hash(w.id) % N` so each shard stays under
# the ceiling. Older years group into wider ranges since pre-2010 papers tend
# to have fewer authors per paper (smaller exploded blow-up).
#
# Order is recent → old so an early-bail still covers most authors' most-
# recent papers (ARG_MAX wins from the latest chunk that resolved them).
YEAR_CHUNKS: list[tuple[str, str]] = [
    ("2024",      "publication_year = 2024"),                                       # 30M ✓
    ("2025_a",    "publication_year = 2025 AND hash(w.id) % 3 = 0"),                # ~15M
    ("2025_b",    "publication_year = 2025 AND hash(w.id) % 3 = 1"),                # ~15M
    ("2025_c",    "publication_year = 2025 AND hash(w.id) % 3 = 2"),                # ~15M
    ("2026_plus", "publication_year BETWEEN 2026 AND 2999"),                        # ~12M
    ("2023",      "publication_year = 2023"),                                       # 22M
    ("2022",      "publication_year = 2022"),                                       # 17M
    ("2021",      "publication_year = 2021"),                                       # 17M
    ("2020",      "publication_year = 2020"),                                       # 16M
    ("2019",      "publication_year = 2019"),                                       # 17M
    ("2018",      "publication_year = 2018"),                                       # 15M
    ("2017",      "publication_year = 2017"),                                       # 15M
    ("2015_2016", "publication_year BETWEEN 2015 AND 2016"),                        # 32M (edge)
    ("2013_2014", "publication_year BETWEEN 2013 AND 2014"),                        # 27M
    ("2011_2012", "publication_year BETWEEN 2011 AND 2012"),                        # 23M
    ("2008_2010", "publication_year BETWEEN 2008 AND 2010"),                        # ~28M
    ("2004_2007", "publication_year BETWEEN 2004 AND 2007"),                        # ~25M
    ("2000_2003", "publication_year BETWEEN 2000 AND 2003"),                        # ~17M
    ("1990_1999", "publication_year BETWEEN 1990 AND 1999"),                        # legacy
    ("pre1990",   "publication_year BETWEEN 1500 AND 1989"),                        # legacy
]


def _chunk_path(label: str) -> str:
    return os.path.join(WORKS_AGG_DIR, f"chunk_{label}.parquet")


def _build_works_agg_chunk(db_path: str, label: str, predicate: str) -> None:
    """Aggregate a single chunk (defined by predicate) into its parquet.
    Idempotent — skips if the parquet already exists and is non-empty."""
    import duckdb  # noqa: F401
    out = _chunk_path(label)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        logger.info("DuckDB chunk %s: reusing %s (%.2f GB)",
                    label, out, os.path.getsize(out) / 1e9)
        return

    # If a zero-byte stub from a killed previous run is here, nuke it.
    if os.path.exists(out) and os.path.getsize(out) == 0:
        os.remove(out)

    os.makedirs(WORKS_AGG_DIR, exist_ok=True)
    logger.info("DuckDB chunk %s: building %s (predicate: %s) …", label, out, predicate)
    con = _open_duckdb(db_path)
    try:
        t0 = time.time()
        con.execute(f"""
            COPY (
              SELECT
                au.author.id                                       AS oa_id,
                ARG_MAX(inst.country_code, w.publication_year)     AS country,
                ARG_MAX(inst.display_name, w.publication_year)     AS inst_name,
                MAX(w.publication_year)                            AS last_year
              FROM works AS w,
                   UNNEST(w.authorships)   AS t1(au),
                   UNNEST(au.institutions) AS t2(inst)
              WHERE inst.country_code IS NOT NULL
                AND ({predicate})
              GROUP BY au.author.id
            ) TO '{out}' (FORMAT PARQUET)
        """)
        size_gb = os.path.getsize(out) / 1e9
        logger.info("DuckDB chunk %s: built (%.2f GB, %.0fs)",
                    label, size_gb, time.time() - t0)
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


def resolve_country_via_works(db_path: str, oa_ids: list[str]) -> dict[str, tuple[str | None, str | None]]:
    """Build (or reuse) year-range chunk Parquets, then filter+merge across
    chunks to derive (country, institution_display_name) per oa_id from each
    author's most-recent paper.

    Returns {oa_id: (country, institution_display_name)}.
    """
    try:
        import duckdb  # noqa: F401
    except ImportError:
        logger.warning("duckdb not installed — skipping works lookup.")
        return {}
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
    logger.info("DuckDB chunks: filtering+merging across %s for %d oa_ids …",
                chunk_glob, len(set(oa_ids)))
    t0 = time.time()
    con = _open_duckdb(db_path)
    try:
        con.register("query_oa_ids", pd.DataFrame({"id": list(set(oa_ids))}))
        # Cross-chunk dedup: ARG_MAX picks the (country, inst_name) from the
        # chunk with the highest last_year per oa_id. Filter via JOIN keeps
        # only the 606K oa_ids of interest.
        rows = con.execute(f"""
            SELECT
              p.oa_id,
              ARG_MAX(p.country,   p.last_year) AS country,
              ARG_MAX(p.inst_name, p.last_year) AS inst_name
            FROM read_parquet('{chunk_glob}') p
            JOIN query_oa_ids q ON p.oa_id = q.id
            GROUP BY p.oa_id
        """).fetchall()
        logger.info("DuckDB chunks: %d oa_ids resolved (%.0fs)",
                    len(rows), time.time() - t0)
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
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            cache = pickle.load(f)
        logger.info("Cache: loaded %d entries from %s", len(cache), path)
        return cache
    except Exception as exc:
        logger.warning("Cache: failed to load %s (%s) — starting empty", path, exc)
        return {}


def save_cache(cache: dict[str, dict | None], path: str | None) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Cache: saved %d entries to %s", len(cache), path)
    except Exception as exc:
        logger.warning("Cache: failed to save %s (%s)", path, exc)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(
    input_path: str,
    output_path: str,
    db_path: str | None = None,
    cache_path: str | None = None,
    skip_works: bool = False,
) -> None:
    logger.info("Loading input: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    logger.info("Rows: %d", len(df))

    norm_series = (
        (df["name"].fillna("").astype(str) + " " + df["lastname"].fillna("").astype(str))
        .str.strip()
        .map(normalize_name)
    )
    unique_norms = sorted({n for n in norm_series.unique() if n})
    logger.info("Unique non-empty author names: %d", len(unique_norms))

    # 1. persistent cache
    cache = load_cache(cache_path)
    pending = [n for n in unique_norms if n not in cache]
    logger.info("Cache: %d hit / %d total (pending: %d)",
                len(unique_norms) - len(pending), len(unique_norms), len(pending))

    # 2. DuckDB authors — exact match (fast)
    if pending and db_path:
        cache.update(resolve_via_authors(db_path, pending))
        save_cache(cache, cache_path)   # checkpoint: safe to Ctrl+C from here

    # 2.5 DuckDB authors — JW fuzzy match for names exact-match missed
    if db_path:
        unresolved = [n for n in unique_norms if n not in cache or not cache[n].get("oa_id")]
        # avoid re-running JW on names we already tried-and-failed in a previous run
        unresolved = [n for n in unresolved
                      if n not in cache or not cache[n].get("_jw_tried")]
        if unresolved:
            jw_resolved = resolve_via_jw(db_path, unresolved)
            for n in unresolved:
                # mark as JW-tried regardless (so re-runs skip)
                if n in jw_resolved:
                    cache[n] = jw_resolved[n]
                cache.setdefault(n, _empty_record())
                cache[n]["_jw_tried"] = True
            save_cache(cache, cache_path)   # checkpoint: safe to Ctrl+C from here

    # mark unresolved names as not_found (so re-runs don't retry)
    for n in unique_norms:
        if n not in cache:
            cache[n] = _empty_record()

    # 3. DuckDB works → fresh country + institution per oa_id (skip already-tried)
    if not skip_works and db_path:
        oa_ids = [
            rec["oa_id"] for rec in cache.values()
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
            logger.info("Works enrichment: filled country/institution for %d oa_ids", updated)
        else:
            logger.info("Works enrichment: nothing to fetch (all oa_ids already attempted)")

    # persist cache
    save_cache(cache, cache_path)

    # 4. assemble output
    records = [cache.get(n) or _empty_record() for n in norm_series]
    for col in OA_COLS:
        df[col] = [r[col] for r in records]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows → %s", len(df), output_path)

    n = len(df)
    logger.info("OpenAlex status distribution:")
    for status, count in df["oa_status"].value_counts().items():
        logger.info("  %-12s %6d  (%.1f%%)", status, count, 100 * count / n)
    has_country = df["oa_country_code"].notna().sum()
    has_inst    = df["oa_last_institution"].notna().sum()
    logger.info("  oa_country_code     populated: %d (%.1f%%)", has_country, 100 * has_country / n)
    logger.info("  oa_last_institution populated: %d (%.1f%%)", has_inst,    100 * has_inst    / n)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAlex enrichment for the factuality pipeline (DuckDB only)")
    parser.add_argument("--input", required=True,
                        help="Input CSV (e.g. factuality_author_jw.csv)")
    parser.add_argument("--output", required=True,
                        help="Output CSV (input columns + oa_* columns)")
    parser.add_argument("--db_path", default=None,
                        help="Path to openalex_latest.duckdb (used for both authors and works lookups)")
    parser.add_argument("--cache", default=None,
                        help="Path to persistent pickle cache (resume across runs)")
    parser.add_argument("--skip_works", action="store_true",
                        help="Skip works streaming (country/institution lookup) — faster but no location data")
    args = parser.parse_args()

    run(
        args.input, args.output,
        db_path=args.db_path,
        cache_path=args.cache,
        skip_works=args.skip_works,
    )


if __name__ == "__main__":
    main()
