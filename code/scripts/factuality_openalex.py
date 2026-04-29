"""
factuality_openalex.py — Verify LLM-recommended authors via OpenAlex.

For each recommendation row (from batch_parse_results.py) searches the OpenAlex
public API for the recommended author and extracts their academic metrics.

Data source priority:
  1. Local DuckDB snapshot (openalex_latest.duckdb, opened read_only=True) when
     --db_path is supplied — one batch scan for all unique names avoids per-row I/O.
  2. OpenAlex public API (https://api.openalex.org) for any name not resolved
     locally, with an in-memory cache so each unique query hits the network once.

Career age is computed as last_pub_year − first_pub_year using counts_by_year.

Output columns added to recommendations CSV:
  oa_status, oa_id, oa_display_name, oa_works_count, oa_cited_by_count,
  oa_h_index, oa_i10_index, oa_first_pub_year, oa_last_pub_year, oa_career_age,
  oa_country_code, oa_last_institution

Usage (from code/scripts/):
  export PYTHONPATH="$PYTHONPATH:../libs"

  # API only
  python factuality_openalex.py \\
      --recommendations ../../results/summary/recommendations.csv \\
      --output          ../../results/summary/factuality_openalex.csv \\
      --email           your@email.com

  # With local DuckDB snapshot (faster, no rate-limit risk)
  python factuality_openalex.py \\
      --recommendations ../../results/summary/recommendations.csv \\
      --output          ../../results/summary/factuality_openalex.csv \\
      --db_path         ../../data/data/openalex_latest.duckdb \\
      --email           your@email.com
"""

import argparse
import logging
import os
import re
import time
import unicodedata
from typing import Any

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

OPENALEX_API   = "https://api.openalex.org/authors"
API_PER_PAGE   = 5        # candidates to consider per query
API_DELAY      = 0.1      # seconds between API calls (polite mode)
API_TIMEOUT    = 15       # seconds per request
API_RETRIES    = 3
API_RETRY_WAIT = 2        # seconds before retry

STATUS_FOUND     = "found"
STATUS_NOT_FOUND = "not_found"


# ── Name helpers ───────────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """Lowercase, strip accents, keep only letters and spaces."""
    if not isinstance(name, str) or not name.strip():
        return ""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^a-z\s]", "", name.lower())
    return " ".join(name.split())


def _name_tokens(norm: str) -> set[str]:
    return set(norm.split())


def _match_score(query_norm: str, candidate_norm: str) -> float:
    """Fraction of query tokens present in the candidate name."""
    q_tokens = _name_tokens(query_norm)
    c_tokens = _name_tokens(candidate_norm)
    if not q_tokens:
        return 0.0
    return len(q_tokens & c_tokens) / len(q_tokens)


# ── Career age ─────────────────────────────────────────────────────────────────

def _career_age(counts_by_year: list[dict]) -> tuple[int | None, int | None, int | None]:
    """
    Returns (first_pub_year, last_pub_year, career_age) from counts_by_year.
    Only considers entries where works_count > 0.
    """
    years = [
        int(entry["year"])
        for entry in counts_by_year
        if entry.get("works_count", 0) > 0
    ]
    if not years:
        return None, None, None
    first, last = min(years), max(years)
    return first, last, last - first


# ── OpenAlex API ───────────────────────────────────────────────────────────────

class OpenAlexClient:
    """
    Thin wrapper around the OpenAlex /authors endpoint with in-memory caching.
    Each unique normalized query string is looked up at most once.
    """

    def __init__(self, email: str | None = None) -> None:
        self._cache: dict[str, dict | None] = {}
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "factuality_openalex/1.0"})
        self._params: dict[str, Any] = {"per_page": API_PER_PAGE}
        if email:
            self._params["mailto"] = email

    # ── public ────────────────────────────────────────────────────────────────

    def lookup(self, name: str, lastname: str) -> dict | None:
        """
        Search for an author by name + lastname.
        Returns a record dict on match, None otherwise.
        Result is cached so each unique query hits the network once.
        """
        full_name = f"{name} {lastname}".strip()
        norm_full = normalize_name(full_name)
        if not norm_full:
            return None

        if norm_full in self._cache:
            return self._cache[norm_full]

        result = self._search(norm_full)
        self._cache[norm_full] = result
        time.sleep(API_DELAY)
        return result

    def cache_from_db(self, records: dict[str, dict | None]) -> None:
        """Pre-populate the cache from duckdb results to avoid API calls."""
        self._cache.update(records)

    def fetch_author_by_id(self, oa_id: str) -> dict | None:
        """
        Fetch the raw author JSON from the OpenAlex public API by ID.
        Used to enrich records found in the DuckDB snapshot when the snapshot
        is missing fields (e.g. last_known_institution, x_concepts).
        """
        if not oa_id:
            return None
        short_id = oa_id.rstrip("/").rsplit("/", 1)[-1]
        url = f"{OPENALEX_API}/{short_id}"
        for attempt in range(1, API_RETRIES + 1):
            try:
                resp = self._session.get(url, params=self._params, timeout=API_TIMEOUT)
                resp.raise_for_status()
                time.sleep(API_DELAY)
                return resp.json()
            except requests.RequestException as exc:
                logger.warning("API by-id attempt %d/%d failed for '%s': %s",
                               attempt, API_RETRIES, oa_id, exc)
                if attempt < API_RETRIES:
                    time.sleep(API_RETRY_WAIT * attempt)
        return None

    # ── private ───────────────────────────────────────────────────────────────

    def _search(self, norm_full: str) -> dict | None:
        params = {**self._params, "search": norm_full}
        for attempt in range(1, API_RETRIES + 1):
            try:
                resp = self._session.get(OPENALEX_API, params=params, timeout=API_TIMEOUT)
                resp.raise_for_status()
                candidates = resp.json().get("results", [])
                return self._best_match(norm_full, candidates)
            except requests.RequestException as exc:
                logger.warning("API attempt %d/%d failed for '%s': %s", attempt, API_RETRIES, norm_full, exc)
                if attempt < API_RETRIES:
                    time.sleep(API_RETRY_WAIT * attempt)
        return None

    def _best_match(self, norm_full: str, candidates: list[dict]) -> dict | None:
        """
        Pick the best candidate from the API result list.
        Strategy: among candidates with match_score >= 0.5 (at least half the
        query tokens present), return the one with the highest cited_by_count.
        """
        scored = []
        for c in candidates:
            norm_candidate = normalize_name(c.get("display_name", ""))
            score = _match_score(norm_full, norm_candidate)
            if score >= 0.5:
                scored.append((score, c.get("cited_by_count", 0), c))

        if not scored:
            return None

        # primary sort: match_score desc; tie-break: citations desc
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return _extract_record(scored[0][2])


# ── Record extraction ──────────────────────────────────────────────────────────

def _extract_record(author: dict) -> dict:
    counts = author.get("counts_by_year") or []
    first_year, last_year, age = _career_age(counts)
    stats = author.get("summary_stats") or {}
    # OpenAlex API may return either `last_known_institution` (legacy, singular)
    # or `last_known_institutions` (current, plural list — first is the most recent).
    inst = author.get("last_known_institution") or {}
    if not inst:
        plural = author.get("last_known_institutions") or []
        inst = plural[0] if plural else {}
    return {
        "oa_status":          STATUS_FOUND,
        "oa_id":              author.get("id"),
        "oa_display_name":    author.get("display_name"),
        "oa_works_count":     author.get("works_count"),
        "oa_cited_by_count":  author.get("cited_by_count"),
        "oa_h_index":         stats.get("h_index"),
        "oa_i10_index":       stats.get("i10_index"),
        "oa_first_pub_year":  first_year,
        "oa_last_pub_year":   last_year,
        "oa_career_age":      age,
        "oa_country_code":    inst.get("country_code"),
        "oa_last_institution":inst.get("display_name"),
    }


def _empty_record() -> dict:
    return {
        "oa_status":          STATUS_NOT_FOUND,
        "oa_id":              None,
        "oa_display_name":    None,
        "oa_works_count":     None,
        "oa_cited_by_count":  None,
        "oa_h_index":         None,
        "oa_i10_index":       None,
        "oa_first_pub_year":  None,
        "oa_last_pub_year":   None,
        "oa_career_age":      None,
        "oa_country_code":    None,
        "oa_last_institution":None,
    }


# ── DuckDB batch lookup ────────────────────────────────────────────────────────

def load_from_duckdb(db_path: str, names: list[str]) -> dict[str, dict | None]:
    """
    Batch-query the local OpenAlex DuckDB snapshot (read_only=True) for a list
    of normalized full names.  One scan is performed for all names at once.

    Returns {norm_name: record_or_None}.
    """
    try:
        import duckdb  # optional dependency
    except ImportError:
        logger.warning("duckdb not installed — skipping local DB lookup.")
        return {}

    if not os.path.exists(db_path):
        logger.warning("DuckDB path not found: %s — skipping.", db_path)
        return {}

    logger.info("Opening DuckDB (read_only=True): %s", db_path)
    con = duckdb.connect(db_path, read_only=True)

    # Deduplicate normalized names (already lowercase, just removing dups)
    unique_names = sorted({n for n in names if n})
    if not unique_names:
        con.close()
        return {}

    # Register the name list as an in-memory table and JOIN against authors.
    # This is dramatically faster than `WHERE col IN (… 1M placeholders …)`.
    import pandas as pd
    name_df = pd.DataFrame({"norm": unique_names})
    con.register("query_names", name_df)

    logger.info("Querying DuckDB for %d unique names …", len(unique_names))
    try:
        rows = con.execute("""
            SELECT
                a.id,
                a.display_name,
                a.works_count,
                a.cited_by_count,
                a.counts_by_year,
                a.last_known_institution.display_name AS inst_name,
                a.last_known_institution.country_code AS inst_country
            FROM authors AS a
            JOIN query_names AS q
              ON lower(a.display_name) = q.norm
            ORDER BY a.cited_by_count DESC
        """).fetchall()
    except Exception as exc:
        logger.warning("DuckDB query failed: %s", exc)
        con.close()
        return {}

    con.close()

    # For each name, keep only the best row (highest cited_by_count, already sorted)
    name_set = set(unique_names)
    results: dict[str, dict | None] = {}
    for row in rows:
        oa_id, display_name, works_count, cited_by_count, counts_by_year, inst_name, inst_country = row
        norm = normalize_name(display_name)
        if norm in name_set and norm not in results:
            counts = counts_by_year if counts_by_year else []
            first_year, last_year, age = _career_age(
                [{"year": e["year"], "works_count": e.get("works_count", 0)} for e in counts]
            )
            results[norm] = {
                "oa_status":          STATUS_FOUND,
                "oa_id":              oa_id,
                "oa_display_name":    display_name,
                "oa_works_count":     works_count,
                "oa_cited_by_count":  cited_by_count,
                "oa_h_index":         None,   # not stored in local snapshot
                "oa_i10_index":       None,
                "oa_first_pub_year":  first_year,
                "oa_last_pub_year":   last_year,
                "oa_career_age":      age,
                "oa_country_code":    inst_country,
                "oa_last_institution":inst_name,
            }

    logger.info("DuckDB resolved %d / %d names.", len(results), len(unique_names))
    return results


# ── Main processing ────────────────────────────────────────────────────────────

def run(
    recommendations_path: str,
    output_path: str,
    email: str | None = None,
    db_path: str | None = None,
) -> None:
    logger.info("Loading recommendations: %s", recommendations_path)
    df = pd.read_csv(recommendations_path, low_memory=False)
    logger.info("Rows: %d", len(df))

    # Build unique (name, lastname) → norm_full mapping
    df["_norm_full"] = (
        (df["name"].fillna("").astype(str) + " " + df["lastname"].fillna("").astype(str))
        .str.strip()
        .apply(normalize_name)
    )
    unique_norms = [n for n in df["_norm_full"].unique() if n]
    logger.info("Unique non-empty author names: %d", len(unique_norms))

    client = OpenAlexClient(email=email)

    # ── Local DB pre-load (avoids API calls for already-known authors) ─────────
    if db_path:
        db_records = load_from_duckdb(db_path, unique_norms)
        client.cache_from_db(db_records)

    # ── Per-row lookup ─────────────────────────────────────────────────────────
    oa_cols: list[str] = [
        "oa_status", "oa_id", "oa_display_name",
        "oa_works_count", "oa_cited_by_count",
        "oa_h_index", "oa_i10_index",
        "oa_first_pub_year", "oa_last_pub_year", "oa_career_age",
        "oa_country_code", "oa_last_institution",
    ]

    records: list[dict] = []
    n_api_calls_before = len(client._cache)

    for idx, row in df.iterrows():
        norm = row["_norm_full"]
        if not norm:
            records.append(_empty_record())
            continue

        name     = str(row.get("name",     "") or "").strip()
        lastname = str(row.get("lastname", "") or "").strip()

        result = client.lookup(name, lastname)
        records.append(result if result is not None else _empty_record())

        if (idx + 1) % 500 == 0:
            logger.info("  Processed %d / %d rows …", idx + 1, len(df))

    n_api_calls = len(client._cache) - n_api_calls_before
    logger.info("New API calls made: %d", n_api_calls)

    # ── Assemble output ───────────────────────────────────────────────────────
    df = df.drop(columns=["_norm_full"])
    for col in oa_cols:
        df[col] = [r[col] for r in records]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows → %s", len(df), output_path)

    n = len(df)
    logger.info("OpenAlex status distribution:")
    for status, count in df["oa_status"].value_counts().items():
        logger.info("  %-12s %6d  (%.1f%%)", status, count, 100 * count / n)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify factuality of LLM recommendations via OpenAlex API"
    )
    parser.add_argument("--recommendations", required=True,
                        help="Path to recommendations CSV (output of batch_parse_results.py)")
    parser.add_argument("--output", required=True,
                        help="Output CSV path")
    parser.add_argument("--email", default=None,
                        help="Email for OpenAlex polite pool (higher rate limits)")
    parser.add_argument("--db_path", default=None,
                        help="Path to openalex_latest.duckdb for local pre-lookup (read_only=True)")
    args = parser.parse_args()

    run(args.recommendations, args.output, email=args.email, db_path=args.db_path)


if __name__ == "__main__":
    main()
