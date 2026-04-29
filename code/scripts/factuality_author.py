"""
factuality_author.py — Step 1 of the factuality pipeline.

For every recommendation in `recommendations.csv`, decide whether the author
exists at all and persist enough metadata from BOTH sources to power downstream
steps (field, seniority, location).

Both sources are consulted for every author — an author may be present in only
one of them, and we want every available signal:
  • Semantic Scholar ground truth (one CSV per field, see factuality_field.GT_FILES).
  • OpenAlex DuckDB snapshot (read_only=True, table `authors`), with a public
    API fallback for names not in the snapshot.

author_status:
  found         author appears in at least one source
  hallucinated  author absent from both

author_source: '+'-joined list of the sources that returned a hit, e.g.
  semantic_scholar
  openalex_duckdb
  openalex_api
  semantic_scholar+openalex_duckdb
  none

Output columns added to the recommendations CSV:
  author_status, author_source,
  gt_name, gt_field, gt_gender, gt_career_age, gt_citations,
  oa_id, oa_display_name, oa_works_count, oa_cited_by_count,
  oa_h_index, oa_i10_index, oa_first_pub_year, oa_last_pub_year,
  oa_career_age, oa_country_code, oa_last_institution, oa_top_concepts

Usage (from code/scripts/):
  python factuality_author.py \\
      --recommendations ../../results/summary/recommendations.csv \\
      --gt_dir          ../../data/data/semantic_scholar_data \\
      --db_path         ../../data/data/openalex_latest.duckdb \\
      --output          ../../results/summary/factuality_author.csv \\
      --email           your@email.com
"""

import argparse
import logging
import os
import time

import pandas as pd

from factuality_field import (
    REC_FIELD_TO_GT,
    STATUS_FOUND_IN_FIELD,
    STATUS_FOUND_OTHER,
    load_gt_index,
    lookup_author,
    normalize_name,
)
from factuality_openalex import (
    API_DELAY,
    OpenAlexClient,
    _empty_record as _empty_oa_record,
    load_from_duckdb,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STATUS_FOUND        = "found"
STATUS_HALLUCINATED = "hallucinated"

SOURCE_SEMANTIC = "semantic_scholar"
SOURCE_DUCKDB   = "openalex_duckdb"
SOURCE_API      = "openalex_api"
SOURCE_NONE     = "none"

GT_COLS = ["gt_name", "gt_field", "gt_gender", "gt_career_age", "gt_citations"]
OA_COLS = [
    "oa_id", "oa_display_name", "oa_works_count", "oa_cited_by_count",
    "oa_h_index", "oa_i10_index", "oa_first_pub_year", "oa_last_pub_year",
    "oa_career_age", "oa_country_code", "oa_last_institution",
]


def _empty_gt() -> dict:
    return {c: None for c in GT_COLS}


def _gt_record(record: dict, matched_field: str | None) -> dict:
    return {
        "gt_name":       record.get("gt_name"),
        "gt_field":      matched_field,
        "gt_gender":     record.get("gt_gender"),
        "gt_career_age": record.get("gt_career_age"),
        "gt_citations":  record.get("gt_citations"),
    }


def _oa_subset(oa_record: dict) -> dict:
    """Strip oa_status from the OpenAlex record (we use author_status instead)."""
    return {c: oa_record.get(c) for c in OA_COLS}


def run(
    recommendations_path: str,
    gt_dir: str,
    output_path: str,
    db_path: str | None = None,
    email: str | None = None,
    use_api: bool = True,
) -> None:
    logger.info("Loading recommendations: %s", recommendations_path)
    df = pd.read_csv(recommendations_path, low_memory=False)
    logger.info("Recommendations: %d rows", len(df))

    df["_norm_full"] = (
        (df["name"].fillna("").astype(str) + " " + df["lastname"].fillna("").astype(str))
        .str.strip()
        .apply(normalize_name)
    )
    unique_norms = [n for n in df["_norm_full"].unique() if n]
    logger.info("Unique non-empty author names: %d", len(unique_norms))

    # ── Pass 1: Semantic Scholar GT lookup for every row ───────────────────────
    logger.info("Loading local ground truth …")
    gt_index = load_gt_index(gt_dir)

    gt_results: list[tuple[dict, bool]] = []  # (gt_record, found_flag)
    for _, row in df.iterrows():
        norm  = row["_norm_full"]
        field = str(row.get("field", "") or "").strip()
        target_gt_key = REC_FIELD_TO_GT.get(field)

        if not norm:
            gt_results.append((_empty_gt(), False))
            continue

        status, record, matched = lookup_author(norm, target_gt_key, gt_index)
        if status in (STATUS_FOUND_IN_FIELD, STATUS_FOUND_OTHER) and record is not None:
            gt_results.append((_gt_record(record, matched), True))
        else:
            gt_results.append((_empty_gt(), False))

    n_gt_hits = sum(1 for _, found in gt_results if found)
    logger.info("Semantic Scholar hits: %d / %d", n_gt_hits, len(df))

    # ── Pass 2: OpenAlex (DuckDB pre-load + API fallback) for every row ────────
    client = OpenAlexClient(email=email)
    duckdb_hits: dict[str, dict | None] = {}
    if db_path and unique_norms:
        duckdb_hits = load_from_duckdb(db_path, unique_norms)
        client.cache_from_db(duckdb_hits)
    duckdb_norms = {n for n, rec in duckdb_hits.items() if rec is not None}
    logger.info("DuckDB pre-resolved: %d / %d unique names", len(duckdb_norms), len(unique_norms))

    if not use_api:
        # Pre-populate the cache with None for every unique name NOT resolved by
        # DuckDB so that client.lookup() short-circuits without hitting the API.
        misses = {n: None for n in unique_norms if n not in duckdb_norms}
        client.cache_from_db(misses)
        logger.info("API disabled: %d unresolved names cached as miss.", len(misses))

    oa_results: list[tuple[dict, str | None]] = []  # (oa_record_or_empty, source_or_None)
    for norm in df["_norm_full"]:
        if not norm:
            oa_results.append((_empty_oa_record(), None))
            continue

        # The cache key matches what we wrote during DuckDB pre-load.
        # Hitting it directly avoids re-normalising raw name/lastname (which
        # produced 'nan' strings for missing values and triggered API calls).
        if norm in client._cache:
            record = client._cache[norm]
        elif use_api:
            # Fall back to API search; this path only runs when --no_api is OFF.
            record = client._search(norm)
            client._cache[norm] = record
            time.sleep(API_DELAY)
        else:
            record = None

        if record is None:
            oa_results.append((_empty_oa_record(), None))
        else:
            source = SOURCE_DUCKDB if norm in duckdb_norms else SOURCE_API
            oa_results.append((record, source))

    n_oa_hits = sum(1 for _, src in oa_results if src is not None)
    logger.info("OpenAlex hits: %d / %d", n_oa_hits, len(df))

    # ── Combine both passes per row ────────────────────────────────────────────
    statuses: list[str]    = []
    sources:  list[str]    = []
    gt_rows:  list[dict]   = []
    oa_rows:  list[dict]   = []

    for (gt_rec, gt_found), (oa_rec, oa_src) in zip(gt_results, oa_results):
        src_tags: list[str] = []
        if gt_found:
            src_tags.append(SOURCE_SEMANTIC)
        if oa_src is not None:
            src_tags.append(oa_src)

        if src_tags:
            statuses.append(STATUS_FOUND)
            sources.append("+".join(src_tags))
        else:
            statuses.append(STATUS_HALLUCINATED)
            sources.append(SOURCE_NONE)

        gt_rows.append(gt_rec)
        oa_rows.append(_oa_subset(oa_rec))

    df = df.drop(columns=["_norm_full"])
    df["author_status"] = statuses
    df["author_source"] = sources
    for col in GT_COLS:
        df[col] = [r[col] for r in gt_rows]
    for col in OA_COLS:
        df[col] = [r[col] for r in oa_rows]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows → %s", len(df), output_path)

    n = len(df)
    logger.info("Author status distribution:")
    for status, count in df["author_status"].value_counts().items():
        logger.info("  %-40s %6d  (%.1f%%)", status, count, 100 * count / n)
    logger.info("Source distribution:")
    for source, count in df["author_source"].value_counts().items():
        logger.info("  %-40s %6d  (%.1f%%)", source, count, 100 * count / n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: verify the LLM-recommended author exists in Semantic Scholar GT and/or OpenAlex"
    )
    parser.add_argument("--recommendations", required=True,
                        help="Path to recommendations CSV (output of batch_parse_results.py)")
    parser.add_argument("--gt_dir", required=True,
                        help="Directory with Semantic Scholar ground truth CSVs (DataFrameRankings_*.csv)")
    parser.add_argument("--output", required=True,
                        help="Output CSV path")
    parser.add_argument("--db_path", default=None,
                        help="Path to openalex_latest.duckdb for local OpenAlex pre-lookup (read_only=True)")
    parser.add_argument("--email", default=None,
                        help="Email for OpenAlex polite pool (higher API rate limits)")
    parser.add_argument("--no_api", action="store_true",
                        help="Skip the OpenAlex public API: rely only on the DuckDB snapshot. "
                             "Authors not in DuckDB will be treated as not-in-OpenAlex.")
    args = parser.parse_args()

    run(
        recommendations_path=args.recommendations,
        gt_dir=args.gt_dir,
        output_path=args.output,
        db_path=args.db_path,
        email=args.email,
        use_api=not args.no_api,
    )


if __name__ == "__main__":
    main()
