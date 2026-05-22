"""
factuality_affiliation.py — Step 3.5 of the factuality pipeline.

PLAN.md Task 2 — Affiliation factuality.
Structure cloned from factuality_location.py (Finding 3). Reuses
normalize_name from factuality_openalex.py (Finding 2) and rapidfuzz
(already in deps via factuality_author_jw.py).

Reads the output of factuality_location.py and decides whether the
LLM-claimed `current_affiliations` for each recommended author match any of
the institutions the author has actually been affiliated with according to
OpenAlex (historical, not just the most-recent).

Ground truth: OpenAlex. Semantic Scholar's deduplicated parquet does not
carry per-researcher institution, so affiliation factuality can only be
verified against OA.

OA institution history is reconstructed from the existing year-chunk
parquets at /data/asanchez/duckdb_enrich/oa_works_agg_chunks (built by
factuality_openalex.py). Each chunk holds one (country, inst_name) per
(oa_id, year_chunk) via ARG_MAX over publication_year, so DISTINCT across
all 20 chunks yields the author's institution history at chunk-granularity.

Match logic: rapidfuzz.fuzz.token_set_ratio with a threshold of 80 between
each LLM affiliation string and each OA historical institution string.
A row is `affiliation_match` if ANY (llm, oa) pair clears the threshold.

Output columns added:
  affiliation_llm                 serialized list of LLM affiliations
  affiliation_oa_all              serialized list of OA historical institutions
  affiliation_best_match_score    best token_set_ratio score (0-100) or None
  affiliation_best_match_oa       OA institution that produced the best score
  affiliation_status              {affiliation_match | affiliation_mismatch
                                   | affiliation_unknown | not_applicable}

Usage (from code/scripts/factuality/):
  python factuality_affiliation.py \\
      --input  ../../../results/results/summary_v2/factuality_location.csv \\
      --output ../../../results/results/summary_v2/factuality_affiliation.csv
"""

import argparse
import ast
import json
import logging
import os

import pandas as pd

# Reuse the existing accent/case normalizer from the OA step so matching is
# stable across mojibake ("Université" vs "Universite") and case differences.
from factuality_openalex import normalize_name as _normalize_for_match
from rapidfuzz import fuzz

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

STATUS_MATCH = "affiliation_match"
STATUS_MISMATCH = "affiliation_mismatch"
STATUS_UNKNOWN = "affiliation_unknown"
STATUS_NOT_APPLICABLE = "not_applicable"

AUTHOR_HALLUCINATED = "hallucinated"

# Threshold on rapidfuzz.fuzz.token_set_ratio (0-100). 85 was chosen because
# 80 lets the shared token "University" alone produce false positives like
# "University of Cape Town" vs "Yale University" (score 80). 85 still passes
# legitimate fuzzy matches (LLM-verbose strings vs short OA names score 100
# via subset-token mechanics) while rejecting the single-shared-token case.
TOKEN_SET_RATIO_THRESHOLD = 85

# Where factuality_openalex.py wrote per-year aggregations.
WORKS_AGG_DIR = "/data/asanchez/duckdb_enrich/oa_works_agg_chunks"


# ── Parsers ────────────────────────────────────────────────────────────────────


def parse_llm_affiliations(raw) -> list[str]:
    """Parse the `current_affiliations` cell into a list of affiliation strings.

    The cell is typically a Python-literal string like
        "[{'position': '...', 'affiliation': 'MIT'}]"
    but may also be None, NaN, empty, or already a list.
    """
    # NaN or None → no affiliations.
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    # If it already comes as a list (rare but possible if pandas didn't serialize it), use it.
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        s = raw.strip()
        # Common "empty" sentinels — before attempting to parse.
        if not s or s in ("[]", "null", "None"):
            return []
        # First ast.literal_eval (accepts Python-style single quotes),
        # then json as a fallback for strict JSON strings.
        try:
            items = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            try:
                items = json.loads(s)
            except Exception:
                return []
    else:
        return []

    # If the parsed value is not a list (e.g. a lone dict), discard.
    if not isinstance(items, list):
        return []

    # Extract the affiliation string from each item: we accept dicts with
    # different key conventions or bare strings.
    affs: list[str] = []
    for it in items:
        if isinstance(it, dict):
            v = it.get("affiliation") or it.get("Affiliation") or it.get("institution")
            if isinstance(v, str) and v.strip():
                affs.append(v.strip())
        elif isinstance(it, str) and it.strip():
            affs.append(it.strip())
    return affs


# ── OA institution history ─────────────────────────────────────────────────────


def fetch_oa_institutions(oa_ids: list[str]) -> dict[str, list[str]]:
    """Return {oa_id: [inst_name, …]} from the existing chunk parquets.

    Uses DuckDB read_parquet on the glob — does not open the full OA DB.
    If no chunks are on disk, returns an empty dict and the caller will
    fall back to oa_last_institution from the input CSV.
    """
    # Glob over the chunks already materialized by factuality_openalex.py
    # (one per year bin; UNNEST over authorships already done there).
    chunk_glob = os.path.join(WORKS_AGG_DIR, "chunk_*.parquet")
    # If the folder does not exist or is empty, abort cleanly: the caller will
    # use oa_last_institution as a proxy for "historical" (not ideal but allows
    # the pipeline to keep running on machines without the pre-generated chunks).
    if not any(
        os.path.exists(os.path.join(WORKS_AGG_DIR, f))
        for f in (os.listdir(WORKS_AGG_DIR) if os.path.isdir(WORKS_AGG_DIR) else [])
    ):
        logger.warning(
            "OA chunk parquets not found at %s — falling back to oa_last_institution only",
            WORKS_AGG_DIR,
        )
        return {}

    # duckdb is a soft dep — if not installed, also fallback.
    try:
        import duckdb  # noqa: F401
    except ImportError:
        logger.warning(
            "duckdb not installed — falling back to oa_last_institution only"
        )
        return {}

    # Deduplicate oa_ids: there are ~4M rows in the CSV but many fewer unique authors.
    unique = sorted({i for i in oa_ids if isinstance(i, str) and i})
    if not unique:
        return {}

    logger.info(
        "OA institutions: scanning %s for %d unique oa_ids …", chunk_glob, len(unique)
    )
    con = duckdb.connect()
    try:
        # Threads = all cores: the query is CPU-bound (LIST DISTINCT over
        # tens of millions of rows).
        cores = os.cpu_count() or 8
        con.execute(f"SET threads = {cores}")
        # Register the oa_ids as a virtual table to do an efficient JOIN
        # instead of a giant IN.
        con.register("query_oa_ids", pd.DataFrame({"id": unique}))
        # Single query: for each author, LIST(DISTINCT inst_name) over
        # all chunks → their full affiliation history.
        rows = con.execute(
            f"""
            SELECT p.oa_id, LIST(DISTINCT p.inst_name) AS institutions
              FROM read_parquet('{chunk_glob}') p
              JOIN query_oa_ids q ON p.oa_id = q.id
             WHERE p.inst_name IS NOT NULL
             GROUP BY p.oa_id
        """
        ).fetchall()
    except Exception as exc:
        # We don't want to kill the pipeline if DuckDB fails — degrade to fallback.
        logger.warning("OA institutions query failed: %s", exc)
        return {}
    finally:
        # Close the connection no matter what (frees RAM and locks).
        try:
            con.close()
        except Exception:
            pass

    # Build the final dict filtering empty or non-string entries.
    out: dict[str, list[str]] = {}
    for oa_id, inst_list in rows:
        out[oa_id] = [s for s in (inst_list or []) if isinstance(s, str) and s.strip()]
    logger.info("OA institutions: resolved %d / %d oa_ids", len(out), len(unique))
    return out


# ── Per-row decision ───────────────────────────────────────────────────────────


def match_affiliations(
    llm_affs: list[str], oa_affs: list[str]
) -> tuple[int | None, str | None]:
    """Return (best_score, best_oa_name) across the cartesian product."""
    # Without affiliations on either side there's no possible match — score = None.
    if not llm_affs or not oa_affs:
        return None, None
    # Pre-normalize once per string (lowercase + strip accents):
    # stabilizes "Université" vs "Universite" and similar.
    llm_norm = [_normalize_for_match(s) for s in llm_affs]
    oa_norm = [_normalize_for_match(s) for s in oa_affs]
    best_score = -1
    best_oa = None
    # Cartesian product LLM × OA — keep the best pair.
    for la, la_n in zip(llm_affs, llm_norm):
        if not la_n:
            continue
        for oa, oa_n in zip(oa_affs, oa_norm):
            if not oa_n:
                continue
            # token_set_ratio handles order differences and
            # subsets well ("MIT" vs "Massachusetts Institute of Technology").
            s = fuzz.token_set_ratio(la_n, oa_n)
            if s > best_score:
                best_score = s
                best_oa = oa  # keep the original OA name, not the normalized one
    return int(best_score), best_oa


def _decide(
    llm_affs: list[str], oa_affs: list[str], author_status: str
) -> tuple[str | None, str | None, int | None, str | None, str]:
    """Pure decision: return (llm_json, oa_json, score, best_oa, status)."""
    # Compute the best score and serialize the lists to JSON (to store them
    # readably in the output CSV).
    score, best_oa = match_affiliations(llm_affs, oa_affs)
    llm_json = json.dumps(llm_affs, ensure_ascii=False) if llm_affs else None
    oa_json = json.dumps(oa_affs, ensure_ascii=False) if oa_affs else None

    # Hallucinated author → no sense checking affiliation: not_applicable
    # (same schema as factuality_location.py).
    if author_status == AUTHOR_HALLUCINATED:
        return llm_json, oa_json, score, best_oa, STATUS_NOT_APPLICABLE
    # Missing info on either side → cannot decide match/mismatch.
    if not llm_affs or not oa_affs:
        return llm_json, oa_json, score, best_oa, STATUS_UNKNOWN
    # Above threshold → match; below → mismatch.
    if score is not None and score >= TOKEN_SET_RATIO_THRESHOLD:
        return llm_json, oa_json, score, best_oa, STATUS_MATCH
    return llm_json, oa_json, score, best_oa, STATUS_MISMATCH


# ── Main ───────────────────────────────────────────────────────────────────────


def run(input_path: str, output_path: str) -> None:
    logger.info("Loading: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    logger.info("Rows: %d", len(df))

    # Lookup OA → list of historical institutions (single DuckDB query
    # for all unique authors in the CSV).
    oa_ids = df["oa_id"].dropna().astype(str).tolist() if "oa_id" in df.columns else []
    oa_inst_cache = fetch_oa_institutions(oa_ids)

    # Cache parsed LLM affiliations by raw string (815k unique vs ~4M rows).
    # Parsing the JSON-ish in each row is expensive, but many rows share the
    # same raw string (same author recommended by multiple LLMs / runs).
    raw_aff_series = df["current_affiliations"]
    unique_raw = raw_aff_series.drop_duplicates()
    logger.info("Parsing %d unique current_affiliations strings …", len(unique_raw))
    llm_parse_cache: dict = {}
    for raw in unique_raw:
        # Store as a tuple so it can be used as a dict key below.
        llm_parse_cache[raw] = tuple(parse_llm_affiliations(raw))

    # Build OA-affs lookup per row: prefer historical cache, fall back to oa_last_institution.
    def _oa_for(oa_id, last_inst) -> tuple:
        # If we have the full DuckDB history, use it (better match).
        if isinstance(oa_id, str) and oa_id in oa_inst_cache:
            return tuple(oa_inst_cache[oa_id])
        # Otherwise, fall back to the last known institution — graceful degradation
        # for machines without the parquet chunks.
        if isinstance(last_inst, str) and last_inst.strip():
            return (last_inst,)
        return ()

    # Decide once per unique (llm_tuple, oa_tuple, author_status). Build the key
    # cheaply via row-wise tuple construction (no fuzzy work here).
    # Important: the fuzzy matching (token_set_ratio) is the expensive part — do it
    # once per unique combo, not per row.
    logger.info("Building per-row keys …")
    llm_tuples = raw_aff_series.map(llm_parse_cache).to_list()
    oa_ids_arr = df["oa_id"].to_list() if "oa_id" in df.columns else [None] * len(df)
    last_inst_arr = (
        df["oa_last_institution"].to_list()
        if "oa_last_institution" in df.columns
        else [None] * len(df)
    )
    status_arr = (
        df["author_status"].to_list()
        if "author_status" in df.columns
        else [None] * len(df)
    )

    decision_cache: dict = {}
    n_rows = len(df)
    # Build the per-row key: (LLM affiliations, OA affiliations, author_status).
    keys: list[tuple] = [None] * n_rows  # type: ignore[list-item]
    for i in range(n_rows):
        oa_tuple = _oa_for(oa_ids_arr[i], last_inst_arr[i])
        keys[i] = (llm_tuples[i], oa_tuple, status_arr[i])
    unique_keys = set(keys)
    logger.info(
        "Deciding %d unique (LLM-affs, OA-affs, author_status) combos …",
        len(unique_keys),
    )

    # The expensive work happens here: one decision per unique combo.
    for k in unique_keys:
        llm_t, oa_t, st = k
        decision_cache[k] = _decide(list(llm_t), list(oa_t), st)

    # Re-expand the results to the N rows of the original CSV.
    logger.info("Materializing columns …")
    aff_llm = [None] * n_rows
    aff_oa = [None] * n_rows
    aff_sc = [None] * n_rows
    aff_best = [None] * n_rows
    aff_st = [None] * n_rows
    for i, k in enumerate(keys):
        aff_llm[i], aff_oa[i], aff_sc[i], aff_best[i], aff_st[i] = decision_cache[k]

    # Attach the 5 new columns to the DataFrame.
    df["affiliation_llm"] = aff_llm
    df["affiliation_oa_all"] = aff_oa
    df["affiliation_best_match_score"] = aff_sc
    df["affiliation_best_match_oa"] = aff_best
    df["affiliation_status"] = aff_st

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows → %s", len(df), output_path)

    n = len(df)
    logger.info("Affiliation status distribution:")
    for status, count in df["affiliation_status"].value_counts().items():
        logger.info("  %-25s %6d  (%.1f%%)", status, count, 100 * count / n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3.5: verify LLM `current_affiliations` against the author's "
        "OpenAlex institution history"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to factuality_location.csv (output of step 3)",
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    run(args.input, args.output)


if __name__ == "__main__":
    main()
