"""
factuality_author_jw.py — Jaro-Winkler name matching (vectorized via rapidfuzz.process.cdist).

Simplified matching: a single display_name (full name) vs display_name comparison
against the SS researcher, with a single threshold. The previous version (scoring 5-of-7 over
6 components — first, last, second, dn_vs_last, dn_vs_first, display_name×2)
generated many homonyms: a name could exceed the threshold by accumulating partial
matches without display_name itself matching well. Manual sampling showed that
most false positives came from that additive effect.

Algorithm:
  • references pre-indexed per 2-char last-name block
  • per block, ONE cdist call computes the (n_queries × n_refs) JW similarity
    matrix between display_names (full names normalized)
  • argmax selects the best candidate per query
  • accepted if JW >= dn_threshold (default 0.85)
  • per-component sims (first/last/second/dn_vs_last/dn_vs_first) are recomputed
    for accepted matches as information, but do NOT affect the decision

Output columns:
  author_status ('found' | 'hallucinated'), matched_name, researcher_id,
  match_score (JW similarity float 0-1, same semantics as oa_match_score),
  matched_fields (always 'display_name' for found, None for hallucinated),
  gt_field, gt_gender, gt_career_age, gt_citations,
  sim_display_name, sim_longest_name, sim_first_name, sim_last_name,
  sim_second_name, sim_dn_vs_last, sim_dn_vs_first, sim_alternative_names

Usage (from code/, with PYTHONPATH=.):
  python scripts/factuality/factuality_author_jw.py \\
      --recommendations ../results/summary/recommendations.csv \\
      --parquet <path_to_ss_parquet> \\
      --output  ../results/summary/factuality_author_jw.csv \\
      [--dn_threshold 0.85] [--workers -1] [--reference-year 2025]

The --parquet default comes from [data].ss_parquet in config.ini.
"""

import argparse
import hashlib
import re
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cdist as rfdist

from libs.utils.ios import load_pickle, save_pickle, write_output_csv
from libs.utils.logging import log_value_counts, setup_logging

logger = setup_logging()

DN_THRESHOLD = 0.85  # JW threshold for (display_name, display_name)
PER_TOKEN_THRESHOLD = (
    0.95  # JW threshold per token (>1 char). Initials (1 char) are ignored.
)

# THRESHOLDS is kept ONLY so that `matched_fields` and the output sim_* columns
# retain the same names as before (downstream compatibility). The match decision
# uses exclusively DN_THRESHOLD + the per-token filter.
THRESHOLDS = {
    "display_name": DN_THRESHOLD,
    "longest_name": DN_THRESHOLD,
    "first_name": 0.70,
    "last_name": 0.70,
    "second_name": 0.70,
    "dn_vs_last": 0.70,
    "dn_vs_first": 0.70,
}

_TITLE_RE = re.compile(
    r"\b(dr|mr|mrs|ms|prof|professor|phd|ph\.d|md|m\.d|dsc|ing|lic|msc)\b\.?",
    re.IGNORECASE,
)
_PAREN_RE = re.compile(r"\(.*?\)")
_JW = JaroWinkler.similarity


# ── Text helpers ─────────────────────────────────────────────────────────────


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _TITLE_RE.sub(" ", text)
    text = _PAREN_RE.sub(" ", text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def parse_name(full: str) -> tuple[str, str, str, str]:
    """Returns (display_name, first_name, second_name, last_name)."""
    tokens = full.split()
    if not tokens:
        return "", "", "", ""
    first = tokens[0]
    last = tokens[-1] if len(tokens) > 1 else ""
    second = tokens[1] if len(tokens) > 2 else ""
    return full, first, second, last


def block_key(last: str) -> str:
    return last[:2] if len(last) >= 2 else last[:1]


# ── Reference index ──────────────────────────────────────────────────────────


REFERENCE_YEAR = 2025


def _cache_path(parquet_path: str, reference_year: int) -> Path:
    p = Path(parquet_path)
    stat = p.stat()
    key = f"{stat.st_size}_{int(stat.st_mtime)}_ry{reference_year}"
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    cache_dir = Path(__file__).resolve().parent.parent.parent.parent / "results" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f".jw_index_{p.stem}_{h}.pkl"


def build_index(
    parquet_path: str, use_cache: bool = True, reference_year: int = REFERENCE_YEAR
) -> dict[str, dict]:
    """
    Load parquet and build a per-block structure:
      block_key → {
          "records":  list[dict],   # full metadata per researcher
          "display":  list[str],
          "first":    list[str],
          "last":     list[str],
          "second":   list[str],
      }
    """
    if use_cache:
        cp = _cache_path(parquet_path, reference_year)
        cached = load_pickle(cp, logger=logger)
        if cached is not None:
            return cached

    logger.info("Loading parquet: %s", parquet_path)
    df = pd.read_parquet(
        parquet_path,
        columns=[
            "Researcher_id",
            "Name",
            "Field",
            "Combined_gender",
            "First_year",
            "Citations",
        ],
    )
    df["gt_career_age"] = (reference_year - df["First_year"]).clip(lower=0)
    logger.info("Parquet rows: %d", len(df))

    raw_blocks: dict[str, list] = defaultdict(list)

    for i, row in enumerate(df.itertuples(index=False)):
        norm = normalize(str(row.Name))
        dn, fn, sn, ln = parse_name(norm)
        raw_blocks[block_key(ln)].append(
            (
                row.Researcher_id,
                row.Name,
                row.Field,
                row.Combined_gender,
                row.gt_career_age,
                row.Citations,
                dn,
                fn,
                sn,
                ln,
            )
        )
        if (i + 1) % 500_000 == 0:
            logger.info("  Indexed %d / %d", i + 1, len(df))

    # Convert to per-block dicts with pre-extracted string lists
    index: dict[str, dict] = {}
    for key, rows in raw_blocks.items():
        records = [
            {
                "researcher_id": r[0],
                "original_name": r[1],
                "field": r[2],
                "gender": r[3],
                "career_age": r[4],
                "citations": r[5],
                "display_name": r[6],
                "first_name": r[7],
                "second_name": r[8],
                "last_name": r[9],
            }
            for r in rows
        ]
        index[key] = {
            "records": records,
            "display": [r[6] for r in rows],
            "first": [r[7] for r in rows],
            "second": [r[8] for r in rows],
            "last": [r[9] for r in rows],
        }

    logger.info("Index built: %d blocks", len(index))

    if use_cache:
        save_pickle(index, _cache_path(parquet_path), logger=logger)

    return index


# ── Vectorized block matching ────────────────────────────────────────────────


def _dn_score_matrix(q_dn, r_dn, workers: int) -> np.ndarray:
    """
    Compute (n_queries × n_refs) float32 matrix of JW similarities between
    query display_names and reference display_names. Single cdist call.
    """
    return rfdist(q_dn, r_dn, scorer=_JW, dtype=np.float32, workers=workers)


def _passes_per_token_check(q_dn: str, r_dn: str, threshold: float) -> bool:
    """
    Post-JW filter: for each token >1 char in the query, require that a
    token >1 char exists in the ref with JW >= threshold. 1-char tokens
    (initials) are ignored on both the query and ref sides.

    Catches cases like:
      "john doe smith" → "john e smith" : doe vs {john,smith} → max ≈ 0.5 → REJECT
      "mariette jacobs" → "maretha jacobs" : mariette vs {maretha,jacobs} → max ≈ 0.85 → REJECT
    Accepts:
      "maria gonzalez" → "maria gonzalez" : all tokens match → ACCEPT
    """
    q_toks = [t for t in q_dn.split() if len(t) > 1]
    r_toks = [t for t in r_dn.split() if len(t) > 1]
    if not q_toks:
        return True  # trivial query, let it pass
    if not r_toks:
        return False  # ref with no useful tokens
    for qt in q_toks:
        best = max(_JW(qt, rt) for rt in r_toks)
        if best < threshold:
            return False
    return True


def _sim_row(q: dict, ref: dict) -> dict:
    """Recompute per-field similarities for a single accepted match (post-pass)."""
    dn = _JW(q["display_name"], ref["display_name"])
    return {
        "sim_display_name": round(dn, 4),
        "sim_longest_name": round(dn, 4),
        "sim_first_name": round(_JW(q["first_name"], ref["first_name"]), 4),
        "sim_last_name": round(_JW(q["last_name"], ref["last_name"]), 4),
        "sim_second_name": round(_JW(q["second_name"], ref["second_name"]), 4),
        "sim_dn_vs_last": round(_JW(q["display_name"], ref["last_name"]), 4),
        "sim_dn_vs_first": round(_JW(q["display_name"], ref["first_name"]), 4),
        "sim_alternative_names": None,
    }


def _matched_fields(sims: dict) -> str:
    return "|".join(
        k.replace("sim_", "")
        for k, v in sims.items()
        if v is not None and v >= THRESHOLDS.get(k.replace("sim_", ""), 1.0)
    )


def match_block(
    queries: list[dict],
    block: dict,
    dn_threshold: float,
    workers: int,
) -> dict[str, dict]:
    """
    Match all queries against a reference block by JW(display_name, display_name).
    Chunks query list to cap memory at ~200 MB per cdist matrix.
    Returns {raw_name: result_dict}.
    """
    r_dn = block["display"]
    recs = block["records"]

    n_refs = len(r_dn)
    # Aim for ≤200 M float32 values per matrix → chunk_size rows
    chunk_size = max(50, 50_000_000 // max(1, n_refs))

    cache: dict[str, dict] = {}

    for start in range(0, len(queries), chunk_size):
        chunk = queries[start : start + chunk_size]
        q_dn = [q["display_name"] for q in chunk]

        scores = _dn_score_matrix(q_dn, r_dn, workers)

        best_idxs = scores.argmax(axis=1)
        best_scores = scores[np.arange(len(chunk)), best_idxs]

        for q, bidx, bscore in zip(chunk, best_idxs, best_scores):
            raw = q["_raw"]
            score = float(bscore)
            ref = recs[int(bidx)]
            passes_token_check = score >= dn_threshold and _passes_per_token_check(
                q["display_name"], ref["display_name"], PER_TOKEN_THRESHOLD
            )
            if passes_token_check:
                sims = _sim_row(q, ref)
                cache[raw] = {
                    "original_name": raw,
                    "matched_name": ref["original_name"],
                    "researcher_id": ref["researcher_id"],
                    "match_score": round(score, 4),
                    "matched_fields": "display_name",
                    "author_status": "found",
                    "gt_field": ref["field"],
                    "gt_gender": ref["gender"],
                    "gt_career_age": ref["career_age"],
                    "gt_citations": ref["citations"],
                    **sims,
                }
            else:
                cache[raw] = {
                    "original_name": raw,
                    "matched_name": None,
                    "researcher_id": None,
                    "match_score": round(score, 4),
                    "matched_fields": None,
                    "author_status": "hallucinated",
                    "gt_field": None,
                    "gt_gender": None,
                    "gt_career_age": None,
                    "gt_citations": None,
                    **{f"sim_{k}": None for k in THRESHOLDS},
                }

    return cache


# ── Main pipeline ────────────────────────────────────────────────────────────


def run(
    recommendations_path: str,
    parquet_path: str,
    output_path: str,
    dn_threshold: float = DN_THRESHOLD,
    workers: int = -1,
    use_cache: bool = True,
    reference_year: int = REFERENCE_YEAR,
) -> None:
    logger.info("Loading recommendations: %s", recommendations_path)
    df = pd.read_csv(recommendations_path, low_memory=False)
    logger.info("Recommendations: %d rows", len(df))

    df["_query"] = (
        df["name"].fillna("").astype(str) + " " + df["lastname"].fillna("").astype(str)
    ).str.strip()

    index = build_index(
        parquet_path, use_cache=use_cache, reference_year=reference_year
    )

    # Parse and group unique queries by block key
    unique_raws = [q for q in df["_query"].dropna().unique() if q.strip()]
    logger.info("Unique non-empty author names: %d", len(unique_raws))

    by_block: dict[str, list[dict]] = defaultdict(list)
    for raw in unique_raws:
        dn, fn, sn, ln = parse_name(normalize(raw))
        key = block_key(ln)
        if key not in index and len(key) == 2:
            key = key[:1]
        by_block[key].append(
            {
                "_raw": raw,
                "display_name": dn,
                "first_name": fn,
                "second_name": sn,
                "last_name": ln,
            }
        )

    # Match block by block
    global_cache: dict[str, dict] = {}
    n_blocks = len(by_block)
    _null_row = lambda raw: {
        "original_name": raw,
        "matched_name": None,
        "researcher_id": None,
        "match_score": 0.0,
        "matched_fields": None,
        "author_status": "hallucinated",
        "gt_field": None,
        "gt_gender": None,
        "gt_career_age": None,
        "gt_citations": None,
        **{f"sim_{k}": None for k in THRESHOLDS},
    }

    for b_idx, (key, queries) in enumerate(by_block.items(), 1):
        if key not in index:
            for q in queries:
                global_cache[q["_raw"]] = _null_row(q["_raw"])
        else:
            block_results = match_block(queries, index[key], dn_threshold, workers)
            global_cache.update(block_results)

        if b_idx % 100 == 0 or b_idx == n_blocks:
            done = sum(
                1 for v in global_cache.values() if v["author_status"] == "found"
            )
            logger.info(
                "  Block %4d / %d  cache=%d  found=%d (%.1f%%)",
                b_idx,
                n_blocks,
                len(global_cache),
                done,
                100 * done / max(1, len(global_cache)),
            )

    # Map results back to all rows
    result_rows = [global_cache.get(q, _null_row(q)) for q in df["_query"]]
    result_df = pd.DataFrame(result_rows)
    out = pd.concat(
        [
            df.drop(columns=["_query"]).reset_index(drop=True),
            result_df.reset_index(drop=True),
        ],
        axis=1,
    )

    write_output_csv(out, output_path, logger=logger)
    log_value_counts(out, "author_status", title="Author status distribution", logger=logger)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vectorized Jaro-Winkler factuality matching (rapidfuzz cdist)"
    )
    parser.add_argument("--recommendations", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dn_threshold",
        type=float,
        default=DN_THRESHOLD,
        help=f"JW threshold for display_name vs display_name (default {DN_THRESHOLD})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="CPU workers for cdist (-1 = all cores, default)",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Rebuild the reference index even if a cache exists",
    )
    parser.add_argument(
        "--reference-year",
        type=int,
        default=REFERENCE_YEAR,
        help=(
            "Year used for career age (year - First_year). Fixed default "
            f"{REFERENCE_YEAR} for reproducibility (NOT datetime.now())."
        ),
    )
    args = parser.parse_args()

    run(
        recommendations_path=args.recommendations,
        parquet_path=args.parquet,
        output_path=args.output,
        dn_threshold=args.dn_threshold,
        workers=args.workers,
        use_cache=not args.no_cache,
        reference_year=args.reference_year,
    )


if __name__ == "__main__":
    main()
