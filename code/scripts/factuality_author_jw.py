"""
factuality_author_jw.py — Jaro-Winkler name matching (vectorized via rapidfuzz.process.cdist).

Matching logic is identical to the original per-record version but ~20–50× faster:
  • references are pre-indexed per 2-char last-name block
  • per block, six cdist calls build a (n_queries × n_refs) int8 score matrix in C/SIMD
  • numpy argmax selects the best candidate per query in one pass
  • individual similarities are recomputed only for accepted matches (cheap post-pass)

Comparison pairs and thresholds:
  1. display_name   vs display_name   (0.85)  — counted twice (display_name + longest_name)
  2. first_name     vs first_name     (0.70)
  3. last_name      vs last_name      (0.70)
  4. second_name    vs second_name    (0.70)
  5. display_name   vs last_name      (0.70)
  6. display_name   vs first_name     (0.70)
  max possible score = 7  (display_name match contributes 2)

Usage (from code/scripts/):
  python factuality_author_jw.py \\
      --recommendations ../../results/summary/recommendations.csv \\
      --parquet /data/datasets/LLMScholar-Personas/data/semantic_scholar_data/clean/Researchers_Deduplicated_Genderize_Namsor.parquet \\
      --output  ../../results/summary/factuality_author_jw.csv \\
      [--min_matches 5] [--workers -1]
"""

import argparse
import hashlib
import logging
import pickle
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cdist as rfdist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

THRESHOLDS = {
    "display_name":  0.85,
    "longest_name":  0.85,
    "first_name":    0.70,
    "last_name":     0.70,
    "second_name":   0.70,
    "dn_vs_last":    0.70,
    "dn_vs_first":   0.70,
}

_TITLE_RE = re.compile(
    r'\b(dr|mr|mrs|ms|prof|professor|phd|ph\.d|md|m\.d|dsc|ing|lic|msc)\b\.?',
    re.IGNORECASE,
)
_PAREN_RE = re.compile(r'\(.*?\)')
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
    first  = tokens[0]
    last   = tokens[-1] if len(tokens) > 1 else ""
    second = tokens[1]  if len(tokens) > 2 else ""
    return full, first, second, last


def block_key(last: str) -> str:
    return last[:2] if len(last) >= 2 else last[:1]


# ── Reference index ──────────────────────────────────────────────────────────

def _cache_path(parquet_path: str) -> Path:
    p = Path(parquet_path)
    stat = p.stat()
    key = f"{stat.st_size}_{int(stat.st_mtime)}"
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    cache_dir = Path(__file__).resolve().parent.parent.parent / "results" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f".jw_index_{p.stem}_{h}.pkl"


def build_index(parquet_path: str, use_cache: bool = True) -> dict[str, dict]:
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
        cp = _cache_path(parquet_path)
        if cp.exists():
            logger.info("Loading cached index: %s", cp)
            with open(cp, "rb") as f:
                return pickle.load(f)

    logger.info("Loading parquet: %s", parquet_path)
    df = pd.read_parquet(parquet_path, columns=[
        "Researcher_id", "Name", "Field", "Combined_gender", "First_year", "Citations",
    ])
    df["gt_career_age"] = (2025 - df["First_year"]).clip(lower=0)
    logger.info("Parquet rows: %d", len(df))

    raw_blocks: dict[str, list] = defaultdict(list)

    for i, row in enumerate(df.itertuples(index=False)):
        norm = normalize(str(row.Name))
        dn, fn, sn, ln = parse_name(norm)
        raw_blocks[block_key(ln)].append((
            row.Researcher_id, row.Name, row.Field, row.Combined_gender,
            row.gt_career_age, row.Citations,
            dn, fn, sn, ln,
        ))
        if (i + 1) % 500_000 == 0:
            logger.info("  Indexed %d / %d", i + 1, len(df))

    # Convert to per-block dicts with pre-extracted string lists
    index: dict[str, dict] = {}
    for key, rows in raw_blocks.items():
        records = [
            {
                "researcher_id": r[0], "original_name": r[1],
                "field": r[2], "gender": r[3], "career_age": r[4], "citations": r[5],
                "display_name": r[6], "first_name": r[7], "second_name": r[8], "last_name": r[9],
            }
            for r in rows
        ]
        index[key] = {
            "records": records,
            "display": [r[6] for r in rows],
            "first":   [r[7] for r in rows],
            "second":  [r[8] for r in rows],
            "last":    [r[9] for r in rows],
        }

    logger.info("Index built: %d blocks", len(index))

    if use_cache:
        cp = _cache_path(parquet_path)
        logger.info("Saving index cache: %s", cp)
        with open(cp, "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    return index


# ── Vectorized block matching ────────────────────────────────────────────────

def _score_matrix(
    q_dn, q_fn, q_sn, q_ln,
    r_dn, r_fn, r_sn, r_ln,
    workers: int,
) -> np.ndarray:
    """
    Compute (n_queries × n_refs) int8 score matrix using six cdist calls.
    display_name match contributes 2 points (display_name + longest_name).
    """
    kw = dict(scorer=_JW, dtype=np.float32, workers=workers)
    scores = np.zeros((len(q_dn), len(r_dn)), dtype=np.int8)

    mat = rfdist(q_dn, r_dn, **kw)
    np.add(scores, (2 * (mat >= 0.85)).astype(np.int8), out=scores)
    del mat

    mat = rfdist(q_fn, r_fn, **kw)
    np.add(scores, (mat >= 0.70).view(np.uint8).astype(np.int8), out=scores)
    del mat

    mat = rfdist(q_ln, r_ln, **kw)
    np.add(scores, (mat >= 0.70).view(np.uint8).astype(np.int8), out=scores)
    del mat

    mat = rfdist(q_sn, r_sn, **kw)
    np.add(scores, (mat >= 0.70).view(np.uint8).astype(np.int8), out=scores)
    del mat

    mat = rfdist(q_dn, r_ln, **kw)
    np.add(scores, (mat >= 0.70).view(np.uint8).astype(np.int8), out=scores)
    del mat

    mat = rfdist(q_dn, r_fn, **kw)
    np.add(scores, (mat >= 0.70).view(np.uint8).astype(np.int8), out=scores)
    del mat

    return scores


def _sim_row(q: dict, ref: dict) -> dict:
    """Recompute per-field similarities for a single accepted match (post-pass)."""
    dn = _JW(q["display_name"], ref["display_name"])
    return {
        "sim_display_name":  round(dn, 4),
        "sim_longest_name":  round(dn, 4),
        "sim_first_name":    round(_JW(q["first_name"],   ref["first_name"]),  4),
        "sim_last_name":     round(_JW(q["last_name"],    ref["last_name"]),   4),
        "sim_second_name":   round(_JW(q["second_name"],  ref["second_name"]), 4),
        "sim_dn_vs_last":    round(_JW(q["display_name"], ref["last_name"]),   4),
        "sim_dn_vs_first":   round(_JW(q["display_name"], ref["first_name"]),  4),
        "sim_alternative_names": None,
    }


def _matched_fields(sims: dict) -> str:
    return "|".join(
        k.replace("sim_", "") for k, v in sims.items()
        if v is not None and v >= THRESHOLDS.get(k.replace("sim_", ""), 1.0)
    )


def match_block(
    queries: list[dict],
    block: dict,
    min_matches: int,
    workers: int,
) -> dict[str, dict]:
    """
    Match all queries against a reference block.
    Chunks query list to cap memory at ~200 MB per cdist matrix.
    Returns {raw_name: result_dict}.
    """
    r_dn = block["display"]
    r_fn = block["first"]
    r_sn = block["second"]
    r_ln = block["last"]
    recs = block["records"]

    n_refs = len(r_dn)
    # Aim for ≤200 M float32 values per matrix → chunk_size rows
    chunk_size = max(50, 50_000_000 // max(1, n_refs))

    cache: dict[str, dict] = {}

    for start in range(0, len(queries), chunk_size):
        chunk = queries[start: start + chunk_size]

        q_dn = [q["display_name"] for q in chunk]
        q_fn = [q["first_name"]   for q in chunk]
        q_sn = [q["second_name"]  for q in chunk]
        q_ln = [q["last_name"]    for q in chunk]

        scores = _score_matrix(q_dn, q_fn, q_sn, q_ln, r_dn, r_fn, r_sn, r_ln, workers)

        best_idxs   = scores.argmax(axis=1)
        best_scores = scores[np.arange(len(chunk)), best_idxs]

        for q, bidx, bscore in zip(chunk, best_idxs, best_scores):
            raw = q["_raw"]
            if int(bscore) >= min_matches:
                ref  = recs[int(bidx)]
                sims = _sim_row(q, ref)
                cache[raw] = {
                    "original_name": raw,
                    "matched_name":  ref["original_name"],
                    "researcher_id": ref["researcher_id"],
                    "match_score":   int(bscore),
                    "matched_fields": _matched_fields(sims),
                    "author_status": "found",
                    "gt_field":      ref["field"],
                    "gt_gender":     ref["gender"],
                    "gt_career_age": ref["career_age"],
                    "gt_citations":  ref["citations"],
                    **sims,
                }
            else:
                cache[raw] = {
                    "original_name": raw,
                    "matched_name":  None,
                    "researcher_id": None,
                    "match_score":   int(bscore),
                    "matched_fields": None,
                    "author_status": "hallucinated",
                    "gt_field":      None,
                    "gt_gender":     None,
                    "gt_career_age": None,
                    "gt_citations":  None,
                    **{f"sim_{k}": None for k in THRESHOLDS},
                }

    return cache


# ── Main pipeline ────────────────────────────────────────────────────────────

def run(
    recommendations_path: str,
    parquet_path: str,
    output_path: str,
    min_matches: int = 5,
    workers: int = -1,
    use_cache: bool = True,
) -> None:
    logger.info("Loading recommendations: %s", recommendations_path)
    df = pd.read_csv(recommendations_path, low_memory=False)
    logger.info("Recommendations: %d rows", len(df))

    df["_query"] = (
        df["name"].fillna("").astype(str) + " " + df["lastname"].fillna("").astype(str)
    ).str.strip()

    index = build_index(parquet_path, use_cache=use_cache)

    # Parse and group unique queries by block key
    unique_raws = [q for q in df["_query"].dropna().unique() if q.strip()]
    logger.info("Unique non-empty author names: %d", len(unique_raws))

    by_block: dict[str, list[dict]] = defaultdict(list)
    for raw in unique_raws:
        dn, fn, sn, ln = parse_name(normalize(raw))
        key = block_key(ln)
        if key not in index and len(key) == 2:
            key = key[:1]
        by_block[key].append({
            "_raw": raw,
            "display_name": dn, "first_name": fn,
            "second_name": sn, "last_name": ln,
        })

    # Match block by block
    global_cache: dict[str, dict] = {}
    n_blocks = len(by_block)
    _null_row = lambda raw: {
        "original_name": raw, "matched_name": None, "researcher_id": None,
        "match_score": 0, "matched_fields": None, "author_status": "hallucinated",
        "gt_field": None, "gt_gender": None, "gt_career_age": None, "gt_citations": None,
        **{f"sim_{k}": None for k in THRESHOLDS},
    }

    for b_idx, (key, queries) in enumerate(by_block.items(), 1):
        if key not in index:
            for q in queries:
                global_cache[q["_raw"]] = _null_row(q["_raw"])
        else:
            block_results = match_block(queries, index[key], min_matches, workers)
            global_cache.update(block_results)

        if b_idx % 100 == 0 or b_idx == n_blocks:
            done = sum(1 for v in global_cache.values() if v["author_status"] == "found")
            logger.info(
                "  Block %4d / %d  cache=%d  found=%d (%.1f%%)",
                b_idx, n_blocks, len(global_cache), done,
                100 * done / max(1, len(global_cache)),
            )

    # Map results back to all rows
    result_rows = [
        global_cache.get(q, _null_row(q)) for q in df["_query"]
    ]
    result_df = pd.DataFrame(result_rows)
    out = pd.concat(
        [df.drop(columns=["_query"]).reset_index(drop=True),
         result_df.reset_index(drop=True)],
        axis=1,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info("Saved %d rows → %s", len(out), output_path)

    n = len(out)
    for status, count in out["author_status"].value_counts().items():
        logger.info("  %-20s %7d  (%.1f%%)", status, count, 100 * count / n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vectorized Jaro-Winkler factuality matching (rapidfuzz cdist)"
    )
    parser.add_argument("--recommendations", required=True)
    parser.add_argument("--parquet",         required=True)
    parser.add_argument("--output",          required=True)
    parser.add_argument("--min_matches", type=int, default=5)
    parser.add_argument("--workers",  type=int, default=-1,
                        help="CPU workers for cdist (-1 = all cores, default)")
    parser.add_argument("--no_cache", action="store_true",
                        help="Rebuild the reference index even if a cache exists")
    args = parser.parse_args()

    run(
        recommendations_path=args.recommendations,
        parquet_path=args.parquet,
        output_path=args.output,
        min_matches=args.min_matches,
        workers=args.workers,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
