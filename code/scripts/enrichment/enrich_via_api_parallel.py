"""
enrich_via_api_parallel.py — Same as enrich_via_api.py but parallelised
with a ThreadPoolExecutor. ~10x faster (10 workers, polite pool).

Strategy:
  • Same /authors?filter=ids.openalex:A1|…|A50 endpoint, batch of 50.
  • N concurrent workers (default 10).
  • Shared records dict guarded by a lock; checkpoint every N batches.
  • Resume-aware: skips oa_ids already in the output CSV.

Usage (from code/scripts/):
  python enrich_via_api_parallel.py \\
      --input  ../../../results/summary/factuality_author.csv \\
      --output ../../../results/summary/oa_enrichment.csv \\
      --email  you@example.com \\
      --workers 10
"""

import argparse
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OPENALEX_API   = "https://api.openalex.org/authors"
BATCH_SIZE     = 50
API_TIMEOUT    = 30
API_RETRIES    = 4
API_RETRY_WAIT = 2
CHECKPOINT_EVERY = 100   # batches


# ── helpers ────────────────────────────────────────────────────────────────────

def _short_id(oa_id: str) -> str:
    return oa_id.rstrip("/").rsplit("/", 1)[-1]


def _load_oa_ids(input_path: str) -> list[str]:
    logger.info("Loading: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False, usecols=["oa_id"])
    ids = df["oa_id"].dropna().unique().tolist()
    logger.info("Unique oa_ids to enrich: %d", len(ids))
    return ids


def _load_existing(output_path: str) -> dict[str, dict]:
    if not os.path.exists(output_path):
        return {}
    df = pd.read_csv(output_path, low_memory=False)
    if df.empty:
        return {}
    df = df.dropna(subset=["oa_id"])
    out = {
        row["oa_id"]: {
            "oa_country_code":     row.get("oa_country_code"),
            "oa_last_institution": row.get("oa_last_institution"),
        }
        for _, row in df.iterrows()
    }
    logger.info("Resuming: %d oa_ids already enriched in %s", len(out), output_path)
    return out


def _save(records: dict[str, dict], output_path: str) -> None:
    df = pd.DataFrame([{"oa_id": k, **v} for k, v in records.items()])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)


def _extract(author: dict) -> dict:
    insts = author.get("last_known_institutions") or []
    inst  = insts[0] if insts else (author.get("last_known_institution") or {})
    return {
        "oa_country_code":     inst.get("country_code"),
        "oa_last_institution": inst.get("display_name"),
    }


# ── worker ─────────────────────────────────────────────────────────────────────

def _fetch_batch(session: requests.Session, ids: list[str], params: dict) -> dict[str, dict]:
    """Fetch one batch with retries; return {full_id: extracted_record}."""
    short = "|".join(_short_id(i) for i in ids)
    p = {**params, "filter": f"ids.openalex:{short}", "per-page": BATCH_SIZE}
    for attempt in range(1, API_RETRIES + 1):
        try:
            resp = session.get(OPENALEX_API, params=p, timeout=API_TIMEOUT)
            if resp.status_code == 429:
                wait = API_RETRY_WAIT * (2 ** (attempt - 1))
                time.sleep(wait)
                continue
            resp.raise_for_status()
            results = resp.json().get("results", [])
            by_short = {_short_id(a.get("id", "")): a for a in results if a.get("id")}
            out: dict[str, dict] = {}
            for full in ids:
                a = by_short.get(_short_id(full))
                out[full] = _extract(a) if a else {"oa_country_code": None, "oa_last_institution": None}
            return out
        except requests.RequestException as exc:
            wait = API_RETRY_WAIT * (2 ** (attempt - 1))
            logger.warning("API error (attempt %d/%d): %s — retry in %ds",
                           attempt, API_RETRIES, exc, wait)
            time.sleep(wait)
    # Final fallback: empty records for the chunk so we don't retry forever
    logger.error("Batch failed after %d attempts: %s …", API_RETRIES, short[:80])
    return {full: {"oa_country_code": None, "oa_last_institution": None} for full in ids}


# ── main ───────────────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str, email: str | None, workers: int) -> None:
    oa_ids  = _load_oa_ids(input_path)
    records = _load_existing(output_path)
    pending = [i for i in oa_ids if i not in records]
    logger.info("Pending: %d  (already enriched: %d)  workers: %d", len(pending), len(records), workers)
    if not pending:
        logger.info("Nothing to do.")
        return

    chunks = [pending[i : i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    logger.info("Total batches to fetch: %d", len(chunks))

    # Each worker gets its own session for connection pooling
    sessions = [requests.Session() for _ in range(workers)]
    for s in sessions:
        s.headers.update({"User-Agent": "factuality_enrich/1.0"})
    params: dict = {}
    if email:
        params["mailto"] = email

    lock = threading.Lock()
    counter = {"done": 0}

    def task(idx_chunk):
        idx, chunk = idx_chunk
        sess = sessions[idx % workers]
        return _fetch_batch(sess, chunk, params)

    pbar = tqdm(total=len(chunks), desc="Batches", unit="batch")
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(task, (i, c)): i for i, c in enumerate(chunks)}
        for fut in as_completed(futures):
            try:
                batch_records = fut.result()
            except Exception as exc:
                logger.error("Worker raised: %s", exc)
                batch_records = {}

            with lock:
                records.update(batch_records)
                counter["done"] += 1
                done = counter["done"]
                if done % CHECKPOINT_EVERY == 0:
                    _save(records, output_path)
                    elapsed = time.time() - t_start
                    rate = done * BATCH_SIZE / elapsed if elapsed > 0 else 0
                    eta = (len(chunks) - done) * BATCH_SIZE / rate if rate > 0 else float("inf")
                    pbar.set_postfix({
                        "saved":   len(records),
                        "rate/s":  f"{rate:.1f}",
                        "eta_min": f"{eta/60:.1f}",
                    })
            pbar.update(1)

    pbar.close()
    _save(records, output_path)

    n_country = sum(1 for v in records.values() if v.get("oa_country_code"))
    elapsed   = time.time() - t_start
    logger.info("Done. Total enriched: %d   with country: %d   wall: %.1f min",
                len(records), n_country, elapsed / 60)
    logger.info("Saved → %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel batch fetch of country/institution per oa_id from OpenAlex"
    )
    parser.add_argument("--input",   required=True)
    parser.add_argument("--output",  required=True)
    parser.add_argument("--email",   default=None)
    parser.add_argument("--workers", type=int, default=10,
                        help="Concurrent worker threads (default: 10)")
    args = parser.parse_args()

    run(args.input, args.output, args.email, args.workers)


if __name__ == "__main__":
    main()
