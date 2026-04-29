"""
enrich_via_api.py — Fetch country + institution for every oa_id via OpenAlex's
batch /authors endpoint (up to 50 ids per request).

Strategy:
  1. Read factuality_author.csv, collect unique oa_ids.
  2. Split into chunks of 50.
  3. GET /authors?filter=ids.openalex:A1|A2|...|A50&per-page=50&mailto=…
  4. Extract last_known_institutions[0].{country_code,display_name}.
  5. Save lookup CSV. Resumes from existing output if interrupted.

Progress is shown via tqdm (stderr).

Usage (from code/scripts/):
  python enrich_via_api.py \\
      --input  ../../results/summary/factuality_author.csv \\
      --output ../../results/summary/oa_enrichment.csv \\
      --email  jmunizagatorres@gmail.com
"""

import argparse
import logging
import os
import time

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OPENALEX_API   = "https://api.openalex.org/authors"
BATCH_SIZE     = 50    # OpenAlex max per filter=ids.openalex
API_TIMEOUT    = 30
API_RETRIES    = 4
API_RETRY_WAIT = 2     # seconds, doubled per attempt
CHECKPOINT_EVERY = 50  # save partial CSV every N batches


def _short_id(oa_id: str) -> str:
    """Convert 'https://openalex.org/A5003003822' → 'A5003003822'."""
    return oa_id.rstrip("/").rsplit("/", 1)[-1]


def _load_oa_ids(input_path: str) -> list[str]:
    logger.info("Loading: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False, usecols=["oa_id"])
    ids = df["oa_id"].dropna().unique().tolist()
    logger.info("Unique oa_ids to enrich: %d", len(ids))
    return ids


def _load_existing(output_path: str) -> dict[str, dict]:
    """Resume support: load any oa_ids already enriched."""
    if not os.path.exists(output_path):
        return {}
    df = pd.read_csv(output_path, low_memory=False)
    if df.empty:
        return {}
    df = df.dropna(subset=["oa_id"])
    out = {}
    for _, row in df.iterrows():
        out[row["oa_id"]] = {
            "oa_country_code":     row.get("oa_country_code"),
            "oa_last_institution": row.get("oa_last_institution"),
        }
    logger.info("Resuming: %d oa_ids already enriched in %s", len(out), output_path)
    return out


def _save(records: dict[str, dict], output_path: str) -> None:
    df = pd.DataFrame(
        [{"oa_id": k, **v} for k, v in records.items()]
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)


def _fetch_batch(session: requests.Session, ids: list[str], params: dict) -> list[dict]:
    """Fetch one batch with retries on 429/5xx."""
    short = "|".join(_short_id(i) for i in ids)
    p = {**params, "filter": f"ids.openalex:{short}", "per-page": BATCH_SIZE}
    for attempt in range(1, API_RETRIES + 1):
        try:
            resp = session.get(OPENALEX_API, params=p, timeout=API_TIMEOUT)
            if resp.status_code == 429:
                wait = API_RETRY_WAIT * (2 ** (attempt - 1))
                logger.warning("429 rate-limited, sleeping %ds (attempt %d/%d)", wait, attempt, API_RETRIES)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json().get("results", [])
        except requests.RequestException as exc:
            wait = API_RETRY_WAIT * (2 ** (attempt - 1))
            logger.warning("API error (attempt %d/%d): %s — retrying in %ds",
                           attempt, API_RETRIES, exc, wait)
            time.sleep(wait)
    logger.error("Batch failed after %d attempts: %s …", API_RETRIES, short[:80])
    return []


def _extract(author: dict) -> dict:
    insts = author.get("last_known_institutions") or []
    inst  = insts[0] if insts else (author.get("last_known_institution") or {})
    return {
        "oa_country_code":     inst.get("country_code"),
        "oa_last_institution": inst.get("display_name"),
    }


def run(input_path: str, output_path: str, email: str | None) -> None:
    oa_ids = _load_oa_ids(input_path)

    # Resume-aware: keep already-fetched records
    records = _load_existing(output_path)
    pending = [i for i in oa_ids if i not in records]
    logger.info("Pending: %d  (already enriched: %d)", len(pending), len(records))
    if not pending:
        logger.info("Nothing to do.")
        return

    session = requests.Session()
    session.headers.update({"User-Agent": "factuality_enrich/1.0"})
    params: dict = {}
    if email:
        params["mailto"] = email

    # Iterate in chunks of 50, with tqdm progress
    n_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE
    pbar = tqdm(total=n_batches, desc="Batches", unit="batch")
    t_start = time.time()

    for batch_idx in range(n_batches):
        chunk = pending[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        results = _fetch_batch(session, chunk, params)

        # Map by short id then back to full id (some short_ids may differ from full)
        by_short = {_short_id(a.get("id", "")): a for a in results if a.get("id")}
        for full_id in chunk:
            short = _short_id(full_id)
            if short in by_short:
                records[full_id] = _extract(by_short[short])
            else:
                # Author not returned by API (deleted/merged). Mark with empty so we don't re-query.
                records[full_id] = {"oa_country_code": None, "oa_last_institution": None}

        pbar.update(1)
        if (batch_idx + 1) % CHECKPOINT_EVERY == 0:
            _save(records, output_path)
            elapsed = time.time() - t_start
            done = (batch_idx + 1) * BATCH_SIZE
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(pending) - done) / rate if rate > 0 else float("inf")
            pbar.set_postfix({"saved": len(records), "rate/s": f"{rate:.1f}", "eta_min": f"{eta/60:.1f}"})

    pbar.close()
    _save(records, output_path)

    n_country = sum(1 for v in records.values() if v.get("oa_country_code"))
    elapsed = time.time() - t_start
    logger.info("Done. Total enriched: %d   with country: %d   wall: %.1f min",
                len(records), n_country, elapsed / 60)
    logger.info("Saved → %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch country + institution per oa_id via OpenAlex batch API"
    )
    parser.add_argument("--input",  required=True, help="Path to factuality_author.csv (oa_id column)")
    parser.add_argument("--output", required=True, help="Output CSV path: oa_id, oa_country_code, oa_last_institution")
    parser.add_argument("--email",  default=None, help="Email for OpenAlex polite pool")
    args = parser.parse_args()

    run(args.input, args.output, args.email)


if __name__ == "__main__":
    main()
