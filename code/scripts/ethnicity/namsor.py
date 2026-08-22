#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# export PYTHONPATH="$PYTHONPATH:../../libs" # <-- run from code/scripts/ethnicity/
"""
NamSor verification of perceived ethnicity.

Pipeline:
  1. Read an input CSV with columns: name, lastname, location.
  2. Build a deduplicated batch of (firstName, lastName, countryIso2) entries.
  3. Send batches of 100 to NamSor /api2/json/usRaceEthnicityBatch.
  4. Append every personalNames item NamSor returns to a JSONL file
     (one JSON object per line) — this is the raw archive.
  5. Optionally write a flat CSV with the canonical category mapped
     to our 4-class scheme.

Endpoint: POST https://v2.namsor.com/NamSorAPIv2/api2/json/usRaceEthnicityBatch
Body: {"personalNames":[{"id","firstName","lastName","countryIso2"}]}
Response field raceEthnicity ∈ {W_NL, HL, A, B_NL, AI_AN, PI}.

Auth: API key from config.ini ([namsor].data_dir → text file with the key)
or from env var NAMSOR_API_KEY (overrides the config).

Examples
--------
# Smoke test on 20 unique names, no API call:
python namsor_verify.py --input ../../../results/summary_v2/recommendations_with_ethnicity.csv --limit 20 --dry-run

# Real run, all unique (firstName, lastName, country) triples, append to JSONL:
python namsor_verify.py --input ../../../results/summary_v2/recommendations_with_ethnicity.csv --output-jsonl ../../../data/ethnicity_inference/namsor_responses.jsonl --output-csv ../../../data/ethnicity_inference/namsor_results.csv
"""

import argparse
import json
import os
import sys
import time
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path

import pandas as pd
import requests

# ─── Constants ───────────────────────────────────────────────────────────────

BATCH_SIZE = 100  # NamSor hard limit per batch call
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 4

# Selectable NamSor endpoints.
#   mode "split" → payload uses firstName + lastName
#   mode "full"  → payload uses a single unsplit `name`
ENDPOINTS = {
    "us_race": {
        "url": "https://v2.namsor.com/NamSorAPIv2/api2/json/usRaceEthnicityBatch",
        "credits": 10,
        "mode": "split",
        "label": "US race/ethnicity (US Census taxonomy)",
    },
    "diaspora": {
        "url": "https://v2.namsor.com/NamSorAPIv2/api2/json/diasporaBatch",
        "credits": 20,
        "mode": "split",
        "label": "Name Diaspora (firstName + lastName)",
    },
    "diaspora_full": {
        "url": "https://v2.namsor.com/NamSorAPIv2/api2/json/diasporaFullBatch",
        "credits": 20,
        "mode": "full",
        "label": "Full Name Diaspora (unsplit name)",
    },
}


def log(msg: str) -> None:
    from datetime import datetime

    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_namsor_key(config_path: Path = Path("../../../config.ini")) -> str | None:
    """Resolve NamSor API key. Env var NAMSOR_API_KEY wins; else read the path
    in config.ini [namsor].data_dir."""
    env = os.environ.get("NAMSOR_API_KEY", "").strip()
    if env:
        return env
    if not config_path.exists():
        return None
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read(config_path, encoding="utf-8")
    key_path = parser.get("namsor", "data_dir", fallback="")
    if not key_path:
        return None
    p = Path(key_path).expanduser()
    if not p.is_absolute():
        p = (config_path.parent / p).resolve()
    if not p.exists():
        log(f"warning: NamSor key file not found at {p}")
        return None
    return p.read_text(encoding="utf-8").strip()


# NamSor raceEthnicity → our 4-class canonical scheme (matches ethnicity_inference.py)
NAMSOR_TO_CANONICAL = {
    "W_NL": "White",
    "HL": "Hispanic or Latino",
    "A": "Asian",
    "B_NL": "Black or African American",
    "AI_AN": "Unknown",  # American Indian / Alaska Native — not in our 4 classes
    "PI": "Unknown",  # Pacific Islander — collapsed
}

# The 5 study locations + a few likely-encountered extras → ISO-2 country codes.
# Used to enrich the NamSor request; countryIso2 is optional but improves accuracy.
COUNTRY_TO_ISO2 = {
    # 5 study locations — English / Spanish / German (the 3 prompt languages)
    "South Africa": "ZA",
    "Sudáfrica": "ZA",
    "Südafrika": "ZA",
    "Germany": "DE",
    "Alemania": "DE",
    "Deutschland": "DE",
    "Canada": "CA",
    "Canadá": "CA",
    "Kanada": "CA",
    "Ecuador": "EC",
    "Japan": "JP",
    "Japón": "JP",
    # extras commonly seen in OpenAlex affiliations
    "United States": "US",
    "United States of America": "US",
    "USA": "US",
    "United Kingdom": "GB",
    "UK": "GB",
    "France": "FR",
    "Spain": "ES",
    "Italy": "IT",
    "Mexico": "MX",
    "Brazil": "BR",
    "Argentina": "AR",
    "Colombia": "CO",
    "Chile": "CL",
    "China": "CN",
    "India": "IN",
    "Australia": "AU",
}


def country_to_iso2(country: str | None) -> str | None:
    if not country or not isinstance(country, str):
        return None
    return COUNTRY_TO_ISO2.get(country.strip())


# ─── NamSor client ───────────────────────────────────────────────────────────


def namsor_batch(
    api_key: str, batch: list[dict], url: str, timeout: int = 60
) -> list[dict]:
    """POST one batch (≤ 100 personalNames) and return the parsed personalNames list.

    Retries 429 / 5xx with exponential backoff. Hard-fails on 401 (bad key)
    and 403 (quota exhausted) — the docs do not recommend retrying those.
    """
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {"personalNames": batch}

    for attempt in range(MAX_RETRIES + 1):
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json().get("personalNames", [])
        if resp.status_code == 401:
            raise SystemExit(
                f"NamSor 401 unauthorized — check your API key. Body: {resp.text[:200]}"
            )
        if resp.status_code == 403:
            raise SystemExit(
                f"NamSor 403 — monthly quota reached (or soft limit). Body: {resp.text[:200]}"
            )
        if resp.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
            wait = 2**attempt
            log(
                f"  HTTP {resp.status_code} — retrying in {wait}s (attempt {attempt+1}/{MAX_RETRIES})"
            )
            time.sleep(wait)
            continue
        # Anything else: raise with body for visibility
        raise requests.HTTPError(
            f"NamSor HTTP {resp.status_code}: {resp.text[:300]}", response=resp
        )
    raise RuntimeError("unreachable")


# ─── Pipeline ────────────────────────────────────────────────────────────────


def build_unique_batch(df: pd.DataFrame, mode: str = "split") -> pd.DataFrame:
    """Return a deduplicated DataFrame keyed on the payload fields the endpoint needs.

    mode "split" → columns id, firstName, lastName, countryIso2
    mode "full"  → columns id, name, countryIso2 (firstName + lastName joined,
                   so NamSor does its own splitting — that is the whole point of
                   the diasporaFull endpoint)
    """
    work = pd.DataFrame(
        {
            "firstName": df["name"].fillna("").astype(str).str.strip(),
            "lastName": df["lastname"].fillna("").astype(str).str.strip(),
            "countryIso2": df["location"].map(country_to_iso2).fillna(""),
        }
    )

    # Drop empty names and dedup
    mask = (work["firstName"] != "") | (work["lastName"] != "")
    work = (
        work[mask]
        .drop_duplicates(subset=["firstName", "lastName", "countryIso2"])
        .reset_index(drop=True)
    )

    # Stable id (deterministic) so JSONL rows can be joined back later
    work["id"] = work.apply(
        lambda r: f"{r['firstName']}|{r['lastName']}|{r['countryIso2']}", axis=1
    )

    if mode == "full":
        work["name"] = (
            (work["firstName"] + " " + work["lastName"]).str.replace(r"\s+", " ", regex=True).str.strip()
        )
        return work[["id", "name", "countryIso2"]]

    return work[["id", "firstName", "lastName", "countryIso2"]]


def load_already_processed(jsonl_path: Path) -> set[str]:
    """Read existing JSONL and return ids already saved (for resume)."""
    if not jsonl_path.exists():
        return set()
    ids: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["id"])
            except Exception:
                continue
    return ids


def run(
    input_path: Path,
    output_jsonl: Path,
    output_csv: Path | None,
    api_key: str | None,
    limit: int | None,
    dry_run: bool,
    sleep: float,
    endpoint: str = "us_race",
    sep: str = ",",
) -> None:
    cfg = ENDPOINTS[endpoint]
    log(f"Endpoint: {endpoint} — {cfg['label']}")
    log(f"  URL: {cfg['url']}  ({cfg['credits']} credits/name)")

    log(f"Reading {input_path} …")
    df = pd.read_csv(
        input_path,
        low_memory=False,
        sep=sep,
        usecols=["name", "lastname", "location"],
    )
    log(f"Loaded {len(df):,} rows.")

    work = build_unique_batch(df, mode=cfg["mode"])
    log(f"Unique (firstName, lastName, countryIso2) triples: {len(work):,}")

    # Country coverage report
    n_with_country = (work["countryIso2"] != "").sum()
    log(
        f"  with countryIso2  : {n_with_country:,}  ({100*n_with_country/max(len(work),1):.1f}%)"
    )
    log(f"  without countryIso2: {len(work)-n_with_country:,}")

    if limit is not None:
        work = work.head(limit).reset_index(drop=True)
        log(f"--limit applied → processing {len(work)} entries.")

    # Resume support: skip ids already in JSONL
    already = load_already_processed(output_jsonl)
    if already:
        before = len(work)
        work = work[~work["id"].isin(already)].reset_index(drop=True)
        log(
            f"Resume: skipping {before - len(work):,} already-saved ids "
            f"({len(work):,} remaining)."
        )

    if work.empty:
        log("Nothing to send. Exiting.")
        return

    # Build personalNames payload (drop empty countryIso2 to keep request clean)
    records = []
    for _, r in work.iterrows():
        if cfg["mode"] == "full":
            rec = {"id": r["id"], "name": r["name"]}
        else:
            rec = {
                "id": r["id"],
                "firstName": r["firstName"],
                "lastName": r["lastName"],
            }
        if r["countryIso2"]:
            rec["countryIso2"] = r["countryIso2"]
        records.append(rec)

    n_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
    est_credits = len(records) * cfg["credits"]
    log(
        f"Plan: {len(records):,} names → {n_batches} batch(es) of ≤{BATCH_SIZE} "
        f"→ ~{est_credits:,} NamSor credits ({cfg['credits']}/name)."
    )

    if dry_run:
        log("DRY RUN — first 5 payload entries:")
        for rec in records[:5]:
            print("  " + json.dumps(rec, ensure_ascii=False))
        return

    if not api_key:
        sys.exit(
            "ERROR: no NamSor API key. Set NAMSOR_API_KEY env var or "
            "place the key at the path in config.ini [namsor].data_dir."
        )

    # Stream raw responses to JSONL, one personalNames item per line.
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []
    sent = 0

    with output_jsonl.open("a", encoding="utf-8") as out:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            log(
                f"POST batch {i//BATCH_SIZE+1}/"
                f"{(len(records)+BATCH_SIZE-1)//BATCH_SIZE} ({len(batch)} names) …"
            )
            try:
                items = namsor_batch(api_key, batch, cfg["url"])
            except requests.HTTPError as exc:
                log(f"  HTTP error: {exc} — body: {exc.response.text[:300]}")
                raise
            except Exception as exc:
                log(f"  network error: {exc}")
                raise

            for item in items:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                all_results.append(item)
            out.flush()
            sent += len(items)
            log(f"  saved {sent:,} responses so far → {output_jsonl}")
            if sleep > 0 and i + BATCH_SIZE < len(records):
                time.sleep(sleep)

    log(f"Done. {sent:,} responses written to {output_jsonl}")

    # ── Optional flat CSV summary ────────────────────────────────────────────
    if output_csv is not None and all_results:
        # Keep every scalar field NamSor returned; the taxonomy column differs per
        # endpoint (raceEthnicity for us_race, ethnicity for the diaspora ones).
        flat = pd.DataFrame(all_results)
        if "ethnicitiesTop" in flat.columns:
            # list → pipe-joined string so the CSV stays flat (raw list is in the JSONL)
            flat["ethnicitiesTop"] = flat["ethnicitiesTop"].apply(
                lambda v: "|".join(map(str, v)) if isinstance(v, list) else v
            )
        flat.insert(0, "endpoint", endpoint)
        if "raceEthnicity" in flat.columns:
            flat["namsor_canonical"] = (
                flat["raceEthnicity"].map(NAMSOR_TO_CANONICAL).fillna("Unknown")
            )

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        # If file exists, append without header
        if output_csv.exists():
            flat.to_csv(output_csv, mode="a", header=False, index=False)
        else:
            flat.to_csv(output_csv, index=False)
        log(f"Flat CSV written → {output_csv}")

        # Quick distribution over whichever taxonomy column this endpoint returned
        dist_col = next(
            (c for c in ("namsor_canonical", "ethnicity") if c in flat.columns), None
        )
        if dist_col:
            log(f"Distribution of `{dist_col}` (this run):")
            for cat, count in flat[dist_col].value_counts().items():
                log(f"  {str(cat):<30} {count:>6,}")


# ─── CLI ─────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify perceived ethnicity using the NamSor API."
    )
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input CSV with columns name, lastname, location.",
    )
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("../../../data/ethnicity_inference/namsor_responses.jsonl"),
        help="JSONL file to append raw NamSor responses (one personalNames item per line).",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("../../../data/ethnicity_inference/namsor_results.csv"),
        help="Flat CSV with parsed result + canonical mapping. Pass empty string to skip.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N unique entries (for testing).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the API; print the first few payload entries.",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between batches (rate-limit friendly).",
    )
    p.add_argument(
        "--endpoint",
        choices=sorted(ENDPOINTS),
        default="us_race",
        help=(
            "Which NamSor endpoint to call. "
            "us_race = usRaceEthnicityBatch (10 credits/name, US Census classes); "
            "diaspora = diasporaBatch (20, firstName+lastName); "
            "diaspora_full = diasporaFullBatch (20, unsplit name)."
        ),
    )
    p.add_argument(
        "--sep",
        default=",",
        help="Input CSV field separator (use ';' for the semicolon-delimited files).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = build_parser()

    api_key = load_namsor_key()
    output_csv = args.output_csv if str(args.output_csv) else None

    run(
        input_path=args.input,
        output_jsonl=args.output_jsonl,
        output_csv=output_csv,
        api_key=api_key or None,
        limit=args.limit,
        dry_run=args.dry_run,
        sleep=args.sleep,
        endpoint=args.endpoint,
        sep=args.sep,
    )
