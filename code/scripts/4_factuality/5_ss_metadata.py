"""
5_ss_metadata.py — Step 6 of run_pipeline.py: the Semantic Scholar metadata of
the researcher picked for each recommendation with factuality_author = True,
from the Semantic Scholar API (POST /author/batch, 1000 researchers per call):

  ss_paper_count     papers of the researcher in Semantic Scholar
  ss_citation_count  citations
  ss_h_index         h-index
  ss_status          found | not_returned (the batch does not return researchers
                     that Semantic Scholar merged into another profile) | ""
                     (factuality_author = False, no researcher picked)

Added to factuality_ss.csv (and factual_authors.csv) as new columns. The calls
are saved in ss_metadata_parts/ as they arrive, so a run that stops resumes; if
[secrets].s2_api_key is set in config.ini (or S2_API_KEY), it is sent.

Usage (from code/, with PYTHONPATH=.):
  python scripts/4_factuality/5_ss_metadata.py --cascade_dir ../results/summary/cascade
"""

import argparse
import configparser
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from libs.utils.logging import setup_logging

logger = setup_logging()

URL = "https://api.semanticscholar.org/graph/v1/author/batch?fields=paperCount,citationCount,hIndex"
BATCH = 1000
WORKERS = 4
COLUMNS = {"paperCount": "ss_paper_count", "citationCount": "ss_citation_count", "hIndex": "ss_h_index"}


def api_key() -> str | None:
    if os.environ.get("S2_API_KEY"):
        return os.environ["S2_API_KEY"]
    cfg = configparser.ConfigParser()
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d / "config.ini").exists():
            cfg.read(d / "config.ini")
            return cfg.get("secrets", "s2_api_key", fallback=None) or None
    return None


def fetch(ids: list[int], key: str | None) -> pd.DataFrame | None:
    headers = {"Content-Type": "application/json", **({"x-api-key": key} if key else {})}
    body = json.dumps({"ids": [str(i) for i in ids]}).encode()
    for attempt in range(40):
        try:
            req = urllib.request.Request(URL, data=body, headers=headers, method="POST")
            answer = json.load(urllib.request.urlopen(req, timeout=120))
            rows = [{"researcher_id": rid, "ss_status": "found" if a else "not_returned",
                     **{new: (a or {}).get(old) for old, new in COLUMNS.items()}}
                    for rid, a in zip(ids, answer)]
            return pd.DataFrame(rows)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(min(60, 2 ** attempt))
    return None


def download(ids: list[int], parts: Path) -> pd.DataFrame:
    parts.mkdir(parents=True, exist_ok=True)
    key = api_key()
    batches = [ids[i:i + BATCH] for i in range(0, len(ids), BATCH)]

    def one(arg):
        i, b = arg
        out = parts / f"part_{i:05d}.parquet"
        if out.exists():
            return True
        df = fetch(b, key)
        if df is None:
            return False
        df.to_parquet(out, index=False)
        return True

    for round_ in range(5):   # what failed (too many 429) is tried again
        with ThreadPoolExecutor(WORKERS) as pool:
            ok = list(pool.map(one, enumerate(batches)))
        logger.info("round %d: %d/%d batches", round_ + 1, sum(ok), len(batches))
        if all(ok):
            break
    return pd.concat(pd.read_parquet(f) for f in sorted(parts.glob("part_*.parquet")))


def add_columns(path: Path, meta: pd.DataFrame) -> None:
    df = pd.read_csv(path, low_memory=False)
    df = df.drop(columns=[c for c in meta.columns if c != "researcher_id" and c in df.columns])
    rid = pd.to_numeric(df["researcher_id"], errors="coerce").astype("Int64")
    m = meta.drop_duplicates("researcher_id").set_index("researcher_id")
    for col in m.columns:
        df[col] = rid.map(m[col]).to_numpy()
    df["ss_status"] = df["ss_status"].fillna("")
    df.to_csv(path, index=False)
    logger.info("Added %s to %s", ", ".join(m.columns), path)


def run(cascade_dir: str) -> None:
    d = Path(cascade_dir)
    factual = pd.read_csv(d / "factual_authors.csv", usecols=["researcher_id"])
    ids = factual["researcher_id"].dropna().astype("int64").drop_duplicates().sort_values().tolist()
    logger.info("%d researchers with factuality_author = True", len(ids))
    meta = download(ids, d / "ss_metadata_parts")
    meta["researcher_id"] = meta["researcher_id"].astype("int64")
    logger.info("found in the API: %d of %d (%.1f%%)", (meta["ss_status"] == "found").sum(),
                len(ids), 100 * (meta["ss_status"] == "found").mean())
    for name in ["factuality_ss.csv", "factual_authors.csv"]:
        add_columns(d / name, meta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cascade_dir", default="../results/summary/cascade")
    args = parser.parse_args()
    run(args.cascade_dir)


if __name__ == "__main__":
    main()
