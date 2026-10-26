"""
1_ss_papers.py — Step 1: the papers (with DOI) of Semantic Scholar researchers,
from the Semantic Scholar API. The DOIs are the bridge to the same person in
OpenAlex (2_jw_match.py). run_pipeline.py runs it for every researcher of the
parquet with --batch_only (step a); steps b–d (merged profiles, one call each)
are run for a smaller list, e.g. the researchers with factuality_author = True.

  a. POST /author/batch, 100 researchers per call (the most the API allows with
     papers; it returns ~5000 papers per call, enough DOIs per researcher).
  b. The researchers with no DOI: which of them the batch does not return at
     all (asked again with only paperCount, which is cheap). Those were merged
     by Semantic Scholar into another profile.
  c. GET /author/{id} follows the merge and gives their current id.
  d. The papers of those current ids, again in batches of 100, kept under the
     original researcher_id.

A few calls in parallel, retried with a growing wait on 429 (the public rate
limit is shared; with a key, calls are spaced to --rps). Every call is saved as it arrives in <output>_parts/, so a run
that stops resumes where it was. If [semantic_scholar].s2_api_key in config.ini (the path of a file with the key) is set
(or the S2_API_KEY environment variable), it is sent.

Output: ss_papers.parquet — researcher_id, doi (lowercase), year; and
ss_merged.parquet — researcher_id, current_id (for the merged profiles).

Usage (from code/, with PYTHONPATH=.):
  python scripts/1_ground_truth/1_ss_papers.py
  # every researcher of the Semantic Scholar parquet (step a only):
  python scripts/1_ground_truth/1_ss_papers.py --ids <ss_parquet> --id_column Researcher_id \\
      --batch_only --output ../results/summary/ground_truth/ss_papers.parquet
  # how many of those without DOIs were merged (step b only, 1000 per call):
  python scripts/1_ground_truth/1_ss_papers.py --ids <ss_parquet> --id_column Researcher_id \\
      --exists_only --output ../results/summary/ground_truth/ss_papers.parquet
"""

import argparse
import configparser
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from libs.utils.logging import setup_logging

logger = setup_logging()

API = "https://api.semanticscholar.org/graph/v1"
PAPERS = "papers.externalIds,papers.year"
BATCH = 100
EXISTS_BATCH = 1000   # the most the batch endpoint takes without papers
WORKERS = 4


def api_key() -> str | None:
    """S2_API_KEY, else the file that [semantic_scholar].s2_api_key in config.ini points to."""
    if os.environ.get("S2_API_KEY"):
        return os.environ["S2_API_KEY"]
    cfg = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d / "config.ini").exists():
            cfg.read(d / "config.ini")
            key_file = cfg.get("semantic_scholar", "s2_api_key", fallback="")
            if not key_file:
                return None
            p = Path(key_file).expanduser()
            return (p if p.is_absolute() else d / p).read_text(encoding="utf-8").strip() or None
    return None


KEY = api_key()

# With a key the limit is per key (1 request per second unless granted more):
# calls from all workers are spaced to stay under it.
RPS = 1.0
REVERSE = False
FIRST, LAST = 0, 10**9   # range of batch numbers to do (to split the work between machines)
_lock = threading.Lock()
_next = [0.0]


def wait_turn() -> None:
    if not KEY:
        return
    with _lock:
        now = time.monotonic()
        start = max(now, _next[0])
        _next[0] = start + 1 / RPS
    time.sleep(start - now)


def call(path: str, body: dict | None = None):
    """One API call, retried on 429 and server errors. None if it never works."""
    headers = {"Content-Type": "application/json", **({"x-api-key": KEY} if KEY else {})}
    for attempt in range(40):
        wait_turn()
        try:
            req = urllib.request.Request(f"{API}{path}", headers=headers, method="POST" if body else "GET",
                                         data=json.dumps(body).encode() if body else None)
            return json.load(urllib.request.urlopen(req, timeout=120))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code not in (429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(min(60, 2 ** attempt))
    return None


def papers_rows(rid: int, author: dict | None) -> list[tuple]:
    return [(rid, p["externalIds"]["DOI"].lower(), p.get("year"))
            for p in (author or {}).get("papers") or []
            if (p.get("externalIds") or {}).get("DOI")]


def in_parallel(fn, items, parts: Path, label: str) -> None:
    """fn(i, item) → DataFrame, saved as parts/<label>_<i>.parquet; skipped if there."""
    def one(arg):
        i, item = arg
        out = parts / f"{label}_{i:05d}.parquet"
        if out.exists():
            return True
        df = fn(item)
        if df is None:
            return False
        df.to_parquet(out, index=False)
        return True

    todo = [(i, x) for i, x in enumerate(items) if FIRST <= i <= LAST]
    if REVERSE:   # from the end: a second process can share the work with one going forward
        todo.reverse()
    for round_ in range(5):   # what failed (too many 429) is tried again
        with ThreadPoolExecutor(WORKERS) as pool:
            ok = list(pool.map(one, todo))
        logger.info("%s, round %d: %d/%d", label, round_ + 1, sum(ok), len(todo))
        if all(ok):
            return


def batch_papers(ids: list[int]) -> pd.DataFrame | None:
    """Papers of up to 100 researchers."""
    answer = call(f"/author/batch?fields={PAPERS}", {"ids": [str(i) for i in ids]})
    if answer is None:
        return None
    rows = [r for rid, a in zip(ids, answer) for r in papers_rows(rid, a)]
    return pd.DataFrame(rows, columns=["researcher_id", "doi", "year"])


def batch_exists(ids: list[int]) -> pd.DataFrame | None:
    """Whether the batch endpoint returns each researcher at all."""
    answer = call("/author/batch?fields=paperCount", {"ids": [str(i) for i in ids]})
    if answer is None:
        return None
    return pd.DataFrame({"researcher_id": ids, "exists": [a is not None for a in answer]})


def current_id(rid: int) -> pd.DataFrame | None:
    answer = call(f"/author/{rid}?fields=authorId")
    if answer is None:
        return pd.DataFrame({"researcher_id": [rid], "current_id": [None]})
    return pd.DataFrame({"researcher_id": [rid], "current_id": [answer.get("authorId")]})


def merged_papers(pairs: list[tuple[int, str]]) -> pd.DataFrame | None:
    """Papers of the current ids, kept under the original researcher_id."""
    answer = call(f"/author/batch?fields={PAPERS}", {"ids": [c for _, c in pairs]})
    if answer is None:
        return None
    rows = [r for (rid, _), a in zip(pairs, answer) for r in papers_rows(rid, a)]
    return pd.DataFrame(rows, columns=["researcher_id", "doi", "year"])


def read_parts(parts: Path, label: str) -> pd.DataFrame:
    files = sorted(parts.glob(f"{label}_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files]) if files else pd.DataFrame()


def read_ids(path: str, column: str) -> set[int]:
    if str(path).endswith(".parquet"):
        s = pd.read_parquet(path, columns=[column])[column]
    else:
        s = pd.read_csv(path, usecols=[column])[column]
    return set(s.dropna().astype("int64"))


def run(ids_path: str, id_column: str, output: str, exclude: str | None = None,
        batch_only: bool = False, exists_only: bool = False) -> None:
    ids = read_ids(ids_path, id_column)
    if exclude:
        ids -= read_ids(exclude, "researcher_id")
    ids = sorted(ids)
    parts = Path(str(output).replace(".parquet", "_parts"))
    parts.mkdir(parents=True, exist_ok=True)
    logger.info("%d researchers (%s API key)", len(ids), "with" if KEY else "no")

    # a. Batches of 100.
    if not exists_only:
        in_parallel(batch_papers, [ids[i:i + BATCH] for i in range(0, len(ids), BATCH)], parts, "batch")
    a = read_parts(parts, "batch")
    if exists_only:
        # Only step b, 1000 per call: which researchers without DOIs were merged.
        no_doi = sorted(set(ids) - set(a["researcher_id"].astype("int64")))
        in_parallel(batch_exists, [no_doi[i:i + EXISTS_BATCH] for i in range(0, len(no_doi), EXISTS_BATCH)],
                    parts, "exists1000")
        e = read_parts(parts, "exists1000")
        e.to_parquet(Path(output).with_name("ss_exists.parquet"), index=False)
        logger.info("Without DOIs: %d; asked: %d; merged into another profile: %d",
                    len(no_doi), len(e), int((~e["exists"]).sum()))
        return
    if batch_only:
        a["researcher_id"] = a["researcher_id"].astype("int64")
        a = a.drop_duplicates(["researcher_id", "doi"])
        a.to_parquet(output, index=False)
        logger.info("Wrote %s: %d DOIs for %d of %d researchers", output, len(a),
                    a["researcher_id"].nunique(), len(ids))
        return

    # b. Without DOIs: does the batch return them at all?
    no_doi = sorted(set(ids) - set(a["researcher_id"].astype("int64")))
    in_parallel(batch_exists, [no_doi[i:i + BATCH] for i in range(0, len(no_doi), BATCH)], parts, "exists")
    e = read_parts(parts, "exists")
    nulls = sorted(e.loc[~e["exists"], "researcher_id"].astype("int64"))
    logger.info("Without DOIs: %d; merged into another profile: %d", len(no_doi), len(nulls))

    # c. Their current id, one call each.
    in_parallel(current_id, nulls, parts, "merged")
    merged = read_parts(parts, "merged").dropna(subset=["current_id"])

    # d. The papers of the current ids.
    pairs = list(zip(merged["researcher_id"].astype("int64"), merged["current_id"].astype(str)))
    in_parallel(merged_papers, [pairs[i:i + BATCH] for i in range(0, len(pairs), BATCH)], parts, "merged_papers")

    df = pd.concat([a, read_parts(parts, "merged_papers")])
    df["researcher_id"] = df["researcher_id"].astype("int64")
    df = df.drop_duplicates(["researcher_id", "doi"])
    df.to_parquet(output, index=False)
    merged.to_parquet(Path(output).with_name("ss_merged.parquet"), index=False)
    logger.info("Wrote %s: %d DOIs for %d of %d researchers (%d through a merged profile)",
                output, len(df), df["researcher_id"].nunique(), len(ids),
                df.loc[df["researcher_id"].isin(merged["researcher_id"]), "researcher_id"].nunique())


def main() -> None:
    global RPS, WORKERS, REVERSE, KEY, FIRST, LAST
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", "--factual", dest="ids", default="../results/summary/cascade/factual_authors.csv",
                        help="CSV or parquet with the researcher ids (default: the factual authors)")
    parser.add_argument("--id_column", default="researcher_id")
    parser.add_argument("--exclude", default=None, help="CSV or parquet with researcher_id to skip")
    parser.add_argument("--batch_only", action="store_true",
                        help="Only step a (no merged-profile recovery, which is one call per researcher)")
    parser.add_argument("--exists_only", action="store_true",
                        help="Only step b for the researchers without DOIs (1000 per call) → ss_exists.parquet")
    parser.add_argument("--output", default="../results/summary/openalex_cascade/ss_papers.parquet")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help="calls in parallel (with a key they are still spaced to --rps)")
    parser.add_argument("--no_key", action="store_true",
                        help="do not send the API key (public limit, separate from the key's)")
    parser.add_argument("--first_batch", type=int, default=0, help="first batch number to do")
    parser.add_argument("--last_batch", type=int, default=10**9, help="last batch number to do")
    parser.add_argument("--reverse", action="store_true",
                        help="go through the batches from the last one (to run next to another process)")
    parser.add_argument("--rps", type=float, default=RPS,
                        help="requests per second allowed for the API key")
    args = parser.parse_args()
    RPS, WORKERS, REVERSE = args.rps, args.workers, args.reverse
    FIRST, LAST = args.first_batch, args.last_batch
    if args.no_key:
        KEY = None
    run(args.ids, args.id_column, args.output, args.exclude, args.batch_only, args.exists_only)


if __name__ == "__main__":
    main()
