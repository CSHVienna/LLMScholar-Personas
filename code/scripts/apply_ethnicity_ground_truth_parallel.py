"""
Parallel ethnicity inference for Semantic Scholar ground truth data.

Splits unique researchers into N chunks and runs one worker process per chunk,
each loading its own BERT model instance to fully utilize CPU cores.

Usage:
    python apply_ethnicity_ground_truth_parallel.py [--workers N] [--batch-size B] [--skip-merge]
"""

import os
import sys
import logging
import subprocess
import tempfile
import shutil
import argparse

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR      = "/data/datasets/LLMScholar-Personas/data/semantic_scholar_data"
LOOKUP_OUT    = os.path.join(DATA_DIR, "researcher_ethnicity_lookup.csv")
WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "ethnicity_worker.py")

# Writable fallback paths (used when DATA_DIR has no write permissions)
_PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_ETHNICITY_DIR = os.path.join(_PROJECT_ROOT, 'results', 'ethnicity')
LOOKUP_FALLBACK = os.path.join(_ETHNICITY_DIR, "researcher_ethnicity_lookup.csv")

FIELDS = [
    "Biology",
    "Computer_Science",
    "Mathematics",
    "Physics",
    "Psychology",
    "Sociology",
]

CHUNK_SIZE = 500_000   # rows per chunk when writing field files


# ─── Step 1: Extract unique researchers ──────────────────────────────────────

def extract_unique_researchers() -> pd.DataFrame:
    logger.info("=== Step 1: Extracting unique researchers from all fields ===")
    frames = []
    for field in FIELDS:
        path = os.path.join(DATA_DIR, f"DataFrameRankings_Genderize_Namsor_{field}.csv")
        logger.info("  Reading %s …", field)
        df = pd.read_csv(path, usecols=["Researcher_id", "Name"],
                         dtype={"Researcher_id": "float64", "Name": str},
                         low_memory=False)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    del frames
    combined.drop_duplicates(subset=["Researcher_id"], keep="first", inplace=True)
    combined.dropna(subset=["Name"], inplace=True)
    combined["Name"] = combined["Name"].astype(str).str.strip()
    combined = combined[combined["Name"] != ""].reset_index(drop=True)
    logger.info("Unique researchers with valid name: %d", len(combined))
    return combined


# ─── Step 2: Split + run parallel workers ────────────────────────────────────

def run_parallel_inference(unique_df: pd.DataFrame, n_workers: int, batch_size: int) -> pd.DataFrame:
    logger.info("=== Step 2: Running inference with %d parallel workers ===", n_workers)

    tmp_dir = tempfile.mkdtemp(prefix="ethnicity_chunks_")
    logger.info("Temp directory: %s", tmp_dir)

    try:
        # Split into chunks
        chunk_size = (len(unique_df) + n_workers - 1) // n_workers
        chunk_paths  = []
        output_paths = []

        for i in range(n_workers):
            chunk = unique_df.iloc[i * chunk_size : (i + 1) * chunk_size]
            if len(chunk) == 0:
                continue
            chunk_path  = os.path.join(tmp_dir, f"chunk_{i:02d}.csv")
            output_path = os.path.join(tmp_dir, f"result_{i:02d}.csv")
            chunk.to_csv(chunk_path, index=False)
            chunk_paths.append(chunk_path)
            output_paths.append(output_path)

        actual_workers = len(chunk_paths)
        logger.info("Launching %d worker processes …", actual_workers)

        # Launch all workers simultaneously
        env = os.environ.copy()
        env["TF_USE_LEGACY_KERAS"] = "1"
        # Limit each worker's thread count so they don't fight for cores
        threads_per_worker = max(1, 36 // actual_workers)
        env["OMP_NUM_THREADS"]   = str(threads_per_worker)
        env["MKL_NUM_THREADS"]   = str(threads_per_worker)
        env["TORCH_NUM_THREADS"] = str(threads_per_worker)

        procs = []
        for i, (chunk_path, output_path) in enumerate(zip(chunk_paths, output_paths)):
            log_path = os.path.join(tmp_dir, f"worker_{i:02d}.log")
            log_fh   = open(log_path, "w")
            proc = subprocess.Popen(
                [sys.executable, WORKER_SCRIPT,
                 chunk_path, output_path, str(i), str(batch_size)],
                env=env,
                stdout=log_fh,
                stderr=log_fh,
            )
            procs.append((proc, log_fh, log_path))
            logger.info("  Worker %d started (PID %d, chunk size %d)",
                        i, proc.pid,
                        len(pd.read_csv(chunk_path, usecols=["Researcher_id"])))

        # Wait for all workers
        logger.info("Waiting for all workers to finish …")
        for i, (proc, log_fh, log_path) in enumerate(procs):
            rc = proc.wait()
            log_fh.close()
            if rc != 0:
                logger.error("Worker %d failed (exit code %d). Log: %s", i, rc, log_path)
                with open(log_path) as f:
                    logger.error(f.read()[-2000:])
            else:
                logger.info("  Worker %d finished OK.", i)

        # Merge results
        logger.info("Merging worker results …")
        result_frames = []
        for output_path in output_paths:
            if os.path.exists(output_path):
                result_frames.append(pd.read_csv(output_path))
            else:
                logger.error("Missing result file: %s", output_path)

        merged = pd.concat(result_frames, ignore_index=True)
        logger.info("Total rows after merge: %d", len(merged))
        return merged

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─── Step 3: Save lookup ──────────────────────────────────────────────────────

def save_lookup(df: pd.DataFrame) -> None:
    logger.info("=== Step 3: Saving lookup to %s ===", LOOKUP_OUT)

    n_total   = len(df)
    n_demo    = (df["__ethnicity_source"] == "demographicx").sum()
    n_ethn    = (df["__ethnicity_source"] == "ethnicolr").sum()
    n_unknown = (df["__ethnicity_source"] == "unknown").sum()

    logger.info("  demographicx : %d  (%.1f%%)", n_demo,    100 * n_demo    / n_total)
    logger.info("  ethnicolr    : %d  (%.1f%%)", n_ethn,    100 * n_ethn    / n_total)
    logger.info("  Unknown      : %d  (%.1f%%)", n_unknown, 100 * n_unknown / n_total)

    logger.info("Ethnicity distribution:")
    dist = df["perceived_ethnicity"].value_counts()
    for cat, count in dist.items():
        logger.info("  %-28s %8d  (%.1f%%)", cat, count, 100 * count / n_total)

    saved = False
    for dest in [LOOKUP_OUT, LOOKUP_FALLBACK]:
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            df[["Researcher_id", "Name", "perceived_ethnicity", "__ethnicity_source"]].to_csv(dest, index=False)
            logger.info("Lookup saved to %s (%d rows).", dest, n_total)
            saved = True
            break
        except PermissionError as exc:
            logger.warning("Cannot write to %s: %s — trying fallback.", dest, exc)
    if not saved:
        logger.error("Could not save lookup to any path.")


# ─── Step 4: Merge lookup into each field file ────────────────────────────────

def merge_into_field_files(lookup: pd.DataFrame) -> None:
    logger.info("=== Step 4: Merging ethnicity into field files ===")
    lookup_map = lookup.set_index("Researcher_id")["perceived_ethnicity"].to_dict()

    for field in FIELDS:
        in_path           = os.path.join(DATA_DIR, f"DataFrameRankings_Genderize_Namsor_{field}.csv")
        out_path          = os.path.join(DATA_DIR, f"DataFrameRankings_Genderize_Namsor_{field}_with_ethnicity.csv")
        out_path_fallback = os.path.join(_ETHNICITY_DIR, f"DataFrameRankings_Genderize_Namsor_{field}_with_ethnicity.csv")

        if os.path.exists(out_path) or os.path.exists(out_path_fallback):
            logger.info("  %s already exists, skipping.", field)
            continue

        dest = out_path if os.access(DATA_DIR, os.W_OK) else out_path_fallback
        if dest == out_path_fallback:
            os.makedirs(_ETHNICITY_DIR, exist_ok=True)
            logger.warning("  No write permission for %s — using fallback path.", DATA_DIR)

        logger.info("  Processing %s → %s …", field, dest)
        first_chunk = True
        for chunk in tqdm(
            pd.read_csv(in_path, chunksize=CHUNK_SIZE, low_memory=False),
            desc=f"  {field}",
        ):
            chunk["perceived_ethnicity"] = (
                chunk["Researcher_id"].map(lookup_map).fillna("Unknown")
            )
            chunk.to_csv(dest, index=False,
                         mode="w" if first_chunk else "a",
                         header=first_chunk)
            first_chunk = False

        logger.info("  Saved %s", dest)


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(lookup_path: str) -> None:
    lookup = pd.read_csv(lookup_path)
    total  = len(lookup)
    print("\n=== Ground Truth Ethnicity Inference Summary ===")
    print(f"Total unique researchers : {total:,}")
    n_demo    = (lookup["__ethnicity_source"] == "demographicx").sum()
    n_ethn    = (lookup["__ethnicity_source"] == "ethnicolr").sum()
    n_unknown = (lookup["__ethnicity_source"] == "unknown").sum()
    print(f"  classified by demographicx : {n_demo:,}  ({100*n_demo/total:.1f}%)")
    print(f"  classified by ethnicolr    : {n_ethn:,}  ({100*n_ethn/total:.1f}%)")
    print(f"  Unknown                    : {n_unknown:,}  ({100*n_unknown/total:.1f}%)")
    print("\nEthnicity distribution:")
    dist = lookup["perceived_ethnicity"].value_counts()
    for cat, count in dist.items():
        print(f"  {cat:<30} {count:>10,}  ({100*count/total:5.1f}%)")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",    type=int, default=12,
                        help="Number of parallel worker processes (default: 12)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size per worker (default: 256)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference, load existing lookup")
    parser.add_argument("--skip-merge",     action="store_true",
                        help="Skip merging into field files")
    args = parser.parse_args()

    os.makedirs(_ETHNICITY_DIR, exist_ok=True)

    if not args.skip_inference:
        unique_df = extract_unique_researchers()
        result_df = run_parallel_inference(unique_df, args.workers, args.batch_size)
        save_lookup(result_df)
    else:
        lookup_path = LOOKUP_OUT if os.path.exists(LOOKUP_OUT) else LOOKUP_FALLBACK
        logger.info("Skipping inference, loading existing lookup from %s.", lookup_path)
        result_df = pd.read_csv(lookup_path)

    if not args.skip_merge:
        merge_into_field_files(result_df)

    summary_path = LOOKUP_OUT if os.path.exists(LOOKUP_OUT) else LOOKUP_FALLBACK
    print_summary(summary_path)
