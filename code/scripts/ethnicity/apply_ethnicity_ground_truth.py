"""
Apply ethnicity inference to Semantic Scholar ground truth data.

Strategy:
  1. Extract unique (Researcher_id, Name) pairs across all 6 field files (one pass)
  2. Deduplicate by Researcher_id — infer each researcher only once
  3. Save lookup CSV:  researcher_ethnicity_lookup.csv
  4. For each field file, merge lookup and save *_with_ethnicity.csv (chunked to handle large files)
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import pandas as pd
from ethnicity_inference import VALID_CATEGORIES, infer_ethnicity_batch
from tqdm import tqdm

from libs.utils.config import get_data_path
from libs.utils.ios import load_pickle, save_pickle
from libs.utils.logging import setup_logging

logger = setup_logging()

try:
    DATA_DIR = get_data_path("ss_data_dir")
    LOOKUP_OUT = os.path.join(DATA_DIR, "researcher_ethnicity_lookup.csv")
except (FileNotFoundError, KeyError, ValueError):
    DATA_DIR = None
    LOOKUP_OUT = None

# Writable fallback paths (used when DATA_DIR has no write permissions)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ETHNICITY_DIR = os.path.join(_PROJECT_ROOT, "results", "ethnicity")
CHECKPOINT_PATH = os.path.join(_ETHNICITY_DIR, "ethnicity_inference_checkpoint.pkl")
LOOKUP_FALLBACK = os.path.join(_ETHNICITY_DIR, "researcher_ethnicity_lookup.csv")

# Save checkpoint every N batches
CHECKPOINT_EVERY = 10

FIELDS = [
    "Biology",
    "Computer_Science",
    "Mathematics",
    "Physics",
    "Psychology",
    "Sociology",
]

BATCH_SIZE = 512
CHUNK_SIZE = 500_000  # rows per chunk when merging back into large files


def build_lookup() -> pd.DataFrame:
    """
    Read only Researcher_id + Name from all field files, deduplicate by
    Researcher_id, run ethnicity inference, return lookup DataFrame.
    """
    logger.info("=== Step 1: Extracting unique researchers ===")
    frames = []
    for field in FIELDS:
        path = os.path.join(DATA_DIR, f"DataFrameRankings_Genderize_Namsor_{field}.csv")
        logger.info("  Reading %s …", field)
        df = pd.read_csv(
            path,
            usecols=["Researcher_id", "Name"],
            dtype={"Researcher_id": "float64", "Name": str},
            low_memory=False,
        )
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    del frames
    logger.info("Total rows collected: %d", len(combined))

    # Deduplicate: keep first occurrence of each Researcher_id
    combined.drop_duplicates(subset=["Researcher_id"], keep="first", inplace=True)
    combined.dropna(subset=["Name"], inplace=True)
    combined["Name"] = combined["Name"].astype(str).str.strip()
    combined = combined[combined["Name"] != ""]
    logger.info("Unique researchers with valid name: %d", len(combined))

    # ── Step 2: Run inference (with checkpoint resume) ───────────────────────
    logger.info(
        "=== Step 2: Running ethnicity inference (batch_size=%d) ===", BATCH_SIZE
    )
    names = combined["Name"].tolist()
    categories: list[str] = []
    sources: list[str] = []
    confidences: list[float] = []

    # Fingerprint: cheap identity check without storing the full names list
    n = len(names)
    fingerprint = (n, tuple(names[:5]), tuple(names[-5:]))

    # Resume from checkpoint if available
    start_idx = 0
    ckpt = load_pickle(CHECKPOINT_PATH, logger=logger)
    if ckpt is not None:
        if ckpt.get("fingerprint") == fingerprint:
            categories = ckpt["categories"]
            sources = ckpt["sources"]
            confidences = ckpt["confidences"]
            start_idx = len(categories)
            logger.info(
                "Resuming from checkpoint: %d / %d names already processed.",
                start_idx, n,
            )
        else:
            logger.warning("Checkpoint found but names don't match — starting from scratch.")

    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    start_batch = start_idx // BATCH_SIZE
    for batch_idx, i in enumerate(
        tqdm(
            range(start_idx, n, BATCH_SIZE),
            total=n_batches - start_batch,
            desc="Inferring",
        )
    ):
        batch = names[i : i + BATCH_SIZE]
        results = infer_ethnicity_batch(batch, batch_size=BATCH_SIZE)
        for r in results:
            categories.append(r["category"] or "Unknown")
            sources.append(r["source"])
            confidences.append(r["confidence"])

        # Periodic checkpoint — only fingerprint + results, NOT the names list
        if (batch_idx + 1) % CHECKPOINT_EVERY == 0:
            save_pickle(
                {
                    "fingerprint": fingerprint,
                    "categories": categories,
                    "sources": sources,
                    "confidences": confidences,
                },
                CHECKPOINT_PATH,
                logger=logger,
            )

    # Final checkpoint
    save_pickle(
        {
            "fingerprint": fingerprint,
            "categories": categories,
            "sources": sources,
            "confidences": confidences,
        },
        CHECKPOINT_PATH,
        logger=logger,
    )

    combined["perceived_ethnicity"] = categories
    combined["__ethnicity_source"] = sources
    combined["__ethnicity_confidence"] = confidences

    # ── Logging ──────────────────────────────────────────────────────────────
    n_total = len(combined)
    n_demo = (combined["__ethnicity_source"] == "demographicx").sum()
    n_ethn = (combined["__ethnicity_source"] == "ethnicolr").sum()
    n_unknown = (combined["__ethnicity_source"] == "unknown").sum()

    logger.info("=== Inference results ===")
    logger.info("  Total unique researchers : %d", n_total)
    logger.info(
        "  demographicx             : %d  (%.1f%%)", n_demo, 100 * n_demo / n_total
    )
    logger.info(
        "  ethnicolr (fallback)     : %d  (%.1f%%)", n_ethn, 100 * n_ethn / n_total
    )
    logger.info(
        "  Unknown                  : %d  (%.1f%%)",
        n_unknown,
        100 * n_unknown / n_total,
    )

    logger.info("Ethnicity distribution:")
    dist = combined["perceived_ethnicity"].value_counts()
    for cat, count in dist.items():
        logger.info("  %-28s %8d  (%.1f%%)", cat, count, 100 * count / n_total)

    lookup = combined[["Researcher_id", "perceived_ethnicity"]].copy()
    lookup.drop(columns=[], inplace=True)

    logger.info("=== Step 3: Saving lookup to %s ===", LOOKUP_OUT)
    lookup_df = combined[
        [
            "Researcher_id",
            "Name",
            "perceived_ethnicity",
            "__ethnicity_confidence",
            "__ethnicity_source",
        ]
    ]
    saved_path = None
    for dest in [LOOKUP_OUT, LOOKUP_FALLBACK]:
        try:
            lookup_df.to_csv(dest, index=False)
            logger.info("Lookup saved to %s (%d rows).", dest, len(combined))
            saved_path = dest
            break
        except PermissionError as exc:
            logger.warning("Cannot write to %s: %s — trying fallback path.", dest, exc)
        except Exception as exc:
            logger.warning("Error writing to %s: %s — trying fallback path.", dest, exc)

    if saved_path is None:
        logger.error(
            "Could not save lookup to any path. Results are in the checkpoint at %s",
            CHECKPOINT_PATH,
        )
    elif saved_path == LOOKUP_FALLBACK:
        logger.warning(
            "Lookup saved to fallback path %s (original path had permission error). "
            "Copy it manually to %s when permissions allow.",
            LOOKUP_FALLBACK,
            LOOKUP_OUT,
        )
    else:
        # Success at primary path — remove checkpoint so a clean re-run starts fresh
        try:
            os.remove(CHECKPOINT_PATH)
            logger.info("Checkpoint removed after successful save.")
        except OSError:
            pass

    return lookup


def merge_into_field_files(lookup: pd.DataFrame) -> None:
    """
    For each field file, merge lookup by Researcher_id (chunked) and save
    *_with_ethnicity.csv next to the original.
    """
    logger.info("=== Step 4: Merging ethnicity into field files ===")
    lookup_map = lookup.set_index("Researcher_id")["perceived_ethnicity"].to_dict()

    for field in FIELDS:
        in_path = os.path.join(
            DATA_DIR, f"DataFrameRankings_Genderize_Namsor_{field}.csv"
        )
        out_path = os.path.join(
            DATA_DIR, f"DataFrameRankings_Genderize_Namsor_{field}_with_ethnicity.csv"
        )
        out_path_fallback = os.path.join(
            _ETHNICITY_DIR,
            f"DataFrameRankings_Genderize_Namsor_{field}_with_ethnicity.csv",
        )

        if os.path.exists(out_path) or os.path.exists(out_path_fallback):
            logger.info("  %s already exists, skipping.", field)
            continue

        # Determine writable output path using os.access on the directory
        dest = out_path if os.access(DATA_DIR, os.W_OK) else out_path_fallback
        if dest == out_path_fallback:
            logger.warning(
                "  No write permission for %s — using fallback path.", out_path
            )

        logger.info("  Processing %s → %s …", field, dest)
        first_chunk = True

        for chunk in tqdm(
            pd.read_csv(in_path, chunksize=CHUNK_SIZE, low_memory=False),
            desc=f"  {field}",
        ):
            chunk["perceived_ethnicity"] = (
                chunk["Researcher_id"].map(lookup_map).fillna("Unknown")
            )
            try:
                chunk.to_csv(
                    dest,
                    index=False,
                    mode="w" if first_chunk else "a",
                    header=first_chunk,
                )
            except PermissionError:
                if dest == out_path_fallback:
                    raise  # already on fallback, nothing else to try
                logger.warning(
                    "  PermissionError on %s — switching to fallback path.", dest
                )
                dest = out_path_fallback
                chunk.to_csv(
                    dest,
                    index=False,
                    mode="w" if first_chunk else "a",
                    header=first_chunk,
                )
            first_chunk = False

        logger.info("  Saved %s", dest)

    logger.info("All field files processed.")


def print_summary(lookup_path: str) -> None:
    lookup = pd.read_csv(lookup_path)
    total = len(lookup)
    print("\n=== Ground Truth Ethnicity Inference Summary ===")
    print(f"Total unique researchers : {total:,}")
    n_demo = (lookup["__ethnicity_source"] == "demographicx").sum()
    n_ethn = (lookup["__ethnicity_source"] == "ethnicolr").sum()
    n_unknown = (lookup["__ethnicity_source"] == "unknown").sum()
    print(f"  classified by demographicx : {n_demo:,}  ({100*n_demo/total:.1f}%)")
    print(f"  classified by ethnicolr    : {n_ethn:,}  ({100*n_ethn/total:.1f}%)")
    print(f"  Unknown                    : {n_unknown:,}  ({100*n_unknown/total:.1f}%)")
    print("\nEthnicity distribution:")
    dist = lookup["perceived_ethnicity"].value_counts()
    for cat, count in dist.items():
        print(f"  {cat:<30} {count:>10,}  ({100*count/total:5.1f}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference and go straight to merging (lookup must exist)",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Only run inference and save lookup, skip merging into field files",
    )
    args = parser.parse_args()

    if DATA_DIR is None:
        raise SystemExit(
            "Set [data].ss_data_dir in config.ini (see config.ini.example)."
        )

    os.makedirs(_ETHNICITY_DIR, exist_ok=True)

    if not args.skip_inference:
        lookup = build_lookup()
    else:
        lookup_path = LOOKUP_OUT if os.path.exists(LOOKUP_OUT) else LOOKUP_FALLBACK
        logger.info("Skipping inference, loading existing lookup from %s", lookup_path)
        lookup = pd.read_csv(
            lookup_path, usecols=["Researcher_id", "perceived_ethnicity"]
        )

    if not args.skip_merge:
        merge_into_field_files(lookup)

    summary_path = LOOKUP_OUT if os.path.exists(LOOKUP_OUT) else LOOKUP_FALLBACK
    print_summary(summary_path)
