"""
Worker script for parallel ethnicity inference.
Called by apply_ethnicity_ground_truth_parallel.py — do not run directly.

Usage:
    python ethnicity_worker.py <chunk_csv> <output_csv> <worker_id> <batch_size>
"""

import logging
import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from tqdm import tqdm

from ethnicity_inference import infer_ethnicity_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker-%(name)s] %(levelname)s - %(message)s",
)


def main():
    chunk_csv  = sys.argv[1]
    output_csv = sys.argv[2]
    worker_id  = int(sys.argv[3])
    batch_size = int(sys.argv[4])

    logger = logging.getLogger(str(worker_id))
    logger.info("Starting — reading %s", chunk_csv)

    df = pd.read_csv(chunk_csv)
    names = df["Name"].tolist()
    logger.info("Names to process: %d", len(names))

    categories, sources = [], []
    n_batches = (len(names) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(names), batch_size), total=n_batches,
                  desc=f"worker-{worker_id}", position=worker_id):
        batch = names[i : i + batch_size]
        results = infer_ethnicity_batch(batch, batch_size=batch_size)
        for r in results:
            categories.append(r["category"] or "Unknown")
            sources.append(r["source"])

    df["perceived_ethnicity"] = categories
    df["__ethnicity_source"]  = sources
    df.to_csv(output_csv, index=False)
    logger.info("Done — saved %s", output_csv)


if __name__ == "__main__":
    main()
