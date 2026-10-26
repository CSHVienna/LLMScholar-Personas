"""
2_model.py — Step 2.2: the coauthorship graph and the similarity PCA over every
author of the ground truth (1_ground_truth/), with the stats of 1_author_stats.py.
Cached in ../results/.cache/ (coauthorship_graph_gt.joblib,
similarity_embeddings_gt.joblib); 8_metrics/2_socials.py takes a subgraph and
the embeddings of each call's authors from them.

Usage (from code/, with PYTHONPATH=.):
  python scripts/2_graph/2_model.py
"""

import argparse

from libs.metrics.structural_model import AUTHOR_STATS, GROUND_TRUTH, graph_and_embeddings
from libs.utils.config import config_default
from libs.utils.logging import setup_logging

logger = setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground_truth", default=GROUND_TRUTH)
    parser.add_argument("--author_stats", default=AUTHOR_STATS)
    parser.add_argument("--oa_duckdb", default=config_default("oa_duckdb"))
    args = parser.parse_args()
    model = graph_and_embeddings(args.oa_duckdb, args.ground_truth, args.author_stats)
    logger.info("Graph over %d authors (%d edges); embeddings of %d authors",
                model["graph"].shape[0], model["graph"].nnz // 2, model["embeddings"].shape[0])


if __name__ == "__main__":
    main()
