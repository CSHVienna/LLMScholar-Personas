"""
structural_model.py — The coauthorship graph and the similarity PCA, built over
every author of the ground truth (scripts/1_ground_truth/), so that every call is
scored in the same space. scripts/2_graph/2_model.py builds them (cached);
scripts/8_metrics/2_socials.py reads them.
"""

from pathlib import Path

import pandas as pd

from libs.metrics.coauthorship import build_coauthorship_graph
from libs.metrics.similarity import (
    build_author_features, build_similarity_embeddings, load_author_scholarly_stats,
)

GROUND_TRUTH = "../results/summary/ground_truth/ground_truth.parquet"
AUTHOR_STATS = "../results/summary/ground_truth/author_stats"
CACHE = Path("../results/.cache")
GRAPH_CACHE = CACHE / "coauthorship_graph_gt.joblib"
EMBEDDINGS_CACHE = CACHE / "similarity_embeddings_gt.joblib"


def ground_truth_population(path: str = GROUND_TRUTH) -> pd.DataFrame:
    """Every author of the ground truth with an OpenAlex id, with the columns the features use."""
    gt = pd.read_parquet(path, columns=["Researcher_id", "oa_id", "oa_works_count", "oa_cited_by_count",
                                        "oa_first_year", "oa_last_year"])
    gt = gt.dropna(subset=["oa_id"]).drop_duplicates("oa_id")
    return pd.DataFrame({
        "author_id": gt["oa_id"].str.split("/").str[-1],
        "researcher_id": gt["Researcher_id"],
        "oa_works_count": gt["oa_works_count"],
        "oa_cited_by_count": gt["oa_cited_by_count"],
        "oa_first_pub_year": gt["oa_first_year"],
        "oa_last_pub_year": gt["oa_last_year"],
    })


def graph_and_embeddings(oa_duckdb: str, ground_truth: str = GROUND_TRUTH,
                         author_stats: str = AUTHOR_STATS) -> dict:
    """The graph and the PCA embeddings over the whole ground truth (loaded from the cache when there)."""
    population = ground_truth_population(ground_truth)
    graph, graph_index = build_coauthorship_graph(
        population["author_id"], cache_path=GRAPH_CACHE, oa_duckdb_path=oa_duckdb, temp_dir=CACHE / "duckdb_tmp")
    features = build_author_features(population, stats=load_author_scholarly_stats(author_stats))
    embeddings, emb_index, _ = build_similarity_embeddings(features, cache_path=EMBEDDINGS_CACHE, index=graph_index)
    return {"graph": graph, "graph_index": graph_index, "embeddings": embeddings, "emb_index": emb_index}
