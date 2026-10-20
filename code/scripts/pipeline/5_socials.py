"""
5_socials.py — Step 5: social representation of every valid call.

Takes the recommendations of step 3 and keeps the factual ones
(factuality_author = True), each researcher once per call. Per call:

  diversity_<x>     how spread the call's authors are over the categories of x
                    (normalised entropy, paper Eq. 9). NaN with fewer than 2.
                    x = gender, ethnicity, location (OpenAlex country),
                    publications, citations (low / med / high in their field)
  parity_<x>        1 − distance to a reference distribution (paper Eq. 11):
                    the ground truth for gender and ethnicity, an even split
                    over low / med / high for publications and citations
  popularity_<x>    share of authors in the high tier of publications / citations
                    (pct_<x>_low / _med / _high: the share in each tier)
  bias_location     share of authors whose OpenAlex country is the prompt's
                    country (mean of bias_location from step 3)
  similarity        mean pairwise scholarly similarity (paper Eq. 8)
  connectedness     coauthorship connectedness (paper Eq. 7)
                    (n_used_<x>: how many authors entered each of the two;
                    n_excluded_<x>: how many factual authors could not, e.g.
                    no OpenAlex id)

Runs once per OpenAlex variant (common.VARIANTS). The coauthorship graph and the
PCA are built once, over the authors of all variants, so the variants are
scored in the same space.

Output: 5_socials/<variant>/<model>.csv — call columns + n_authors + the metrics
above. Calls without factual authors get NaN.

Usage (from code/, with PYTHONPATH=.):
  python scripts/pipeline/5_socials.py
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CALL, OUT_DIR, VARIANTS, read_step, write_step  # noqa: E402

from libs.metrics.coauthorship import build_coauthorship_graph, compute_connectedness  # noqa: E402
from libs.metrics.constants import (  # noqa: E402
    ETHNICITY_MAP, ETHNICITY_ORDER, FIELD_NORM_MAP, GENDER_MAP, GENDER_ORDER,
)
from libs.metrics.productivity import assign_productivity_tier, productivity_thresholds  # noqa: E402
from libs.metrics.similarity import (  # noqa: E402
    build_author_features, build_similarity_embeddings, compute_similarity,
    load_author_scholarly_stats,
)
from libs.utils import ios  # noqa: E402
from libs.utils.config import config_default  # noqa: E402

# The coauthorship graph and the PCA are expensive, so they are cached here.
CACHE = Path("../results/.cache")
ETHNICITY_GT = "../results/ethnicity/DataFrameRankings_Genderize_Namsor_*_with_ethnicity.csv"
TIERS = ["low", "med", "high"]


# ── Small helpers ─────────────────────────────────────────────────────────────

def shares(authors: pd.DataFrame, col: str, categories=None) -> pd.DataFrame:
    """Per call (rows) and category (columns): share of the call's authors.
    Authors with an unknown category are left out."""
    counts = pd.crosstab(authors["call"], authors[col])
    if categories is not None:
        counts = counts.reindex(columns=categories, fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0)


def entropy(p: pd.DataFrame) -> pd.Series:
    p = p.where(p > 0)
    return -(p * np.log(p)).sum(axis=1)


def diversity(authors: pd.DataFrame, col: str, categories) -> pd.Series:
    """Entropy over a fixed list of categories, divided by its maximum."""
    p = shares(authors, col, categories)
    n = authors.dropna(subset=[col]).groupby("call").size()
    return (entropy(p) / np.log(len(categories))).where(n >= 2)


def parity(authors: pd.DataFrame, col: str, categories, reference: pd.Series) -> pd.Series:
    """1 − total variation distance between the call's shares and `reference`."""
    p = shares(authors, col, categories)
    return 1 - 0.5 * (p - reference).abs().sum(axis=1)


# ── Inputs ────────────────────────────────────────────────────────────────────

def ground_truth_shares(parquet: str) -> dict[str, pd.Series]:
    """Gender and ethnicity distribution of the Semantic Scholar researchers."""
    eth = pd.concat(pd.read_csv(f, usecols=["Researcher_id", "Year", "perceived_ethnicity"])
                    for f in sorted(glob.glob(ETHNICITY_GT)))
    eth = eth.sort_values("Year").groupby("Researcher_id").last().reset_index()
    gender = pd.read_parquet(parquet, columns=["Researcher_id", "Combined_gender"])
    gt = eth.merge(gender, on="Researcher_id", how="left")
    gt["gender"] = gt["Combined_gender"].str.strip().str.lower().map(GENDER_MAP)
    gt["ethnicity"] = gt["perceived_ethnicity"].map(ETHNICITY_MAP)
    return {
        "gender": gt["gender"].value_counts(normalize=True).reindex(GENDER_ORDER, fill_value=0),
        "ethnicity": gt["ethnicity"].value_counts(normalize=True).reindex(ETHNICITY_ORDER, fill_value=0),
    }


def openalex_id(oa_id: pd.Series) -> pd.Series:
    """oa_id as a plain id: it may be a URL, a float (5003003822.0) or a string."""
    return oa_id.dropna().map(
        lambda v: (str(int(v)) if isinstance(v, float) else str(v)).split("/")[-1].strip()
    ).reindex(oa_id.index)


# ── Similarity and connectedness ──────────────────────────────────────────────

def graph_and_embeddings(population: pd.DataFrame, oa_duckdb: str, oa_author_stats: str) -> dict:
    """The coauthorship graph and the PCA embeddings, built once over every
    recommended author with an OpenAlex id in any variant, so that all variants
    are scored in the same space."""
    population = population.dropna(subset=["author_id"]).drop_duplicates("author_id")
    graph, graph_index = build_coauthorship_graph(
        population["author_id"], cache_path=CACHE / "coauthorship_graph.joblib",
        oa_duckdb_path=oa_duckdb, temp_dir=CACHE / "duckdb_tmp")
    features = build_author_features(population, stats=load_author_scholarly_stats(oa_author_stats))
    embeddings, emb_index, _ = build_similarity_embeddings(
        features, cache_path=CACHE / "similarity_embeddings.joblib", index=graph_index)
    return {"graph": graph, "graph_index": graph_index,
            "embeddings": embeddings, "emb_index": emb_index}


def structural(authors: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Similarity and connectedness of each call's factual authors. Authors
    without an OpenAlex id are skipped (and counted in n_excluded_*)."""
    graph, graph_index = model["graph"], model["graph_index"]
    embeddings, emb_index = model["embeddings"], model["emb_index"]
    rows = []
    for call, ids in authors.groupby("call")["author_id"]:
        no_oa_id = int(ids.isna().sum())          # factual authors OpenAlex did not find
        ids = ids.dropna().unique().tolist()
        sim = compute_similarity(ids, embeddings, emb_index)
        conn = compute_connectedness(ids, graph, graph_index)
        rows.append({"call": call,
                     "similarity": sim.value, "n_used_similarity": sim.n_used,
                     "n_excluded_similarity": sim.n_excluded + no_oa_id,
                     "connectedness": conn.value, "n_used_connectedness": conn.n_used,
                     "n_excluded_connectedness": conn.n_excluded + no_oa_id})
    return pd.DataFrame(rows).set_index("call")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(parquet: str, oa_duckdb: str, oa_author_stats: str) -> None:
    recs = {}
    for variant in VARIANTS:
        r = read_step(3, variant=variant)
        r["call"] = r.groupby(CALL, dropna=False).ngroup()
        r["author_id"] = openalex_id(r["oa_id"])
        r["field_en"] = r["field"].map(FIELD_NORM_MAP).fillna(r["field"])
        recs[variant] = r
    model = graph_and_embeddings(pd.concat(recs.values()), oa_duckdb, oa_author_stats)
    # The ground truth takes minutes to read, so it is cached; delete the
    # pickle if the ground-truth files change.
    gt = ios.cached_pickle(OUT_DIR / ".gt_shares.pkl", lambda: ground_truth_shares(parquet))
    for variant in VARIANTS:
        write_step(socials(recs[variant], model, gt), step=5, variant=variant)


def socials(recs: pd.DataFrame, model: dict, gt: dict) -> pd.DataFrame:
    """All the metrics of step 5 for one variant."""

    # The authors we score: factual, each researcher once per call.
    authors = recs[recs["factuality_author"]].drop_duplicates(["call", "researcher_id"]).copy()
    authors["gender"] = authors["gt_gender"].str.strip().str.lower().map(GENDER_MAP)
    authors["ethnicity"] = authors["perceived_ethnicity"].map(ETHNICITY_MAP)

    out = recs.drop_duplicates("call").set_index("call")[CALL]
    out["n_authors"] = authors.groupby("call").size()
    out["n_authors"] = out["n_authors"].fillna(0).astype(int)

    # Gender and ethnicity.
    for col, categories in [("gender", GENDER_ORDER), ("ethnicity", ETHNICITY_ORDER)]:
        out[f"diversity_{col}"] = diversity(authors, col, categories)
        out[f"parity_{col}"] = parity(authors, col, categories, gt[col])

    # Location: the categories are the countries seen in each call.
    p = shares(authors, "location_oa_iso")
    n_countries = (p > 0).sum(axis=1)
    out["diversity_location"] = (entropy(p) / np.log(n_countries)).where(n_countries >= 2)
    out["bias_location"] = authors["bias_location"].astype(float).groupby(authors["call"]).mean()

    # Publications and citations: low / med / high within the author's field.
    # The per-field cut points come from every recommended author found.
    found = recs[recs["factuality_name"] | recs["author_id"].notna()]
    for col, name in [("oa_works_count", "publications"), ("oa_cited_by_count", "citations")]:
        cuts = productivity_thresholds(found, field_col="field_en", metric_cols=[col])[col]
        authors[name] = assign_productivity_tier(authors[col], authors["field_en"], cuts)
        even = pd.Series(1 / len(TIERS), index=TIERS)
        out[f"diversity_{name}"] = diversity(authors, name, TIERS)
        out[f"parity_{name}"] = parity(authors, name, TIERS, even)
        tier_shares = shares(authors, name, TIERS)
        for tier in TIERS:
            out[f"pct_{name}_{tier}"] = tier_shares[tier]
        out[f"popularity_{name}"] = tier_shares["high"]

    out = out.join(structural(authors, model))

    metrics = ["diversity_gender", "diversity_ethnicity", "diversity_location",
               "diversity_publications", "diversity_citations",
               "parity_gender", "parity_ethnicity", "parity_publications", "parity_citations",
               "popularity_publications", "popularity_citations",
               *[f"pct_{x}_{t}" for x in ["publications", "citations"] for t in TIERS],
               "bias_location",
               "similarity", "n_used_similarity", "n_excluded_similarity",
               "connectedness", "n_used_connectedness", "n_excluded_connectedness"]
    return out[CALL + ["n_authors"] + metrics]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", default=config_default("ss_parquet"))
    parser.add_argument("--oa_duckdb", default=config_default("oa_duckdb"))
    parser.add_argument("--oa_author_stats", default=config_default("oa_author_stats"))
    args = parser.parse_args()
    run(args.parquet, args.oa_duckdb, args.oa_author_stats)


if __name__ == "__main__":
    main()
