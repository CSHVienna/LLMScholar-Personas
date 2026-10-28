"""
2_socials.py — Step 8.2: social representation of every valid call.

Takes the recommendations of 1_factuality.py and keeps the factual ones
(factuality_author = True), each researcher once per call. Per call:

  diversity_<x>     how spread the call's authors are over the classes F_a of x
                    (normalised entropy, paper Eq. 9: divided by log |F_a|).
                    NaN with fewer than 2 authors.
  parity_<x>        1 − total variation distance between the call's shares and
                    the ground truth's, over the same classes (paper Eq. 11)
  popularity_<x>    share of authors in the high tier of publications / citations
                    (pct_<x>_low / _med / _high: the share in each tier)

The classes F_a are every category that exists in the ground truth (every
Semantic Scholar researcher), not only the ones the recommendations show:
  gender       Female, Male, Unknown (Combined_gender of ground_truth.parquet;
               empty = Unknown)
  ethnicity    Asian, Black, White, Hispanic, Unknown (perceived_ethnicity of
               results/1_ground_truth/ethnicity/researcher_ethnicity_lookup.csv)
  location     every OpenAlex country of ground_truth.parquet (226)
  publications, citations   low / med / high: the per-field terciles (p33, p67)
               of oa_works_count / oa_cited_by_count in ground_truth.parquet, so
               the ground truth has about a third in each
Diversity leaves the authors with an unknown value out, and Unknown is not one of
its classes (paper Eq. 9: "Authors with unknown attribute values are excluded").
Parity keeps Unknown as a class, in the call and in the ground-truth reference.
The authors' own values come from the same tables, through the researcher_id
the factuality cascade picked.
  bias_location     share of authors whose OpenAlex country is the prompt's
                    country (mean of bias_location from step 3)
  similarity        mean pairwise scholarly similarity (paper Eq. 8)
  connectedness     coauthorship connectedness (paper Eq. 7)
                    (n_used_<x>: how many authors entered each of the two;
                    n_excluded_<x>: how many factual authors could not, e.g.
                    no OpenAlex id)

Similarity and connectedness use the coauthorship graph and the PCA built over
the whole ground truth (scripts/2_graph/, cached): each call's authors get a
subgraph and their embeddings from them, so every call is scored in the same space.

Output: ../results/8_metrics/socials/<model>.csv — call columns + n_authors + the
metrics above. Calls without factual authors get NaN.

Usage (from code/, with PYTHONPATH=.):
  python scripts/8_metrics/2_socials.py
"""

import argparse

import numpy as np
import pandas as pd

from libs.metrics.coauthorship import compute_connectedness
from libs.metrics.constants import (
    ETHNICITY_MAP, ETHNICITY_ORDER, FIELD_NORM_MAP, GENDER_MAP, GENDER_ORDER,
)
from libs.metrics.productivity import assign_productivity_tier, productivity_thresholds
from libs.metrics.similarity import compute_similarity
from libs.metrics.structural_model import GROUND_TRUTH, graph_and_embeddings
from libs.pipeline_common import CALL, OUT_DIR, logger, read_step, write_step
from libs.utils import ios
from libs.utils.config import config_default

ETHNICITY_GT = "../results/1_ground_truth/ethnicity/researcher_ethnicity_lookup.csv"   # the same table as 7_enrichment/4_ethnicity.py
# "Unknown" is a class of parity; diversity leaves it out (paper Eq. 9).
ETHNICITY = {**ETHNICITY_MAP, "Unknown": "Unknown"}
ETHNICITY_CLASSES = ETHNICITY_ORDER + ["Unknown"]
# Gender: an empty Combined_gender (a name the tool does not know) is "Unknown".
GENDER_CLASSES = GENDER_ORDER + ["Unknown"]
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

def ground_truth_classes(ground_truth: str) -> dict:
    """What the metrics take from the ground truth (every Semantic Scholar
    researcher, not the recommendations):
      categories  the classes F_a of each attribute (paper Eq. 9): every value
                  that exists in the ground truth (gender, ethnicity, OpenAlex
                  country; unknown values are not a class)
      shares      the gender and ethnicity distribution, the reference of parity
      cuts        per field p33 / p67 of OpenAlex works and citations, the
                  low / med / high tiers"""
    eth = pd.read_csv(ETHNICITY_GT, usecols=["Researcher_id", "perceived_ethnicity"])
    gt = pd.read_parquet(ground_truth, columns=["Researcher_id", "Field", "Combined_gender", "oa_country_code",
                                                 "oa_works_count", "oa_cited_by_count"])
    gt = gt.merge(eth, on="Researcher_id", how="left")
    gt["gender"] = gt["Combined_gender"].str.strip().str.lower().map(GENDER_MAP).fillna("Unknown")
    gt["ethnicity"] = gt["perceived_ethnicity"].map(ETHNICITY)
    categories = {
        "gender": [c for c in GENDER_CLASSES if c in set(gt["gender"].dropna())],
        "ethnicity": [c for c in ETHNICITY_CLASSES if c in set(gt["ethnicity"].dropna())],
        "location": sorted(gt["oa_country_code"].dropna().unique()),
    }
    return {
        "categories": categories,
        "shares": {col: gt[col].value_counts(normalize=True).reindex(categories[col], fill_value=0)
                   for col in ["gender", "ethnicity"]},
        "cuts": productivity_thresholds(gt.rename(columns={"Field": "field_en"}), field_col="field_en",
                                        metric_cols=["oa_works_count", "oa_cited_by_count"]),
    }


def openalex_id(oa_id: pd.Series) -> pd.Series:
    """oa_id as a plain id: it may be a URL, a float (5003003822.0) or a string."""
    return oa_id.dropna().map(
        lambda v: (str(int(v)) if isinstance(v, float) else str(v)).split("/")[-1].strip()
    ).reindex(oa_id.index)


# ── Similarity and connectedness ──────────────────────────────────────────────

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

def run(ground_truth: str, oa_duckdb: str) -> None:
    recs = read_step("factuality")
    recs["call"] = recs.groupby(CALL, dropna=False).ngroup()
    recs["author_id"] = openalex_id(recs["oa_id"])
    recs["field_en"] = recs["field"].map(FIELD_NORM_MAP).fillna(recs["field"])
    # The graph and the PCA of the whole ground truth (2_graph/), the same for every call.
    model = graph_and_embeddings(oa_duckdb)
    # The ground truth takes minutes to read, so it is cached; delete the
    # pickle if the ground-truth files change.
    gt = ios.cached_pickle(OUT_DIR / ".gt_classes_unknown_v2.pkl", lambda: ground_truth_classes(ground_truth))
    logger.info("Classes from the ground truth: %s",
                {k: len(v) for k, v in gt["categories"].items()})
    write_step(socials(recs, model, gt), step="socials")


def socials(recs: pd.DataFrame, model: dict, gt: dict) -> pd.DataFrame:
    """All the metrics of this step."""

    # The authors we score: factual, each researcher once per call.
    authors = recs[recs["factuality_author"]].drop_duplicates(["call", "researcher_id"]).copy()
    authors["gender"] = authors["gt_gender"].str.strip().str.lower().map(GENDER_MAP).fillna("Unknown")
    authors["ethnicity"] = authors["perceived_ethnicity"].map(ETHNICITY)

    out = recs.drop_duplicates("call").set_index("call")[CALL]
    out["n_authors"] = authors.groupby("call").size()
    out["n_authors"] = out["n_authors"].fillna(0).astype(int)

    # Gender and ethnicity; the classes and the parity reference come from the ground truth.
    # Diversity without the Unknown authors (paper Eq. 9); parity with Unknown as a class.
    for col in ["gender", "ethnicity"]:
        known = authors.assign(**{col: authors[col].where(authors[col] != "Unknown")})
        out[f"diversity_{col}"] = diversity(known, col, [c for c in gt["categories"][col] if c != "Unknown"])
        out[f"parity_{col}"] = parity(authors, col, gt["categories"][col], gt["shares"][col])

    # Location: the classes are every OpenAlex country of the ground truth.
    out["diversity_location"] = diversity(authors, "location_oa_iso", gt["categories"]["location"])
    out["bias_location"] = authors["bias_location"].astype(float).groupby(authors["call"]).mean()

    # Publications and citations: low / med / high within the author's field.
    # The per-field cut points come from every researcher of the ground truth.
    for col, name in [("oa_works_count", "publications"), ("oa_cited_by_count", "citations")]:
        cuts = gt["cuts"][col]
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
    parser.add_argument("--ground_truth", default=GROUND_TRUTH)
    parser.add_argument("--oa_duckdb", default=config_default("oa_duckdb"))
    args = parser.parse_args()
    run(args.ground_truth, args.oa_duckdb)


if __name__ == "__main__":
    main()
