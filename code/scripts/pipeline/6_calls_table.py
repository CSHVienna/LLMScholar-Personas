"""
6_calls_table.py — Step 6: one table with every metric of every call, for the
notebooks.

Joins the outputs of steps 1, 2, 3 and 5 (one row per call and OpenAlex
variant; consistency is per setup, so every call of a setup gets the same value)
and adds:
  oa_variant                    how the OpenAlex author was chosen (common.VARIANTS);
                                filter on it — every call appears once per variant
  factuality_name / _author     share of the call's authors that pass (step 3,
                                Semantic Scholar; the same in every variant)
  factuality_field / _seniority the same, among the authors whose name exists
  oa_factuality_*               the same four shares for the checks in OpenAlex
  n_authors_found               how many of the call's authors exist
  model_access, model_size, model_class, model_family, model_short_name
                                from data/models/
  <param>_en                    the params in English (language_en, field_en, ...)
  duplicates_c, refusals_c      1 − duplicates, 1 − refusals

The columns keep the names the notebooks use (location, target, refusals,
div_*, *_works, ...).

Output: ../results/pipeline/6_calls.csv (one file, all models).

Usage (from code/, with PYTHONPATH=.):
  python scripts/pipeline/6_calls_table.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CALL, OUT_DIR, SETUP, VARIANTS, author_key, logger, read_step  # noqa: E402

from libs.metrics.constants import (  # noqa: E402
    FIELD_NORM_MAP, LANGUAGE_NORM_MAP, LOCATION_NORM_MAP, ROLE_NORM_MAP,
    SUBFIELD_NORM_MAP, TARGET_NORM_MAP, TASK_NORM_MAP,
)
from libs.utils import ios  # noqa: E402

MODELS_DIR = Path("../data/models")
OUTPUT = OUT_DIR / "6_calls.csv"

# Pipeline name → name used by the notebooks.
NOTEBOOK_NAMES = {
    "country": "location", "seniority": "target", "refusal": "refusals",
    "diversity_gender": "div_gender", "diversity_ethnicity": "div_ethnicity",
    "diversity_location": "div_location",
    "diversity_publications": "div_productivity_works",
    "diversity_citations": "div_productivity_citations",
    "parity_publications": "parity_works", "popularity_publications": "popularity_works",
    "pct_publications_low": "pct_works_low", "pct_publications_med": "pct_works_med",
    "pct_publications_high": "pct_works_high",
    "n_authors": "n_factual_authors",
}
ENGLISH = {"language": LANGUAGE_NORM_MAP, "location": LOCATION_NORM_MAP,
           "field": FIELD_NORM_MAP, "subfield": SUBFIELD_NORM_MAP,
           "task": TASK_NORM_MAP, "target": TARGET_NORM_MAP, "role": ROLE_NORM_MAP}


def factuality_per_call(variant: str) -> pd.DataFrame:
    """Share of each call's authors that pass each check (each name once), in
    Semantic Scholar (factuality_*) and in OpenAlex (oa_factuality_*)."""
    checks = [f"{p}factuality_{s}" for p in ["", "oa_"] for s in ["name", "field", "seniority", "author"]]
    recs = read_step(3, CALL + ["name", "lastname"] + checks, variant=variant)
    recs = recs.assign(author=author_key(recs)).drop_duplicates(CALL + ["author"])
    calls = recs.groupby(CALL, dropna=False)
    out = calls[[f"{p}factuality_{s}" for p in ["", "oa_"] for s in ["name", "author"]]].mean()
    out["n_authors_found"] = calls["factuality_name"].sum()
    for p in ["", "oa_"]:   # field and seniority: among the authors whose name exists
        found = recs[recs[f"{p}factuality_name"]].groupby(CALL, dropna=False)
        out = out.join(found[[f"{p}factuality_field", f"{p}factuality_seniority"]].mean())
    return out.reset_index()


def model_metadata(models: pd.Series) -> pd.DataFrame:
    """Access, size, class, family and short name of each model."""
    clean = models.str.replace(":", "-").str.replace(".", "_")   # as in metadata.json
    meta = ios.load_json(MODELS_DIR / "metadata.json")
    short = {
        m: m.replace("-2025-04-14", "")
            .replace("b-maverick-128e-instruct-q4_K_M", "B-maverick")
            .replace("b-scout-16e-instruct-q4_K_M", "B-scout")
            .split("b-")[0].lower()
        for m in ios.read_list_from_file(MODELS_DIR / "models.txt")
    }
    out = pd.DataFrame({"model": clean})
    for col, key in [("model_size", "size"), ("model_access", "access"),
                     ("model_class", "reasoning"), ("model_family", "family")]:
        out[col] = clean.map(lambda m: meta[m][key] if m in meta else "Unknown")
    out["model_short_name"] = clean.map(lambda m: short.get(m, m))
    return out


def run() -> None:
    base = read_step(1)                                    # every call, valid or not
    base = base.merge(read_step(2), on=SETUP, how="left")
    tables = []
    for variant in VARIANTS:
        t = base.merge(factuality_per_call(variant), on=CALL, how="left")
        t = t.merge(read_step(5, variant=variant), on=CALL, how="left")
        tables.append(t.assign(oa_variant=variant))
    calls = pd.concat(tables, ignore_index=True)

    calls[["validity", "refusal"]] = calls[["validity", "refusal"]].astype(int)
    calls = calls.rename(columns=NOTEBOOK_NAMES)
    calls["duplicates_c"] = 1 - calls["duplicates"]
    calls["refusals_c"] = 1 - calls["refusals"]
    for col, mapping in ENGLISH.items():
        calls[f"{col}_en"] = calls[col].map(mapping).fillna(calls[col])

    meta = model_metadata(calls["model"])
    calls["model"] = meta["model"].to_numpy()
    for col in meta.columns.drop("model"):
        calls[col] = meta[col].to_numpy()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    calls.to_csv(OUTPUT, index=False)
    logger.info("Wrote %s (%d rows = calls × variants, %d columns)", OUTPUT, *calls.shape)


if __name__ == "__main__":
    run()
