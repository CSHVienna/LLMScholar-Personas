"""
build_ethnicity_distributions.py — Pre-compute ethnicity distributions for the
analysis notebook.

Reads the BERT/ethnicolr ground-truth CSVs (one per field), the unique-researcher
lookup, and the LLM recommendations with their inferred ethnicity. Writes one
small CSV per distribution so the notebook can plot directly without any heavy
groupby on ~140M rows.

Outputs under <results_dir>/ethnicity/distributions/:
  - gt_overall.csv          rows: ethnicity, count
  - gt_per_field.csv        rows: field, ethnicity, count
  - rec_overall.csv         rows: ethnicity, count
  - rec_per_field.csv       rows: field_en, ethnicity, count
  - rec_per_model.csv       rows: model, ethnicity, count

Usage (from code/, with PYTHONPATH=.):
  python scripts/metrics/build_ethnicity_distributions.py
  python scripts/metrics/build_ethnicity_distributions.py --results <results_dir>

The results directory defaults to [data].results_dir in config.ini.
"""

import argparse
from pathlib import Path

import pandas as pd

from libs.metrics.constants import FIELD_NORM_MAP, VALID_FLAGS
from libs.utils.config import get_results_path
from libs.utils.logging import setup_logging
from libs.visuals.constants import ETHNICITY_PLOT_ORDER

logger = setup_logging()

FIELDS = ["Biology", "Computer_Science", "Mathematics", "Physics", "Psychology", "Sociology"]


def _gt_field_path(eth_dir: Path, field: str) -> Path:
    return eth_dir / f"DataFrameRankings_Genderize_Namsor_{field}_with_ethnicity.csv"


def _vc_to_df(series: pd.Series, name: str) -> pd.DataFrame:
    return (
        series.reindex([e for e in ETHNICITY_PLOT_ORDER if e in series.index])
        .rename("count")
        .rename_axis(name)
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=None, help="Results directory.")
    args = parser.parse_args()

    results = Path(args.results) if args.results else get_results_path()
    eth_dir = results / "ethnicity"
    out_dir = eth_dir / "distributions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. GT — overall
    lookup_path = eth_dir / "researcher_ethnicity_lookup.csv"
    logger.info("Loading %s", lookup_path)
    lookup = pd.read_csv(lookup_path)
    gt_overall = lookup["perceived_ethnicity"].value_counts()
    _vc_to_df(gt_overall, "ethnicity").to_csv(out_dir / "gt_overall.csv", index=False)

    # 2. GT — per field
    rows = []
    for field in FIELDS:
        path = _gt_field_path(eth_dir, field)
        if not path.exists():
            logger.warning("Missing %s — skipping", path)
            continue
        logger.info("Loading %s", path)
        df = pd.read_csv(
            path,
            usecols=["Researcher_id", "perceived_ethnicity"],
            dtype={"perceived_ethnicity": "category"},
        ).drop_duplicates("Researcher_id")
        vc = df["perceived_ethnicity"].value_counts()
        for eth, count in vc.items():
            rows.append({"field": field.replace("_", " "), "ethnicity": eth, "count": int(count)})
    pd.DataFrame(rows).to_csv(out_dir / "gt_per_field.csv", index=False)

    # 3. Recommendations — overall, per field, per model
    rec_path = results / "summary" / "recommendations_with_ethnicity.csv"
    logger.info("Loading %s", rec_path)
    rdf = pd.read_csv(rec_path, low_memory=False)
    rdf_valid = rdf[rdf["valid_flag"].isin(VALID_FLAGS)].copy()
    rdf_valid["field_en"] = rdf_valid["field"].map(FIELD_NORM_MAP).fillna(rdf_valid["field"])

    _vc_to_df(rdf_valid["perceived_ethnicity"].value_counts(), "ethnicity").to_csv(
        out_dir / "rec_overall.csv", index=False
    )

    (
        rdf_valid.groupby(["field_en", "perceived_ethnicity"])
        .size()
        .rename("count")
        .reset_index()
        .to_csv(out_dir / "rec_per_field.csv", index=False)
    )
    (
        rdf_valid.groupby(["model", "perceived_ethnicity"])
        .size()
        .rename("count")
        .reset_index()
        .to_csv(out_dir / "rec_per_model.csv", index=False)
    )

    logger.info("Wrote 5 distribution CSVs to %s", out_dir)


if __name__ == "__main__":
    main()
