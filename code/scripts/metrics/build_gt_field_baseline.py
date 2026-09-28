"""
build_gt_field_baseline.py — Field mix of the name-matching reference population.

The Sankey figures that end in `gt_field` are read as "field drift": ask for a
number theorist, get a biologist. That reading only holds against a baseline,
because the reference table the matcher searches is not balanced across fields —
Biology alone is ~31% of it. A subfield sending 40% of its recommendations to
Biology is therefore *below* what picking a name at random would give, not above.

Writes one tiny CSV, <results_dir>/factualities/tables/gt_field_baseline.csv,
with the share of each field among the deduplicated Semantic Scholar researchers
(the exact table factuality_author_jw.py matches against).

Usage (from code/, with PYTHONPATH=.):
  python scripts/metrics/build_gt_field_baseline.py
"""

import argparse

import pandas as pd

from libs.utils.config import config_default, get_results_path
from libs.utils.logging import setup_logging

logger = setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=None)
    parser.add_argument("--ss-parquet", default=config_default("ss_parquet"))
    args = parser.parse_args()

    ss_parquet = args.ss_parquet
    if not ss_parquet:
        parser.error("Set [data].ss_parquet in config.ini or pass --ss-parquet")
    results = get_results_path(path=args.results)

    logger.info("Reading %s", ss_parquet)
    fields = pd.read_parquet(ss_parquet, columns=["Field"])["Field"]
    counts = fields.value_counts(dropna=False)
    out = (
        counts.rename("n_researchers")
        .rename_axis("field")
        .reset_index()
        .assign(share=lambda d: 100 * d.n_researchers / d.n_researchers.sum())
        .sort_values("n_researchers", ascending=False)
    )

    fn = results / "factualities" / "tables" / "gt_field_baseline.csv"
    fn.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(fn, index=False)
    logger.info("Wrote %s\n%s", fn, out.to_string(index=False))


if __name__ == "__main__":
    main()
