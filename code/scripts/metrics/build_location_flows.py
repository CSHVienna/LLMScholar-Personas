"""
build_location_flows.py — Pre-compute the prompt-country → author-country flow
table behind the location Sankey figure (issue #37).

For every recommendation that was matched to a real author, records the country
embedded in the prompt (`location`, normalised to ISO-2) and the country
OpenAlex assigns to that author (`location_oa_iso`). The output is a long table
that the plotting notebook aggregates however it needs — globally, per field,
per language or per model.

Output: <results_dir>/factualities/tables/location_flows.csv

Usage (from code/, with PYTHONPATH=.):
  python scripts/metrics/build_location_flows.py
  python scripts/metrics/build_location_flows.py --results <results_dir>

Defaults come from [data] in config.ini.
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

from libs.metrics.constants import (
    FIELD_NORM_MAP,
    LANGUAGE_NORM_MAP,
    LLM_COUNTRY_TO_ISO,
    VALID_FLAGS,
)
from libs.utils.config import get_results_path
from libs.utils.logging import setup_logging

warnings.filterwarnings("ignore")
logger = setup_logging()

# Columns pulled from factuality_full.csv — the flow needs per-author rows, which
# the per-call metrics table does not carry.
FLOW_COLS = [
    "location",
    "location_oa_iso",
    "valid_flag",
    "author_status",
    "oa_id",
    "field",
    "language",
    "model",
]
# Dimensions kept in the long output so the notebook can slice without re-reading
# the 6.7 GB source.
GROUP_COLS = ["src_iso", "dst_iso", "field_en", "language_en", "model"]


def build_flows(fact_path: Path, chunksize: int = 2_000_000) -> pd.DataFrame:
    """Aggregate (prompt country → author country) counts over factual authors.

    Read in chunks and reduced as we go: the source CSV does not fit in memory
    comfortably, but the grouped result is tiny.

    A recommendation counts as factual under the same rule the rest of the
    pipeline uses — matched in Semantic Scholar OR resolved in OpenAlex. Rows
    without `location_oa_iso` are dropped: OpenAlex has no country for that
    author, so there is no flow to draw (they are reported as a coverage figure
    by the caller, not silently ignored).
    """
    parts = []
    n_total = n_no_country = 0
    for i, chunk in enumerate(
        pd.read_csv(fact_path, usecols=FLOW_COLS, chunksize=chunksize, low_memory=False)
    ):
        valid = chunk[chunk["valid_flag"].isin(VALID_FLAGS)]
        factual = valid[
            (valid["author_status"] == "found")
            | (valid["oa_id"].notna() & (valid["oa_id"].astype(str) != ""))
        ]
        n_total += len(factual)
        n_no_country += int(factual["location_oa_iso"].isna().sum())

        flows = factual[factual["location_oa_iso"].notna()].copy()
        flows["src_iso"] = flows["location"].map(LLM_COUNTRY_TO_ISO)
        flows["dst_iso"] = flows["location_oa_iso"].astype(str).str.upper()
        flows["field_en"] = flows["field"].map(FIELD_NORM_MAP).fillna(flows["field"])
        flows["language_en"] = (
            flows["language"].map(LANGUAGE_NORM_MAP).fillna(flows["language"])
        )
        # Unmapped prompt country would silently vanish from the figure.
        unmapped = flows["src_iso"].isna().sum()
        if unmapped:
            logger.warning(
                "chunk %d: %d rows with an unmapped prompt country %s",
                i,
                unmapped,
                flows.loc[flows["src_iso"].isna(), "location"].unique()[:5],
            )
        flows = flows.dropna(subset=["src_iso"])
        parts.append(flows.groupby(GROUP_COLS, dropna=False).size().rename("n"))
        logger.info("chunk %d: %d factual rows, %d flows", i, len(factual), len(flows))

    out = (
        pd.concat(parts)
        .groupby(level=GROUP_COLS)
        .sum()
        .reset_index()
        .sort_values("n", ascending=False)
    )
    logger.info(
        "Factual recommendations: %d | without an OpenAlex country: %d (%.1f%%)",
        n_total,
        n_no_country,
        100 * n_no_country / n_total if n_total else 0,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default=None,
        help="Results directory (default: [data].results_dir in config.ini).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=2_000_000,
        help="Rows per read_csv chunk (default: 2,000,000).",
    )
    args = parser.parse_args()

    results = Path(args.results) if args.results else get_results_path()
    fact_path = results / "summary" / "factuality_full.csv"
    out_path = results / "factualities" / "tables" / "location_flows.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Reading %s", fact_path)
    flows = build_flows(fact_path, chunksize=args.chunksize)

    flows.to_csv(out_path, index=False)
    logger.info(
        "Saved %d flow rows (%d prompt countries → %d author countries) → %s",
        len(flows),
        flows["src_iso"].nunique(),
        flows["dst_iso"].nunique(),
        out_path,
    )


if __name__ == "__main__":
    main()
