"""
Lookup the raw LLM output for a given summary.csv row index.

Usage:
  python scripts/annotation/lookup_output.py <index> [--results_dir ../results] [--summary_csv ../results/summary/summary.csv]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from annotate_responses import load_raw_content


def main():
    parser = argparse.ArgumentParser(
        description="Print the raw LLM output for a summary row."
    )
    parser.add_argument(
        "index", type=int, help="Row index in summary.csv (= original_index)"
    )
    parser.add_argument("--results_dir", default="../results")
    parser.add_argument(
        "--summary_csv", default="../results/summary/summary.csv"
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_csv)
    header = pd.read_csv(summary_path, nrows=0).columns

    # Read only the requested row (skip header + earlier rows)
    row_df = pd.read_csv(
        summary_path,
        skiprows=range(1, args.index + 1),
        nrows=1,
        header=0,
        names=header,
        low_memory=False,
    )
    if row_df.empty:
        print(f"Index {args.index} out of range.", file=sys.stderr)
        sys.exit(1)
    row = row_df.iloc[0]

    print(f"index      : {args.index}")
    for c in [
        "model",
        "language",
        "role",
        "task",
        "location",
        "k",
        "target",
        "field",
        "subfield",
        "run_id",
        "valid_flag",
    ]:
        print(f"{c:10s} : {row[c]}")
    print("─" * 80)

    content = load_raw_content(args.results_dir, row)
    if content is None:
        print("[Could not locate response in JSON]")
    elif content.strip() == "":
        print("[Empty response]")
    else:
        print(content)


if __name__ == "__main__":
    main()
