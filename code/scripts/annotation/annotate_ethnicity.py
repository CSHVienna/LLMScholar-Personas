"""
Interactive CLI tool for manually annotating ethnicity inference labels.

Labels:
  a = Asian
  w = White
  b = Black or African American
  h = Hispanic or Latino
  u = Unknown
  s = skip         (not sure, skip this sample)
  q = quit         (stop and compute metrics from labeled so far)

Usage:
  python scripts/annotation/annotate_ethnicity.py \
    --lookup results/ethnicity/researcher_ethnicity_lookup.csv \
    --output data/ethnicity_inference/manual_labels_annotator1.csv \
    --sample_csv data/ethnicity_inference/sample_100.csv

After labeling, computes accuracy/precision/recall/F1 vs algorithmic label.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Constants ──────────────────────────────────────────────────────────────────

VALID_LABELS = {
    "a": "Asian",
    "w": "White",
    "b": "Black or African American",
    "h": "Hispanic or Latino",
    "u": "Unknown",
    "s": "skip",
    "q": "quit",
}

ETHNICITY_CATEGORIES = [
    "Asian",
    "White",
    "Black or African American",
    "Hispanic or Latino",
    "Unknown",
]

from libs.metrics.agreement import compute_classification_metrics
from libs.utils.cli import (
    BLUE,
    BOLD,
    CYAN,
    GRAY,
    GREEN,
    MAGENTA,
    RED,
    RESET,
    YELLOW,
    clear_screen,
    colorize,
    print_separator,
)

CATEGORY_COLORS = {
    "Asian": CYAN,
    "White": GREEN,
    "Black or African American": MAGENTA,
    "Hispanic or Latino": YELLOW,
    "Unknown": GRAY,
}

SOURCE_COLORS = {
    "demographicx": BLUE,
    "ethnicolr": CYAN,
    "unknown": GRAY,
}


# ── Helpers ────────────────────────────────────────────────────────────────────


def confidence_bar(conf: float, width: int = 20) -> str:
    filled = round(conf * width)
    bar = "█" * filled + "░" * (width - filled)
    color = GREEN if conf >= 0.75 else YELLOW if conf >= 0.5 else RED
    return colorize(f"[{bar}] {conf:.2%}", color)


# ── Display ────────────────────────────────────────────────────────────────────


def display_sample(i: int, total: int, row: pd.Series):
    clear_screen()
    print_separator("═")
    print(colorize(f"  ETHNICITY ANNOTATOR  [{i}/{total}]", BOLD + CYAN))
    print_separator("═")

    # Name
    name = str(row.get("Name", ""))
    researcher_id = row.get("Researcher_id", "")
    print(f"  {BOLD}Name:{RESET}           {BOLD}{name}{RESET}")
    print(f"  {BOLD}Researcher ID:{RESET}  {GRAY}{researcher_id}{RESET}")
    print_separator()

    print_separator()

    # Legend
    print(
        f"  {BOLD}Label:{RESET} "
        f"{colorize('[a]sian', CYAN)}  "
        f"{colorize('[w]hite', GREEN)}  "
        f"{colorize('[b]lack or African American', MAGENTA)}  "
        f"{colorize('[h]ispanic or Latino', YELLOW)}  "
        f"{colorize('[u]nknown', GRAY)}  "
        f"{GRAY}[s]kip  [q]uit{RESET}"
    )
    print_separator("═")


# ── Metrics ────────────────────────────────────────────────────────────────────


def compute_metrics(df_labeled: pd.DataFrame):
    stats = compute_classification_metrics(df_labeled, pred_col="perceived_ethnicity")
    if stats is None:
        print(colorize("No labeled samples (excluding skips). Cannot compute metrics.", RED))
        return

    print_separator("═")
    print(colorize("  EVALUATION RESULTS", BOLD + CYAN))
    print_separator("═")
    acc_str = f"{stats['accuracy']:.4f}"
    print(f"  Samples labeled (excl. skip): {stats['n']}")
    print(f"  Overall Accuracy: {colorize(acc_str, BOLD + GREEN)}")
    print()
    print("  Per-class metrics (manual = truth, algo = prediction):")
    print_separator()
    for line in stats["classification_report"].splitlines():
        print("  " + line)
    print_separator()
    print("  Confusion matrix (rows=manual, cols=algo):")
    print(f"  Labels: {stats['labels']}")
    for idx, row_cm in enumerate(stats["confusion_matrix"]):
        print(f"  {stats['labels'][idx]:30s} | {row_cm}")
    print_separator("═")


# ── Sampling ───────────────────────────────────────────────────────────────────


def sample_rows(
    df: pd.DataFrame, n: int, seed: int, already_done_idx: set
) -> pd.DataFrame:
    df_remaining = df[~df.index.isin(already_done_idx)]
    categories = df_remaining["perceived_ethnicity"].unique()
    per_class = max(1, n // len(categories))
    frames = []
    for cat in categories:
        sub = df_remaining[df_remaining["perceived_ethnicity"] == cat]
        k = min(per_class, len(sub))
        frames.append(sub.sample(n=k, random_state=seed))
    sample = pd.concat(frames).sample(frac=1, random_state=seed)
    if len(sample) < n:
        leftover = df_remaining[~df_remaining.index.isin(sample.index)]
        extra = leftover.sample(
            n=min(n - len(sample), len(leftover)), random_state=seed
        )
        sample = pd.concat([sample, extra]).sample(frac=1, random_state=seed)
    return sample.head(n).reset_index(drop=False)


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Manually annotate ethnicity inference labels"
    )
    parser.add_argument(
        "--lookup", required=True, help="Path to researcher_ethnicity_lookup.csv"
    )
    parser.add_argument(
        "--output", required=True, help="CSV to save/resume manual labels"
    )
    parser.add_argument(
        "--n", type=int, default=100, help="Number of samples to label (default: 100)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample_csv",
        default=None,
        help="Use a pre-defined sample CSV instead of sampling",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    # Load existing labels (for resumability)
    if output_path.exists():
        df_done = pd.read_csv(output_path)
        if "manual_label" in df_done.columns:
            df_done = df_done[
                df_done["manual_label"].notna() & (df_done["manual_label"] != "")
            ]
        else:
            df_done = pd.DataFrame()
        if len(df_done) > 0:
            print(
                colorize(f"Resuming: {len(df_done)} samples already labeled.", YELLOW)
            )
        already_done_idx = (
            set(df_done["sample_index"].tolist()) if not df_done.empty else set()
        )
    else:
        df_done = pd.DataFrame()
        already_done_idx = set()

    # Load sample
    if args.sample_csv:
        print(colorize(f"Using pre-defined sample: {args.sample_csv}", CYAN))
        sample = pd.read_csv(args.sample_csv)
        sample = sample[~sample["sample_index"].isin(already_done_idx)].reset_index(
            drop=True
        )
    else:
        print(f"Loading lookup: {args.lookup}")
        df = pd.read_csv(args.lookup, low_memory=False)
        print(f"Total rows: {len(df):,}")
        sample = sample_rows(df, args.n, args.seed, already_done_idx)

    remaining = len(sample)

    if remaining == 0:
        print(colorize("All sampled rows already labeled. Showing metrics.", GREEN))
        compute_metrics(df_done)
        return

    print(f"Labeling {remaining} rows. Press Enter to begin...")
    input()

    new_labels = []

    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        display_sample(
            i + len(already_done_idx), len(already_done_idx) + remaining, row
        )

        while True:
            try:
                key = input("  Your label: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                key = "q"

            if key in VALID_LABELS:
                break
            print(
                colorize(
                    f"  Invalid key '{key}'. Use: {list(VALID_LABELS.keys())}", RED
                )
            )

        if key == "q":
            print(colorize("\nQuitting early...", YELLOW))
            break

        label = VALID_LABELS[key]
        new_labels.append(
            {
                "sample_index": row.get("sample_index", row.name),
                "Researcher_id": row.get("Researcher_id", ""),
                "Name": row.get("Name", ""),
                "perceived_ethnicity": row.get("perceived_ethnicity", ""),
                "__ethnicity_confidence": row.get("__ethnicity_confidence", ""),
                "__ethnicity_source": row.get("__ethnicity_source", ""),
                "manual_label": label,
            }
        )

        # Save incrementally
        df_new = pd.DataFrame(new_labels)
        df_all = (
            pd.concat([df_done, df_new], ignore_index=True)
            if not df_done.empty
            else df_new
        )
        df_all.to_csv(output_path, index=False)

    df_new = pd.DataFrame(new_labels)
    df_all = (
        pd.concat([df_done, df_new], ignore_index=True) if not df_done.empty else df_new
    )

    if not df_all.empty:
        df_all.to_csv(output_path, index=False)
        print(colorize(f"\nLabels saved to {output_path}", GREEN))
        compute_metrics(df_all)
    else:
        print(colorize("No labels saved.", RED))


if __name__ == "__main__":
    main()
