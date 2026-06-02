"""Inter-annotator agreement metrics shared by the agreement notebooks."""

from __future__ import annotations

import krippendorff
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    cohen_kappa_score,
    confusion_matrix,
)


def compute_agreement(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """Return raw agreement, Cohen κ, Krippendorff α on the two label columns.

    Both columns must be present in `df` and aligned by row.
    """
    agreed = int((df[col1] == df[col2]).sum())
    n = len(df)

    kappa = cohen_kappa_score(df[col1], df[col2])

    cats = sorted({*df[col1].dropna().unique(), *df[col2].dropna().unique()})
    cat_to_id = {c: i for i, c in enumerate(cats)}
    reliability = np.array(
        [
            df[col1].map(cat_to_id).to_numpy(),
            df[col2].map(cat_to_id).to_numpy(),
        ]
    )
    alpha = krippendorff.alpha(
        reliability_data=reliability, level_of_measurement="nominal"
    )

    return {
        "n": n,
        "agreed": agreed,
        "raw_agreement": agreed / n if n else float("nan"),
        "cohen_kappa": float(kappa),
        "krippendorff_alpha": float(alpha),
    }


def plot_confusion(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    *,
    labels: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (7, 6),
):
    """Render a confusion matrix for two annotator label columns.

    Returns the (fig, ax) pair. Caller is responsible for plt.show() / savefig.
    """
    import matplotlib.pyplot as plt

    if labels is None:
        labels = sorted({*df[col1].dropna().unique(), *df[col2].dropna().unique()})
    cm = confusion_matrix(df[col1], df[col2], labels=labels)
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    if title:
        ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig, ax


def compute_classification_metrics(
    df_labeled: pd.DataFrame,
    pred_col: str,
    *,
    truth_col: str = "manual_label",
    skip_label: str = "skip",
):
    """Compute accuracy / per-class metrics for a manual-vs-algorithmic labels table.

    Returns ``None`` if the labelled set is empty after dropping ``skip_label``.
    Otherwise returns a dict with keys: ``n``, ``accuracy``, ``labels``,
    ``classification_report`` (str), ``confusion_matrix`` (2D ndarray).
    Used by every `annotate_*` CLI to summarise inter-rater fit.
    """
    df = df_labeled[df_labeled[truth_col] != skip_label].copy()
    if df.empty:
        return None

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    y_true = df[truth_col]
    y_pred = df[pred_col]
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "n": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "labels": labels,
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def report_binary_dimension(
    dim_name: str,
    auto_pos: pd.Series,
    manual_pos: pd.Series,
    mask: pd.Series,
) -> dict | None:
    """Print TP/FP/FN/TN + accuracy/precision/recall/F1 for one binary dimension.

    `mask` selects the rows where a manual annotation exists; the two series
    must already be bool-coercible and aligned with `mask`. Returns a dict
    summary, or None if no rows are comparable.
    """
    auto_pos = auto_pos[mask].astype(bool).reset_index(drop=True)
    manual_pos = manual_pos[mask].astype(bool).reset_index(drop=True)
    n = len(auto_pos)
    if n == 0:
        print(f"── {dim_name} ──")
        print("  (no rows with manual annotation; nothing to compute)\n")
        return None

    TP = int((auto_pos & manual_pos).sum())
    FP = int((auto_pos & ~manual_pos).sum())
    FN = int((~auto_pos & manual_pos).sum())
    TN = int((~auto_pos & ~manual_pos).sum())

    prec = TP / (TP + FP) if (TP + FP) else float("nan")
    rec = TP / (TP + FN) if (TP + FN) else float("nan")
    acc = (TP + TN) / n
    f1 = (
        2 * prec * rec / (prec + rec)
        if (prec == prec and rec == rec and (prec + rec))
        else float("nan")
    )

    def _pct(x):
        return f"{100*x:.1f}%" if x == x else "N/A"

    print(f"── {dim_name} ──")
    print(f"  Comparable rows: {n}")
    print()
    print("  Confusion matrix:")
    print("                           manual positive   manual negative")
    print(f"     auto positive  →  TP={TP:<3}             FP={FP}")
    print(f"     auto negative  →  FN={FN:<3}             TN={TN}")
    print()
    print(f"  Accuracy  = (TP+TN)/N  = ({TP}+{TN})/{n} = {_pct(acc)}")
    prec_note = "  = " + _pct(prec) if (TP + FP) else " = N/A (auto marked no positives)"
    rec_note = "  = " + _pct(rec) if (TP + FN) else " = N/A (no real positives)"
    print(f"  Precision = TP/(TP+FP) = {TP}/{TP+FP}{prec_note}")
    print(f"  Recall    = TP/(TP+FN) = {TP}/{TP+FN}{rec_note}")
    print(f"  F1        = {_pct(f1)}")
    print()
    return {
        "dim": dim_name,
        "n": n,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "accuracy_%": round(100 * acc, 1),
        "precision_%": round(100 * prec, 1) if prec == prec else None,
        "recall_%": round(100 * rec, 1) if rec == rec else None,
        "f1_%": round(100 * f1, 1) if f1 == f1 else None,
    }


def print_agreement_summary(stats: dict) -> None:
    """Pretty-print the dict returned by `compute_agreement`."""
    print(f"N items:              {stats['n']}")
    print(f"Items agreed:         {stats['agreed']}")
    print()
    print(f"Raw agreement (p):    {stats['raw_agreement']:.4f}")
    print(f"Cohen's kappa (κ):    {stats['cohen_kappa']:.4f}")
    print(f"Krippendorff's α:     {stats['krippendorff_alpha']:.4f}")
