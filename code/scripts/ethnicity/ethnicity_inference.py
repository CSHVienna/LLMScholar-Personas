"""
Ethnicity inference module for LLMScholar-Personas.

Implements the cascade model from:
"Whose Name Comes Up? Auditing LLM-Based Scholar Recommendations" (arXiv:2506.00074)

Cascade order:
  1. demographicx (BERT-based, liamliang/demographics_race_v2)
     → use if max probability >= DEMOGRAPHICX_THRESHOLD
  2. ethnicolr (LSTM-based, Florida voter registry model)
     → fallback if demographicx confidence is too low
  3. Unknown — if both fail or confidence is below threshold

Valid output categories:
  - Asian
  - White
  - Black or African American
  - Hispanic or Latino
  - Unknown
"""

import os

# Must be set before any TensorFlow/Keras import
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pandas as pd
import torch

from libs.utils.ios import read_input_csv, write_output_csv
from libs.utils.logging import setup_logging

logger = setup_logging()

# ─── Constants ───────────────────────────────────────────────────────────────

DEMOGRAPHICX_THRESHOLD = 0.5  # min max-probability to trust demographicx
ETHNICOLR_THRESHOLD = 0.5  # min max-probability to trust ethnicolr

# Canonical 5 categories
VALID_CATEGORIES = [
    "Asian",
    "White",
    "Black or African American",
    "Hispanic or Latino",
    "Unknown",
]

# demographicx output indices → canonical categories
# Model output order: [white, hispanic, black, asian]  (from classifier.py)
_DEMOGRAPHICX_LABELS = [
    "White",
    "Hispanic or Latino",
    "Black or African American",
    "Asian",
]

# ethnicolr 'race' column values → canonical categories
_ETHNICOLR_MAP = {
    "nh_white": "White",
    "asian": "Asian",
    "hispanic": "Hispanic or Latino",
    "nh_black": "Black or African American",
}
_ETHNICOLR_PROB_COLS = ["nh_white", "asian", "hispanic", "nh_black"]

# ─── Lazy model singletons ───────────────────────────────────────────────────

_demographicx_model = None
_demographicx_tokenizer = None
_ethnicolr_ready = False


def _load_demographicx():
    """Load BERT model for ethnicity inference (once)."""
    global _demographicx_model, _demographicx_tokenizer
    if _demographicx_model is None:
        logger.info("Loading demographicx BERT model (liamliang/demographics_race_v2)…")
        from transformers import AutoTokenizer, BertForSequenceClassification

        _demographicx_model = BertForSequenceClassification.from_pretrained(
            "liamliang/demographics_race_v2"
        )
        _demographicx_model.eval()
        _demographicx_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        logger.info("demographicx model loaded.")


def _load_ethnicolr():
    """Trigger ethnicolr model download/cache (once)."""
    global _ethnicolr_ready
    if not _ethnicolr_ready:
        import ethnicolr  # noqa: F401 — side-effect: downloads models on first use

        _ethnicolr_ready = True


# ─── Core inference ──────────────────────────────────────────────────────────


def _demographicx_batch(names: list[str]) -> list[dict]:
    """
    Run demographicx BERT inference on a list of full names.

    Returns a list of dicts with keys:
      {'category': str, 'confidence': float, 'source': 'demographicx'}
    or {'category': None, 'confidence': 0.0, 'source': 'demographicx'}
    if confidence < threshold.
    """
    _load_demographicx()

    # Replicate demographicx.classifier.get_name_pair logic exactly:
    # input is (word-level name, char-level name), both lowercased;
    # double spaces from embedded spaces in name are collapsed.
    def _make_pair(n: str):
        s = str(n).lower()
        chars = " ".join(s).replace("  ", " ").replace("  ", " ")
        return s, chars

    pairs = [_make_pair(n) for n in names]
    try:
        encoded = _demographicx_tokenizer(
            pairs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        with torch.no_grad():
            logits = _demographicx_model(**encoded).logits
        probs = torch.softmax(logits, dim=1).numpy()
    except Exception as exc:
        logger.warning("demographicx batch failed: %s", exc)
        return [{"category": None, "confidence": 0.0, "source": "demographicx"}] * len(
            names
        )

    results = []
    for prob_row in probs:
        max_idx = int(np.argmax(prob_row))
        confidence = float(prob_row[max_idx])
        if confidence >= DEMOGRAPHICX_THRESHOLD:
            results.append(
                {
                    "category": _DEMOGRAPHICX_LABELS[max_idx],
                    "confidence": confidence,
                    "source": "demographicx",
                }
            )
        else:
            results.append(
                {"category": None, "confidence": confidence, "source": "demographicx"}
            )
    return results


def _ethnicolr_batch(full_names: list[str]) -> list[dict]:
    """
    Run ethnicolr Florida voter registry model on a list of full names.
    Splits each name into first + last by whitespace (last token = last name).

    Returns a list of dicts with keys:
      {'category': str, 'confidence': float, 'source': 'ethnicolr'}
    or {'category': None, 'confidence': 0.0, 'source': 'ethnicolr'} on failure.
    """
    _load_ethnicolr()
    import ethnicolr

    rows = []
    for fn in full_names:
        parts = str(fn).strip().split()
        if len(parts) >= 2:
            rows.append({"__idx": len(rows), "first": parts[0], "last": parts[-1]})
        else:
            rows.append(
                {"__idx": len(rows), "first": parts[0] if parts else "", "last": ""}
            )

    df = pd.DataFrame(rows)
    defaults = [{"category": None, "confidence": 0.0, "source": "ethnicolr"}] * len(
        full_names
    )

    # Names with no last name cannot be processed by ethnicolr
    has_last = df["last"].str.strip() != ""
    if not has_last.any():
        return defaults

    try:
        sub = df[has_last].copy()
        result = ethnicolr.pred_fl_reg_name(sub, lname_col="last", fname_col="first")

        for _, row in result.iterrows():
            idx = int(row["__idx"])
            max_prob_col = max(
                _ETHNICOLR_PROB_COLS, key=lambda c: float(row.get(c, 0.0))
            )
            confidence = float(row.get(max_prob_col, 0.0))
            race_label = str(row.get("race", "")).strip()
            canonical = _ETHNICOLR_MAP.get(race_label)

            if canonical and confidence >= ETHNICOLR_THRESHOLD:
                defaults[idx] = {
                    "category": canonical,
                    "confidence": confidence,
                    "source": "ethnicolr",
                }
            else:
                defaults[idx] = {
                    "category": None,
                    "confidence": confidence,
                    "source": "ethnicolr",
                }
    except Exception as exc:
        logger.warning("ethnicolr batch failed: %s", exc)

    return defaults


def infer_ethnicity(full_name: str) -> str:
    """
    Infer perceived ethnicity from a full name using the cascade:
      demographicx → ethnicolr → Unknown

    Parameters
    ----------
    full_name : str
        The scholar's full name (first + last, at minimum).

    Returns
    -------
    str
        One of: 'Asian', 'White', 'Black or African American',
                'Hispanic or Latino', 'Unknown'
    """
    result = infer_ethnicity_batch([full_name])[0]
    return result["category"] or "Unknown"


def infer_ethnicity_batch(
    full_names: list[str],
    batch_size: int = 64,
) -> list[dict]:
    """
    Infer perceived ethnicity for a list of full names using the cascade model.

    Parameters
    ----------
    full_names : list[str]
    batch_size : int
        Batch size for BERT inference.

    Returns
    -------
    list of dicts with keys: 'category', 'confidence', 'source'
      category: one of VALID_CATEGORIES (or None before Unknown assignment)
      confidence: float probability from winning model
      source: 'demographicx', 'ethnicolr', or 'unknown'
    """
    n = len(full_names)
    final_results: list[dict | None] = [None] * n

    # ── Step 1: demographicx pass ────────────────────────────────────────────
    demo_results = []
    for i in range(0, n, batch_size):
        batch = full_names[i : i + batch_size]
        demo_results.extend(_demographicx_batch(batch))

    fallback_indices = []
    for i, res in enumerate(demo_results):
        if res["category"] is not None:
            final_results[i] = res
        else:
            fallback_indices.append(i)

    # ── Step 2: ethnicolr fallback ───────────────────────────────────────────
    if fallback_indices:
        fallback_names = [full_names[i] for i in fallback_indices]
        ethn_results = []
        for i in range(0, len(fallback_names), batch_size):
            batch = fallback_names[i : i + batch_size]
            ethn_results.extend(_ethnicolr_batch(batch))

        for local_i, global_i in enumerate(fallback_indices):
            res = ethn_results[local_i]
            if res["category"] is not None:
                final_results[global_i] = res
            else:
                final_results[global_i] = {
                    "category": "Unknown",
                    "confidence": 0.0,
                    "source": "unknown",
                }

    return final_results  # type: ignore[return-value]


# ─── Application entry point ─────────────────────────────────────────────────


def _apply_to_recommendations(
    input_path: str,
    output_path: str,
    batch_size: int = 64,
    checkpoint_every: int = 10000,
) -> None:
    """
    Load recommendations CSV, infer ethnicity for all scholars, save results.

    Deduplicates names before inference to avoid redundant computation,
    then joins back to the full dataset.
    """
    from tqdm import tqdm

    df = read_input_csv(input_path, logger=logger)
    total_rows = len(df)

    # Build full_name column
    df["__full_name"] = (
        df["name"].fillna("").astype(str).str.strip()
        + " "
        + df["lastname"].fillna("").astype(str).str.strip()
    ).str.strip()

    # Rows without any name → Unknown immediately
    no_name_mask = (df["name"].isna() & df["lastname"].isna()) | (
        df["__full_name"] == ""
    )

    # Unique names that need inference
    names_to_infer = df.loc[~no_name_mask, "__full_name"].unique().tolist()
    logger.info(
        "Unique names to infer: %d (out of %d rows)", len(names_to_infer), total_rows
    )

    # ── Run cascade inference ────────────────────────────────────────────────
    categories: list[str] = []
    sources: list[str] = []

    for i in tqdm(
        range(0, len(names_to_infer), batch_size), desc="Inferring ethnicity"
    ):
        batch = names_to_infer[i : i + batch_size]
        results = infer_ethnicity_batch(batch, batch_size=batch_size)
        for r in results:
            categories.append(r["category"] or "Unknown")
            sources.append(r["source"])

    # ── Build lookup map ─────────────────────────────────────────────────────
    name_to_category = dict(zip(names_to_infer, categories))
    name_to_source = dict(zip(names_to_infer, sources))

    df["perceived_ethnicity"] = (
        df["__full_name"].map(name_to_category).fillna("Unknown")
    )
    df["__ethnicity_source"] = df["__full_name"].map(name_to_source).fillna("unknown")

    # Rows with no name get Unknown
    df.loc[no_name_mask, "perceived_ethnicity"] = "Unknown"
    df.loc[no_name_mask, "__ethnicity_source"] = "unknown"

    # ── Logging summary ──────────────────────────────────────────────────────
    n_demo = (df["__ethnicity_source"] == "demographicx").sum()
    n_ethn = (df["__ethnicity_source"] == "ethnicolr").sum()
    n_unknown = (df["__ethnicity_source"] == "unknown").sum()

    logger.info("Classification breakdown (by row):")
    logger.info("  demographicx : %d  (%.1f%%)", n_demo, 100 * n_demo / total_rows)
    logger.info("  ethnicolr    : %d  (%.1f%%)", n_ethn, 100 * n_ethn / total_rows)
    logger.info(
        "  unknown      : %d  (%.1f%%)", n_unknown, 100 * n_unknown / total_rows
    )

    logger.info("Perceived ethnicity distribution:")
    dist = df["perceived_ethnicity"].value_counts()
    for cat, count in dist.items():
        logger.info("  %-28s %7d  (%.1f%%)", cat, count, 100 * count / total_rows)

    # ── Drop internal helper columns and save ────────────────────────────────
    df.drop(columns=["__full_name", "__ethnicity_source"], inplace=True)

    write_output_csv(df, output_path, logger=logger)

    # ── Print summary to stdout ──────────────────────────────────────────────
    print("\n=== Ethnicity Inference Summary ===")
    print(f"Total rows processed  : {total_rows:,}")
    print(f"Unique names inferred : {len(names_to_infer):,}")
    print(
        f"  classified by demographicx : {n_demo:,} rows ({100*n_demo/total_rows:.1f}%)"
    )
    print(
        f"  classified by ethnicolr    : {n_ethn:,} rows ({100*n_ethn/total_rows:.1f}%)"
    )
    print(
        f"  Unknown (no inference)     : {n_unknown:,} rows ({100*n_unknown/total_rows:.1f}%)"
    )
    print("\nPerceived ethnicity distribution:")
    for cat, count in dist.items():
        print(f"  {cat:<30} {count:>8,}  ({100*count/total_rows:5.1f}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Infer perceived ethnicity for LLMScholar-Personas recommendations."
    )
    from libs.utils.config import get_results_path

    try:
        _results = get_results_path()
        _default_input = str(_results / "summary" / "recommendations.csv")
        _default_output = str(_results / "summary" / "recommendations_with_ethnicity.csv")
    except (FileNotFoundError, KeyError, ValueError):
        _default_input = None
        _default_output = None

    parser.add_argument(
        "--input",
        default=_default_input,
        required=_default_input is None,
        help="Path to input recommendations CSV (default from [data].results_dir).",
    )
    parser.add_argument(
        "--output",
        default=_default_output,
        required=_default_output is None,
        help="Path to output CSV with perceived_ethnicity column added (default from [data].results_dir).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for BERT inference (default: 64)",
    )
    args = parser.parse_args()

    _apply_to_recommendations(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
    )
