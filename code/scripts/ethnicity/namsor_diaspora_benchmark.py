#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark the NamSor `diaspora` endpoint against a hand-built ground-truth set
of real authors, measuring how much the optional `countryIso2` parameter moves
the answer.

Three arms are run over the SAME names:

  country    countryIso2 = the author's real country of affiliation
  nocountry  countryIso2 omitted from the payload entirely
  us         countryIso2 = "US" for everyone

The third arm exists because NamSor appears to echo `countryIso2: "US"` on
responses where no country was sent (see data/ethnicity_inference/
lifted_nocountry.jsonl). If `nocountry` and `us` come back identical, then
"no country" is not a neutral condition — it is a US prior — and the two-arm
comparison people usually run is measuring the wrong contrast.

Ground truth CSV must carry at least:
  id, first_name, last_name, work_country_iso2, expected_diaspora, stratum

`stratum` is `native` (works in country of origin) or `diaspora` (works abroad).
The split matters: passing the country should help the natives and hurt the
diaspora cases, so a single pooled accuracy number hides the real effect.

Examples
--------
python namsor_diaspora_benchmark.py --input ../../../data/ethnicity_inference/diaspora_benchmark_authors.csv --dry-run
python namsor_diaspora_benchmark.py --input ../../../data/ethnicity_inference/diaspora_benchmark_authors.csv --outdir ../../../results/ethnicity/diaspora_benchmark
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from namsor import BATCH_SIZE, ENDPOINTS, load_namsor_key, log, namsor_batch  # noqa: E402

ARMS = ("country", "nocountry", "us")
REQUIRED_COLS = [
    "id",
    "first_name",
    "last_name",
    "work_country_iso2",
    "expected_diaspora",
    "stratum",
]


def build_records(df: pd.DataFrame, arm: str) -> list[dict]:
    """One personalNames payload per author for the given arm."""
    records = []
    for _, r in df.iterrows():
        rec = {
            "id": str(r["id"]),
            "firstName": str(r["first_name"]).strip(),
            "lastName": str(r["last_name"]).strip(),
        }
        if arm == "country":
            iso2 = str(r["work_country_iso2"]).strip().upper()
            if iso2 and iso2 != "NAN":
                rec["countryIso2"] = iso2
        elif arm == "us":
            rec["countryIso2"] = "US"
        # arm == "nocountry" → send no countryIso2 at all
        records.append(rec)
    return records


def run_arm(api_key: str, df: pd.DataFrame, arm: str, outdir: Path, sleep: float) -> pd.DataFrame:
    cfg = ENDPOINTS["diaspora"]
    records = build_records(df, arm)
    log(f"[{arm}] {len(records)} names → ~{len(records)*cfg['credits']:,} credits")

    items: list[dict] = []
    jsonl = outdir / f"namsor_diaspora_{arm}.jsonl"
    with jsonl.open("w", encoding="utf-8") as out:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            got = namsor_batch(api_key, batch, cfg["url"])
            for item in got:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
            items.extend(got)
            log(f"  [{arm}] {len(items)}/{len(records)} done")

    flat = pd.DataFrame(items)
    flat["arm"] = arm
    # ethnicitiesTop is a ranked list; keep it as a list for top-k, plus a flat copy
    if "ethnicitiesTop" in flat.columns:
        flat["ethnicitiesTop_str"] = flat["ethnicitiesTop"].apply(
            lambda v: "|".join(map(str, v)) if isinstance(v, list) else ""
        )
    return flat


# Label pairs NamSor does not document a difference between, so a swap between
# them is not really a wrong answer. `Hispanic` and `HispanoLatino` are both in
# the diaspora taxonomy (140 classes) with no published rule separating them;
# scoring them as distinct would charge the model for our own labelling guess.
# Scored both ways below, so the choice is visible instead of baked in.
EQUIVALENT_LABELS = [{"Hispanic", "HispanoLatino"}]


def canonical(label: str) -> str:
    """Collapse each undocumented-synonym group to one representative label."""
    for group in EQUIVALENT_LABELS:
        if label in group:
            return sorted(group)[0]
    return label


def top_k_hit(row, k: int, collapse: bool = False) -> bool:
    top = row.get("ethnicitiesTop")
    if not isinstance(top, list):
        return False
    expected = row["expected_diaspora"]
    if collapse:
        return canonical(expected) in [canonical(t) for t in top[:k]]
    return expected in top[:k]


def score(merged: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per arm, overall and split by stratum."""
    merged["correct_top1"] = merged["ethnicity"] == merged["expected_diaspora"]
    merged["correct_top3"] = merged.apply(lambda r: top_k_hit(r, 3), axis=1)
    merged["correct_top1_collapsed"] = merged["ethnicity"].map(canonical) == merged[
        "expected_diaspora"
    ].map(canonical)
    merged["correct_top3_collapsed"] = merged.apply(
        lambda r: top_k_hit(r, 3, collapse=True), axis=1
    )

    rows = []
    for arm in merged["arm"].unique():
        sub = merged[merged["arm"] == arm]
        for stratum in ["ALL"] + sorted(sub["stratum"].dropna().unique().tolist()):
            s = sub if stratum == "ALL" else sub[sub["stratum"] == stratum]
            if s.empty:
                continue
            rows.append(
                {
                    "arm": arm,
                    "stratum": stratum,
                    "n": len(s),
                    "top1_acc": round(s["correct_top1"].mean(), 4),
                    "top3_acc": round(s["correct_top3"].mean(), 4),
                    "top1_acc_collapsed": round(s["correct_top1_collapsed"].mean(), 4),
                    "top3_acc_collapsed": round(s["correct_top3_collapsed"].mean(), 4),
                    "mean_prob": round(pd.to_numeric(s.get("probabilityCalibrated"), errors="coerce").mean(), 4),
                }
            )
    return pd.DataFrame(rows)


def check_nocountry_defaults_to_us(merged: pd.DataFrame) -> None:
    """Is 'no country' actually a neutral condition, or a US prior in disguise?"""
    piv = merged.pivot_table(
        index="id", columns="arm", values="ethnicity", aggfunc="first"
    )
    if not {"nocountry", "us"}.issubset(piv.columns):
        return
    same = (piv["nocountry"] == piv["us"]).mean()
    log(f"\nnocountry vs us — identical top-1 label on {same:.1%} of names")
    if same > 0.98:
        log("  → 'no country' behaves as countryIso2=US. Not a neutral baseline.")
    else:
        log("  → the two arms genuinely differ; omitting the country is its own condition.")

    echoed = merged[merged["arm"] == "nocountry"].get("countryIso2")
    if echoed is not None:
        log(f"  countryIso2 echoed back on the nocountry arm: {echoed.dropna().unique().tolist()[:5]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="Ground-truth authors CSV.")
    p.add_argument("--outdir", type=Path, default=Path("../../../results/ethnicity/diaspora_benchmark"))
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--dry-run", action="store_true", help="Print payloads and credit cost, call nothing.")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: input is missing required columns: {missing}")
    log(f"Loaded {len(df)} authors from {args.input}")
    log(f"  strata: {df['stratum'].value_counts().to_dict()}")
    log(f"  distinct expected_diaspora labels: {df['expected_diaspora'].nunique()}")

    cost = len(df) * len(args.arms) * ENDPOINTS["diaspora"]["credits"]
    log(f"  planned cost: {len(df)} names x {len(args.arms)} arms x 20 credits = {cost:,} credits")

    if args.dry_run:
        for arm in args.arms:
            log(f"\n[{arm}] first 3 payload entries:")
            for rec in build_records(df, arm)[:3]:
                print("  " + json.dumps(rec, ensure_ascii=False))
        return

    api_key = load_namsor_key()
    if not api_key:
        sys.exit("ERROR: no NamSor API key (NAMSOR_API_KEY or config.ini [namsor].data_dir).")

    args.outdir.mkdir(parents=True, exist_ok=True)
    frames = [run_arm(api_key, df, arm, args.outdir, args.sleep) for arm in args.arms]
    responses = pd.concat(frames, ignore_index=True)

    merged = responses.merge(
        df[REQUIRED_COLS + [c for c in ("full_name", "origin_country_iso2") if c in df.columns]],
        left_on="id",
        right_on="id",
        how="left",
    )

    per_name = args.outdir / "per_name_results.csv"
    merged.drop(columns=["ethnicitiesTop"], errors="ignore").to_csv(per_name, index=False)
    log(f"\nPer-name results → {per_name}")

    summary = score(merged)
    summary_path = args.outdir / "accuracy_summary.csv"
    summary.to_csv(summary_path, index=False)
    log(f"Accuracy summary → {summary_path}\n")
    print(summary.to_string(index=False))

    check_nocountry_defaults_to_us(merged)


if __name__ == "__main__":
    main()
