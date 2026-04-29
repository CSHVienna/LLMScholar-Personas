"""
Interactive CLI tool for manually annotating LLM response quality labels.

Labels:
  i = invalid      (empty or unparseable)
  c = cleaned      (answered but with extra verbosity/text)
  u = unchanged    (perfect JSON response)
  r = refused      (LLM explained why it can't answer)
  f = fixed_dict   (JSON present but corrupted/partially fixed)
  s = skip         (not sure, skip this sample)
  q = quit         (stop and compute metrics from labeled so far)

Usage:
  python annotate_responses.py \
    --results_dir ../../results \
    --summary_csv ../../results/summary/summary.csv \
    --output ../../results/manual_labels.csv \
    --n 100 \
    [--stratified] [--seed 42] [--model gpt-4.1-2025-04-14] [--language english]

After labeling, computes accuracy/precision/recall/F1 vs algorithmic valid_flag.
"""

import argparse
import json
import os
import re
import sys
import textwrap
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)


# ── Constants ──────────────────────────────────────────────────────────────────

VALID_LABELS = {'i': 'invalid', 'c': 'cleaned', 'u': 'unchanged',
                'r': 'refused', 'f': 'fixed_dict', 'e': 'empty',
                's': 'skip', 'q': 'quit'}

# Labels that count as "valid" for binary accuracy computation
VALID_GROUP = {'cleaned', 'unchanged', 'fixed_dict'}
INVALID_GROUP = {'invalid', 'empty', 'refused'}

RESULTS_PATH_TEMPLATE = '{root}/responses/results_{source}_{language}'

# Colors (ANSI)
RESET  = '\033[0m'
BOLD   = '\033[1m'
CYAN   = '\033[96m'
YELLOW = '\033[93m'
GREEN  = '\033[92m'
RED    = '\033[91m'
GRAY   = '\033[90m'
MAGENTA = '\033[95m'
BLUE   = '\033[94m'

LABEL_COLORS = {
    'invalid':    RED,
    'cleaned':    YELLOW,
    'unchanged':  GREEN,
    'refused':    MAGENTA,
    'fixed_dict': CYAN,
    'empty':      GRAY,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def colorize(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def _compute_cleaned(text: str) -> tuple[str, bool]:
    """Apply the same cleaning logic as batch_parse_results/text.py.
    Returns (cleaned_text, was_changed) so we can show both lengths."""
    original = text
    m = re.search(r'```[a-zA-Z]*\s*([\s\S]*?)```', text)
    if m:
        text = m.group(1).strip()
    for v in "aeiou":
        text = text.replace(f'\\\"{v}', f"{v}\u0308")
        text = text.replace(f'\\\"{v.upper()}', f"{v.upper()}\u0308")
    text = re.sub(r'\\\"([^"]+)\\\"', r'\1', text)
    try:
        decoded = text.encode('raw_unicode_escape').decode('unicode_escape')
        text = re.sub(r'[\ud800-\udfff]', '', decoded)
    except Exception:
        pass
    text = text.replace('\\\"\"', '\"')
    text = text.replace('\\\",', '\",')
    text = text.replace('\\\"', '\"')
    text = text.replace('},\n    \"', '\",\n    \"')
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text, text != original


def get_source(model: str) -> str:
    m = model.lower()
    if 'gemini' in m:
        return 'gemini'
    if 'gpt' in m and 'gpt-oss' not in m:
        return 'gpt'
    return 'ollama'


def model_to_filename(model: str) -> str:
    """Convert model name to filename format (dots → underscores, colons → hyphens)."""
    return model.replace('.', '_').replace(':', '-')


def build_json_path(results_dir: str, source: str, language: str, model: str) -> Path:
    folder = RESULTS_PATH_TEMPLATE.format(root=results_dir, source=source, language=language)
    filename = f"{source}_{language}_{model_to_filename(model)}.json"
    return Path(folder) / filename


def extract_raw_content(response: dict, source: str) -> str:
    """Extract raw text content from a response object depending on source."""
    if source == 'gemini':
        return (response.get('response', {})
                        .get('candidates', [{}])[0]
                        .get('content', {})
                        .get('parts', [{}])[0]
                        .get('text', ''))
    elif source == 'ollama':
        return response.get('message', {}).get('content', '')
    else:  # gpt
        msg = (response.get('response', {})
                       .get('body', {})
                       .get('choices', [{}])[0]
                       .get('message', {}))
        return msg.get('content', '') or msg.get('refusal', '') or ''


def _find_matching_entry(data, row):
    """Return the raw response dict matching the row parameters, or None.

    run_id in the CSV is 1-indexed (first run = 1), so run_idx = run_id - 1
    converts it to a 0-based list index. The bounds check guards against
    missing runs in the JSON file.
    """
    for key, obj in data.items():
        p = obj.get('parameters', {})
        pc = p.get('persona_context', {})
        ur = p.get('user_request', {})
        if (pc.get('role', '') == row['role'] and
                pc.get('task', '') == row['task'] and
                pc.get('location', '') == row['location'] and
                ur.get('k') == row['k'] and
                ur.get('target', '') == row['target'] and
                ur.get('field') == row['field'] and
                ur.get('subfield') == row['subfield']):
            run_idx = int(row['run_id']) - 1   # run_id is 1-indexed
            responses = obj.get('responses', [])
            if 0 <= run_idx < len(responses):  # bounds check
                return responses[run_idx]
    return None


def load_raw_content(results_dir: str, row: pd.Series) -> str | None:
    """Load the raw response text for a given summary row."""
    source = get_source(row['model'])
    json_path = build_json_path(results_dir, source, row['language'], row['model'])

    if not json_path.exists():
        return None

    try:
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return f"[Error loading file: {e}]"

    response = _find_matching_entry(data, row)
    return extract_raw_content(response, source) if response is not None else None


# ── Sampling ───────────────────────────────────────────────────────────────────

def sample_rows(df: pd.DataFrame, n: int, stratified: bool, seed: int,
                already_done_idx: set) -> pd.DataFrame:
    """Sample n rows, optionally stratified by valid_flag, excluding already done."""
    df_remaining = df[~df.index.isin(already_done_idx)]

    if stratified:
        classes = df_remaining['valid_flag'].unique()
        per_class = max(1, n // len(classes))
        frames = []
        for cls in classes:
            sub = df_remaining[df_remaining['valid_flag'] == cls]
            k = min(per_class, len(sub))
            frames.append(sub.sample(n=k, random_state=seed))
        sample = pd.concat(frames).sample(frac=1, random_state=seed)  # shuffle
        # If still short, top up randomly
        if len(sample) < n:
            leftover = df_remaining[~df_remaining.index.isin(sample.index)]
            extra = leftover.sample(n=min(n - len(sample), len(leftover)), random_state=seed)
            sample = pd.concat([sample, extra]).sample(frac=1, random_state=seed)
    else:
        k = min(n, len(df_remaining))
        sample = df_remaining.sample(n=k, random_state=seed)

    return sample.head(n).reset_index(drop=False)  # keep original index


# ── Display ────────────────────────────────────────────────────────────────────

def print_separator(char='─', width=80):
    print(colorize(char * width, GRAY))


def display_sample(i: int, total: int, row: pd.Series, content: str | None,
                   show_algo_label: bool = False):
    os.system('clear' if os.name == 'posix' else 'cls')
    print_separator('═')
    print(colorize(f"  RESPONSE ANNOTATOR  [{i}/{total}]", BOLD + CYAN))
    print_separator('═')

    # Context
    algo_flag = row['valid_flag']
    flag_color = LABEL_COLORS.get(algo_flag, RESET)
    print(f"  {BOLD}Model:{RESET}     {row['model']}")
    print(f"  {BOLD}Language:{RESET}  {row['language']}")
    print(f"  {BOLD}Role:{RESET}      {row['role']} — {row['task']}")
    print(f"  {BOLD}Location:{RESET}  {row['location']}")
    print(f"  {BOLD}Request:{RESET}   k={row['k']} {row['target']} in {row['field']} / {row['subfield']}")
    print(f"  {BOLD}Run ID:{RESET}    {row['run_id']}")
    if show_algo_label:
        print(f"  {BOLD}Algo label:{RESET} {colorize(algo_flag, flag_color)}")
    print_separator()

    # Raw content + lengths
    if content is None:
        print(colorize("  [Could not load raw content]", RED))
    elif content.strip() == '':
        print(colorize("  [Empty response]", RED))
    else:
        cleaned, was_changed = _compute_cleaned(content)
        len_raw = len(content)
        len_cleaned = len(cleaned)
        diff = len_raw - len_cleaned
        if was_changed:
            len_info = colorize(f"raw={len_raw}  cleaned={len_cleaned}  (diff={diff:+d} → cleaned)", YELLOW)
        else:
            len_info = colorize(f"raw={len_raw}  cleaned={len_cleaned}  (unchanged)", GREEN)
        print(f"  {BOLD}Response:{RESET} {len_info}")
        print()
        if was_changed:
            print(colorize("  ── RAW ──────────────────────────────────────────────────────────────────────────", GRAY))
        for line in content.splitlines():
            wrapped = textwrap.fill(line, width=100, subsequent_indent='    ')
            print(f"  {wrapped}")
        if was_changed:
            print()
            print(colorize("  ── CLEANED ──────────────────────────────────────────────────────────────────────", YELLOW))
            for line in cleaned.splitlines():
                wrapped = textwrap.fill(line, width=100, subsequent_indent='    ')
                print(f"  {wrapped}")

    print_separator()

    # Legend
    print(f"  {BOLD}Label:{RESET} "
          f"{colorize('[i]nvalid', RED)}  "
          f"{colorize('[c]leaned', YELLOW)}  "
          f"{colorize('[u]nchanged', GREEN)}  "
          f"{colorize('[r]efused', MAGENTA)}  "
          f"{colorize('[f]ixed_dict', CYAN)}  "
          f"{colorize('[e]mpty', GRAY)}  "
          f"{GRAY}[s]kip  [q]uit{RESET}")
    print_separator('═')


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(df_labeled: pd.DataFrame):
    """Compute and print accuracy / per-class precision, recall, F1."""
    df = df_labeled[df_labeled['manual_label'] != 'skip'].copy()

    if df.empty:
        print(colorize("No labeled samples (excluding skips). Cannot compute metrics.", RED))
        return

    y_true = df['manual_label']
    y_pred = df['valid_flag']
    labels = sorted(set(y_true) | set(y_pred))

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Binary valid/invalid accuracy — 'valid','cleaned','unchanged','fixed_dict' → valid;
    # 'invalid','empty','refused' → invalid; 'v' (generic valid label) maps to valid side.
    df_binary = df[df['manual_label'].isin(VALID_GROUP | INVALID_GROUP)].copy()
    if not df_binary.empty:
        to_binary = lambda s: 'valid' if s in VALID_GROUP else 'invalid'
        y_true_b = df_binary['manual_label'].map(to_binary)
        y_pred_b = df_binary['valid_flag'].map(to_binary)
        acc_b = accuracy_score(y_true_b, y_pred_b)
    else:
        acc_b = None

    print_separator('═')
    print(colorize("  EVALUATION RESULTS", BOLD + CYAN))
    print_separator('═')
    print(f"  Samples labeled (excl. skip): {len(df)}")
    print(f"  Overall Accuracy (exact label): {colorize(f'{acc:.4f}', BOLD + GREEN)}")
    if acc_b is not None:
        print(f"  Binary Accuracy  (valid/invalid): {colorize(f'{acc_b:.4f}', BOLD + BLUE)}  "
              f"{GRAY}(n={len(df_binary)}){RESET}")
    print()
    print("  Per-class metrics (manual = truth, algo = prediction):")
    print_separator()
    for line in report.splitlines():
        print("  " + line)
    print_separator()
    print("  Confusion matrix (rows=manual, cols=algo):")
    print(f"  Labels: {labels}")
    for i, row in enumerate(cm):
        print(f"  {labels[i]:12s} | {row}")
    print_separator('═')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Manually annotate LLM response labels')
    parser.add_argument('--results_dir', required=True, help='Root results directory (contains responses/)')
    parser.add_argument('--summary_csv', required=True, help='Path to summary.csv')
    parser.add_argument('--output', required=True, help='CSV to save/resume manual labels')
    parser.add_argument('--n', type=int, default=100, help='Number of samples to label (default: 100)')
    parser.add_argument('--stratified', action='store_true', help='Stratify sample by valid_flag')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', default=None, help='Filter by model name')
    parser.add_argument('--language', default=None, help='Filter by language')
    parser.add_argument('--export_sample', default=None, help='Export the sample to a CSV and exit (for sharing)')
    parser.add_argument('--sample_csv', default=None, help='Use a pre-defined sample CSV instead of sampling')
    parser.add_argument('--show_algo_label', action='store_true',
                        help='Show the algorithmic label during annotation (may introduce bias)')
    args = parser.parse_args()

    # Load summary
    print(f"Loading summary CSV: {args.summary_csv}")
    df = pd.read_csv(args.summary_csv, low_memory=False)

    if args.model:
        df = df[df['model'] == args.model]
    if args.language:
        df = df[df['language'] == args.language]

    if df.empty:
        print(colorize("No rows match filters.", RED))
        sys.exit(1)

    print(f"Total rows after filters: {len(df):,}")
    print(f"Label distribution:\n{df['valid_flag'].value_counts().to_string()}\n")

    # Load existing labels (for resumability)
    output_path = Path(args.output)
    if output_path.exists():
        df_done = pd.read_csv(output_path)
        if 'manual_label' in df_done.columns:
            df_done = df_done[df_done['manual_label'].notna() & (df_done['manual_label'] != '')]
        else:
            df_done = pd.DataFrame()
        if len(df_done) > 0:
            print(colorize(f"Resuming: {len(df_done)} samples already labeled.", YELLOW))
        already_done_idx = set(df_done['original_index'].tolist()) if not df_done.empty else set()
    else:
        df_done = pd.DataFrame()
        already_done_idx = set()

    # Sample — either from pre-defined CSV or by sampling summary
    if args.sample_csv:
        print(colorize(f"Using pre-defined sample: {args.sample_csv}", CYAN))
        sample = pd.read_csv(args.sample_csv)
        sample = sample[~sample['index'].isin(already_done_idx)].reset_index(drop=True)
    else:
        sample = sample_rows(df, args.n, args.stratified, args.seed, already_done_idx)

    # Export sample and exit if requested
    if args.export_sample:
        export_path = Path(args.export_sample)
        sample.to_csv(export_path, index=False)
        print(colorize(f"Sample exported to {export_path} ({len(sample)} rows). Share this file.", GREEN))
        return

    remaining = len(sample)

    if remaining == 0:
        print(colorize("All sampled rows already labeled. Showing metrics.", GREEN))
        compute_metrics(df_done)
        return

    print(f"Sampling {remaining} rows to label. Press Enter to begin...")
    input()

    new_labels = []

    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        # Load raw content
        content = load_raw_content(args.results_dir, row)

        display_sample(i + len(already_done_idx), len(already_done_idx) + remaining, row, content,
                       show_algo_label=args.show_algo_label)

        # Get label
        while True:
            try:
                key = input("  Your label: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                key = 'q'

            if key in VALID_LABELS:
                break
            print(colorize(f"  Invalid key '{key}'. Use: {list(VALID_LABELS.keys())}", RED))

        if key == 'q':
            print(colorize("\nQuitting early...", YELLOW))
            break

        label = VALID_LABELS[key]
        new_labels.append({
            'original_index': row['index'],
            'model': row['model'],
            'language': row['language'],
            'role': row['role'],
            'task': row['task'],
            'location': row['location'],
            'k': row['k'],
            'target': row['target'],
            'field': row['field'],
            'subfield': row['subfield'],
            'run_id': row['run_id'],
            'valid_flag': row['valid_flag'],
            'manual_label': label,
        })

        # Save incrementally
        df_new = pd.DataFrame(new_labels)
        df_all = pd.concat([df_done, df_new], ignore_index=True) if not df_done.empty else df_new
        df_all.to_csv(output_path, index=False)

    # Final metrics
    df_new = pd.DataFrame(new_labels)
    df_all = pd.concat([df_done, df_new], ignore_index=True) if not df_done.empty else df_new

    if not df_all.empty:
        df_all.to_csv(output_path, index=False)
        print(colorize(f"\nLabels saved to {output_path}", GREEN))
        compute_metrics(df_all)
    else:
        print(colorize("No labels saved.", RED))


if __name__ == '__main__':
    main()
