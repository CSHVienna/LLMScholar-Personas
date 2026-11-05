# Batch Processing Scripts

This folder contains the prompt-generation scripts and the audit pipeline scripts for generating and managing prompts in multiple languages: `batch_params.py` and `batch_prompt.py`.

## Overview

These scripts work together to create localized prompt combinations:

1. **`batch_params.py`** - Must be run first. Translates parameter files (instructions, locations, inputs) into the target language.
2. **`batch_prompt.py`** - Run after params. Generates prompt combinations from the translated parameters.
3. **Audit pipeline** - Run after collecting data from LLMs (see *Audit pipeline* below).

## Prerequisites
`
Before running either script, set the `PYTHONPATH`. If you are inside `code/scripts` then:
```bash
export PYTHONPATH="$PYTHONPATH:../libs"
```

## batch_params.py

Generates all English params, and translates them to a target language using OpenAI's API.

### Parameters

- `-l, --language` (optionsl): Target language code. Must be one of the supported languages defined in `cons.LANGUAGES`. Default: `cons.LANG_EN` (english).
- `-o, --output-dir` (required): Output directory. Default: `../data/context/`

### Configuration

Requires a `config.ini` file with OpenAI API credentials at `../../config.ini`.

### Example Usage
```bash
# Translate parameters to German
python batch_params.py -l german -o ../data/context/

# Translate parameters to Spanish
python batch_params.py -l spanish -o ../data/context/
```

## batch_prompt.py

Generates prompt combinations from the translated parameter files.

### Parameters

- `-c, --combination_id` (optional): The combination ID to display (0-indexed).
- `-l, --language` (optional): Target language code (must match the language used in `batch_params.py`).
- `-o, --output-dir` (required): Output directory. Default: `../data/context/`

### Example Usage
```bash
# See total combinations available
python batch_prompt.py -c 0 -l german -o ../data/context/

# Display combination ID 5
python batch_prompt.py -c 5 -l german

# Display combination ID 42 in Spanish
python batch_prompt.py -c 42 -l spanish
```


## Audit pipeline

All audit scripts read from `data/` and write to `results/`. Default paths are resolved from the project root,
so scripts run **from any folder** (project root, `code/scripts`, ...) and need no `PYTHONPATH`
(`_bootstrap.py` adds `code/libs`). Every path can be overridden from the command line.
Use the `py311_llmscholar` conda environment.

### Running everything — `run_responses_pipeline.sh`

Runs steps B.1–B.3 (summary → recommendations) with GNU parallel, one process per model.

```bash
conda activate py311_llmscholar
bash code/scripts/run_responses_pipeline.sh                        # all steps, all models, 10 jobs
bash code/scripts/run_responses_pipeline.sh -j 20 -s summary       # one step: summary | recommendations
bash code/scripts/run_responses_pipeline.sh -x "--overwrite"       # recompute existing files
bash code/scripts/run_responses_pipeline.sh -r /tmp/smoke -l /tmp/smoke/logs -x "--limit_prompts 5"   # smoke test
```

Options: `-j` jobs (10), `-m` models file (`data/models/models.txt`), `-s` step (`all`), `-p` python executable,
`-r` results root (`results/`), `-l` logs folder (`logs/`), `-x` extra arguments for the per-model scripts.
If a step has failed models it stops and prints the `--resume-failed` command that re-runs only those.

### Logging

- Every script run: `logs/<timestamp>_<script>_<params>.log`
  (e.g. `2026-11-04-23:10_responses_summary_gpt-4_1-nano-2025-04-14.log`; `params` = model, languages if a subset,
  `limit<N>` for smoke tests). Starts with the full command and all arguments; failures are logged with tracebacks.
- Every GNU parallel step of the runner: `logs/<timestamp>_<step>_<models file>.joblog`
  (e.g. `2026-11-04-23:10_responses_summary_models.joblog`; column `Exitval` must be 0 for every model).
- Running GNU parallel by hand: it writes a joblog only if you pass `--joblog <file>`.

### Running a single script

```bash
python code/scripts/batch_responses_summary.py --models gpt-4_1-nano-2025-04-14     # from the project root
cd code/scripts && python batch_responses_summary.py --models gpt-4_1-nano-2025-04-14  # or from code/scripts
```

### Tags (multi-label)

Responses and recommendations are tagged with **every** tag that applies. Each output file has a joined column
(`response_tags` / `rec_tags`, e.g. `title_removed|fixed`) and one boolean column per tag
(`response_<tag>` / `rec_<tag>`). "Valid" is defined later as a combination of tags, after reading the report.

### Step B.1-2 — `batch_responses_summary.py`

Reads every raw response, decodes it, and tags it (rules in `libs/responses/taxonomy.py`).
A response with >= 1 named recommendation gets every **issue** tag that applies, or `correct` (alone) if none
applies; a response without names gets exactly one **failure** tag. Recommendations are extracted from every
response with names, whatever its tags (e.g. from the list inside a `wrapped` object).

| Group | Tag | Meaning |
|---|---|---|
| — | `correct` | none of the other tags apply: a strict JSON array of complete recommendations, no disclaimer (always alone) |
| issue | `python_literal` | python syntax instead of JSON (single quotes, `True`/`None`, trailing commas) |
| issue | `embedded` | JSON surrounded by prose (`"Here you go: [...]"`) |
| issue | `fixed` | truncated (e.g. token budget hit mid-record): parsed up to the last complete record, trailing partial entry discarded |
| issue | `wrapped` | JSON root is an object (`{"candidates": [...]}`), not the requested array |
| issue | `extra_incomplete` | some recommendation lacks a requested non-name field (current_affiliations, areas_of_research_or_work, reason, source) |
| issue | `toy` | names disclaimed as illustrative / fictional (`toy_keyword` records the match) |
| failure | `empty` | no content (empty string or whitespace only) |
| failure | `empty_list` | an object wrapping only empty lists, e.g. `{"candidates": []}` |
| failure | `refused` | declines to recommend (natural-language justification, no names) |
| failure | `invalid` | API error, unparsable / broken JSON, `[]`, no person objects, or only template names |

JSON broken anywhere other than at the end (e.g. a missing comma) is not repaired: it is `invalid`.
Writes `results/responses_metadata/<model>.parquet` (one row per response, with tags,
`parse_method`, `json_root`, counts of persons / named / complete persons, tokens, finish reason, errors).

| Argument | Default | Description |
|---|---|---|
| `--responses_dir` | `data/responses` | Folder with `results_<source>_<language>/` |
| `--context_dir` | `data/context` | Prompt parameters per language (maps ES/DE values back to English) |
| `--collection_dates_file` | `data/models/collection_dates.json` | Dates for sources without timestamps (Gemini) |
| `--models_file` | `data/models/models.txt` | One model per line |
| `--models` | all | Models to process (one with GNU parallel: `--models {}`) |
| `--languages` | all | Subset of languages |
| `--limit_prompts` | none | Only the first N prompts per file (smoke tests) |
| `--overwrite` | off | Recompute existing files |
| `--output_dir` | `results/responses_metadata` | Output folder (`<model>.parquet`) |
| `--log_dir` | `logs` | Log folder |

### Step B.3 — `batch_recommendations.py`

Tags every recommended person (rules in `libs/responses/recommendations.py`):

| Tag | Meaning |
|---|---|
| `correct` | none of the other tags apply (always alone) |
| `title_removed` | honorifics/degrees removed (Dr., Prof., PhD) |
| `fixed` | both parts given, one repeated inside the other (`"John Smith"` / `"Smith"`): repetition removed |
| `split` | only one part given, with > 1 word: split into first + last name |
| `initials_name` | first name is initials only (`"M."` / `"Smith"`) |
| `initials_lastname` | last name is initials only (`"Michael"` / `"G."`) |
| `initials` | both first and last name are initials only (`"S. M."` / `"D."`) |
| `placeholder` | template instead of a name (`"Candidate 1"`, `"Last Name"`); always alone |
| `invalid` | no usable name (missing, or a single word only) |

Stores the recommendations of every response with named recommendations (whatever its tags). Per recommendation: `missing_fields` / `is_complete` (non-name
requested fields), `name_fix` (how a name was split / fixed), `name_key` (name tokens without case/accents,
any order; used later by the duplicates metric — duplicates are not a tag). Only the names are used by the audit, but nothing the
LLM returned is lost: the requested metadata is in `affiliations`, `areas`, `reason`, `source_url`, additional keys
are listed in `extra_fields`, and the full original object of every person is kept in `record_json`.
Writes `results/recommendations_metadata/<model>.parquet`.
Same arguments as above, plus `--responses_metadata_dir`.

### Inspecting the tags — `code/notebooks/responses_tags.ipynb`

Plots the distribution of tag combinations (no batch report): responses and recommendations per model, and for
all models (mean of the model shares, every model weighs equally, with a 95% hierarchical bootstrap CI: models,
then configurations within models; `tag_stats.model_mean_shares`, constants `DEFAULT_BOOTSTRAP_SAMPLES`,
`DEFAULT_CONFIDENCE`, `DEFAULT_SEED`). Valid combinations
(all tags in the valid set; defaults `DEFAULT_VALID_RESPONSE_TAGS` / `DEFAULT_VALID_REC_TAGS` in `constants.py`,
overridable in the notebook) are green, the others red and hatched. Plots: `results/plots/tags_*.{png,pdf}`.
Paper figures (`tags_responses_paper`, `tags_recommendations_paper`): per-model panel (4/5) + overall panel (1/5),
both vertical with a shared 0–100 axis, A4 width × 2.9 in; models sorted by valid share (desc.); short model labels
(`viz/labels.py`, rules in `constants.MODEL_LABEL_RULES`, checked to be unique) and short tag labels
(`constants.TAG_SHORT_LABELS`, e.g. `fixed+wrap+extra_inc`; define them in the caption); $N_m$ and $N_r$ annotated
inside the overall panel (no per-model title). Each paper cell prints the figure caption (`viz/captions.py`). Rule for all plots: titles only in PNG, never in PDF (PNG titles
include $N_m$ / $N_r$). Style (`viz/style.apply_style()`, called by the notebook): seaborn `ticks` + `paper`
context × 0.75 (figures drawn at print size: ticks ~6.6 pt, labels ~7 pt), Helvetica-like TrueType font
(Liberation Sans here), fonts embedded as TrueType (Type 42) in PDFs for submission checkers.

### Order

`batch_responses_summary.py` → `batch_recommendations.py` (needs the summary); `run_responses_pipeline.sh` runs
them in this order. Then open `code/notebooks/responses_tags.ipynb`.

### Manual validation of the tags — `batch_annotation_sample.py`, `code/tools/annotation_tool.html`, `batch_annotation_evaluate.py`

Blind check of the inferred tags (the annotator never sees them):

```bash
python code/scripts/batch_annotation_sample.py --task responses          # or --task recommendations
python code/scripts/serve_annotation_tool.py                             # on the server; Ctrl+C to stop
# open http://localhost:8765 in your local browser (VS Code forwards the port; or: ssh -L 8765:localhost:8765 <server>)
# choose the task, annotate: every item is saved to results/annotations/annotation_<task>_<annotator>.csv
python code/scripts/batch_annotation_evaluate.py --task responses
```

Offline alternative: open `code/tools/annotation_tool.html` directly, load the items file, and put the exported
CSV into `results/annotations/`. The server binds to 127.0.0.1 only and never serves the key file.

- Sample: up to `--per_stratum` (20) items per inferred tag combination (rare ones all checked), at most
  `--max_items` (400), seed 42. Items file = codebook + content only; key file = inferred tags (evaluation only).
  An existing sample is never overwritten (annotations refer to its item ids) unless `--overwrite`.
- Tool: one item per screen, tick all tags that apply (keys 1–9, 0, -, =), "unsure", notes, for recommendations
  the first / last name as you would record them; order shuffled per annotator; autosaved in the browser.
  Responses: the prompt is collapsed (requested k in its header) and a **Checks** panel shows facts computed in the
  browser with `JSON.parse`, independently of the pipeline (never its labels or tag names): whether the raw / cleaned
  text parses, the error location (end of text = truncation, or middle), a JSON block inside surrounding text,
  the root (array / object), the number of person records vs k, and the missing requested fields per record.
  Report it in the paper as "annotators were assisted by a JSON syntax check".
  Responses also ask for the **number of recommended people** (compared with the pipeline's count of person records,
  i.e. `k_returned` of `k_ratio`). To add new codebook fields to an existing sample without changing its items:
  `python code/scripts/batch_annotation_sample.py --task responses --refresh_codebook`.
- Evaluation (`results/annotations/annotation_<task>_eval_<table>.csv`): `summary` (exact match of tag sets,
  accuracy of the valid / not valid decision, name match), `tags` (precision / recall / F1 per tag), `agreement`
  (Krippendorff's alpha per tag, >= 2 annotators), `disagreements` (items to inspect). Gold = majority of
  non-unsure annotators; tied items are excluded and counted.

### Tests

```bash
# from the project root
python -m unittest discover -s code/tests -v
```

The previous parser (`batch_parse_results.py`) is kept in `_old/` for reference only.

## Complete Workflow Example

Here's a complete example for German language with combination ID 0:
```bash
# Step 1: Set PYTHONPATH
export PYTHONPATH="$PYTHONPATH:../libs"

# Step 2: Translate parameters to German
python batch_params.py -l de -o ../data/context/

# Step 3: Generate and view prompt combination 0
python batch_prompt.py -c 0 -l de -o ../data/context/
```

## Parallel Processing with GNU Parallel

To process multiple combinations in parallel using GNU Parallel:
```bash
# Process combinations 0-719 in parallel (8 jobs at a time)
parallel -j 8 python batch_prompt.py -c {} -l germam -o ../data/context/ ::: {0..719}

# Process all combinations for multiple languages
parallel -j 4 python batch_params.py -l {} -o ../data/context/ ::: english german spanish
parallel -j 8 python batch_prompt.py -c {1} -l {2} ::: {0..719} -o ../data/context/ ::: english german spanish

# Save output to separate files
parallel -j 8 "python batch_prompt.py -c {} -l german -o ../data/context/ > output_{}.txt" ::: {0..719}

# Process with progress bar
parallel --bar -j 8 python batch_prompt.py -c {} -l german -o ../data/context/ ::: {0..719}
```

## Output

- `batch_params.py` creates translated JSON files in `{output_dir}/{language}/`:
  - `instructions.json`
  - `locations.json`
  - `input.json`

- `batch_prompt.py` creates:
  - `all_prompts.txt` - Contains all generated prompt combinations
  - Console output showing the selected combination's instructions, input, and full prompt

## Notes

- Always run `batch_params.py` before `batch_prompt.py` for each language
- The combination ID in `batch_prompt.py` must be within the valid range (check output for total combinations)
- GNU Parallel is not required but highly recommended for processing multiple combinations efficiently