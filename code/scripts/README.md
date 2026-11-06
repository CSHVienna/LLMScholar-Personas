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
bash code/scripts/run_responses_pipeline.sh -r tmp/smoke -l tmp/smoke/logs -x "--limit_prompts 5"   # smoke test
```

Options: `-j` jobs (10), `-m` models file (`data/models/models.txt`), `-s` step (`all`), `-p` python executable,
`-r` results root (`results/`), `-l` logs folder (`logs/`), `-x` extra arguments for the per-model scripts.
If a step has failed models it stops and prints the `--resume-failed` command that re-runs only those.


### Checking the latest runs — `check_eval_runs.sh`

Read-only. For every model it takes the newest log of `batch_eval_responses.py` and of
`batch_eval_recommendations.py` (only files directly in `logs/`; sub-folders such as `logs/2025/` are
ignored) and reports models whose step FAILED, is UNFINISHED or MISSING, whose recommendations log is older
than its responses log, or whose stored tags in `results/*_metadata/<model>.parquet` are not current tag names
(e.g. written before a rename). Prints the commands that re-run only those models; exit code 1 if anything
is wrong.

```bash
bash code/scripts/check_eval_runs.sh          # problems only
bash code/scripts/check_eval_runs.sh -v       # one line per model
```

### Logging

- Every script run: `logs/<timestamp>_<script>_<params>.log`
  (e.g. `2026-11-04-23:10_responses_summary_gpt-4_1-nano-2025-04-14.log`; `params` = model, languages if a subset,
  `limit<N>` for smoke tests). Starts with the full command and all arguments; failures are logged with tracebacks.
- Every GNU parallel step of the runner: `logs/<timestamp>_<step>_<models file>.joblog`
  (e.g. `2026-11-04-23:10_responses_summary_models.joblog`; column `Exitval` must be 0 for every model).
- Running GNU parallel by hand: it writes a joblog only if you pass `--joblog <file>`.

### Running a single script

```bash
python code/scripts/batch_eval_responses.py --models gpt-4_1-nano-2025-04-14     # from the project root
cd code/scripts && python batch_eval_responses.py --models gpt-4_1-nano-2025-04-14  # or from code/scripts
```

### Tags (multi-label)

Responses and recommendations are tagged with **every** tag that applies. Each output file has a joined column
(`response_tags` / `rec_tags`, e.g. `title_removed|repetition_removed`) and one boolean column per tag
(`response_<tag>` / `rec_<tag>`). "Valid" is defined later as a combination of tags, after reading the report.

### Step B.1-2 — `batch_eval_responses.py`

Reads every raw response, decodes it, and tags it (rules in `libs/responses/taxonomy.py`).
A response with >= 1 named recommendation gets every **issue** tag that applies, or `correct` (alone) if none
applies; a response without names gets exactly one **failure** tag. Recommendations are extracted from every
response with names, whatever its tags (e.g. from the list inside a `wrapped` object).

| Group | Tag | Meaning |
|---|---|---|
| — | `correct` | none of the other tags apply: a strict JSON array of recommendations, no disclaimer (always alone; completeness is per recommendation, see B.3) |
| issue | `python_literal` | python syntax instead of JSON (single quotes, `True`/`None`, trailing commas) |
| issue | `embedded` | JSON surrounded by prose (`"Here you go: [...]"`) |
| issue | `truncated` | cut off (e.g. token budget hit mid-record): parsed up to the last complete record, trailing partial entry discarded |
| issue | `wrapped` | JSON root is an object (`{"candidates": [...]}`), not the requested array |
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

### Step B.3 — `batch_eval_recommendations.py`

Tags every recommended person (rules in `libs/responses/recommendations.py`):

| Tag | Meaning |
|---|---|
| `correct` | none of the other tags apply (always alone) |
| `title_removed` | honorifics/degrees removed (Dr., Prof., PhD) |
| `repetition_removed` | both parts given, one repeated inside the other (`"John Smith"` / `"Smith"`): repetition removed |
| `split` | only one part given, with > 1 word: split into first + last name |
| `initials_name` | first name is initials only (`"M."` / `"Smith"`) |
| `initials_lastname` | last name is initials only (`"Michael"` / `"G."`) |
| `initials` | both first and last name are initials only (`"S. M."` / `"D."`) |
| `placeholder` | template instead of a name (`"Candidate 1"`, `"Last Name"`); always alone |
| `invalid` | no usable name (missing, or a single word only) |

Stores the recommendations of every response with named recommendations (whatever its tags). Per recommendation: `missing_fields` / `is_complete` (non-name
requested keys absent from the person's object: current_affiliations, areas_of_research_or_work, reason, source; a present key counts whatever its value, also empty), `name_fix` (how a name was split / fixed), `name_key` (name tokens without case/accents,
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
(`viz/labels.py`, rules in `viz/constants.MODEL_LABEL_RULES`, checked to be unique) and short tag labels
(`viz/constants.TAGS_SHORT_LABELS`, e.g. `py_lit+wrap+toy`; define them in the caption); $N_m$ and $N_r$ annotated
inside the overall panel (no per-model title). Each paper cell prints the figure caption (`viz/captions.py`). Rule for all plots: titles only in PNG, never in PDF (PNG titles
include $N_m$ / $N_r$). Visual constants: `viz/constants.py` (imported as `viscons`). Style (`viz/style.apply_style()`, called by the notebook): seaborn `ticks` + `paper`
context × 0.75 (figures drawn at print size: ticks ~6.6 pt, labels ~7 pt), Helvetica-like TrueType font
(Liberation Sans here), fonts embedded as TrueType (Type 42) in PDFs for submission checkers.

### Order

`batch_eval_responses.py` → `batch_eval_recommendations.py` (needs the summary); `run_responses_pipeline.sh` runs
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
- Verification tasks (`--task matching`, `--task authors`; `libs/annotation/verification.py`): the pipeline's
  output is shown and checked against OpenAlex (links are clickable; one label per group). `matching`: distinct valid
  recommended names x field x seniority, stratified by match outcome (`<rule>|linked`, `<rule>|name`, `none`);
  `authors`: GT memberships, stratified by field x seniority (field, seniority, country, merged profile). The
  evaluation writes `rates` (share of every label per stratum with Wilson CIs, and overall re-weighted to the
  stratum sizes: e.g. precision of the matches, share of missed matches) and `agreement` (per label group).

### Ground truth (step A) — `run_gt_pipeline.sh`, `batch_gt_*.py`

OpenAlex (`data/openalex_20260923.duckdb`) is opened **read-only**; outputs go to `results/groundtruth/`, scratch /
spill space to `<project>/tmp/` (git-ignored). Code: `libs/groundtruth/`, `libs/demographics/`.

```bash
# on the shared server: always inside a capped scope (only this pipeline can be killed, never the server)
systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% \
    bash code/scripts/run_gt_pipeline.sh                    # all steps, in order; re-run the same command to resume
bash code/scripts/run_gt_pipeline.sh -s bibliometrics       # one step
bash code/scripts/run_gt_pipeline.sh -h                     # all options
```

DuckDB steps run one chunk / bucket at a time (DuckDB is multi-threaded: `-d 1` job × `-t 12` threads × `-m 40GB`);
the Python-bound steps run in parallel with GNU parallel (`-j 4` jobs × `-c 3` threads). Everything is resumable:
outputs are written to a temporary name and renamed when complete, finished chunks / buckets are skipped, and
demographics continues inside a name shard from its last stored part (every 20k names).

| Step | Script | Output | What it does |
|---|---|---|---|
| A.0 | `batch_gt_authorships.py` | `authorships/chunk=<c>/bucket=<b>/` | the only scan of `works`, in chunks of 2M rows (together the whole table). Works kept: not paratext, not retracted, not xpac, year 1900–2025. One row per (author, work): year, citations, team size, primary-topic field / subfield. The lookup every later step reads |
| A.0b | `batch_gt_pairs.py` | `coauthor_pairs/chunk=<c>/bucket=<b>/` | coauthor pairs per chunk (self-join inside the chunk; works with ≤ 100 authors). Intermediate, ~1–3 s per chunk |
| A.1a | `batch_gt_edges.py` | `coauthor_edges/bucket-<bb>.parquet` | **the coauthorship edge list**: author_id, coauthor_id, n_shared_works, newman_weight (Σ 1/(team−1)), min_team, first/last year. Used for the coauthor counts (A.1) and the network (A.5) |
| A.1 | `batch_gt_bibliometrics.py` | `authors/bucket-<bb>.parquet` | per author: first/last year, # works, # works with a primary topic, # citations, h / i10 / e index, # unique coauthors (rows in the edge list), works per prompt field, OpenAlex stats, latest affiliation country, is_person |
| A.2 | `batch_gt_names.py` | `names/bucket-<bb>.parquet` | name index over ALL authors (display + alternative names, cleaned with `utils/text.py`): `name_key` (= recommendations' `name_key`), `name_key_short`, `name_norm`, `first_norm` / `last_norm` (OpenAlex has full names only: split like the recommendations), `initials_key`, is_person |
| A.3 | `batch_gt_demographics.py --stage keys / gender / ethnicity / merge` | `demographics/bucket-<bb>.parquet` (+ `keys/`, `gender/first_names.parquet`, `ethnicity/shard-<ss>.parquet`) | people with ≥ 1 field work: **probabilities only** — p_female (nomquamgender, first name), p_asian … p_other (raceBERT, full name). Both models run **once per distinct name over all buckets** (gender: one process, ~2 min; raceBERT: 40.4M authors → 31.3M names, sorted by length, in 64 shards); `-C DIR` reuses the probabilities of an earlier demographics output by name. Slowest step (CPU) |
| A.4 | `batch_gt_population.py` | `gt_authors`, `gt_members`, `gt_shares`, `gt_cutoffs` | field shares (`share_<field>`) and membership (`--min_field_share`, default 0 = any work), academic age, seniority, gender / ethnicity labels (`--gender_uncertainty` 0.1, `--ethnicity_min_prob` 0.5), L/M/H bins, top 5 %, GT shares. Cheap: re-run with other thresholds any time |
| A.5 | `batch_gt_coauthorship.py` | `coauthorship/coauthorship_<ids>.parquet` | network among given authors (e.g. the factual recommended ones): edges of the edge list with both ends in the set |

**Memory**: DuckDB takes 80 % of the RAM per process unless limited; every script has bounded defaults
(`--threads 12 --memory_limit 40GB --temp_dir <project>/tmp/duckdb`).

Smoke test without touching the full database: build a mini copy (a few thousand works and their authors) in
`<project>/tmp/smoke/` and run `run_gt_pipeline.sh -o tmp/smoke/mini_openalex.duckdb -n 4 -k 5000 -r tmp/smoke -l tmp/smoke/logs` (`-r` = results root: the ground truth goes to `tmp/smoke/groundtruth/`).

### Author matching (step B.6) — `run_matching_pipeline.sh`, `batch_match_candidates.py`, `batch_match_authors.py`

Needs B.3 and A.1 / A.2. Code: `libs/matching/` (the factuality metric is computed later from its `matched_*` columns). Two steps, like probabilities → labels in A.3 / A.4:

| Step | Script | Output | What it does |
|---|---|---|---|
| B.6a | `batch_match_candidates.py --stage rec_names / exact / fuzzy_recs / oa_tokens / fuzzy / merge` | `matching/candidates/bucket-<bb>.parquet` | OpenAlex people (all of OpenAlex) per distinct recommended name, with every rule: `exact_display`, `exact_alternative`, `short`, `initials`, and Jaro–Winkler `jw_full` / `jw_first` / `jw_last` (as written or flipped; OpenAlex names containing the recommended surname token; stored if ≥ `-f` 0.90; by default only for names without a stricter match). Heavy, once |
| B.6b | `batch_match_authors.py` | `matching/authors/main/<model>.parquet` (variants: `matching/authors/<variant>/`) | per recommendation: strictest rule with a match (`match_rule`), candidates of every rule, `n_name` / `n_name_field` / `n_name_seniority` / `n_author`, `matched_*`, `author_id` (seeded random among the n_author people), else `matched_author_field` / `matched_author_seniority` with their id when only one of the two sets is non-empty. Thresholds (fuzzy, provisional: both parts `--jw_first` / `--jw_last` ≥ 0.90, `--jw_full` at the floor; `--min_field_share`, `--seniority_definition three/single`, `--junior_max` / `--senior_min` / `--single_cutoff`) recorded in `_params.json`; other thresholds → another variant (`run_matching_pipeline.sh -V <name> -a "<args>"`) |

### Analysis config — `configure_analysis.py`

Every analysis decision in one YAML file: valid response / recommendation tags, seniority (reference year,
junior ≤, senior ≥, single cut-off, definition for matching), minimum field share, perceived gender (cut points on
p_female + one label per interval, any number of classes), ethnicity minimum probability, prominence bins and top
group, author-matching thresholds (fuzzy, seed, candidate floor). `config/analysis.yaml` (full runs) and
`config/analysis_smoke.yaml` (smoke tests), created by the wizard; once a file exists it is loaded as it is.

Two phases. **Early** settings are needed to build the ground truth and the matching candidates (seniority, field
share, L/M/H bins, top percentile, candidate floor); **late** settings are decided after looking at the data and the
metrics (valid tags, gender / ethnicity thresholds, social settings, Jaro-Winkler thresholds). Until reviewed, the
late settings hold provisional defaults (marked in the YAML): the pipeline runs with them, but the research questions
(`run_metrics_pipeline.sh -s rq1,rq2`, notebook 4) refuse to run. Reviewing them lists the steps to re-run. The
Jaro-Winkler thresholds can never go below the early `candidates_floor`.

```bash
python code/scripts/configure_analysis.py --phase early      # before the GT (config/analysis.yaml)
python code/scripts/configure_analysis.py --phase late       # after the metrics, before the RQs
python code/scripts/configure_analysis.py --config config/analysis_smoke.yaml   # every setting (Enter keeps, ? explains, b back, q quit)
python code/scripts/configure_analysis.py --show                                # print config/analysis.yaml
```

Scripts take `--config`; options given on the command line win (robustness variants). The runners
(`run_gt_pipeline.sh` population, `run_matching_pipeline.sh`, `run_metrics_pipeline.sh`) take `-y FILE`; if the
file is missing they start the wizard (in a terminal) or stop.

### Metrics (steps 7–8) — `run_metrics_pipeline.sh`, `batch_metrics.py`

Code: `libs/metrics/technical.py`. Per model: `metrics/<variant>/config/<model>.parquet` (validity, refusals,
behaviour / format / recommendation consistency) and `metrics/<variant>/response/<model>.parquet` (outcome, k ratio,
duplicates, factuality at 4 levels on the unique valid names). Variant `main` by default; robustness checks:
`-V robustness/<check> -a "--keep_duplicates"` (or `--order_sensitive_keys`, `--consistency_all_runs`).

### Social metrics (step 9) — `batch_social_metrics.py` (`run_metrics_pipeline.sh -s social`)

Code: `libs/metrics/social.py`. Per response, on the matched authors of its unique valid names: parity (1 − TV to
the GT target, `config: social.parity_target`), diversity (normalised entropy) and unknown share for gender,
ethnicity, location, productivity, influence, collaboration; popularity (top group by works / citations);
connectedness (coauthorship components); similarity (PCA cosine). `social.include_unknown` (config) decides whether
'unknown' is a class. Outputs `metrics/<variant>/social/<model>.parquet` and `social_gaps/<model>.parquet`.

### Metric figures — `code/notebooks/3_metrics.ipynb`

One function for every bar figure: `viz/metric_plots.plot_metric_bars(summary.estimate(configs, rows, metrics), rows,
metrics)` (`libs/metrics/summary.py`). Rows: `model_rows(models)`, `attribute_rows(['access', 'size', 'reasoning',
'origin'])` or `prompt_rows([...])` (persona variables in one colour, context variables in another); columns:
`select_metrics('all' | 'technical' | 'factuality' | 'social' | a subgroup such as 'consistency' | [keys])`.
Unit = configuration (response metrics averaged within it). CIs: a model row bootstraps its configurations; any other
row is the mean of its models (models weigh equally) with a hierarchical bootstrap (models, then configurations).
Bold = best within a group of rows on the displayed (rounded) values.

### Research questions — `batch_rq1.py`, `batch_rq2.py`, `code/notebooks/4_research_questions.ipynb`

RQ1 (`run_metrics_pipeline.sh -s rq1`, one process per metric): partial ω² (type-II ANOVA) and Shapley R² of the
model and of every configuration variable on each metric, cluster-bootstrap CIs over models →
`metrics/<variant>/rq1/<metric>.parquet`. Options for the RQ scripts via `-q`, e.g. `-q "--n_boot 200 --metrics factuality"`.

RQ2 (`run_metrics_pipeline.sh -s rq2`, one process per metric): per value of every configuration variable, the
estimated marginal mean and its deviation from the grand mean (weighted effect coding, same model as RQ1),
cluster-bootstrap CIs over models, plus the contrast vs a reference value (`cons.RQ2_REFERENCE_LEVELS`) and R² /
adjusted R² / ΔR² (LLM) → `metrics/<variant>/rq2/<metric>.parquet`.

RQ3 (no script; notebook only): per model, technical score = validity + (1 − refusals) + (1 − duplicates) + the
4 factualities, social score = the 6 parities (`cons.RQ3_*_METRICS`), from the configuration table; scatter with
quadrants split at the model medians (`relative`) or at half of each axis maximum (`absolute`).

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