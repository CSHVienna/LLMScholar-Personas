# Scripts

All scripts run from the `code/` directory with `PYTHONPATH=.` so that `libs.*` imports resolve. Relative paths in defaults point at `../data/` and `../results/` (one level up from `code/`).

```
scripts/
├── 1_ground_truth/      Every Semantic Scholar researcher + their OpenAlex author and metadata → ground_truth.parquet
├── 2_graph/             Author stats, coauthorship graph and similarity PCA over the whole ground truth
├── 3_prompting/         Build the prompts and parse the LLM answers → recommendations.csv
├── 4_llmcaller/         Send the prompts to Ollama / OpenAI / Gemini
├── 5_validity/          Validity, refusal, duplicates of every call; consistency of the valid ones
├── 6_factuality/        Factuality cascade on Semantic Scholar (valid calls): name → field → seniority → author
├── 7_enrichment/        The LLM's answer vs the picked researcher's data: country, affiliation; ethnicity → factuality_full.csv
├── 8_metrics/           Factuality, socials (factuality_author = True), calls.csv
├── 9_plot_data/         Data for specific figures (Sankey flows, countries, field baseline, ethnicity)
├── run_all.py           Runs 1, 2 and 5 → 9
├── annotation/          Interactive CLIs for manual labelling (optional)
├── ethnicity/           Ethnicity inference (optional)
└── legacy/              Scripts not used any more (replaced or one-time experiments)
```

The order to run them is in [`code/README.md`](../README.md). For the details of a script, read its docstring or run it with `--help`.

---

## Prerequisites

```bash
cd code
export PYTHONPATH=.
```

That single `PYTHONPATH=.` makes `libs.*` importable from every script under `scripts/`.

---

## Ground truth (`1_ground_truth/`)

Once, for every researcher of the Semantic Scholar parquet: their papers with DOI (API) → their OpenAlex author, the one who signs those papers, chosen by Jaro-Winkler on the names (threshold 1.35, `notebooks/analysis/checks/jw_threshold.ipynb`) → for the researchers without one, the cascade by name, field and seniority → the metadata of that author joined to the parquet: `1_ground_truth/ground_truth.parquet`. See `1_ground_truth/README.md`.

## Graph (`2_graph/`)

`run_pipeline.py` runs `1_author_stats.py` (h / i10 / e-index of every OpenAlex author of the ground truth → `2_graph/author_stats/`) and `2_model.py` (coauthorship graph and similarity PCA over all of them, cached in `../results/2_graph/cache/*_gt.joblib`; `libs/metrics/structural_model.py`).

## Batch processing (`3_prompting/`)

Three scripts work together to build and parse localised prompt combinations:

1. **`batch_params.py`** — first. Translates parameter files (instructions, locations, inputs) into the target language.
2. **`batch_prompt.py`** — runs after `batch_params`. Builds all prompt combinations from the translated parameters.
3. **`batch_parse_results.py`** — runs after responses have been collected from the LLMs. Consolidates everything into `recommendations.csv` and `summary.csv`.

### batch_params.py

Generates the canonical English parameters and translates them into the target language with the OpenAI API.

**Arguments**
- `-l, --language` (optional): language code; must match one of `cons.LANGUAGES`. Default: `cons.LANG_EN`.
- `-o, --output-dir` (required): output directory. Default: `../data/context/`.

**Configuration**

Requires a `config.ini` (at the repo root) with the OpenAI credentials. See `config.ini.example`.

**Examples**

```bash
# From code/
python scripts/3_prompting/batch_params.py -l german  -o ../data/context/
python scripts/3_prompting/batch_params.py -l spanish -o ../data/context/
```

### batch_prompt.py

Builds prompt combinations from the translated parameters.

**Arguments**
- `-c, --combination_id` (optional): combination ID to display (0-indexed).
- `-l, --language` (optional): language (must match what `batch_params.py` produced).
- `-o, --output-dir` (required): output directory. Default: `../data/context/`.

**Examples**

```bash
# From code/
python scripts/3_prompting/batch_prompt.py -c 0  -l german  -o ../data/context/
python scripts/3_prompting/batch_prompt.py -c 5  -l german
python scripts/3_prompting/batch_prompt.py -c 42 -l spanish
```

### batch_parse_results.py

**Arguments**
- `--results_dir` (required): directory with the LLM responses (`.json`), e.g. `../results`.
- `--output_dir` (required): output directory.
- `--model` (optional): model name (see `data/context/models.txt`).
- `--language` (optional): language (`spanish`, `english`, `german`).

**Example**

```bash
# From code/, parse every combination in parallel
nice -n 10 parallel -j 20 python scripts/3_prompting/batch_parse_results.py --results_dir ../results --output_dir ../results/3_recommendations --model {1} --language {2} :::: ../data/context/models.txt ::: english german spanish
```

## LLMs Caller (`4_llmcaller/`)

### Requirements
* Data must be available at `data/context/...`.
* Add API keys for gpt an gemini in `.env`.
* Install packages (see requirements.txt).

### Execution
* Run ollama: `scripts/4_llmcaller/ollama_requests_multiprocessing.py` (see args for details)
* Run gpt/gemini:
  1. `scripts/4_llmcaller/gpt_gemini_01_create_batch_files.py` (modify language using args; uncomment models ony by one)
  2. `scripts/4_llmcaller/gpt_gemini_02_create_batches.py` (modify language and model family using args)
  3. `scripts/4_llmcaller/gpt_gemini_03_retrieve_results.py` (modify model family using args)


### End-to-end workflow

```bash
# From code/
export PYTHONPATH=.

# 1. Translate parameters into German
python scripts/3_prompting/batch_params.py -l german -o ../data/context/

# 2. Build and inspect combination 0
python scripts/3_prompting/batch_prompt.py -c 0 -l german -o ../data/context/

# 3. Data collection (querying LLMs via Ollama, OpenAI, and Vertex APIs)
python scripts/4_llmcaller/ollama_requests_multiprocessing.py -m <model>  -r <repetitions> -l <language>
python scripts/4_llmcaller/gpt_gemini_01_create_batch_files.py -m <model> -r <repetitions> -l <language>
python scripts/4_llmcaller/gpt_gemini_02_create_batches.py -mf <model_familiy> -r <repetitions> -l <language>
python scripts/4_llmcaller/gpt_gemini_03_retrieve_results.py -mf <model_familiy>

```

### GNU Parallel

```bash
# Process combinations 0–719 in parallel (8 simultaneous jobs)
parallel -j 8 python scripts/3_prompting/batch_prompt.py -c {} -l german -o ../data/context/ ::: {0..719}

# Multiple languages
parallel -j 4 python scripts/3_prompting/batch_params.py -l {} -o ../data/context/ ::: english german spanish

# With a progress bar
parallel --bar -j 8 python scripts/3_prompting/batch_prompt.py -c {} -l german -o ../data/context/ ::: {0..719}
```

### Outputs

- `batch_params.py` writes translated JSONs under `{output_dir}/{language}/`:
  - `instructions.json`, `locations.json`, `input.json`.
- `batch_prompt.py` writes:
  - `all_prompts.txt` — every generated combination.
  - The selected combination is printed to stdout.

### Notes

- Always run `batch_params.py` before `batch_prompt.py` for each language.
- `combination_id` must be within the valid range (check the total count printed by the script).
- GNU Parallel is not required but strongly recommended for the parsing step.

---

## Validity and consistency (`5_validity/`)

| Script | One row per | Writes |
|---|---|---|
| `1_validity.py` | call | validity, refusal, duplicates (names normalized: `libs/pipeline_common.author_key`) → `../results/5_validity/validity/`; the recommendations of the valid calls, each author once per call (repeated names left out after measuring duplicates) and only complete names (`has_full_name`: a single first or last name is left out) → `../results/5_validity/recommendations_valid.csv`, the input of `6_factuality/`. Both filters are applied only here |
| `2_consistency.py` | setup (model + params) | consistency of the valid calls → `../results/5_validity/consistency/` |

## Factuality cascade (`6_factuality/`)

`run_pipeline.py` reads the recommendations of valid calls, each author once per call and only complete names (`../results/5_validity/recommendations_valid.csv`, from `5_validity/1_validity.py`; the names are searched with the same normalization as duplicates), runs `1_name.py` (all researchers with that name; the slow step) → `2_field.py` → `3_seniority.py` → `4_author.py` (counts and the pick: uniform among the candidates that pass, reproducible per name), and then itself adds the status columns and the ground truth of the picked researcher (`6_factuality/factuality_ss.csv`) with the ground-truth data of the picked researcher (OpenAlex included, from `1_ground_truth/`), and keeps the rows with `factuality_author = True` (`6_factuality/factual_authors.csv`). `names.py` holds how names are compared. Output under `../results/6_factuality/`.

## Enrichment (`7_enrichment/`)

`run_enrichment.py` runs `1_openalex.py` (the OpenAlex columns of the picked researcher, from the ground truth; only factuality_author = True, the rest get `oa_status = not_searched`) → `2_location.py` (prompt country vs OpenAlex country) → `3_affiliation.py` (LLM affiliations vs OpenAlex institutions) → `4_ethnicity.py` (perceived ethnicity of the picked researcher, from the ground-truth table `ethnicity/researcher_ethnicity_lookup.csv`), each reading the previous one's output, and writes `7_enrichment/factuality_full.csv`.

## Metrics (`8_metrics/`)

`run_pipeline.py` runs the steps in order (`--from_step N` to resume). Each writes one CSV per model under `../results/8_metrics/<step>/` (see `libs/pipeline_common.py`).

| Script | One row per | Writes |
|---|---|---|
| `1_factuality.py` | recommendation | factuality_*, matched_* (Semantic Scholar), bias_location (only factuality_author = True) → `factuality/` |
| `2_socials.py` | call | diversity, parity, popularity, bias_location, similarity, connectedness, over the authors with factuality_author = True (graph and PCA of `2_graph/`; classes, parity references and tercile cuts from the ground truth) → `socials/` |
| `3_calls_table.py` | call | validity, consistency, steps 1 and 2 in one table, with the column names the notebooks use → `calls.csv` |

## Data for specific figures (`9_plot_data/`)

They read the output of the funnel (`7_enrichment/factuality_full.csv`), only the recommendations with `factuality_author = True`, and write one table per figure to `9_figures/tables/`. `build_gt_field_baseline.py` is the exception: it describes the Semantic Scholar population, not the recommendations.

| Script | Writes | Read by |
|---|---|---|
| `build_sankey_flows.py` | `9_figures/tables/{seniority,ethnicity,field,subfield,language,gender}_flows.csv` | `analysis/figures/metrics.ipynb` |
| `build_location_flows.py` | `9_figures/tables/location_flows.csv` | `analysis/figures/metrics.ipynb` |
| `build_gt_field_baseline.py` | `9_figures/tables/gt_field_baseline.csv` | `analysis/figures/metrics.ipynb` |
| `build_ethnicity_distributions.py` | `9_figures/ethnicity_distributions/*.csv` | `analysis/figures/ethnicity.ipynb` |

## Legacy (`legacy/`)

Not part of the pipeline any more; kept for reference. See `legacy/README.md`.
