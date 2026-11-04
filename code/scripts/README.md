# Scripts

All scripts run from the `code/` directory with `PYTHONPATH=.` so that `libs.*` imports resolve (the exception is `4_llmcaller/`, run from the repository root with `PYTHONPATH=code`). Relative paths in defaults point at `../data/` and `../results/` (one level up from `code/`).

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
├── ethnicity/           Ethnicity inference and NamSor checks (optional)
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

Once, for every researcher of the Semantic Scholar parquet: their papers with DOI (API) → their OpenAlex author, the one who signs those papers, chosen by Jaro-Winkler on the names (threshold 1.35, `notebooks/analysis/checks/jw_threshold.ipynb`) → for the researchers without one, the cascade by name, field and seniority → the metadata of that author joined to the parquet: `1_ground_truth/ground_truth.parquet`. `run_pipeline.py` runs `1_ss_papers.py` → `2_jw_match.py` → `3_candidates.py` → `4_author_fields.py` → `5_pick.py` → `6_oa_metadata.py` (`--from_step N`, `--ids`, `--out_dir`). See `1_ground_truth/README.md`.

## Graph (`2_graph/`)

`run_pipeline.py` runs `1_author_stats.py` and `2_model.py`.

- `1_author_stats.py`: per OpenAlex author of the ground truth, h-index, i10-index and 2-year mean citedness read from OpenAlex's `authors.summary_stats` (needs the 2026-09 snapshot or later), and the e-index computed from per-paper citation counts (a full pass over `oa.works`). Stages checkpointed to `../results/2_graph/author_stats/` (`authors.parquet`, `summary_stats.parquet`, `author_work_citations.parquet`, `eindex.parquet`) and skipped when present; `--rebuild <stage>` redoes one. Other flags: `--ground_truth`, `--duckdb`, `--out_dir`, `--temp_dir` (default `[data].oa_works_tmp_dir`), `--memory_limit`.
- `2_model.py`: coauthorship graph and similarity PCA over all of them (8 features, see `code/README.md`), cached in `../results/2_graph/cache/` (`coauthorship_graph_gt.joblib`, `similarity_embeddings_gt.joblib`; `libs/metrics/structural_model.py`). Flags: `--ground_truth`, `--author_stats`, `--oa_duckdb`.

## Batch processing (`3_prompting/`)

Three scripts work together to build and parse localised prompt combinations:

1. **`batch_params.py`** — first. Translates parameter files (instructions, locations, inputs) into the target language.
2. **`batch_prompt.py`** — runs after `batch_params`. Builds all prompt combinations from the translated parameters.
3. **`batch_parse_results.py`** — runs after responses have been collected from the LLMs. Consolidates everything into `recommendations.csv` and `summary.csv`.

### batch_params.py

Generates the canonical English parameters and translates them into the target language with the OpenAI API.

**Arguments**
- `-l, --language` (optional): `english`, `spanish` or `german` (`cons.LANGUAGES`). Default: `english`.
- `-o, --output-dir` (required): output directory, normally `../data/context/`.

**Configuration**

Requires a `config.ini` (found by walking up from the current directory) whose `[openai].data_dir` points to the file with the OpenAI key. See `config.ini.example`.

**Examples**

```bash
# From code/
python scripts/3_prompting/batch_params.py -l german  -o ../data/context/
python scripts/3_prompting/batch_params.py -l spanish -o ../data/context/
```

### batch_prompt.py

Builds prompt combinations from the translated parameters.

**Arguments**
- `-c, --combination_id` (optional): combination ID to display (0-indexed). Default: 0.
- `-l, --language` (optional): language (must match what `batch_params.py` produced). Default: `english`.
- `-o, --output-dir` (required): directory with the `<language>/` parameter files, normally `../data/context/`.

**Examples**

```bash
# From code/
python scripts/3_prompting/batch_prompt.py -c 0  -l german  -o ../data/context/
python scripts/3_prompting/batch_prompt.py -c 5  -l german  -o ../data/context/
python scripts/3_prompting/batch_prompt.py -c 42 -l spanish -o ../data/context/
```

### batch_parse_results.py

**Arguments**
- `--results_dir` (required): the folder whose `responses/results_<source>_<language>/` holds the LLM responses (`.json`), e.g. `../results`.
- `--output_dir` (required): output directory.
- `--model` (optional): model name (see `data/models/models.txt`).
- `--language` (optional): language (`spanish`, `english`, `german`).

Writes `summary.csv` (one row per response) and `recommendations.csv` (one row per recommended author); with both `--model` and `--language` the names get a `_<source>_<model>_<language>` suffix. `5_validity/` reads `../results/3_recommendations/recommendations.csv`.

**Example**

```bash
# From code/, parse every combination in parallel
nice -n 10 parallel -j 20 python scripts/3_prompting/batch_parse_results.py --results_dir ../results --output_dir ../results/3_recommendations --model {1} --language {2} :::: ../data/models/models.txt ::: english german spanish
```

## LLMs Caller (`4_llmcaller/`)

### Requirements
* Run from the **repository root** with `PYTHONPATH=code`: the prompts are read from `data/context/<language>/all_params_and_prompts.json` (relative to the current directory; `batch_prompt.py` writes `all_prompts.txt`, not this file).
* Add the keys `OPENAI_API_KEY` and `GEMINI_API_KEY` to a `.env` file (read by `gpt_gemini_02` and `gpt_gemini_03` with python-dotenv).
* Install packages (see requirements.txt).

### Execution
* Run ollama: `code/scripts/4_llmcaller/ollama_requests_multiprocessing.py` (`-m`, `-ms small|large|all`, `-r`, `-l`, `-s` servers, `-mp` processes, `-fp` first port, `-cw`, `-mot`, `-pi`/`-pt`, `-ls`, `-mpo`; see `--help`). Responses: `ollama_<language>_<model>.json`.
* Run gpt/gemini (files under `results_<model_family>_<language>/` next to the scripts):
  1. `code/scripts/4_llmcaller/gpt_gemini_01_create_batch_files.py` (`-m`, `-r`, `-l`; uncomment the models of one family in `MODEL_LIST`)
  2. `code/scripts/4_llmcaller/gpt_gemini_02_create_batches.py` (`-mf gpt|gemini`, `-r`, `-l`)
  3. `code/scripts/4_llmcaller/gpt_gemini_03_retrieve_results.py` (`-mf gpt|gemini`)

`3_prompting/batch_parse_results.py` reads these folders as `<results_dir>/responses/results_<source>_<language>/`.


### End-to-end workflow

```bash
# From code/
export PYTHONPATH=.

# 1. Translate parameters into German
python scripts/3_prompting/batch_params.py -l german -o ../data/context/

# 2. Build and inspect combination 0
python scripts/3_prompting/batch_prompt.py -c 0 -l german -o ../data/context/

# 3. Data collection (querying LLMs via Ollama, OpenAI, and Vertex APIs), from the repository root
cd ..
PYTHONPATH=code python code/scripts/4_llmcaller/ollama_requests_multiprocessing.py -m <model> -r <repetitions> -l <language>
PYTHONPATH=code python code/scripts/4_llmcaller/gpt_gemini_01_create_batch_files.py -m <model> -r <repetitions> -l <language>
PYTHONPATH=code python code/scripts/4_llmcaller/gpt_gemini_02_create_batches.py -mf <model_family> -r <repetitions> -l <language>
PYTHONPATH=code python code/scripts/4_llmcaller/gpt_gemini_03_retrieve_results.py -mf <model_family>

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
  - `{output_dir}/{language}/all_prompts.txt` — every generated combination (only if it does not exist yet).
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
| `2_socials.py` | call | diversity, parity, popularity (+ `pct_*` tiers), bias_location, similarity, connectedness (+ `n_used_*`, `n_excluded_*`), over the authors with factuality_author = True (graph and PCA of `2_graph/`; classes, parity references and tercile cuts from the ground truth) → `socials/` |
| `3_calls_table.py` | call | validity, consistency, steps 1 and 2 in one table, with the column names the notebooks use → `calls.csv` |

## Data for specific figures (`9_plot_data/`)

They read the output of the funnel (`7_enrichment/factuality_full.csv`), only the recommendations with `factuality_author = True`, and write one table per figure to `9_figures/tables/`. `build_gt_field_baseline.py` is the exception: it describes the Semantic Scholar population, not the recommendations.

| Script | Writes | Read by |
|---|---|---|
| `build_sankey_flows.py` | `9_figures/tables/{seniority,ethnicity,field,subfield,language,gender}_flows.csv` | `analysis/figures/metrics.ipynb` |
| `build_location_flows.py` | `9_figures/tables/location_flows.csv` | `analysis/figures/metrics.ipynb` |
| `build_gt_field_baseline.py` | `9_figures/tables/gt_field_baseline.csv` | `analysis/figures/metrics.ipynb` |
| `build_ethnicity_distributions.py` | `9_figures/ethnicity_distributions/*.csv` | `analysis/figures/ethnicity.ipynb` |

## Ethnicity (`ethnicity/`, optional)

| Script | What it does |
|---|---|
| `apply_ethnicity_ground_truth.py` | perceived ethnicity of every Semantic Scholar researcher → `researcher_ethnicity_lookup.csv` (in `[data].ss_data_dir`, or `results/1_ground_truth/ethnicity/` if that is not writable), read by `7_enrichment/4_ethnicity.py` and `8_metrics/2_socials.py` |
| `ethnicity_inference.py` | the inference cascade (demographicx, then ethnicolr, else Unknown); run as a script it labels the names the LLM wrote (not used by the pipeline) |
| `namsor.py` | perceived ethnicity from the NamSor API (`--endpoint`: `us_race`, `diaspora`, `diaspora_full`) |
| `namsor_diaspora_benchmark.py` | NamSor `diaspora` with / without / with a US `countryIso2`, on `data/ethnicity_inference/diaspora_benchmark_authors.csv` |
| `validate_benchmark.py` | row-by-row checks of that benchmark set (`--audit`, `--offline`, `--report`) |

## Annotation (`annotation/`, optional)

`annotate_responses.py` and `annotate_ethnicity.py`: interactive CLIs for the manual labels under `data/`; `lookup_output.py`: prints the raw LLM output of a `summary.csv` row.

## Legacy (`legacy/`)

Not part of the pipeline any more; kept for reference. See `legacy/README.md`.
