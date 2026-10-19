# Scripts

All scripts run from the `code/` directory with `PYTHONPATH=.` so that `libs.*` imports resolve. Relative paths in defaults point at `../data/` and `../results/` (one level up from `code/`).

```
scripts/
├── prompting/           Build the prompts and parse the LLM answers → recommendations.csv
├── llmcaller/           Send the prompts to Ollama / OpenAI / Gemini
├── enrichment/          OpenAlex, location, affiliation and ethnicity → factuality_full.csv
├── factuality/          Factuality cascade: name → field → seniority → author (names.py: how names are compared)
├── pipeline/            Computes every metric, steps 1 → 6 in the order they run
├── plot_data/           Data for specific figures (Sankey flows, countries, field baseline, ethnicity)
├── legacy/              Scripts not used any more (replaced or one-time experiments)
├── ethnicity/           Ethnicity inference
└── annotation/          Interactive CLIs for manual labelling
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

## Batch processing (`prompting/`)

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
python scripts/prompting/batch_params.py -l german  -o ../data/context/
python scripts/prompting/batch_params.py -l spanish -o ../data/context/
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
python scripts/prompting/batch_prompt.py -c 0  -l german  -o ../data/context/
python scripts/prompting/batch_prompt.py -c 5  -l german
python scripts/prompting/batch_prompt.py -c 42 -l spanish
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
nice -n 10 parallel -j 20 python scripts/prompting/batch_parse_results.py --results_dir ../results --output_dir ../results/summary_parallel --model {1} --language {2} :::: ../data/context/models.txt ::: english german spanish
```

## LLMs Caller

### Requirements
* Data must be available at `data/context/...`.
* Add API keys for gpt an gemini in `.env`.
* Install packages (see requirements.txt).

### Execution
* Run ollama: `scripts/llmcaller/ollama_requests_multiprocessing.py` (see args for details)
* Run gpt/gemini:
  1. `scripts/llmcaller/gpt_gemini_01_create_batch_files.py` (modify language using args; uncomment models ony by one)
  2. `scripts/llmcaller/gpt_gemini_02_create_batches.py` (modify language and model family using args)
  3. `scripts/llmcaller/gpt_gemini_03_retrieve_results.py` (modify model family using args)


### End-to-end workflow

```bash
# From code/
export PYTHONPATH=.

# 1. Translate parameters into German
python scripts/prompting/batch_params.py -l german -o ../data/context/

# 2. Build and inspect combination 0
python scripts/prompting/batch_prompt.py -c 0 -l german -o ../data/context/

# 3. Data collection (querying LLMs via Ollama, OpenAI, and Vertex APIs)
python scripts/llmcaller/ollama_requests_multiprocessing.py -m <model>  -r <repetitions> -l <language>
python scripts/llmcaller/gpt_gemini_01_create_batch_files.py -m <model> -r <repetitions> -l <language>
python scripts/llmcaller/gpt_gemini_02_create_batches.py -mf <model_familiy> -r <repetitions> -l <language>
python scripts/llmcaller/gpt_gemini_03_retrieve_results.py -mf <model_familiy>

```

### GNU Parallel

```bash
# Process combinations 0–719 in parallel (8 simultaneous jobs)
parallel -j 8 python scripts/prompting/batch_prompt.py -c {} -l german -o ../data/context/ ::: {0..719}

# Multiple languages
parallel -j 4 python scripts/prompting/batch_params.py -l {} -o ../data/context/ ::: english german spanish

# With a progress bar
parallel --bar -j 8 python scripts/prompting/batch_prompt.py -c {} -l german -o ../data/context/ ::: {0..719}
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

## Metrics pipeline (`pipeline/`)

`run_pipeline.py` runs the four steps in order; each one writes one CSV per model under `../results/pipeline/<step>/`.

| Script | One row per | Writes |
|---|---|---|
| `1_validity.py` | call | validity, refusal, duplicates |
| `2_consistency.py` | setup (model + params) | consistency |
| `3_factuality.py` | recommendation | runs `factuality/`; factuality_*, matched_*, bias_location |
| `4_author_stats.py` | author | h-index, i10-index, e-index from OpenAlex (only when step 3 brings new authors) |
| `5_socials.py` | call | diversity, parity, popularity, bias_location, similarity, connectedness |
| `6_calls_table.py` | call | steps 1–5 in one table (`6_calls.csv`), with the column names the notebooks use |

`common.py` holds the column names (call, setup, params) and the helpers that read and write the per-model files.

```bash
python scripts/pipeline/run_pipeline.py --skip_name
```

## Factuality cascade (`factuality/`)

`run_pipeline.py` runs `1_name.py` (all researchers with that name; the slow step) → `2_field.py` → `3_seniority.py` → `4_author.py` (counts and the random pick), and then itself adds the old status columns and the OpenAlex / location / affiliation / ethnicity columns from `enrichment/`'s `factuality_full.csv`. `names.py` holds how names are compared. Output under `../results/summary/cascade/`.

## Data for specific figures (`plot_data/`)

They read the factuality cascade output (`summary/cascade/factuality_full.csv`) and write one table per figure.

| Script | Writes | Read by |
|---|---|---|
| `build_sankey_flows.py` | `factualities/tables/{seniority,ethnicity,field,subfield,language,gender}_flows.csv` | `metrics_pipeline.ipynb` |
| `build_location_flows.py` | `factualities/tables/location_flows.csv` | `metrics_pipeline.ipynb` |
| `build_gt_field_baseline.py` | `factualities/tables/gt_field_baseline.csv` | `metrics_pipeline.ipynb` |
| `build_ethnicity_distributions.py` | `ethnicity/distributions/*.csv` | `ethnicity_metrics.ipynb` |

## Legacy (`legacy/`)

Not part of the pipeline any more; kept for reference. See `legacy/README.md`.
