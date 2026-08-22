# Scripts

All scripts run from the `code/` directory with `PYTHONPATH=.` so that `libs.*` imports resolve. Relative paths in defaults point at `../data/` and `../results/` (one level up from `code/`).

```
scripts/
├── prompting/    Generate and parse batch prompts sent to the LLMs.
├── llmcaller/    Data collection using Ollama, OpenAI, and Vertex APIs.
├── annotation/   Interactive CLIs for manual annotation.
├── factuality/   Factuality-check pipeline (7 ordered steps + orchestrator).
├── ethnicity/    Ethnicity inference cascade over the ground truth.
└── metrics/      Pre-compute the aggregated tables consumed by the plotting notebooks.
```

For per-script details, see each script's docstring.

---

4. **`ollama_requests_multiprocessing.py`** - Calling oLLama API to access multiple LLMs
5. **`gpt_gemini_01_create_batch_files.py`** - 
6. **`gpt_gemini_02_create_batches.py`** - 
7. **`gpt_gemini_03_retrieve_results.py`** - 


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

## Metrics (`metrics/`)

Two scripts pre-compute the aggregated tables consumed by `notebooks/analysis/`:

- **`build_valid_calls.py`** — reads `summary/factuality_full.csv`, joins ground truth, derives every per-call metric (factuality, diversity, parity, consistency, duplicates, popularity, connectedness, similarity), and writes `factualities/tables/valid_requests_metadata.csv`.
  - The two structural metrics (paper Eqs. 6-8) build a coauthorship graph from the OpenAlex snapshot (`[data].oa_duckdb`) and a PCA embedding of author features. Both are cached under `<results_dir>/.cache`, so the pass over `oa.works` runs once; use `--rebuild_structural` to force it. Without `--oa_duckdb` they are skipped and no other metric changes.
  - That pass is the expensive step: it explodes the `authorships` list of all 492M works. It is chunked by `works.id` to bound peak memory — a single-pass version exhausted 56 GiB of DuckDB temp space. Budget hours, not minutes, and run it once.
- **`build_location_flows.py`** — reads `summary/factuality_full.csv` in chunks and writes `factualities/tables/location_flows.csv`: one row per (prompt country → author country, field, language, model) with a recommendation count. Feeds the location Sankey figure (issue #37). Recommendations whose author has no OpenAlex country (15.2%) cannot form a flow and are excluded, reported as a coverage line in the log.

- **`build_ethnicity_distributions.py`** — reads the per-field ground-truth CSVs and `recommendations_with_ethnicity.csv`, computes the distributions needed by `notebooks/analysis/ethnicity_metrics.ipynb`, and writes them under `ethnicity/distributions/`.

Both scripts read their default paths from the `[data]` section of `config.ini`.

```bash
# From code/
python scripts/metrics/build_valid_calls.py
python scripts/metrics/build_ethnicity_distributions.py
```
