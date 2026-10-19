# LLMScholar-Personas — Code

Turns the raw LLM responses (one JSON per persona × language × field × subfield × k × run) into metric tables and the paper's figures.

Everything runs from `code/` with `PYTHONPATH=.`. External paths (Semantic Scholar parquet, OpenAlex DuckDB, results folder) come from `[data]` in `config.ini` (copy `config.ini.example`).

```bash
cd code
export PYTHONPATH=.
PYTHONPATH=. python -m pytest tests/        # 178 tests
```

---

## Folder layout

```
code/
├── scripts/                 Programs you run. Each one says what it does in its docstring (--help).
│   ├── prompting/           Build the prompts and parse the LLM answers → recommendations.csv
│   ├── llmcaller/           Send the prompts to Ollama / OpenAI / Gemini
│   ├── enrichment/          OpenAlex, location, affiliation and ethnicity → factuality_full.csv
│   ├── factuality/          Factuality cascade: name → field → seniority → author
│   ├── pipeline/            Computes every metric, steps 1 → 6 in the order they run
│   ├── plot_data/           Data for specific figures (Sankey flows, countries, field baseline, ethnicity)
│   ├── legacy/              Scripts not used any more (replaced or one-time experiments)
│   ├── ethnicity/           Ethnicity inference for the ground truth and the recommendations
│   └── annotation/          Interactive CLIs for manual labelling
├── libs/                    Code shared by scripts and notebooks (see "Where things live")
├── notebooks/
│   ├── analysis/            Paper figures (plot only: they read the tables, they compute nothing heavy)
│   ├── agreement/           Inter-annotator agreement
│   └── legacy/              Old notebooks
└── tests/                   pytest
```

---

## Running everything, in order

### 1. Prompts and LLM answers

```bash
python scripts/prompting/batch_params.py -l english -o ../data/context/   # also spanish, german
python scripts/prompting/batch_prompt.py -l english -o ../data/context/
# send the prompts: scripts/llmcaller/ (see scripts/README.md)
python scripts/prompting/batch_parse_results.py --results_dir ../results --output_dir ../results/summary
```

Output: `../results/summary/recommendations.csv`, one row per recommended author with the response's `valid_flag` and `run_id`.

### 2. OpenAlex, location, affiliation, ethnicity

```bash
python scripts/enrichment/run_enrichment.py
```

Output: `../results/summary/factuality_full.csv`. The cascade below copies its OpenAlex / location / affiliation / ethnicity columns from this file, so it must exist first.

### 3. Metrics pipeline

```bash
python scripts/pipeline/run_pipeline.py --skip_name     # --skip_name reuses the slow name matching
python scripts/pipeline/run_pipeline.py --from_step 5   # re-run from a given step
```

| step | script | one row per | columns |
|---|---|---|---|
| 1 | `1_validity.py` | call | validity, refusal, duplicates |
| 2 | `2_consistency.py` | setup (model + params) | consistency |
| 3 | `3_factuality.py` (runs `factuality/`) | recommendation | factuality_name/field/seniority/author + matched_* + bias_location |
| 4 | `4_author_stats.py` | author | h-index, i10-index, e-index from OpenAlex (~95 min, only when step 3 brings new authors) |
| 5 | `5_socials.py` | call | diversity, parity, popularity, bias_location, similarity, connectedness |
| 6 | `6_calls_table.py` | call | steps 1–5 in one table, with the column names the notebooks use |

Output: `../results/pipeline/<step>/<model>.csv` (one CSV per model) and `../results/pipeline/6_calls.csv` (all calls, read by the notebooks). Step 4 writes to `[data].oa_author_stats`.

Step 3 leaves out recommendations whose name has a single word (only a first name or only a last name).

**Factuality cascade** (`scripts/factuality/`, run by step 3). For each recommended name:
`factuality_name` = someone in Semantic Scholar has that name; `factuality_field` = one of them is in the asked field; `factuality_seniority` = one of them has the asked seniority; `factuality_author` = one of them has both. The field and seniority come from the prompt, not from the LLM's answer. The reported researcher is picked at random among those that pass.

### 4. Data for specific figures

```bash
python scripts/plot_data/build_sankey_flows.py             # asked → actual: seniority, field, gender, ...
python scripts/plot_data/build_location_flows.py           # prompt country → author country
python scripts/plot_data/build_gt_field_baseline.py        # field mix of Semantic Scholar
python scripts/plot_data/build_ethnicity_distributions.py  # for ethnicity_metrics.ipynb
```

They read the factuality cascade output (`summary/cascade/factuality_full.csv`) and write to `factualities/tables/`.

### 5. Figures

```bash
cd notebooks/analysis && jupyter nbconvert --execute metrics_pipeline.ipynb
```

| Notebook | Reads |
|---|---|
| `analysis/metrics_pipeline.ipynb` | `pipeline/6_calls.csv` + `factualities/tables/*_flows.csv` |
| `analysis/ethnicity_metrics.ipynb` | `ethnicity/distributions/*.csv` |
| `analysis/oa_productivity_coverage.ipynb` | `summary/factuality_full.csv` |
| `analysis/sensitivity.ipynb` | `pipeline/6_calls.csv` |
| `analysis/manual_classification_metrics.ipynb` | manual labels vs `valid_flag` |
| `analysis/scholarly_stats_check.ipynb` | checks of the h/i10/e-index |
| `analysis/incomplete_names_check.ipynb` | recommendations with only a name or only a last name |
| `agreement/*.ipynb` | manual labels of two annotators |

### Optional branches

- **Ethnicity** — `scripts/ethnicity/apply_ethnicity_ground_truth.py` (ground truth) and `ethnicity_inference.py` (recommendations).
- **Annotation** — `scripts/annotation/` (manual labels under `data/`).

---

## Legacy scripts (`scripts/legacy/`)

Scripts that are not part of the pipeline any more, kept for reference: `build_valid_calls.py` (the old per-call table; replaced by `pipeline/`), the old field and seniority checks and `resolve_homonyms.py` (replaced by the factuality cascade), and one-time tools (`run_similarity.py`, `merge_structural_metrics.py`, `export_coauthor_links.py`, `factuality_seniority_oa.py`). See `scripts/legacy/README.md`.

---

## Where things live (`libs/`)

| Module | Contents |
|---|---|
| `metrics/constants.py` | Column names, valid flags, category orders and maps (`GENDER_MAP`, `ETHNICITY_MAP`), seniority rule (`REFERENCE_YEAR`, `JUNIOR_MAX`, `SENIOR_MIN`), translations to English |
| `metrics/coauthorship.py` | Coauthorship graph and `compute_connectedness` (Eqs. 6-7) |
| `metrics/similarity.py` | Author features, PCA embeddings and `compute_similarity` (Eq. 8) |
| `metrics/productivity.py` | Low / med / high tiers of publications and citations |
| `metrics/aggregators.py` | Averages and confidence intervals for the plots |
| `metrics/io.py` | Reading the structural-metric variant files |
| `metrics/agreement.py` | Kappa / alpha for annotator agreement |
| `visuals/` | Plots and plot styling |
| `utils/` | Config (`config.py`), files (`ios.py`), logging (`logging.py`) |
| `prompt/`, `llm/` | Prompt building and the OpenAI client |
| `legacy/` | Code nothing uses any more, kept for reference. Do not import it in new code. |

---

## Similarity and connectedness, in short

- Both need the OpenAlex id of the authors. Authors found only in Semantic Scholar are skipped.
- The coauthorship graph (~2.5 h the first time) and the PCA are built once over all recommended authors with an OpenAlex id (305,893) and cached in `../results/.cache/`. Each response is then scored with that same graph and PCA, so values are comparable across models.
- Both are NaN when fewer than 2 authors can be scored (for example k = 1).

---

## Troubleshooting

- `ModuleNotFoundError: No module named 'libs'` → run from `code/` with `PYTHONPATH=.`.
- `KeyError: Missing [data].ss_parquet` → fill in `config.ini`.
- A notebook fails on its first data cell → run the script that writes the table it reads (step 4 above).
- DuckDB runs out of temp space → set `[data].oa_works_tmp_dir` to a disk with room (not `/tmp`).
