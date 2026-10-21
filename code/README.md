# LLMScholar-Personas — Code

Turns the raw LLM responses (one JSON per persona × language × field × subfield × k × run) into metric tables and the paper's figures.

Everything runs from `code/` with `PYTHONPATH=.`. External paths (Semantic Scholar parquet, OpenAlex DuckDB, results folder) come from `[data]` in `config.ini` (copy `config.ini.example`).

```bash
cd code
export PYTHONPATH=.
PYTHONPATH=. python -m pytest tests/        # 159 tests
```

---

## Folder layout

```
code/
├── scripts/                 Programs you run. Each one says what it does in its docstring (--help).
│   ├── prompting/           Build the prompts and parse the LLM answers → recommendations.csv
│   ├── llmcaller/           Send the prompts to Ollama / OpenAI / Gemini
│   ├── openalex_cascade/    The OpenAlex author of each recommendation: name → field → seniority → subfield
│   ├── enrichment/          OpenAlex, location, affiliation and ethnicity of that author → factuality_full_<variant>.csv
│   ├── factuality/          Factuality cascade on Semantic Scholar: name → field → seniority → author
│   ├── pipeline/            Computes every metric, steps 1 → 6 in the order they run
│   ├── plot_data/           Data for specific figures (Sankey flows, countries, field baseline, ethnicity)
│   ├── run_all.py           Runs openalex_cascade → enrichment → factuality → pipeline → plot_data
│   ├── legacy/              Scripts not used any more (replaced or one-time experiments)
│   ├── ethnicity/           Ethnicity inference for the ground truth and the recommendations
│   └── annotation/          Interactive CLIs for manual labelling
├── libs/                    Code shared by scripts and notebooks (see "Where things live")
├── notebooks/
│   ├── analysis/figures/    Paper figures (plot only: they read the tables)
│   ├── analysis/checks/     Checks of the data behind the metrics
│   ├── agreement/           Inter-annotator agreement
│   └── legacy/              Notebooks of the pipeline before the factuality cascade
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

### 2. Everything else: one command

```bash
python scripts/run_all.py                    # all stages
python scripts/run_all.py --skip_name        # reuse the Semantic Scholar name matches (the slow step)
python scripts/run_all.py --from pipeline    # start at a given stage
```

| stage | folder | what it does |
|---|---|---|
| 1 | `openalex_cascade/` | every OpenAlex author with each recommended name; their field (two published classifications, see its README) and subfields; the pick per recommendation, once per **variant** |
| 2 | `enrichment/` | OpenAlex, location, affiliation and ethnicity of the picked author → `summary/factuality_full_<variant>.csv` |
| 3 | `factuality/` | the factuality cascade on Semantic Scholar → `summary/cascade/factuality_full_<variant>.csv` |
| 4 | `pipeline/` | every metric (table below) → `pipeline/6_calls.csv` |
| 5 | `plot_data/` | data for specific figures → `factualities/tables/<variant>/` |

**Variants.** When several OpenAlex authors with the recommended name pass the same checks (field, seniority, subfield), the last tie-break is either **random** or the **most cited**. Random can land on small or split OpenAlex profiles and the most cited on a famous namesake, so both are computed and reported: `6_calls.csv` has every call once per variant (`oa_variant` column). Semantic Scholar metrics are the same in both.

**Factuality cascade** (`scripts/factuality/`). For each recommended name:
`factuality_name` = someone in Semantic Scholar has that name; `factuality_field` = one of them is in the asked field; `factuality_seniority` = one of them has the asked seniority; `factuality_author` = one of them has both. The field and seniority come from the prompt, not from the LLM's answer. The reported researcher is picked at random among those that pass. The same four checks on OpenAlex are `oa_factuality_*`.

**Metrics pipeline** (`scripts/pipeline/`, also runnable alone with `run_pipeline.py [--from_step N]`):

| step | script | one row per | columns |
|---|---|---|---|
| 1 | `1_validity.py` | call | validity, refusal, duplicates |
| 2 | `2_consistency.py` | setup (model + params) | consistency |
| 3 | `3_factuality.py` | recommendation, per variant | factuality_* + matched_* (Semantic Scholar), oa_factuality_* + oa_matched_* (OpenAlex), bias_location |
| 4 | `4_author_stats.py` | author | h-index, i10-index, e-index from OpenAlex, for the authors of all variants (~95 min, only when step 3 brings new authors) |
| 5 | `5_socials.py` | call, per variant | diversity, parity, popularity, bias_location, similarity, connectedness |
| 6 | `6_calls_table.py` | call × variant | steps 1–5 in one table, with the column names the notebooks use |

Names are compared normalized (lowercase, no punctuation or extra spaces: `common.author_key`), so "J. Smith" and "j smith" are the same author in duplicates, consistency and the per-call factuality. Step 3 leaves out recommendations whose name has a single word (only a first name or only a last name). Step 5 scores the authors whose name exists in Semantic Scholar or in OpenAlex (`factuality_name` or `oa_factuality_name`), each name once per call; it builds the coauthorship graph and the PCA once, over the authors of all variants, so both variants are scored in the same space.

### 3. Figures and checks

```bash
cd notebooks/analysis/figures && jupyter nbconvert --execute metrics.ipynb
```

The notebooks only read tables written by `run_all.py`; they compute nothing heavy. `OA_VARIANT = 'random'` (or `'most_cited'`) at the top picks the OpenAlex variant.

**`analysis/figures/`: the paper's figures**

| Notebook | Shows | Reads |
|---|---|---|
| `metrics.ipynb` | every per-call metric by persona, context, field, model…, the trade-off plots and the Sankeys | `pipeline/6_calls.csv` + `factualities/tables/<variant>/*_flows.csv` |
| `sensitivity.ipynb` | which prompt variables move each metric (sensitivity analysis) | `pipeline/6_calls.csv` |
| `ethnicity.ipynb` | perceived ethnicity: ground truth vs recommendations | `ethnicity/distributions/*.csv` |

**`analysis/checks/`: checks of the data behind the metrics**

| Notebook | Checks | Reads |
|---|---|---|
| `openalex_field.ipynb` | how often the OpenAlex field agrees with Semantic Scholar's (72% / 78% / 84%) | `openalex_cascade/*.parquet`, `cascade/candidates_name.parquet` |
| `openalex_coverage.ipynb` | why publications / citations are NaN for some authors | `summary/cascade/factuality_full_<variant>.csv` |
| `scholarly_stats.ipynb` | h-index, i10-index and e-index against a hand-written formula | `[data].oa_author_stats` |
| `incomplete_names.ipynb` | recommendations with only a first name or only a last name | `cascade/factuality_cascade.csv` |
| `validity_flags.ipynb` | the automatic `valid_flag` against manual labels | `summary/summary.csv` + manual labels |

**`agreement/`**: inter-annotator agreement of the manual labels. **`legacy/`**: notebooks of the pipeline before the factuality cascade.

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

- They score the same authors as the rest of step 5 (name in Semantic Scholar or OpenAlex), but need their OpenAlex id: authors found only in Semantic Scholar are skipped and counted in `n_excluded_*`.
- The coauthorship graph (~1 h the first time, resumable) and the PCA are built once over all recommended authors with an OpenAlex id, in any variant, and cached in `../results/.cache/`. Each response is then scored with that same graph and PCA, so values are comparable across models and variants.
- Both are NaN when fewer than 2 authors can be scored (for example k = 1).

---

## Troubleshooting

- `ModuleNotFoundError: No module named 'libs'` → run from `code/` with `PYTHONPATH=.`.
- `KeyError: Missing [data].ss_parquet` → fill in `config.ini`.
- A notebook fails on its first data cell → run `scripts/run_all.py` (or the stage that writes the table it reads).
- DuckDB runs out of temp space → set `[data].oa_works_tmp_dir` to a disk with room (not `/tmp`).
