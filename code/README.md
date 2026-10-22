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
├── scripts/                 Programs you run, folders numbered in the order they run
│   ├── 1_prompting/         Build the prompts and parse the LLM answers → recommendations.csv
│   ├── 2_llmcaller/         Send the prompts to Ollama / OpenAI / Gemini
│   ├── 3_validity/          Validity, refusal, duplicates of every call; consistency of the valid ones
│   ├── 4_factuality/        Factuality cascade on Semantic Scholar (valid calls): name → field → seniority → author
│   ├── 5_openalex_cascade/  The OpenAlex author of each confirmed recommendation (factuality_author = True)
│   ├── 6_enrichment/        Their country, institution, works, citations; ethnicity for all → factuality_full.csv
│   ├── 7_metrics/           Factuality, author stats, socials (factuality_author = True), calls.csv
│   ├── 8_plot_data/         Data for specific figures (Sankey flows, countries, field baseline, ethnicity)
│   ├── run_all.py           Runs 3 → 8
│   ├── annotation/          Interactive CLIs for manual labelling (optional)
│   ├── ethnicity/           Ethnicity inference for the ground truth and the recommendations (optional)
│   └── legacy/              Scripts not used any more (replaced or one-time experiments)
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

### 1–2. Prompts and LLM answers

```bash
python scripts/1_prompting/batch_params.py -l english -o ../data/context/   # also spanish, german
python scripts/1_prompting/batch_prompt.py -l english -o ../data/context/
# send the prompts: scripts/2_llmcaller/ (see scripts/README.md)
python scripts/1_prompting/batch_parse_results.py --results_dir ../results --output_dir ../results/summary
```

Output: `../results/summary/recommendations.csv`, one row per recommended author with the response's `valid_flag` and `run_id`.

### 3–8. Everything else: one command

```bash
python scripts/run_all.py                        # 3 → 8
python scripts/run_all.py --skip_name            # reuse the Semantic Scholar name matches (the slow step)
python scripts/run_all.py --from 7_metrics       # start at a given folder
```

It is a **funnel** (the order of the team's email): each folder only passes on what survived the previous one.

| folder | on what | what it does → output |
|---|---|---|
| `3_validity/` | every call | `1_validity.py`: validity, refusal, duplicates → `pipeline/validity/`; `2_consistency.py`: consistency of the valid calls → `pipeline/consistency/` |
| `4_factuality/` | valid calls | the factuality cascade on Semantic Scholar → `summary/cascade/factuality_ss.csv`, and the rows with `factuality_author = True` → `summary/cascade/factual_authors.csv` |
| `5_openalex_cascade/` | factuality_author = True | their OpenAlex author: name → field (two published classifications, see its README) → seniority → subfield, then one at random → `picks.csv` |
| `6_enrichment/` | factuality_author = True (ethnicity: all) | country, institution, works, citations from OpenAlex; prompt country vs OpenAlex country; LLM affiliations vs OpenAlex institutions; perceived ethnicity → `summary/factuality_full.csv` |
| `7_metrics/` | valid calls; socials only factuality_author = True | every metric (table below) → `pipeline/calls.csv` |
| `8_plot_data/` | | data for specific figures → `factualities/tables/` |

**Factuality cascade** (`4_factuality/`). For each recommended name:
`factuality_name` = someone in Semantic Scholar has that name; `factuality_field` = one of them is in the asked field; `factuality_seniority` = one of them has the asked seniority; `factuality_author` = one of them has both. The field and seniority come from the prompt, not from the LLM's answer. The reported researcher is picked at random among those that pass. Only the recommendations with `factuality_author = True` go on to OpenAlex; OpenAlex adds data about the author, it does not decide factuality.

**Metrics** (`7_metrics/`, also runnable alone with `run_pipeline.py [--from_step N]`):

| step | script | one row per | columns |
|---|---|---|---|
| 1 | `1_factuality.py` | recommendation | factuality_* + matched_* (Semantic Scholar), bias_location (only factuality_author = True) |
| 2 | `2_author_stats.py` | author | h-index, i10-index, e-index from OpenAlex (~95 min, only when step 1 brings new authors) |
| 3 | `3_socials.py` | call | diversity, parity, popularity, bias_location, similarity, connectedness (factuality_author = True) |
| 4 | `4_calls_table.py` | call | validity, consistency and steps 1 and 3 in one table, with the column names the notebooks use → `pipeline/calls.csv` |

Names are compared normalized (lowercase, no punctuation or extra spaces: `libs/pipeline_common.author_key`), so "J. Smith" and "j smith" are the same author in duplicates, consistency and the per-call factuality. `1_factuality.py` leaves out recommendations whose name has a single word (only a first name or only a last name). `3_socials.py` scores the authors with `factuality_author = True`, each researcher once per call.

**The funnel in numbers** (run of 2026-10-21):

```
928,800 calls · 3.91 M recommendations
  │ 3_validity
  ▼
797,301 valid calls · 3.74 M recommendations
  │ 4_factuality (Semantic Scholar)
  ▼
694 k recommendations with factuality_author = True (18.5%)
  │ 5_openalex_cascade + 6_enrichment (OpenAlex)
  ▼
573 k found in OpenAlex · 88,843 distinct authors
  │ 7_metrics
  ▼
pipeline/calls.csv → notebooks
```

**What each metric means** (per call; paper equations in brackets):

| metric | how it is computed | on what |
|---|---|---|
| `validity` | the answer is a populated JSON (`valid_flag` cleaned or unchanged) | every call |
| `refusal` | the model declined and said why | every call |
| `duplicates` | 1 − unique names / names (Eq. 3) | valid calls |
| `consistency` | mean Jaccard of the authors of consecutive runs of the same setup (Eq. 4) | valid calls |
| `factuality_name / field / seniority / author` | share of the call's names that pass each check in Semantic Scholar | valid calls |
| `diversity_<x>` | normalised entropy of gender, ethnicity, country, publications, citations (Eq. 9) | factuality_author = True |
| `parity_<x>` | 1 − total variation distance to a reference: Semantic Scholar for gender and ethnicity, equal thirds for publications and citations (Eq. 11) | factuality_author = True |
| `popularity_<x>` | share of authors in the top third of publications or citations of their field | factuality_author = True |
| `bias_location` | share of authors whose OpenAlex country is the prompt's country | factuality_author = True |
| `similarity` | mean pairwise cosine of the authors' scholarly profiles (PCA of 8 features) (Eq. 8) | factuality_author = True, in OpenAlex |
| `connectedness` | 1 − normalised entropy of the coauthorship components among the authors (Eqs. 6–7) | factuality_author = True, in OpenAlex |

### Figures and checks

```bash
cd notebooks/analysis/figures && jupyter nbconvert --execute metrics.ipynb
```

The notebooks only read tables written by `run_all.py`; they compute nothing heavy.

**`analysis/figures/`: the paper's figures**

| Notebook | Shows | Reads |
|---|---|---|
| `metrics.ipynb` | every per-call metric by persona, context, field, model…, the trade-off plots and the Sankeys | `pipeline/calls.csv` + `factualities/tables/*_flows.csv` |
| `sensitivity.ipynb` | which prompt variables move each metric (sensitivity analysis) | `pipeline/calls.csv` |
| `ethnicity.ipynb` | perceived ethnicity: ground truth vs recommendations | `ethnicity/distributions/*.csv` |

**`analysis/checks/`: checks of the data behind the metrics**

| Notebook | Checks | Reads |
|---|---|---|
| `openalex_field.ipynb` | how often the OpenAlex field agrees with Semantic Scholar's (72% / 78% / 84%) | `openalex_cascade/*.parquet`, `cascade/candidates_name.parquet` |
| `openalex_coverage.ipynb` | why publications / citations are NaN for some authors | `summary/factuality_full.csv` |
| `scholarly_stats.ipynb` | h-index, i10-index and e-index against a hand-written formula | `[data].oa_author_stats` |
| `incomplete_names.ipynb` | recommendations with only a first name or only a last name | `cascade/factuality_cascade.csv` |
| `validity_flags.ipynb` | the automatic `valid_flag` against manual labels | `summary/summary.csv` + manual labels |

**`agreement/`**: inter-annotator agreement of the manual labels. **`legacy/`**: notebooks of the pipeline before the factuality cascade.

### Optional branches

- **Ethnicity** — `scripts/ethnicity/apply_ethnicity_ground_truth.py` (ground truth) and `ethnicity_inference.py` (recommendations).
- **Annotation** — `scripts/annotation/` (manual labels under `data/`).

---

## Legacy scripts (`scripts/legacy/`)

Scripts that are not part of the pipeline any more, kept for reference: `build_valid_calls.py` (the old per-call table; replaced by `3_validity/` and `7_metrics/`), the old field and seniority checks and `resolve_homonyms.py` (replaced by the factuality cascade), and one-time tools (`run_similarity.py`, `merge_structural_metrics.py`, `export_coauthor_links.py`, `factuality_seniority_oa.py`). See `scripts/legacy/README.md`.

---

## Where things live (`libs/`)

| Module | Contents |
|---|---|
| `pipeline_common.py` | What `3_validity/` and `7_metrics/` share: call / setup columns, `author_key`, `has_full_name`, reading and writing the per-model files |
| `metrics/constants.py` | Column names, valid flags, category orders and maps (`GENDER_MAP`, `ETHNICITY_MAP`), seniority rule (`REFERENCE_YEAR`, `JUNIOR_MAX`, `SENIOR_MIN`), translations to English |
| `metrics/coauthorship.py` | Coauthorship graph and `compute_connectedness` (Eqs. 6-7) |
| `metrics/similarity.py` | Author features, PCA embeddings and `compute_similarity` (Eq. 8) |
| `metrics/productivity.py` | Low / med / high tiers of publications and citations |
| `metrics/aggregators.py` | Averages and confidence intervals for the plots |
| `metrics/io.py` | `validate_path` (create the folder of an output file) |
| `metrics/agreement.py` | Kappa / alpha for annotator agreement |
| `visuals/` | Plots and plot styling |
| `utils/` | Config (`config.py`), files (`ios.py`), logging (`logging.py`) |
| `prompt/`, `llm/` | Prompt building and the OpenAI client |
| `legacy/` | Code nothing uses any more, kept for reference. Do not import it in new code. |

---

## Similarity and connectedness, in short

- They score the same authors as the rest of `3_socials.py` (factuality_author = True), but need their OpenAlex id: factual authors OpenAlex did not find are skipped and counted in `n_excluded_*`.
- The coauthorship graph (~1 h the first time, resumable) and the PCA are built once over all factual authors with an OpenAlex id and cached in `../results/.cache/`. Each response is then scored with that same graph and PCA, so values are comparable across models.
- Both are NaN when fewer than 2 authors can be scored (for example k = 1).

---

## Troubleshooting

- `ModuleNotFoundError: No module named 'libs'` → run from `code/` with `PYTHONPATH=.`.
- `KeyError: Missing [data].ss_parquet` → fill in `config.ini`.
- A notebook fails on its first data cell → run `scripts/run_all.py` (or the stage that writes the table it reads).
- DuckDB runs out of temp space → set `[data].oa_works_tmp_dir` to a disk with room (not `/tmp`).
