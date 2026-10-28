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
│   ├── 1_ground_truth/      Every Semantic Scholar researcher + their OpenAlex author and metadata → ground_truth.parquet
│   ├── 2_graph/             Author stats, coauthorship graph and similarity PCA over the whole ground truth
│   ├── 3_prompting/         Build the prompts and parse the LLM answers → recommendations.csv
│   ├── 4_llmcaller/         Send the prompts to Ollama / OpenAI / Gemini
│   ├── 5_validity/          Validity, refusal, duplicates of every call; consistency of the valid ones
│   ├── 6_factuality/        Factuality cascade on Semantic Scholar (valid calls): name → field → seniority → author
│   ├── 7_enrichment/        The LLM's answer vs the picked researcher's data: country, affiliation; ethnicity → factuality_full.csv
│   ├── 8_metrics/           Factuality, socials (factuality_author = True), calls.csv
│   ├── 9_plot_data/         Data for specific figures (Sankey flows, countries, field baseline, ethnicity)
│   ├── run_all.py           Runs 1, 2 and 5 → 9
│   ├── annotation/          Interactive CLIs for manual labelling (optional)
│   ├── ethnicity/           Ethnicity inference for the ground truth and the recommendations (optional)
│   └── legacy/              Scripts not used any more (replaced or one-time experiments)
├── libs/                    Code shared by scripts and notebooks (see "Where things live")
├── notebooks/
│   ├── analysis/figures/    Paper figures (plot only: they read the tables)
│   ├── analysis/checks/     Checks of the data behind the metrics
│   ├── agreement/           Inter-annotator agreement
│   └── legacy/              Old notebooks
└── tests/                   pytest
```

---

## Where the results go (`../results/`)

Each folder of results has the number of the script folder that writes it:

```
results/
├── 0_responses/        raw LLM answers (JSON), from 4_llmcaller/
├── 1_ground_truth/     ground_truth.parquet + its intermediate files (ss_papers, jw_match, candidates, picks, ...)
│   ├── ss_papers_all/        Semantic Scholar API: papers of every researcher, which ids no longer exist
│   ├── factual_authors_api/  the same for factuality_author = True, plus the merged profiles followed one by one
│   ├── merged_recovered/     those recovered researchers, run through steps 2–6
│   └── ethnicity/            perceived ethnicity of every researcher (researcher_ethnicity_lookup.csv)
├── 2_graph/            author_stats/ (h, i10, e-index) and cache/ (coauthorship graph, PCA)
├── 3_recommendations/  recommendations.csv, summary.csv (parsed LLM answers)
├── 5_validity/         validity/, consistency/, recommendations_valid.csv (valid calls, each author once)
├── 6_factuality/       candidates_*, factuality_cascade.csv, factuality_ss.csv, factual_authors.csv
├── 7_enrichment/       factuality_oa / _location / _affiliation / _full.csv
├── 8_metrics/          factuality/, socials/, calls.csv (every metric of every call)
├── 9_figures/          tables/, ethnicity_distributions/, plots/, sensitivity_analysis/
├── manual_labels/      manual labels and validations
├── logs/               the logs of the runs (reorg_2026-10-26.tsv: where each file was before the reorganisation)
├── runs/               snapshots of earlier runs
└── legacy/             files the pipeline no longer uses
```

## Running everything, in order

### 1–2. The ground truth and the graph (once, before the LLM answers)

```bash
python scripts/1_ground_truth/run_pipeline.py    # every Semantic Scholar researcher (the API part takes ~1–2 days)
python scripts/2_graph/run_pipeline.py           # author stats, graph and PCA over the ground truth
```

| folder | what it does → output |
|---|---|
| `1_ground_truth/` | for every Semantic Scholar researcher: their papers (API), their OpenAlex author through those papers (Jaro-Winkler on the names, threshold 1.35) or else by name, field and seniority (the cascade), and that author's metadata (works, citations, h/i10, years, field, country and institution) → `1_ground_truth/ground_truth.parquet` (see its README) |
| `2_graph/` | h / i10 / e-index of every OpenAlex author of the ground truth → `2_graph/author_stats/`; coauthorship graph and similarity PCA over all of them → `.cache/*_gt.joblib` |

### 3–4. Prompts and LLM answers

```bash
python scripts/3_prompting/batch_params.py -l english -o ../data/context/   # also spanish, german
python scripts/3_prompting/batch_prompt.py -l english -o ../data/context/
# send the prompts: scripts/4_llmcaller/ (see scripts/README.md)
python scripts/3_prompting/batch_parse_results.py --results_dir ../results --output_dir ../results/3_recommendations
```

Output: `../results/3_recommendations/recommendations.csv`, one row per recommended author with the response's `valid_flag` and `run_id`.

### 5–9. Everything else: one command

```bash
python scripts/run_all.py --from 5_validity      # the ground truth and the graph already built
python scripts/run_all.py --skip_name            # reuse the Semantic Scholar name matches (the slow step)
```

It is a **funnel**: each folder only passes on what survived the previous one.

| folder | on what | what it does → output |
|---|---|---|
| `5_validity/` | every call | `1_validity.py`: validity, refusal, duplicates → `5_validity/validity/`, and the recommendations of the valid calls → `5_validity/recommendations_valid.csv`; `2_consistency.py`: consistency of the valid calls → `5_validity/consistency/` |
| `6_factuality/` | valid calls | the factuality cascade on Semantic Scholar; the picked researcher (only when `factuality_author = True`) brings their ground-truth data, OpenAlex included → `6_factuality/factuality_ss.csv` |
| `7_enrichment/` | factuality_author = True | prompt country vs OpenAlex country; LLM affiliations vs OpenAlex institutions; perceived ethnicity of the picked researcher (ground truth) → `7_enrichment/factuality_full.csv` |
| `8_metrics/` | valid calls; socials only factuality_author = True | every metric (table below) → `8_metrics/calls.csv` |
| `9_plot_data/` | factuality_author = True | data for specific figures → `9_figures/tables/` |

**Factuality cascade** (`6_factuality/`). For each recommended name:
`factuality_name` = someone in Semantic Scholar has that name; `factuality_field` = one of them is in the asked field; `factuality_seniority` = one of them has the asked seniority; `factuality_author` = one of them has both. The field and seniority come from the prompt, not from the LLM's answer. The reported researcher is picked uniformly at random among those that pass, with a draw that depends only on the name, field and seniority (reproducible; other rows never change it). Names are compared with one normalization everywhere (`author_key`: "name lastname", lowercase, no punctuation): duplicates, consistency, the repeated names left out before factuality, and the name searched in Semantic Scholar. OpenAlex adds data about the author (through the ground truth), it does not decide factuality.

**Metrics** (`8_metrics/`, also runnable alone with `run_pipeline.py [--from_step N]`):

| step | script | one row per | columns |
|---|---|---|---|
| 1 | `1_factuality.py` | recommendation | factuality_* + matched_* (Semantic Scholar), bias_location (only factuality_author = True) |
| 2 | `2_socials.py` | call | diversity, parity, popularity, bias_location, similarity, connectedness (factuality_author = True; graph and PCA of `2_graph/`) |
| 3 | `3_calls_table.py` | call | validity, consistency and steps 1 and 2 in one table, with the column names the notebooks use → `8_metrics/calls.csv` |

Names are compared normalized (lowercase, no punctuation or extra spaces: `libs/pipeline_common.author_key`), so "J. Smith" and "j smith" are the same author in duplicates, consistency and the per-call factuality. `1_factuality.py` leaves out recommendations whose name has a single word (only a first name or only a last name). `2_socials.py` scores the authors with `factuality_author = True`, each researcher once per call.

**The funnel in numbers** (run of 2026-10-21, before the ground truth was built for all researchers):

```
928,800 calls · 3.91 M recommendations
  │ 5_validity
  ▼
797,301 valid calls · 3.74 M recommendations
  │ 6_factuality (Semantic Scholar)
  ▼
694 k recommendations with factuality_author = True (18.5%)
  │ their OpenAlex author
  ▼
573 k found in OpenAlex · 88,843 distinct authors
  │ 8_metrics
  ▼
8_metrics/calls.csv → notebooks
```

**What each metric means** (per call; paper equations in brackets):

| metric | how it is computed | on what |
|---|---|---|
| `validity` | the answer is a populated JSON (`valid_flag` cleaned or unchanged) | every call |
| `refusal` | the model declined and said why | every call |
| `duplicates` | 1 − unique names / names (Eq. 3) | valid calls |
| `consistency` | mean Jaccard of the authors of consecutive runs of the same setup (Eq. 4) | valid calls |
| `factuality_name / field / seniority / author` | share of the call's names that pass each check in Semantic Scholar | valid calls |
| `diversity_<x>` | normalised entropy of gender, ethnicity, country, publications, citations, divided by log of the number of classes (Eq. 9) | factuality_author = True |
| `parity_<x>` | 1 − total variation distance to the ground truth's distribution over the same classes (Eq. 11) | factuality_author = True |
| `popularity_<x>` | share of authors in the top third of publications or citations of their field in the ground truth | factuality_author = True |

The classes are every category that exists in the ground truth, not only those the recommendations show: gender Female / Male / Unknown, ethnicity Asian / Black / White / Hispanic / Unknown, country every OpenAlex country of the ground truth (226), publications and citations low / med / high (the per-field terciles of the ground truth). Diversity leaves the Unknown authors out (paper Eq. 9); parity keeps Unknown as a class. See the docstring of `scripts/8_metrics/2_socials.py`.
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
| `metrics.ipynb` | every per-call metric by persona, context, field, model…, the trade-off plots and the Sankeys | `8_metrics/calls.csv` + `9_figures/tables/*_flows.csv` |
| `sensitivity.ipynb` | which prompt variables move each metric (sensitivity analysis) | `8_metrics/calls.csv` |
| `ethnicity.ipynb` | perceived ethnicity: ground truth vs recommendations | `9_figures/ethnicity_distributions/*.csv` |

**`analysis/checks/`: checks of the data behind the metrics**

| Notebook | Checks | Reads |
|---|---|---|
| `jw_threshold.ipynb` | how the Jaro-Winkler threshold of the OpenAlex author match (1.35) was chosen, on labels that do not use names | `1_ground_truth/factual_authors_api/checks/jw_signers_sample.parquet` |
| `ss_vs_oa_metadata.ipynb` | which author metadata each source has (checked with queries) and how often it is filled | Semantic Scholar API, OpenAlex DB |
| `ss_merged_profiles.ipynb` | that the batch call returns null for a merged Semantic Scholar profile and only the individual call gives the new id | Semantic Scholar API |
| `ss_api_metadata.ipynb` | which author metadata the Semantic Scholar API gives (no affiliations) | Semantic Scholar API |
| `openalex_field.ipynb` | how often the OpenAlex field agrees with Semantic Scholar's (72% / 78% / 84%) | `1_ground_truth/factual_authors_api/*.parquet`, `6_factuality/candidates_name.parquet` |
| `openalex_coverage.ipynb` | why publications / citations are NaN for some authors | `7_enrichment/factuality_full.csv` |
| `scholarly_stats.ipynb` | h-index, i10-index and e-index against a hand-written formula | `[data].oa_author_stats` |
| `incomplete_names.ipynb` | recommendations with only a first name or only a last name | `6_factuality/factuality_cascade.csv` |
| `validity_flags.ipynb` | the automatic `valid_flag` against manual labels | `3_recommendations/summary.csv` + manual labels |

**`agreement/`**: inter-annotator agreement of the manual labels. **`legacy/`**: notebooks of the pipeline before the factuality cascade, and one-time checks (`duplicates_july.ipynb`).

### Optional branches

- **Ethnicity** — `scripts/ethnicity/apply_ethnicity_ground_truth.py` builds the ground-truth table (`results/1_ground_truth/ethnicity/researcher_ethnicity_lookup.csv`) that `7_enrichment/4_ethnicity.py` and `8_metrics/2_socials.py` read; `ethnicity_inference.py` infers it from the names the LLM wrote (no longer used by the pipeline).
- **Annotation** — `scripts/annotation/` (manual labels under `data/`).

---

## Legacy scripts (`scripts/legacy/`)

Scripts that are not part of the pipeline any more, kept for reference: `build_valid_calls.py` (the old per-call table; replaced by `5_validity/` and `8_metrics/`), the old field and seniority checks and `resolve_homonyms.py` (replaced by the factuality cascade), and one-time tools (`run_similarity.py`, `merge_structural_metrics.py`, `export_coauthor_links.py`, `factuality_seniority_oa.py`). See `scripts/legacy/README.md`.

---

## Where things live (`libs/`)

| Module | Contents |
|---|---|
| `openalex_common.py` | `normalize_name` (how names are compared with OpenAlex) and the folder of the per-year country/institution chunks |
| `metrics/structural_model.py` | The coauthorship graph and the similarity PCA over the ground truth (built by `2_graph/`, read by `8_metrics/`) |
| `pipeline_common.py` | What `5_validity/` and `8_metrics/` share: call / setup columns, `author_key`, `has_full_name`, reading and writing the per-model files |
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

- They score the same authors as the rest of `2_socials.py` (factuality_author = True), but need their OpenAlex id: factual authors without one are skipped and counted in `n_excluded_*`.
- The coauthorship graph and the PCA are built once over **every author of the ground truth** with an OpenAlex id (`2_graph/`) and cached in `../results/2_graph/cache/` (`*_gt.joblib`). Each response takes a subgraph and the embeddings of its authors from them, so values are comparable across models.
- Both are NaN when fewer than 2 authors can be scored (for example k = 1).

---

## Troubleshooting

- `ModuleNotFoundError: No module named 'libs'` → run from `code/` with `PYTHONPATH=.`.
- `KeyError: Missing [data].ss_parquet` → fill in `config.ini`.
- A notebook fails on its first data cell → run `scripts/run_all.py` (or the stage that writes the table it reads).
- DuckDB runs out of temp space → set `[data].oa_works_tmp_dir` to a disk with room (not `/tmp`).
