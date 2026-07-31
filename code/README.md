# LLMScholar-Personas — Code

Pipeline that turns raw LLM responses (one JSON per persona × language × field × subfield × k × run) into the per-call metrics tables and figures used in the paper. The whole pipeline is driven from `code/` with `PYTHONPATH=.` and reads its external paths from `[data]` in `config.ini` (see `config.ini.example`).

```
code/
├── libs/               Shared libraries imported by scripts and notebooks.
│   ├── llm/            OpenAI / Ollama / Gemini wrappers.
│   ├── prompt/         Prompt scaffolding (constants + generation).
│   ├── metrics/        Metric definitions, normalisation maps, aggregators, I/O.
│   ├── visuals/        Paper-style plots and plot constants.
│   └── utils/          Config helpers, file I/O, text cleaning.
├── scripts/            Executable CLIs that produce data on disk.
│   ├── prompting/      batch_params → batch_prompt → batch_parse_results
│   ├── annotation/     annotate_responses, annotate_ethnicity, lookup_output
│   ├── factuality/     7-step factuality cascade + run_factuality_pipeline orchestrator
│   ├── ethnicity/      BERT + ethnicolr ethnicity cascade
│   └── metrics/        build_valid_calls, build_ethnicity_distributions
├── notebooks/          Plot-only Jupyter notebooks.
│   ├── analysis/       Paper figures (metrics, ethnicity, OA coverage, manual validation)
│   └── agreement/      Inter-annotator agreement (Cohen κ, Krippendorff α)
└── tests/              pytest suite (structural metrics, coauthorship builder)
```

Run the tests from `code/` with `PYTHONPATH=.`:

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

---

## Pipeline overview

```
       prompting                  factuality                      metrics             notebooks
 ┌────────────────┐    ┌─────────────────────────────────┐    ┌──────────────┐    ┌──────────────┐
 │ batch_params   │    │ factuality_author_jw            │    │ build_valid_ │    │ analysis/    │
 │ batch_prompt   │ →  │ → factuality_openalex           │ →  │   calls      │ →  │ metrics_     │
 │ batch_parse_   │    │   → factuality_field_check      │    │ build_ethni- │    │ pipeline     │
 │ results        │    │     → factuality_seniority      │    │   city_dist- │    │              │
 └────────────────┘    │       → factuality_location     │    │   ributions  │    │ ethnicity_   │
       │                │         → factuality_affiliation│    └──────────────┘    │ metrics      │
       ↓                │           → factuality_ethnicity│           │            │ ...          │
  ../results/           └─────────────────────────────────┘           ↓            └──────────────┘
  summary/                          │                          ../results/                │
  recommendations.csv               ↓                          factualities/               ↓
  summary.csv                  ../results/                     tables/                ../results/
                               summary/                        valid_requests_        plots/
                               factuality_full.csv             metadata.csv           *.pdf

                               ethnicity (optional, in parallel):
                               scripts/ethnicity/ethnicity_inference.py
                               scripts/ethnicity/apply_ethnicity_ground_truth.py
                               → ../results/ethnicity/, ../results/summary/recommendations_with_ethnicity.csv
```

The seven factuality steps run in order — each reads the output of the previous one.

| # | Script | Input → Output | What it adds |
|---|---|---|---|
| 0   | `factuality_author_jw.py`    | `recommendations.csv` → `factuality_author_jw.csv` | `author_status`, `gt_career_age`, `gt_field`, `gt_gender`, JW sim columns |
| 0.5 | `factuality_openalex.py`     | `factuality_author_jw.csv` → `factuality_oa.csv` | `oa_id`, `oa_country_code`, `oa_works_count`, `oa_cited_by_count`, etc. (12 cols) |
| 1   | `factuality_field_check.py`  | `factuality_oa.csv` → `factuality_field.csv` | `field_status`, `field_check_source`, `field_evidence` |
| 2   | `factuality_seniority.py`    | `factuality_field.csv` → `factuality_seniority.csv` | `seniority_career_age`, `seniority_bucket`, `seniority_status` |
| 3   | `factuality_location.py`     | `factuality_seniority.csv` → `factuality_location.csv` | `location_llm_iso`, `location_oa_iso`, `location_status` |
| 3.5 | `factuality_affiliation.py`  | `factuality_location.csv` → `factuality_affiliation.csv` | `affiliation_llm`, `affiliation_oa_all`, `affiliation_status` |
| 4   | `factuality_ethnicity.py`    | `factuality_affiliation.csv` → `factuality_full.csv` | `perceived_ethnicity` |

---

## How to run

### Setup (once)

```bash
# Install dependencies
pip install -r requirements.txt
# (optional) ethnicity inference cascade:
pip install tensorflow tf-keras ethnicolr

# Configure local paths
cp config.ini.example config.ini
$EDITOR config.ini   # fill in [secrets].keys_dir and every [data].* path
```

Every script is launched from `code/` with `PYTHONPATH=.` so `libs.*` imports resolve:

```bash
cd code
export PYTHONPATH=.
```

### Step by step

#### 1. Prompting — generate prompts and parse LLM responses

```bash
# 1a. Translate parameters into each language.
python scripts/prompting/batch_params.py -l english -o ../data/context/
python scripts/prompting/batch_params.py -l spanish -o ../data/context/
python scripts/prompting/batch_params.py -l german  -o ../data/context/

# 1b. Expand all (role × task × target × language × location × k × field × subfield) combinations.
python scripts/prompting/batch_prompt.py -l english -o ../data/context/
python scripts/prompting/batch_prompt.py -l spanish -o ../data/context/
python scripts/prompting/batch_prompt.py -l german  -o ../data/context/

# 1c. Hit the LLMs (out of scope here — drop JSON responses under ../results/responses/).

# 1d. Consolidate the JSON responses into a single recommendations.csv + summary.csv.
python scripts/prompting/batch_parse_results.py \
  --results_dir ../results \
  --output_dir ../results/summary

# Parallel variant — one model × language per worker:
parallel -j 20 python scripts/prompting/batch_parse_results.py \
  --results_dir ../results \
  --output_dir ../results/summary_parallel \
  --model {1} --language {2} \
  :::: ../data/context/models.txt ::: english german spanish
```

**Outputs:**
- `../results/summary/recommendations.csv` — one row per LLM-recommended author (~3.9M rows).
- `../results/summary/summary.csv` — one row per LLM call (~929K rows) with `valid_flag`.

#### 2. Factuality pipeline — 7 ordered steps

```bash
# Orchestrator (recommended) — runs every step in order, reads paths from config.ini.
python scripts/factuality/run_factuality_pipeline.py \
  --results ../results/summary \
  --parquet <path_to_ss_parquet> \
  --duckdb  <path_to_oa_duckdb>

# Skip individual steps if their outputs already exist:
python scripts/factuality/run_factuality_pipeline.py --skip_jw --skip_oa --skip_field
```

Or call each step individually (every step accepts `--help`):

```bash
python scripts/factuality/factuality_author_jw.py    --recommendations ../results/summary/recommendations.csv     --parquet <path_to_ss_parquet> --output ../results/summary/factuality_author_jw.csv
python scripts/factuality/factuality_openalex.py     --input ../results/summary/factuality_author_jw.csv          --db_path <path_to_oa_duckdb> --output ../results/summary/factuality_oa.csv
python scripts/factuality/factuality_field_check.py  --input ../results/summary/factuality_oa.csv                 --output ../results/summary/factuality_field.csv
python scripts/factuality/factuality_seniority.py    --input ../results/summary/factuality_field.csv              --output ../results/summary/factuality_seniority.csv
python scripts/factuality/factuality_location.py     --input ../results/summary/factuality_seniority.csv          --output ../results/summary/factuality_location.csv
python scripts/factuality/factuality_affiliation.py  --input ../results/summary/factuality_location.csv           --output ../results/summary/factuality_affiliation.csv
python scripts/factuality/factuality_ethnicity.py    --input ../results/summary/factuality_affiliation.csv        --ethnicity_lookup ../results/summary/recommendations_with_ethnicity.csv --output ../results/summary/factuality_full.csv
```

**Output:** `../results/summary/factuality_full.csv` — same rows as `recommendations.csv`, plus all factuality columns added by every step.

#### 3. Ethnicity inference (optional, runs in parallel to factuality)

```bash
# 3a. Infer perceived ethnicity for every researcher in the ground-truth field CSVs.
python scripts/ethnicity/apply_ethnicity_ground_truth.py

# 3b. Apply the cascade to the LLM recommendations themselves.
python scripts/ethnicity/ethnicity_inference.py \
  --input  ../results/summary/recommendations.csv \
  --output ../results/summary/recommendations_with_ethnicity.csv
```

Cascade: BERT `liamliang/demographics_race_v2` (confidence ≥ 0.5) → `ethnicolr` LSTM (confidence ≥ 0.5) → `Unknown`. Categories: White / Asian / Hispanic or Latino / Black or African American / Unknown.

**Outputs:**
- `../results/ethnicity/researcher_ethnicity_lookup.csv` — unique researcher → ethnicity.
- `../results/ethnicity/DataFrameRankings_Genderize_Namsor_<Field>_with_ethnicity.csv` — per-field GT CSVs.
- `../results/summary/recommendations_with_ethnicity.csv` — LLM recs with `perceived_ethnicity`.

#### 4. Annotation (optional, manual)

Interactive CLIs used to produce the manual labels shipped under `data/annotator_agreement/` and `data/ethnicity_inference/`:

```bash
python scripts/annotation/annotate_responses.py \
  --results_dir ../results --summary_csv ../results/summary/summary.csv \
  --output ../data/annotator_agreement/manual_labels_annotator2.csv \
  --n 100

python scripts/annotation/annotate_ethnicity.py \
  --lookup ../results/ethnicity/researcher_ethnicity_lookup.csv \
  --output ../data/ethnicity_inference/manual_labels_annotator1.csv \
  --sample_csv ../data/ethnicity_inference/sample_100.csv

# Inspect a single LLM response by index:
python scripts/annotation/lookup_output.py 42 \
  --results_dir ../results --summary_csv ../results/summary/summary.csv
```

#### 5. Metrics — pre-compute the analysis tables

```bash
# 5a. Build the per-call metrics table consumed by every plotting notebook.
python scripts/metrics/build_valid_calls.py
# → ../results/factualities/tables/valid_requests_metadata.csv

# 5b. Build the GT/rec ethnicity distributions used by ethnicity_metrics.ipynb.
python scripts/metrics/build_ethnicity_distributions.py
# → ../results/ethnicity/distributions/{gt_overall,gt_per_field,rec_overall,rec_per_field,rec_per_model}.csv
```

Both scripts read their defaults (`ss_parquet`, `results_dir`) from `[data]` in `config.ini`.

##### Structural metrics: connectedness & similarity (paper Eqs. 6-8)

`build_valid_calls.py` also derives the two structural metrics. They are the only metrics
that need the **OpenAlex snapshot** (`[data].oa_duckdb`) rather than just `factuality_full.csv`,
because one of them needs a coauthorship graph. Both are computed over `Û_i`, the set of
unique *factual* authors of a response.

| Metric | What it measures | Definition |
|---|---|---|
| `connectedness` | Whether the recommended authors form one collaborating group or scattered individuals | `1 − NormEntropy` over the connected components of the induced subgraph `G[Û_i]` |
| `similarity` | Whether the recommended authors have similar academic profiles | Mean pairwise cosine of PCA embeddings of 5 author features |

Two cached artefacts are built **once** and reused (`--rebuild_structural` forces a rebuild):

| Artefact | Cost | Cached at |
|---|---|---|
| Coauthorship graph (CSR + `author_id → index`) | **~2.5 h** — explodes the `authorships` of all 492M works, chunked by `works.id` quantiles to bound memory | `<results_dir>/.cache/coauthorship_graph.joblib` |
| Scaler + PCA + author embeddings | seconds | `<results_dir>/.cache/similarity_embeddings.joblib` |

The graph is restricted to the benchmark's ~306k factual authors. That is exact, not an
approximation: every `Û_i` is a subset of that population and induced subgraphs compose, so
`G[Û_i] = (G[population])[Û_i]`. It keeps the matrix at 306k² sparse instead of 113M².

The scaler and PCA are fitted **once** over the whole author population and only *applied*
per response. This departs from a literal reading of the paper (which fits inside each
response) and is deliberate: with n≈10 authors a per-response PCA is unstable and the values
stop being comparable across models — see the docstring of `build_similarity_embeddings`.

Four columns reach the output CSV per metric pair:

```
connectedness, similarity                            the metrics (NaN where undefined)
n_used_connectedness, n_used_similarity              the n that entered each formula
n_excluded_connectedness, n_excluded_similarity      Û_i members dropped (no oa_id / no features)
```

**Both are NaN when fewer than 2 authors are evaluable** — connectedness because `log n = 0`
at n=1, similarity because there is no pair. No default value is invented. In practice this
makes them undefined for ~63% of responses, for two structural reasons: `k=1` cannot produce
a pair at all, and authors matched only in Semantic Scholar have no `oa_id`, hence no graph
node and no features (~13% of factual authors, reported in `n_excluded_*`).

Use `n_used_*` as the denominator, **not** `n_authors_found`: the latter collapses every
author without an `oa_id` into a single entry, so subtracting the two goes negative.

#### 6. Notebooks — plot the figures

```bash
cd notebooks/analysis
jupyter nbconvert --execute metrics_pipeline.ipynb
jupyter nbconvert --execute ethnicity_metrics.ipynb
jupyter nbconvert --execute manual_classification_metrics.ipynb
jupyter nbconvert --execute oa_productivity_coverage.ipynb
# Figures land under ../results/plots/.
```

#### 7. Inter-annotator agreement (optional)

```bash
cd notebooks/agreement
jupyter nbconvert --execute inter_annotator_agreement.ipynb
jupyter nbconvert --execute inter_annotator_agreement_ethnicity.ipynb
jupyter nbconvert --execute manual_factuality_validation.ipynb
```

### Run it all at once

For a full clean run, chain the orchestrators:

```bash
cd code
export PYTHONPATH=.
python scripts/prompting/batch_parse_results.py --results_dir ../results --output_dir ../results/summary
python scripts/factuality/run_factuality_pipeline.py --results ../results/summary
python scripts/ethnicity/apply_ethnicity_ground_truth.py        # optional, slow
python scripts/ethnicity/ethnicity_inference.py                 # optional, slow
python scripts/metrics/build_valid_calls.py
python scripts/metrics/build_ethnicity_distributions.py
( cd notebooks/analysis  && jupyter nbconvert --execute *.ipynb )
( cd notebooks/agreement && jupyter nbconvert --execute *.ipynb )
```

---

## Plots and notebooks

| Notebook | Inputs | What it produces |
|---|---|---|
| `analysis/metrics_pipeline.ipynb` | `factualities/tables/valid_requests_metadata.csv` | Per-call metric panels by language / location / role / task / field / subfield / k / seniority / model / model family — every figure in the paper's main results section. PDFs under `../results/plots/`. |
| `analysis/ethnicity_metrics.ipynb` | `ethnicity/distributions/*.csv` | Ground-truth vs. LLM-recommendation ethnicity distributions: overall, per field, per model, side-by-side. |
| `analysis/manual_classification_metrics.ipynb` | `../data/annotator_agreement/manual_labels_annotator2.csv` + `summary/summary.csv` | Confusion matrix and accuracy/precision/recall/F1 of the algorithmic `valid_flag` against manual labels. |
| `analysis/oa_productivity_coverage.ipynb` | `summary/factuality_full.csv` | OpenAlex coverage diagnostics for productivity tiers (low / med / high `works_count` and `cited_by_count`). |
| `agreement/inter_annotator_agreement.ipynb` | `../data/annotator_agreement/manual_labels_annotator{1,2}.csv` | Cohen κ and Krippendorff α between two annotators on response validity. |
| `agreement/inter_annotator_agreement_ethnicity.ipynb` | `../data/ethnicity_inference/manual_labels_annotator{1,2}.csv` | Cohen κ and Krippendorff α between two annotators on perceived ethnicity. |
| `agreement/manual_factuality_validation.ipynb` | `factuality_full.csv`, `<path_to_ss_parquet>`, `<path_to_oa_duckdb>` | Interactive validator: sample 20 personas, fetch the underlying response JSONs, render SS / OA / LLM side-by-side. |

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'libs'`** — you forgot to set `PYTHONPATH`. From `code/`: `export PYTHONPATH=.`.
- **`KeyError: 'Missing [data].ss_parquet in config.ini'`** — copy `config.ini.example` to `config.ini` and fill in the `[data]` paths.
- **`FileNotFoundError: ../results/summary/factuality_full.csv`** — the factuality pipeline hasn't run yet, or you point `--results` at the wrong dir.
- **A notebook fails on the first data-loading cell** — run the corresponding `scripts/metrics/build_*.py` first; notebooks are plotting-only.
- **`from openai import OpenAI` fails** — install the OpenAI client: `pip install openai` (already in `requirements.txt`).
- **`grouped_metrics` import error** — make sure the renamed module is on the path; if you imported as `libs.visuals.grouped_metrics_leen`, update to `libs.visuals.grouped_metrics`.
- **DuckDB / parquet temp space errors** — set `[data].oa_works_tmp_dir` in `config.ini` to a directory with plenty of free space (the OpenAlex enrichment step writes ~20 chunk parquets there).
