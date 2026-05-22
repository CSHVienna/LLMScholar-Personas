# LLMScholar-Personas

Auditing LLMs as recommender systems for people through persona prompting.
The study evaluates how **language**, **country**, and **role** in the
persona prompt affect the *factuality*, *diversity*, *parity*, *consistency*,
and *connectedness* of recommended researchers, using Semantic Scholar and
OpenAlex as ground truth.

---

## Repository structure

```
LLMScholar-Personas/
├── code/
│   ├── scripts/
│   │   ├── prompting/          # batch prompt generation and response parsing
│   │   ├── annotation/         # interactive CLI tools for manual annotation
│   │   ├── factuality/         # 7-step factuality pipeline + orchestrator
│   │   └── ethnicity/          # cascade ethnicity inference (BERT + ethnicolr)
│   ├── libs/
│   │   ├── llm/                # OpenAI wrapper
│   │   ├── metrics/            # aggregators and metric I/O
│   │   ├── prompt/             # prompt generation and combination
│   │   ├── utils/              # config, I/O, text utilities
│   │   └── visuals/            # paper-style plots (panels, grids, scatter)
│   └── notebooks/
│       ├── agreement_v2/       # inter-annotator agreement
│       └── analysis_v2/        # benchmark metrics and exploratory analyses
├── data/                       # per-language contexts, manual labels
├── results/ → /data/datasets/LLMScholar-Personas/results   (symlink)
├── pyproject.toml              # Black + isort configuration
└── config.ini                  # paths to API key files
```

---

## Setup

### Dependencies

Python 3.10 or newer is required.

```bash
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn tqdm rapidfuzz requests duckdb pyarrow openai anthropic torch transformers
```

Ethnicity inference additionally requires:

```bash
pip install tensorflow tf-keras ethnicolr
```

### Environment

All scripts expect `code/libs` to be on the `PYTHONPATH`:

```bash
export PYTHONPATH="$PYTHONPATH:../../libs"
```

### API keys

`config.ini` points to plain-text files containing the keys:

```ini
[secrets]
keys_dir = ../../../.keys/

[openai]
data_dir = ${secrets:keys_dir}/openai_api_key.txt
```

Create `.keys/openai_api_key.txt` with the key on a single line.

---

## Factuality pipeline

The pipeline checks whether recommended authors exist and whether the
attributes assigned by the LLM (field, seniority, location, affiliation,
ethnicity) match ground truth. Each step reads the output of the previous one.

| # | Script | Input → Output |
|---|---|---|
| 0   | `factuality_author_jw.py`    | `recommendations.csv` → `factuality_author_jw.csv` |
| 0.5 | `factuality_openalex.py`     | `factuality_author_jw.csv` → `factuality_oa.csv` |
| 1   | `factuality_field_check.py`  | `factuality_oa.csv` → `factuality_field.csv` |
| 2   | `factuality_seniority.py`    | `factuality_field.csv` → `factuality_seniority.csv` |
| 3   | `factuality_location.py`     | `factuality_seniority.csv` → `factuality_location.csv` |
| 3.5 | `factuality_affiliation.py`  | `factuality_location.csv` → `factuality_affiliation.csv` |
| 4   | `factuality_ethnicity.py`    | `factuality_affiliation.csv` → `factuality_full.csv` |

Run the full pipeline with the orchestrator:

```bash
cd code/scripts/factuality/ && python run_factuality_pipeline.py --results ../../../results/summary_v2 --parquet /data/datasets/LLMScholar-Personas/data/semantic_scholar_data/clean/Researchers_Deduplicated_Genderize_Namsor.parquet --duckdb /data/datasets/LLMScholar-Personas/data/openalex_latest.duckdb
```

Each step accepts `--help`. Individual steps can be skipped via
`--skip_jw`, `--skip_oa`, `--skip_field`.

---

## Other pipelines

### Prompting

`code/scripts/prompting/` contains three scripts:

- `batch_params.py` — generate per-language prompt parameter combinations.
- `batch_prompt.py` — build and inspect individual prompts.
- `batch_parse_results.py` — parse raw LLM responses into
  `recommendations.csv` (consolidated) or per-model/per-language CSVs
  when called with `--model` and `--language`.

### Annotation

`code/scripts/annotation/` contains interactive CLI tools that produce
manual labels for inter-annotator agreement studies:

- `annotate_responses.py` — label LLM responses for validity.
- `annotate_ethnicity.py` — label perceived researcher ethnicity.
- `lookup_output.py` — print the raw LLM output for a `summary.csv` row.

### Ethnicity

`code/scripts/ethnicity/` provides cascade inference
(BERT `liamliang/demographics_race_v2` → `ethnicolr` LSTM → `Unknown`):

- `ethnicity_inference.py` — model loader and inference functions.
- `apply_ethnicity_ground_truth.py` — apply the cascade to a CSV.

---

## Analysis notebooks

| Notebook | Purpose |
|---|---|
| `analysis_v2/metrics_pipeline_leen.ipynb` | Benchmark metrics (Diversity, Parity, Factuality, Consistency, Duplicates) per dimension; produces all paper figures under `factualities_v2/plots/leen/`. |
| `analysis_v2/ethnicity_metrics.ipynb`     | Ground-truth vs. recommendation ethnicity distributions. |
| `analysis_v2/manual_classification_metrics.ipynb` | Manual-validation accuracy/precision/recall. |
| `analysis_v2/oa_productivity_coverage.ipynb` | OpenAlex coverage diagnostics for productivity tiers. |
| `agreement_v2/inter_annotator_agreement.ipynb`           | Cohen κ / Krippendorff α for response validity. |
| `agreement_v2/inter_annotator_agreement_ethnicity.ipynb` | Cohen κ / Krippendorff α for ethnicity labels. |
| `agreement_v2/manual_factuality_validation.ipynb`        | Manual review of factuality pipeline outputs. |

---

## Data sources

- **Semantic Scholar** ground truth: `data/semantic_scholar_data/clean/Researchers_Deduplicated_Genderize_Namsor.parquet`
  (216 MB, deduplicated by researcher).
- **OpenAlex** snapshot: `data/openalex_latest.duckdb` (DuckDB).
- **Manual labels** for inter-annotator agreement: `data/annotator_agreement/` and `data/ethnicity_inference/`.

---

## Code style

The project uses [Black](https://black.readthedocs.io/) (line length 88) and
[isort](https://pycqa.github.io/isort/) with the `black` profile.
Configuration is in `pyproject.toml`. To format the codebase:

```bash
black code/scripts code/libs && isort code/scripts code/libs
```
