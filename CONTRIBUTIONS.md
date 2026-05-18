# Contributions — LLMScholar-Personas

Detailed documentation of the pipeline parts I worked on: **automatic classification**, **manual classification**, **inter-annotator agreement**, **ethnicity**, **factuality**, and **enrichment**. Each section lists the scripts, their CLI arguments, input/output columns, thresholds, and key logic.

> Path convention: all paths are relative to the repo root (`/home/asanchez/code/asanchez/LLMScholar-Personas`).

---

## 0. Pipeline order

```
LLM responses (raw JSON)
        │
        ▼
[1] Automatic classification   ──► summary.csv + recommendations.csv
        │
        ├──► [2] Manual classification (human annotation on a sample)
        │             │
        │             ▼
        │       [3] Inter-annotator agreement
        │
        ▼
[4] Enrichment (OpenAlex)       ──► oa_country_code, institution, h-index
        │
        ▼
[5] Factuality (5 steps)        ──► factuality_full.csv
        │
        ▼
[6] Ethnicity (inference + NamSor cross-check)
```

---

## 1. Automatic classification (LLM-response parser)

Turns the raw LLM output (free text, sometimes malformed JSON, sometimes a refusal) into two structured CSVs.

### 1.1 Scripts and libraries

| File | Purpose |
|------|---------|
| `code/scripts/prompting/batch_parse_results.py` | Orchestrator. Walks all responses and produces `summary.csv` + `recommendations.csv` |
| `code/libs/utils/text.py` | `clean_content` (markdown / unicode / diacritics cleanup) and `parse_valid_dicts` (recover valid dicts from a truncated JSON) |
| `code/libs/utils/discover_keys.py` | Utility for discovering new *wrapper keys* in unseen responses |
| `code/libs/utils/constants.py` | `INSTRUCTIONS`, `LOCATIONS`, `INPUTS`, EN/ES/DE prompts, `REFUSAL_KEYWORDS`, `OUTPUT_*` flags |

### 1.2 CLI

```bash
python code/scripts/prompting/batch_parse_results.py --results_dir results/responses --output_dir results/summary [--model gpt-4o] [--language english]
```

Args:
- `--results_dir`: root with `results_<SOURCE>_<LANGUAGE>/` subfolders (source ∈ `gemini` | `gpt` | `ollama`)
- `--output_dir`: output
- `--model`, `--language`: optional filters

### 1.3 Outputs

**`summary.csv`** — one row per `run_id`:

```
role, task, location, k, target, field, subfield, language, model, run_id,
done, done_reason, prompt_eval_count, eval_count, eval_duration,
reasoning_tokens, response_role, response_content, response_thinking,
error_message, valid_flag
```

**`recommendations.csv`** — exploded from `response_content` (each dict in the array → one row):

```
<all summary cols> + name, lastname, current_affiliations,
areas_of_research_or_work, reason, source
```

### 1.4 `valid_flag` flags (constants.py:138-143)

| Flag | Meaning |
|------|---------|
| `unchanged` | Valid JSON on the first try |
| `cleaned` | Valid JSON after cleaning markdown / unicode / quotes (`clean_content`) |
| `fixed_dict` | Truncated JSON: parseable dicts were recovered one by one (`parse_valid_dicts`) |
| `invalid` | Structure present but not parseable / empty list / placeholders |
| `empty` | Empty response |
| `refused` | Model refused (detection by length `<300` chars without JSON structure, or by keyword in `REFUSAL_KEYWORDS`) |

`valid` group = {`unchanged`, `cleaned`, `fixed_dict`}. `invalid` group = {`invalid`, `empty`, `refused`}.

### 1.5 Key parser details

- **Per-provider parsers** (`batch_parse_results.py:723`, `:797`, `:886`): `_parse_ollama`, `_parse_gemini`, `_parse_gpt`.
- **Wrapper keys** (`:24-610`): list of ~600 keys (`candidates`, `professors`, `advisors`, `results`, …) used to unwrap responses that wrap the array in a dict (`{"candidates": [...]}`).
- **Short-refusal threshold** (`:21`): if the text has `<300` chars and contains no `[`/`{`, it is marked `refused`.
- **Post-processing** (`:641-701`): error-key detection, single-dict unwrapping, empty / placeholder-filled lists marked as `invalid`.

### 1.6 `clean_content` (`code/libs/utils/text.py:8`)

Cleanup pipeline, in this order:

1. Extract block from ` ```json … ``` ` fences if present.
2. Normalize umlauts (`ä → a¨` and back).
3. Decode `raw_unicode_escape`.
4. Strip diacritics via `unicodedata.normalize("NFD", …)`.
5. Replace typographic quotes with ASCII.
6. Return `(cleaned, OUTPUT_CLEANED | OUTPUT_UNCHANGED)`.

### 1.7 `parse_valid_dicts` (`code/libs/utils/text.py:44`)

For truncated JSON: splits the list on top-level commas (respecting bracket depth), tries `ast.literal_eval` on each item, and returns the ones that parse. Backs the `fixed_dict` flag.

### 1.8 `discover_keys.py`

Utility to find new wrapper keys that are not in the hardcoded list:

```bash
python code/libs/utils/discover_keys.py --results_dir results/responses --model gpt-4o --language english
```

Filter: keys matching `^[a-z][a-z0-9_]*$` with length ≤40 (`:64-69`). Compares against `KNOWN_KEYS` and highlights new ones.

---

## 2. Manual classification (annotation CLI)

Human labeling over a stratified sample to audit the automatic parser.

### 2.1 `code/scripts/annotation/annotate_responses.py`

**Labels** (8 keys):

| Key | Label | Group |
|-----|-------|-------|
| `c` | `cleaned` | valid |
| `u` | `unchanged` | valid |
| `f` | `fixed_dict` | valid |
| `i` | `invalid` | invalid |
| `e` | `empty` | invalid |
| `r` | `refused` | invalid |
| `s` | `skip` | (excluded) |
| `q` | `quit` | (close) |

**CLI**:

```bash
python code/scripts/annotation/annotate_responses.py --results_dir results/responses --summary_csv results/summary/summary.csv --output data/annotator_agreement/manual_labels.csv --n 100 --stratified --seed 42
```

Main args:
- `--n`: sample size (default 100)
- `--stratified`: stratify by `valid_flag` so each flag has enough weight
- `--model`, `--language`: pre-sampling filters
- `--export_sample` + `--sample_csv`: export the chosen sample to share with another annotator (key for reproducible IAA)
- `--show_algo_label`: show the parser's label during annotation (review only; biases IAA unless agreed otherwise)

**Output** (`manual_labels.csv`):

```
original_index, model, language, role, task, location, k, target, field,
subfield, run_id, valid_flag, manual_label
```

**Resumable**: the script detects already-labeled rows and continues from where it left off.

**Metrics printed on exit** (`:281-327`): exact accuracy over 6 classes, binary valid/invalid accuracy, and sklearn `classification_report` (precision/recall/F1 per class) comparing `manual_label` vs algorithmic `valid_flag`. Also `confusion_matrix`.

### 2.2 `code/scripts/annotation/annotate_ethnicity.py`

Human ethnicity annotation over the lookup already produced by the BERT→ethnicolr cascade.

**Keys**:

| Key | Label |
|-----|-------|
| `a` | Asian |
| `w` | White |
| `b` | Black or African American |
| `h` | Hispanic or Latino |
| `u` | Unknown |
| `s` | skip |
| `q` | quit |

**CLI**:

```bash
python code/scripts/annotation/annotate_ethnicity.py --lookup data/ethnicity_inference/researcher_ethnicity_lookup.csv --output data/annotator_agreement/manual_labels_ethnicity.csv --n 100 --seed 42
```

Stratifies by automatic `perceived_ethnicity`, shows a confidence bar, and computes accuracy against the automatic prediction.

### 2.3 `code/scripts/annotation/lookup_output.py`

Utility to inspect the raw response for a given `original_index` during annotation (debug):

```bash
python code/scripts/annotation/lookup_output.py 42 --results_dir results/responses --summary_csv results/summary/summary.csv
```

---

## 3. Inter-annotator agreement (IAA)

Notebooks: `code/notebooks/agreement/inter_annotator_agreement.ipynb` (response validity) and `code/notebooks/agreement/inter_annotator_agreement_ethnicity.ipynb` (ethnicity).

### 3.1 Three measures

1. **Raw agreement** `p = #matches / N`.
2. **Cohen's κ** — pairwise agreement corrected for chance:

$$ \kappa = \frac{p_o - p_e}{1 - p_e} $$

3. **Krippendorff's α** — nominal multi-annotator reliability (more robust to missing data and to >2 annotators).

Rows with `skip` are excluded from the computation.

### 3.2 Results — response validity

Data: 100 items, annotators Leen vs v6 (`data/annotator_agreement/manual_labels_v1_leen.csv` vs `manual_labels_v6.csv`).

| Metric | Value |
|--------|-------|
| Raw agreement | 0.8600 (86/100) |
| Cohen's κ | 0.8288 |
| Krippendorff's α | 0.8296 |

Distribution (Leen | v6): `unchanged` 23/20, `cleaned` 23/25, `refused` 18/17, `empty` 17/16, `fixed_dict` 9/12, `invalid` 10/10.

The 14 disagreements cluster on adjacent categories (`cleaned ↔ unchanged`, `fixed_dict ↔ cleaned`), with few far jumps (`refused ↔ cleaned`, `empty ↔ invalid`).

### 3.3 Results — ethnicity

Data: 100 items, annotators Leen vs v2 (`manual_labels_v1_leen.csv` vs `manual_labels_v2.csv`).

| Metric | Value |
|--------|-------|
| Raw agreement | 0.6400 (64/100) |
| Cohen's κ | 0.4874 |
| Krippendorff's α | 0.4747 |

Distribution (Leen | v2): White 62/39, Hispanic or Latino 14/17, Asian 13/11, Unknown 6/13, Black or African American 5/20. *Moderate* agreement: Leen over-assigns White; v2 assigns much more Black. 36 disagreements.

> Interpretation: low IAA on ethnicity is expected — name-based ethnic classification is ambiguous for humans and the ground truth is subjective. This justifies reporting `factuality_ethnicity` carefully and preferring NamSor cross-verification (§6.4).

---

## 4. Enrichment (OpenAlex)

Resolves each recommended author to an `oa_id` and enriches with country, institution, and h-index. Two routes: API and DuckDB.

### 4.1 `code/scripts/enrichment/enrich_via_api.py`

Serial route using the public OpenAlex API (polite pool, batches of 50 via the `ids.openalex:A1|…|A50` filter).

```bash
python code/scripts/enrichment/enrich_via_api.py --input results/summary/factuality_author.csv --output results/summary/oa_enrichment.csv --email <your_email>
```

- **Retries**: 429/5xx with exponential backoff `2^attempt × 2 s`, max 4 attempts.
- **Resume-aware**: skips `oa_id`s already in the output.
- **Checkpoint**: every 50 batches.
- **Extraction**: `last_known_institutions[0]` (fallback to `last_known_institution`) → `oa_country_code` and `oa_last_institution`.

Output `oa_enrichment.csv`: `oa_id, oa_country_code, oa_last_institution`.

### 4.2 `code/scripts/enrichment/enrich_via_api_parallel.py`

Same contract but with `ThreadPoolExecutor` (default 10 workers, one `requests.Session` per worker, shared lock-guarded dict). Checkpoint every 100 batches.

```bash
python code/scripts/enrichment/enrich_via_api_parallel.py --input results/summary/factuality_author.csv --output results/summary/oa_enrichment.csv --email <your_email> --workers 10
```

### 4.3 `code/scripts/enrichment/enrich_from_works.py`

When `last_known_institution` is empty in OpenAlex's `authors` table (the usual case), country is derived from the *most recent paper* over the `works` table.

```bash
python code/scripts/enrichment/enrich_from_works.py --input results/summary/factuality_author.csv --db_path /data/asanchez/openalex.duckdb --output results/summary/oa_enrichment.csv
```

- DuckDB stream of `(oa_id, country, year)` via `UNNEST(authorships) × UNNEST(institutions)`.
- Python-side filtering on the `oa_id` set (a DuckDB `SEMI JOIN` materializes post-UNNEST and blows memory).
- Deduplication: `ARG_MAX` by `publication_year` → most recent paper per author.
- Tuning: `memory_limit=96GB`, `temp_directory=/data/asanchez/duckdb_enrich`.
- `--benchmark`: limit to 5 M works to estimate ETA.

### 4.4 `code/scripts/enrichment/apply_enrichment.py`

Merges `oa_enrichment.csv` into `factuality_author.csv`. **Does not overwrite** existing values — only fills `oa_country_code` where empty.

```bash
python code/scripts/enrichment/apply_enrichment.py --input results/summary/factuality_author.csv --enrichment results/summary/oa_enrichment.csv --output results/summary/factuality_author_enriched.csv
```

### 4.5 `code/scripts/enrichment/merge_h_index.py`

Standalone (no argparse). Reads hardcoded paths: `results/summary/factuality_full.csv` + `results/summary/oa_h_index.csv`, merges `oa_h_index`, reorders the column next to `oa_cited_by_count`, and rewrites `factuality_full.csv`. Prints hit rate and quantiles (.25, .5, .75, .9, .99).

---

## 5. Factuality — 5-step pipeline

Orchestrator: `code/scripts/factuality/run_factuality_pipeline.py`.

```bash
python code/scripts/factuality/run_factuality_pipeline.py --results results/summary --parquet data/researchers/Researchers_Deduplicated_Genderize_Namsor.parquet --duckdb /data/asanchez/openalex.duckdb
```

Skip flags: `--skip_jw`, `--skip_oa`, `--skip_field`. Every step **only adds columns** — never drops rows.

### 5.1 Step 0 — `factuality_author_jw.py`

Jaro-Winkler matching against ground truth (Researchers parquet).

- **Input**: `recommendations.csv` + parquet (`Researcher_id, Name, Field, Combined_gender, First_year, Citations`).
- **Output**: `factuality_author_jw.csv` with added columns:
  - `original_name`, `matched_name`, `researcher_id`, `match_score` (0-7), `matched_fields`, `author_status` (`found` | `hallucinated`)
  - `gt_field`, `gt_gender`, `gt_career_age`, `gt_citations`
  - `sim_*` (7 raw Jaro-Winkler scores)

- **Thresholds** (`:44-52`):
  - `display_name ≥ 0.85` (worth 2 points)
  - `first_name`, `last_name`, `second_name`, `dn_vs_last`, `dn_vs_first`, etc. `≥ 0.70` (1 point each)
  - Max possible score = 7
  - `--min_matches` (default 5): score ≥5 → `found`; <5 → `hallucinated`

- **Vectorization**: uses `rapidfuzz.process.cdist` in chunked blocks to cap matrix size at ~200 MB; ~20-50× faster than per-record looping.
- **Block-index cache** (`:92-167`): indexes the parquet by the first 2 letters of the last name; pickle cached with MD5 hash of parquet `(size, mtime)`.

### 5.2 Step 0.5 — `factuality_openalex.py`

OpenAlex resolution (local DuckDB or API), 3 stages in order:

1. **Exact match** (`:169-227`): normalized scan against `display_name` + `alternatives`; ties broken by `cited_by_count` desc.
2. **JW fuzzy** (`:306-369`) for unresolved names:
   - JW ≥ 0.85 on full name
   - **AND** JW ≥ 0.80 on first token
   - **AND** exact match on last token (surname)
   - Tie-break by `cited_by_count`.
3. **Works scan** (`:501-555`) for resolved `oa_id`s: aggregates country + institution from the most recent paper. Year chunked in 20 segments. Cache in `WORKS_AGG_DIR=/data/asanchez/duckdb_enrich/oa_works_agg_chunks`.

Added columns: `oa_status, oa_id, oa_display_name, oa_works_count, oa_cited_by_count, oa_h_index, oa_i10_index, oa_first_pub_year, oa_last_pub_year, oa_career_age, oa_country_code, oa_last_institution, oa_match_score` (1.0 = exact, <1.0 = JW).

Persistent pickle cache, resume-aware.

### 5.3 Step 1 — `factuality_field_check.py`

Decides `field_status` per row:

| Case | `field_status` |
|------|----------------|
| `author_status == hallucinated` | `not_applicable` |
| `gt_field` present and == `field_LLM` (normalized) | `field_match` |
| `gt_field` present and ≠ `field_LLM` | `field_mismatch` |
| `gt_field` missing | `field_unknown` |

Added columns: `field_status, field_check_source` (`gt` | `openalex_concepts` | `none`), `field_evidence`.

EN/ES/DE → canonical EN field translation (`:49-62`), e.g. *"Ciencias de la computación"* → *"Computer Science"*.

`factuality_field.py` is the variant that looks at the 6 ground-truth CSVs per field (`DataFrameRankings_*_<FIELD>.csv`) with priority lookup: (i) full name in target field, (ii) `surname + initial` in target field, (iii) full name in other fields, (iv) abbreviated in other fields.

### 5.4 Step 2 — `factuality_seniority.py`

Bucketing by career age:

| `career_age` | Bucket |
|--------------|--------|
| ≤ 10 | `Junior` |
| 11–19 | `None` (unclassifiable, excluded) |
| ≥ 20 | `Senior` |

LLM `target` → bucket (`:68-75`): "Junior Professor" / "Profesor Júnior" / "Juniorprofessor(in)" → Junior; "Senior Professor" / "Profesor Sénior" / "Seniorprofessor(in)" → Senior.

Decision:

| Case | `seniority_status` |
|------|--------------------|
| Hallucinated | `not_applicable` |
| LLM bucket == real bucket | `seniority_match` |
| LLM bucket ≠ real bucket | `seniority_mismatch` |
| Missing data / `None` | `seniority_unknown` |

Added columns: `seniority_career_age` (2025 − `First_year`), `seniority_age_source`, `seniority_bucket`, `seniority_llm_bucket`, `seniority_status`.

### 5.5 Step 3 — `factuality_location.py`

LLM country match vs `oa_country_code` (ISO-3166 alpha-2).

Mapping (`:44-57`): Ecuador→EC, Japan→JP, Germany→DE, Canada→CA, South Africa→ZA (with EN/ES/DE variants normalized).

Added columns: `location_llm_country` (raw string), `location_llm_iso`, `location_oa_iso`, `location_oa_institution`, `location_status` (`location_match` | `location_mismatch` | `location_unknown` | `not_applicable`).

### 5.6 Step 4 — `factuality_ethnicity.py`

**Does not re-infer ethnicity** (it's expensive and only depends on the name): looks up against `recommendations_with_ethnicity.csv` by `(name, lastname)`.

Final output: `factuality_full.csv` with column `perceived_ethnicity ∈ {Asian, White, Black or African American, Hispanic or Latino, Unknown}`. Missing → `Unknown`.

---

## 6. Ethnicity — inference and verification

### 6.1 Inference cascade (`code/scripts/ethnicity/ethnicity_inference.py`)

```
demographicx (BERT, conf ≥ 0.5)  →  ethnicolr (LSTM, conf ≥ 0.5)  →  Unknown
```

- **BERT**: `liamliang/demographics_race_v2`, output in order `[white, hispanic, black, asian]` (`:54`).
- **ethnicolr**: surname LSTM (Florida voter registry); mapping `:57-62`: `nh_white → White`, `asian → Asian`, `hispanic → Hispanic or Latino`, `nh_black → Black or African American`, others → `Unknown`.

Public function: `infer_ethnicity_batch(names, batch_size=64)` returns `list[dict]` with `category, confidence, source ∈ {demographicx, ethnicolr, unknown}`.

### 6.2 `apply_ethnicity_ground_truth.py`

Applies the cascade to the 6 ground-truth CSVs `DataFrameRankings_Genderize_Namsor_<FIELD>.csv`.

1. Extract unique `(Researcher_id, Name)` pairs, dedupe by `Researcher_id`.
2. Run `infer_ethnicity_batch` with `batch_size=512`. Checkpoint every 10 batches to `.pkl` (resume-aware).
3. Write `researcher_ethnicity_lookup.csv` (`Researcher_id, Name, perceived_ethnicity, __ethnicity_confidence, __ethnicity_source`).
4. Merge back into each field CSV, chunked at 500 k rows → adds `perceived_ethnicity` column.

### 6.3 `apply_ethnicity_ground_truth_parallel.py` + `ethnicity_worker.py`

Multi-process wrapper around the above:

- Split uniques into N chunks (default 12 workers).
- Each worker runs `ethnicity_worker.py` with `TF_USE_LEGACY_KERAS=1`, threads per worker = `36 // N`.
- Temp spill in `/data/asanchez/duckdb_enrich`.
- Each worker uses `tqdm` at `position=worker_id` so the bars don't overlap.

### 6.4 `namsor_verify.py` — cross-verification

Endpoint: `POST https://v2.namsor.com/NamSorAPIv2/api2/json/usRaceEthnicityBatch`.

```bash
python code/scripts/ethnicity/namsor_verify.py --input data/ethnicity_inference/researcher_ethnicity_lookup.csv --output-jsonl data/ethnicity_inference/namsor_responses.jsonl --output-csv data/ethnicity_inference/namsor_results.csv --sleep 0.5
```

- Deduplicates `(firstName, lastName, countryIso2)`.
- Batch of 100 (NamSor hard limit); 10 credits per name.
- 429/5xx retries with exponential backoff (max 4 attempts).
- Append-only JSONL, resume-aware (skips already-written ids).
- Optional CSV with `raceEthnicity → canonical` mapping: `W_NL → White`, `HL → Hispanic`, `A → Asian`, `B_NL → Black`, `AI_AN | PI → Unknown`.
- ISO-2 country mapping covers the 5 study locations + 15 common countries (`:90-114`).
- `--dry-run`, `--limit`, `--sleep` for responsible credit use.

---

## 7. Quick file map

```
code/
├── libs/
│   ├── metrics/
│   │   ├── aggregators.py        ← aggregation + CIs (Wilson / t)
│   │   ├── constants.py          ← BENCHMARK_*, ETHNICITY_LIST, etc.
│   │   └── io.py
│   ├── prompt/generation.py      ← persona prompts (role × country × lang)
│   ├── utils/
│   │   ├── constants.py          ← INSTRUCTIONS, LOCATIONS, INPUTS, REFUSAL_KEYWORDS
│   │   ├── text.py               ← clean_content, parse_valid_dicts
│   │   ├── discover_keys.py      ← discover new wrapper keys
│   │   └── ios.py / config.py
│   └── visuals/                  ← gridbar, gridcons, grouped_metrics, vis
│
├── scripts/
│   ├── prompting/
│   │   ├── batch_prompt.py          ← generates prompts
│   │   ├── batch_params.py          ← EN→ES/DE translations
│   │   └── batch_parse_results.py   ← [§1] parser (automatic classification)
│   │
│   ├── annotation/
│   │   ├── annotate_responses.py    ← [§2] manual CLI (validity)
│   │   ├── annotate_ethnicity.py    ← [§2] manual CLI (ethnicity)
│   │   └── lookup_output.py         ← debug helper
│   │
│   ├── enrichment/
│   │   ├── enrich_via_api.py        ← [§4.1] OpenAlex API serial
│   │   ├── enrich_via_api_parallel.py ← [§4.2] parallel
│   │   ├── enrich_from_works.py     ← [§4.3] DuckDB works scan
│   │   ├── apply_enrichment.py      ← [§4.4] non-destructive merge
│   │   └── merge_h_index.py         ← [§4.5] adds oa_h_index
│   │
│   ├── factuality/
│   │   ├── run_factuality_pipeline.py  ← [§5] orchestrator
│   │   ├── factuality_author_jw.py     ← [§5.1] step 0
│   │   ├── factuality_openalex.py      ← [§5.2] step 0.5
│   │   ├── factuality_field.py / factuality_field_check.py ← [§5.3] step 1
│   │   ├── factuality_seniority.py     ← [§5.4] step 2
│   │   ├── factuality_location.py      ← [§5.5] step 3
│   │   └── factuality_ethnicity.py     ← [§5.6] step 4
│   │
│   └── ethnicity/
│       ├── ethnicity_inference.py             ← [§6.1] BERT→ethnicolr cascade
│       ├── apply_ethnicity_ground_truth.py    ← [§6.2] serial application
│       ├── apply_ethnicity_ground_truth_parallel.py ← [§6.3] parallel
│       ├── ethnicity_worker.py                ← worker for parallel
│       └── namsor_verify.py                   ← [§6.4] cross-verification
│
└── notebooks/
    ├── agreement/
    │   ├── inter_annotator_agreement.ipynb            ← [§3.2] validity
    │   └── inter_annotator_agreement_ethnicity.ipynb  ← [§3.3] ethnicity
    └── analysis/
        ├── metrics_pipeline.ipynb        ← canonical per-call pipeline
        ├── factuality_metrics.ipynb
        ├── ethnicity_metrics.ipynb
        └── manual_classification_metrics.ipynb
```

---

## 8. Operational notes

- **Resume-aware**: all heavy scripts (`enrich_*`, `factuality_openalex`, `ethnicity_inference`, `namsor_verify`) detect prior work and continue from the last checkpoint. Safe to re-run.
- **Caches**: `factuality_author_jw` caches the parquet block-index under `.cache/`; `factuality_openalex` caches per name in `.pkl`. Delete only if inputs change.
- **DuckDB**: `memory_limit=96GB`, `temp_directory=/data/asanchez/duckdb_enrich`. Adjust before running on another machine.
- **NamSor**: 10 credits/name. Use `--limit` and `--dry-run` before a full run. JSONL is always append-only.
- **Determinism**: all samplings use `--seed 42` by convention.
