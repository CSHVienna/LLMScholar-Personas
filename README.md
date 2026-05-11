# LLMScholar-Personas

Auditoría de LLMs como sistemas recomendadores de personas usando *prompting* con personas. Extiende el paper anterior incorporando **idioma**, **país** y **rol** como variables del persona prompt, y mide *factuality*, *diversity*, *parity*, *consistency* y *connectedness* de las recomendaciones contra Semantic Scholar + OpenAlex como ground truth.

---

## Estructura del proyecto

```
LLMScholar-Personas/
├── code/
│   ├── scripts/              # 5 pipelines ejecutables (CLI)
│   │   ├── prompting/        # generación batch de prompts y parsing de respuestas
│   │   ├── annotation/       # herramientas interactivas de anotación manual
│   │   ├── enrichment/       # enriquecimiento con OpenAlex (API + DuckDB)
│   │   ├── factuality/       # pipeline de chequeo factual (5 pasos)
│   │   └── ethnicity/        # inferencia étnica (BERT + LSTM + verificación NamSor)
│   ├── libs/                 # librerías compartidas
│   │   ├── llm/              # wrapper de OpenAI
│   │   ├── metrics/          # agregadores, constantes y I/O de métricas
│   │   ├── prompt/           # generación y combinación de prompts
│   │   ├── utils/            # config, I/O, texto, descubrimiento de claves
│   │   └── visuals/          # plots (paper-style, paneles, grids)
│   └── notebooks/
│       ├── analysis/         # métricas, factuality, ethnicity, APS demographics
│       ├── agreement/        # inter-annotator agreement
│       └── scratch/          # notebooks exploratorios
├── data/                     # contextos por idioma, etnicidad, anotación manual
├── results/ → /data/datasets/LLMScholar-Personas/results  (symlink)
├── logs/                     # logs de ejecuciones
└── config.ini                # rutas a las API keys
```

---

## Setup

### Dependencias

Instalar las dependencias en tu entorno Python (recomendado 3.10+):

```bash
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn tqdm rapidfuzz requests duckdb pyarrow openai anthropic torch transformers
```

Para *ethnicity inference* se requiere además `tensorflow` (legacy keras), `ethnicolr`:

```bash
pip install tensorflow tf-keras ethnicolr
```

### Variables de entorno

Casi todos los scripts esperan que `code/libs` esté en el `PYTHONPATH`. Desde cualquier `code/scripts/<pipeline>/`:

```bash
export PYTHONPATH="$PYTHONPATH:../../libs"
```

### API Keys

El archivo `config.ini` apunta a los archivos planos con las claves:

```ini
[secrets]
keys_dir = ../../../.keys/

[openai]
data_dir = ${secrets:keys_dir}/openai_api_key.txt

[namsor]
data_dir = ${secrets:keys_dir}/namsor_api_key.txt
```

Crear `.keys/openai_api_key.txt` y `.keys/namsor_api_key.txt` (un archivo por servicio, con la clave en una sola línea). La variable de entorno `NAMSOR_API_KEY` también funciona para NamSor.

---

## Pipelines y comandos importantes

Todos los comandos están en **una sola línea** para copy-paste directo. Cada uno asume que estás en `code/scripts/<subcarpeta>/` salvo que se diga otra cosa.

### 1. Prompting (`code/scripts/prompting/`)

Genera, distribuye y consolida las salidas del LLM.

```bash
cd code/scripts/prompting/ && export PYTHONPATH="$PYTHONPATH:../../libs" && python batch_params.py -l english -o ../../data/context/
```

```bash
cd code/scripts/prompting/ && export PYTHONPATH="$PYTHONPATH:../../libs" && python batch_params.py -l spanish -o ../../data/context/
```

```bash
cd code/scripts/prompting/ && export PYTHONPATH="$PYTHONPATH:../../libs" && python batch_params.py -l german -o ../../data/context/
```

```bash
cd code/scripts/prompting/ && export PYTHONPATH="$PYTHONPATH:../../libs" && python batch_prompt.py -c 0 -l english -o ../../data/context/
```

```bash
parallel -j 8 python batch_prompt.py -c {} -l german -o ../../data/context/ ::: {0..719}
```

```bash
nice -n 10 parallel -j 20 python batch_parse_results.py --results_dir ../../../results --output_dir ../../../results/summary_parallel --model {1} --language {2} :::: ../../../data/context/models.txt ::: english german spanish
```

### 2. Annotation (`code/scripts/annotation/`)

CLIs interactivas para anotación manual; al terminar imprimen accuracy/precision/recall/F1 contra la etiqueta algorítmica.

```bash
python annotate_responses.py --results_dir ../../../results --summary_csv ../../../results/summary/summary.csv --output ../../../results/manual_labels.csv --n 100 --stratified --seed 42
```

```bash
python annotate_ethnicity.py --lookup ../../../results/ethnicity/researcher_ethnicity_lookup.csv --output ../../../data/ethnicity_inference/manual_labels_v1.csv --sample_csv ../../../data/ethnicity_inference/sample_100.csv --n 100 --seed 42
```

### 3. Enrichment (`code/scripts/enrichment/`)

Resuelve `oa_id → (country_code, institution)` para autores recomendados, usando el snapshot DuckDB local o el API público de OpenAlex.

```bash
python enrich_from_works.py --input ../../../results/summary/factuality_author.csv --db_path /data/datasets/LLMScholar-Personas/data/openalex_latest.duckdb --output ../../../results/summary/oa_enrichment.csv
```

```bash
python enrich_via_api.py --input ../../../results/summary/factuality_author.csv --output ../../../results/summary/oa_enrichment.csv --email you@example.com
```

```bash
python enrich_via_api_parallel.py --input ../../../results/summary/factuality_author.csv --output ../../../results/summary/oa_enrichment.csv --email you@example.com --workers 10
```

```bash
python apply_enrichment.py --input ../../../results/summary/factuality_author.csv --enrichment ../../../results/summary/oa_enrichment.csv --output ../../../results/summary/factuality_author_enriched.csv
```

### 4. Factuality (`code/scripts/factuality/`) — pipeline de 5 pasos

Verifica si los autores recomendados son reales y si los atributos (campo, seniority, ubicación, etnicidad) que el LLM les asigna coinciden con el ground truth. Cada paso lee la salida del anterior.

**Pipeline completo (recomendado):**

```bash
cd code/scripts/factuality/ && python run_factuality_pipeline.py --results ../../../results/summary --parquet /data/datasets/LLMScholar-Personas/data/semantic_scholar_data/clean/Researchers_Deduplicated_Genderize_Namsor.parquet --duckdb /data/datasets/LLMScholar-Personas/data/openalex_latest.duckdb
```

**Pasos individuales:**

```bash
python factuality_author_jw.py --recommendations ../../../results/summary/recommendations.csv --parquet /data/datasets/LLMScholar-Personas/data/semantic_scholar_data/clean/Researchers_Deduplicated_Genderize_Namsor.parquet --output ../../../results/summary/factuality_author_jw.csv
```

```bash
python factuality_openalex.py --input ../../../results/summary/factuality_author_jw.csv --output ../../../results/summary/factuality_oa.csv --db_path /data/datasets/LLMScholar-Personas/data/openalex_latest.duckdb --cache ../../../results/summary/.oa_cache.pkl
```

```bash
python factuality_field_check.py --input ../../../results/summary/factuality_oa.csv --output ../../../results/summary/factuality_field.csv
```

```bash
python factuality_seniority.py --input ../../../results/summary/factuality_field.csv --output ../../../results/summary/factuality_seniority.csv
```

```bash
python factuality_location.py --input ../../../results/summary/factuality_seniority.csv --output ../../../results/summary/factuality_location.csv
```

```bash
python factuality_ethnicity.py --input ../../../results/summary/factuality_location.csv --ethnicity_lookup ../../../results/summary/recommendations_with_ethnicity.csv --output ../../../results/summary/factuality_ethnicity.csv
```

### 5. Ethnicity (`code/scripts/ethnicity/`)

Inferencia étnica en cascada (BERT `liamliang/demographics_race_v2` → fallback `ethnicolr` LSTM → `Unknown`). Verificación opcional con NamSor.

```bash
python apply_ethnicity_ground_truth.py
```

```bash
python apply_ethnicity_ground_truth_parallel.py --workers 8 --batch-size 256
```

```bash
python namsor_verify.py --input ../../../results/summary/recommendations_with_ethnicity.csv --limit 20 --dry-run
```

```bash
python namsor_verify.py --input ../../../results/summary/recommendations_with_ethnicity.csv --output-jsonl ../../../data/ethnicity_inference/namsor_responses.jsonl --output-csv ../../../data/ethnicity_inference/namsor_results.csv
```

---

## Notebooks de análisis

- `code/notebooks/analysis/metrics_pipeline.ipynb` — pipeline de métricas (refusals, validity, duplicates, consistency, similarity, diversity, parity)
- `code/notebooks/analysis/factuality_metrics.ipynb` — métricas de factuality (author, field, seniority, location, epoch)
- `code/notebooks/analysis/ethnicity_metrics.ipynb` — métricas de inferencia étnica
- `code/notebooks/analysis/manual_classification_metrics.ipynb` — métricas de la anotación manual
- `code/notebooks/analysis/aps_demographics_analysis.ipynb` — análisis demográfico APS
- `code/notebooks/agreement/inter_annotator_agreement.ipynb` — agreement entre anotadores
- `code/notebooks/agreement/inter_annotator_agreement_ethnicity.ipynb` — agreement étnico
- `code/notebooks/scratch/` — exploración rápida (parquet, semantic scholar, prompts)

---

## Convenciones

- Cada comando va **en una sola línea** para copiar y pegar.
- Salidas del pipeline en `results/summary/{factuality_*.csv, recommendations.csv, summary.csv}`.
- Plots: **izquierda = settings**, **derecha = métricas**.
- Datos: `data/semantic_scholar_data/clean/Researchers_Deduplicated_Genderize_Namsor.parquet` es el archivo canónico para *factuality* (216 MB, deduplicado por researcher_id).

---

## Documentación adicional

- `code/scripts/README.md` — guía detallada del pipeline de *prompting* (parámetros, output, ejemplos con GNU Parallel).
- Cada script Python contiene un docstring superior con uso, parámetros y ejemplos.
