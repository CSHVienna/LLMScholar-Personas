# Datasets

Detalle de las dos fuentes de datos usadas en el pipeline: **Semantic Scholar** (ground truth de investigadores) y **OpenAlex** (enrichment + verificación de factuality).

---

## 1. Semantic Scholar (SS)

**Origen:** aporte de Ana María Jaramillo (AJ), extensión del paper previo.
**Ubicación:** [/data/datasets/LLMScholar-Personas/data/semantic_scholar_data/](/data/datasets/LLMScholar-Personas/data/semantic_scholar_data/)

| Carpeta / archivo | Tamaño | Contenido |
|---|---|---|
| `all/DataFrameRankings_Genderize_Namsor_<Field>.csv` (6 archivos) | 14 GB | Originales de AJ — un CSV por campo (Biology, Computer_Science, Mathematics, Physics, Psychology, Sociology), stats por autor **por año 1950‑2019** |
| `clean/DataFrameRankings_Genderize_Namsor.parquet` | 4.4 GB | Merge de los 6 CSV en un solo parquet (mismo eje longitudinal) |
| `clean/Researchers_Deduplicated_Genderize_Namsor.parquet` | 226 MB | **Únicos por `Researcher_id`** con stats al cierre de 2019 — el que se usa en factuality |

**Cutoff temporal:** stats acumulados hasta **2019** (nota en `README.txt`; `Last_year` llega como mucho a 2021 en algunos registros, primer año mínimo 1605 — outliers históricos).

### Columnas relevantes (parquet dedup, 17 en total)

`Researcher_id, Name, Clean_standarized_name, Field, First_year, Last_year, Career_age, Citations, Productivity, Collaborations, Ranking_position_citations, Gender` (Genderize), `Probability, Gender_Namsor, Gender_probability_Namsor, Number_of_genders, Combined_gender`.

**Género usado:** `Combined_gender` — instrucción explícita de AJ en el README (combina Genderize + Namsor).

---

## 2. OpenAlex (OA) — DuckDB local

**Archivo:** `/data/datasets/LLMScholar-Personas/data/openalex_latest.duckdb`
**Resolución del symlink:** → `/code/openalex/openalex_latest.duckdb` → `/code/openalex/openalex_20260330.duckdb` (224 GB)

**Cutoff del snapshot:** **2026‑03‑30** (en el nombre del archivo).

Acceso siempre `read_only=True`, `memory_limit=64–96 GB`, spill en `/data/asanchez/duckdb_enrich/`.

### Qué se usa de OA y para qué

| Tabla / campo | Para qué | Script |
|---|---|---|
| `authors` (exact normalized name) → `id, display_name, works_count, cited_by_count, counts_by_year, x_concepts` | Resolver `oa_id` por nombre; derivar `oa_first_pub_year`, `oa_last_pub_year`, `oa_career_age`, top concepts | [factuality_openalex.py:124-228](code/scripts/factuality/factuality_openalex.py#L124-L228) |
| `authors` (Jaro‑Winkler ≥ 0.85) | Fallback fuzzy para nombres no resueltos por exact match | [factuality_openalex.py:230-398](code/scripts/factuality/factuality_openalex.py#L230-L398) |
| `works` con `UNNEST(authorships) × UNNEST(institutions)` | País (`oa_country_code`) e institución (`oa_last_institution`) del **paper más reciente** por `oa_id` — porque `authors.last_known_institution` viene casi siempre vacío en el snapshot | [factuality_openalex.py:400-555](code/scripts/factuality/factuality_openalex.py#L400-L555), [enrich_from_works.py](code/scripts/enrichment/enrich_from_works.py) |
| OpenAlex public **API** (ruta alternativa) | Mismo país + institución + `h_index` (no está en el snapshot DuckDB) | [enrich_via_api.py](code/scripts/enrichment/enrich_via_api.py), [enrich_via_api_parallel.py](code/scripts/enrichment/enrich_via_api_parallel.py) |
| `authors.x_concepts` | Verificación de campo (`field_check_source = openalex_concepts`) cuando no hay match en SS ground truth | [factuality_field_check.py](code/scripts/factuality/factuality_field_check.py) |

**Notas:**
- `h_index` / `i10_index` no están materializados en el dump DuckDB → solo se llenan vía API.
- El works scan se chunkea en 20 segmentos por año; cache en `WORKS_AGG_DIR=/data/asanchez/duckdb_enrich/oa_works_agg_chunks`.

---

## 3. Estadísticas del parquet SS

Computado sobre [Researchers_Deduplicated_Genderize_Namsor.parquet](/data/datasets/LLMScholar-Personas/data/semantic_scholar_data/clean/Researchers_Deduplicated_Genderize_Namsor.parquet) — **6,686,108 investigadores únicos** (`Researcher_id` único en el parquet, no hay duplicados).

| Field | # Scholars | female | male | NULL | % female (sobre total) | % female (solo conocidos) |
|---|---:|---:|---:|---:|---:|---:|
| Biology          | 2,053,232 |   625,408 |   937,293 |   490,531 | 30.46 % | 40.02 % |
| Computer Science | 1,426,900 |   264,493 |   980,551 |   181,856 | 18.54 % | 21.24 % |
| Mathematics      |   707,656 |   111,877 |   457,009 |   138,770 | 15.81 % | 19.67 % |
| Physics          |   902,877 |   152,618 |   493,355 |   256,904 | 16.90 % | 23.63 % |
| Psychology       | 1,236,641 |   471,956 |   590,365 |   174,320 | 38.16 % | 44.43 % |
| Sociology        |   358,802 |   134,938 |   200,643 |    23,221 | 37.61 % | 40.21 % |
| **Total**        | **6,686,108** | **1,761,290** | **3,659,216** | **1,265,602** | **26.34 %** | **32.49 %** |

**Lectura rápida:**
- Biology es el campo más grande (~31 % de la base); Sociology el más pequeño (~5 %).
- Sociology tiene la **mejor cobertura de género** (93.5 % con etiqueta) y Physics la **peor** (71.6 %).
- Psychology y Sociology son los campos más femeninos (~38 %); Mathematics y Physics los más masculinos (~65 % / ~55 %).
- Normalizando solo sobre conocidos, la fracción femenina sube **+3.6 a +9.6 puntos** según el campo (Biology +9.6, Psychology +6.3, Physics +6.7, Mathematics +3.9).
