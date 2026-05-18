# Métricas y Estrategias — LLMScholar-Personas

Documentación completa de las métricas calculadas en el benchmark, sus fórmulas, dónde están implementadas y las estrategias de evaluación. Extiende el paper anterior agregando *persona prompting* sobre tres dimensiones: **lenguaje**, **país** y **rol**.

---

## 1. Pipeline de datos

```
Prompts (persona)  →  LLM responses  →  Parsing  →  Enrichment (OpenAlex)
                                                          ↓
                                                    Annotation (manual)
                                                          ↓
                                                 Factuality / Ethnicity checks
                                                          ↓
                                                       Métricas
```

| Etapa | Carpeta | Script principal | Propósito |
|------|---------|------------------|-----------|
| 1. Prompting | [code/scripts/prompting/](code/scripts/prompting/) | `batch_prompt.py`, `batch_params.py` | Genera combinaciones (rol × país × idioma × campo × k) y traduce parámetros EN→ES/DE |
| 2. Parseo | [code/scripts/prompting/](code/scripts/prompting/) | `batch_parse_results.py` | Convierte respuestas LLM en `recommendations.csv` y `summary.csv` |
| 3. Enriquecimiento | [code/scripts/enrichment/](code/scripts/enrichment/) | `enrich_via_api.py`, `enrich_from_works.py`, `apply_enrichment.py` | Resuelve cada autor a su ID en OpenAlex, agrega `oa_country_code`, institución, métricas |
| 4. Anotación manual | [code/scripts/annotation/](code/scripts/annotation/) | `annotate_responses.py`, `annotate_ethnicity.py` | Etiquetas humanas para validez y etnicidad |
| 5. Factualidad | [code/scripts/factuality/](code/scripts/factuality/) | 6 pasos secuenciales (ver §5) | Verifica que cada autor exista y matchee field/seniority/location/ethnicity |
| 6. Etnicidad | [code/scripts/ethnicity/](code/scripts/ethnicity/) | `apply_ethnicity_ground_truth.py`, `namsor_verify.py` | Predicción etnia: BERT → ethnicolr → Unknown; cruce opcional con NamSor |
| 7. Métricas | [code/libs/metrics/aggregators.py](code/libs/metrics/aggregators.py) | — | Agregación y CIs |

---

## 2. Estrategia de Persona Prompting

Implementación: [code/libs/prompt/generation.py:100-173](code/libs/prompt/generation.py#L100-L173).

Cada prompt se construye con **4 dimensiones** controladas:

| Dimensión | Fuente | Valores |
|-----------|--------|---------|
| **Rol / Tarea** | `instructions.json` | Combinación `role` + `task` (ej. estudiante de posgrado pidiendo recomendaciones) |
| **País / Locación** | `locations.json` | Ecuador, Japón, Alemania, Canadá, Sudáfrica |
| **Idioma** | clave dict | `EN`, `ES`, `DE` |
| **Campo + Subcampo + K** | `input.json` | 6 campos (Biology, CS, Maths, Physics, Psychology, Sociology); `top_k`, `epoch`, `seniority`, `twins` |

Plantilla modular (línea 148-173):
```
INSTRUCTION[lang]:  rol + tarea + locación
INPUT[lang]:        k + target + field + subfield
```

Salida: tuple `(instructions_str, input_str)` que se envía al LLM.

**Combinaciones totales** ≈ `idiomas × países × campos × subcampos × valores_k × roles`.

---

## 3. Semántica "rate" — todas las métricas son tasas

Casi todas las métricas son **rates (fracciones en [0,1])**, no conteos absolutos. Existen dos denominadores distintos según la métrica (ver [metrics_pipeline.ipynb cell `Build per-call metrics table`](code/notebooks/analysis/metrics_pipeline.ipynb)):

| Tipo de tasa | Denominador | Métricas |
|--------------|-------------|----------|
| **Per request/attempt** | n autores devueltos | `refusal_pct`, `validity_pct`, `duplicates` |
| **Found rate** | n autores únicos solicitados (`n_unique`) | `factuality` (= `n_found / n_unique`) |
| **Match rate among found** | n encontrados con status `*_match` ∪ `*_mismatch` | `factuality_field`, `factuality_seniority`, `factuality_epoch`, `factuality_location`, `factuality_ethnicity` |
| **Distribución** | fracciones por categoría | `diversity_*`, `parity_*`, `pct_low/med/high_*` |

Importante: los factualitys de atributo (field, seniority, etc.) solo se evalúan sobre autores **encontrados** (`author_status == 'found'`). Autores alucinados no entran al denominador.

---

## 4. Métricas técnicas (calidad de respuesta)

Agrupación a nivel "request" o "attempt":
- `BENCHMARK_PER_REQUEST_COLS` = `[model_access, model_size, model_class, model, grounded, temperature, date, time, task_name, task_param]`
- `BENCHMARK_PER_ATTEMPT_COLS` = idem + `task_attempt`

Definido en [code/libs/metrics/constants.py:774-775](code/libs/metrics/constants.py#L774-L775).

### 3.1 Refusal — `refusal_pct`
Fracción de respuestas donde el modelo rehusó la tarea.

$$ \text{refusal\_pct} = \frac{n_{\text{refusals}}}{n_{\text{total}}} $$

Implementación: [aggregators.py:92-95](code/libs/metrics/aggregators.py#L92-L95). Métrica binaria → CI **Wilson score**.

### 3.2 Validity — `validity_pct`
Fracción de requests con al menos un intento válido (`valid_attempt.any()`).

$$ \text{validity\_pct} = \frac{n_{\text{valid requests}}}{n_{\text{requests}}} $$

Implementación: [aggregators.py:98-102](code/libs/metrics/aggregators.py#L98-L102). Binaria → CI **Wilson**.

### 3.3 Duplicates — `duplicates`
Fracción de autores duplicados por request.

$$ \text{duplicates} = 1 - \frac{n_{\text{unique\_authors}}}{n_{\text{total\_authors}}} $$

Implementación: [aggregators.py:104-107](code/libs/metrics/aggregators.py#L104-L107). No-binaria → CI **t-Student**.

### 3.4 Consistency — `consistency`
Estabilidad de los sets de autores entre runs sucesivos del mismo prompt: media del Jaccard entre el set de la corrida `i` y la corrida `i-1`.

$$ J(A_i, A_{i-1}) = \frac{|A_i \cap A_{i-1}|}{|A_i \cup A_{i-1}|}, \quad \text{consistency} = \frac{1}{T-1}\sum_{i=2}^{T} J(A_i, A_{i-1}) $$

Implementación: [aggregators.py:109-141](code/libs/metrics/aggregators.py#L109-L141). CI **t-Student**.

---

## 5. Métricas de Factualidad

Dos implementaciones equivalentes:
- **Agregador**: [aggregators.py:153-168](code/libs/metrics/aggregators.py#L153-L168), mapeo en [constants.py:701-703](code/libs/metrics/constants.py#L701-L703).
- **Notebook per-call** (canónico en `metrics_pipeline.ipynb`): construye `calls` por `_cid` (call id) y calcula match rates con denominador "evaluable".

### 5.1 `factuality_author` — Found rate
$$ \text{factuality\_author} = \frac{n_{\text{found}}}{n_{\text{unique}}} $$
`n_found` = autores con `author_status == 'found'` (resueltos en Semantic Scholar **o** OpenAlex).

### 5.2 `factuality_field` — Field match rate
$$ \text{factuality\_field} = \frac{n_{\text{field\_match}}}{n_{\text{field\_match}} + n_{\text{field\_mismatch}}} $$
Solo cuenta autores encontrados cuyo `field_status ∈ {field_match, field_mismatch}` (excluye `field_unknown`).

### 5.3 `factuality_seniority` — Seniority match rate
Misma forma con `seniority_status`. Buckets:
- Junior: `career_age ≤ 10`
- Senior: `career_age ≥ 20`
- Unknown (excluido): 11–19

### 5.4 `factuality_epoch`, `factuality_location`, `factuality_ethnicity`
Mismo patrón "match rate among evaluable":
- `factuality_epoch`: matchea década de primera publicación contra la solicitada (ej. 1950s vs 2000s).
- `factuality_location`: matchea `oa_country_code` ISO contra la persona-location (EC/JP/DE/CA/ZA).
- `factuality_ethnicity`: matchea `perceived_ethnicity` (LLM) contra ground-truth (cascada BERT → ethnicolr).

Fórmula genérica:

$$ \text{factuality}_X = \frac{n_{X\text{\_match}}}{n_{X\text{\_match}} + n_{X\text{\_mismatch}}} $$

### 5.5 Pipeline de factualidad (`code/scripts/factuality/`)
Orquestado por `run_factuality_pipeline.py`:

| Paso | Script | Input → Output |
|------|--------|----------------|
| 0 | `factuality_author_jw.py` | recommendations → factuality_author_jw.csv (Jaro-Winkler vs Semantic Scholar) |
| 0.5 | `factuality_openalex.py` | author_jw → factuality_oa.csv (fallback OpenAlex DuckDB + API) |
| 1 | `factuality_field_check.py` | oa → factuality_field.csv (vs 6 CSVs GT por campo) |
| 2 | `factuality_seniority.py` | field → factuality_seniority.csv (buckets career_age) |
| 3 | `factuality_location.py` | seniority → factuality_location.csv (ISO-3166 alpha-2) |
| 3.5 | `factuality_affiliation.py` | location → factuality_affiliation.csv (LLM `current_affiliations` vs OA institution history) |
| 4 | `factuality_ethnicity.py` | affiliation → **factuality_full.csv** (final) |

Cada paso solo añade columnas `*_status`; nunca elimina filas.

#### 5.6 `factuality_affiliation` — Affiliation match rate

$$ \text{factuality\_affiliation} = \frac{n_{\text{aff\_match}}}{n_{\text{aff\_match}} + n_{\text{aff\_mismatch}}} $$

Compara las afiliaciones declaradas por el LLM en `current_affiliations` contra el **histórico completo** de instituciones del autor en OpenAlex (no solo la última). El histórico se reconstruye a partir de las chunks parquet ya pre-calculadas por `factuality_openalex.py` (`/data/asanchez/duckdb_enrich/oa_works_agg_chunks/chunk_*.parquet`) — cada chunk guarda un `ARG_MAX(inst.display_name, publication_year)` por (oa_id, año), así que `LIST(DISTINCT inst_name)` cross-chunks aproxima la trayectoria institucional del autor a granularidad de chunk.

Match string-a-string con `rapidfuzz.fuzz.token_set_ratio ≥ 85` sobre nombres normalizados (lowercase + NFD strip accents, reusando `factuality_openalex.normalize_name`). Si **cualquier** afiliación LLM matchea **cualquier** institución OA → `affiliation_match`; si ninguna → `affiliation_mismatch`; si el autor fue clasificado `hallucinated` → `not_applicable`; si falta LLM o OA → `affiliation_unknown`.

**Limitación de la GT**: Semantic Scholar (`Researchers_Deduplicated_*` parquet y los CSVs en `all/`) **no almacena institución por researcher**, por eso la única fuente de verdad para afiliación es OpenAlex.

Columnas añadidas: `affiliation_llm, affiliation_oa_all, affiliation_best_match_score, affiliation_best_match_oa, affiliation_status`.

---

## 6. Métricas de Diversidad

Implementación: [aggregators.py:189-256](code/libs/metrics/aggregators.py#L189-L256) y notebook `metrics_pipeline.ipynb` (`_div_fixed`, `_div_geo`, `_tier_stats`).

Fórmula base: **entropía de Shannon normalizada** sobre las etiquetas observadas (excluyendo `Unknown`).

$$ H = -\sum_{i=1}^{K} p_i \ln(p_i), \quad \text{diversity} = \frac{H}{\ln(K)} \in [0,1] $$

| Métrica | Atributo | K | Tipo |
|---------|----------|---|------|
| `diversity_gender` / `div_gender` | gender | 3 (Female, Male, Unisex) | K fijo |
| `diversity_ethnicity` / `div_ethnicity` | ethnicity | 5 (Black, Asian, White, Latino, American Indian) | K fijo |
| `div_geography` | `location_oa_iso` | **variable** (n distintos países observados) | K dinámico |
| `diversity_prominence_pub` / `div_productivity_works` | tier de `oa_works_count` | 3 (low/med/high) | K fijo |
| `diversity_prominence_cit` / `div_productivity_citations` | tier de `oa_cited_by_count` | 3 (low/med/high) | K fijo |

Reglas adicionales (del notebook):
- Si `total < 2` autores con etiqueta conocida → `NaN` (no se reporta diversidad sobre 1 punto).
- `div_geography` usa `K = #países distintos en esa call`, no un K fijo.

> Nota: `PROMINENCE_CATEGORIES` en [constants.py:157](code/libs/metrics/constants.py#L157) lista 4 (`low, mid, high, elite`) pero el notebook usa 3 tiers (`low, med, high`). El agregador llama a `get_number_of_categories` para K.

---

## 7. Métricas de Productividad (Prominence)

Implementación canónica: notebook [metrics_pipeline.ipynb — Step 1.5 "Productivity tiers"](code/notebooks/analysis/metrics_pipeline.ipynb).

### 7.1 Construcción de tiers (per-field terciles)
Solo se calculan sobre autores **encontrados** (los nombres alucinados no tienen productividad real). Para cada campo:

1. Calcula umbrales `p33` y `p67` sobre `oa_works_count` y `oa_cited_by_count`.
2. Asigna cada autor a tier:
   - `low` si `value ≤ p33`
   - `med` si `p33 < value ≤ p67`
   - `high` si `value > p67`

> Los terciles son **per-field**: Biology, CS, Maths, Physics, Psychology, Sociology tienen umbrales distintos (productividad es campo-dependiente).

### 7.2 Métricas por call

Para `lab ∈ {works, citations}`:

| Métrica | Definición |
|---------|-----------|
| `pct_low_{lab}` | fracción de autores found en tier low |
| `pct_med_{lab}` | fracción en tier med |
| `pct_high_{lab}` | fracción en tier high |
| `div_productivity_{lab}` | Shannon normalizada sobre los 3 tiers (`K=3`) |

Coverage: si menos de 2 autores tienen tier asignado → `NaN`.

---

## 8. Métricas de Paridad

Implementación: [aggregators.py:263-305](code/libs/metrics/aggregators.py#L263-L305).

Fórmula: **1 − distancia de variación total** entre la distribución del recomendador y la del ground truth (OpenAlex).

$$ TV = \frac{1}{2}\sum_{c \in C} |p^{rec}_c - p^{gt}_c|, \quad \text{parity} = 1 - TV \in [0,1] $$

Si una categoría aparece en GT pero no en el resultado, se imputa 0 (línea 279-281).

| Métrica | Atributo |
|---------|----------|
| `parity_gender` | gender |
| `parity_ethnicity` | ethnicity |
| `parity_prominence_pub` | prominence_pub |
| `parity_prominence_cit` | prominence_cit |

---

## 9. Métricas de Similaridad / Conectividad

Definidas en [constants.py:708-714](code/libs/metrics/constants.py#L708-L714) y consumidas vía [aggregators.py:309-319](code/libs/metrics/aggregators.py#L309-L319). Vienen pre-calculadas en un `df_similarity` externo (red de co-autoría OpenAlex).

| Benchmark | Columna interna | Significado |
|-----------|----------------|-------------|
| `connectedness` | `connectedness_entropy` | Entropía del tamaño de componentes |
| `connectedness_density` | `recommended_author_pairs_are_coauthors` | Fracción de pares recomendados que son co-autores |
| `connectedness_norm_entropy` | `normalized_component_entropy` | Entropía normalizada |
| `connectedness_ncomponents` | `normalized_n_components` | Conteo normalizado de componentes |
| `similarity_pca` | `scholarly_pca_similarity_mean` | Cosine PCA-similarity scholar a scholar |

---

## 10. Agregación y Confidence Intervals

Implementación: [aggregators.py:12-86](code/libs/metrics/aggregators.py#L12-L86).

### 10.1 Per-attempt
Agrupa por `BENCHMARK_PER_ATTEMPT_COLS` y aplica la función de la métrica (mean / Jaccard / parity / entropía).

### 10.2 Per-group
Para cada métrica calcula `mean, std, median, sum, n`. La elección del **CI a 95%** depende del tipo de métrica:

| Tipo | Métricas | CI |
|------|----------|-----|
| **Bernoulli** (`BENCHMARK_BINARY_METRICS`) | `refusal_pct`, `validity_pct` | **Wilson score** vía `proportion_confint` |
| **No-Bernoulli** (todas las demás) | duplicates, consistency, factuality, diversity, parity, similarity | **t-Student**: $CI = t_{1-\alpha/2, n-1} \cdot \frac{\sigma}{\sqrt{n}}$ |

Constants en [constants.py:689-692](code/libs/metrics/constants.py#L689-L692).

---

## 11. Inter-annotator agreement

Notebooks: [code/notebooks/agreement/inter_annotator_agreement.ipynb](code/notebooks/agreement/inter_annotator_agreement.ipynb), [code/notebooks/agreement/inter_annotator_agreement_ethnicity.ipynb](code/notebooks/agreement/inter_annotator_agreement_ethnicity.ipynb).

Tres medidas sobre pares de anotadores:

1. **Raw agreement** = `# coincidencias / N`
2. **Cohen's κ** — agreement pairwise corregido por azar.
3. **Krippendorff's α** — confiabilidad nominal multi-anotador (categorías: valid / refusal / invalid / empty / illustrative para validez; 5 categorías para etnicidad).

Excluye filas con `"skip"`.

---

## 12. Etnicidad: estrategia de inferencia

Cascada (en `code/scripts/ethnicity/apply_ethnicity_ground_truth.py`):

1. **BERT** — `liamliang/demographics_race_v2` (clasificador transformer).
2. **Fallback ethnicolr** — LSTM basado en surname si BERT no produce predicción confiable.
3. **Unknown** — si ambos fallan.
4. **Verificación opcional** — `namsor_verify.py` cruza con la API NamSor.

Cada autor termina con `ethnicity_inference` + probabilidades.

---

## 13. Constantes y mapas relevantes

| Constante | Valor / Definición | Ubicación |
|-----------|-------------------|-----------|
| `PROMINENCE_CATEGORIES` | `["low", "mid", "high", "elite"]` | [constants.py:157](code/libs/metrics/constants.py#L157) |
| `GENDER_LIST` | `[Female, Male, Unisex, Unknown]` | constants.py:~95 |
| `ETHNICITY_LIST` | `[Black, Asian, White, Latino, Unknown]` | constants.py:~110 |
| `BENCHMARK_TECHNICAL_METRICS` | refusal, validity, duplicates, consistency, factuality_* | [constants.py:737-746](code/libs/metrics/constants.py#L737-L746) |
| `BENCHMARK_SOCIAL_METRICS` | connectedness_*, similarity_pca, diversity_*, parity_* | [constants.py:717-731](code/libs/metrics/constants.py#L717-L731) |
| `BENCHMARK_METRICS_PLOT_ORDER` | Orden de columnas en gridbar | [constants.py:687](code/libs/metrics/constants.py#L687) |
| `BENCHMARK_METRIC_HIGHLIGHT_RULES` | `max` / `min` por métrica para resaltado | [constants.py:791+](code/libs/metrics/constants.py#L791) |

---

## 14. Estilo de figuras

Convención del paper anterior (heredada): **izquierda = settings/condiciones, derecha = métricas**. Las figuras se generan en [code/libs/visuals/](code/libs/visuals/) (`gridbar.py`, `gridcons.py`, `vis.py`) y se invocan desde los notebooks de `code/notebooks/analysis/`.

| Notebook | Foco |
|----------|------|
| `metrics_pipeline.ipynb` | Pipeline completo de cálculo |
| `factuality_metrics.ipynb` | Factualidad por modelo / tarea |
| `ethnicity_metrics.ipynb` | Diversidad y paridad de etnicidad |
| `manual_classification_metrics.ipynb` | Métricas sobre etiquetas manuales |
| `aps_demographics_analysis.ipynb` | Análisis demográfico APS |

---

## Resumen rápido de fórmulas

| Métrica | Fórmula | Tipo |
|---------|---------|------|
| `refusal_pct` | `n_refusal / n_total` | rate per request |
| `validity_pct` | `n_valid / n_requests` | rate per request |
| `duplicates` | `1 − n_unique/n_total` | rate per request |
| `consistency` | `mean Jaccard(A_i, A_{i-1})` | similarity |
| `factuality_author` | `n_found / n_unique` | **found rate** |
| `factuality_field` | `n_field_match / (n_field_match + n_field_mismatch)` | **match rate among found** |
| `factuality_seniority` | `n_sen_match / (n_sen_match + n_sen_mismatch)` | match rate |
| `factuality_epoch` | idem epoch | match rate |
| `factuality_location` | idem location (ISO match) | match rate |
| `factuality_ethnicity` | idem ethnicity | match rate |
| `div_*` | `−Σ p_i ln(p_i) / ln(K)` | entropy ∈ [0,1] |
| `div_geography` | misma fórmula, **K dinámico** = #países distintos | entropy |
| `pct_low/med/high_works` | tier counts / total (per-field terciles) | distribución |
| `pct_low/med/high_citations` | idem para citaciones | distribución |
| `div_productivity_works/citations` | Shannon sobre 3 tiers | entropy |
| `parity_*` | `1 − ½ Σ|p_rec − p_gt|` | TV-based ∈ [0,1] |
| `factuality_affiliation` | `n_aff_match / (n_aff_match + n_aff_mismatch)` | match rate (OA history) |
| CI Bernoulli (refusal, validity) | Wilson score (`proportion_confint`) | — |
| CI no-Bernoulli (resto) | `t_{1−α/2, n−1} · σ/√n` | — |

---

## 15. Métricas por sub-población

Implementación: nueva sección "Step 6 — Social metrics per sub-population" en [metrics_pipeline.ipynb](code/notebooks/analysis/metrics_pipeline.ipynb); constantes en [code/libs/metrics/constants.py](code/libs/metrics/constants.py) (`FIELD_ORDER`, `LOCATION_ORDER`, `BENCHMARK_SUBPOPULATION_COMBOS`).

Cuando el persona prompt restringe la búsqueda a un subgrupo (p.ej. `location=Japan`, `field=Biology`), el ground truth contra el que se calcula `parity` y los terciles de popularity también se filtran a ese subgrupo. Diversity, por no usar GT, simplemente se calcula dentro del subgrupo.

Dimensiones de sub-población:

| Combo | Fuente del GT filtrado |
|-------|------------------------|
| `field` | SS GT filtrado por la columna `Field` (cargada del nombre del CSV `DataFrameRankings_<Field>_with_ethnicity.csv`) joineado con `Combined_gender` del parquet dedup. |
| `location` | **OA-derived proxy** — la distribución de gender/ethnicity de los autores en `factuality_full.csv` con ese `location_oa_iso`. SS no tiene país por researcher, así que esta es la GT más informativa disponible sin recorrer OA completo. |
| `field × location` | AND de los dos anteriores sobre el proxy OA. |

Popularity (terciles `low`/`med`/`high`) se recalcula con `p33`/`p67` dentro del pool del subgrupo (`field` solo, `location` solo, o el cruce). Para subgrupos con < 30 autores se omite (terciles no significativos).

Output: long-format `results/summary/metrics_per_subpop.csv` con columnas `CALL_KEYS + [subpop_dim, subpop_field, subpop_location, metric_name, value]`. Cada fila es una métrica calculada para una llamada LLM contra una porción del GT.
