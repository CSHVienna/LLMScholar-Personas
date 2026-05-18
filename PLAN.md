# Plan — Validación manual + factuality de afiliación + métricas sociales por sub-población

## Context

Lisette ve los resultados poco novedosos y pidió tres tareas urgentes:

1. **Validación manual** de 5 requests para auditar `author_status` en `factuality_full.csv` contra inspección humana en SS/OA.
2. **Factuality de afiliación** — hoy solo se chequea `name+lastname` (`factuality_field.py:180`). El JSON ya trae `current_affiliations`.
3. **Parity / diversity / popularity por sub-población** — hoy se comparan contra GT completo (`aggregators.py:276`). Lisette pidió que cuando el persona prompt restringe a un subgrupo (location, field, field×location), el GT también se filtre.

Restricción: **reutilizar TODO lo que ya existe**, sin regenerar CSVs pesados ni el pipeline de enriquecimiento.

## Hallazgos de la auditoría que cambian el plan

### Hallazgo 1 — `recommendations.csv` YA tiene los campos pedidos
`batch_parse_results.py:1064` extrae `name, lastname, current_affiliations, areas_of_research_or_work, reason, source`. Verificado en header del CSV. **No regenerar nada.**

### Hallazgo 2 — Mucho código fuzzy/DuckDB es reutilizable directo
- `factuality_author_jw.py:38-208` ya importa `rapidfuzz.distance.JaroWinkler` y tiene `_score_matrix` con thresholds 0.85/0.70 → **reuso la importación; uso `rapidfuzz.fuzz.token_set_ratio` para afiliaciones** (es el mismo paquete, solo distinta métrica).
- `factuality_openalex.py:78-85` `normalize_name()` lowercase + NFD strip accents + letras+espacios. **Reusable tal cual para normalizar nombres de instituciones.**
- `factuality_openalex.py:103-121` `_open_duckdb()` ya configura conexión read-only, threads, memory_limit=64GB. **Reuso directo; importo desde mi nuevo script.**
- `factuality_openalex.py:462-476` ya hace UNNEST sobre `works.authorships[].institutions[]` con `ARG_MAX` para `last institution`. **Modifico el query para devolver `LIST(DISTINCT inst.display_name)` (histórico completo).**

### Hallazgo 3 — `factuality_location.py` es el template casi exacto para afiliación
Sigue exactamente la forma que necesito: lee CSV anterior, aplica `classify_row()` (`location_match`/`mismatch`/`unknown`/`not_applicable`), escribe nuevo CSV. **Copio su estructura — mismo CLI args, misma forma de respetar `author_status==hallucinated` → `not_applicable`.**

### Hallazgo 4 — `run_factuality_pipeline.py` tiene un patrón limpio para añadir un step
`run_factuality_pipeline.py:80-123` define `steps = [(label, cmd_list, optional)…]`. Inserto entre location y ethnicity con el mismo formato CLI (`--input/--output`). **0 código nuevo de orquestación.**

### Hallazgo 5 — `factuality_full.csv` ya tiene `oa_country_code` por autor
Esto significa que para popularity/diversity por sub-poblacion de **país**, NO necesito join SS↔OA — la columna ya está enriquecida (`factuality_openalex.py` la pone vía `ARG_MAX(inst.country_code, w.publication_year)`).

### Hallazgo 6 — Terciles per-field de prominence YA están calculados en el notebook
`metrics_pipeline.ipynb:753-757` ya hace `prod.groupby('field_en')[col].quantile([0.33, 0.67])`. **Reutilizo este bloque y solo añado `.groupby(['field_en', 'oa_country_code'])` para terciles per (field, country).**

### Hallazgo 7 — `grouped_metrics.py:108-159` `compute_grouped_metrics_table` YA acepta `filter={'field_en': 'Biology'}` por configuración
Esto es exactamente lo que necesito para sub-poblaciones. **No hay que refactorizar nada — solo construir una lista de configs con filters por (field, location).**

### Hallazgo 8 — `aggregate_parity` (`aggregators.py:263-305`) acepta `df_gt` por kwargs
**No necesito refactorizar la función**: solo paso un `df_gt` ya filtrado al subgrupo. La lógica per-attempt sigue igual; la única decisión es construir el GT correcto por subgrupo afuera.

### Hallazgo 9 — Limitación de GT para parity por país
El GT de SS (`Researchers_Deduplicated_Genderize_Namsor.parquet`) tiene `Field` y `Combined_gender` pero **no tiene país**. Solo OA tiene país. No existe join SS↔OA `Researcher_id → country` ya hecho en el repo. Implicación práctica:
- **parity_gender por field**: trivial (SS filtrada por field).
- **parity_gender por location**: necesita un join OA→SS para añadir país al GT, **o** usar como GT proxy el universo OA per `(field, country)`.
- **Decisión:** para urgencia, uso el universo de autores **enriquecidos en `factuality_full.csv`** como GT-OA per `(field, country)` — es la población que las LLMs pueden potencialmente recomendar y está pre-calculada. Lo documento en `METRICS.md` como "OA-derived sub-population GT".

### Hallazgo 10 — `LANGUAGES` existe pero `FIELD_ORDER` y `LOCATION_ORDER` NO
`code/libs/utils/constants.py:57` tiene `LANGUAGES = [LANG_EN, LANG_ES, LANG_DE]`. Añado dos constantes nuevas mínimas en `metrics/constants.py`.

### Hallazgo 11 — `grid.py` y `gridbar.py` están INCOMPLETOS (depende de un módulo inexistente)
No me apoyo en ellos. **Reuso `grouped_metrics.py` que sí está completo y funcional.**

---

## Tarea 1 — Validación manual de 5 requests

**Entregable:** `code/notebooks/agreement/manual_factuality_validation.ipynb` + `data/datasets/LLMScholar-Personas/results/manual/manual_validation_5.csv`.

**REUSO:**
- `ios.load_csv`, `ios.load_json`, `ios.list_files_in_folder`, `ios.to_csv` (`code/libs/utils/ios.py`).
- `cons.RESULTS_PATH = '<ROOT>/responses/results_<SOURCE>_<LANGUAGE>'` (`code/libs/utils/constants.py:145`).
- Patrón de iteración sobre JSONs de `batch_parse_results.py:1000-1043` — copio el bloque para abrir un JSON y leer `parameters.persona_context` + `parameters.user_request`.
- `factuality_full.csv` para `author_status_auto` y `field_status_auto`.

**NUEVO:**
- Notebook con celdas:
  1. Muestreo: `df_summary.query("valid_flag in ['cleaned','unchanged']").sample(5, random_state=42)`.
  2. Para cada fila, ubicar JSON (`RESULTS_PATH` template) + buscar entrada por match exacto de `(role, task, location, k, target, field, subfield)` dentro del dict del JSON. Función helper local `find_request_in_json(json_path, params)` — 15 líneas.
  3. Extraer las k recomendaciones, join con `factuality_full.csv` por `(name, lastname, model, run_id, role, task, location, field, subfield, language)`.
  4. Generar columnas: `request_id, persona params, name, lastname, current_affiliations, areas_of_research_or_work, reason, source, author_status_auto, affiliation_status_auto (cuando Tarea 2 termine), found_in_ss_manual, found_in_oa_manual, notes_manual`.
  5. Imprimir links de búsqueda pre-formados: `https://www.semanticscholar.org/search?q=...` y `https://api.openalex.org/authors?search=...` por persona.
  6. Celda final (post-llenado manual): `agree = (found_in_ss_manual | found_in_oa_manual) == (author_status_auto != 'hallucinated')`; reporta % por persona y por request.

**Archivos críticos:**
- Lee: `/data/.../summary.csv`, `/data/.../factuality_full.csv`, JSONs.
- Escribe: `/data/.../results/manual/manual_validation_5.csv` + notebook.

---

## Tarea 2 — Factuality de afiliación

**Entregable:** `code/scripts/factuality/factuality_affiliation.py` (clon estructural de `factuality_location.py`) + nuevo step en `run_factuality_pipeline.py`.

**REUSO:**
- **Estructura del script y CLI:** copio el esqueleto exacto de `factuality_location.py` (`classify_row`, `parse_args`, `main`, mismo logging, mismo patrón `--input/--output`).
- `factuality_openalex.py:103-121` `_open_duckdb()` — **importo directo** desde mi script: `from factuality_openalex import _open_duckdb, normalize_name`.
- `factuality_openalex.py:78-85` `normalize_name()` — **uso tal cual** para normalizar afiliaciones (lowercase + strip accents + letras+espacios funciona para "MIT" vs "Massachusetts Institute of Technology" → no, ahí necesito token_set_ratio; pero sí ayuda con "Université de Montréal").
- `rapidfuzz` ya está en deps (lo usa `factuality_author_jw.py`). Uso `from rapidfuzz import fuzz`.
- `factuality_openalex.py:462-476` query DuckDB de instituciones — modifico para histórico:
  ```sql
  SELECT au.author.id AS oa_id,
         LIST(DISTINCT inst.display_name) AS oa_all_institutions
    FROM works AS w,
         UNNEST(w.authorships) AS t1(au),
         UNNEST(au.institutions) AS t2(inst)
   WHERE au.author.id IN (...)
     AND inst.display_name IS NOT NULL
   GROUP BY au.author.id
  ```
  Una sola query por todo el batch de `oa_id`s, no por autor.
- `run_factuality_pipeline.py:80-123` patrón de steps — añado:
  ```python
  ("4/5    factuality_affiliation", [sys.executable, "factuality_affiliation.py",
       "--input",  f"{r}/factuality_location.csv",
       "--output", f"{r}/factuality_affiliation.csv",
       "--duckdb", DUCKDB_PATH], False),
  ```
  Renombro los pasos posteriores en sus labels.

**NUEVO:**
- ~150 líneas en `factuality_affiliation.py`:
  - `parse_llm_affiliations(raw: str) -> list[str]`: `ast.literal_eval` + extracción de `affiliation` keys. Maneja `None`, `[]`, strings malformadas → `[]`.
  - `fetch_oa_institutions(con, oa_ids: list[str]) -> dict[str, list[str]]`: una query DuckDB, retorna dict.
  - `match_affiliations(llm_affs: list[str], oa_affs: list[str]) -> tuple[status, score, best_oa]`: producto cartesiano `token_set_ratio`, umbral 80, retorna el mejor.
  - `classify_row(row, oa_inst_cache)` análogo a `factuality_location.classify_row`.
  - Status: `affiliation_match | affiliation_mismatch | affiliation_unknown | not_applicable` (mismo schema que location).
- Columnas nuevas en `factuality_affiliation.csv`: `affiliation_llm, affiliation_oa_all, affiliation_best_match_score, affiliation_best_match_oa, affiliation_status`.
- Edita `factuality_ethnicity.py` (step 5, ya genera `factuality_full.csv`) para leer de `factuality_affiliation.csv` en lugar de `factuality_location.csv` — 1 línea.
- Documentación en `METRICS.md`: "Affiliation factuality uses OpenAlex as ground truth because SS dedup parquet lacks per-researcher institution; we use the full OA institution history per author (not only `last_institution`)."

**Validación:** smoke test con 100 filas aleatorias; verifico 5 matches y 5 mismatches a mano.

**Archivos críticos:**
- Lee: `factuality_location.csv`, OA DuckDB (`/data/.../openalex_latest.duckdb`).
- Escribe: `factuality_affiliation.csv`, sobrescribe `factuality_full.csv` (vía el step 5 ya existente).
- Nuevo: `code/scripts/factuality/factuality_affiliation.py`.
- Edita: `code/scripts/factuality/run_factuality_pipeline.py`, `code/scripts/factuality/factuality_ethnicity.py` (1 línea), `METRICS.md`.

---

## Tarea 3 — Parity / Diversity / Popularity por sub-población

**Dimensiones:** `field`, `location`, `field × location`. `language`, `role`, `task`, `subfield` siguen como groupby per-attempt (no son atributos del GT).

**REUSO:**
- `aggregators.aggregate_parity(df, attribute, gt=df_gt)` (`aggregators.py:263`) — **no se refactoriza**; recibe `df_gt` ya filtrado al subgrupo desde fuera.
- `aggregators.aggregate_diversity(df, attribute)` (`aggregators.py:189`) — **no se refactoriza**; basta añadir `subpop_col` al `groupby` que la envuelve.
- `metrics_pipeline.ipynb:753-757` bloque que calcula terciles per-field — **lo extiendo** cambiando `.groupby('field_en')` por `.groupby(['field_en', 'oa_country_code'])` cuando subpop es location.
- `grouped_metrics.py:108-159` `compute_grouped_metrics_table` ya acepta `filter` por config — **uso esto** para construir una matriz de resultados subpop sin tocar la firma.
- `metrics_pipeline.ipynb:207-227` carga del GT (parquet SS + dedup last-year) — reutilizo.
- `factuality_full.csv` ya tiene `oa_country_code` por autor → no se necesita join nuevo.

**NUEVO (mínimo):**
- 5 líneas en `code/libs/metrics/constants.py`:
  ```python
  FIELD_ORDER = ['Biology', 'Computer Science', 'Mathematics', 'Physics', 'Psychology', 'Sociology']
  LOCATION_ORDER = ['EC', 'JP', 'DE', 'CA', 'ZA']
  BENCHMARK_SUBPOPULATION_DIMS = ['field', 'location']
  BENCHMARK_SUBPOPULATION_COMBOS = [['field'], ['location'], ['field', 'location']]
  ```
- Función helper nueva en `aggregators.py` (~40 líneas):
  ```python
  def aggregate_by_subpop(df, attribute, agg_fn, gt_provider, subpop_cols, **kwargs):
      """For each (subpop value, per-attempt key), calls agg_fn with gt = gt_provider(subpop_value)."""
  ```
  Donde `gt_provider(subpop)` retorna el GT filtrado. Para `subpop_cols=['field']`: filtra el parquet SS por `Field`. Para `['location']`: filtra `factuality_full` por `oa_country_code` (proxy OA-derived). Para `['field','location']`: combina ambos sobre OA.
- Sección nueva en `code/notebooks/analysis/metrics_pipeline.ipynb` titulada "Per sub-population metrics" — itera `BENCHMARK_SUBPOPULATION_COMBOS`, genera tabla larga `(model, language, ..., subpop_dim, subpop_value, metric_name, value)`, guarda como `results/summary/metrics_per_subpop.csv`.
- Plots: **reuso `grouped_metrics.plot_grouped_metrics`** llamándolo una vez por sub-pop combo (loop externo). No refactorizo la función — es más rápido producir N figuras pequeñas que reescribir el plotter. Respeto `LANGUAGE_ORDER`, `FIELD_ORDER`, `LOCATION_ORDER`.

**Archivos críticos:**
- Edita: `code/libs/metrics/aggregators.py` (1 función nueva, no toca existentes), `code/libs/metrics/constants.py` (4 constantes), `code/notebooks/analysis/metrics_pipeline.ipynb` (sección nueva).
- Lee: `factuality_full.csv`, `Researchers_Deduplicated_Genderize_Namsor.parquet`.
- Escribe: `results/summary/metrics_per_subpop.csv` + PNGs en `data/datasets/LLMScholar-Personas/results/figures/` (siguiendo la convención que ya use el notebook).

---

## Orden de ejecución (todo en serie ágil — ~6 h reales)

1. **Tarea 2 primero (~1.5 h):** desbloquea una nueva métrica (afiliación) y enriquece `factuality_full.csv` con la columna que la Tarea 1 puede mostrar en su CSV.
2. **Tarea 1 (~45 min):** el notebook de validación ya puede aprovechar `affiliation_status_auto` para que tu chequeo manual sea más completo.
3. **Tarea 3 (~3 h):** la más densa pero también la que más reutiliza. Empieza por field (más simple, GT directo de SS), luego location (GT proxy de OA), luego field×location.

## Verificación end-to-end

- **Tarea 2:** smoke test 100 filas + 10 verificaciones a mano. Métrica de salud: `affiliation_match / (affiliation_match + affiliation_mismatch) ≥ 0.5` para autores no halucinados (si baja mucho, revisar umbral 80 de `token_set_ratio`).
- **Tarea 1:** concordancia ≥ 80% con `author_status_auto` tras llenado manual. Si baja → discutir con Lisette.
- **Tarea 3:** sanity check — promedio ponderado de parity per-subpop debe aproximarse al parity global (no idéntico, pero del mismo orden). Si difiere por >0.2, hay bug.

## Lo que **no** voy a hacer

- No regenerar `recommendations.csv` (los campos ya están — Hallazgo 1).
- No refactorizar `aggregate_parity`/`aggregate_diversity` (Hallazgo 8 — aceptan GT externo).
- No tocar `gridbar.py`/`gridcons.py`/`grid.py` (Hallazgo 11 — están incompletos).
- No hacer join completo SS↔OA por país (Hallazgo 9 — uso proxy OA-derived y lo documento).
- No introducir abstracciones (clases, registries) ni feature flags.
- No regenerar el pipeline completo de factuality — solo añadir un step (Hallazgo 4).
