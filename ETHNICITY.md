# Inferencia y Evaluación de Etnia Percibida

Documento descriptivo de los procedimientos implementados en el repositorio `LLMScholar-Personas` para (a) inferir la etnia percibida de investigadores y autores recomendados, y (b) evaluar la calidad de dicha inferencia. El objetivo es servir de insumo para una redacción académica posterior.

---

## 1. Motivación

El benchmark mide, entre otras dimensiones, la **diversidad** y **paridad demográfica** de las recomendaciones de autores generadas por LLMs bajo prompting de personas. Para ello se requiere una etiqueta de etnia consistente tanto sobre el ground truth (investigadores reales en Semantic Scholar) como sobre los autores recomendados por los modelos. Dado que ni Semantic Scholar ni OpenAlex publican esta información, se infiere a partir del **nombre** del autor.

Se adopta explícitamente la noción de **etnia percibida** (no auto-declarada): la etiqueta no pretende representar la identidad real del individuo, sino la categoría que un observador externo le asignaría a partir de su nombre. Esto está en línea con la literatura previa sobre auditoría algorítmica de sesgo demográfico (p. ej. trabajos basados en `ethnicolr` y modelos derivados).

---

## 2. Método de inferencia

Se implementa un **modelo en cascada de dos niveles** con fallback explícito a `Unknown`. La cascada se prioriza por capacidad esperada del modelo (transformer > LSTM) y se controla por un umbral de confianza común.

### 2.1 Nivel 1 — DemographicX (transformer)

- **Modelo:** `liamliang/demographics_race_v2`, distribuido en HuggingFace Hub.
- **Arquitectura:** transformer tipo BERT con embeddings combinados a nivel de palabra y de carácter, lo que permite generalizar a nombres no vistos en entrenamiento.
- **Entrada:** nombre completo (una sola cadena).
- **Salida:** distribución de probabilidad sobre cuatro clases: `White`, `Hispanic or Latino`, `Black or African American`, `Asian`.
- **Procesamiento:** inferencia en *batch* de tamaño 64 sobre GPU cuando está disponible.

### 2.2 Nivel 2 — EthnicolR (fallback LSTM)

Cuando la predicción del Nivel 1 no supera el umbral de confianza, se delega en:

- **Modelo:** LSTM entrenado sobre el registro de votantes del estado de Florida, provisto por el paquete `ethnicolr`.
- **Entrada:** nombre y apellido separados.
- **Salida:** mapeada al mismo espacio de etiquetas que Nivel 1 mediante:
  - `nh_white` → `White`
  - `asian` → `Asian`
  - `hispanic` → `Hispanic or Latino`
  - `nh_black` → `Black or African American`

### 2.3 Umbral y fallback

- **Umbral de confianza:** 0.5, aplicado de forma uniforme a ambos niveles.
- **Categoría residual:** si tras los dos niveles ninguna predicción supera el umbral, la observación recibe la etiqueta `Unknown`.
- **Metadatos persistidos por observación:**
  - `perceived_ethnicity`: clase final.
  - `__ethnicity_confidence`: probabilidad asociada.
  - `__ethnicity_source`: `demographicx`, `ethnicolr` o `unknown` (permite auditoría posterior y análisis de sensibilidad por origen).

### 2.4 Aplicación

- Sobre el **ground truth** (`Researchers_Deduplicated_Genderize_Namsor.parquet`, Semantic Scholar) la inferencia se ejecuta **una sola vez por investigador único** (deduplicación por `Researcher_id`), produciendo el artefacto `researcher_ethnicity_lookup.csv`.
- Sobre las **recomendaciones del LLM**, la asignación se hace por *lookup* `(nombre, apellido) → perceived_ethnicity`; los pares no encontrados se etiquetan como `Unknown`. Esto evita doble inferencia y garantiza consistencia entre las dos poblaciones que después se comparan.

### 2.5 Esquema de etiquetas

Conjunto final de cinco clases: `{White, Asian, Black or African American, Hispanic or Latino, Unknown}`.

---

## 3. Evaluación

La evaluación combina (a) una medición del **techo de la tarea** mediante acuerdo entre anotadores humanos y (b) métricas indirectas en los análisis downstream.

### 3.1 Inter-annotator agreement sobre etnia percibida

- **Diseño:** dos anotadores humanos (`Leen`, `v2`) etiquetan de forma independiente la etnia percibida a partir del **nombre únicamente**, sin acceso a fotografía, afiliación ni publicaciones.
- **Tamaño de muestra:** 100 ítems tras descartar *skips*.
- **Métricas calculadas:**
  - Acuerdo bruto (`p`): **0.6400** (64/100).
  - Cohen's κ: **0.4874**.
  - Krippendorff's α (nominal): **0.4747**.
- **Distribución de etiquetas (asimétrica):**
  - `Leen`: 62 White, 14 Latino, 13 Asian, 6 Unknown, 5 Black.
  - `v2`: 39 White, 20 Black, 17 Latino, 13 Unknown, 11 Asian.
- **Patrones de desacuerdo:** concentrados en los pares `White ↔ Black` (14 casos) y `White ↔ Latino` (4 casos).
- **Lectura:** la tarea de inferir etnia desde el nombre es **intrínsecamente ambigua**, lo que impone un techo a la calidad alcanzable por cualquier modelo automático. Cualquier evaluación del cascada debe interpretarse contra ese techo, no contra un oráculo perfecto.

### 3.2 Comparación con una tarea de etiquetado adyacente

Para contextualizar el κ obtenido, se replicó el mismo protocolo de doble anotación sobre una tarea de **validez de respuesta del LLM** (categorías: `cleaned`, `unchanged`, `refused`, `empty`, `invalid`, `fixed_dict`), también con n=100:

- Acuerdo bruto: **0.8600**.
- Cohen's κ: **0.8288**.
- Krippendorff's α: **0.8296**.

La brecha (κ ≈ 0.83 vs κ ≈ 0.49) confirma que el déficit de acuerdo en la tarea de etnia no proviene del protocolo de anotación, sino de la ambigüedad inherente del constructo.

### 3.3 Validación manual de factuality (procedimiento análogo)

Para los autores recomendados, una muestra aleatoria (`seed=42`, n=5 *requests*) fue verificada manualmente contra Semantic Scholar y OpenAlex sin asistencia del LLM, registrando `found_in_ss_manual`, `found_in_oa_manual`, `affiliation_correct_manual`, `field_correct_manual`. La concordancia con el clasificador automático se computa como
`(found_in_ss_manual | found_in_oa_manual) == (author_status_auto != 'hallucinated')`.
Este procedimiento es paralelo en espíritu al de §3.1 pero aplica a *factuality*; se documenta aquí para mostrar que el mismo principio (anclar las inferencias a juicio humano sobre una muestra) se usa de forma consistente en el pipeline.

### 3.4 Uso downstream: diversity y parity

La etnia inferida se consume en dos métricas agregadas:

- **Diversidad** — entropía de Shannon normalizada
  H<sub>norm</sub> = H / log(K),
  con K = 4 (clases efectivas, excluyendo `Unknown` del soporte para no inflar artificialmente la dispersión).

- **Paridad** — complemento de la distancia de variación total
  Parity = 1 − TV, con TV = ½·Σ<sub>k</sub>|p<sub>rec</sub>(k) − p<sub>gt</sub>(k)|,
  comparando la distribución de etnia entre las recomendaciones y el ground truth correspondiente.

Existen variantes por sub-población (por campo disciplinar, por país, y por la interacción campo × país) que reusan la misma definición sobre cortes del ground truth.

---

## 4. Resumen del flujo

1. Inferir etnia sobre investigadores únicos del ground truth → `researcher_ethnicity_lookup.csv`.
2. Resolver la etnia de cada autor recomendado por el LLM vía *lookup* sobre el mismo artefacto.
3. Calibrar las expectativas evaluativas con un κ de doble anotación humana sobre 100 nombres (κ ≈ 0.49).
4. Alimentar las etiquetas resultantes en las métricas de diversidad (entropía normalizada) y paridad (1 − TV) agregadas por modelo, temperatura, tarea y sub-población.

---

## 5. Limitaciones declaradas

- La etiqueta es **percibida**, no auto-declarada; no debe interpretarse como identidad real.
- Los modelos base (`liamliang/demographics_race_v2`, `ethnicolr`) están entrenados con datos predominantemente estadounidenses, lo que sesga el espacio de clases y la distribución a priori.
- El esquema de cuatro clases más `Unknown` colapsa heterogeneidad relevante (p. ej. dentro de `Asian`).
- El techo humano (κ ≈ 0.49) implica que evaluaciones automáticas de exactitud por encima de ese valor deben interpretarse con cautela.
- `Unknown` se excluye del soporte de entropía y paridad; esto es una decisión metodológica explícita y altera la sensibilidad de ambas métricas frente a muestras con alta tasa de fallo del clasificador.
