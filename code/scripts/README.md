# Scripts

Esta carpeta contiene los scripts del proyecto, organizados por dominio:

```
scripts/
├── prompting/    Genera y procesa prompts batch a los LLMs
├── annotation/   Herramientas interactivas de anotación manual
├── enrichment/   Enriquece resultados con datos de OpenAlex
├── factuality/   Pipeline de chequeo factual (5 pasos)
└── ethnicity/    Inferencia étnica sobre el ground truth
```

Para detalles de cada pipeline, ver los docstrings de los scripts dentro de cada subcarpeta.

---

## Batch Processing (`prompting/`)

Tres scripts trabajan en conjunto para crear y procesar combinaciones de prompts localizados:

1. **`batch_params.py`** — primero. Traduce los archivos de parámetros (instrucciones, ubicaciones, inputs) al idioma objetivo.
2. **`batch_prompt.py`** — después de `batch_params`. Genera combinaciones de prompts a partir de los parámetros traducidos.
3. **`batch_parse_results.py`** — después de recolectar respuestas de los LLMs. Unifica todas las respuestas en `recommendations.csv` y `summary.csv`.

### Prerequisites

Antes de ejecutar cualquier script, configurar `PYTHONPATH` desde `code/scripts/prompting/`:
```bash
export PYTHONPATH="$PYTHONPATH:../../libs"
```

### batch_params.py

Genera todos los parámetros en inglés y los traduce al idioma objetivo usando la API de OpenAI.

**Parámetros**
- `-l, --language` (opcional): Código del idioma. Debe ser uno de los definidos en `cons.LANGUAGES`. Default: `cons.LANG_EN`.
- `-o, --output-dir` (requerido): Directorio de salida. Default: `../../data/context/`

**Configuración**

Requiere un `config.ini` con credenciales de la API de OpenAI en `../../../config.ini`.

**Ejemplos**
```bash
# Desde code/scripts/prompting/
python batch_params.py -l german -o ../../data/context/
python batch_params.py -l spanish -o ../../data/context/
```

### batch_prompt.py

Genera combinaciones de prompts desde los parámetros traducidos.

**Parámetros**
- `-c, --combination_id` (opcional): ID de combinación a mostrar (0-indexed).
- `-l, --language` (opcional): Idioma (debe coincidir con el usado en `batch_params.py`).
- `-o, --output-dir` (requerido): Directorio de salida. Default: `../../data/context/`

**Ejemplos**
```bash
# Desde code/scripts/prompting/
python batch_prompt.py -c 0 -l german -o ../../data/context/
python batch_prompt.py -c 5 -l german
python batch_prompt.py -c 42 -l spanish
```

### batch_parse_results.py

**Parámetros**
- `--results_dir` (requerido): Directorio con las respuestas (`.json`), ej. `../../../results`
- `--output_dir` (requerido): Directorio de salida.
- `--model` (opcional): Nombre del modelo (ver `data/context/models.txt`).
- `--language` (opcional): Idioma (`spanish`, `english`, `german`).

**Ejemplo**
```bash
# Desde code/scripts/prompting/, parsear en paralelo todas las combinaciones
nice -n 10 parallel -j 20 python batch_parse_results.py \
  --results_dir ../../../results \
  --output_dir ../../../results/summary_parallel \
  --model {1} --language {2} \
  :::: ../../../data/context/models.txt ::: english german spanish
```

### Workflow completo

```bash
# Desde code/scripts/prompting/
export PYTHONPATH="$PYTHONPATH:../../libs"

# 1. Traducir parámetros a alemán
python batch_params.py -l german -o ../../data/context/

# 2. Generar y visualizar combinación 0
python batch_prompt.py -c 0 -l german -o ../../data/context/
```

### GNU Parallel

```bash
# Procesar combinaciones 0-719 en paralelo (8 jobs simultáneos)
parallel -j 8 python batch_prompt.py -c {} -l german -o ../../data/context/ ::: {0..719}

# Múltiples idiomas
parallel -j 4 python batch_params.py -l {} -o ../../data/context/ ::: english german spanish

# Guardar salida en archivos separados
parallel -j 8 "python batch_prompt.py -c {} -l german -o ../../data/context/ > output_{}.txt" ::: {0..719}

# Con barra de progreso
parallel --bar -j 8 python batch_prompt.py -c {} -l german -o ../../data/context/ ::: {0..719}
```

### Salida

- `batch_params.py` produce JSON traducidos en `{output_dir}/{language}/`:
  - `instructions.json`, `locations.json`, `input.json`
- `batch_prompt.py` produce:
  - `all_prompts.txt` — todas las combinaciones generadas
  - Output en consola con la combinación seleccionada

### Notas

- Siempre correr `batch_params.py` antes de `batch_prompt.py` para cada idioma.
- El `combination_id` debe estar dentro del rango válido (revisar el output total).
- GNU Parallel no es requerido pero es altamente recomendado.
