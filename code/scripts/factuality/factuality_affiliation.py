"""
factuality_affiliation.py — Step 3.5 of the factuality pipeline.

PLAN.md Tarea 2 — Factuality de afiliación.
Estructura clonada de factuality_location.py (Hallazgo 3). Reusa
normalize_name de factuality_openalex.py (Hallazgo 2) y rapidfuzz
(ya en deps por factuality_author_jw.py).

Reads the output of factuality_location.py and decides whether the
LLM-claimed `current_affiliations` for each recommended author match any of
the institutions the author has actually been affiliated with according to
OpenAlex (historical, not just the most-recent).

Ground truth: OpenAlex. Semantic Scholar's deduplicated parquet does not
carry per-researcher institution, so affiliation factuality can only be
verified against OA.

OA institution history is reconstructed from the existing year-chunk
parquets at /data/asanchez/duckdb_enrich/oa_works_agg_chunks (built by
factuality_openalex.py). Each chunk holds one (country, inst_name) per
(oa_id, year_chunk) via ARG_MAX over publication_year, so DISTINCT across
all 20 chunks yields the author's institution history at chunk-granularity.

Match logic: rapidfuzz.fuzz.token_set_ratio with a threshold of 80 between
each LLM affiliation string and each OA historical institution string.
A row is `affiliation_match` if ANY (llm, oa) pair clears the threshold.

Output columns added:
  affiliation_llm                 serialized list of LLM affiliations
  affiliation_oa_all              serialized list of OA historical institutions
  affiliation_best_match_score    best token_set_ratio score (0-100) or None
  affiliation_best_match_oa       OA institution that produced the best score
  affiliation_status              {affiliation_match | affiliation_mismatch
                                   | affiliation_unknown | not_applicable}

Usage (from code/scripts/factuality/):
  python factuality_affiliation.py \\
      --input  ../../../results/summary/factuality_location.csv \\
      --output ../../../results/summary/factuality_affiliation.csv
"""

import argparse
import ast
import json
import logging
import os

import pandas as pd
from rapidfuzz import fuzz

# Reuse the existing accent/case normalizer from the OA step so matching is
# stable across mojibake ("Université" vs "Universite") and case differences.
from factuality_openalex import normalize_name as _normalize_for_match

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

STATUS_MATCH          = "affiliation_match"
STATUS_MISMATCH       = "affiliation_mismatch"
STATUS_UNKNOWN        = "affiliation_unknown"
STATUS_NOT_APPLICABLE = "not_applicable"

AUTHOR_HALLUCINATED = "hallucinated"

# Threshold on rapidfuzz.fuzz.token_set_ratio (0-100). 85 was chosen because
# 80 lets the shared token "University" alone produce false positives like
# "University of Cape Town" vs "Yale University" (score 80). 85 still passes
# legitimate fuzzy matches (LLM-verbose strings vs short OA names score 100
# via subset-token mechanics) while rejecting the single-shared-token case.
TOKEN_SET_RATIO_THRESHOLD = 85

# Where factuality_openalex.py wrote per-year aggregations.
WORKS_AGG_DIR = "/data/asanchez/duckdb_enrich/oa_works_agg_chunks"


# ── Parsers ────────────────────────────────────────────────────────────────────

def parse_llm_affiliations(raw) -> list[str]:
    """Parse the `current_affiliations` cell into a list of affiliation strings.

    The cell is typically a Python-literal string like
        "[{'position': '...', 'affiliation': 'MIT'}]"
    but may also be None, NaN, empty, or already a list.
    """
    # NaN o None → sin afiliaciones.
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    # Si ya viene como lista (raro pero posible si pandas no la serializó), úsala.
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        s = raw.strip()
        # Sentinels comunes de "vacío" — antes de intentar parsear.
        if not s or s in ("[]", "null", "None"):
            return []
        # Primero ast.literal_eval (acepta comillas simples al estilo Python),
        # luego json como fallback para strings JSON estrictas.
        try:
            items = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            try:
                items = json.loads(s)
            except Exception:
                return []
    else:
        return []

    # Si lo que se parseó no es una lista (e.g. un dict suelto), descartar.
    if not isinstance(items, list):
        return []

    # Extraer el string de afiliación de cada item: aceptamos dicts con
    # distintas convenciones de keys o strings sueltos.
    affs: list[str] = []
    for it in items:
        if isinstance(it, dict):
            v = it.get("affiliation") or it.get("Affiliation") or it.get("institution")
            if isinstance(v, str) and v.strip():
                affs.append(v.strip())
        elif isinstance(it, str) and it.strip():
            affs.append(it.strip())
    return affs


# ── OA institution history ─────────────────────────────────────────────────────

def fetch_oa_institutions(oa_ids: list[str]) -> dict[str, list[str]]:
    """Return {oa_id: [inst_name, …]} from the existing chunk parquets.

    Uses DuckDB read_parquet on the glob — does not open the full OA DB.
    If no chunks are on disk, returns an empty dict and the caller will
    fall back to oa_last_institution from the input CSV.
    """
    # Glob sobre los chunks ya materializados por factuality_openalex.py
    # (uno por bin de years; UNNEST sobre authorships ya hecho ahí).
    chunk_glob = os.path.join(WORKS_AGG_DIR, "chunk_*.parquet")
    # Si la carpeta no existe o está vacía, abortamos limpio: el caller usará
    # oa_last_institution como proxy de "histórico" (no es lo ideal pero permite
    # seguir corriendo el pipeline en máquinas sin los chunks pre-generados).
    if not any(os.path.exists(os.path.join(WORKS_AGG_DIR, f))
               for f in (os.listdir(WORKS_AGG_DIR) if os.path.isdir(WORKS_AGG_DIR) else [])):
        logger.warning("OA chunk parquets not found at %s — falling back to oa_last_institution only",
                       WORKS_AGG_DIR)
        return {}

    # duckdb es dep blanda — si no está instalado, también fallback.
    try:
        import duckdb  # noqa: F401
    except ImportError:
        logger.warning("duckdb not installed — falling back to oa_last_institution only")
        return {}

    # Deduplicar oa_ids: hay ~4M rows en el CSV pero muchos menos autores únicos.
    unique = sorted({i for i in oa_ids if isinstance(i, str) and i})
    if not unique:
        return {}

    logger.info("OA institutions: scanning %s for %d unique oa_ids …",
                chunk_glob, len(unique))
    con = duckdb.connect()
    try:
        # Threads = todos los cores: la query es CPU-bound (LIST DISTINCT sobre
        # decenas de millones de rows).
        cores = os.cpu_count() or 8
        con.execute(f"SET threads = {cores}")
        # Registramos los oa_ids como tabla virtual para hacer un JOIN
        # eficiente en lugar de un IN gigante.
        con.register("query_oa_ids", pd.DataFrame({"id": unique}))
        # Una sola query: por cada autor, LIST(DISTINCT inst_name) sobre
        # todos los chunks → su historia de afiliaciones completa.
        rows = con.execute(f"""
            SELECT p.oa_id, LIST(DISTINCT p.inst_name) AS institutions
              FROM read_parquet('{chunk_glob}') p
              JOIN query_oa_ids q ON p.oa_id = q.id
             WHERE p.inst_name IS NOT NULL
             GROUP BY p.oa_id
        """).fetchall()
    except Exception as exc:
        # No queremos tirar el pipeline si DuckDB falla — degradamos a fallback.
        logger.warning("OA institutions query failed: %s", exc)
        return {}
    finally:
        # Cerrar la conexión sí o sí (libera RAM y locks).
        try:
            con.close()
        except Exception:
            pass

    # Construir el dict final filtrando entradas vacías o no-string.
    out: dict[str, list[str]] = {}
    for oa_id, inst_list in rows:
        out[oa_id] = [s for s in (inst_list or []) if isinstance(s, str) and s.strip()]
    logger.info("OA institutions: resolved %d / %d oa_ids", len(out), len(unique))
    return out


# ── Per-row decision ───────────────────────────────────────────────────────────

def match_affiliations(llm_affs: list[str], oa_affs: list[str]) -> tuple[int | None, str | None]:
    """Return (best_score, best_oa_name) across the cartesian product."""
    # Sin afiliaciones de un lado u otro no hay match posible — score = None.
    if not llm_affs or not oa_affs:
        return None, None
    # Pre-normalizar una vez por string (lowercase + strip accents):
    # estabiliza "Université" vs "Universite" y similares.
    llm_norm = [_normalize_for_match(s) for s in llm_affs]
    oa_norm  = [_normalize_for_match(s) for s in oa_affs]
    best_score = -1
    best_oa = None
    # Producto cartesiano LLM × OA — quedarse con el mejor par.
    for la, la_n in zip(llm_affs, llm_norm):
        if not la_n:
            continue
        for oa, oa_n in zip(oa_affs, oa_norm):
            if not oa_n:
                continue
            # token_set_ratio maneja bien diferencias de orden y
            # subconjuntos ("MIT" vs "Massachusetts Institute of Technology").
            s = fuzz.token_set_ratio(la_n, oa_n)
            if s > best_score:
                best_score = s
                best_oa = oa  # guardamos el nombre OA original, no el normalizado
    return int(best_score), best_oa


def _decide(llm_affs: list[str], oa_affs: list[str], author_status: str
            ) -> tuple[str | None, str | None, int | None, str | None, str]:
    """Pure decision: return (llm_json, oa_json, score, best_oa, status)."""
    # Calcular el mejor score y serializar las listas a JSON (para guardarlas
    # legibles en el CSV de salida).
    score, best_oa = match_affiliations(llm_affs, oa_affs)
    llm_json = json.dumps(llm_affs, ensure_ascii=False) if llm_affs else None
    oa_json  = json.dumps(oa_affs,  ensure_ascii=False) if oa_affs  else None

    # Autor halucinado → no tiene sentido chequear afiliación: not_applicable
    # (mismo schema que factuality_location.py).
    if author_status == AUTHOR_HALLUCINATED:
        return llm_json, oa_json, score, best_oa, STATUS_NOT_APPLICABLE
    # Falta info de un lado u otro → no podemos decidir match/mismatch.
    if not llm_affs or not oa_affs:
        return llm_json, oa_json, score, best_oa, STATUS_UNKNOWN
    # Por encima del umbral → match; por debajo → mismatch.
    if score is not None and score >= TOKEN_SET_RATIO_THRESHOLD:
        return llm_json, oa_json, score, best_oa, STATUS_MATCH
    return llm_json, oa_json, score, best_oa, STATUS_MISMATCH


# ── Main ───────────────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str) -> None:
    logger.info("Loading: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    logger.info("Rows: %d", len(df))

    # Lookup OA → lista de instituciones históricas (una sola query DuckDB
    # para todos los autores únicos del CSV).
    oa_ids = df["oa_id"].dropna().astype(str).tolist() if "oa_id" in df.columns else []
    oa_inst_cache = fetch_oa_institutions(oa_ids)

    # Cache parsed LLM affiliations by raw string (815k unique vs ~4M rows).
    # Parsear el JSON-ish de cada row es caro, pero muchos rows comparten el
    # mismo raw string (mismo autor recomendado por múltiples LLMs / runs).
    raw_aff_series = df["current_affiliations"]
    unique_raw = raw_aff_series.drop_duplicates()
    logger.info("Parsing %d unique current_affiliations strings …", len(unique_raw))
    llm_parse_cache: dict = {}
    for raw in unique_raw:
        # Guardamos como tuple para poder usarlas como key de dict abajo.
        llm_parse_cache[raw] = tuple(parse_llm_affiliations(raw))

    # Build OA-affs lookup per row: prefer historical cache, fall back to oa_last_institution.
    def _oa_for(oa_id, last_inst) -> tuple:
        # Si tenemos el histórico completo de DuckDB, úsalo (mejor match).
        if isinstance(oa_id, str) and oa_id in oa_inst_cache:
            return tuple(oa_inst_cache[oa_id])
        # Sino, caemos a la última institución conocida — degradación elegante
        # para máquinas sin los chunks parquet.
        if isinstance(last_inst, str) and last_inst.strip():
            return (last_inst,)
        return ()

    # Decide once per unique (llm_tuple, oa_tuple, author_status). Build the key
    # cheaply via row-wise tuple construction (no fuzzy work here).
    # Importante: el fuzzy matching (token_set_ratio) es lo caro — lo hacemos
    # una vez por combo único, no por row.
    logger.info("Building per-row keys …")
    llm_tuples = raw_aff_series.map(llm_parse_cache).to_list()
    oa_ids_arr   = df["oa_id"].to_list() if "oa_id" in df.columns else [None]*len(df)
    last_inst_arr = df["oa_last_institution"].to_list() if "oa_last_institution" in df.columns else [None]*len(df)
    status_arr   = df["author_status"].to_list() if "author_status" in df.columns else [None]*len(df)

    decision_cache: dict = {}
    n_rows = len(df)
    # Construir la key por row: (afiliaciones LLM, afiliaciones OA, author_status).
    keys: list[tuple] = [None]*n_rows  # type: ignore[list-item]
    for i in range(n_rows):
        oa_tuple = _oa_for(oa_ids_arr[i], last_inst_arr[i])
        keys[i] = (llm_tuples[i], oa_tuple, status_arr[i])
    unique_keys = set(keys)
    logger.info("Deciding %d unique (LLM-affs, OA-affs, author_status) combos …", len(unique_keys))

    # Aquí se hace el trabajo caro: una decisión por combo único.
    for k in unique_keys:
        llm_t, oa_t, st = k
        decision_cache[k] = _decide(list(llm_t), list(oa_t), st)

    # Re-expandir los resultados a las N rows del CSV original.
    logger.info("Materializing columns …")
    aff_llm  = [None]*n_rows
    aff_oa   = [None]*n_rows
    aff_sc   = [None]*n_rows
    aff_best = [None]*n_rows
    aff_st   = [None]*n_rows
    for i, k in enumerate(keys):
        aff_llm[i], aff_oa[i], aff_sc[i], aff_best[i], aff_st[i] = decision_cache[k]

    # Adjuntar las 5 columnas nuevas al DataFrame.
    df["affiliation_llm"]              = aff_llm
    df["affiliation_oa_all"]           = aff_oa
    df["affiliation_best_match_score"] = aff_sc
    df["affiliation_best_match_oa"]    = aff_best
    df["affiliation_status"]           = aff_st

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows → %s", len(df), output_path)

    n = len(df)
    logger.info("Affiliation status distribution:")
    for status, count in df["affiliation_status"].value_counts().items():
        logger.info("  %-25s %6d  (%.1f%%)", status, count, 100 * count / n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3.5: verify LLM `current_affiliations` against the author's "
                    "OpenAlex institution history"
    )
    parser.add_argument("--input",  required=True,
                        help="Path to factuality_location.csv (output of step 3)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    run(args.input, args.output)


if __name__ == "__main__":
    main()
