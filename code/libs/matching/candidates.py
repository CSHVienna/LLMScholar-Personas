"""
Step B.6a: OpenAlex candidates of every distinct recommended name, with the result of every name-match rule.
Heavy, run once; the decision (thresholds, levels, random pick) is B.6b (matching/decide.py).

Only OpenAlex PEOPLE are candidates (A.2 `is_person`), from ALL of OpenAlex (not only the field population A.4):
a real person outside the field is "exists, not in the field", not "fabricated".

Names are compared as normalised tokens (no case / accents, transliterated; responses.recommendations.normalize_tokens)
on both sides. The recommended first / last name come from the LLM (cleaned, B.3); OpenAlex names (display + every
alternative name) are split by A.2 (groundtruth/names.py).

Rules (cons.MATCH_RULES, strictest first):
    exact_display / exact_alternative   all name tokens equal (any order) to the display / an alternative name
    short                               first + last token equal ('John A. Smith' ~ 'John Smith'); only if the first token
                                        has >= 2 letters (a single letter is the initials rule's job)
    initials                            the recommended first name is initials only: initial + last name equal
    fuzzy                               Jaro-Winkler on full names: the OpenAlex name contains the recommended
                                        surname token (its last token, >= 2 letters) anywhere; score = the better of
                                        the name as written and flipped ('john smith' vs 'smith john' -> 1.0);
                                        jw_first / jw_last compare the parts in that orientation, edit_first is the
                                        relative edit distance of the first names (Damerau-Levenshtein / longer
                                        length). Pairs with jw_full >= floor are kept; per author, the scores of its
                                        best name (highest jw_full) are stored, and that name (jw_name).

Stages (each resumable; outputs written to a temporary name and renamed when complete):
    rec_names   <dir>/rec_names.parquet: every distinct (first_name, last_name) of the recommendations ->
                rec_name ('first|last', normalised) and its keys
    exact       per OpenAlex bucket: <dir>/exact/bucket-<bb>.parquet  (rec_name, author_id, rule booleans)
    fuzzy_recs  <dir>/fuzzy_recs.parquet: recommended names scored by the fuzzy rule (default: only those without
                any exact / short / initials match, the fuzzy rule being the last resort; --fuzzy_all: all)
    oa_tokens   per OpenAlex bucket: <dir>/oa_tokens/bucket=<b>/shard=<s>/  OpenAlex names containing a surname
                token of fuzzy_recs, partitioned by the token's shard
    fuzzy       per shard: <dir>/fuzzy/shard-<ss>.parquet  (rec_name, name_norm, jw_full, jw_first, jw_last,
                jw_flipped, edit_first)
    merge       per OpenAlex bucket: <dir>/bucket-<bb>.parquet, the candidates:
                rec_name, author_id, exact_display, exact_alternative, short, initials,
                jw_full, jw_first, jw_last, jw_flipped, edit_first, jw_name, jw_is_display
                (NULL if no fuzzy pair >= floor)
"""

import shutil
import zlib
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import pandas as pd

from groundtruth.db import sql_path, write_parquet_atomic
from responses.recommendations import is_initials, normalize_tokens

REC_NAMES_FILE = 'rec_names.parquet'
FUZZY_RECS_FILE = 'fuzzy_recs.parquet'
EXACT_DIR, OA_TOKENS_DIR, FUZZY_DIR = 'exact', 'oa_tokens', 'fuzzy'
CANDIDATE_COLUMNS = ['rec_name', 'author_id', 'exact_display', 'exact_alternative', 'short', 'initials',
                     'jw_full', 'jw_first', 'jw_last', 'jw_flipped', 'edit_first', 'jw_name', 'jw_is_display']
_INDEX_COLUMNS = {'name_key', 'name_key_short', 'name_norm', 'first_norm', 'last_norm', 'initials_key',
                  'author_id', 'is_display_name', 'is_person'}


def token_shard(token: str, n_shards: int) -> int:
    """Deterministic shard of a surname token (same in every process and Python version)."""
    return zlib.crc32(token.encode('utf-8')) % n_shards


def shard_file(folder: Path | str, shard: int) -> Path:
    return Path(folder) / f"shard-{shard:02d}.parquet"


# --- rec_names ---------------------------------------------------------------------------------------------------
def rec_name_row(first_name: Optional[str], last_name: Optional[str], n_shards: int) -> Optional[Dict]:
    """Keys of one recommended name (both parts needed), or None."""
    first_tokens, last_tokens = normalize_tokens(first_name or ''), normalize_tokens(last_name or '')
    if not first_tokens or not last_tokens:
        return None
    tokens = first_tokens + last_tokens
    rec_first, rec_last = ' '.join(first_tokens), ' '.join(last_tokens)
    block = last_tokens[-1] if len(last_tokens[-1]) > 1 else None
    return {'rec_name': f"{rec_first}|{rec_last}", 'rec_first': rec_first, 'rec_last': rec_last,
            'name_key': ' '.join(sorted(tokens)),
            'name_key_short': ' '.join(sorted([tokens[0], tokens[-1]])) if len(tokens[0]) > 1 else None,
            'initials_key': f"{first_tokens[0][0]} {rec_last}" if is_initials(first_name) else None,
            'block_token': block, 'shard': token_shard(block, n_shards) if block else None}


def write_rec_names(con: duckdb.DuckDBPyConnection, recommendations_glob: str, output_file: Path | str,
                    n_shards: int) -> pd.DataFrame:
    """Every distinct (first_name, last_name) of the recommendations with its rec_name and keys."""
    pairs = con.execute(f"SELECT DISTINCT first_name, last_name FROM read_parquet({sql_path(recommendations_glob)}, "
                        f"union_by_name = true) WHERE first_name IS NOT NULL AND last_name IS NOT NULL").fetchall()
    rows = []
    for first, last in pairs:
        row = rec_name_row(first, last, n_shards)
        if row:
            rows.append({'first_name': first, 'last_name': last, **row})
    df = pd.DataFrame(rows, columns=['first_name', 'last_name', 'rec_name', 'rec_first', 'rec_last', 'name_key',
                                     'name_key_short', 'initials_key', 'block_token', 'shard'])
    df['shard'] = df['shard'].astype('Int32')
    write_parquet_atomic(df, output_file)
    return df


def check_index(con: duckdb.DuckDBPyConnection, index_file: Path | str) -> None:
    """The A.2 index must have the name parts (re-run A.2 if it was built by an older version)."""
    columns = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet({sql_path(index_file)})").fetchall()}
    missing = _INDEX_COLUMNS - columns
    if missing:
        raise SystemExit(f"{index_file} lacks {sorted(missing)}: re-run A.2 (run_gt_pipeline.sh -s names)")


# --- exact (per OpenAlex bucket) ---------------------------------------------------------------------------------
def exact_query(rec_names_file: Path | str, index_file: Path | str) -> str:
    """Exact / short / initials candidates (three equi-joins; OR-joins would not be hash joins)."""
    r = f"(SELECT DISTINCT rec_name, name_key, name_key_short, initials_key FROM read_parquet({sql_path(rec_names_file)}))"
    i = f"(SELECT * FROM read_parquet({sql_path(index_file)}) WHERE is_person)"
    return f"""
        WITH hits AS (
            SELECT r.rec_name, i.author_id, i.is_display_name AS display, true AS exact, false AS short, false AS initials
            FROM {r} r JOIN {i} i ON i.name_key = r.name_key
            UNION ALL
            SELECT r.rec_name, i.author_id, false, false, true, false
            FROM {r} r JOIN {i} i ON i.name_key_short = r.name_key_short
            UNION ALL
            SELECT r.rec_name, i.author_id, false, false, false, true
            FROM {r} r JOIN {i} i ON i.initials_key = r.initials_key
        )
        SELECT rec_name, author_id,
               bool_or(exact AND display) AS exact_display,
               bool_or(exact AND NOT display) AS exact_alternative,
               bool_or(short) AS short, bool_or(initials) AS initials
        FROM hits GROUP BY rec_name, author_id
    """


def write_exact_bucket(con: duckdb.DuckDBPyConnection, rec_names_file: Path | str, index_file: Path | str,
                       output_file: Path | str) -> int:
    return _copy(con, exact_query(rec_names_file, index_file), output_file)


# --- fuzzy_recs ----------------------------------------------------------------------------------------------------
def write_fuzzy_recs(con: duckdb.DuckDBPyConnection, rec_names_file: Path | str, exact_dir: Path | str,
                     output_file: Path | str, fuzzy_all: bool) -> int:
    """Recommended names scored by the fuzzy rule: with a surname token, and (unless fuzzy_all) without any
    exact / short / initials candidate."""
    unmatched = '' if fuzzy_all else (f"AND rec_name NOT IN (SELECT DISTINCT rec_name FROM "
                                      f"read_parquet({sql_path(Path(exact_dir) / 'bucket-*.parquet')}))")
    query = f"""
        SELECT DISTINCT rec_name, rec_first, rec_last, block_token, shard
        FROM read_parquet({sql_path(rec_names_file)}) WHERE block_token IS NOT NULL {unmatched}
    """
    return _copy(con, query, output_file)


# --- oa_tokens (per OpenAlex bucket) -------------------------------------------------------------------------------
def oa_tokens_query(index_file: Path | str, fuzzy_recs_file: Path | str) -> str:
    return f"""
        WITH t AS (SELECT DISTINCT block_token AS token, shard FROM read_parquet({sql_path(fuzzy_recs_file)})),
        n AS (SELECT DISTINCT name_norm, first_norm, last_norm FROM read_parquet({sql_path(index_file)})
              WHERE is_person AND first_norm IS NOT NULL),
        u AS (SELECT DISTINCT n.*, u.token FROM n, unnest(string_split(n.name_norm, ' ')) AS u(token))
        SELECT t.shard, u.token, u.name_norm, u.first_norm, u.last_norm FROM u JOIN t USING (token)
    """


def write_oa_tokens_bucket(con: duckdb.DuckDBPyConnection, index_file: Path | str, fuzzy_recs_file: Path | str,
                           output_dir: Path | str, bucket: int) -> int:
    """<output_dir>/bucket=<b>/shard=<s>/*.parquet (via a temporary folder, renamed when complete)."""
    final = Path(output_dir) / f"bucket={bucket}"
    tmp = Path(output_dir) / f"_tmp_bucket={bucket}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    row = con.execute(f"COPY ({oa_tokens_query(index_file, fuzzy_recs_file)}) TO {sql_path(tmp)} "
                      f"(FORMAT parquet, PARTITION_BY (shard), OVERWRITE_OR_IGNORE, COMPRESSION zstd)").fetchone()
    tmp.rename(final)
    return int(row[0]) if row else 0


def oa_tokens_done(output_dir: Path | str, bucket: int) -> bool:
    return (Path(output_dir) / f"bucket={bucket}").is_dir()


# --- fuzzy (per shard) ------------------------------------------------------------------------------------------------
def fuzzy_query(fuzzy_recs_file: Path | str, oa_files: List[str], shard: int, floor: float) -> str:
    files = '[' + ', '.join(sql_path(f) for f in oa_files) + ']'
    return f"""
        WITH r AS (SELECT rec_name, rec_first, rec_last, block_token FROM read_parquet({sql_path(fuzzy_recs_file)})
                   WHERE shard = {int(shard)}),
        o AS (SELECT DISTINCT token, name_norm, first_norm, last_norm FROM read_parquet({files}, hive_partitioning = false)),
        p AS (SELECT r.rec_name, r.rec_first, r.rec_last, o.name_norm, o.first_norm, o.last_norm,
                     jaro_winkler_similarity(r.rec_first || ' ' || r.rec_last, o.name_norm) AS as_written,
                     jaro_winkler_similarity(r.rec_last || ' ' || r.rec_first, o.name_norm) AS flipped
              FROM r JOIN o ON o.token = r.block_token)
        SELECT rec_name, name_norm, greatest(as_written, flipped) AS jw_full,
               CASE WHEN flipped > as_written THEN jaro_winkler_similarity(rec_first, last_norm)
                    ELSE jaro_winkler_similarity(rec_first, first_norm) END AS jw_first,
               CASE WHEN flipped > as_written THEN jaro_winkler_similarity(rec_last, first_norm)
                    ELSE jaro_winkler_similarity(rec_last, last_norm) END AS jw_last,
               flipped > as_written AS jw_flipped,
               damerau_levenshtein(rec_first, CASE WHEN flipped > as_written THEN last_norm ELSE first_norm END)
                   / greatest(length(rec_first), length(CASE WHEN flipped > as_written THEN last_norm ELSE first_norm END))
                   AS edit_first
        FROM p WHERE greatest(as_written, flipped) >= {float(floor)}
    """


def write_fuzzy_shard(con: duckdb.DuckDBPyConnection, fuzzy_recs_file: Path | str, oa_tokens_dir: Path | str,
                      output_file: Path | str, shard: int, floor: float) -> int:
    oa_files = sorted(str(p) for p in Path(oa_tokens_dir).glob(f"bucket=*/shard={shard}/*.parquet"))
    if not oa_files:                                      # no OpenAlex name holds a surname token of this shard
        query = ("SELECT NULL::VARCHAR AS rec_name, NULL::VARCHAR AS name_norm, NULL::DOUBLE AS jw_full, "
                 "NULL::DOUBLE AS jw_first, NULL::DOUBLE AS jw_last, NULL::BOOLEAN AS jw_flipped, "
                 "NULL::DOUBLE AS edit_first WHERE false")
    else:
        query = fuzzy_query(fuzzy_recs_file, oa_files, shard, floor)
    return _copy(con, query, output_file)


# --- merge (per OpenAlex bucket) --------------------------------------------------------------------------------------
def merge_query(exact_file: Path | str, fuzzy_dir: Path | str, index_file: Path | str) -> str:
    """Candidates of one bucket: exact rules + the author's best fuzzy pair (by jw_full)."""
    return f"""
        WITH f AS (
            SELECT f.rec_name, i.author_id, max(f.jw_full) AS jw_full,
                   arg_max(f.jw_first, f.jw_full) AS jw_first, arg_max(f.jw_last, f.jw_full) AS jw_last,
                   arg_max(f.jw_flipped, f.jw_full) AS jw_flipped, arg_max(f.edit_first, f.jw_full) AS edit_first,
                   arg_max(f.name_norm, f.jw_full) AS jw_name, arg_max(i.is_display_name, f.jw_full) AS jw_is_display
            FROM read_parquet({sql_path(Path(fuzzy_dir) / 'shard-*.parquet')}) f
            JOIN (SELECT * FROM read_parquet({sql_path(index_file)}) WHERE is_person) i ON i.name_norm = f.name_norm
            GROUP BY f.rec_name, i.author_id
        )
        SELECT rec_name, author_id,
               coalesce(e.exact_display, false) AS exact_display, coalesce(e.exact_alternative, false) AS exact_alternative,
               coalesce(e.short, false) AS short, coalesce(e.initials, false) AS initials,
               f.jw_full, f.jw_first, f.jw_last, f.jw_flipped, f.edit_first, f.jw_name, f.jw_is_display
        FROM read_parquet({sql_path(exact_file)}) e
        FULL OUTER JOIN f USING (rec_name, author_id)
        ORDER BY rec_name, author_id
    """


def write_merged_bucket(con: duckdb.DuckDBPyConnection, exact_file: Path | str, fuzzy_dir: Path | str,
                        index_file: Path | str, output_file: Path | str) -> int:
    return _copy(con, merge_query(exact_file, fuzzy_dir, index_file), output_file)


def _copy(con: duckdb.DuckDBPyConnection, query: str, output_file: Path | str) -> int:
    """COPY a query to <output_file> via a temporary file; returns the number of rows."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_file.with_name(output_file.name + '.tmp')
    row = con.execute(f"COPY ({query}) TO {sql_path(tmp)} (FORMAT parquet, COMPRESSION zstd)").fetchone()
    tmp.replace(output_file)
    return int(row[0]) if row else 0

