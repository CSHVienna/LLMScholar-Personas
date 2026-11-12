"""
Step A.3 (demographics) in five resumable stages. The models depend on the name only, and names recur across buckets
(40.4M authors, 31.3M distinct full names), so every distinct name is inferred ONCE per model, over all buckets: the
first name for nomquamgender, the full name for the raceBERT race and ethnicity models (the slow part; each with its
own normalised name, MODELS; sorted by length: batches of similar length need little padding).

    keys       per author bucket:  <demographics_dir>/keys/bucket-<bb>.parquet
                   author_id, gender_first_name, gender_name (lower-case first name),
                   race_name, race_shard, ethnicity_name, ethnicity_shard
               (population: people with >= 1 work in a prompt field; <model>_shard = crc32(<model>_name) % n_shards)
    gender     one process (needs every keys bucket; a lookup table, ~1 min): <demographics_dir>/gender/first_names.parquet
                   gender_name, p_female
    race       per name shard (needs every keys bucket): <demographics_dir>/race/shard-<ss>.parquet
                   race_name, p_race_asian, p_race_black, p_race_hispanic, p_race_white, p_race_other    (CPU)
    ethnicity  per name shard (needs every keys bucket): <demographics_dir>/ethnicity/shard-<ss>.parquet
                   ethnicity_name, p_eth_<13 classes>                                              (GPU if any)
               both checkpointed every chunk_names names in shard-<ss>.parts/ (a crash resumes after the last part)
    merge      per author bucket (needs gender + every shard of both models): <demographics_dir>/bucket-<bb>.parquet
                   author_id, gender_first_name, p_female, p_race_<5 classes>, p_eth_<13 classes>   (the A.4 input)

Each output is written to a temporary name and renamed when complete; <demographics_dir>/_params.json records the
bucket / shard counts (runs with other counts are refused).

Cache (race / ethnicity stages): probabilities computed earlier can be reused (one or more folders; cache_files());
only the names not in the cache go through the model. Two kinds of files are recognised, each used only if it has
the model's columns:
    name level    <model>/shard-<ss>.parquet (+ unfinished shard-<ss>.parts/part-*.parquet) of an earlier run:
                  <model>_name + probabilities, matched by the exact name (preferred)
    author level  a previous demographics output (bucket-<bb>.parquet + unfinished bucket-<bb>.parts/):
                  author_id + probabilities, matched through the keys (author_id -> name)
A name whose text changed since (e.g. text clean-up) is not found by name and is inferred again.
"""

import glob
import hashlib
import json
import math
import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from demographics import ethnicity, race
from demographics.gender import first_name_of
from groundtruth import db
from groundtruth.authorships import check_params
from groundtruth.fields import FieldMapping
from utils import ios

LAYOUT = 'global_names_v3'
KEYS_DIR, GENDER_DIR = 'keys', 'gender'
GENDER_FILE = 'first_names.parquet'
ProbabilityFn = Callable[[List[str]], List[Dict[str, float]]]   # RaceInferrer / EthnicityInferrer.probabilities


@dataclass(frozen=True)
class NameModel:
    """A full-name model of A.3: its folder of name shards (<demographics_dir>/<name>/), key columns and outputs."""
    name: str
    normalise: Callable[[Optional[str]], Optional[str]]
    prob_columns: Tuple[str, ...]

    @property
    def name_column(self) -> str:
        return f"{self.name}_name"

    @property
    def shard_column(self) -> str:
        return f"{self.name}_shard"


RACE = NameModel('race', race.race_name, tuple(race.PROB_COLUMNS))
ETHNICITY = NameModel('ethnicity', ethnicity.ethnicity_name, tuple(ethnicity.PROB_COLUMNS))
MODELS = {m.name: m for m in (RACE, ETHNICITY)}
KEY_COLUMNS = ['author_id', 'gender_first_name', 'gender_name',
               *[c for m in MODELS.values() for c in (m.name_column, m.shard_column)]]
OUTPUT_COLUMNS = ['author_id', 'gender_first_name', 'p_female', *[c for m in MODELS.values() for c in m.prob_columns]]


# --- layout / parameters -----------------------------------------------------------------------------------------
def shard_file(folder: Path | str, shard: int) -> Path:
    return Path(folder) / f"shard-{shard:02d}.parquet"


def model_dir(demographics_dir: Path | str, model: NameModel) -> Path:
    return Path(demographics_dir) / model.name


def name_shard(name: str, n_shards: int) -> int:
    """Deterministic shard of a normalised name (the same in every process and Python version)."""
    return zlib.crc32(name.encode('utf-8')) % n_shards


def check_layout(demographics_dir: Path | str, n_buckets: int, n_shards: int, limit_authors: Optional[int]) -> None:
    """Record the parameters once; refuse to mix outputs written with other counts."""
    check_params(demographics_dir, {'layout': LAYOUT, 'n_buckets': n_buckets, 'n_shards': n_shards,
                                    'limit_authors': limit_authors})


def missing_files(paths: List[Path]) -> List[str]:
    return [p.name for p in paths if not p.exists()]


# --- stage 1: keys per author bucket -----------------------------------------------------------------------------
def population(authors_file: Path | str, fields: List[FieldMapping], limit_authors: Optional[int] = None) -> pd.DataFrame:
    """People with >= 1 work in a prompt field (the broadest population; later thresholds only select subsets)."""
    columns = ['author_id', 'display_name', 'is_person'] + [f.column for f in fields]
    authors = ios.load_parquet(authors_file, columns=columns)
    in_field = (authors[[f.column for f in fields]] >= 1).any(axis=1)
    authors = authors[in_field & authors['is_person'].fillna(False).astype(bool)]
    authors = authors.sort_values('author_id').reset_index(drop=True)
    return authors.head(limit_authors) if limit_authors else authors


def build_keys(authors: pd.DataFrame, n_shards: int) -> pd.DataFrame:
    """Per author: the first name for nomquamgender (and its lower-case lookup key) and, per full-name model, the
    normalised name with its shard."""
    names = authors['display_name'].tolist()
    first = [first_name_of(n) for n in names]
    keys = pd.DataFrame({'author_id': authors['author_id'].to_numpy(), 'gender_first_name': first,
                         'gender_name': [f.lower() if f else None for f in first],
                         **{m.name_column: [m.normalise(n) for n in names] for m in MODELS.values()}})
    keys = ios.replace_surrogates(keys)          # what is written is what is sharded (never two shards for a name)
    for m in MODELS.values():
        keys[m.shard_column] = pd.array([name_shard(n, n_shards) if isinstance(n, str) else None
                                         for n in keys[m.name_column]], dtype='Int32')
    return keys[KEY_COLUMNS]


def write_keys_bucket(authors_file: Path | str, keys_file: Path | str, fields: List[FieldMapping], n_shards: int,
                      limit_authors: Optional[int] = None) -> pd.DataFrame:
    if not Path(authors_file).exists():
        raise FileNotFoundError(f"No bibliometrics file (step A.1): {authors_file}")
    keys = build_keys(population(authors_file, fields, limit_authors), n_shards)
    db.write_parquet_atomic(keys, keys_file)
    return keys


# --- stage 2: nomquamgender once per distinct first name ---------------------------------------------------------
def gender_names(con: duckdb.DuckDBPyConnection, keys_dir: Path | str) -> List[str]:
    """Distinct lower-case first names over all buckets (sorted)."""
    rows = con.execute(f"SELECT DISTINCT gender_name FROM read_parquet({db.sql_path(Path(keys_dir) / 'bucket-*.parquet')}) "
                       f"WHERE gender_name IS NOT NULL").fetchall()
    return sorted(r[0] for r in rows)


def write_gender_table(names: List[str], output_file: Path | str, p_female_fn: Callable) -> pd.DataFrame:
    """p_female per first name (NaN = not in nomquamgender's reference -> unknown later)."""
    p_female = p_female_fn(names) if names else []
    if len(p_female) != len(names):
        raise RuntimeError(f"nomquamgender returned {len(p_female)} results for {len(names)} names")
    df = pd.DataFrame({'gender_name': pd.Series(names, dtype=object), 'p_female': pd.Series(p_female, dtype=float)})
    db.write_parquet_atomic(df, output_file)
    return df


# --- stages 3-4: a full-name model once per distinct name, per shard ----------------------------------------------
def _shard_names_sql(keys_dir: Path | str, shard: int, model: NameModel) -> str:
    return (f"SELECT DISTINCT {model.name_column} FROM read_parquet({db.sql_path(Path(keys_dir) / 'bucket-*.parquet')}) "
            f"WHERE {model.shard_column} = {int(shard)} AND {model.name_column} IS NOT NULL")


def shard_names(con: duckdb.DuckDBPyConnection, keys_dir: Path | str, shard: int, model: NameModel) -> List[str]:
    """Distinct names of one shard over all buckets, sorted by (length, name): deterministic, and batches of
    similar length need little padding."""
    rows = con.execute(_shard_names_sql(keys_dir, shard, model)).fetchall()
    return sorted((r[0] for r in rows), key=lambda n: (len(n), n))


def cache_files(cache_dirs: Optional[Path | str | Sequence[Path | str]]) -> List[str]:
    """Probability files of earlier runs in one or more folders: name-level shard-<ss>.parquet /
    shard-<ss>.parts/part-*.parquet and author-level bucket-<bb>.parquet / bucket-<bb>.parts/part-*.parquet.
    Temporary (.tmp) files are never matched."""
    if not cache_dirs:
        return []
    dirs = [cache_dirs] if isinstance(cache_dirs, (str, Path)) else list(cache_dirs)
    files: List[str] = []
    for cache_dir in dirs:
        if not Path(cache_dir).is_dir():
            raise FileNotFoundError(f"Cache folder not found: {cache_dir}")
        for kind in ('shard', 'bucket'):
            files += sorted(glob.glob(str(Path(cache_dir) / f'{kind}-*.parquet'))
                            + glob.glob(str(Path(cache_dir) / f'{kind}-*.parts' / 'part-*.parquet')))
    return files


def is_name_level(file: str) -> bool:
    """A name-level cache file (a model's shard or its part), else author level."""
    path = Path(file)
    return (path.parent.name if path.parent.name.endswith('.parts') else path.name).startswith('shard-')


def model_cache_files(files: Sequence[str], model: NameModel) -> Tuple[List[str], List[str]]:
    """(name-level, author-level) cache files that hold the model's columns (other files are ignored)."""
    by_name, by_author = [], []
    for f in files:
        columns = set(pq.read_schema(f).names)
        if not set(model.prob_columns) <= columns:
            continue
        if is_name_level(f) and model.name_column in columns:
            by_name.append(f)
        elif not is_name_level(f) and 'author_id' in columns:
            by_author.append(f)
    return by_name, by_author


def cached_probabilities(con: duckdb.DuckDBPyConnection, keys_dir: Path | str, shard: int, files: List[str],
                         model: NameModel) -> pd.DataFrame:
    """Probabilities of the shard's names found in the cache: by exact name in the name-level files, else through
    the keys (author_id -> name) in the author-level files; a name cached several times takes the name-level row
    first, then the row of the smallest author_id (deterministic)."""
    name, cols = model.name_column, list(model.prob_columns)
    empty = pd.DataFrame({name: pd.Series(dtype=object), **{c: pd.Series(dtype=float) for c in cols}})
    by_name, by_author = model_cache_files(files, model)
    if not by_name and not by_author:
        return empty
    complete = ' AND '.join(f"c.{col} IS NOT NULL" for col in cols)
    probs = ', '.join(f"c.{c}" for c in cols)
    parts = []
    if by_name:
        parts.append(f"""SELECT c.{name}, {probs}, 0 AS priority, 0 AS author_id
            FROM read_parquet({'[' + ', '.join(db.sql_path(f) for f in by_name) + ']'}, union_by_name = true) c
            WHERE c.{name} IN ({_shard_names_sql(keys_dir, shard, model)}) AND {complete}""")
    if by_author:
        parts.append(f"""SELECT k.{name}, {probs}, 1 AS priority, c.author_id
            FROM read_parquet({db.sql_path(Path(keys_dir) / 'bucket-*.parquet')}) k
            JOIN read_parquet({'[' + ', '.join(db.sql_path(f) for f in by_author) + ']'}, union_by_name = true) c
              USING (author_id)
            WHERE k.{model.shard_column} = {int(shard)} AND k.{name} IS NOT NULL AND {complete}""")
    return con.execute(f"""
        SELECT {name}, {', '.join(cols)} FROM (
            SELECT *, row_number() OVER (PARTITION BY {name} ORDER BY priority, author_id) AS rn
            FROM ({' UNION ALL '.join(parts)})
        ) WHERE rn = 1
    """).df()


def _parts_dir(output_file: Path, settings: Dict, overwrite: bool) -> Path:
    """Checkpoint folder <output>.parts/; parts written for another name list / chunk size are discarded."""
    parts_dir = output_file.with_name(output_file.stem + '.parts')
    meta = parts_dir / '_settings.json'
    if parts_dir.exists() and (overwrite or not meta.exists() or json.loads(meta.read_text()) != settings):
        ios.printf(f"Discarding stale checkpoints: {parts_dir}")
        shutil.rmtree(parts_dir)
    if not parts_dir.exists():
        tmp = parts_dir.with_name(parts_dir.name + '.tmp')
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        (tmp / '_settings.json').write_text(json.dumps(settings))
        tmp.rename(parts_dir)                      # the folder exists only with its settings file
    return parts_dir


def infer_shard(names: List[str], output_file: Path | str, probabilities_fn: ProbabilityFn, chunk_names: int,
                model: NameModel, overwrite: bool = False, log: Callable[[str], None] = ios.printf,
                cached: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Probabilities of every name: cached ones are taken as they are, the others are inferred in checkpointed parts
    of chunk_names names. Returns the merged frame (one row per name, in the order of ``names``)."""
    output_file = Path(output_file)
    name_column, prob_columns = model.name_column, list(model.prob_columns)
    cached = cached[cached[name_column].isin(set(names))] if cached is not None else None
    known = set(cached[name_column]) if cached is not None else set()
    todo = [n for n in names if n not in known]                        # keeps the (length, name) order
    digest = hashlib.md5('\n'.join(todo).encode('utf-8')).hexdigest()
    parts_dir = _parts_dir(output_file, {'n_names': len(todo), 'chunk_names': chunk_names, 'md5': digest}, overwrite)
    n_parts = max(1, math.ceil(len(todo) / chunk_names))
    for part in range(n_parts):
        part_file = parts_dir / f"part-{part:05d}.parquet"
        if part_file.exists():
            continue
        chunk = todo[part * chunk_names:(part + 1) * chunk_names]
        probs = probabilities_fn(chunk) if chunk else []
        if len(probs) != len(chunk):
            raise RuntimeError(f"Model returned {len(probs)} results for {len(chunk)} names")
        df = pd.DataFrame(probs, columns=prob_columns, dtype=float)
        df.insert(0, name_column, pd.Series(chunk, dtype=object))
        db.write_parquet_atomic(df, part_file)
        log(f"[{model.name} {output_file.stem}] part {part + 1}/{n_parts} stored ({len(known)} names cached)")
    inferred = pd.concat([ios.load_parquet(f) for f in sorted(parts_dir.glob('part-*.parquet'))], ignore_index=True)
    if inferred[name_column].tolist() != todo:
        raise RuntimeError(f"{parts_dir}: parts do not match the names to infer ({len(inferred)} vs {len(todo)})")
    frames = [inferred] if cached is None or cached.empty else [cached[[name_column, *prob_columns]], inferred]
    df = pd.concat(frames, ignore_index=True).set_index(name_column).loc[names].reset_index()
    if len(df) != len(names) or df[prob_columns].isna().any().any():
        raise RuntimeError(f"{output_file.name}: {len(df)} rows for {len(names)} names, or missing probabilities")
    db.write_parquet_atomic(df, output_file)
    shutil.rmtree(parts_dir)
    return df


# --- stage 5: merge per author bucket ----------------------------------------------------------------------------
def write_merged_bucket(con: duckdb.DuckDBPyConnection, keys_file: Path | str, gender_file: Path | str,
                        demographics_dir: Path | str, output_file: Path | str) -> int:
    """Keys of one bucket joined with the probabilities of their names (gender table + the shards of every model);
    checks that no author is lost or duplicated and that every name was found in its lookup table."""
    output_file = Path(output_file)
    tmp = output_file.with_name(output_file.name + '.tmp')
    aliases = {m.name: f"m{i}" for i, m in enumerate(MODELS.values())}
    probs = ', '.join(f"{aliases[m.name]}.{c}" for m in MODELS.values() for c in m.prob_columns)
    joins = '\n'.join(f"LEFT JOIN read_parquet({db.sql_path(model_dir(demographics_dir, m) / 'shard-*.parquet')}) "
                      f"{aliases[m.name]} ON {aliases[m.name]}.{m.name_column} = k.{m.name_column}"
                      for m in MODELS.values())
    joined = f"""
        FROM read_parquet({db.sql_path(keys_file)}) k
        LEFT JOIN read_parquet({db.sql_path(gender_file)}) g ON g.gender_name = k.gender_name
        {joins}
    """
    con.execute(f"COPY (SELECT k.author_id, k.gender_first_name, g.p_female, {probs} {joined} ORDER BY k.author_id) "
                f"TO {db.sql_path(tmp)} (FORMAT parquet, COMPRESSION zstd)")
    lost = ', '.join(f"count(*) FILTER (k.{m.name_column} IS NOT NULL AND {aliases[m.name]}.{m.name_column} IS NULL)"
                     for m in MODELS.values())
    n_keys, lost_gender, *lost_models = con.execute(
        f"SELECT count(*), count(*) FILTER (k.gender_name IS NOT NULL AND g.gender_name IS NULL), {lost} {joined}"
    ).fetchone()
    n_out, n_ids = con.execute(f"SELECT count(*), count(DISTINCT author_id) FROM read_parquet({db.sql_path(tmp)})").fetchone()
    n_authors = con.execute(f"SELECT count(*) FROM read_parquet({db.sql_path(keys_file)})").fetchone()[0]
    if not (n_out == n_ids == n_keys == n_authors and lost_gender == 0 and not any(lost_models)):
        tmp.unlink()
        missing = '; '.join(f"names missing in the {m} shards: {n}" for m, n in zip(MODELS, lost_models))
        raise RuntimeError(f"{output_file.name}: {n_authors} authors -> {n_out} rows, {n_ids} distinct ids; "
                           f"first names missing in the gender table: {lost_gender}; {missing}")
    tmp.replace(output_file)
    return n_out
