"""
Step A.2: name index over ALL OpenAlex authors of one bucket (display name + alternative names).

One row per (author, normalised name); names are cleaned (utils.text.clean_text) and normalised exactly like the
recommended names (responses.recommendations.normalize_tokens: no case / accents, transliterated):
    name_key        all name tokens, sorted: the same key as `name_key` of the recommendations (exact matching = join)
    name_key_short  first + last token, sorted (e.g. 'John A. Smith' -> 'john smith'); NULL for 1-token names
    name_norm       the tokens in written order ('john a smith'; for fuzzy scores, B.6a)
    first_norm, last_norm   first name / last name (last token plus lower-case particles, as the recommendations
                    are split: 'ludwig van beethoven' -> 'ludwig' / 'van beethoven'); NULL for 1-token names
    initials_key    first initial + last name ('j smith'); NULL for 1-token names
    author_id, is_display_name (the key comes from the display name),
    is_person (False = organisation-like display name without ORCID; kept in the index, flagged)

OpenAlex has no first / last name fields, only full-name strings (display name + alternative names), hence the split.
The matching rules (B.6a) and the decision (B.6b) are in libs/matching/.
"""

from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from groundtruth.db import is_person_sql
from responses.recommendations import normalize_tokens, split_full_name
from utils.text import clean_text

NAME_INDEX_SCHEMA = pa.schema([('name_key', pa.string()), ('name_key_short', pa.string()),
                               ('name_norm', pa.string()), ('first_norm', pa.string()), ('last_norm', pa.string()),
                               ('initials_key', pa.string()),
                               ('author_id', pa.int64()), ('is_display_name', pa.bool_()),
                               ('is_person', pa.bool_())])
_PARTS = ('name_key', 'name_key_short', 'name_norm', 'first_norm', 'last_norm', 'initials_key')


def name_keys(name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """(name_key, name_key_short) of a raw name."""
    return name_parts(name)[:2]


def name_parts(name: Optional[str]) -> Tuple[Optional[str], ...]:
    """(name_key, name_key_short, name_norm, first_norm, last_norm, initials_key) of a raw name (see module doc).
    The first / last split works on words, so a hyphenated surname stays whole ('José García-López' ->
    'jose' / 'garcia lopez'); 'Last, First' is reordered first."""
    tokens = normalize_tokens(name or '')
    if not tokens:
        return (None,) * len(_PARTS)
    key = ' '.join(sorted(tokens))
    text = clean_text(name)
    if ',' in text:
        last, _, first = text.partition(',')
        text = f"{first} {last}"
    words = [w for w in (' '.join(normalize_tokens(word)) for word in text.split()) if w]
    if len(tokens) == 1 or len(words) < 2:
        return key, None, ' '.join(tokens), None, None, None
    ordered = ' '.join(words).split()                                 # tokens in reading order ('Smith, John' reordered)
    first, last = split_full_name(words)
    return key, ' '.join(sorted([ordered[0], ordered[-1]])), ' '.join(words), first, last, f"{words[0][0]} {last}"


def names_query(bucket: int, n_buckets: int, limit_authors: Optional[int] = None) -> str:
    limit = f"LIMIT {int(limit_authors)}" if limit_authors else ''
    return f"""
        SELECT id AS author_id, display_name, {is_person_sql()} AS is_person,
               list_distinct(list_append(coalesce(display_name_alternatives, []), display_name)) AS names
        FROM authors WHERE id % {int(n_buckets)} = {int(bucket)}
        {limit}
    """


def iter_name_rows(con: duckdb.DuckDBPyConnection, bucket: int, n_buckets: int,
                   limit_authors: Optional[int] = None, batch_rows: int = 200_000) -> Iterator[pa.Table]:
    """Name-index tables, one per batch of authors (keys cached per batch)."""
    reader = con.execute(names_query(bucket, n_buckets, limit_authors)).to_arrow_reader(batch_rows)
    for batch in reader:
        cache: Dict[str, Tuple[Optional[str], ...]] = {}
        columns = {name: [] for name in NAME_INDEX_SCHEMA.names}
        for author_id, display_name, is_person, names in zip(*(batch.column(i).to_pylist() for i in range(4))):
            display_key = None
            if display_name is not None:
                display_key = cache.setdefault(display_name, name_parts(display_name))[0]
            seen = set()
            for name in names or []:
                if name is None:
                    continue
                parts = cache.setdefault(name, name_parts(name))
                key = parts[0]
                if key is None or key in seen:
                    continue
                seen.add(key)
                for column, value in zip(_PARTS, parts):
                    columns[column].append(value)
                columns['author_id'].append(author_id)
                columns['is_display_name'].append(key == display_key)
                columns['is_person'].append(bool(is_person))
        yield pa.table(columns, schema=NAME_INDEX_SCHEMA)


def write_name_index(con: duckdb.DuckDBPyConnection, output_file: Path | str, bucket: int, n_buckets: int,
                     limit_authors: Optional[int] = None) -> int:
    """Write the bucket's name index (to a temporary file first); returns the number of rows."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_file.with_suffix('.tmp')
    n_rows = 0
    with pq.ParquetWriter(tmp, NAME_INDEX_SCHEMA, compression='zstd') as writer:
        for table in iter_name_rows(con, bucket, n_buckets, limit_authors):
            writer.write_table(table)
            n_rows += table.num_rows
    tmp.replace(output_file)
    return n_rows
