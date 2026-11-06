"""
Step A.1: bibliometrics of the authors of one bucket, from their authorships (step A.0) and
the OpenAlex `authors` table.

One row per author with >= 1 counted work:
    author_id, display_name, orcid, is_person (False = organisation-like name without ORCID, see constants),
    first_pub_year, last_pub_year, n_works, n_citations, h_index, i10_index, e_index,
    n_coauthors (unique coauthors = rows in the A.1a edge list, i.e. on works with <= max_team authors; 0 if none),
    n_works_topic (counted works with a primary topic: the denominator of the field shares),
    works_<field> (works whose primary topic is in the mapped OpenAlex fields of each prompt field),
    oa_works_count, oa_cited_by_count, oa_h_index, oa_i10_index, two_year_mean_citedness (OpenAlex, all works),
    country (latest affiliation country; fallback: last known institution), last_known_country

h, i10 and e indices are computed on the counted works (e-index, Zhang 2009: e^2 = sum of the citations
of the h-core papers - h^2).
"""

from pathlib import Path
import time
from typing import Callable, List, Optional, Tuple

import duckdb
import pandas as pd

from groundtruth.db import authorships_glob, is_person_sql, sql_path
from groundtruth.fields import FieldMapping


def bibliometrics_steps(authorships_dir: Path | str, edges_dir: Path | str, bucket: int, n_buckets: int,
                        fields: List[FieldMapping]) -> List[Tuple[str, str]]:
    """(temp table, query) steps, run one after the other.

    Staged on purpose (one big query ran out of memory at full scale); each step is a temp table (spillable to
    --temp_dir). Coauthors come from this bucket's edge list (A.1a, coauthorship.py): no join against the full
    authorships table (DuckDB hashed its ~700M rows per job).
    """
    field_counts = ',\n'.join(f"count(*) FILTER (WHERE {f.sql_condition()}) AS {f.column}" for f in fields)
    bucket_files = sql_path(authorships_glob(authorships_dir, bucket))
    edges_file = sql_path(Path(edges_dir) / f"bucket-{int(bucket):02d}.parquet")
    in_bucket = f"id % {int(n_buckets)} = {int(bucket)}"
    return [
        ('aw', f"""   -- one row per (author, work)
            SELECT author_id, work_id, any_value(year) AS year, any_value(cited_by_count) AS c,
                   any_value(n_authors) AS n_authors, any_value(field_id) AS field_id,
                   any_value(subfield_id) AS subfield_id
            FROM read_parquet({bucket_files}, hive_partitioning = false)
            GROUP BY author_id, work_id"""),
        ('agg', f"""
            SELECT author_id, min(year) AS first_pub_year, max(year) AS last_pub_year,
                   count(*) AS n_works, count(*) FILTER (WHERE field_id IS NOT NULL) AS n_works_topic,
                   sum(c)::BIGINT AS n_citations,
                   count(*) FILTER (WHERE c >= rn) AS h_index,          -- c is non-increasing in rn
                   count(*) FILTER (WHERE c >= 10) AS i10_index,
                   coalesce(sum(c) FILTER (WHERE c >= rn), 0)::BIGINT AS h_core_citations,
                   {field_counts}
            FROM (SELECT *, row_number() OVER (PARTITION BY author_id ORDER BY c DESC, work_id) AS rn FROM aw)
            GROUP BY author_id"""),
        ('co', f"""   -- unique coauthors = rows of the (directed, deduplicated) edge list
            SELECT author_id, count(*) AS n_coauthors FROM read_parquet({edges_file}) GROUP BY author_id"""),
        ('info', f"""
            SELECT id AS author_id, display_name, orcid, {is_person_sql()} AS is_person,
                   works_count AS oa_works_count, cited_by_count AS oa_cited_by_count,
                   summary_stats.h_index AS oa_h_index, summary_stats.i10_index AS oa_i10_index,
                   summary_stats."2yr_mean_citedness" AS two_year_mean_citedness,
                   last_known_institutions[1].country_code AS last_known_country
            FROM authors WHERE {in_bucket}"""),
        ('aff', f"""
            SELECT author_id, arg_max(country_code, last_year) AS latest_affiliation_country
            FROM (SELECT id AS author_id, af.institution.country_code AS country_code, list_max(af.years) AS last_year
                  FROM (SELECT id, unnest(affiliations) AS af FROM authors WHERE {in_bucket}))
            WHERE country_code IS NOT NULL AND last_year IS NOT NULL
            GROUP BY author_id"""),
        ('result', f"""
            SELECT agg.author_id, info.display_name, info.orcid, info.is_person,
                   agg.first_pub_year, agg.last_pub_year, agg.n_works, agg.n_citations,
                   agg.h_index, agg.i10_index,
                   sqrt(greatest(agg.h_core_citations - agg.h_index * agg.h_index, 0)) AS e_index,
                   coalesce(co.n_coauthors, 0) AS n_coauthors, agg.n_works_topic,
                   {', '.join(f'agg.{f.column}' for f in fields)},
                   info.oa_works_count, info.oa_cited_by_count, info.oa_h_index, info.oa_i10_index,
                   info.two_year_mean_citedness,
                   coalesce(aff.latest_affiliation_country, info.last_known_country) AS country,
                   info.last_known_country
            FROM agg
            LEFT JOIN co USING (author_id)
            LEFT JOIN info USING (author_id)
            LEFT JOIN aff USING (author_id)
            ORDER BY agg.author_id"""),
    ]


# Temp tables no longer needed once a step is done (freed early to keep memory low)
_DROP_AFTER = {'agg': ['aw'], 'result': ['agg', 'co', 'info', 'aff']}


def compute_bibliometrics(con: duckdb.DuckDBPyConnection, authorships_dir: Path | str, edges_dir: Path | str,
                          bucket: int, n_buckets: int, fields: List[FieldMapping],
                          log: Optional[Callable[[str], None]] = None) -> pd.DataFrame:
    """Run the steps as temp tables and return the result (one row per author)."""
    for name, query in bibliometrics_steps(authorships_dir, edges_dir, bucket, n_buckets, fields):
        start = time.time()
        con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS {query}")
        if log:
            n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            log(f"[bucket {bucket}] step {name}: {n} rows in {time.time() - start:.0f} s")
        for table in _DROP_AFTER.get(name, []):
            con.execute(f"DROP TABLE IF EXISTS {table}")
    df = con.execute("SELECT * FROM result").df()
    con.execute("DROP TABLE result")
    return df
