"""
Step B.6b: author matching of every recommendation from the B.6a candidates, with thresholds chosen here (cheap; re-run
whenever a threshold changes). Everything is stored so that the rules can be compared and plotted later.

1. Name match. A candidate matches by a rule if (exact_display, exact_alternative, short, initials) holds, or
   (fuzzy) jw_full >= jw_full_min and jw_first >= jw_first_min and jw_last >= jw_last_min (None: not used).
   Defaults (constants, provisional): both parts >= 0.90, jw_full at the B.6a floor.
   A recommendation uses only its STRICTEST rule with any match (exact_display > exact_alternative > short >
   initials > fuzzy): `match_rule`; its candidates are the people matching by that rule. The number of
   candidates of EVERY rule is stored as well (n_rule_<rule>, overlapping), and the best fuzzy scores
   (jw_full_max; jw_parts_max = the highest min(jw_first, jw_last)).
2. Levels, within the candidates of match_rule (field and seniority are those requested by the prompt):
       n_name             all candidates
       n_name_field       with >= 1 work in the field (primary topic, A.1) and field share >= min_field_share
       n_name_seniority   with the requested seniority: academic age = reference_year - first publication year,
                          'three': junior <= junior_max, senior >= senior_min (mid matches neither);
                          'single': junior <= single_cutoff < senior
       n_author           both
   matched_<level> = n_<level> > 0. A candidate without counted works (not in A.1) is a name match only.
3. The OpenAlex person (one pick, seeded; the same pick whatever the order or subset of the rows):
       matched_author              -> author_id: random among the n_author people (used downstream)
       else, if exactly one of the field / seniority sets is non-empty:
       matched_author_field        -> author_id_field: random among the n_name_field people
       matched_author_seniority    -> author_id_seniority: random among the n_name_seniority people
       (both non-empty: ambiguous, no pick; both empty: no pick)
   The random draw takes, among the eligible people, the one with the smallest md5(seed | recommendation | author).
"""

from pathlib import Path
from typing import Sequence

import duckdb

from groundtruth.db import sql_path
from groundtruth.fields import FieldMapping
from utils import constants as cons

REC_ID_COLUMNS = cons.RESPONSE_ID_COLUMNS + [cons.COL_REC_INDEX]
_RULE_SQL = {
    cons.MATCH_EXACT_DISPLAY: 'c.exact_display',
    cons.MATCH_EXACT_ALTERNATIVE: 'c.exact_alternative',
    cons.MATCH_SHORT: 'c.short',
    cons.MATCH_INITIALS: 'c.initials',
}


def fuzzy_sql(jw_full_min: float, jw_first_min: float | None, jw_last_min: float | None) -> str:
    parts = [f"coalesce(c.jw_full >= {float(jw_full_min)}, false)"]
    if jw_first_min is not None:
        parts.append(f"coalesce(c.jw_first >= {float(jw_first_min)}, false)")
    if jw_last_min is not None:
        parts.append(f"coalesce(c.jw_last >= {float(jw_last_min)}, false)")
    return '(' + ' AND '.join(parts) + ')'


def match_query(recommendations_file: Path | str, rec_names_file: Path | str, candidates_glob: str,
                     authors_glob: str, fields: Sequence[FieldMapping], seed: int, jw_full_min: float,
                     jw_first_min: float | None = None, jw_last_min: float | None = None,
                     min_field_share: float = cons.DEFAULT_GT_MIN_FIELD_SHARE,
                     seniority_definition: str = cons.MATCH_SENIORITY_THREE, reference_year: int = cons.GT_REFERENCE_YEAR,
                     junior_max: int = cons.DEFAULT_SENIORITY_JUNIOR_MAX,
                     senior_min: int = cons.DEFAULT_SENIORITY_SENIOR_MIN,
                     single_cutoff: int = cons.DEFAULT_SENIORITY_SINGLE_CUTOFF) -> str:
    rules = {**_RULE_SQL, cons.MATCH_FUZZY: fuzzy_sql(jw_full_min, jw_first_min, jw_last_min)}
    rank = 'CASE ' + ' '.join(f"WHEN {cond} THEN {i}" for i, cond in enumerate(rules.values())) + ' END'
    rule_name = 'CASE match_rank ' + ' '.join(f"WHEN {i} THEN '{name}'" for i, name in enumerate(rules)) + ' END'
    rule_counts = ', '.join(f"count(*) FILTER ({cond}) AS n_rule_{name}" for name, cond in rules.items())
    works = 'CASE r.field ' + ' '.join(f"WHEN '{f.field}' THEN a.{f.column}" for f in fields) + ' END'
    age = f"({int(reference_year)} - a.first_pub_year)"
    if seniority_definition == cons.MATCH_SENIORITY_THREE:
        senior_ok = (f"((r.seniority = '{cons.SENIORITY_JUNIOR}' AND {age} <= {int(junior_max)}) OR "
                     f"(r.seniority = '{cons.SENIORITY_SENIOR}' AND {age} >= {int(senior_min)}))")
    elif seniority_definition == cons.MATCH_SENIORITY_SINGLE:
        senior_ok = (f"((r.seniority = '{cons.SENIORITY_JUNIOR}' AND {age} <= {int(single_cutoff)}) OR "
                     f"(r.seniority = '{cons.SENIORITY_SENIOR}' AND {age} > {int(single_cutoff)}))")
    else:
        raise ValueError(f"Unknown seniority definition: {seniority_definition}")
    ids = ', '.join(f"r.{c}" for c in REC_ID_COLUMNS)
    uid = "concat_ws('|', " + ', '.join(f"m.{c}::VARCHAR" for c in REC_ID_COLUMNS) + ")"   # one recommendation
    draw = f"md5('{int(seed)}|' || r.rec_uid || '|' || c.author_id::VARCHAR)"
    a_cols = ', '.join(['author_id', 'first_pub_year', 'n_works_topic'] + [f.column for f in fields])
    return f"""
        WITH r AS (
            SELECT {', '.join(f"m.{c}" for c in REC_ID_COLUMNS)}, m.first_name, m.last_name, n.rec_name,
                   {uid} AS rec_uid
            FROM read_parquet({sql_path(recommendations_file)}) m
            LEFT JOIN read_parquet({sql_path(rec_names_file)}) n
                   ON n.first_name = m.first_name AND n.last_name = m.last_name
        ),
        c AS (                                             -- candidates of this model's names, with their rule rank
            SELECT c.*, {rank} AS match_rank
            FROM read_parquet({sql_path(candidates_glob)}) c
            WHERE c.rec_name IN (SELECT rec_name FROM r WHERE rec_name IS NOT NULL)
        ),
        per_name AS (                                      -- name-level statistics (independent of the prompt)
            SELECT c.rec_name, min(match_rank) AS match_rank, {rule_counts}, max(c.jw_full) AS jw_full_max,
                   max(least(c.jw_first, c.jw_last)) AS jw_parts_max
            FROM c GROUP BY c.rec_name
        ),
        a AS (SELECT {a_cols} FROM read_parquet({sql_path(authors_glob)})
              WHERE author_id IN (SELECT author_id FROM c)),
        tier AS (                                          -- candidates of the strictest rule, judged per recommendation
            SELECT r.rec_uid, c.author_id,
                   coalesce({works} >= 1 AND {works} / nullif(a.n_works_topic, 0) >= {float(min_field_share)}, false)
                       AS in_field,
                   coalesce({senior_ok}, false) AS in_seniority,
                   {draw} AS draw
            FROM r JOIN per_name p USING (rec_name)
            JOIN c ON c.rec_name = r.rec_name AND c.match_rank = p.match_rank
            LEFT JOIN a ON a.author_id = c.author_id
        ),
        levels AS (
            SELECT rec_uid, count(*) AS n_name, count(*) FILTER (in_field) AS n_name_field,
                   count(*) FILTER (in_seniority) AS n_name_seniority, count(*) FILTER (in_field AND in_seniority) AS n_author,
                   arg_min(author_id, draw) FILTER (in_field AND in_seniority) AS pick_author,
                   arg_min(author_id, draw) FILTER (in_field) AS pick_field,
                   arg_min(author_id, draw) FILTER (in_seniority) AS pick_seniority
            FROM tier GROUP BY rec_uid
        )
        SELECT {ids}, r.first_name, r.last_name, r.rec_name,
               {rule_name.replace('match_rank', 'p.match_rank')} AS match_rule,
               {', '.join(f"coalesce(p.n_rule_{name}, 0) AS n_rule_{name}" for name in rules)},
               p.jw_full_max, p.jw_parts_max,
               coalesce(l.n_name, 0) AS n_name, coalesce(l.n_name_field, 0) AS n_name_field,
               coalesce(l.n_name_seniority, 0) AS n_name_seniority, coalesce(l.n_author, 0) AS n_author,
               coalesce(l.n_name, 0) > 0 AS matched_name, coalesce(l.n_name_field, 0) > 0 AS matched_name_field,
               coalesce(l.n_name_seniority, 0) > 0 AS matched_name_seniority, coalesce(l.n_author, 0) > 0 AS matched_author,
               l.pick_author AS author_id,
               coalesce(l.n_author = 0 AND l.n_name_field > 0 AND l.n_name_seniority = 0, false) AS matched_author_field,
               CASE WHEN l.n_author = 0 AND l.n_name_field > 0 AND l.n_name_seniority = 0 THEN l.pick_field END
                   AS author_id_field,
               coalesce(l.n_author = 0 AND l.n_name_seniority > 0 AND l.n_name_field = 0, false) AS matched_author_seniority,
               CASE WHEN l.n_author = 0 AND l.n_name_seniority > 0 AND l.n_name_field = 0 THEN l.pick_seniority END
                   AS author_id_seniority
        FROM r
        LEFT JOIN per_name p ON p.rec_name = r.rec_name
        LEFT JOIN levels l ON l.rec_uid = r.rec_uid
        ORDER BY {', '.join(f"r.{c}" for c in REC_ID_COLUMNS)}
    """


def write_matches(con: duckdb.DuckDBPyConnection, output_file: Path | str, **kwargs) -> int:
    """Write the author matching of one model's recommendations (via a temporary file); returns the number of rows."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_file.with_name(output_file.name + '.tmp')
    row = con.execute(f"COPY ({match_query(**kwargs)}) TO {sql_path(tmp)} (FORMAT parquet, COMPRESSION zstd)").fetchone()
    tmp.replace(output_file)
    return int(row[0]) if row else 0

