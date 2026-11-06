"""
Step A.4: target population, seniority, prominence classes and GT shares (from the per-bucket outputs
of A.1 bibliometrics and A.3 demographics). Everything runs in DuckDB (no full table in pandas).

Outputs (in <gt_dir>/):
    gt_authors.parquet   one row per scholar of the target population (people only, i.e. is_person; >= 1 work in
                         at least one prompt field, and share >= min_field_share, default 0): bibliometrics,
                         share_<field> (share of the works with a primary topic), academic_age, seniority (junior /
                         mid / senior), seniority_single (junior / senior at one cut-off), fields (list),
                         p_female / p_<ethnicity> and the labels gender / ethnicity (thresholds applied here),
                         citations_per_paper_per_year
    gt_members.parquet   one row per (scholar, prompt field): field_share, seniority, prominence bins productivity /
                         influence / collaboration (L / M / H / unknown, rank-based within field x seniority),
                         is_top_productivity / is_top_influence (top x % by works / citations within field x
                         seniority), gender, ethnicity, country
    gt_shares.parquet    GT shares per scope: field x seniority, field, seniority, all
                         (columns scope, field, seniority, attribute, class, n, share). The seniority / all
                         scopes pool the (scholar, field) rows: a scholar in two fields counts twice.
    gt_cutoffs.parquet   per field x seniority x metric x bin: n, min and max metric value

Bins use the percent rank (ties share a bin, so bins are not exactly equal-sized when values tie, e.g. many
authors with one work). Authors without a value get 'unknown'.
"""

from pathlib import Path
from typing import List, Sequence

import duckdb

from groundtruth.db import sql_path
from groundtruth.fields import FieldMapping
from utils import constants as cons

TOP_COLUMNS = [f"top_{name}" for name in cons.GT_TOP_METRICS]
SHARE_ATTRIBUTES = ['gender', 'ethnicity', 'country', *cons.GT_BIN_METRICS, *TOP_COLUMNS]
SCOPE_FIELD_SENIORITY, SCOPE_FIELD, SCOPE_SENIORITY, SCOPE_ALL = 'field_x_seniority', 'field', 'seniority', 'all'
UNKNOWN = 'unknown'


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def share_column(field: FieldMapping) -> str:
    """Column with the share of an author's works (with a primary topic) in a field, e.g. share_computer_science."""
    return field.column.replace('works_', 'share_', 1)


def gender_label_sql(edges: Sequence[float] = cons.DEFAULT_GENDER_EDGES,
                     labels: Sequence[str] = cons.DEFAULT_GENDER_LABELS) -> str:
    """Perceived gender from p_female: label i for [e_i, e_{i+1}) (cut points edges, len(labels) = len(edges) + 1);
    'unknown' without p_female. Any number of classes (see cons.DEFAULT_GENDER_EDGES)."""
    check_scheme(edges, labels, 'Gender')
    whens = ' '.join(f"WHEN p_female < {float(edge)} THEN {_sql_str(label)}" for edge, label in zip(edges, labels))
    return (f"CASE WHEN p_female IS NULL OR isnan(p_female) THEN '{cons.GENDER_UNKNOWN}' {whens} "
            f"ELSE {_sql_str(labels[-1])} END")


def check_scheme(edges: Sequence[float], labels: Sequence[str], what: str) -> None:
    """Cut points increasing in (0, 1) and one distinct label per interval (not 'unknown')."""
    edges = list(edges)
    if not edges or any(not 0 < e < 1 for e in edges) or edges != sorted(set(edges)):
        raise ValueError(f"{what} cut points must be increasing and in (0, 1): {edges}")
    if len(labels) != len(edges) + 1 or len(set(labels)) != len(labels) or UNKNOWN in labels:
        raise ValueError(f"Need {len(edges) + 1} distinct {what.lower()} labels (not '{UNKNOWN}') for {edges}: {list(labels)}")


def ethnicity_label_sql(min_prob: float) -> str:
    """raceBERT: argmax class if its probability >= min_prob, else unknown."""
    classes = list(dict.fromkeys(cons.RACEBERT_LABELS.values()))
    best = f"greatest({', '.join(f'p_{c}' for c in classes)})"
    whens = ' '.join(f"WHEN p_{c} = {best} THEN '{c}'" for c in classes)
    return f"CASE WHEN {best} IS NULL OR {best} < {float(min_prob)} THEN '{cons.ETHNICITY_UNKNOWN}' {whens} END"


def create_views(con: duckdb.DuckDBPyConnection, authors_glob: str, demographics_glob: str | None,
                 fields: List[FieldMapping], min_field_share: float, reference_year: int, junior_max: int,
                 senior_min: int, single_cutoff: int, bin_edges: Sequence[float], top_percentile: float,
                 bin_labels: Sequence[str] = cons.DEFAULT_GT_BIN_LABELS,
                 gender_edges: Sequence[float] = cons.DEFAULT_GENDER_EDGES,
                 gender_labels: Sequence[str] = cons.DEFAULT_GENDER_LABELS,
                 ethnicity_min_prob: float = cons.DEFAULT_ETHNICITY_MIN_PROB) -> None:
    """Create the views gt_authors and gt_members on the per-bucket files.

    A scholar belongs to a field if they have >= 1 work there and its share of their works (with a primary topic)
    is >= min_field_share (default 0: any work). Gender / ethnicity labels are derived here from the stored
    probabilities (A.3), so thresholds can change without re-running the models.
    """
    check_scheme(bin_edges, bin_labels, 'Prominence bin')
    share_cols = ',\n'.join(f"a.{f.column} / nullif(a.n_works_topic, 0) AS {share_column(f)}" for f in fields)
    member = {f.field: f"(a.{f.column} >= 1 AND a.{f.column} / nullif(a.n_works_topic, 0) >= {float(min_field_share)})"
              for f in fields}
    in_any = ' OR '.join(member.values())
    field_list = ', '.join(f"CASE WHEN {cond} THEN {_sql_str(name)} END" for name, cond in member.items())
    prob_cols = ['p_female', *[f"p_{c}" for c in dict.fromkeys(cons.RACEBERT_LABELS.values())]]
    if demographics_glob:
        demo_join = f"LEFT JOIN read_parquet({sql_path(demographics_glob)}) d USING (author_id)"
        demo_cols = "d.gender_first_name, " + ', '.join(f"d.{c}" for c in prob_cols)
    else:
        demo_join = ''
        demo_cols = "NULL::VARCHAR AS gender_first_name, " + ', '.join(f"NULL::DOUBLE AS {c}" for c in prob_cols)

    con.execute(f"""
        CREATE OR REPLACE VIEW gt_authors AS
        WITH base AS (
            SELECT a.*,
                   {share_cols},
                   {int(reference_year)} - a.first_pub_year AS academic_age,
                   CASE WHEN {int(reference_year)} - a.first_pub_year <= {int(junior_max)} THEN '{cons.SENIORITY_JUNIOR}'
                        WHEN {int(reference_year)} - a.first_pub_year >= {int(senior_min)} THEN '{cons.SENIORITY_SENIOR}'
                        ELSE '{cons.SENIORITY_MID}' END AS seniority,
                   CASE WHEN {int(reference_year)} - a.first_pub_year <= {int(single_cutoff)}
                        THEN '{cons.SENIORITY_JUNIOR}' ELSE '{cons.SENIORITY_SENIOR}' END AS seniority_single,
                   list_filter([{field_list}], x -> x IS NOT NULL) AS fields,
                   a.n_citations / a.n_works / greatest({int(reference_year)} - a.first_pub_year, 1)
                       AS citations_per_paper_per_year,
                   {demo_cols}
            FROM read_parquet({sql_path(authors_glob)}) a
            {demo_join}
            WHERE a.is_person AND ({in_any})
        )
        SELECT *, {gender_label_sql(gender_edges, gender_labels)} AS gender,
                  {ethnicity_label_sql(ethnicity_min_prob)} AS ethnicity
        FROM base
    """)

    rank_cols = ',\n'.join(
        f"CASE WHEN {metric} IS NULL THEN NULL ELSE percent_rank() OVER "
        f"(PARTITION BY field, seniority, {metric} IS NULL ORDER BY {metric}) END AS pr_{name}"
        for name, metric in cons.GT_BIN_METRICS.items())
    bin_cols = ',\n'.join(
        f"CASE WHEN pr_{name} IS NULL THEN '{UNKNOWN}' "
        + ' '.join(f"WHEN pr_{name} < {float(edge)} THEN {_sql_str(label)}" for edge, label in zip(bin_edges, bin_labels))
        + f" ELSE {_sql_str(bin_labels[-1])} END AS {name}"
        for name in cons.GT_BIN_METRICS)
    metrics = ', '.join(dict.fromkeys([*cons.GT_BIN_METRICS.values(), *cons.GT_TOP_METRICS.values()]))
    top_ranks = ', '.join(f"percent_rank() OVER (PARTITION BY field, seniority ORDER BY {metric}) AS pr_top_{name}"
                          for name, metric in cons.GT_TOP_METRICS.items())
    top_flags = ', '.join(f"pr_top_{name} >= {1 - float(top_percentile) / 100} AS is_top_{name}"
                          for name in cons.GT_TOP_METRICS)
    field_share_case = ' '.join(f"WHEN {_sql_str(f.field)} THEN {share_column(f)}" for f in fields)
    con.execute(f"""
        CREATE OR REPLACE VIEW gt_members AS
        WITH m AS (
            SELECT author_id, unnest(fields) AS field, seniority, seniority_single, {metrics},
                   {', '.join(share_column(f) for f in fields)},
                   gender, ethnicity, coalesce(country, '{UNKNOWN}') AS country
            FROM gt_authors
        ), r AS (
            SELECT *, CASE field {field_share_case} END AS field_share FROM m
        ), r2 AS (
            SELECT *, {rank_cols}, {top_ranks}
            FROM r
        )
        SELECT author_id, field, field_share, seniority, seniority_single, {metrics},
               {bin_cols},
               {top_flags},
               gender, ethnicity, country
        FROM r2
    """)


def shares_query() -> str:
    """GT shares of every attribute class, per scope (see module docstring)."""
    attributes = ', '.join(SHARE_ATTRIBUTES)
    return f"""
        WITH long AS (
            UNPIVOT (SELECT field, seniority, gender, ethnicity, country,
                            {', '.join(cons.GT_BIN_METRICS)},
                            {', '.join(f"CASE WHEN is_top_{n} THEN 'top' ELSE 'rest' END AS top_{n}" for n in cons.GT_TOP_METRICS)}
                     FROM gt_members)
            ON {attributes} INTO NAME attribute VALUE class
        ), counts AS (
            SELECT CASE WHEN GROUPING(field) = 0 AND GROUPING(seniority) = 0 THEN '{SCOPE_FIELD_SENIORITY}'
                        WHEN GROUPING(field) = 0 THEN '{SCOPE_FIELD}'
                        WHEN GROUPING(seniority) = 0 THEN '{SCOPE_SENIORITY}'
                        ELSE '{SCOPE_ALL}' END AS scope,
                   field, seniority, attribute, class, count(*) AS n
            FROM long
            GROUP BY GROUPING SETS ((field, seniority, attribute, class), (field, attribute, class),
                                    (seniority, attribute, class), (attribute, class))
        )
        SELECT scope, field, seniority, attribute, class, n,
               n / sum(n) OVER (PARTITION BY scope, field, seniority, attribute) AS share
        FROM counts
        ORDER BY scope, field, seniority, attribute, n DESC
    """


def cutoffs_query() -> str:
    parts = [f"""SELECT field, seniority, '{name}' AS attribute, '{metric}' AS metric, {name} AS class,
                        count(*) AS n, min({metric}) AS min_value, max({metric}) AS max_value
                 FROM gt_members GROUP BY field, seniority, {name}"""
             for name, metric in cons.GT_BIN_METRICS.items()]
    return ' UNION ALL '.join(parts) + ' ORDER BY field, seniority, attribute, class'


def write_outputs(con: duckdb.DuckDBPyConnection, gt_dir: Path | str) -> dict:
    """Write gt_authors / gt_members / gt_shares / gt_cutoffs; returns their row counts."""
    gt_dir = Path(gt_dir)
    gt_dir.mkdir(parents=True, exist_ok=True)
    queries = {'gt_authors': 'SELECT * FROM gt_authors ORDER BY author_id',
               'gt_members': 'SELECT * FROM gt_members ORDER BY field, seniority, author_id',
               'gt_shares': shares_query(),
               'gt_cutoffs': cutoffs_query()}
    counts = {}
    for name, query in queries.items():
        target = gt_dir / f"{name}.parquet"
        tmp = gt_dir / f"{name}.parquet.tmp"
        con.execute(f"COPY ({query}) TO {sql_path(tmp)} (FORMAT parquet, COMPRESSION zstd)")
        tmp.replace(target)
        counts[name] = con.execute(f"SELECT count(*) FROM read_parquet({sql_path(target)})").fetchone()[0]
        if name == 'gt_members':   # shares and cut-offs read the written file (window functions run once)
            con.execute(f"CREATE OR REPLACE VIEW gt_members AS SELECT * FROM read_parquet({sql_path(target)})")
    return counts
