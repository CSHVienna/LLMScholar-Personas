"""
GT descriptives (notebook 0_gt.ipynb): small tables from the A.1 and A.4 outputs. The big tables (gt_authors: one row
per scholar; gt_members: one row per scholar x field; tens of millions of rows) are aggregated in DuckDB with bounded
resources and never loaded into pandas; gt_shares / gt_cutoffs are small and read directly.

    cached(cache_dir, name, compute, sources)
                                      a table computed once and re-read while it is newer than its source files
    connect()                         in-memory DuckDB with the GT_STATS_* threads / memory and the project spill folder
    funnel(con, gt_dir)               OpenAlex authors with >= 1 counted work (A.1) -> people -> GT scholars -> members
    population(con, gt_dir, by)       scholars per field x seniority / gender / ethnicity (gt_members)
    fields_per_scholar(con, gt_dir)   how many prompt fields a scholar belongs to
    field_overlap(con, gt_dir)        scholars in field A that are also in field B (counts; diagonal = field size)
    bibliometric_quantiles(...)       n, mean and quantiles of each bibliometric per field x seniority (+ all fields),
                                      optionally per perceived gender / ethnicity (by=)
    histogram(con, gt_dir, column)    counts in equal-width bins of a gt_authors column (optionally per field)
    gender_coverage(con, gt_dir)      perceived gender label by why it is unknown (no first name / no p_female)
    composition(shares, attribute)    class shares per field x seniority from gt_shares (countries pooled to the top)
    shares_by_prominence(...)         perceived gender / ethnicity inside each L / M / H class and the top x % group
    cutoffs_table(cutoffs)            L / M / H value ranges per field x seniority (gt_cutoffs)

Pooled rows ('All fields') count a scholar once per field (as gt_shares). Labels (gender, ethnicity, bins) are those
written by A.4, i.e. with the config of the A.4 run.
"""

from pathlib import Path
from typing import Callable, List, Optional, Sequence

import duckdb
import numpy as np
import pandas as pd

from groundtruth.db import sql_path
from groundtruth.population import SCOPE_FIELD_SENIORITY, SCOPE_SENIORITY, UNKNOWN
from utils import constants as cons

ALL_FIELDS = 'All fields'
OTHER = 'other'
SENIORITY_ORDER = list(cons.SENIORITY_CLASSES)
GT_FILES = ['gt_authors', 'gt_members', 'gt_shares', 'gt_cutoffs']
BY_ATTRIBUTES = ('gender', 'ethnicity')
POPULATION_BY = ('seniority', *BY_ATTRIBUTES)
PROMINENCE_GROUPS = [*cons.GT_BIN_METRICS, *(f'top_{n}' for n in cons.GT_TOP_METRICS)]


# --- plumbing ----------------------------------------------------------------------------------------------------
def connect(threads: int = cons.GT_STATS_DUCKDB_THREADS, memory_limit: str = cons.GT_STATS_DUCKDB_MEMORY,
            temp_dir: str = cons.DEFAULT_DUCKDB_TEMP_DIR) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(':memory:')
    con.execute(f"SET threads = {int(threads)}")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = {sql_path(temp_dir)}")
    con.execute("SET preserve_insertion_order = false")
    return con


def cached(cache_dir: Optional[Path | str], name: str, compute: Callable[[], pd.DataFrame],
           sources: Sequence[Path | str] = ()) -> pd.DataFrame:
    """compute() once, saved as <cache_dir>/<name>.parquet; re-read while the file is newer than every source (e.g.
    the GT parquet files: a re-run of A.4 invalidates it). cache_dir None: always compute."""
    if cache_dir is None:
        return compute()
    path = Path(cache_dir) / f"{name}.parquet"
    newest = max((Path(s).stat().st_mtime for s in sources if Path(s).exists()), default=0)
    if path.exists() and path.stat().st_mtime > newest:
        return pd.read_parquet(path)
    df = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.parquet.tmp')
    df.to_parquet(tmp)
    tmp.replace(path)
    return df


def gt_file(gt_dir: Path | str, name: str) -> Path:
    path = Path(gt_dir) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No {path}: run run_gt_pipeline.sh (step population, A.4)")
    return path


def _src(gt_dir: Path | str, name: str) -> str:
    return f"read_parquet({sql_path(gt_file(gt_dir, name))})"


def field_order(fields: Sequence[str]) -> List[str]:
    known = [str(f) for f in cons.VARIABLE_VALUES[cons.VAR_FIELD]]
    present = set(map(str, fields))
    return [f for f in known if f in present] + sorted(present - set(known) - {ALL_FIELDS}) + \
        ([ALL_FIELDS] if ALL_FIELDS in present else [])


def _sort_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Rows in the design order: fields (All fields last), then junior / mid / senior."""
    fields, seniorities = field_order(df['field'].unique()), SENIORITY_ORDER
    key = pd.DataFrame({'f': df['field'].map({f: i for i, f in enumerate(fields)}),
                        's': df['seniority'].map({s: i for i, s in enumerate(seniorities)}).fillna(len(seniorities))})
    return df.loc[key.sort_values(['f', 's'], kind='stable').index].reset_index(drop=True)


# --- population --------------------------------------------------------------------------------------------------
def funnel(con: duckdb.DuckDBPyConnection, gt_dir: Path | str, authors_dir: Path | str = None) -> pd.DataFrame:
    """Steps from the A.1 authors to the GT memberships: n and share of the previous step."""
    authors_dir = Path(authors_dir or Path(gt_dir) / 'authors')
    rows = []
    files = sorted(authors_dir.glob('bucket-*.parquet'))
    if files:
        glob = sql_path(authors_dir / 'bucket-*.parquet')
        n_all, n_person = con.execute(f"SELECT count(*), count(*) FILTER (WHERE is_person) FROM read_parquet({glob})").fetchone()
        rows += [('OpenAlex authors with >= 1 counted work (A.1)', n_all), ('people (organisations excluded)', n_person)]
    n_gt = con.execute(f"SELECT count(*) FROM {_src(gt_dir, 'gt_authors')}").fetchone()[0]
    n_members = con.execute(f"SELECT count(*) FROM {_src(gt_dir, 'gt_members')}").fetchone()[0]
    rows += [('GT scholars (>= 1 work in a prompt field)', n_gt), ('scholar x field memberships', n_members)]
    out = pd.DataFrame(rows, columns=['step', 'n'])
    out['share_of_previous'] = out['n'] / out['n'].shift(1)
    return out


def population(con: duckdb.DuckDBPyConnection, gt_dir: Path | str, by: str = 'seniority') -> pd.DataFrame:
    """Scholars per field x `by` (+ All fields = memberships pooled): field, <by>, n, share_of_field. by: seniority,
    gender or ethnicity (the gt_members columns; 'unknown' included as a class)."""
    if by not in POPULATION_BY:
        raise ValueError(f"by must be one of {POPULATION_BY}: {by}")
    df = con.execute(f"""
        SELECT coalesce(field, '{ALL_FIELDS}') AS field, {by}, count(*) AS n
        FROM {_src(gt_dir, 'gt_members')}
        GROUP BY GROUPING SETS ((field, {by}), ({by}))""").df()
    df['share_of_field'] = df['n'] / df.groupby('field')['n'].transform('sum')
    if by == 'seniority':
        return _sort_cells(df)
    df = df.sort_values(['n'], ascending=False, kind='stable')
    order = {f: i for i, f in enumerate(field_order(df['field'].unique()))}
    return df.sort_values('field', key=lambda c: c.map(order), kind='stable').reset_index(drop=True)


def fields_per_scholar(con: duckdb.DuckDBPyConnection, gt_dir: Path | str) -> pd.DataFrame:
    df = con.execute(f"""SELECT len(fields) AS n_fields, count(*) AS n FROM {_src(gt_dir, 'gt_authors')}
                         GROUP BY 1 ORDER BY 1""").df()
    df['share'] = df['n'] / df['n'].sum()
    return df


def field_overlap(con: duckdb.DuckDBPyConnection, gt_dir: Path | str) -> pd.DataFrame:
    """Square table (rows, columns = fields): scholars in the row field that also belong to the column field."""
    df = con.execute(f"""
        WITH f AS (SELECT unnest(fields) AS field_a, fields FROM {_src(gt_dir, 'gt_authors')}),
             p AS (SELECT field_a, unnest(fields) AS field_b FROM f)
        SELECT field_a, field_b, count(*) AS n FROM p GROUP BY ALL""").df()
    order = field_order(pd.concat([df['field_a'], df['field_b']]).unique())
    table = df.pivot(index='field_a', columns='field_b', values='n').reindex(index=order, columns=order).fillna(0)
    table.index.name, table.columns.name = 'field', 'also in'
    return table.astype('int64')


# --- bibliometrics -----------------------------------------------------------------------------------------------
def bibliometric_quantiles(con: duckdb.DuckDBPyConnection, gt_dir: Path | str,
                           metrics: Sequence[str] = cons.GT_STATS_METRICS,
                           quantiles: Sequence[float] = cons.GT_STATS_QUANTILES,
                           by: Optional[str] = None) -> pd.DataFrame:
    """Long table field, seniority, [by,] metric, n, mean, q<..> (approximate quantiles, t-digest; one pass over
    gt_authors with the fields unnested; All fields = memberships pooled). by: 'gender' or 'ethnicity' (perceived
    labels of A.4, 'unknown' included as a class)."""
    if by not in (None, *BY_ATTRIBUTES):
        raise ValueError(f"by must be one of {BY_ATTRIBUTES}: {by}")
    keys = ['seniority', *([by] if by else [])]
    qs = ', '.join(str(float(q)) for q in quantiles)
    aggs = ', '.join(f"count({m}) AS {m}__n, avg({m}) AS {m}__mean, approx_quantile({m}, [{qs}]) AS {m}__q"
                     for m in metrics)
    df = con.execute(f"""
        WITH m AS (SELECT unnest(fields) AS field, {', '.join(keys)}, {', '.join(metrics)}
                   FROM {_src(gt_dir, 'gt_authors')})
        SELECT coalesce(field, '{ALL_FIELDS}') AS field, {', '.join(keys)}, {aggs}
        FROM m GROUP BY GROUPING SETS ((field, {', '.join(keys)}), ({', '.join(keys)}))""").df()
    rows = []
    for _, r in df.iterrows():
        for m in metrics:
            values = r[f"{m}__q"]
            row = {'field': r['field'], **{k: r[k] for k in keys}, 'metric': m, 'n': int(r[f"{m}__n"]),
                   'mean': r[f"{m}__mean"]}
            row.update({quantile_column(q): (float(v) if v is not None else np.nan)
                        for q, v in zip(quantiles, values if values is not None else [None] * len(quantiles))})
            rows.append(row)
    out = pd.DataFrame(rows)
    if by:
        out = out.sort_values(by, kind='stable')
    out = _sort_cells(out)
    out['metric'] = pd.Categorical(out['metric'], list(metrics))
    return out.sort_values('metric', kind='stable').reset_index(drop=True)


def quantile_column(q: float) -> str:
    return f"q{round(q * 100):02d}"


def histogram(con: duckdb.DuckDBPyConnection, gt_dir: Path | str, column: str, low: float, high: float,
              n_bins: int, by_field: bool = False) -> pd.DataFrame:
    """Counts of a gt_authors numeric column in n_bins equal bins on [low, high] (values outside are clipped to the
    first / last bin; NULL / NaN left out): bin, left, right, n (+ field, memberships, if by_field)."""
    classes = list(dict.fromkeys(cons.RACEBERT_LABELS.values()))
    if column not in ('academic_age', 'p_female', 'p_ethnicity_max', *(f'p_{c}' for c in classes), *cons.GT_STATS_METRICS):
        raise ValueError(f"Unknown column for a GT histogram: {column}")
    value = f"greatest({', '.join(f'p_{c}' for c in classes)})" if column == 'p_ethnicity_max' else column
    width = (high - low) / n_bins
    source = (f"(SELECT unnest(fields) AS field, {value} AS v FROM {_src(gt_dir, 'gt_authors')})" if by_field
              else f"(SELECT {value} AS v FROM {_src(gt_dir, 'gt_authors')})")
    group = 'field, ' if by_field else ''
    df = con.execute(f"""
        SELECT {group}least(greatest(floor((v - {low}) * {n_bins / (high - low)!r}), 0), {n_bins - 1})::INTEGER AS bin, count(*) AS n
        FROM {source} WHERE v IS NOT NULL AND NOT isnan(v::DOUBLE)
        GROUP BY ALL""").df()
    if by_field:
        grid = pd.MultiIndex.from_product([field_order(df['field'].unique()), range(n_bins)],
                                          names=['field', 'bin']).to_frame(index=False)
        out = grid.merge(df, on=['field', 'bin'], how='left')
    else:
        out = pd.DataFrame({'bin': range(n_bins)}).merge(df, on='bin', how='left')
    out['n'] = out['n'].fillna(0).astype('int64')
    out['left'] = low + out['bin'] * width
    out['right'] = out['left'] + width
    return out[[*(['field'] if by_field else []), 'bin', 'left', 'right', 'n']]


# --- perceived demographics --------------------------------------------------------------------------------------
def gender_coverage(con: duckdb.DuckDBPyConnection, gt_dir: Path | str) -> pd.DataFrame:
    """Scholars by perceived gender label, with the reason of 'unknown': no usable first name (initials only, ...) or
    a first name without a nomquamgender probability. Columns label, n, share."""
    df = con.execute(f"""
        SELECT CASE WHEN gender <> '{UNKNOWN}' THEN gender
                    WHEN gender_first_name IS NULL OR gender_first_name = '' THEN 'unknown: no usable first name'
                    ELSE 'unknown: first name not in nomquamgender' END AS label, count(*) AS n
        FROM {_src(gt_dir, 'gt_authors')} GROUP BY 1 ORDER BY n DESC""").df()
    df['share'] = df['n'] / df['n'].sum()
    return df


def composition(shares: pd.DataFrame, attribute: str, include_unknown: bool = True, top: Optional[int] = None,
                keep: Sequence[str] = (), add_all: bool = True) -> pd.DataFrame:
    """Class shares per field x seniority (rows field, seniority; columns = classes) from gt_shares. With add_all, the
    seniority scope is added as field 'All fields'. top: keep the `top` classes with the largest pooled share (+ keep,
    e.g. the prompt countries) and pool the rest into 'other'. Without unknown, shares are renormalised over the known
    classes (as parity does with include_unknown = no)."""
    scopes = [SCOPE_FIELD_SENIORITY, SCOPE_SENIORITY] if add_all else [SCOPE_FIELD_SENIORITY]
    df = shares[(shares['attribute'] == attribute) & shares['scope'].isin(scopes)].copy()
    if df.empty:
        raise ValueError(f"No GT shares for attribute '{attribute}'")
    df['field'] = df['field'].fillna(ALL_FIELDS)
    if not include_unknown:
        df = df[df['class'] != UNKNOWN]
    if top is not None:
        pooled = df.groupby('class')['n'].sum().drop(UNKNOWN, errors='ignore').sort_values(ascending=False)
        kept = set(pooled.index[:top]) | set(keep)
        df['class'] = df['class'].where(df['class'].isin(kept | {UNKNOWN}), OTHER)
    table = df.pivot_table(index=['field', 'seniority'], columns='class', values='n', aggfunc='sum', fill_value=0)
    table = table.div(table.sum(axis=1), axis=0)
    table.columns.name = attribute
    order = _sort_cells(table.reset_index()[['field', 'seniority']])
    return table.reindex(pd.MultiIndex.from_frame(order))


def shares_by_prominence(con: duckdb.DuckDBPyConnection, gt_dir: Path | str, attribute: str,
                         groups: Sequence[str] = tuple(PROMINENCE_GROUPS), include_unknown: bool = False) -> pd.DataFrame:
    """Composition of a perceived attribute (gender / ethnicity) inside each prominence class, per field x seniority
    (+ All fields): columns field, seniority, group (productivity / influence / collaboration / top_<metric>),
    level (L / M / H, or 'top' for the top x % group), class, n, share (within field x seniority x group x level).
    Without unknown, shares are over the known labels (as parity). The top groups compare 'top' with everyone."""
    if attribute not in BY_ATTRIBUTES:
        raise ValueError(f"attribute must be one of {BY_ATTRIBUTES}: {attribute}")
    unknown = '' if include_unknown else f"WHERE class <> '{UNKNOWN}'"
    parts = []
    for group in groups:
        if group.startswith('top_'):
            # 'all' = every member (top included): the reference the top group is compared with
            parts.append(f"SELECT field, seniority, '{group}' AS grp, 'top' AS level, {attribute} AS class "
                         f"FROM {_src(gt_dir, 'gt_members')} WHERE is_{group}")
            parts.append(f"SELECT field, seniority, '{group}' AS grp, 'all' AS level, {attribute} AS class "
                         f"FROM {_src(gt_dir, 'gt_members')}")
        else:
            parts.append(f"SELECT field, seniority, '{group}' AS grp, {group} AS level, {attribute} AS class "
                         f"FROM {_src(gt_dir, 'gt_members')} WHERE {group} <> '{UNKNOWN}'")
    df = con.execute(f"""
        WITH u AS ({' UNION ALL '.join(parts)}), k AS (SELECT * FROM u {unknown})
        SELECT coalesce(field, '{ALL_FIELDS}') AS field, seniority, grp AS "group", level, class, count(*) AS n
        FROM k GROUP BY GROUPING SETS ((field, seniority, grp, level, class), (seniority, grp, level, class))""").df()
    df['share'] = df['n'] / df.groupby(['field', 'seniority', 'group', 'level'])['n'].transform('sum')
    return _sort_cells(df)


def cutoffs_table(cutoffs: pd.DataFrame, bin_labels: Sequence[str] = cons.DEFAULT_GT_BIN_LABELS) -> pd.DataFrame:
    """One row per field x seniority, one column per attribute: 'L 0-4 | M 5-13 | H 14-2352' (value ranges of the
    rank-based bins; ties share a bin, so the bins are not equal-sized: see share_<class>)."""
    rows = []
    for (field, seniority, attribute), g in cutoffs.groupby(['field', 'seniority', 'attribute'], sort=False):
        g = g.set_index('class')
        known = [c for c in bin_labels if c in g.index]
        total = g['n'].sum()
        text = ' | '.join(f"{c} {_fmt(g.at[c, 'min_value'])}-{_fmt(g.at[c, 'max_value'])}" for c in known)
        row = {'field': field, 'seniority': seniority, 'attribute': attribute, 'ranges': text}
        row.update({f"share_{c}": g.at[c, 'n'] / total for c in g.index})
        rows.append(row)
    out = _sort_cells(pd.DataFrame(rows))
    return out


def _fmt(value) -> str:
    value = float(value)
    return f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}"

