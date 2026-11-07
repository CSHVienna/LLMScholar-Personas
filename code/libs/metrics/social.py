"""
Social metrics (plan §2.3, step 9): per response, on its PEOPLE = the matched authors (B.6 author_id, i.e.
matched_author) of its unique valid names, in a valid response; a person named twice counts once.

Attributes of a person come from the GT row of the REQUESTED field (A.4 gt_members): perceived gender, perceived
ethnicity, location (country), productivity, influence, collaboration (classes within field x seniority), and the
top groups by works / citations.

    parity_<a>        1 - TV(p, q): p = the people's class distribution, q = the GT target (gt_shares, scope
                      field_x_seniority = requested field and seniority by default; field / seniority / all = robustness)
    diversity_<a>     normalised Shannon entropy of p: H(p) / ln K, K = number of classes of the target; undefined for
                      fewer than 2 people (e.g. k = 1)
    unknown_<a>       share of the people with an unknown class ('unknown')
    include_unknown   False (default): 'unknown' is dropped from p and q, both renormalised over the known classes;
                      True: 'unknown' is one more class
    popularity_<m>    share of the people in the GT top group by m (works = productivity, citations = influence);
                      expected = the target's top share
    popularity_location  share of the people (known countries) from the TOP COUNTRIES of the target: the
                      `top_countries` countries with the most GT scholars (known countries; ties with the k-th
                      included); expected = the GT share of those countries
    connectedness     1 - normalised entropy of the sizes of the coauthorship components among the people
                      (1 = all connected, 0 = nobody coauthored); undefined for fewer than 2 people
    similarity        mean pairwise cosine of the people in a PCA space (>= 90 % of the variance) of log1p-standardised
                      bibliometrics, fitted on the GT population; undefined for fewer than 2 people
    locality          share of the people (known countries) from the PROMPTED country (prompt location ->
                      cons.LOCATION_COUNTRY; NaN if not mapped); expected = the GT share of that country in the
                      target (known countries)
The signed gaps p - q per class are returned as a long table (gaps).
Every social metric also gets <metric>_expected (its value under a random recommender) and the size-adjusted
<metric>_adj = observed - expected (metrics/expected.py):
popularity and locality from their expected share (here), the others from an expected-values table (Expected,
optional).
"""

from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd

from groundtruth.db import shared_gt_dir, sql_path
from metrics import technical as tm
from utils import constants as cons

UNKNOWN = 'unknown'
ATTRIBUTES = {'gender': 'gender', 'ethnicity': 'ethnicity', 'location': 'country', 'productivity': 'productivity',
              'influence': 'influence', 'collaboration': 'collaboration'}      # metric name -> gt_members / gt_shares
POPULARITY = list(cons.GT_TOP_METRICS)                                            # productivity (works), influence
PCA_FEATURES = ['n_works', 'n_citations', 'h_index', 'i10_index', 'e_index', 'two_year_mean_citedness', 'n_coauthors',
                'citations_per_paper_per_year', 'academic_age']
PCA_VARIANCE = 0.90
SCOPES = ['field_x_seniority', 'field', 'seniority', 'all']
METRIC_KEYS = ([f"parity_{a}" for a in ATTRIBUTES] + [f"diversity_{a}" for a in ATTRIBUTES]
               + [f"popularity_{m}" for m in POPULARITY] + ['popularity_location', 'connectedness', 'similarity', 'locality'])
ADJUSTED_KEYS = [f"{key}_adj" for key in METRIC_KEYS]                                  # metrics/expected.py
EXPECTED_KEYS = [f"{key}_expected" for key in METRIC_KEYS]                        # random recommender
DEFAULT_TOP_COUNTRIES = 2


# --- people ------------------------------------------------------------------------------------------------------
def response_people(responses: pd.DataFrame, recs: pd.DataFrame, matches: pd.DataFrame,
                    valid_response_tags: Iterable[str], valid_rec_tags: Iterable[str],
                    order_sensitive_keys: bool = False) -> pd.DataFrame:
    """RESPONSE_KEY + author_id: the matched authors of the unique valid names of every valid response (each once)."""
    valid_responses = responses[tm.outcome(responses[cons.COL_RESPONSE_TAGS], valid_response_tags) == tm.OUTCOME_VALID]
    keep = set(tm.row_id(valid_responses, tm.RESPONSE_KEY))
    valid = recs[tm.is_valid(recs[cons.COL_REC_TAGS], valid_rec_tags)].copy()
    valid['rid'], valid['xid'] = tm.row_id(valid, tm.RESPONSE_KEY), tm.row_id(valid, tm.REC_KEY)
    valid = valid[valid['rid'].isin(keep)]
    valid['dkey'] = tm.duplicate_keys(valid, order_sensitive_keys)
    valid = valid.sort_values(tm.REC_KEY).drop_duplicates(['rid', 'dkey'])
    matched = matches.assign(xid=tm.row_id(matches, tm.REC_KEY)).set_index('xid')[['matched_author', 'author_id']]
    valid = valid.join(matched, on='xid')
    people = valid[valid['matched_author'].fillna(False).astype(bool) & valid['author_id'].notna()]
    people = people.drop_duplicates(['rid', 'author_id'])
    return people[tm.RESPONSE_KEY + ['author_id']].assign(author_id=lambda d: d['author_id'].astype('int64'))


# --- GT reference ------------------------------------------------------------------------------------------------
class GroundTruth:
    """The GT tables the social metrics need: A.4 outputs in gt_dir (population/<variant>/), the shared
    coauthorship edges (A.1a) in the GT root."""

    def __init__(self, gt_dir: Path | str, con: Optional[duckdb.DuckDBPyConnection] = None):
        self.dir = Path(gt_dir)
        self.con = con or duckdb.connect()
        self.shares = pd.read_parquet(self.dir / 'gt_shares.parquet')
        self._pca = None

    def members(self, author_ids: Sequence[int]) -> pd.DataFrame:
        """gt_members rows (all fields) of the given authors."""
        ids = pd.DataFrame({'author_id': pd.Series(list(set(author_ids)), dtype='int64')})
        self.con.register('wanted_ids', ids)
        return self.con.execute(f"SELECT m.* FROM read_parquet({sql_path(self.dir / 'gt_members.parquet')}) m "
                                f"SEMI JOIN wanted_ids USING (author_id)").df()

    def edges(self, author_ids: Sequence[int]) -> pd.DataFrame:
        """Coauthorship edges (undirected, a < b) with both ends among the given authors."""
        files = sorted((shared_gt_dir(self.dir) / 'coauthor_edges').glob('bucket-*.parquet'))
        if not files or not len(author_ids):
            return pd.DataFrame({'a': pd.Series(dtype='int64'), 'b': pd.Series(dtype='int64')})
        self.con.register('wanted_ids', pd.DataFrame({'author_id': pd.Series(list(set(author_ids)), dtype='int64')}))
        return self.con.execute(f"""
            SELECT e.author_id AS a, e.coauthor_id AS b FROM read_parquet({[str(f) for f in files]}) e
            WHERE e.author_id < e.coauthor_id
              AND e.author_id IN (SELECT author_id FROM wanted_ids) AND e.coauthor_id IN (SELECT author_id FROM wanted_ids)
        """).df()

    def pca(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(mean, std, components) of the log1p features over the GT population; components keep >= PCA_VARIANCE."""
        if self._pca is None:
            authors = sql_path(self.dir / 'gt_authors.parquet')
            logs_ = ', '.join(f"ln(1 + greatest(coalesce({f}, 0), 0)) AS {f}" for f in PCA_FEATURES)
            self.con.execute(f"CREATE OR REPLACE TEMP VIEW feats AS SELECT {logs_} FROM read_parquet({authors})")
            stats = self.con.execute("SELECT " + ', '.join(f"avg({f}), stddev_samp({f})" for f in PCA_FEATURES)
                                     + " FROM feats").fetchone()
            mean, std = np.array(stats[0::2], dtype=float), np.array(stats[1::2], dtype=float)
            std[~np.isfinite(std) | (std == 0)] = 1.0
            pairs = [(i, j) for i in range(len(PCA_FEATURES)) for j in range(i, len(PCA_FEATURES))]
            cov_values = self.con.execute("SELECT " + ', '.join(
                f"covar_samp({PCA_FEATURES[i]}, {PCA_FEATURES[j]}) / ({std[i]} * {std[j]})" for i, j in pairs)
                + " FROM feats").fetchone()
            corr = np.zeros((len(PCA_FEATURES), len(PCA_FEATURES)))
            for (i, j), v in zip(pairs, cov_values):
                corr[i, j] = corr[j, i] = v if v is not None else 0.0
            values, vectors = np.linalg.eigh(corr)
            order = np.argsort(values)[::-1]
            values, vectors = values[order], vectors[:, order]
            keep = int(np.searchsorted(np.cumsum(values) / values.sum(), PCA_VARIANCE) + 1)
            self._pca = (mean, std, vectors[:, :keep])
        return self._pca

    def features(self, author_ids: Sequence[int]) -> pd.DataFrame:
        """PCA coordinates of the given authors (index author_id)."""
        mean, std, components = self.pca()
        ids = pd.DataFrame({'author_id': pd.Series(list(set(author_ids)), dtype='int64')})
        self.con.register('wanted_ids', ids)
        logs_ = ', '.join(f"ln(1 + greatest(coalesce({f}, 0), 0)) AS {f}" for f in PCA_FEATURES)
        raw = self.con.execute(f"SELECT author_id, {logs_} FROM read_parquet({sql_path(self.dir / 'gt_authors.parquet')}) "
                               f"SEMI JOIN wanted_ids USING (author_id)").df().set_index('author_id')
        return pd.DataFrame(((raw[PCA_FEATURES].to_numpy() - mean) / std) @ components, index=raw.index)

    def target(self, scope: str, field: Optional[str], seniority: Optional[str], attribute: str) -> pd.Series:
        """GT class shares (class -> share) of an attribute in a scope (NaN-free; may be empty)."""
        s = self.shares
        rows = s[(s['scope'] == scope) & (s['attribute'] == attribute)]
        if scope in ('field_x_seniority', 'field'):
            rows = rows[rows['field'] == field]
        if scope in ('field_x_seniority', 'seniority'):
            rows = rows[rows['seniority'] == seniority]
        return rows.set_index('class')['share']


# --- measures ----------------------------------------------------------------------------------------------------
def known(distribution: pd.Series, include_unknown: bool) -> pd.Series:
    """The distribution without 'unknown' (renormalised) unless include_unknown."""
    d = distribution if include_unknown else distribution.drop(UNKNOWN, errors='ignore')
    total = d.sum()
    return d / total if total > 0 else d.iloc[0:0]


def top_countries(shares: pd.Series, k: int) -> set:
    """The k countries with the largest shares of the scholars (known countries); countries tied with the k-th are
    included (all countries if there are fewer than k)."""
    ranked = known(shares, include_unknown=False).sort_values(ascending=False)
    if ranked.empty or k < 1:
        return set()
    last = min(int(k), len(ranked)) - 1
    return set(ranked.index[:last + 1]) | set(ranked.index[np.isclose(ranked.to_numpy(), ranked.iloc[last])])


def parity(p: pd.Series, q: pd.Series) -> float:
    """1 - total variation distance between two class distributions (on the union of the classes)."""
    if p.empty or q.empty:
        return np.nan
    classes = p.index.union(q.index)
    return 1 - 0.5 * float(np.abs(p.reindex(classes, fill_value=0) - q.reindex(classes, fill_value=0)).sum())


def diversity(p: pd.Series, n_classes: int, n_people: int) -> float:
    """Shannon entropy of p normalised by ln(n_classes); NaN for < 2 people or < 2 classes."""
    if n_people < 2 or n_classes < 2 or p.empty:
        return np.nan
    nonzero = p[p > 0].to_numpy()
    return float(-(nonzero * np.log(nonzero)).sum() / np.log(n_classes))


def connectedness(people: Sequence[int], edges: Iterable[Tuple[int, int]]) -> float:
    """1 - normalised entropy of the coauthorship component sizes (union-find); NaN for < 2 people."""
    people = list(dict.fromkeys(people))
    n = len(people)
    if n < 2:
        return np.nan
    parent = {p: p for p in people}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        if a in parent and b in parent:
            parent[root(a)] = root(b)
    sizes = pd.Series([root(p) for p in people]).value_counts().to_numpy() / n
    return float(1 + (sizes * np.log(sizes)).sum() / np.log(n))


def similarity(vectors: np.ndarray) -> float:
    """Mean pairwise cosine similarity of the rows; NaN for < 2 rows."""
    if len(vectors) < 2:
        return np.nan
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
    return float(np.mean([unit[i] @ unit[j] for i, j in combinations(range(len(unit)), 2)]))


# --- per response ------------------------------------------------------------------------------------------------
def social_metrics(responses: pd.DataFrame, people: pd.DataFrame, gt: GroundTruth, include_unknown: bool = False,
                   scope: str = 'field_x_seniority', top_k_countries: int = DEFAULT_TOP_COUNTRIES,
                   expectations=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(metrics: one row per response, gaps: one row per response x attribute x class). ``expectations``
    (metrics.expected.Expected) adds <metric>_expected and _adj of the simulated metrics; those of popularity_* and
    locality are always added."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    ids = people['author_id'].unique()
    members = gt.members(ids)
    edges = gt.edges(ids)
    edge_list = list(zip(edges['a'], edges['b']))
    vectors = gt.features(ids) if len(ids) else pd.DataFrame()
    people = people.assign(rid=tm.row_id(people, tm.RESPONSE_KEY))
    by_response = {rid: g for rid, g in people.groupby('rid')}
    member_index = members.set_index(['author_id', 'field'])
    targets: Dict[tuple, pd.Series] = {}

    rows, gaps = [], []
    for _, response in responses.iterrows():
        rid = tm.row_id(response.to_frame().T, tm.RESPONSE_KEY).iloc[0]
        field, seniority = response[cons.VAR_FIELD], response[cons.VAR_SENIORITY]
        group = by_response.get(rid)
        ids_r = list(group['author_id']) if group is not None else []
        rows_r = member_index.reindex([(a, field) for a in ids_r]).dropna(how='all')
        row = {**{c: response[c] for c in tm.RESPONSE_KEY}, 'n_people': len(rows_r)}
        for name, column in ATTRIBUTES.items():
            key = (scope, field, seniority, column)
            if key not in targets:
                targets[key] = gt.target(scope, field, seniority, column)
            q = known(targets[key], include_unknown)
            counts = rows_r[column].fillna(UNKNOWN).value_counts() if len(rows_r) else pd.Series(dtype=float)
            p = known(counts / counts.sum(), include_unknown) if len(counts) else counts
            n_known = int(counts.drop(UNKNOWN, errors='ignore').sum()) if not include_unknown else int(counts.sum())
            row[f"parity_{name}"] = parity(p, q)
            row[f"diversity_{name}"] = diversity(p, len(q), n_known)
            row[f"unknown_{name}"] = float(counts.get(UNKNOWN, 0) / counts.sum()) if len(counts) else np.nan
            for cls in p.index.union(q.index):
                gaps.append({**{c: response[c] for c in tm.RESPONSE_KEY}, 'attribute': name, 'class': cls,
                             'p': float(p.get(cls, 0.0)), 'q': float(q.get(cls, 0.0)),
                             'gap': float(p.get(cls, 0.0) - q.get(cls, 0.0))})
        for name in POPULARITY:
            key = (scope, field, seniority, f"top_{name}")
            if key not in targets:
                targets[key] = gt.target(scope, field, seniority, f"top_{name}")
            expected = float(targets[key].get('top', np.nan))
            observed = float(rows_r[f"is_top_{name}"].astype(float).mean()) if len(rows_r) else np.nan
            row[f"popularity_{name}"] = observed
            row[f"popularity_{name}_expected"] = expected
            row[f"popularity_{name}_adj"] = observed - expected
        key = (scope, field, seniority, ATTRIBUTES['location'])
        if key not in targets:
            targets[key] = gt.target(scope, field, seniority, ATTRIBUTES['location'])
        countries = known(targets[key], include_unknown=False)
        top = top_countries(countries, top_k_countries)
        located = rows_r[ATTRIBUTES['location']].fillna(UNKNOWN) if len(rows_r) else pd.Series(dtype=object)
        located = located[located != UNKNOWN]
        observed = float(located.isin(top).mean()) if len(located) else np.nan
        expected = float(countries[countries.index.isin(top)].sum()) if len(top) else np.nan
        row['popularity_location'] = observed
        row['popularity_location_expected'] = expected
        row['popularity_location_adj'] = observed - expected
        home = cons.LOCATION_COUNTRY.get(str(response.get(cons.VAR_LOCATION)))
        observed = float((located == home).mean()) if home and len(located) else np.nan
        expected = float(countries.get(home, 0.0)) if home and len(countries) else np.nan
        row['locality'] = observed
        row['locality_expected'] = expected
        row['locality_adj'] = observed - expected
        present = [a for a, _ in rows_r.index] if len(rows_r) else []
        row['connectedness'] = connectedness(present, edge_list)
        row['similarity'] = similarity(vectors.reindex(present).dropna().to_numpy()) if len(present) else np.nan
        rows.append(row)
    metrics = pd.DataFrame(rows)
    if expectations is not None and len(metrics):
        metrics = expectations.adjust(metrics.assign(field=responses[cons.VAR_FIELD].to_numpy(),
                                                 seniority=responses[cons.VAR_SENIORITY].to_numpy()))
    return metrics, pd.DataFrame(gaps)
