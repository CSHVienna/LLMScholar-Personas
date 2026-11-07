"""
Expected social metrics under a RANDOM recommender (plan D.7.2): the mean of each social metric over random lists of
n people drawn from the GT of the requested field x seniority. The social step stores

    <metric>_adj = observed - expected(field, seniority, n)

a difference in the metric's own units: 0 = what a random list of the same size gets on average, > 0 = higher,
< 0 = lower (not a z-score: dividing by the SD, which shrinks with n, would bring the size effect back).
n = the people of the response (matched authors of the requested field), not k.

How each expected value is obtained:
    parity_<a>, diversity_<a>   only the class labels matter, and the labels of random members occur with the GT class
                                shares of the field x seniority (unknown included): LABEL_DRAWS lists of n labels per
                                n, the metric computed exactly as for the observed lists (same unknown handling, same
                                target q = the parity target of the config)
    popularity_<m>, _location   the target's top share (a share is unbiased at any n): no simulation, done in social.py
    connectedness               real people: PEOPLE_DRAWS random lists of N_MAX members (without replacement; the first
                                n of a list are a random list of n) and their coauthorship edges
    similarity                  does not depend on n: the mean pairwise cosine of SIMILARITY_SAMPLE random members
Lists longer than N_MAX use the value at N_MAX. Random numbers are seeded per (seed, field, seniority, metric).
"""

import json
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from groundtruth.db import sql_path
from metrics import social as so

N_MAX = 40                  # longest list simulated (observed: <= 31)
LABEL_DRAWS = 10_000        # random label lists per (field, seniority, attribute, n)
PEOPLE_DRAWS = 1_000        # random people lists per (field, seniority) for connectedness
SIMILARITY_SAMPLE = 5_000   # random members per (field, seniority) for similarity
COLUMNS = ['field', 'seniority', 'metric', 'n', 'expected']
ADJ_SUFFIX = '_adj'
SIMULATED = ([f"parity_{a}" for a in so.ATTRIBUTES] + [f"diversity_{a}" for a in so.ATTRIBUTES]
             + ['connectedness', 'similarity'])                          # from the table (popularity: in social.py)


def rng_for(seed: int, *keys: str) -> np.random.Generator:
    """Reproducible generator per (seed, keys), whatever the order things are computed in."""
    return np.random.default_rng([seed] + [zlib.crc32(str(k).encode()) for k in keys])


# --- labels: parity, diversity -----------------------------------------------------------------------------------
def label_expectation(population: pd.Series, q: pd.Series, n: int, include_unknown: bool,
                      rng: np.random.Generator, draws: int = LABEL_DRAWS) -> Tuple[float, float]:
    """(expected parity, expected diversity) of n labels drawn with the shares ``population`` (unknown included),
    computed like social.parity / social.diversity: p over the known classes unless include_unknown, against q."""
    classes = list(population.index)
    shares = population.to_numpy(dtype=float)
    counts = rng.multinomial(n, shares / shares.sum(), size=draws).astype(float)
    if not include_unknown and so.UNKNOWN in classes:
        keep = [i for i, c in enumerate(classes) if c != so.UNKNOWN]
        counts, classes = counts[:, keep], [classes[i] for i in keep]
    n_known = counts.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        p = counts / n_known[:, None]
    union = list(dict.fromkeys(classes + list(q.index)))
    p_full = np.zeros((draws, len(union)))
    p_full[:, [union.index(c) for c in classes]] = p
    q_full = q.reindex(union, fill_value=0.0).to_numpy(dtype=float)
    parity = 1 - 0.5 * np.abs(p_full - q_full).sum(axis=1)
    parity[n_known == 0] = np.nan
    if len(q) >= 2:
        with np.errstate(invalid='ignore', divide='ignore'):
            entropy = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1) / np.log(len(q))
        entropy[n_known < 2] = np.nan
    else:
        entropy = np.full(draws, np.nan)
    with np.errstate(invalid='ignore'):
        mean = lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
        return mean(parity), mean(entropy)


def label_table(gt: 'so.GroundTruth', groups: Iterable[Tuple[str, str]], scope: str, include_unknown: bool,
                seed: int, n_max: int = N_MAX, draws: int = LABEL_DRAWS) -> pd.DataFrame:
    rows = []
    for field, seniority in groups:
        for name, column in so.ATTRIBUTES.items():
            population = gt.target('field_x_seniority', field, seniority, column)   # who a random list is drawn from
            q = so.known(gt.target(scope, field, seniority, column), include_unknown)
            if population.empty or q.empty:
                continue
            rng = rng_for(seed, field, seniority, name)
            for n in range(1, n_max + 1):
                parity, diversity = label_expectation(population, q, n, include_unknown, rng, draws)
                rows += [(field, seniority, f"parity_{name}", n, parity), (field, seniority, f"diversity_{name}", n, diversity)]
    return pd.DataFrame(rows, columns=COLUMNS)


# --- real people: connectedness, similarity ---------------------------------------------------------------------
def sample_members(gt: 'so.GroundTruth', field: str, seniority: str, size: int, seed: int) -> np.ndarray:
    """``size`` distinct random author ids among the members of a field x seniority (fewer if the group is smaller)."""
    ids = gt.con.execute(
        f"SELECT author_id FROM (SELECT author_id FROM read_parquet({sql_path(gt.dir / 'gt_members.parquet')}) "
        f"WHERE field = ? AND seniority = ?) USING SAMPLE {int(size)} ROWS (reservoir, {int(seed) % (2 ** 31)})",
        # (filter first: DuckDB samples BEFORE a WHERE clause of the same SELECT)
        [field, seniority]).df()['author_id'].to_numpy(dtype='int64')
    return rng_for(seed, field, seniority, 'order').permutation(ids)


def connectedness_curve(lists: np.ndarray, edges: Iterable[Tuple[int, int]], n_max: int) -> Dict[int, float]:
    """{n: mean connectedness of the first n people of every list}, for n = 2..n_max."""
    neighbours: Dict[int, set] = {}
    for a, b in edges:
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    values = {n: [] for n in range(2, n_max + 1)}
    for people in lists:
        people = [int(p) for p in people]
        position = {p: i for i, p in enumerate(people)}
        local = [(position[a], position[b]) for a in people for b in neighbours.get(a, ()) if b in position
                 and position[a] < position[b]]
        for n in range(2, min(n_max, len(people)) + 1):
            values[n].append(so.connectedness(range(n), [(i, j) for i, j in local if j < n]))
    return {n: float(np.mean(v)) if v else np.nan for n, v in values.items()}


def mean_pairwise_cosine(vectors: np.ndarray) -> float:
    """Mean cosine over all pairs (like social.similarity, zero vectors count as cosine 0), in closed form."""
    if len(vectors) < 2:
        return np.nan
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.divide(vectors, norms, out=np.zeros_like(vectors, dtype=float), where=norms > 0)
    total = unit.sum(axis=0)
    pairs_sum = float(total @ total - (unit * unit).sum())          # sum over i != j of u_i . u_j
    return pairs_sum / (len(unit) * (len(unit) - 1))


def people_table(gt: 'so.GroundTruth', groups: Sequence[Tuple[str, str]], seed: int, n_max: int = N_MAX,
                 draws: int = PEOPLE_DRAWS, sample: int = SIMILARITY_SAMPLE) -> pd.DataFrame:
    lists = {}
    for field, seniority in groups:
        ids = sample_members(gt, field, seniority, draws * n_max, seed)
        length = min(n_max, len(ids))                                  # small groups (tests, smoke): shorter lists
        if length >= 2:
            lists[(field, seniority)] = ids[:(len(ids) // length) * length].reshape(-1, length)
    all_ids = np.unique(np.concatenate([l.ravel() for l in lists.values()])) if lists else np.array([], dtype='int64')
    edges = gt.edges(all_ids)                                          # one scan of the edge list for every group
    edge_list = list(zip(edges['a'], edges['b']))
    rows = []
    for (field, seniority), group_lists in lists.items():
        for n, value in connectedness_curve(group_lists, edge_list, n_max).items():
            rows.append((field, seniority, 'connectedness', n, value))
        sample_ids = sample_members(gt, field, seniority, sample, seed + 1)
        vectors = gt.features(sample_ids).to_numpy() if len(sample_ids) else np.empty((0, 0))
        mu = mean_pairwise_cosine(vectors)
        rows += [(field, seniority, 'similarity', n, mu) for n in range(2, n_max + 1)]
    return pd.DataFrame(rows, columns=COLUMNS)


def expected_table(gt: 'so.GroundTruth', groups: Sequence[Tuple[str, str]], scope: str, include_unknown: bool,
                   seed: int, n_max: int = N_MAX) -> pd.DataFrame:
    """All expected values: one row per (field, seniority, metric, n)."""
    return pd.concat([label_table(gt, groups, scope, include_unknown, seed, n_max),
                      people_table(gt, groups, seed, n_max)], ignore_index=True)


# --- use ---------------------------------------------------------------------------------------------------------
class Expected:
    """Look-up of an expected-values table: value(metric, field, seniority, n), n capped at the table's maximum."""

    def __init__(self, table: pd.DataFrame):
        self.n_max = int(table['n'].max()) if len(table) else 0
        self.values = {(r.metric, r.field, r.seniority, int(r.n)): float(r.expected) for r in table.itertuples()}

    def value(self, metric: str, field: str, seniority: str, n: int) -> float:
        if not n or n < 1:
            return np.nan
        return self.values.get((metric, field, seniority, min(int(n), self.n_max)), np.nan)

    def adjust(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """``metrics`` (social rows with field, seniority, n_people) with <metric>_adj for the simulated metrics."""
        out = metrics.copy()
        for metric in SIMULATED:
            if metric in out:
                expected = [self.value(metric, f, s, n) for f, s, n in
                            zip(out['field'], out['seniority'], out['n_people'])]
                out[metric + ADJ_SUFFIX] = out[metric] - np.array(expected, dtype=float)
        return out


# --- files -------------------------------------------------------------------------------------------------------
EXPECTED_DIR = 'social_expected'                  # <metrics dir>/social_expected/expected.parquet + _params.json


def expected_file(metrics_dir: Path | str) -> Path:
    return Path(metrics_dir) / EXPECTED_DIR / 'expected.parquet'


def depends_on(gt_dir, include_unknown: bool, parity_target: str) -> dict:
    """What the expected values depend on (recorded with the table; the social step checks they match)."""
    return {'gt_dir': str(Path(gt_dir).resolve()), 'include_unknown': bool(include_unknown), 'parity_target': parity_target}


def load(metrics_dir, gt_dir, include_unknown: bool, parity_target: str) -> Expected:
    """The expected-values table of a metrics folder, checked against the settings of the social step."""
    path = expected_file(metrics_dir)
    if not path.exists():
        raise FileNotFoundError(f"No expected values {path}: run the step 'expected' first "
                                f"(run_metrics_pipeline.sh -s expected, or batch_social_expected.py)")
    stored = json.loads((path.parent / '_params.json').read_text())
    wanted = depends_on(gt_dir, include_unknown, parity_target)
    differ = {k: (stored.get(k), v) for k, v in wanted.items() if stored.get(k) != v}
    if differ:
        raise ValueError(f"{path} was made with other settings {differ} (table, now): re-run the step 'expected' "
                         f"with --overwrite")
    return Expected(pd.read_parquet(path))
