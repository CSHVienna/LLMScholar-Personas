"""
Unit tests for the ground-truth steps (A.0-A.4) and the name-based demographics helpers,
on a tiny synthetic OpenAlex database (same column names / nesting as the real one). No model is loaded.

Run from the project root:
    python -m unittest discover -s code/tests -v
"""

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'libs'))

import duckdb                                                            # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

from demographics.ethnicity import racebert_name                         # noqa: E402
from demographics.gender import first_name_of                            # noqa: E402
from demographics import stages                                          # noqa: E402
from groundtruth import db                                               # noqa: E402
from groundtruth.authorships import check_params, missing_chunks, n_chunks, write_chunk  # noqa: E402
from groundtruth.bibliometrics import compute_bibliometrics              # noqa: E402
from groundtruth.coauthorship import write_edges_bucket, write_network, write_pairs_chunk  # noqa: E402
from groundtruth.fields import load_field_mapping                        # noqa: E402
from groundtruth.names import name_keys, write_name_index                # noqa: E402
from groundtruth.population import create_views, gender_label_sql, write_outputs  # noqa: E402
from responses.recommendations import NameTagger                         # noqa: E402
from utils import constants as cons                                      # noqa: E402
from utils.ios import to_parquet as ios_to_parquet                       # noqa: E402

N_BUCKETS = 2
CHUNK_SIZE = 4   # works rows per A.0 chunk
CS, MATH, SOCIOLOGY_SUBFIELD = 17, 26, 3312


def _work(wid, year, cites, authors, field=None, subfield=None, paratext=False, retracted=False, xpac=False):
    topic = 'NULL' if field is None else f"{{'field': {{'id': {field}}}, 'subfield': {{'id': {subfield or 0}}}}}"
    authorships = '[' + ', '.join(f"{{'author': {{'id': {a}}}}}" for a in authors) + ']'
    return f"({wid}, {year}, {cites}, {authorships}, {topic}, {paratext}, {retracted}, {xpac})"


# Author 1: w1 (10), w2 (5), w3 (3), w7 (0, listed twice) -> h = 3, e = sqrt(18 - 9) = 3
# w4 (paratext), w5 (year 1850), w8 (retracted), w9 (xpac) are not counted. Author 6 is an organisation.
WORKS = [
    _work(1, 2000, 10, [1, 2], CS),
    _work(2, 2010, 5, [1], CS),
    _work(3, 2020, 3, [1, 3], 33, SOCIOLOGY_SUBFIELD),
    _work(4, 2015, 100, [1], CS, paratext=True),
    _work(5, 1850, 1, [2], CS),
    _work(6, 2018, 2, [2, 3, 4], MATH),
    _work(7, 2019, 0, [1, 1]),
    _work(8, 2005, 7, [5], CS, retracted=True),
    _work(9, 2001, 50, [1], CS, xpac=True),
    _work(10, 2010, 1, [6], CS),
]


def _author(aid, name, alternatives, affiliations='[]', last_known="[]"):
    alts = '[' + ', '.join(f"'{a}'" for a in alternatives) + ']'
    return (f"({aid}, '{name}', {alts}, NULL, 10, 100, "
            f"{{'2yr_mean_citedness': 1.5, 'h_index': 7, 'i10_index': 2}}, {affiliations}, {last_known})")


AUTHORS = [
    _author(1, 'José García-López', ['J. García-López', 'Jose Garcia Lopez'],
            "[{'institution': {'country_code': 'DE'}, 'years': [2010, 2011]},"
            " {'institution': {'country_code': 'US'}, 'years': [2020]}]",
            "[{'country_code': 'FR'}]"),
    _author(2, 'Ada Lovelace', [], last_known="[{'country_code': 'GB'}]"),
    _author(3, 'Wei Zhang', ['Zhang Wei']),
    _author(4, 'Kim Lee', []),
    _author(5, 'Retracted Only', []),
    _author(6, 'Oak Ridge National Lab. (ORNL), Oak Ridge, TN (United States)', []),
]


def make_openalex(path: Path) -> None:
    con = duckdb.connect(str(path))
    con.execute("""CREATE TABLE works (id BIGINT, publication_year INTEGER, cited_by_count INTEGER,
                   authorships STRUCT(author STRUCT(id BIGINT))[],
                   primary_topic STRUCT(field STRUCT(id BIGINT), subfield STRUCT(id BIGINT)),
                   is_paratext BOOLEAN, is_retracted BOOLEAN, is_xpac BOOLEAN)""")
    con.execute(f"INSERT INTO works VALUES {', '.join(WORKS)}")
    con.execute("""CREATE TABLE authors (id BIGINT, display_name VARCHAR, display_name_alternatives VARCHAR[],
                   orcid VARCHAR, works_count INTEGER, cited_by_count INTEGER,
                   summary_stats STRUCT("2yr_mean_citedness" DOUBLE, h_index INTEGER, i10_index INTEGER),
                   affiliations STRUCT(institution STRUCT(country_code VARCHAR), "years" INTEGER[])[],
                   last_known_institutions STRUCT(country_code VARCHAR)[])""")
    con.execute(f"INSERT INTO authors VALUES {', '.join(AUTHORS)}")
    con.close()


class GroundTruthPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        make_openalex(root / 'oa.duckdb')
        cls.args = argparse.Namespace(openalex_db=str(root / 'oa.duckdb'), threads=2)
        cls.fields = load_field_mapping()
        con = db.connect(cls.args)
        cls.authorships_dir = root / 'authorships'
        cls.n_chunks = n_chunks(con, CHUNK_SIZE)                    # 10 works -> 3 chunks
        check_params(cls.authorships_dir, {'n_chunks': cls.n_chunks})
        cls.missing_before = missing_chunks(cls.authorships_dir)
        (cls.authorships_dir / '_tmp_chunk=0').mkdir()               # leftover of a crashed run
        cls.n_authorships = sum(write_chunk(con, cls.authorships_dir, c, CHUNK_SIZE, cons.GT_MIN_YEAR,
                                            cons.GT_REFERENCE_YEAR, N_BUCKETS) for c in range(cls.n_chunks))
        for c in range(cls.n_chunks):                                # A.0b
            write_pairs_chunk(con, cls.authorships_dir, root / 'pairs', c, N_BUCKETS, max_team=100)
            write_pairs_chunk(con, cls.authorships_dir, root / 'pairs_cap2', c, N_BUCKETS, max_team=2)
        cls.edges_dir = root / 'edges'
        for bucket in range(N_BUCKETS):                              # A.1a
            write_edges_bucket(con, root / 'pairs', db.bucket_file(cls.edges_dir, bucket), bucket)
        cls.edges = pd.concat(pd.read_parquet(f) for f in sorted(cls.edges_dir.glob('*.parquet')))
        (root / 'authors').mkdir()
        frames, names = [], []
        for bucket in range(N_BUCKETS):
            df = compute_bibliometrics(con, cls.authorships_dir, cls.edges_dir, bucket, N_BUCKETS, cls.fields)
            df.to_parquet(db.bucket_file(root / 'authors', bucket), index=False)
            frames.append(df)
            names_file = db.bucket_file(root / 'names', bucket)
            write_name_index(con, names_file, bucket, N_BUCKETS)
            names.append(pd.read_parquet(names_file))
        cls.bib = pd.concat(frames).set_index('author_id')
        cls.names = pd.concat(names)
        con.close()
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # --- A.0 ---------------------------------------------------------------------------------------------------
    def test_authorships_filter_and_partition(self):
        # kept works: 1, 2, 3, 6, 7, 10 -> 2 + 1 + 2 + 3 + 2 + 1 authorships (the duplicate on w7 is kept here)
        self.assertEqual(self.n_authorships, 11)
        self.assertEqual(sorted(p.name for p in (self.authorships_dir / 'chunk=0').iterdir()), ['bucket=0', 'bucket=1'])
        rows = duckdb.sql(f"SELECT * FROM read_parquet('{db.authorships_glob(self.authorships_dir)}')").df()
        self.assertEqual(len(rows), 11)
        self.assertTrue(((rows['author_id'] % N_BUCKETS) == rows['bucket']).all())
        self.assertNotIn(4, set(rows['work_id']))
        self.assertNotIn(5, set(rows['work_id']))
        self.assertNotIn(8, set(rows['work_id']))
        self.assertNotIn(9, set(rows['work_id']))                   # xpac

    def test_chunks_resumable(self):
        self.assertEqual(self.n_chunks, 3)
        self.assertEqual(self.missing_before, [0, 1, 2])
        self.assertEqual(missing_chunks(self.authorships_dir), [])
        self.assertFalse(any(p.name.startswith('_tmp') for p in self.authorships_dir.iterdir()))
        with self.assertRaises(ValueError):                          # other parameters cannot be mixed in
            check_params(self.authorships_dir, {'n_chunks': 99})

    def test_check_params_overwrite(self):
        out = Path(self.tmp.name) / 'params_overwrite'
        check_params(out, {'a': [1, 2]})
        with self.assertRaises(ValueError):
            check_params(out, {'a': [2, 1]})
        check_params(out, {'a': [2, 1]}, overwrite=True)            # replaces the stored parameters
        check_params(out, {'a': [2, 1]})

    # --- A.1 ---------------------------------------------------------------------------------------------------
    def test_indices(self):
        a1 = self.bib.loc[1]
        self.assertEqual((a1.n_works, a1.n_citations, a1.h_index, a1.i10_index), (4, 18, 3, 1))
        self.assertAlmostEqual(a1.e_index, 3.0)
        a2 = self.bib.loc[2]
        self.assertEqual((a2.n_works, a2.h_index, a2.first_pub_year), (2, 2, 2000))   # the 1850 work is excluded
        self.assertAlmostEqual(a2.e_index, np.sqrt(12 - 4))

    def test_field_works_primary_topic(self):
        a1 = self.bib.loc[1]
        self.assertEqual((a1.works_computer_science, a1.works_sociology, a1.works_mathematics), (2, 1, 0))
        self.assertEqual(self.bib.loc[4].works_mathematics, 1)

    def test_coauthors_all_works(self):
        n = self.bib['n_coauthors']
        self.assertEqual(n.loc[1], 2)                               # w1 -> 2, w3 -> 3 (w7: only itself)
        self.assertEqual(n.loc[2], 3)                               # 1, and 3 + 4 on w6 (no team cap)
        self.assertEqual(n.loc[4], 2)
        self.assertEqual(n.loc[6], 0)                               # single-author work only
        self.assertEqual(self.bib.loc[1].n_works_topic, 3)          # w7 has no primary topic

    def test_edge_list(self):
        e = self.edges.set_index(['author_id', 'coauthor_id'])
        self.assertEqual(len(e), 10)                                # 5 undirected pairs (1-2, 1-3, 2-3, 2-4, 3-4) x 2
        self.assertEqual(tuple(e.loc[(1, 2), ['n_shared_works', 'min_team']]), (1, 2))
        self.assertAlmostEqual(e.loc[(1, 2), 'newman_weight'], 1.0)  # team of 2 -> 1 / (2 - 1)
        self.assertAlmostEqual(e.loc[(2, 3), 'newman_weight'], 0.5)  # team of 3
        self.assertEqual(e.loc[(2, 3), 'min_team'], 3)
        self.assertNotIn((1, 1), e.index)                           # duplicate authorship on w7 is no self-loop
        capped = duckdb.sql(f"SELECT author_id, coauthor_id FROM read_parquet('{self.root}/pairs_cap2/chunk=*/bucket=*/*.parquet')").fetchall()
        self.assertEqual(sorted(capped), [(1, 2), (1, 3), (2, 1), (3, 1)])   # w6 (3 authors) > max_team 2
        self.assertTrue(((e.reset_index()['author_id'] % N_BUCKETS) ==
                         [b for b, f in enumerate(sorted(self.edges_dir.glob('*.parquet')))
                          for _ in range(len(pd.read_parquet(f)))]).all())   # an author's edges sit in its bucket

    def test_authors_without_counted_works_are_absent(self):
        self.assertNotIn(5, self.bib.index)

    def test_is_person(self):
        self.assertTrue(self.bib.loc[1].is_person)
        self.assertFalse(self.bib.loc[6].is_person)
        names = ['Ada Lovelace', 'Kathryn L. Corp', 'William H. Press', 'Thomas H. Service', 'Wei Zhang',
                 'Cisco Team', 'Fryer Library', 'University of Washington', 'USDOE, Washington, DC (United States)',
                 'Korea Ministry of Education', 'Nelios media']
        rows = duckdb.sql(f"SELECT name, {db.is_person_sql('name', 'orcid')} AS p FROM (VALUES "
                          + ', '.join(f"('{n}', NULL)" for n in names) + ") t(name, orcid)").fetchall()
        self.assertEqual({n for n, p in rows if p}, set(names[:5]))
        with_orcid = duckdb.sql(f"SELECT {db.is_person_sql('n', 'o')} FROM (VALUES ('Team Smith', 'x')) t(n, o)")
        self.assertTrue(with_orcid.fetchone()[0])                   # an ORCID always means a person

    def test_coauthorship_edges(self):
        ids = self.root / 'ids.parquet'
        pd.DataFrame({'author_id': [1, 2, 3, 3]}).to_parquet(ids, index=False)
        con = duckdb.connect()
        n = write_network(con, self.root / 'network.parquet', self.edges_dir, ids)
        net = pd.read_parquet(self.root / 'network.parquet')
        self.assertEqual(n, 3)                                      # w1: 1-2, w3: 1-3, w6: 2-3 (4 is not in the set)
        self.assertEqual(set(zip(net.author_a, net.author_b)), {(1, 2), (1, 3), (2, 3)})
        self.assertEqual(len(net[net.min_team <= 2]), 2)            # filter later: 2-3 only met on a 3-author work

    def test_country(self):
        self.assertEqual(self.bib.loc[1].country, 'US')             # latest affiliation year wins
        self.assertEqual(self.bib.loc[1].last_known_country, 'FR')
        self.assertEqual(self.bib.loc[2].country, 'GB')             # fallback: last known institution
        self.assertTrue(pd.isna(self.bib.loc[3].country))

    # --- A.2 ---------------------------------------------------------------------------------------------------
    def test_name_key_matches_recommendations(self):
        rec_key = NameTagger().tag('José', 'García-López').key
        self.assertEqual(name_keys('José García-López')[0], rec_key)
        self.assertEqual(name_keys('Jose Garcia Lopez')[0], rec_key)
        self.assertEqual(name_keys('Zhang Wei')[0], name_keys('Wei Zhang')[0])
        self.assertEqual(name_keys('John A. Smith')[1], 'john smith')
        self.assertEqual(name_keys('  '), (None, None))

    def test_name_index_all_authors(self):
        self.assertIn(5, set(self.names['author_id']))              # also authors without counted works
        a1 = self.names[self.names['author_id'] == 1]
        self.assertEqual(len(a1), 2)                                # 3 names, 2 distinct keys
        self.assertEqual(int(a1['is_display_name'].sum()), 1)

    # --- A.4 ---------------------------------------------------------------------------------------------------
    def test_population(self):
        nan = float('nan')   # probabilities only (A.3); labels are derived in A.4
        demo = pd.DataFrame({'author_id': [1, 2, 3, 4], 'gender_first_name': ['Jose', 'Ada', 'Wei', None],
                             'p_female': [0.05, 0.95, 0.02, nan],
                             'p_asian': [0.02, 0.05, 0.99, 0.45], 'p_black': [0.03, 0.05, 0.0, 0.30],
                             'p_hispanic': [0.90, 0.10, 0.0, 0.05], 'p_white': [0.05, 0.80, 0.01, 0.15],
                             'p_other': [0.0, 0.0, 0.0, 0.05]})
        (self.root / 'demo').mkdir(exist_ok=True)
        demo.to_parquet(self.root / 'demo' / 'bucket-00.parquet', index=False)
        con = duckdb.connect()
        create_views(con, str(self.root / 'authors' / 'bucket-*.parquet'), str(self.root / 'demo' / '*.parquet'),
                     self.fields, min_field_share=0.0, reference_year=2025, junior_max=10, senior_min=20,
                     single_cutoff=15, bin_edges=[1 / 3, 2 / 3], top_percentile=50)
        counts = write_outputs(con, self.root / 'gt')
        authors = pd.read_parquet(self.root / 'gt' / 'gt_authors.parquet').set_index('author_id')
        members = pd.read_parquet(self.root / 'gt' / 'gt_members.parquet')
        shares = pd.read_parquet(self.root / 'gt' / 'gt_shares.parquet')

        self.assertEqual(authors.loc[1, 'seniority'], 'senior')     # age 25
        self.assertEqual(authors.loc[3, 'seniority'], 'junior')     # first work 2018 -> age 7
        self.assertEqual(sorted(authors.loc[1, 'fields']), ['Computer Science', 'Sociology'])
        self.assertEqual(counts['gt_members'], 7)                   # 1: CS+Soc, 2: CS+Math, 3: Soc+Math, 4: Math

        math_junior = members[(members.field == 'Mathematics') & (members.seniority == 'junior')].set_index('author_id')
        self.assertEqual(math_junior.loc[4, 'productivity'], cons.BIN_LOW)     # 1 work vs 2 works
        self.assertEqual(math_junior.loc[3, 'productivity'], cons.BIN_HIGH)
        self.assertEqual(math_junior.loc[4, 'collaboration'], cons.BIN_LOW)    # 2 vs 3 coauthors
        self.assertAlmostEqual(authors.loc[1, 'share_computer_science'], 2 / 3)
        self.assertEqual(tuple(authors.loc[[1, 2, 3, 4], 'gender']), ('male', 'female', 'male', 'unknown'))
        self.assertEqual(tuple(authors.loc[[1, 2, 3, 4], 'ethnicity']), ('hispanic', 'white', 'asian', 'unknown'))

        con = duckdb.connect()   # a field-share threshold applied later: Sociology is 1/3 of author 1's works
        create_views(con, str(self.root / 'authors' / 'bucket-*.parquet'), None, self.fields, min_field_share=0.5,
                     reference_year=2025, junior_max=10, senior_min=20, single_cutoff=15, bin_edges=[1 / 3, 2 / 3],
                     top_percentile=50, ethnicity_min_prob=0.4)
        a1 = con.execute("SELECT fields, ethnicity FROM gt_authors WHERE author_id = 1").fetchone()
        self.assertEqual(a1[0], ['Computer Science'])
        self.assertEqual(a1[1], 'unknown')                          # no demographics given

        con = duckdb.connect()   # any number of prominence classes (here 4)
        create_views(con, str(self.root / 'authors' / 'bucket-*.parquet'), None, self.fields, min_field_share=0.0,
                     reference_year=2025, junior_max=10, senior_min=20, single_cutoff=15, bin_edges=[0.25, 0.5, 0.75],
                     top_percentile=50, bin_labels=['q1', 'q2', 'q3', 'q4'])
        classes = {r[0] for r in con.execute("SELECT DISTINCT productivity FROM gt_members").fetchall()}
        self.assertTrue(classes and classes <= {'q1', 'q2', 'q3', 'q4', 'unknown'})
        with self.assertRaises(ValueError):
            create_views(con, str(self.root / 'authors' / 'bucket-*.parquet'), None, self.fields, min_field_share=0.0,
                         reference_year=2025, junior_max=10, senior_min=20, single_cutoff=15, bin_edges=[0.5],
                         top_percentile=50, bin_labels=['L', 'M', 'H'])

        totals = shares.groupby(['scope', 'field', 'seniority', 'attribute'], dropna=False)['share'].sum()
        np.testing.assert_allclose(totals.to_numpy(), 1.0)
        self.assertEqual(set(shares['scope']), {'field_x_seniority', 'field', 'seniority', 'all'})
        all_gender = shares[(shares.scope == 'all') & (shares.attribute == 'gender')].set_index('class')['n']
        self.assertEqual(all_gender['male'], 4)                     # pooled (scholar, field) rows: 1 x2, 3 x2


class TestDemographicsStages(unittest.TestCase):
    """A.3 keys -> ethnicity (each distinct name once, sorted by length, resumable) -> merge, with a fake model."""
    N_SHARDS = 3

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.fields = load_field_mapping()
        works = {f.column: 0 for f in self.fields}
        rows = [(1, 'Maria Garcia', True, 1), (2, 'maria garcia', True, 2), (3, 'Wei Zhang', True, 1),
                (4, 'Wei Zhang', True, 3), (5, 'J. Smith', True, 1), (6, ',', True, 1),
                (7, 'Oak Ridge Lab', False, 1), (8, 'No Field', True, 0), (10, 'Ada Lovelace', True, 1)]
        self.authors_dir = self.root / 'authors'
        for bucket in range(N_BUCKETS):
            df = pd.DataFrame([{'author_id': a, 'display_name': n, 'is_person': p, **works,
                                'works_computer_science': w} for a, n, p, w in rows if a % N_BUCKETS == bucket])
            ios_to_parquet(df, db.bucket_file(self.authors_dir, bucket))
        self.out = self.root / 'demographics'
        self.keys_dir, self.eth_dir = self.out / stages.KEYS_DIR, self.out / stages.ETHNICITY_DIR
        self.gender_file = self.out / stages.GENDER_DIR / stages.GENDER_FILE
        self.calls, self.gender_calls = [], []

    def tearDown(self):
        self.tmp.cleanup()

    def fake_probs(self, names):
        self.calls.append(list(names))
        return [{'p_asian': len(n) / 100, 'p_black': 0.1, 'p_hispanic': 0.1, 'p_white': 0.2, 'p_other': 0.0}
                for n in names]

    def fake_gender(self, first_names):
        self.gender_calls.append(list(first_names))
        return np.array([0.95 if n == 'maria' else np.nan if n == 'ada' else 0.4 for n in first_names])

    def run_keys(self):
        stages.check_layout(self.out, N_BUCKETS, self.N_SHARDS, None)
        keys = pd.concat([stages.write_keys_bucket(db.bucket_file(self.authors_dir, b), db.bucket_file(self.keys_dir, b),
                                                   self.fields, self.N_SHARDS)
                          for b in range(N_BUCKETS)]).set_index('author_id')
        stages.write_gender_table(stages.gender_names(duckdb.connect(), self.keys_dir), self.gender_file, self.fake_gender)
        return keys

    def run_ethnicity(self, chunk_names=100):
        con = duckdb.connect()
        for s in range(self.N_SHARDS):
            stages.infer_shard(stages.shard_names(con, self.keys_dir, s), stages.shard_file(self.eth_dir, s),
                               self.fake_probs, chunk_names, log=lambda m: None)

    def run_merge(self):
        con = duckdb.connect()
        return pd.concat([pd.read_parquet(f) for f in [
            (stages.write_merged_bucket(con, db.bucket_file(self.keys_dir, b), self.gender_file, self.eth_dir,
                                        db.bucket_file(self.out, b)), db.bucket_file(self.out, b))[1]
            for b in range(N_BUCKETS)]]).set_index('author_id')

    def test_end_to_end(self):
        keys = self.run_keys()
        self.assertEqual(sorted(keys.index), [1, 2, 3, 4, 5, 6, 10])         # no organisation, no field-less author
        self.assertEqual(keys.loc[1, 'racebert_name'], keys.loc[2, 'racebert_name'])   # 'maria GARCIA'
        self.assertTrue(pd.isna(keys.loc[6, 'racebert_name']) and pd.isna(keys.loc[6, 'shard']))
        self.assertEqual((keys.loc[1, 'gender_first_name'], keys.loc[2, 'gender_first_name']), ('Maria', 'maria'))
        self.assertEqual(self.gender_calls, [['ada', 'maria', 'wei']])        # once per distinct first name, globally
        self.assertEqual(int(keys.loc[3, 'shard']), stages.name_shard('wei ZHANG', self.N_SHARDS))

        self.run_ethnicity()
        inferred = [n for call in self.calls for n in call]
        self.assertEqual(sorted(inferred), sorted({'maria GARCIA', 'wei ZHANG', 'j. SMITH', 'ada LOVELACE'}))  # once each
        for call in self.calls:
            self.assertEqual(call, sorted(call, key=lambda n: (len(n), n)))   # by length
        self.assertEqual(sum(1 for _ in self.eth_dir.glob('*.parts')), 0)

        merged = self.run_merge()
        self.assertEqual(list(merged.columns), stages.OUTPUT_COLUMNS[1:])
        self.assertEqual(sorted(merged.index), [1, 2, 3, 4, 5, 6, 10])
        self.assertAlmostEqual(merged.loc[1, 'p_asian'], len('maria GARCIA') / 100)
        self.assertEqual(merged.loc[1, 'p_asian'], merged.loc[2, 'p_asian'])
        self.assertEqual(merged.loc[3, 'p_asian'], merged.loc[4, 'p_asian'])
        self.assertTrue(merged.loc[6, stages.PROB_COLUMNS].isna().all())       # no name -> unknown later
        self.assertEqual((merged.loc[1, 'p_female'], merged.loc[2, 'p_female']), (0.95, 0.95))
        self.assertEqual((merged.loc[1, 'gender_first_name'], merged.loc[2, 'gender_first_name']), ('Maria', 'maria'))
        self.assertTrue(pd.isna(merged.loc[10, 'p_female']))                   # not in the reference -> unknown later
        self.assertTrue(pd.isna(merged.loc[5, 'gender_first_name']))           # 'J. Smith': initial only
        self.assertTrue(pd.isna(merged.loc[5, 'p_female']))

    def test_resume_after_crash(self):
        self.run_keys()
        con = duckdb.connect()
        shard = next(s for s in range(self.N_SHARDS) if len(stages.shard_names(con, self.keys_dir, s)) >= 2)
        names = stages.shard_names(con, self.keys_dir, shard)
        out = stages.shard_file(self.eth_dir, shard)

        def crash_on_second(chunk):
            if len(self.calls) == 1:
                raise RuntimeError('killed')
            return self.fake_probs(chunk)
        with self.assertRaises(RuntimeError):
            stages.infer_shard(names, out, crash_on_second, 1, log=lambda m: None)
        self.assertEqual(len(list(out.with_name(out.stem + '.parts').glob('part-*.parquet'))), 1)
        df = stages.infer_shard(names, out, self.fake_probs, 1, log=lambda m: None)
        self.assertEqual(len(self.calls), len(names))                         # the stored part was not redone
        self.assertEqual(df['racebert_name'].tolist(), names)
        self.assertFalse(out.with_name(out.stem + '.parts').exists())

        self.calls.clear()                                                    # parts of another chunk size are discarded
        with self.assertRaises(RuntimeError):
            stages.infer_shard(names, out, crash_on_second, 1, log=lambda m: None)
        self.calls.clear()
        stages.infer_shard(names, out, self.fake_probs, 2, log=lambda m: None)
        self.assertEqual([n for c in self.calls for n in c], names)

    def test_merge_refuses_missing_names(self):
        self.run_keys()
        self.run_ethnicity()
        con = duckdb.connect()
        bucket = 3 % N_BUCKETS                                                # holds author 3 ('Wei Zhang')
        args = (db.bucket_file(self.keys_dir, bucket), self.gender_file, self.eth_dir, db.bucket_file(self.out, bucket))
        gender = pd.read_parquet(self.gender_file)                            # a first name missing in the gender table
        gender[gender['gender_name'] != 'wei'].to_parquet(self.gender_file, index=False)
        with self.assertRaises(RuntimeError):
            stages.write_merged_bucket(con, *args)
        gender.to_parquet(self.gender_file, index=False)
        for s in range(self.N_SHARDS):                                        # a full name missing in its shard
            f = stages.shard_file(self.eth_dir, s)
            df = pd.read_parquet(f)
            if 'wei ZHANG' in set(df['racebert_name']):
                df[df['racebert_name'] != 'wei ZHANG'].to_parquet(f, index=False)
        with self.assertRaises(RuntimeError):
            stages.write_merged_bucket(con, *args)
        self.assertFalse(db.bucket_file(self.out, bucket).exists())
        self.assertFalse(db.bucket_file(self.out, bucket).with_suffix('.parquet.tmp').exists())

    def test_cache_reused_by_name(self):
        """Author-level probabilities of an earlier run (final file + unfinished parts) are reused for every author
        with the same name; only the other names are inferred."""
        self.run_keys()
        cache = self.root / 'cache'
        row = {'p_asian': 0.7, 'p_black': 0.1, 'p_hispanic': 0.1, 'p_white': 0.05, 'p_other': 0.05}
        ios_to_parquet(pd.DataFrame([{'author_id': 1, 'gender_first_name': 'Maria', 'p_female': 0.9, **row},
                                     {'author_id': 6, 'gender_first_name': None, 'p_female': np.nan,
                                      **{c: np.nan for c in row}}]), cache / 'bucket-00.parquet')
        ios_to_parquet(pd.DataFrame([{'author_id': 3, 'gender_first_name': 'Wei', 'p_female': 0.3, **row,
                                      'p_asian': 0.8}]), cache / 'bucket-01.parts' / 'part-00000.parquet')
        (cache / 'bucket-01.parts' / 'part-00001.parquet.tmp').write_text('unfinished write')
        files = stages.cache_files(cache)
        self.assertEqual(len(files), 2)
        con = duckdb.connect()
        for s in range(self.N_SHARDS):
            stages.infer_shard(stages.shard_names(con, self.keys_dir, s), stages.shard_file(self.eth_dir, s),
                               self.fake_probs, 100, log=lambda m: None,
                               cached=stages.cached_probabilities(con, self.keys_dir, s, files))
        self.assertEqual(sorted(n for c in self.calls for n in c), ['ada LOVELACE', 'j. SMITH'])
        merged = self.run_merge()
        self.assertEqual(merged.loc[2, 'p_asian'], 0.7)          # author 2 shares author 1's (cached) name
        self.assertEqual(merged.loc[4, 'p_asian'], 0.8)          # from the unfinished part
        self.assertAlmostEqual(merged.loc[10, 'p_asian'], len('ada LOVELACE') / 100)
        self.assertEqual(stages.cache_files(None), [])
        with self.assertRaises(FileNotFoundError):
            stages.cache_files(self.root / 'nope')

    def test_cache_name_level_and_several_folders(self):
        """Ethnicity shards of an earlier run are matched by exact name and win over the author-level cache; a name
        not in any cache (e.g. changed by the text clean-up) is inferred."""
        self.run_keys()
        old_shards, old_buckets = self.root / 'old_ethnicity', self.root / 'old_demographics'
        row = {'p_asian': 0.1, 'p_black': 0.1, 'p_hispanic': 0.1, 'p_white': 0.6, 'p_other': 0.1}
        ios_to_parquet(pd.DataFrame([{'racebert_name': 'wei ZHANG', **row, 'p_asian': 0.9},
                                     {'racebert_name': 'stale NAME', **row}]), old_shards / 'shard-00.parquet')
        ios_to_parquet(pd.DataFrame([{'racebert_name': 'maria GARCIA', **row, 'p_asian': 0.2}]),
                       old_shards / 'shard-01.parts' / 'part-00000.parquet')
        ios_to_parquet(pd.DataFrame([{'author_id': 3, 'gender_first_name': 'Wei', 'p_female': 0.3, **row,
                                      'p_asian': 0.5}]), old_buckets / 'bucket-00.parquet')
        files = stages.cache_files([old_shards, old_buckets])
        self.assertEqual([stages.is_name_level(f) for f in files], [True, True, False])
        con = duckdb.connect()
        cached = pd.concat([stages.cached_probabilities(con, self.keys_dir, s, files) for s in range(self.N_SHARDS)])
        cached = cached.set_index('racebert_name')['p_asian']
        self.assertEqual(cached['wei ZHANG'], 0.9)                 # name level wins over author 3's row
        self.assertEqual(cached['maria GARCIA'], 0.2)              # from an unfinished shard's part
        self.assertNotIn('stale NAME', cached.index)               # not a current name
        self.assertNotIn('ada LOVELACE', cached.index)             # not cached: inferred

    def test_params_guard(self):
        stages.check_layout(self.out, N_BUCKETS, self.N_SHARDS, None)
        stages.check_layout(self.out, N_BUCKETS, self.N_SHARDS, None)         # same parameters: fine
        with self.assertRaises(ValueError):
            stages.check_layout(self.out, N_BUCKETS, self.N_SHARDS + 1, None)


class TestGenderScheme(unittest.TestCase):
    def label(self, p, *args):
        return duckdb.connect().execute(f"SELECT {gender_label_sql(*args)} FROM (SELECT ?::DOUBLE AS p_female)", [p]).fetchone()[0]

    def test_three_classes_default(self):
        self.assertEqual([self.label(p) for p in [0.05, 0.1, 0.5, 0.89, 0.9, 1.0, None]],
                         ['male', 'neutral', 'neutral', 'neutral', 'female', 'female', 'unknown'])

    def test_any_number_of_classes(self):
        five = ([0.1, 0.4, 0.6, 0.9], ['male', 'mostly male', 'neutral', 'mostly female', 'female'])
        self.assertEqual([self.label(p, *five) for p in [0.0, 0.3, 0.5, 0.7, 0.95]], five[1])
        self.assertEqual(self.label(0.7, [0.5], ['m', 'f']), 'f')

    def test_invalid_schemes(self):
        for edges, labels in [([0.9, 0.1], ['a', 'b', 'c']), ([0.1, 0.9], ['a', 'b']), ([0, 0.5], ['a', 'b', 'c']),
                              ([0.5], ['unknown', 'x']), ([0.5], ['x', 'x'])]:
            with self.assertRaises(ValueError):
                gender_label_sql(edges, labels)


class TestNameHelpers(unittest.TestCase):
    def test_first_name(self):
        self.assertEqual(first_name_of('J. Michael Smith'), 'Michael')
        self.assertEqual(first_name_of('Smith, John'), 'John')
        self.assertEqual(first_name_of('Jean-Pierre Dupont'), 'Jean-Pierre')
        self.assertIsNone(first_name_of('H. W. Raadsma'))
        self.assertIsNone(first_name_of(''))

    def test_racebert_name(self):
        self.assertEqual(racebert_name('Maria García'), 'maria GARCÍA')
        self.assertEqual(racebert_name('Smith, John'), 'john SMITH')
        self.assertEqual(racebert_name('김민수'), 'minsu GIM')
        self.assertEqual(racebert_name('Караваев Алексей'), 'karavaev ALEKSEI')
        self.assertIsNone(racebert_name(' '))
        self.assertIsNone(racebert_name(','))

    def test_names_are_cleaned_first(self):
        """The shared text clean-up (utils.text.clean_text) runs before the demographic name helpers and the name keys."""
        self.assertEqual(first_name_of('Tarcí\xadsio Gonçalves de Brito'), 'Tarcísio')       # soft hyphen
        self.assertEqual(racebert_name('Claudia Y. Wilson-GarcÃ\xada'), 'claudia y. WILSON-GARCÍA')   # mojibake
        self.assertEqual(racebert_name('Richard Smith\ufeff'), 'richard SMITH')
        self.assertEqual(name_keys('Jin-\xadJu Jung'), ('jin ju jung', 'jin jung'))


if __name__ == '__main__':
    unittest.main()
