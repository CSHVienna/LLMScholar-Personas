"""
Unit tests for author matching (B.6a candidates -> B.6b decision) on a tiny synthetic OpenAlex.

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
import pandas as pd                                                      # noqa: E402

from matching import candidates as mc                                  # noqa: E402
from matching import stats                                               # noqa: E402
from matching.decide import REC_ID_COLUMNS, write_matches           # noqa: E402
from groundtruth import db                                               # noqa: E402
from groundtruth.fields import load_field_mapping                        # noqa: E402
from groundtruth.names import write_name_index                           # noqa: E402
from utils import constants as cons                                      # noqa: E402

N_BUCKETS, N_SHARDS = 2, 3
CS, PHYS, BIO, SOC, PSY = 'Computer Science', 'Physics', 'Biology', 'Sociology', 'Psychology'

# author_id: (display name, alternative names, field of all works, first publication year); None field = no A.1 row
OPENALEX = {
    1: ('John Smith', [], CS, 2020),                  # junior
    2: ('John Smith', [], PHYS, 1995),                # senior
    3: ('Jon Smith', [], CS, 1990),                   # senior
    4: ('Wei Zhang', ['Zhang Wei'], CS, 2019),
    5: ('María García-López', [], BIO, 1990),
    7: ('Ludmila Petrova', [], SOC, 2021),
    8: ('Oak Ridge National Lab (ORNL)', [], CS, 1990),          # organisation: never a candidate
    9: ("Kevin O'Connell", [], PSY, 1985),
    11: ('Hiroshi Yamamoto', [], PHYS, 2018),
    12: ('Anna Weber', ['A. Weber'], CS, None),     # no counted works: a name match only
    13: ('Alexander Kellner', [], CS, 2000),           # shares only the long token 'alexander' with rec 13
}

# (rec_index, first name, last name, field, seniority)
RECS = [
    (1, 'John', 'Smith', CS, 'junior'),          # exact: {1, 2}; field {1}, junior {1}         -> author 1
    (2, 'John', 'Smith', PHYS, 'junior'),        # field {2}, junior {1}, author {}             -> ambiguous
    (3, 'John', 'Smith', BIO, 'senior'),         # field {}, senior {2}                         -> seniority only: 2
    (4, 'Zhang', 'Wei', CS, 'junior'),           # order-free key                               -> author 4
    (5, 'J.', 'Smith', CS, 'senior'),            # initials: {1, 2, 3}; field {1, 3}, senior {2, 3} -> author 3
    (6, 'Ludmilla', 'Petrova', SOC, 'junior'),   # fuzzy                                        -> author 7
    (7, 'Yamamotto', 'Hiroshi', PHYS, 'junior'), # fuzzy, flipped                               -> author 11
    (8, 'Fake', 'Person', CS, 'junior'),         # nothing
    (9, 'Kevin', "O'Connell", PSY, 'senior'),    # exact                                        -> author 9
    (10, 'Maria', 'Garcia Lopez', BIO, 'senior'),  # accents / hyphen: exact                    -> author 5
    (11, 'Anna', 'Weber', CS, 'junior'),         # exact, but no A.1 row: name only
    (12, 'A.', 'Weber', CS, 'junior'),           # exact via the alternative name 'A. Weber'
    (13, 'Neville', 'Alexander', CS, 'senior'),  # flipped jw_full ~0.94 with 'Alexander Kellner', parts far apart
]


def make_openalex(path: Path) -> None:
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE authors (id BIGINT, display_name VARCHAR, display_name_alternatives VARCHAR[], orcid VARCHAR)")
    for author_id, (name, alternatives, _, _) in OPENALEX.items():
        con.execute("INSERT INTO authors VALUES (?, ?, ?, NULL)", [author_id, name, alternatives])
    con.close()


class MatchingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.fields = load_field_mapping()
        make_openalex(root / 'oa.duckdb')
        con = duckdb.connect(str(root / 'oa.duckdb'), read_only=True)
        for b in range(N_BUCKETS):                                                   # A.2
            write_name_index(con, db.bucket_file(root / 'names', b), b, N_BUCKETS)
        con.close()
        rows = []                                                                    # A.1 (only the needed columns)
        for author_id, (_, _, field, year) in OPENALEX.items():
            if year is not None:
                rows.append({'author_id': author_id, 'first_pub_year': year, 'n_works_topic': 4,
                             **{f.column: 4 if f.field == field else 0 for f in cls.fields}})
        authors = pd.DataFrame(rows)
        (root / 'authors').mkdir()
        for b in range(N_BUCKETS):
            authors[authors.author_id % N_BUCKETS == b].to_parquet(db.bucket_file(root / 'authors', b), index=False)
        base = {c: 'x' for c in REC_ID_COLUMNS}
        recs = pd.DataFrame([{**base, cons.COL_MODEL: 'm1', cons.COL_RUN_ID: 1, cons.COL_REC_INDEX: i,
                              'field': f, 'seniority': s, 'first_name': fn, 'last_name': ln}
                             for i, fn, ln, f, s in RECS])
        (root / 'recs').mkdir()
        recs.to_parquet(root / 'recs' / 'm1.parquet', index=False)

        cls.cand = root / 'candidates'
        con = duckdb.connect()
        cls.rec_names = mc.write_rec_names(con, str(root / 'recs' / '*.parquet'), cls.cand / mc.REC_NAMES_FILE, N_SHARDS)
        for b in range(N_BUCKETS):
            mc.write_exact_bucket(con, cls.cand / mc.REC_NAMES_FILE, db.bucket_file(root / 'names', b),
                                  db.bucket_file(cls.cand / mc.EXACT_DIR, b))
        cls.n_fuzzy_recs = mc.write_fuzzy_recs(con, cls.cand / mc.REC_NAMES_FILE, cls.cand / mc.EXACT_DIR,
                                               cls.cand / mc.FUZZY_RECS_FILE, fuzzy_all=False)
        for b in range(N_BUCKETS):
            mc.write_oa_tokens_bucket(con, db.bucket_file(root / 'names', b), cls.cand / mc.FUZZY_RECS_FILE,
                                      cls.cand / mc.OA_TOKENS_DIR, b)
        for s in range(N_SHARDS):
            mc.write_fuzzy_shard(con, cls.cand / mc.FUZZY_RECS_FILE, cls.cand / mc.OA_TOKENS_DIR,
                                 mc.shard_file(cls.cand / mc.FUZZY_DIR, s), s, floor=0.90)
        for b in range(N_BUCKETS):
            mc.write_merged_bucket(con, db.bucket_file(cls.cand / mc.EXACT_DIR, b), cls.cand / mc.FUZZY_DIR,
                                   db.bucket_file(root / 'names', b), db.bucket_file(cls.cand, b))
        cls.candidates = pd.concat(pd.read_parquet(db.bucket_file(cls.cand, b)) for b in range(N_BUCKETS))
        cls.kwargs = dict(recommendations_file=root / 'recs' / 'm1.parquet', rec_names_file=cls.cand / mc.REC_NAMES_FILE,
                          candidates_glob=str(cls.cand / 'bucket-*.parquet'),
                          authors_glob=str(root / 'authors' / 'bucket-*.parquet'), fields=cls.fields, seed=42,
                          jw_full_min=0.95)
        write_matches(con, root / 'match' / 'm1.parquet', **cls.kwargs)
        cls.fact = pd.read_parquet(root / 'match' / 'm1.parquet').set_index(cons.COL_REC_INDEX)
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def cands(self, rec_name):
        return self.candidates[self.candidates.rec_name == rec_name].set_index('author_id')

    # --- B.6a ------------------------------------------------------------------------------------------------------
    def test_rec_names(self):
        names = self.rec_names.set_index(['first_name', 'last_name'])['rec_name']
        self.assertEqual(names[('Maria', 'Garcia Lopez')], 'maria|garcia lopez')
        self.assertEqual(self.rec_names.set_index('rec_name').loc['j|smith', 'initials_key'], 'j smith')
        self.assertTrue(pd.isna(self.rec_names.set_index('rec_name').loc['j|smith', 'name_key_short']))   # initials rule only
        self.assertTrue(pd.isna(self.rec_names.set_index('rec_name').loc['john|smith', 'initials_key']))

    def test_exact_rules(self):
        c = self.cands('john|smith')
        self.assertEqual(sorted(c.index), [1, 2])                          # not 3 (Jon), never 8 (organisation)
        self.assertTrue(c['exact_display'].all() and c['short'].all())
        self.assertTrue(self.cands('zhang|wei').loc[4, 'exact_display'])   # any order
        self.assertTrue(self.cands('a|weber').loc[12, 'exact_alternative'])
        self.assertEqual(sorted(self.cands('j|smith').query('initials').index), [1, 2, 3])
        self.assertTrue(self.cands('maria|garcia lopez').loc[5, 'exact_display'])
        self.assertNotIn(8, set(self.candidates.author_id))

    def test_fuzzy_only_for_unmatched_names(self):
        fuzzy = pd.read_parquet(self.cand / mc.FUZZY_RECS_FILE)
        self.assertEqual(sorted(fuzzy.rec_name), ['fake|person', 'ludmilla|petrova', 'neville|alexander', 'yamamotto|hiroshi'])
        c = self.cands('ludmilla|petrova').loc[7]
        self.assertGreater(c['jw_full'], 0.95)
        self.assertFalse(c['jw_flipped'])
        self.assertEqual(c['jw_name'], 'ludmila petrova')                    # the scored OpenAlex name
        self.assertAlmostEqual(c['edit_first'], 1 / 8)                       # 'ludmilla' -> 'ludmila': 1 edit / 8
        y = self.cands('yamamotto|hiroshi').loc[11]
        self.assertTrue(y['jw_flipped'])                                    # 'hiroshi yamamotto' ~ 'hiroshi yamamoto'
        self.assertGreater(y['jw_first'], 0.95)
        self.assertEqual(y['jw_last'], 1.0)

    # --- B.6b ------------------------------------------------------------------------------------------------------
    def row(self, index):
        return self.fact.loc[index]

    def test_levels_and_pick(self):
        r = self.row(1)
        self.assertEqual((r.match_rule, r.n_name, r.n_name_field, r.n_name_seniority, r.n_author), ('exact_display', 2, 1, 1, 1))
        self.assertTrue(r.matched_author and r.author_id == 1)
        self.assertEqual((r.n_rule_exact_display, r.n_rule_short, r.n_rule_initials), (2, 2, 0))

    def test_partial_author_matching(self):
        r = self.row(2)                                          # field {2}, junior {1}: ambiguous
        self.assertFalse(r.matched_author or r.matched_author_field or r.matched_author_seniority)
        self.assertTrue(pd.isna(r.author_id_field) and pd.isna(r.author_id_seniority))
        r = self.row(3)                                          # field {}, senior {2}
        self.assertTrue(r.matched_author_seniority and r.author_id_seniority == 2)
        self.assertFalse(r.matched_author or r.matched_author_field)

    def test_rules_order(self):
        self.assertEqual(self.row(4).match_rule, 'exact_display')
        r = self.row(5)
        self.assertEqual((r.match_rule, r.n_name, r.author_id), ('initials', 3, 3))
        self.assertEqual((self.row(6).match_rule, self.row(6).author_id), ('fuzzy', 7))
        self.assertEqual((self.row(7).match_rule, self.row(7).author_id), ('fuzzy', 11))
        self.assertEqual((self.row(9).author_id, self.row(10).author_id), (9, 5))
        self.assertEqual(self.row(12).match_rule, 'exact_alternative')

    def test_no_match_and_no_works(self):
        r = self.row(8)
        self.assertTrue(pd.isna(r.match_rule))
        self.assertEqual((r.n_name, r.matched_name, r.matched_author), (0, False, False))
        r = self.row(11)                                         # Anna Weber exists, but without counted works
        self.assertEqual((r.matched_name, r.matched_name_field, r.matched_name_seniority), (True, False, False))

    def test_both_parts_must_be_close(self):
        k = self.cands('neville|alexander').loc[13]
        self.assertTrue(k['jw_flipped'] and k['jw_full'] > 0.93 and k['jw_first'] < 0.8)
        self.assertTrue(pd.isna(self.row(13).match_rule))                    # jw_full 0.94 < 0.95
        con = duckdb.connect()
        loose = self.root / 'match_loose' / 'm1.parquet'
        write_matches(con, loose, **{**self.kwargs, 'jw_full_min': 0.90})
        self.assertEqual(pd.read_parquet(loose).set_index(cons.COL_REC_INDEX).loc[13, 'match_rule'], 'fuzzy')
        parts = self.root / 'match_parts' / 'm1.parquet'
        write_matches(con, parts, **{**self.kwargs, 'jw_full_min': 0.90, 'jw_first_min': 0.9, 'jw_last_min': 0.9})
        both = pd.read_parquet(parts).set_index(cons.COL_REC_INDEX)
        self.assertTrue(pd.isna(both.loc[13, 'match_rule']))                 # rejected: 'neville' vs 'kellner'
        self.assertEqual((both.loc[6, 'match_rule'], both.loc[7, 'match_rule']), ('fuzzy', 'fuzzy'))  # typos still match

    def test_stricter_fuzzy_threshold(self):
        con = duckdb.connect()
        write_matches(con, self.root / 'match_strict' / 'm1.parquet', **{**self.kwargs, 'jw_full_min': 0.999})
        strict = pd.read_parquet(self.root / 'match_strict' / 'm1.parquet').set_index(cons.COL_REC_INDEX)
        self.assertFalse(strict.loc[6, 'matched_name'])             # Ludmilla vs Ludmila no longer matches
        self.assertEqual(strict.loc[6, 'n_rule_fuzzy'], 0)

    def test_stats(self):
        df = self.fact.reset_index()
        out = stats.outcome(df).set_axis(df[cons.COL_REC_INDEX])
        self.assertEqual((out[1], out[2], out[3], out[8], out[11]),
                         ('author', 'ambiguous', 'seniority only', 'no match', 'name only'))
        rules = stats.rule(df).set_axis(df[cons.COL_REC_INDEX])
        self.assertEqual((rules[5], rules[6], rules[8]), ('initials', 'fuzzy', 'no match'))
        shares = stats.shares(df, stats.rule(df), stats.RULE_ORDER)
        self.assertAlmostEqual(shares.loc['m1'].sum(), 1.0)
        self.assertAlmostEqual(shares.loc['m1', 'exact_display'], 7 / len(RECS))  # recs 1-4, 9, 10, 11
        self.assertAlmostEqual(stats.level_rates(df).loc['m1', 'matched_author'], df['matched_author'].mean())
        self.assertEqual(stats.tier_overlap(df).loc['fuzzy', 'n_recommendations'], 2)

    def test_pick_is_reproducible_and_order_free(self):
        """A homonym pick depends on the seed and the recommendation only (not on other rows)."""
        con = duckdb.connect()
        recs = pd.read_parquet(self.kwargs['recommendations_file'])
        sub = self.root / 'recs_sub'
        sub.mkdir(exist_ok=True)
        recs[recs.rec_index == 5].to_parquet(sub / 'm1.parquet', index=False)
        write_matches(con, self.root / 'match_sub' / 'm1.parquet', **{**self.kwargs, 'recommendations_file': sub / 'm1.parquet'})
        alone = pd.read_parquet(self.root / 'match_sub' / 'm1.parquet').iloc[0]
        self.assertEqual(alone.author_id, self.row(5).author_id)


if __name__ == '__main__':
    unittest.main()
