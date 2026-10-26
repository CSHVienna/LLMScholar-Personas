"""Tests for scripts/2_graph/1_author_stats.py on a tiny snapshot shaped like
OpenAlex 2026-09 (authors referred to by number in works.authorships).

Run from code/ with PYTHONPATH=.:
    python -m pytest tests/test_author_stats.py -v
"""

import importlib.util
from pathlib import Path

import duckdb
import pandas as pd

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "2_graph" / "1_author_stats.py"
_spec = importlib.util.spec_from_file_location("author_stats", _SCRIPT)
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)


def make_snapshot(path):
    """Authors 1 and 2. Paper 10: both (5 citations); paper 11: author 1 (0);
    paper 12: author 2 listed twice (12 citations)."""
    con = duckdb.connect(str(path))
    con.execute("""CREATE TABLE works (id BIGINT, cited_by_count INTEGER,
                   authorships STRUCT(author STRUCT(id BIGINT))[])""")
    con.execute("""INSERT INTO works VALUES
        (10, 5,  [{'author': {'id': 1}}, {'author': {'id': 2}}]),
        (11, 0,  [{'author': {'id': 1}}]),
        (12, 12, [{'author': {'id': 2}}, {'author': {'id': 2}}])""")
    con.execute("""CREATE TABLE authors (id BIGINT, works_count INTEGER, cited_by_count INTEGER,
                   updated_date DATE,
                   summary_stats STRUCT("2yr_mean_citedness" DOUBLE, h_index INTEGER, i10_index INTEGER),
                   counts_by_year STRUCT("year" INTEGER, works_count INTEGER)[])""")
    con.execute("""INSERT INTO authors VALUES
        (1, 2, 5,  '2026-09-01', {'2yr_mean_citedness': 1.5, 'h_index': 1, 'i10_index': 0}, []),
        (2, 2, 17, '2026-09-01', {'2yr_mean_citedness': 3.0, 'h_index': 2, 'i10_index': 1}, [])""")
    con.close()


def authors_file(path):
    pd.DataFrame({"author_id": ["A1", "A2"],
                  "id": ["https://openalex.org/A1", "https://openalex.org/A2"]}).to_parquet(path)


def test_the_three_stages_on_numeric_author_ids(tmp_path, monkeypatch):
    db, authors = tmp_path / "oa.duckdb", tmp_path / "authors.parquet"
    make_snapshot(db)
    authors_file(authors)
    monkeypatch.setattr(st, "N_CHUNKS", 2)

    st.stage_summary_stats(duckdb.connect(), db, authors, tmp_path / "summary_stats.parquet")
    st.stage_author_work_citations(duckdb.connect(), db, authors, tmp_path / "awc.parquet")
    st.stage_eindex(duckdb.connect(), tmp_path / "awc.parquet", tmp_path / "eindex.parquet")

    summary = pd.read_parquet(tmp_path / "summary_stats.parquet").set_index("id")
    assert summary.loc["https://openalex.org/A2", "h_index"] == 2
    assert summary.loc["https://openalex.org/A1", "two_year_mean_citedness"] == 1.5

    pairs = pd.read_parquet(tmp_path / "awc.parquet")
    got = set(map(tuple, pairs[["author_url", "work_id", "c"]].itertuples(index=False)))
    assert got == {("https://openalex.org/A1", 10, 5), ("https://openalex.org/A1", 11, 0),
                   ("https://openalex.org/A2", 10, 5), ("https://openalex.org/A2", 12, 12)}
    # The folder of chunks is removed once the table is written.
    assert not list(tmp_path.glob("awc_parts_*"))

    eindex = pd.read_parquet(tmp_path / "eindex.parquet").set_index("author_url")
    assert eindex.loc["https://openalex.org/A1", "n_cited"] == 1      # the uncited paper does not count
    assert eindex.loc["https://openalex.org/A2", "h_index_calc"] == 2
