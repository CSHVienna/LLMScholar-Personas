"""Tests for the homonym tie-break (resolve_homonyms.py).

The matcher picks the first reference row in file order when several people
share a name, which hands every tie to whichever field sits earlier in the
parquet. These tests pin the replacement rule: prefer the homonym whose field is
the field the prompt asked for, and say so in `tiebreak`.

Run from code/ with PYTHONPATH=.:
    python -m pytest tests/test_resolve_homonyms.py -v
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "factuality"
sys.path.insert(0, str(_SCRIPTS))
_spec = importlib.util.spec_from_file_location("rh", _SCRIPTS / "resolve_homonyms.py")
rh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rh)

import factuality_field_check as ffc  # noqa: E402
import factuality_seniority as fs  # noqa: E402


@pytest.fixture
def reference():
    """Three names: one ambiguous across fields, one ambiguous within a field,
    one unique. Row order is the file order the matcher would have followed."""
    return pd.DataFrame(
        {
            "Researcher_id": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Name": ["Ann Lee", "Ann Lee", "Bob Ray", "Bob Ray", "Cy Poe"],
            "Field": ["Biology", "Mathematics", "Biology", "Biology", "Psychology"],
            "Combined_gender": ["female", "female", "male", "male", "male"],
            "First_year": [1995.0, 2018.0, 2000.0, 2001.0, 2015.0],
            "Citations": [500, 10, 300, 200, 50],
        }
    )


@pytest.fixture
def candidates(reference, tmp_path):
    fn = tmp_path / "ref.parquet"
    reference.to_parquet(fn)
    return rh.build_candidates(str(fn))


def _calls(rows):
    """Minimal factuality_full-shaped frame."""
    base = {
        "author_status": "found", "target": "Junior Professor",
        "field": "Mathematics", "matched_name": "Ann Lee", "researcher_id": 1.0,
        "gt_field": "Biology", "gt_gender": "female", "gt_career_age": 30.0,
        "gt_citations": 500,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_tie_goes_to_the_asked_field(candidates):
    out = rh.resolve_chunk(_calls([{}]), *candidates)
    assert out.tiebreak.iloc[0] == "asked_field"
    assert out.gt_field.iloc[0] == "Mathematics"
    assert out.researcher_id.iloc[0] == 2.0
    # Everything copied from the record moves with it.
    assert out.gt_citations.iloc[0] == 10
    assert out.gt_career_age.iloc[0] == rh.REFERENCE_YEAR - 2018


def test_asked_field_absent_leaves_the_file_order_pick(candidates):
    out = rh.resolve_chunk(_calls([{"field": "Sociology"}]), *candidates)
    assert out.tiebreak.iloc[0] == "file_order"
    assert out.gt_field.iloc[0] == "Biology"
    assert out.researcher_id.iloc[0] == 1.0


def test_unique_name_is_untouched(candidates):
    call = _calls([{"matched_name": "Cy Poe", "field": "Mathematics",
                    "gt_field": "Psychology", "researcher_id": 5.0}])
    out = rh.resolve_chunk(call, *candidates)
    assert out.tiebreak.iloc[0] == "unique"
    assert out.gt_field.iloc[0] == "Psychology"
    assert out.homonym_count.iloc[0] == 1


def test_homonyms_within_one_field_are_not_ambiguous(candidates):
    """Two Bob Rays, both biologists — the tie changes nothing we measure."""
    call = _calls([{"matched_name": "Bob Ray", "field": "Biology",
                    "gt_field": "Biology", "researcher_id": 3.0}])
    out = rh.resolve_chunk(call, *candidates)
    assert out.homonym_count.iloc[0] == 2
    assert out.homonym_fields.iloc[0] == 1
    assert out.tiebreak.iloc[0] == "unique"


def test_hallucinated_rows_are_left_alone(candidates):
    call = _calls([{"author_status": "hallucinated", "matched_name": None,
                    "gt_field": None}])
    out = rh.resolve_chunk(call, *candidates)
    assert out.tiebreak.iloc[0] == ""
    assert out.field_status.iloc[0] == rh._FIELD["NOT_APPLICABLE"]


def test_translated_prompt_fields_still_match(candidates):
    """The prompt may name the field in Spanish or German."""
    out = rh.resolve_chunk(_calls([{"field": "Matemáticas"}]), *candidates)
    assert out.tiebreak.iloc[0] == "asked_field"
    assert out.gt_field.iloc[0] == "Mathematics"


def test_field_status_matches_the_original_classifier(candidates):
    calls = _calls([{}, {"field": "Sociology"}, {"author_status": "hallucinated"},
                    {"field": ""}])
    out = rh.resolve_chunk(calls, *candidates)
    # Re-run the original row-wise classifier over the resolved frame.
    expected = pd.DataFrame(
        [ffc.classify_row(r) for _, r in out.iterrows()],
        columns=["field_status", "field_check_source", "field_evidence"],
        index=out.index,
    )
    assert out.field_status.equals(expected.field_status)
    assert out.field_check_source.equals(expected.field_check_source)


def test_seniority_matches_the_original_classifier(candidates):
    calls = _calls([{}, {"target": "Senior Professor"}, {"gt_career_age": None},
                    {"target": "Profesor(a) Júnior"}, {"gt_career_age": 15.0}])
    out = rh.resolve_chunk(calls, *candidates)
    expected = pd.DataFrame([fs.classify_row(r) for _, r in out.iterrows()],
                            index=out.index)
    for col in ("seniority_bucket", "seniority_llm_bucket", "seniority_status",
                "seniority_age_source"):
        assert out[col].fillna("~").equals(expected[col].fillna("~")), col


def test_normalize_series_equals_the_matchers_normaliser():
    from factuality_author_jw import normalize

    names = pd.Series(["Dr. José Pérez-Gómez (UCE)", "Ann-Marie O'Neill",
                       "Prof Müller, H.", "Sto­tny", None, ""])
    assert list(rh.normalize_series(names)) == [normalize(n) if n else "" for n in names]
