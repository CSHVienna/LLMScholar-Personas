"""Helpers for the manual factuality-validation notebook.

The notebook used to define ~20 functions inline (load LLM responses, look the
recommended person up in Semantic Scholar / OpenAlex, print per-row context for
the human reviewer). They are extracted here so the notebook stays plotting /
orchestration only.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd


# ── Pure utilities ────────────────────────────────────────────────────────────


def norm(s: Any) -> str:
    """Lower-case + strip accents (NFD)."""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()


def norm_eq(a: Any, b: Any) -> bool:
    return norm(a) == norm(b) if a and b else False


def truthy(v: Any) -> bool:
    """Truthy in the sense of a manual annotator: 'yes', 'y', 'true', '1', 't'."""
    return str(v).strip().lower() in ("yes", "y", "true", "1", "t")


def has_text(s: Any):
    """True if `s` is non-empty, non-NaN text.

    Works on either a scalar (returns ``bool``) or a pandas ``Series``
    (returns a boolean ``Series`` aligned with ``s``).
    """
    if isinstance(s, pd.Series):
        return s.fillna("").astype(str).str.strip() != ""
    return isinstance(s, str) and bool(s.strip())


def search_urls(name: str, lastname: str) -> dict[str, str]:
    """Return ``{ss_search_url, oa_search_url}`` for the manual reviewer."""
    q = quote(f"{name} {lastname}".strip())
    return {
        "ss_search_url": f"https://www.semanticscholar.org/search?q={q}",
        "oa_search_url": f"https://api.openalex.org/authors?search={q}",
    }


def lookup_affiliation_row(
    aff_table: pd.DataFrame | None,
    row: pd.Series,
    name: str,
    lastname: str,
) -> dict:
    """Return the 4 ``*_auto`` affiliation columns for one persona.

    ``aff_table`` is the loaded ``factuality_affiliation.csv``; ``row`` carries
    the join keys (model, language, run_id, role, task, location, field,
    subfield). Returns ``{}`` if no match or if ``aff_table`` is ``None``.
    """
    if aff_table is None:
        return {}
    sub = aff_table[
        (aff_table["model"] == row["model"])
        & (aff_table["language"] == row["language"])
        & (aff_table["run_id"] == row["run_id"])
        & (aff_table["name"] == name)
        & (aff_table["lastname"] == lastname)
        & (aff_table["role"] == row["role"])
        & (aff_table["task"] == row["task"])
        & (aff_table["location"] == row["location"])
        & (aff_table["field"] == row["field"])
        & (aff_table["subfield"] == row["subfield"])
    ]
    if len(sub) == 0:
        return {}
    r = sub.iloc[0]
    return {
        "affiliation_status_auto": r.get("affiliation_status"),
        "affiliation_best_match_score": r.get("affiliation_best_match_score"),
        "affiliation_best_match_oa": r.get("affiliation_best_match_oa"),
        "affiliation_oa_all": r.get("affiliation_oa_all"),
    }


# ── LLM-response parsing ──────────────────────────────────────────────────────


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
    return s


def _coerce_list(obj: Any) -> list[dict]:
    """Normalise the LLM payload into a list of persona dicts.

    Accepts ``[persona, …]``, ``{"candidates": [...]}``, ``{"recommendations":
    [...]}`` (and a few other plural keys), or a single persona dict.
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("candidates", "recommendations", "persons", "people", "results", "items"):
            if isinstance(obj.get(key), list):
                return obj[key]
        if "name" in obj or "lastname" in obj:
            return [obj]
    return []


def _extract_text(r: dict) -> str | None:
    """Pull the LLM textual content out of a response entry.

    Handles Gemini (``response.candidates[].content.parts[].text``),
    OpenAI batch (``response.body.choices[].message.content``),
    and Ollama (top-level ``message.content``).
    """
    resp = r.get("response") if isinstance(r.get("response"), dict) else None
    if resp:
        cands = resp.get("candidates")
        if cands:
            parts = cands[0].get("content", {}).get("parts", [])
            if parts and parts[0].get("text"):
                return parts[0]["text"]
        body = resp.get("body") if isinstance(resp.get("body"), dict) else None
        choices = (body or resp).get("choices") if isinstance(body or resp, dict) else None
        if choices:
            msg = choices[0].get("message", {})
            if msg.get("content"):
                return msg["content"]
    msg = r.get("message") if isinstance(r.get("message"), dict) else None
    if msg and msg.get("content"):
        return msg["content"]
    return None


def extract_recommendations(req_obj: dict, run_id: int) -> list[dict]:
    """Decode the LLM response for ``run_id`` (1-indexed) into a list of personas."""
    responses = req_obj.get("responses", [])
    if not responses:
        return []
    idx = max(0, min(int(run_id) - 1, len(responses) - 1))
    txt = _extract_text(responses[idx])
    if not txt:
        return []
    try:
        return _coerce_list(json.loads(_strip_code_fence(txt)))
    except json.JSONDecodeError:
        return []


def find_request(
    responses_dir: Path, summary_row: pd.Series
) -> tuple[Path | None, str | None, dict | None]:
    """Walk ``results_*_{language}/`` and return the JSON entry that matches
    every persona variable in ``summary_row``.

    Returns ``(json_path, json_key, request_obj)``, or ``(None, None, None)`` if
    no match is found.
    """
    target = {
        "model": str(summary_row["model"]),
        "role": str(summary_row["role"]),
        "task": str(summary_row["task"]),
        "location": str(summary_row["location"]),
        "k": int(summary_row["k"]),
        "f_target": str(summary_row["target"]),
        "field": str(summary_row["field"]),
        "subfield": str(summary_row["subfield"]),
    }
    language = summary_row["language"]
    candidate_dirs = [
        d
        for d in Path(responses_dir).iterdir()
        if d.is_dir() and d.name.endswith(f"_{language}")
    ]
    for d in candidate_dirs:
        for jpath in sorted(d.glob("*.json")):
            with open(jpath) as f:
                data = json.load(f)
            # Pre-filter: only inspect files whose first entry matches the model.
            sample_obj = next(iter(data.values()), {})
            if str(sample_obj.get("model", "")) != target["model"]:
                continue
            for key, obj in data.items():
                pc = obj.get("parameters", {}).get("persona_context", {})
                ur = obj.get("parameters", {}).get("user_request", {})
                if (
                    str(pc.get("role", "")) == target["role"]
                    and str(pc.get("task", "")) == target["task"]
                    and str(pc.get("location", "")) == target["location"]
                    and int(ur.get("k", -1)) == target["k"]
                    and str(ur.get("target", "")) == target["f_target"]
                    and str(ur.get("field", "")) == target["field"]
                    and str(ur.get("subfield", "")) == target["subfield"]
                ):
                    return jpath, key, obj
    return None, None, None


# ── Stateful validator (loads heavy datasets once) ────────────────────────────


class ManualValidator:
    """Side-by-side validator for the manual factuality notebook.

    Loads Semantic Scholar (parquet), OpenAlex (DuckDB), and
    ``factuality_full.csv`` once, then exposes the lookups the human
    reviewer needs for each persona row:

      - ``find_ss(name, lastname)`` — substring search in SS.
      - ``find_oa(name, lastname)`` — author search in OA (ordered by citations).
      - ``oa_institutions(oa_id)`` — institution history from the pre-aggregated
        chunks.
      - ``oa_author_topics(oa_id)`` — top OA topics for an author.
      - ``lookup_pipeline(name, lastname)`` — what each factuality step decided.
      - ``validate(name, lastname)`` — print SS + OA + pipeline.
      - ``validate_row(row)`` — full context (prompt, LLM output, pipeline
        verdict, SS, OA, suggestions) for one row of the manual CSV.
    """

    def __init__(
        self,
        ss_parquet: str | Path,
        oa_duckdb: str | Path,
        oa_works_tmp_dir: str | Path,
        factuality_full: str | Path,
    ):
        import duckdb

        self._df_ss = pd.read_parquet(ss_parquet)
        self._oa_con = duckdb.connect(str(oa_duckdb), read_only=True)
        self._df_full = pd.read_csv(factuality_full, low_memory=False)

        self._ss_full_col = "Name" if "Name" in self._df_ss.columns else None
        self._ss_field_col = "Field" if "Field" in self._df_ss.columns else None
        if self._ss_full_col:
            self._ss_name_norm = self._df_ss[self._ss_full_col].astype(str).map(norm)
        else:
            self._ss_name_norm = pd.Series(dtype=str)

        self._works_agg_glob = str(
            Path(oa_works_tmp_dir) / "oa_works_agg_chunks" / "chunk_*.parquet"
        )

    # ── lookups ──────────────────────────────────────────────────────────────

    def find_ss(self, name: str, lastname: str, limit: int = 20) -> pd.DataFrame:
        """Case- and accent-insensitive substring search over SS author names."""
        if self._ss_full_col is None:
            return pd.DataFrame()
        n = norm(name)
        ln = norm(lastname)
        mask = self._ss_name_norm.str.contains(
            re.escape(n), na=False
        ) & self._ss_name_norm.str.contains(re.escape(ln), na=False)
        cols = [
            c
            for c in (
                self._ss_full_col,
                self._ss_field_col,
                "Researcher_id",
                "Combined_gender",
                "Citations",
                "Productivity",
                "First_year",
                "Last_year",
            )
            if c and c in self._df_ss.columns
        ]
        return self._df_ss.loc[mask, cols].head(limit)

    def find_oa(self, name: str, lastname: str, limit: int = 10) -> pd.DataFrame:
        """Search OA authors by name + lastname (display_name + alternatives)."""
        n = name.replace("'", "''").lower()
        ln = lastname.replace("'", "''").lower()
        q = f"""
            WITH filtered AS (
              SELECT id, display_name, works_count, cited_by_count, last_known_institution,
                     list_filter(list_concat([display_name], COALESCE(display_name_alternatives, [])),
                                 x -> LOWER(x) LIKE '%{n}%' AND LOWER(x) LIKE '%{ln}%') AS matches
              FROM authors
            )
            SELECT id, display_name, works_count, cited_by_count, last_known_institution,
                   matches[1] AS matched_via
            FROM filtered WHERE len(matches) > 0
            ORDER BY cited_by_count DESC NULLS LAST
            LIMIT {limit}
        """
        try:
            return self._oa_con.execute(q).fetchdf()
        except Exception as exc:
            print(f"OA query failed: {exc}")
            return pd.DataFrame()

    def oa_institutions(self, oa_id: str) -> pd.DataFrame:
        """Institution history for an author, from the pre-aggregated chunks."""
        q = f"""
            SELECT inst_name AS institution, country,
                   MIN(last_year) AS first_year, MAX(last_year) AS last_year
              FROM read_parquet('{self._works_agg_glob}')
             WHERE oa_id = '{oa_id}'
               AND inst_name IS NOT NULL
             GROUP BY inst_name, country
             ORDER BY last_year DESC
        """
        try:
            return self._oa_con.execute(q).fetchdf()
        except Exception as exc:
            print(f"OA institutions query failed: {exc}")
            return pd.DataFrame()

    def oa_author_topics(self, oa_id: str, limit: int = 5) -> pd.DataFrame:
        """Top OA topics for an author. Returns empty if the column doesn't exist."""
        for col_path in ("UNNEST(a.topics) AS u(t)", "UNNEST(a.x_concepts) AS u(t)"):
            q = f"""
                SELECT t.display_name AS topic,
                       t.field.display_name AS field,
                       t.subfield.display_name AS subfield,
                       t.count
                  FROM authors AS a, {col_path}
                 WHERE a.id = '{oa_id}'
                 ORDER BY t.count DESC
                 LIMIT {limit}
            """
            try:
                df = self._oa_con.execute(q).fetchdf()
                if len(df):
                    return df
            except Exception:
                continue
        return pd.DataFrame()

    def lookup_pipeline(self, name: str, lastname: str) -> pd.DataFrame:
        """What every factuality step said about this (name, lastname)."""
        q = self._df_full[
            (self._df_full["name"].astype(str).str.lower() == name.lower())
            & (self._df_full["lastname"].astype(str).str.lower() == lastname.lower())
        ]
        cols = [
            c
            for c in (
                "model", "language", "role", "task", "location", "field", "subfield",
                "author_status", "field_status", "seniority_status", "location_status",
                "affiliation_status", "oa_id", "oa_country_code", "oa_last_institution",
            )
            if c in q.columns
        ]
        return q[cols].drop_duplicates().head(20)

    # ── printers (notebook UX) ───────────────────────────────────────────────

    def validate(self, name: str, lastname: str) -> None:
        """Print SS + OA + pipeline for one persona."""
        from IPython.display import display

        print(f"═══ {name} {lastname} ═══")
        print("\n— Semantic Scholar (parquet):")
        ss = self.find_ss(name, lastname)
        display(ss) if len(ss) else print("  (no matches)")
        print("\n— OpenAlex (DuckDB, ordered by citations):")
        oa = self.find_oa(name, lastname)
        if len(oa):
            display(oa)
            top_id = oa.iloc[0]["id"]
            print(f"\n— OA institution history for top hit ({top_id}):")
            display(self.oa_institutions(top_id))
        else:
            print("  (no matches)")
        print("\n— Pipeline (factuality_full.csv):")
        pp = self.lookup_pipeline(name, lastname)
        display(pp) if len(pp) else print("  (not in factuality_full)")

    def validate_row(self, row: pd.Series) -> None:
        """Print full context + suggestions for one row of the manual CSV."""
        print("═" * 70)
        print(f"REQUEST {row['request_id']} — {row['name']} {row['lastname']}")
        print("═" * 70)

        print("\n┌─ PROMPT sent to LLM ─")
        print(f"│  model       = {row['model']}  ({row['language']})")
        print(f"│  role        = {row['role']}")
        print(f"│  task        = {row['task']}     location = {row['location']}")
        print(f"│  field       = {row['field']}")
        print(f"│  subfield    = {row['subfield']}")

        print("\n┌─ LLM said ─")
        print(f"│  name             = {row['name']} {row['lastname']}")
        print(f"│  current_affil.   = {row['current_affiliations']}")
        print(f"│  areas_of_work    = {row['areas_of_research_or_work']}")
        reason_snippet = str(row.get("reason", ""))[:120]
        print(f"│  reason snippet   = {reason_snippet}…")

        print("\n┌─ PIPELINE automatic verdict ─")
        for c in (
            "author_status_auto", "oa_status_auto", "field_status_auto",
            "seniority_status_auto", "location_status_auto", "affiliation_status_auto",
            "affiliation_best_match_score", "affiliation_best_match_oa",
        ):
            if c in row.index and pd.notna(row[c]) and row[c] != "":
                print(f"│  {c:32s} = {row[c]}")

        print("\n┌─ PIPELINE matches (SS + OA, what the pipeline found) ─")
        if has_text(row.get("matched_name")):
            print(
                f"│  SS matched_name    = {row['matched_name']}  "
                f"(researcher_id={row.get('researcher_id')}, "
                f"score={row.get('match_score')}, gt_field={row.get('gt_field')})"
            )
        else:
            print("│  SS                  → no match in SS parquet")
        if has_text(row.get("oa_id")):
            score = row.get("oa_match_score")
            kind = (
                "exact" if pd.notna(score) and float(score) == 1.0
                else f"fuzzy ({score})"
            )
            print(
                f"│  OA display_name    = {row['oa_display_name']}  "
                f"(oa_id={row['oa_id']}, {kind})"
            )
        else:
            print("│  OA                  → no match in OpenAlex")

        print("\n┌─ Semantic Scholar (parquet) ─")
        ss = self.find_ss(row["name"], row["lastname"])
        if len(ss):
            print(ss.to_string(index=False))
        else:
            print("  (no matches — try removing initials or particles)")

        print("\n┌─ OpenAlex (DuckDB, other similar ones, top 5 by citations) ─")
        oa = self.find_oa(row["name"], row["lastname"], limit=5)
        if len(oa):
            print(oa.to_string(index=False))
            top_id = oa.iloc[0]["id"]
            print(f"\n  Institutions for top hit ({top_id}):")
            inst = self.oa_institutions(top_id)
            if len(inst):
                print("  " + inst.to_string(index=False).replace("\n", "\n  "))
            else:
                print("  (no institutions)")
            print("\n  Topics/Fields for top hit:")
            topics = self.oa_author_topics(top_id)
            if len(topics):
                print("  " + topics.to_string(index=False).replace("\n", "\n  "))
            else:
                print("  (no topics — or the column does not exist in this dump)")
        else:
            print("  (no matches)")

        print("\n┌─ SUGGESTIONS for filling the CSV ─")
        print(
            "│  found_in_ss_manual         → 'yes' if you see a clear match above, "
            "'no' if the list is empty or all are a different person"
        )
        print("│  found_in_oa_manual         → same for OA")
        print(
            "│  field_correct_manual       → compare requested field "
            f"({row['field']!r}) with SS's Field or OA's Topics"
        )
        print(
            "│  affiliation_correct_manual → does the LLM's affiliation "
            f"({row['current_affiliations']}) appear in the OA institution list above?"
        )
        print("│  notes_manual               → free text (e.g. 'same name but different field')")
        print()


# ── Concordance helpers (used by the per-row report cell) ────────────────────


def manual_found_any_row(r: pd.Series) -> bool | None:
    """``True`` if found in SS or OA; ``False`` if 'no' in both; ``None`` if both empty."""
    ss = r.get("found_in_ss_manual", "")
    oa = r.get("found_in_oa_manual", "")
    if not ss and not oa:
        return None
    return truthy(ss) or truthy(oa)


def auto_found_row(r: pd.Series) -> bool | None:
    """``auto_found = author_status_auto == 'found' OR oa_status_auto == 'found'``."""
    ss = r.get("author_status_auto", "")
    oa = r.get("oa_status_auto", "")
    if not ss and not oa:
        return None
    return (ss == "found") or (oa == "found")
