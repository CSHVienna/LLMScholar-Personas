# scripts/openalex_cascade

Chooses the OpenAlex author of each recommendation the same way
`scripts/factuality/` chooses the Semantic Scholar one: among the OpenAlex
authors with the recommended name, those of the asked field, then of the asked
seniority, then of the asked subfield.

| step | script | what it does |
|---|---|---|
| 1 | `1_candidates.py` | every OpenAlex author whose name (or one of its alternative names) matches the recommended name |
| 2 | `2_author_fields.py` | each author's field from two published classifications (topics, built on Scopus ASJC; level-0 concepts, inherited from Microsoft Academic Graph) and their top subfields |
| 3 | `3_pick.py` | the cascade and the pick, once per **variant** — the last tie-break among the authors that pass the same checks: `random` or `most_cited` |

`run_pipeline.py` runs the three; `scripts/run_all.py` runs it first. Output in
`../results/summary/openalex_cascade/`: `candidates.parquet`,
`author_fields.parquet`, `picks_random.csv`, `picks_most_cited.csv` (with the
`n_name`, `n_field`, `n_seniority`, `n_author` counts of each check).

An author is of the asked field when their main topics field **or** their main
concept is that field. `fields.py` has both mappings and how they were validated
(5,671 authors identified in both Semantic Scholar and OpenAlex: 72.4% / 77.8%
agreement with Semantic Scholar's field, 83.9% when both agree;
`notebooks/analysis/checks/openalex_field.ipynb`). It needs the
2026-09 snapshot or later: the DuckDB database for topics and the snapshot's
parquet author files for concepts (`x_concepts`, not loaded in the database).
