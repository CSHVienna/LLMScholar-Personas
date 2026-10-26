# scripts/legacy

Scripts that are not part of the pipeline any more. Kept for reference only.

| Script | Why it is here |
|---|---|
| `build_valid_calls.py`, `test_valid_calls_metrics.py` | The old per-call metrics table; replaced by `pipeline/` (step 5 writes the table the notebooks read) |
| `factuality_field_check.py`, `factuality_seniority.py` | Old field and seniority checks; replaced by `factuality/` |
| `resolve_homonyms.py` | Old way of choosing among researchers with the same name; replaced by `factuality/` |
| `test_resolve_homonyms.py` | Its tests (`python -m pytest scripts/legacy/`) |
| `run_similarity.py`, `merge_structural_metrics.py` | One-time recomputation of similarity variants (`oa`, `oa_ss`, `oa_inter`) |
| `export_coauthor_links.py` | One-time export of the coauthorship graph to `results/coauthorship/` |
| `factuality_seniority_oa.py` | One-time analysis: seniority with OpenAlex career ages |
| `2_doi_match.py` | OpenAlex author through the papers by last name + most shared papers; replaced by `5_openalex_cascade/2_jw_match.py` (Jaro-Winkler), 2026-10-23 |
