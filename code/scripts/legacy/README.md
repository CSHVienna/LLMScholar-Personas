# scripts/legacy

Scripts that are not part of the pipeline any more. Kept for reference only.

| Script | Why it is here |
|---|---|
| `build_valid_calls.py`, `test_valid_calls_metrics.py` | The old per-call metrics table; replaced by `5_validity/` and `8_metrics/` (`8_metrics/3_calls_table.py` writes the table the notebooks read) |
| `factuality_field_check.py`, `factuality_seniority.py` | Old field and seniority checks; replaced by `factuality/` |
| `resolve_homonyms.py` | Old way of choosing among researchers with the same name; replaced by `factuality/` |
| `test_resolve_homonyms.py` | Its tests (`python -m pytest scripts/legacy/`) |
| `run_similarity.py`, `merge_structural_metrics.py` | One-time recomputation of similarity variants (`oa`, `oa_ss`, `oa_inter`) |
| `export_coauthor_links.py` | One-time export of the coauthorship graph to `results/coauthorship/` |
| `factuality_seniority_oa.py` | One-time analysis: seniority with OpenAlex career ages |
| `2_doi_match.py` | OpenAlex author through the papers by last name + most shared papers; replaced by `1_ground_truth/2_jw_match.py` (Jaro-Winkler), 2026-10-23 |
| `ss_metadata.py` | `6_factuality/5_ss_metadata.py`: papers, citations and h-index of the picked researcher from the Semantic Scholar API (ss_paper_count, ss_citation_count, ss_h_index); no metric used them, the pipeline keeps only OpenAlex, 2026-10-27 |
