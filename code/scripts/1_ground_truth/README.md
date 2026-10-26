# scripts/1_ground_truth

Step 1 of everything, done once before the LLM answers are looked at: every
researcher of the Semantic Scholar parquet (the ground truth) with their
OpenAlex author and its metadata → `results/summary/ground_truth/ground_truth.parquet`.
`6_factuality/` takes the data of the picked researcher from it, and `2_graph/`
builds the coauthorship graph and the similarity PCA over it.

**First through the papers** (the same person, because they signed the same papers), **then by name** for the researchers without a match:

| step | script | what it does |
|---|---|---|
| 1 | `1_ss_papers.py` | the researcher's papers with DOI, from the Semantic Scholar API (batch calls) → `ss_papers.parquet` |
| 2 | `2_jw_match.py` | those papers in OpenAlex; on each paper the author whose name is most similar (Jaro-Winkler on the display name + the longest alternative name, at least 1.35) gets a vote; from the most recent paper, the first author with 2 votes is the one (ties: up to 50 papers) → `jw_match.parquet` |
| — | `run_pipeline.py` | the researchers without a match, with their name, Semantic Scholar field and seniority → `cascade_requests.csv` |
| 3 | `3_candidates.py` | (cascade) every OpenAlex author whose name (or one of its alternative names) matches |
| 4 | `4_author_fields.py` | field from two published classifications (topics, built on Scopus ASJC; level-0 concepts, inherited from Microsoft Academic Graph) and top subfields, of the cascade's candidates and of the authors of step 2 |
| 5 | `5_pick.py` | (cascade) name → field → seniority, then one at random → `picks.csv` |
| 6 | `6_oa_metadata.py` | the metadata of the chosen author (works, citations, h/i10, 2-year citedness, years, field, ORCID; country and institution of the researcher's most recent paper signed by that author, or else OpenAlex's last known institution), joined to the Semantic Scholar parquet → `ground_truth.parquet` |

`run_pipeline.py` runs them (`--ids` for a subset, `--out_dir`, `--from_step`).

- **Threshold of Jaro-Winkler (1.35):** chosen from data in `notebooks/analysis/checks/jw_threshold.ipynb` (labels that do not use names).
- **Field:** an author is of the asked field when their main topics field **or** their main concept is that field (`fields.py`; validation in `notebooks/analysis/checks/openalex_field.ipynb`).
- **The API:** 100 researchers per call with papers (the most it allows); with a key (`[semantic_scholar].s2_api_key` in `config.ini`) calls are spaced to 1 per second. Researchers that Semantic Scholar merged into another profile are not returned by the batch endpoint; following the merge is one call per researcher (`1_ss_papers.py` without `--batch_only`), too slow for the whole ground truth.
