# Persona Prompting Effects in LLM-Based Scholar Recommendation

> Auditing how the **persona** in a prompt shapes which researchers an LLM recommends.

When you ask an LLM for "5 senior Physics professors", does the answer change if you ask **in Spanish from Ecuador** vs **in English from Germany**? Does it change if the persona is a **PhD student looking for an advisor** vs a **recruiter scouting hires**? **LLMScholar-Personas** runs the same prompt across every combination of those three persona variables — **language**, **country**, **role** — and measures what shifts in the recommendations:

- **Factuality** — do the recommended people actually exist? Are their field, seniority, and country correct?
- **Diversity** — gender, ethnicity, country, productivity tier of the recommended set.
- **Parity** — distance from the Semantic Scholar / OpenAlex ground-truth distribution.
- **Consistency** — overlap between repeated runs of the same prompt.

The result is one CSV per LLM call with every metric attached, plus a notebook that turns that table into the figures used in the paper.

---

## Quick start

```bash
git clone <repo-url>
cd LLMScholar-Personas

# 1. Install dependencies (Python 3.10+).
pip install -r requirements.txt

# 2. Point the project at your local data.
cp config.ini.example config.ini
$EDITOR config.ini    # fill in [secrets].keys_dir and every [data].* path

# 3. Run the pipeline from code/.
cd code
export PYTHONPATH=.
python scripts/run_all.py      # ground truth → graph → validity → factuality → enrichment → metrics → figure data
jupyter nbconvert --execute notebooks/analysis/figures/metrics.ipynb
```

The end-to-end pipeline (collect responses → factuality → metrics → figures) lives in **[`code/README.md`](code/README.md)**.

---

## Repository structure

```
LLMScholar-Personas/
├── code/                 see code/README.md
│   ├── libs/             Shared code (metrics, plots, prompts, utils; legacy/ = unused).
│   ├── scripts/          Programs you run, folders numbered in the order they run (see code/scripts/README.md).
│   ├── notebooks/        Plot-only notebooks (analysis/, agreement/).
│   └── tests/            pytest.
├── data/                 see data/README.md
│   ├── context/          Per-language prompt scaffolding.
│   ├── models/           List of LLMs and their metadata.
│   └── annotator_agreement/, ethnicity_inference/   Manual labels.
├── results/              Pipeline outputs. Path configurable in config.ini.
├── requirements.txt
└── config.ini.example    Template — copy to config.ini and fill in your paths.
```

Detailed instructions live in the sub-READMEs:

- **[`code/README.md`](code/README.md)** — end-to-end pipeline (data flow, every step, expected outputs, troubleshooting).
- **[`code/scripts/README.md`](code/scripts/README.md)** — what each script folder is for, and the details of each script.
- **[`data/README.md`](data/README.md)** — what lives in each subfolder of `data/`.

---

## How the pipeline fits together

```
scripts/run_all.py
─────────────────────────────────────────────────────────────────────────────────────────────
1_ground_truth/   every Semantic Scholar researcher + their OpenAlex author (through their
                  papers, Jaro-Winkler; by name otherwise) and its metadata → ground_truth.parquet
2_graph/          author stats, coauthorship graph and similarity PCA over the ground truth
3_prompting/ + 4_llmcaller/  prompts and LLM answers → recommendations.csv   (run by hand)
5_validity/       validity, refusal, duplicates; consistency          (every call)
6_factuality/     the factuality cascade on Semantic Scholar          (valid calls)
                  the picked researcher brings their ground-truth data
7_enrichment/     LLM answer vs that data: country, affiliation; ethnicity
8_metrics/        factuality, socials (factuality_author = True) → 8_metrics/calls.csv
9_plot_data/      data for specific figures
                                                                  →  notebooks/analysis/
```

A funnel: each stage only passes on what survived the previous one. See `code/README.md` for the ordered command list, the schema of every intermediate CSV, and the optional branches (manual annotation, ethnicity inference, inter-annotator agreement).

---

## Related work

This codebase extends the auditor framework introduced in *Whose Name Comes Up? II: Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation* — [arXiv:2602.08873](https://arxiv.org/abs/2602.08873) — with the persona-prompting axis (language × country × role).

---

## Status

Research code. APIs and intermediate CSV schemas may change between paper revisions.
