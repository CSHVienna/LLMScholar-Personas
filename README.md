# LLMScholar-Personas

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
python scripts/factuality/run_factuality_pipeline.py    # 7-step factuality cascade
python scripts/metrics/build_valid_calls.py             # per-call metrics table
jupyter nbconvert --execute notebooks/analysis/metrics_pipeline.ipynb
```

The end-to-end pipeline (collect responses → factuality → metrics → figures) lives in **[`code/README.md`](code/README.md)**.

---

## Repository structure

```
LLMScholar-Personas/
├── code/                 see code/README.md
│   ├── libs/             Reusable libraries (LLM clients, prompts, metrics, plots).
│   ├── scripts/          CLI entry points — one folder per pipeline stage.
│   └── notebooks/        Plot-only notebooks (analysis/, agreement/).
├── data/                 see data/README.md
│   ├── context/          Per-language prompt scaffolding.
│   ├── models/           List of LLMs and their metadata.
│   └── annotator_agreement/, ethnicity_inference/   Manual labels.
├── results/              Pipeline outputs. Path configurable in config.ini.
├── requirements.txt
├── config.ini.example    Template — copy to config.ini and fill in your paths.
└── REFACTOR.md           Change log for the latest refactor.
```

Detailed instructions live in the sub-READMEs:

- **[`code/README.md`](code/README.md)** — end-to-end pipeline (data flow, every step, expected outputs, troubleshooting).
- **[`code/scripts/README.md`](code/scripts/README.md)** — per-CLI documentation for the batch-processing scripts.
- **[`data/README.md`](data/README.md)** — what lives in each subfolder of `data/`.

---

## How the pipeline fits together

```
prompts (per language)                LLM responses              factuality + metrics            figures
─────────────────────                ───────────────             ────────────────────            ───────
scripts/prompting/      →  (collected externally)  →  scripts/factuality/         →  notebooks/analysis/
batch_params, batch_prompt,                            run_factuality_pipeline       metrics_pipeline.ipynb,
batch_parse_results                                    scripts/metrics/              ethnicity_metrics.ipynb,
                                                       build_valid_calls,            ...
                                                       build_ethnicity_distributions
```

Each stage reads what the previous one wrote. See `code/README.md` for the ordered command list, the schema of every intermediate CSV, and the optional branches (manual annotation, ethnicity inference, inter-annotator agreement).

---

## Related work

This codebase extends the auditor framework introduced in *Whose Name Comes Up? I: Auditing LLM-Based Scholar Recommendations* — [arXiv:2506.00074](https://arxiv.org/abs/2506.00074) — with the persona-prompting axis (language × country × role).

---

## Status

Research code. APIs and intermediate CSV schemas may change between paper revisions.
