# Full run (GT to audit)

From scratch, everything after the LLM data collection. One step per block, in this order. Every step emails
OK / FAILED when it ends (`notify.sh`; setup: code/scripts/README.md). Finished outputs are skipped, so a failed step
is simply run again.

## Setup
- [ ] done | date:

Environment, the capped scope of the shared server ($CAP) and the email wrapper ($N)
``` bash
conda activate py311_llmscholar
export PYTHONPATH="$PYTHONPATH:code/libs"
CAP="systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% nice -n 10"
N="bash code/scripts/notify.sh"
```

Config complete (early + late settings; no output = ok)
``` bash
python code/scripts/configure_analysis.py --config config/analysis.yaml --check --phase late
```

Annotations back from the backup (manual work, needed by Validation)
``` bash
cp -a /data/datasets/LLMScholar-PersonasContext/backups/2026-11-13_main_full_before_rerun/results/annotations results/
```

## All at once
Alternative to the steps below: the same sequence, one email per step. A failure stops the later steps; run it again
after the fix: completed steps are skipped (list: results/_run_all_done.txt; steps run by hand below are not in it).
`-n` lists the steps (done / run), `-f -s <step> -e <step>` runs one again.
``` bash
$CAP bash code/scripts/run_all.sh
```

## Ground truth

### Authorships
- [ ] done | date:

A.0: works x authors from OpenAlex, per chunk (the longest DuckDB step)
``` bash
$CAP $N bash code/scripts/run_gt_pipeline.sh -s authorships
```

### Coauthor pairs
- [ ] done | date:
``` bash
$CAP $N bash code/scripts/run_gt_pipeline.sh -s pairs
```

### Coauthor edges
- [ ] done | date:
``` bash
$CAP $N bash code/scripts/run_gt_pipeline.sh -s edges
```

### Bibliometrics
- [ ] done | date:

A.1: per author (works, citations, h-index, field works, country)
``` bash
$CAP $N bash code/scripts/run_gt_pipeline.sh -s bibliometrics
```

### Name index
- [ ] done | date:

A.2: normalised names of every OpenAlex author (input of the matching)
``` bash
$CAP $N bash code/scripts/run_gt_pipeline.sh -s names
```

### Demographics
- [ ] done | date:

A.3: gender, race, ethnicity from names (raceBERT: days). Faster: add
`-C /data/datasets/LLMScholar-PersonasContext/backups/2026-11-13_main_full_before_rerun/results/groundtruth/demographics/race,/data/datasets/LLMScholar-PersonasContext/backups/2026-11-13_main_full_before_rerun/results/groundtruth/demographics/ethnicity`
(reuses the earlier probabilities by name)
``` bash
$CAP $N bash code/scripts/run_gt_pipeline.sh -s demographics
```

### Population
- [ ] done | date:

A.4: field members, seniority, labels, GT shares (thresholds from the config)
``` bash
$CAP $N bash code/scripts/run_gt_pipeline.sh -s population
```

## Response tags

### Responses
- [ ] done | date:

B.1-B.2: parse and tag every response
``` bash
$CAP $N bash code/scripts/run_responses_pipeline.sh -s summary
```

### Recommendations
- [ ] done | date:

B.3: clean and tag every recommended name
``` bash
$CAP $N bash code/scripts/run_responses_pipeline.sh -s recommendations
```

### Check both
- [ ] done | date:
``` bash
bash code/scripts/check_eval_runs.sh
```

## Author matching

### Candidates
- [ ] done | date:

B.6a: OpenAlex candidates of every name (heavy, once)
``` bash
$CAP $N bash code/scripts/run_matching_pipeline.sh -s candidates
```

### Matches
- [ ] done | date:

B.6b: thresholds, levels, picked author per recommendation
``` bash
$CAP $N bash code/scripts/run_matching_pipeline.sh -s authors
```

## Metrics

### Technical
- [ ] done | date:

Validity, refusals, consistency, duplicates, k balance, factuality
``` bash
$CAP $N bash code/scripts/run_metrics_pipeline.sh -s technical
```

### Expected (random)
- [ ] done | date:

Social metrics of a random recommender (for the _adj metrics)
``` bash
$CAP $N bash code/scripts/run_metrics_pipeline.sh -s expected
```

### Social
- [ ] done | date:

Parity, diversity, popularity, cohesion, locality
``` bash
$CAP $N bash code/scripts/run_metrics_pipeline.sh -s social
```

### Flows
- [ ] done | date:
``` bash
$CAP $N bash code/scripts/run_metrics_pipeline.sh -s flows
```

## Research questions

### RQ1
- [ ] done | date:
``` bash
$CAP $N bash code/scripts/run_metrics_pipeline.sh -s rq1
```

### RQ2
- [ ] done | date:
``` bash
$CAP $N bash code/scripts/run_metrics_pipeline.sh -s rq2
```

### Interactions
- [ ] done | date:

~27 min per metric at 1000 replicates
``` bash
$CAP $N bash code/scripts/run_metrics_pipeline.sh -s interactions -j 7 -c 2
```

## Validation

Annotated samples vs the new outputs (the samples themselves are not redrawn)

### Responses
- [ ] done | date:
``` bash
python code/scripts/batch_annotation_evaluate.py --task responses
```

### Recommendations
- [ ] done | date:
``` bash
python code/scripts/batch_annotation_evaluate.py --task recommendations
```

### Matching
- [ ] done | date:
``` bash
python code/scripts/batch_annotation_evaluate.py --task matching --per_stratum 10
```

### Authors
- [ ] done | date:
``` bash
python code/scripts/batch_annotation_evaluate.py --task authors --per_stratum 10
```

### Demographics
- [ ] done | date:
``` bash
python code/scripts/batch_annotation_evaluate.py --task demographics
```

## Notebooks

Figures: main text -> plots/main/, the rest -> plots/main/si/

### GT descriptives
- [ ] done | date:
``` bash
$CAP $N jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 code/notebooks/0_gt.ipynb
```

### Tags
- [ ] done | date:
``` bash
$CAP $N jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 code/notebooks/1_responses_tags.ipynb
```

### Matching stats
- [ ] done | date:
``` bash
$CAP $N jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 code/notebooks/2_matching_stats.ipynb
```

### Metric bars
- [ ] done | date:
``` bash
$CAP $N jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 code/notebooks/3_metrics.ipynb
```

### Research questions
- [ ] done | date:
``` bash
$CAP $N jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 code/notebooks/4_research_questions.ipynb
```

### Flows
- [ ] done | date:
``` bash
$CAP $N jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 code/notebooks/5_flows.ipynb
```


# Inputs and LLM responses

## 1. Prompt parameters 
- [ ] done | date:
``` bash
python code/scripts/batch_input_params.py -l english
```

## 2. Prompts of all languages
- [ ] done | date:
``` bash
python code/scripts/batch_input_prompts.py
```

## 3. Model list from metadata.json
- [ ] done | date:
``` bash
python code/scripts/batch_input_models.py
```

## 4. Collect GPT / Gemini 
- [ ] done | date:
``` bash
python code/scripts/batch_collect_cloud_prepare.py --models gpt-4_1-nano-2025-04-14 gpt-4_1-mini-2025-04-14 gpt-4_1-2025-04-14 gemini-2_5-flash-lite gemini-2_5-flash gemini-2_5-pro
```

``` bash
python code/scripts/batch_collect_cloud_submit.py --models gpt-4_1-nano-2025-04-14 gpt-4_1-mini-2025-04-14 gpt-4_1-2025-04-14 gemini-2_5-flash-lite gemini-2_5-flash gemini-2_5-pro
```

``` bash
python code/scripts/batch_collect_cloud_retrieve.py --models gpt-4_1-nano-2025-04-14 gpt-4_1-mini-2025-04-14 gpt-4_1-2025-04-14 gemini-2_5-flash-lite gemini-2_5-flash gemini-2_5-pro
```

## 5. Collect Ollama 
- [ ] done | date:
``` bash
python code/scripts/batch_input_models.py --apis ollama --output tmp/models_ollama.txt
```

``` bash
parallel -j 1 --joblog logs/collect_local.joblog python code/scripts/batch_collect_local.py --nr_servers 4 --max_processes 8 --models {} :::: tmp/models_ollama.txt
```




# Smoke-test

``` bash
conda activate py311_llmscholar
export PYTHONPATH="$PYTHONPATH:code/libs" LOGS=logs; MODELS=data/models/models.txt
```

## LLMCaller

### Collect responses 
(3 prompts, 1 run, English, into tmp/smoke/responses)

``` bash
python code/scripts/batch_input_prompts.py
```

``` bash
python code/scripts/batch_collect_cloud_prepare.py --models gpt-4_1-nano-2025-04-14 gemini-2_5-flash-lite --languages english --repetitions 1 --limit_prompts 3 --responses_dir tmp/smoke/responses --log_dir tmp/smoke/logs
```

``` bash
python code/scripts/batch_collect_cloud_submit.py --models gpt-4_1-nano-2025-04-14 gemini-2_5-flash-lite --languages english --responses_dir tmp/smoke/responses --log_dir tmp/smoke/logs
```

``` bash
python code/scripts/batch_collect_cloud_retrieve.py --models gpt-4_1-nano-2025-04-14 gemini-2_5-flash-lite --languages english --responses_dir tmp/smoke/responses --log_dir tmp/smoke/logs
```

``` bash
python code/scripts/batch_collect_local.py --models gemma3-4b-it-q4_K_M --languages english --repetitions 1 --limit_prompts 3 --responses_dir tmp/smoke/responses --log_dir tmp/smoke/logs
```

### Parse responses

``` bash
printf "deepseek-r1-8b-0528-qwen3-q4_K_M\ngemma3-4b-it-q4_K_M\nphi4-reasoning-14b-q4_K_M\n" > tmp/smoke/models.txt
```

``` bash
bash code/scripts/run_responses_pipeline.sh -M tmp/smoke/models.txt -r tmp/smoke -l tmp/smoke/logs -x "--overwrite"
```

``` bash
bash code/scripts/check_eval_runs.sh -M tmp/smoke/models.txt -r tmp/smoke -l tmp/smoke/logs -v
```


## GTBuilder

### Sample GT

``` bash
python code/scripts/configure_analysis.py --config config/analysis_smoke.yaml
```

``` bash
rm -r tmp/smoke/groundtruth/population tmp/smoke/matching tmp/smoke/metrics
```

``` bash
bash code/scripts/run_gt_pipeline.sh -s names,demographics,population -o tmp/smoke/mini_openalex.duckdb -n 4 -S 3 -r tmp/smoke -y config/analysis_smoke.yaml -l tmp/smoke/logs
```


### Inference
Infer demographics (gender, race, ethnicity) and bibliometrics

``` bash
mkdir -p tmp/smoke/groundtruth/archive && mv tmp/smoke/groundtruth/demographics tmp/smoke/groundtruth/archive/demographics_v2
```

``` bash
bash code/scripts/run_gt_pipeline.sh -s demographics,population -o tmp/smoke/mini_openalex.duckdb -n 4 -S 3 -r tmp/smoke -y config/analysis_smoke.yaml -l tmp/smoke/logs -a "--overwrite"
```

``` bash
bash code/scripts/run_gt_pipeline.sh -s population -v smoke_pf02 -r tmp/smoke -l tmp/smoke/logs -a "--overwrite"
```

### Audit

#### Find name-author matches

``` bash 
bash code/scripts/run_matching_pipeline.sh -n 4 -S 4 -t 4 -m 8GB -r tmp/smoke -M tmp/smoke/models.txt -y config/analysis_smoke.yaml -l tmp/smoke/logs
```

#### Raw metrics stats

``` bash
# Technical?
bash code/scripts/run_metrics_pipeline.sh -r tmp/smoke -M tmp/smoke/models.txt -y config/analysis_smoke.yaml -l tmp/smoke/logs
```

``` bash
# Socials
bash code/scripts/run_metrics_pipeline.sh -s expected,social,flows -r tmp/smoke -M tmp/smoke/models.txt -y config/analysis_smoke.yaml -l tmp/smoke/logs -x "--overwrite"
```

#### RQs

``` bash
bash code/scripts/run_metrics_pipeline.sh -s rq1,rq2,interactions -r tmp/smoke -M tmp/smoke/models.txt -y config/analysis_smoke.yaml -l tmp/smoke/logs -x "--overwrite" -q "--n_boot 50"
```

### Audit (variant)

#### Variant 
(smoke: p_female 0.2 / 0.8 -> own GT population, main's matching)

``` bash
python code/scripts/configure_variant.py create smoke_pf02 --base config/analysis_smoke.yaml --set pf=0.2,0.8
```

``` bash
bash code/scripts/run_gt_pipeline.sh -s population -v smoke_pf02 -r tmp/smoke -l tmp/smoke/logs
```

``` bash
bash code/scripts/run_matching_pipeline.sh -s authors -v smoke_pf02 -r tmp/smoke -M tmp/smoke/models.txt -l tmp/smoke/logs
```

``` bash
bash code/scripts/run_metrics_pipeline.sh -v smoke_pf02 -r tmp/smoke -M tmp/smoke/models.txt -l tmp/smoke/logs
```

---------------------------------------------
