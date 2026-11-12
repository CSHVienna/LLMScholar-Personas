
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


# Ground-truth

## 1. Setup
``` bash
conda activate py311_llmscholar
export PYTHONPATH="$PYTHONPATH:code/libs" LOGS=logs; MODELS=data/models/models.txt
```

## 2. Extract bibliometrics from OA (A.0-A.1)
- [x] done | date: 2026-11-05
``` bash
systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% nice -n 10 bash code/scripts/run_gt_pipeline.sh -s authorships,pairs,edges,bibliometrics
```

## 3. Analysis config, early settings 
(seniority, field share, L/M/H bins, matching candidates)
- [x] done | date: 2026-11-06
``` bash
python code/scripts/configure_analysis.py --config config/analysis.yaml --phase early
```

## 4. Perceived demographics and population from OA (A.2-A.4)
- [x] done | date: 2026-11-06
``` bash
systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% nice -n 10 bash code/scripts/run_gt_pipeline.sh -s names,demographics,population -C _old/ethnicity,_old/demographics
```

## 4b. Ethnicity (switch between race and ethnicity)
- [ ] done | date:

Dry run: checks the old run is complete and prints what would be written to results/groundtruth/demographics.new/
``` bash
python tmp/migrate_demographics.py prepare
```

Build demographics.new/ (keys joined, race shards renamed, gender / ethnicity shards hard-linked)
``` bash
python tmp/migrate_demographics.py prepare --apply
```

The new merge stage on demographics.new/ (reads the race and ethnicity shards for every bucket)
``` bash
systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% nice -n 10 python code/scripts/batch_gt_demographics.py --stage merge --output_dir results/groundtruth/demographics.new --threads 12 --memory_limit 40GB
```

Compare every new bucket with the old files, and recompute the keys of bucket 0 with the current code
``` bash
python tmp/migrate_demographics.py verify
```

Move the old folder to results/groundtruth/archive/demographics_v2/ (kept) and demographics.new/ into its place
``` bash
python tmp/migrate_demographics.py swap --apply
```

A.4 again: race and ethnicity labels and shares (thresholds from the config: race / ethnicity min_prob)
``` bash
systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% nice -n 10 bash code/scripts/run_gt_pipeline.sh -s population -a "--overwrite"
```

Demographics annotation key: ethnicity -> race, + the inferred ethnicity (original kept as .v2)
``` bash
python tmp/migrate_demographics.py annotation_key --apply
```


# LLM outcomes

## 1. Setup
``` bash
conda activate py311_llmscholar
export PYTHONPATH="$PYTHONPATH:code/libs" LOGS=logs; MODELS=data/models/models.txt
```

## 2. Pre-processing

### 2.1 Parsing LLM outputs

#### Classifying responses
- [x] done | date: 2026-11-06
``` bash
parallel -j 10 --nice 10 python code/scripts/batch_eval_responses.py --models {} :::: $MODELS
```

#### Classifying recommnedations
- [x] done | date: 2026-11-06
``` bash
parallel -j 10 nice -n 10 python code/scripts/batch_eval_recommendations.py --models {} :::: $MODELS
```

#### Checks the latest run of both steps (status, order, current tag names)
- [x] done | date: 2026-11-06
``` bash
bash code/scripts/check_eval_runs.sh   
```

### 2.2 Evaluate classificaiton

##### Draw the blind samples: responses
- [x] done | date: 2026-11-06
``` python
python code/scripts/batch_annotation_sample.py --task responses --per_stratum 10 --refresh_codebook
```

##### Draw the blind samples: recommendations
- [x] done | date: 2026-11-06
``` python
python code/scripts/batch_annotation_sample.py --task recommendations --per_stratum 10 --refresh_codebook
```

##### Start the tool server and leave it running:
``` python
python code/scripts/serve_annotation_tool.py
```

##### Stop the server with Ctrl+C when you're done


##### Compare results: responses
- [...] in-progress | date: 2026-11-05 - 2026-11-xx
``` python
python code/scripts/batch_annotation_evaluate.py --task responses
```

##### Compare results: recommendations
- [...] on-hold | date: 2026-11-xx - 2026-11-xx
``` python
python code/scripts/batch_annotation_evaluate.py --task recommendations
```

Note:
- Look at the results, and decide what accounts for 'valid'
- For example: 
- `valid responses: correct, wrapped, incomplete, python_literal, embedded`
- `valid recommendations: correct, title_removed `


### 2.3 Mathching authors
- [x] done | date: 2026-11-06
``` bash
systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% nice -n 10 bash code/scripts/run_matching_pipeline.sh
```

#### Validate the matching (sample of recommended names vs OpenAlex)
- [x] done | date: 2026-11-06
``` bash
python code/scripts/batch_annotation_sample.py --task matching --config config/analysis.yaml --per_stratum 10
```

#### Validate the GT field, seniority and country (sample of GT scholars)
- [x] done | date: 2026-11-06
``` bash
python code/scripts/batch_annotation_sample.py --task authors --config config/analysis.yaml --per_stratum 10
```

#### Perceived gender and ethnicity from the name (sample of GT scholars + recommended people)
- [x] done | date: 2026-11-06
``` bash
python code/scripts/batch_annotation_sample.py --task demographics --per_stratum 10
```

#### Annotate (choose matching / authors / demographics in the tool), then stop with Ctrl+C
``` bash
python code/scripts/serve_annotation_tool.py
```

#### Compare results: matching
- [...] on-hold | date:
``` bash
python code/scripts/batch_annotation_evaluate.py --task matching --per_stratum 10
```

#### Compare results: authors
- [...] on-hold | date:
``` bash
python code/scripts/batch_annotation_evaluate.py --task authors --per_stratum 10
```

#### Compare results: demographics
- [...] on-hold | date:
``` bash
python code/scripts/batch_annotation_evaluate.py --task demographics
```


## 3. Metrics all (technical, expected, social, flows)
- [x] done | date: 2026-11-06
``` bash
bash code/scripts/run_metrics_pipeline.sh -s technical,expected,social,flows
```

Again after D.7.14 (race + ethnicity): expected, social and flows only (technical unchanged)
- [ ] done | date:
``` bash
systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% nice -n 10 bash code/scripts/run_metrics_pipeline.sh -s expected,social,flows -x "--overwrite"
```

## 4. Research questions (RQ1, RQ2 & interactions; RQ3)

### Setup thresholds
- [x] done | date: 2026-11-06
``` bash
python code/scripts/configure_analysis.py --config config/analysis.yaml --phase late
```

### Audit
- [x] done | date: 2026-11-06
``` bash
bash code/scripts/run_metrics_pipeline.sh -s rq1,rq2
```

Again after D.7.14 (metrics of config study.ethnicity_attribute)
- [ ] done | date:
``` bash
systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% nice -n 10 bash code/scripts/run_metrics_pipeline.sh -s rq1,rq2 -x "--overwrite"
```

### Interactions (D.7.3; ~27 min per metric at 1000 replicates, 45 metrics; study.ethnicity_attribute both: 49)
- [ ] done | date:
``` bash
systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% nice -n 10 bash code/scripts/run_metrics_pipeline.sh -s interactions -j 7 -c 2
```

## 5. Notebooks (paper main-text figures -> plots/main/, the rest -> plots/main/si/; outputs saved in the notebooks)
- [ ] done | date:

GT descriptives and composition (race and ethnicity)
``` bash
systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% nice -n 10 jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 code/notebooks/0_gt.ipynb
```

response / recommendation tags
``` bash
systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% nice -n 10 jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 code/notebooks/1_responses_tags.ipynb
```

author matching
``` bash
systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% nice -n 10 jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 code/notebooks/2_matching_stats.ipynb
```

metric bars (sets of config study.ethnicity_attribute)
``` bash
systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% nice -n 10 jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 code/notebooks/3_metrics.ipynb
```

RQ1, RQ2, RQ3 (study.rq3_ceilings), interactions (needs the interactions run)
``` bash
systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% nice -n 10 jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 code/notebooks/4_research_questions.ipynb
```

flows (ethnicity panels: study.ethnicity_attribute)
``` bash
systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% nice -n 10 jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 code/notebooks/5_flows.ipynb
```

