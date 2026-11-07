
# Smoke-test

``` bash
conda activate py311_llmscholar
export PYTHONPATH="$PYTHONPATH:code/libs" LOGS=logs; MODELS=data/models/models.txt
```
## Collect responses (3 prompts, 1 run, English, into tmp/smoke/responses)

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

## Sample GT

``` bash
python code/scripts/configure_analysis.py --config config/analysis_smoke.yaml
```

``` bash
rm -r tmp/smoke/groundtruth/population tmp/smoke/matching tmp/smoke/metrics
```

``` bash
bash code/scripts/run_gt_pipeline.sh -s names,demographics,population -o tmp/smoke/mini_openalex.duckdb -n 4 -S 3 -r tmp/smoke -y config/analysis_smoke.yaml -l tmp/smoke/logs
```


## Parse responses

``` bash
printf "deepseek-r1-8b-0528-qwen3-q4_K_M\ngemma3-4b-it-q4_K_M\nphi4-reasoning-14b-q4_K_M\n" > tmp/smoke/models.txt
```

``` bash
bash code/scripts/run_responses_pipeline.sh -M tmp/smoke/models.txt -r tmp/smoke -l tmp/smoke/logs -x "--overwrite"
```

``` bash
bash code/scripts/check_eval_runs.sh -M tmp/smoke/models.txt -r tmp/smoke -l tmp/smoke/logs -v
```

## Find name-author matches

``` bash 
bash code/scripts/run_matching_pipeline.sh -n 4 -S 4 -t 4 -m 8GB -r tmp/smoke -M tmp/smoke/models.txt -y config/analysis_smoke.yaml -l tmp/smoke/logs
```

## Compute metrics

``` bash
bash code/scripts/run_metrics_pipeline.sh -r tmp/smoke -M tmp/smoke/models.txt -y config/analysis_smoke.yaml -l tmp/smoke/logs
```

## Address RQs

``` bash
bash code/scripts/run_metrics_pipeline.sh -s flows -r tmp/smoke -M tmp/smoke/models.txt -y config/analysis_smoke.yaml -l tmp/smoke/logs
```

``` bash
bash code/scripts/run_metrics_pipeline.sh -s rq1 -r tmp/smoke -M tmp/smoke/models.txt -y config/analysis_smoke.yaml -l tmp/smoke/logs
```

``` bash
bash code/scripts/run_metrics_pipeline.sh -s rq2 -r tmp/smoke -M tmp/smoke/models.txt -y config/analysis_smoke.yaml -l tmp/smoke/logs
```

``` bash
bash code/scripts/run_metrics_pipeline.sh -s flows -r tmp/smoke -M tmp/smoke/models.txt -y config/analysis_smoke.yaml -l tmp/smoke/logs
```


## Variant (smoke: p_female 0.2 / 0.8 -> own GT population, main's matching)

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

## 3. Analysis config, early settings (seniority, field share, L/M/H bins, matching candidates); the rest stays provisional
- [x] done | date: 2026-11-06
``` bash
python code/scripts/configure_analysis.py --config config/analysis.yaml --phase early
```

## 4. Perceived demographics and population from OA (A.2-A.4)
- [x] done | date: 2026-11-06
``` bash
systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% nice -n 10 bash code/scripts/run_gt_pipeline.sh -s names,demographics,population -C _old/ethnicity,_old/demographics
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

#### Evaluate classificaiton

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


### 2.2 Mathching authors
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


## 3. Metrics (technical, social, people table of the flows; with the provisional late settings)
- [x] done | date: 2026-11-06
``` bash
bash code/scripts/run_metrics_pipeline.sh -s technical,social,flows
```

## 4. Analysis config, late settings (valid tags, gender / ethnicity thresholds, social, Jaro-Winkler), then re-run what it lists
- [x] done | date: 2026-11-06
``` bash
python code/scripts/configure_analysis.py --config config/analysis.yaml --phase late
```

## 5. Research questions (RQ1, RQ2; RQ3 in the notebook)
- [x] done | date: 2026-11-06
``` bash
bash code/scripts/run_metrics_pipeline.sh -s rq1,rq2
```

## 6. Size-adjusted social metrics (expected values, then the social metrics again with the _adj columns)
- [ ] done | date:
``` bash
bash code/scripts/run_metrics_pipeline.sh -s expected,social -x "--overwrite"
```

``` bash
bash code/scripts/run_metrics_pipeline.sh -s rq1,rq2
```
