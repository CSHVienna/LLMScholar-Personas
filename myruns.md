
# Smoke-test

``` bash
conda activate py311_llmscholar
export PYTHONPATH="$PYTHONPATH:code/libs" LOGS=logs; MODELS=data/models/models.txt
```
## Sample GT

``` bash
python code/scripts/configure_analysis.py --config config/analysis_smoke.yaml
```

``` bash
rm -r tmp/smoke/groundtruth/gt_*.parquet tmp/smoke/matching tmp/smoke/metrics
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
bash code/scripts/run_metrics_pipeline.sh -s rq1 -r tmp/smoke -M tmp/smoke/models.txt -y config/analysis_smoke.yaml -l tmp/smoke/logs
```

``` bash
bash code/scripts/run_metrics_pipeline.sh -s rq2 -r tmp/smoke -M tmp/smoke/models.txt -y config/analysis_smoke.yaml -l tmp/smoke/logs
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
- [x] done | date:
``` bash
python code/scripts/configure_analysis.py --config config/analysis.yaml --phase early
```

## 4. Perceived demographics and population from OA (A.2-A.4)
- [ ] done | date:
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
- [x] done | date: 2026-11-05
``` bash
parallel -j 10 --nice 10 python code/scripts/batch_eval_responses.py --models {} :::: $MODELS
```

#### Classifying recommnedations
- [x] done | date: 2026-11-05
``` bash
parallel -j 10 nice -n 10 python code/scripts/batch_eval_recommendations.py --models {} :::: $MODELS
```

#### Checks the latest run of both steps (status, order, current tag names)
- [x] done | date: 2026-11-05
``` bash
bash code/scripts/check_eval_runs.sh   
```

#### Evaluate classificaiton

##### Draw the blind samples: responses
- [ ] done | date:
``` python
python code/scripts/batch_annotation_sample.py --task responses --refresh_codebook
```

##### Draw the blind samples: recommendations
- [ ] done | date:
``` python
python code/scripts/batch_annotation_sample.py --task recommendations --refresh_codebook
```

##### Start the tool server and leave it running:
- [ ] done | date:
``` python
python code/scripts/serve_annotation_tool.py
```

##### Stop the server with Ctrl+C when you're done

##### Compare results: responses
- [ ] done | date:
``` python
python code/scripts/batch_annotation_evaluate.py --task responses
```

##### Compare results: recommendations
- [ ] done | date:
``` python
python code/scripts/batch_annotation_evaluate.py --task recommendations
```

Note:
- Look at the results, and decide what accounts for 'valid'
- For example: 
- `valid responses: correct, wrapped, incomplete, python_literal, embedded`
- `valid recommendations: correct, title_removed `


### 2.2 Mathching authors
- [ ] done | date:
``` bash
systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% nice -n 10 bash code/scripts/run_matching_pipeline.sh
```

#### Validate the matching (sample of recommended names vs OpenAlex)
- [ ] done | date:
``` bash
python code/scripts/batch_annotation_sample.py --task matching --config config/analysis.yaml
```

#### Validate the GT field, seniority and country (sample of GT scholars)
- [ ] done | date:
``` bash
python code/scripts/batch_annotation_sample.py --task authors
```

#### Annotate (choose matching / authors in the tool), then stop with Ctrl+C
- [ ] done | date:
``` bash
python code/scripts/serve_annotation_tool.py
```

#### Compare results: matching
- [ ] done | date:
``` bash
python code/scripts/batch_annotation_evaluate.py --task matching
```

#### Compare results: authors
- [ ] done | date:
``` bash
python code/scripts/batch_annotation_evaluate.py --task authors
```


## 3. Metrics (technical, social; with the provisional late settings)
- [ ] done | date:
``` bash
bash code/scripts/run_metrics_pipeline.sh -s technical,social
```

## 4. Analysis config, late settings (valid tags, gender / ethnicity thresholds, social, Jaro-Winkler), then re-run what it lists
- [ ] done | date:
``` bash
python code/scripts/configure_analysis.py --config config/analysis.yaml --phase late
```

## 5. Research questions (RQ1, RQ2; RQ3 in the notebook)
- [ ] done | date:
``` bash
bash code/scripts/run_metrics_pipeline.sh -s rq1,rq2
```
