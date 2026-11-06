"""
Ground truth (step A): OpenAlex authors of the prompt fields, their bibliometrics, name index,
seniority, prominence classes and GT shares. OpenAlex is opened read-only.

Order of the steps (one process per author bucket, GNU parallel; see scripts/run_gt_pipeline.sh):
    authorships.py   works x authorships (filtered works), partitioned by author bucket
    bibliometrics.py per-author bibliometrics and works per prompt field
    names.py         name index over ALL OpenAlex authors (name_key as in the recommendations)
    demographics/    perceived gender and ethnicity of the field authors (libs/demographics)
    population.py    target population, seniority, L/M/H bins, top group, GT shares
"""
