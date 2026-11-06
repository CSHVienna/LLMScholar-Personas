#!/usr/bin/env bash
# =============================================================================
# Step A (ground truth from OpenAlex). Works from any working directory.
#   authorships   A.0  works x authorships, per chunk of works rows (~240 chunks)       DuckDB, one job at a time
#   pairs         A.0b coauthor pairs per chunk (self-join inside the chunk)            DuckDB, one job at a time
#   edges         A.1a coauthorship edge list per author bucket                         DuckDB, one job at a time
#   bibliometrics A.1  per author bucket (needs A.0 + A.1a)                             DuckDB, one job at a time
#   names         A.2  per author bucket (Python normalisation)                         GNU parallel
#   demographics  A.3  needs A.1; four stages, models run ONCE per distinct name:     GNU parallel
#                        keys (per bucket: names) -> gender (one process, ~1 min) -> ethnicity (raceBERT per
#                        name shard, CPU; the slow part) -> merge (per bucket)
#   population    A.4  single process (needs A.1 + A.3; thresholds live here)           DuckDB
#
# DuckDB already uses many threads per process, so its steps run one chunk / bucket after the other
# (-d 1 job x -t 12 threads x -m 40GB). The Python-bound steps run -j jobs in parallel (-j 4 x -c / -t 3 threads).
# Parallel/sequential jobs write a GNU parallel joblog logs/<YYYY-mm-dd-HH:MM>_<step>_<chunks|buckets>.joblog and
# every job writes its own log. Everything is resumable: re-running the same command skips finished chunks /
# buckets, and demographics continues inside a name shard from its last stored part (every 20k names).
# All scratch / spill space is inside the project: <project>/tmp/ (git-ignored).
#
# On the shared server, run the whole pipeline inside a capped scope (only this pipeline can be killed):
#   systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% \
#       bash code/scripts/run_gt_pipeline.sh
#
# Usage:
#   bash code/scripts/run_gt_pipeline.sh                       # all steps (defaults fit 64 GB / 14 cores)
#   bash code/scripts/run_gt_pipeline.sh -s bibliometrics      # one step
#   bash code/scripts/run_gt_pipeline.sh -o tmp/smoke/mini_openalex.duckdb -n 4 -r tmp/smoke -l tmp/smoke/logs   # smoke
#
# Options (same letters as in run_matching_pipeline.sh / run_responses_pipeline.sh):
#   -s STEPS  comma-separated: authorships,pairs,edges,bibliometrics,names,demographics,population, or all (default);
#             always run in this order, whatever order they are given in
#   -d N      jobs at a time for the DuckDB steps (default 1)
#   -t N      DuckDB threads per DuckDB-step job (default 12)
#   -m SIZE   DuckDB memory limit per DuckDB-step job (default 40GB)
#   -j N      parallel jobs for the Python steps names / demographics (default 4)
#   -S N      raceBERT name shards (default 64; must stay the same until demographics is complete)
#   -C DIRS   demographics: reuse raceBERT probabilities of earlier runs, comma-separated folders: ethnicity shards
#             (matched by name) and / or demographics buckets (by author_id -> name), e.g. -C _old/ethnicity,_old/demographics
#   -c N      threads per Python-step job (raceBERT / DuckDB reads; default 3)
#   -T DIR    DuckDB spill folder (default <project>/tmp/duckdb)
#   -n N      number of author buckets (default 64; must be the same in every step)
#   -k N      works rows per A.0 chunk (default 2000000; must stay the same until A.0 is complete)
#   -o FILE   OpenAlex DuckDB file (default data/openalex_20260923.duckdb)
#   -p ENV    python executable (default: python of the active environment)
#   -r DIR    results root (default <project>/results): ground truth in DIR/groundtruth
#   -l DIR    logs folder (default <project>/logs)
#   -x ARGS   extra arguments for ALL per-bucket scripts (only common ones, e.g. "--overwrite")
#   -a ARGS   extra arguments for population (e.g. "--overwrite"; thresholds come from the config, -y)
#   -y FILE   analysis config (default config/analysis.yaml; smoke: config/analysis_smoke.yaml); population reads it.
#             Missing: in a terminal the wizard (configure_analysis.py --phase early) creates it, otherwise the pipeline stops
# =============================================================================
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPTS_DIR}/../.." && pwd)"
LOGS_DIR="${PROJECT_ROOT}/logs"
RESULTS_DIR="${PROJECT_ROOT}/results"
OPENALEX_DB="${PROJECT_ROOT}/data/openalex_20260923.duckdb"
TEMP_DIR="${PROJECT_ROOT}/tmp/duckdb"

N_SHARDS=64; STEP="all"; DUCK_JOBS=1; THREADS=12; MEMORY="40GB"; JOBS=4; PY_THREADS=3; N_BUCKETS=64; CHUNK_SIZE=2000000
PYTHON="python"; EXTRA_ARGS=""; SINGLE_ARGS=""; CACHE_ARGS=""; CONFIG="${PROJECT_ROOT}/config/analysis.yaml"
while getopts "s:d:t:m:j:S:C:c:T:n:k:o:p:r:l:x:a:y:h" option; do
    case "${option}" in
        s) STEP="${OPTARG}" ;;
        d) DUCK_JOBS="${OPTARG}" ;;
        t) THREADS="${OPTARG}" ;;
        m) MEMORY="${OPTARG}" ;;
        j) JOBS="${OPTARG}" ;;
        S) N_SHARDS="${OPTARG}" ;;
        C) CACHE_ARGS="--cache_dir ${OPTARG//,/ }" ;;
        c) PY_THREADS="${OPTARG}" ;;
        T) TEMP_DIR="${OPTARG}" ;;
        n) N_BUCKETS="${OPTARG}" ;;
        k) CHUNK_SIZE="${OPTARG}" ;;
        o) OPENALEX_DB="${OPTARG}" ;;
        p) PYTHON="${OPTARG}" ;;
        r) RESULTS_DIR="${OPTARG}" ;;
        l) LOGS_DIR="${OPTARG}" ;;
        x) EXTRA_ARGS="${OPTARG}" ;;
        a) SINGLE_ARGS="${OPTARG}" ;;
        y) CONFIG="${OPTARG}" ;;
        h|*) sed -n '2,53p' "$0"; exit 0 ;;
    esac
done
GT_DIR="${RESULTS_DIR}/groundtruth"
ALL_STEPS="authorships pairs edges bibliometrics names demographics population"
for s in ${STEP//,/ }; do
    if [[ "${s}" != "all" && " ${ALL_STEPS} " != *" ${s} "* ]]; then
        echo "ERROR: unknown step '${s}' in -s ${STEP} (steps: ${ALL_STEPS// /,} or all)"; exit 1
    fi
done
selected() { [[ ",${STEP}," == *",all,"* || ",${STEP}," == *",$1,"* ]]; }   # selected <step>: is it in -s?
mkdir -p "${LOGS_DIR}"
if ! mkdir -p "${TEMP_DIR}" || [[ ! -w "${TEMP_DIR}" ]]; then
    echo "ERROR: DuckDB spill folder ${TEMP_DIR} cannot be created / written (-T)"; exit 1
fi
echo "[$(date +%H:%M:%S)] DuckDB steps: ${DUCK_JOBS} job(s) x ${THREADS} threads x ${MEMORY} | Python steps: ${JOBS} jobs x ${PY_THREADS} threads | spill ${TEMP_DIR}"
DB="--openalex_db ${OPENALEX_DB}"
DUCK_RES="--threads ${THREADS} --memory_limit ${MEMORY} --temp_dir ${TEMP_DIR}"
PY_RES="--threads ${PY_THREADS} --memory_limit 8GB --temp_dir ${TEMP_DIR}"

# ensure_config: the analysis config must exist before the steps that read it; in a terminal the wizard creates it
ensure_config() {   # the config must exist and be complete (settings added later): else the wizard, in a terminal
    local problem
    if problem="$("${PYTHON}" "${SCRIPTS_DIR}/configure_analysis.py" --config "${CONFIG}" --check)"; then
        echo "[$(date +%H:%M:%S)] analysis config: ${CONFIG}"
    elif [[ -t 0 && -t 1 ]]; then
        echo "[$(date +%H:%M:%S)] ${problem}"
        echo "[$(date +%H:%M:%S)] starting the wizard"
        "${PYTHON}" "${SCRIPTS_DIR}/configure_analysis.py" --config "${CONFIG}" --phase early || { echo "No config written: stopping."; exit 1; }
    else
        echo "ERROR: ${problem}"
        exit 1
    fi
}

# run_jobs <step name> <jobs> <unit: chunks|buckets> <count> <script> <script args>: GNU parallel, timestamped joblog
run_jobs() {
    local name="$1" jobs="$2" unit="$3" count="$4" script="$5" args="$6 --log_dir ${LOGS_DIR}"
    local joblog="${LOGS_DIR}/$(date +%Y-%m-%d-%H:%M)_${name}_${unit}.joblog"
    echo "[$(date +%H:%M:%S)] ${name}: ${count} ${unit}, ${jobs} job(s) at a time, joblog ${joblog}"
    if ! parallel -j "${jobs}" --nice 10 --joblog "${joblog}" \
            "${PYTHON}" "${SCRIPTS_DIR}/${script}" --${unit} {} ${args} '>' /dev/null \
            ::: $(seq 0 $((count - 1))); then
        echo "[$(date +%H:%M:%S)] ${name}: some ${unit} FAILED (see ${joblog}, column Exitval, and their logs in ${LOGS_DIR})"
        echo "Re-run the same command: finished ${unit} are skipped."
        exit 1
    fi
    echo "[$(date +%H:%M:%S)] ${name}: done"
}
per_bucket() { run_jobs "$1" "$2" buckets "${N_BUCKETS}" "$3" "$4 --n_buckets ${N_BUCKETS} ${EXTRA_ARGS}"; }

N_CHUNKS="$("${PYTHON}" "${SCRIPTS_DIR}/batch_gt_authorships.py" ${DB} --chunk_size "${CHUNK_SIZE}" --count_chunks)"
if selected authorships; then
    run_jobs "gt_authorships" "${DUCK_JOBS}" chunks "${N_CHUNKS}" "batch_gt_authorships.py" \
        "${DB} ${DUCK_RES} --chunk_size ${CHUNK_SIZE} --n_buckets ${N_BUCKETS} --output_dir ${GT_DIR}/authorships"
fi
if selected pairs; then
    run_jobs "gt_pairs" "${DUCK_JOBS}" chunks "${N_CHUNKS}" "batch_gt_pairs.py" \
        "${DUCK_RES} --authorships_dir ${GT_DIR}/authorships --output_dir ${GT_DIR}/coauthor_pairs"
fi
if selected edges; then
    per_bucket "gt_edges" "${DUCK_JOBS}" "batch_gt_edges.py" \
        "${DUCK_RES} --pairs_dir ${GT_DIR}/coauthor_pairs --output_dir ${GT_DIR}/coauthor_edges"
fi
if selected bibliometrics; then
    per_bucket "gt_bibliometrics" "${DUCK_JOBS}" "batch_gt_bibliometrics.py" \
        "${DB} ${DUCK_RES} --authorships_dir ${GT_DIR}/authorships --edges_dir ${GT_DIR}/coauthor_edges --output_dir ${GT_DIR}/authors"
fi
if selected names; then
    per_bucket "gt_names" "${JOBS}" "batch_gt_names.py" "${DB} ${PY_RES} --output_dir ${GT_DIR}/names"
fi
if selected demographics; then
    DEMO="--output_dir ${GT_DIR}/demographics --n_shards ${N_SHARDS} ${PY_RES}"
    per_bucket "gt_demographics_keys" "${JOBS}" "batch_gt_demographics.py" \
        "--stage keys --authors_dir ${GT_DIR}/authors ${DEMO}"
    echo "[$(date +%H:%M:%S)] gt_demographics_gender: start"
    "${PYTHON}" "${SCRIPTS_DIR}/batch_gt_demographics.py" --stage gender --n_buckets "${N_BUCKETS}" ${DEMO} \
        --log_dir "${LOGS_DIR}" ${EXTRA_ARGS} > /dev/null
    echo "[$(date +%H:%M:%S)] gt_demographics_gender: done"
    run_jobs "gt_demographics_ethnicity" "${JOBS}" shards "${N_SHARDS}" "batch_gt_demographics.py" \
        "--stage ethnicity --n_buckets ${N_BUCKETS} --torch_threads ${PY_THREADS} ${CACHE_ARGS} ${DEMO} ${EXTRA_ARGS}"
    per_bucket "gt_demographics_merge" "${JOBS}" "batch_gt_demographics.py" "--stage merge ${DEMO}"
fi
if selected population; then
    ensure_config
    echo "[$(date +%H:%M:%S)] gt_population: start"
    "${PYTHON}" "${SCRIPTS_DIR}/batch_gt_population.py" ${DUCK_RES} --config "${CONFIG}" \
        --authors_dir "${GT_DIR}/authors" --demographics_dir "${GT_DIR}/demographics" --gt_dir "${GT_DIR}" \
        --log_dir "${LOGS_DIR}" ${SINGLE_ARGS}
    echo "[$(date +%H:%M:%S)] gt_population: done"
fi
