#!/usr/bin/env bash
# =============================================================================
# Step B.6 (author matching of the recommended scholars against OpenAlex). Works from any working directory.
# Needs B.3 (recommendations_metadata) and A.1 / A.2 (authors, names) of the ground truth.
#   candidates   B.6a  rec_names -> exact (per bucket) -> fuzzy_recs -> oa_tokens (per bucket)
#                      -> fuzzy (per shard; the heavy part) -> merge (per bucket)      heavy, run once
#   authors      B.6b  per model, with the thresholds (-a)                            cheap, re-run per threshold
#
# Per-bucket stages and per-model jobs run -j jobs in parallel (-c threads each); the fuzzy shards run -d jobs
# (-t threads x -m memory each). Every stage is resumable: re-running the same command skips finished files.
# Folders are derived from the results root (-r): DIR/groundtruth/{names,authors}, DIR/recommendations_metadata,
# DIR/matching/candidates (B.6a, shared by every variant), DIR/matching/authors/<variant> (B.6b, -V).
#
# Usage:
#   systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% \
#       nice -n 10 bash code/scripts/run_matching_pipeline.sh
#   bash code/scripts/run_matching_pipeline.sh -s authors -V jw097_single -a "--jw_full 0.97 --seniority_definition single"
#
# Options (same letters as in run_gt_pipeline.sh / run_responses_pipeline.sh):
#   -s STEPS  comma-separated: candidates,authors, or all (default)
#   -j N      parallel jobs for the per-bucket stages and the per-model step (default 4)
#   -c N      threads per such job (default 3; 8GB each)
#   -d N      parallel jobs for the fuzzy shards (default 1)
#   -t N      DuckDB threads per fuzzy job (default 12)
#   -m SIZE   DuckDB memory per fuzzy job (default 40GB)
#   -T DIR    DuckDB spill folder (default <project>/tmp/duckdb)
#   -n N      OpenAlex buckets of the A.2 index (default 64)
#   -S N      surname-token shards of the fuzzy rule (default 64)
#   -y FILE   analysis config (default config/analysis.yaml; smoke: config/analysis_smoke.yaml): thresholds of both
#             steps. Missing: in a terminal the wizard (configure_analysis.py --phase early) creates it, otherwise the pipeline stops
#   -f X      override the config: fuzzy floor (pairs with jw_full >= X are stored)
#   -F        override the config: fuzzy rule for ALL recommended names
#   -V NAME   variant of B.6b: output DIR/matching/authors/NAME (default main)
#   -v NAME   analysis variant (configure_variant.py): sets -y and -V (B.6b skipped if the variant changes no
#             B.6b setting: it reads main's matching); the candidates are shared by every variant
#   -M FILE   models file (default data/models/models.txt)
#   -r DIR    results root (default <project>/results)
#   -p ENV    python executable (default: python of the active environment)
#   -l DIR    logs folder (default <project>/logs)
#   -x ARGS   extra arguments for ALL scripts (only common ones, e.g. "--overwrite")
#   -a ARGS   extra arguments for B.6b, e.g. "--jw_full 0.97 --jw_last 0.9 --min_field_share 0.1 --seniority_definition single"
# =============================================================================
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPTS_DIR}/../.." && pwd)"
LOGS_DIR="${PROJECT_ROOT}/logs"
RESULTS_DIR="${PROJECT_ROOT}/results"
MODELS_FILE="${PROJECT_ROOT}/data/models/models.txt"
TEMP_DIR="${PROJECT_ROOT}/tmp/duckdb"

STEP="all"; JOBS=4; PY_THREADS=3; DUCK_JOBS=1; THREADS=12; MEMORY="40GB"; N_BUCKETS=64; N_SHARDS=64; FLOOR=""
FUZZY_ALL=""; CONFIG="${PROJECT_ROOT}/config/analysis.yaml"; VARIANT="main"; EXTRA_ARGS=""; SINGLE_ARGS=""; PYTHON="python"
VARIANT_NAME=""; CONFIG_GIVEN=""
while getopts "s:j:c:d:t:m:T:n:S:f:FV:M:r:p:l:x:a:y:v:h" option; do
    case "${option}" in
        s) STEP="${OPTARG}" ;;
        j) JOBS="${OPTARG}" ;;
        c) PY_THREADS="${OPTARG}" ;;
        d) DUCK_JOBS="${OPTARG}" ;;
        t) THREADS="${OPTARG}" ;;
        m) MEMORY="${OPTARG}" ;;
        T) TEMP_DIR="${OPTARG}" ;;
        n) N_BUCKETS="${OPTARG}" ;;
        S) N_SHARDS="${OPTARG}" ;;
        f) FLOOR="${OPTARG}" ;;
        F) FUZZY_ALL="--fuzzy_all" ;;
        y) CONFIG="${OPTARG}"; CONFIG_GIVEN=1 ;;
        v) VARIANT_NAME="${OPTARG}" ;;
        V) VARIANT="${OPTARG}" ;;
        M) MODELS_FILE="${OPTARG}" ;;
        r) RESULTS_DIR="${OPTARG}" ;;
        p) PYTHON="${OPTARG}" ;;
        l) LOGS_DIR="${OPTARG}" ;;
        x) EXTRA_ARGS="${OPTARG}" ;;
        a) SINGLE_ARGS="${OPTARG}" ;;
        h|*) sed -n '2,42p' "$0"; exit 0 ;;
    esac
done
# -v NAME: analysis variant (configure_variant.py): its config, and the variants whose GT population / author
# matching it reads (its own if it changes their settings, else main's)
VARIANT_POPULATION="main"; VARIANT_AUTHORS="main"; VARIANT_METRICS="main"
if [[ -n "${VARIANT_NAME}" ]]; then
    variant_lines="$("${PYTHON}" "${SCRIPTS_DIR}/configure_variant.py" paths "${VARIANT_NAME}" --results_dir "${RESULTS_DIR}" ${CONFIG_GIVEN:+--config "${CONFIG}"})" || exit 1
    eval "${variant_lines}"
    CONFIG="${VARIANT_CONFIG}"
    echo "[$(date +%H:%M:%S)] variant ${VARIANT_NAME}: config ${CONFIG} | GT population: ${VARIANT_POPULATION} | author matching: ${VARIANT_AUTHORS}"
fi
[[ -n "${VARIANT_NAME}" ]] && VARIANT="${VARIANT_AUTHORS}"
GT_DIR="${RESULTS_DIR}/groundtruth"
RECS_DIR="${RESULTS_DIR}/recommendations_metadata"
CAND_DIR="${RESULTS_DIR}/matching/candidates"
AUTHORS_DIR="${RESULTS_DIR}/matching/authors/${VARIANT}"
ALL_STEPS="candidates authors"
for s in ${STEP//,/ }; do
    if [[ "${s}" != "all" && " ${ALL_STEPS} " != *" ${s} "* ]]; then
        echo "ERROR: unknown step '${s}' in -s ${STEP} (steps: ${ALL_STEPS// /,} or all)"; exit 1
    fi
done
selected() { [[ ",${STEP}," == *",all,"* || ",${STEP}," == *",$1,"* ]]; }
mkdir -p "${LOGS_DIR}"
if ! mkdir -p "${TEMP_DIR}" || [[ ! -w "${TEMP_DIR}" ]]; then
    echo "ERROR: DuckDB spill folder ${TEMP_DIR} cannot be created / written (-T)"; exit 1
fi
SMALL="--threads ${PY_THREADS} --memory_limit 8GB --temp_dir ${TEMP_DIR}"
BIG="--threads ${THREADS} --memory_limit ${MEMORY} --temp_dir ${TEMP_DIR}"
CAND="${PYTHON} ${SCRIPTS_DIR}/batch_match_candidates.py --config ${CONFIG} --n_buckets ${N_BUCKETS} --n_shards ${N_SHARDS}"
CAND="${CAND} ${FLOOR:+--jw_floor ${FLOOR}} ${FUZZY_ALL}"
CAND="${CAND} --recommendations_metadata_dir ${RECS_DIR} --names_dir ${GT_DIR}/names --output_dir ${CAND_DIR} --log_dir ${LOGS_DIR} ${EXTRA_ARGS}"

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

# parallel_stage <name> <jobs> <unit: buckets|shards> <count> <command>: GNU parallel, timestamped joblog
parallel_stage() {
    local name="$1" jobs="$2" unit="$3" count="$4" command="$5"
    local joblog="${LOGS_DIR}/$(date +%Y-%m-%d-%H:%M)_${name}_${unit}.joblog"
    echo "[$(date +%H:%M:%S)] ${name}: ${count} ${unit}, ${jobs} job(s) at a time, joblog ${joblog}"
    if ! parallel -j "${jobs}" --nice 10 --joblog "${joblog}" ${command} --${unit} {} '>' /dev/null \
            ::: $(seq 0 $((count - 1))); then
        echo "[$(date +%H:%M:%S)] ${name}: some ${unit} FAILED (see ${joblog}, column Exitval, and their logs in ${LOGS_DIR})"
        echo "Re-run the same command: finished ${unit} are skipped."
        exit 1
    fi
    echo "[$(date +%H:%M:%S)] ${name}: done"
}
single_stage() {   # single_stage <name> <command>
    echo "[$(date +%H:%M:%S)] $1: start"
    $2 > /dev/null
    echo "[$(date +%H:%M:%S)] $1: done"
}

ensure_config
if selected candidates; then
    single_stage "matching_rec_names" "${CAND} --stage rec_names ${SMALL}"
    parallel_stage "matching_exact" "${JOBS}" buckets "${N_BUCKETS}" "${CAND} --stage exact ${SMALL}"
    single_stage "matching_fuzzy_recs" "${CAND} --stage fuzzy_recs ${SMALL}"
    parallel_stage "matching_oa_tokens" "${JOBS}" buckets "${N_BUCKETS}" "${CAND} --stage oa_tokens ${SMALL}"
    parallel_stage "matching_fuzzy" "${DUCK_JOBS}" shards "${N_SHARDS}" "${CAND} --stage fuzzy ${BIG}"
    parallel_stage "matching_merge" "${JOBS}" buckets "${N_BUCKETS}" "${CAND} --stage merge ${SMALL}"
fi
if selected authors && [[ -n "${VARIANT_NAME}" && "${VARIANT_AUTHORS}" != "${VARIANT_NAME}" ]]; then
    echo "[$(date +%H:%M:%S)] match_authors: skipped, variant ${VARIANT_NAME} changes no B.6b setting (reads ${AUTHORS_DIR})"
elif selected authors; then
    joblog="${LOGS_DIR}/$(date +%Y-%m-%d-%H:%M)_match_authors_models.joblog"
    echo "[$(date +%H:%M:%S)] match_authors (${VARIANT}): models of ${MODELS_FILE}, ${JOBS} job(s) at a time, -> ${AUTHORS_DIR}, joblog ${joblog}"
    if ! parallel -j "${JOBS}" --nice 10 --joblog "${joblog}" \
            "${PYTHON}" "${SCRIPTS_DIR}/batch_match_authors.py" --config "${CONFIG}" --models {} ${SMALL} --candidates_dir "${CAND_DIR}" \
            --recommendations_metadata_dir "${RECS_DIR}" --authors_dir "${GT_DIR}/authors" --output_dir "${AUTHORS_DIR}" \
            --log_dir "${LOGS_DIR}" ${EXTRA_ARGS} ${SINGLE_ARGS} '>' /dev/null :::: "${MODELS_FILE}"; then
        echo "[$(date +%H:%M:%S)] match_authors: some models FAILED (see ${joblog}); re-run the same command (finished models are skipped)"
        exit 1
    fi
    echo "[$(date +%H:%M:%S)] match_authors: done"
fi
