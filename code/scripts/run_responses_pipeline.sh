#!/usr/bin/env bash
# =============================================================================
# Steps B.1-B.3: response summary -> recommendation tagging,
# one process per model with GNU parallel. Works from any working directory.
#
# Each step writes a GNU parallel joblog named like the script logs:
#   logs/<YYYY-mm-dd-HH:MM>_<step>_<params>.joblog
# and every model writes its own log (logs/<timestamp>_<step>_<model>.log).
# If a step has failed models, the pipeline stops and prints the command that
# re-runs only the failed ones.
#
# Usage:
#   bash code/scripts/run_responses_pipeline.sh                       # all steps, all models, 10 jobs
#   bash code/scripts/run_responses_pipeline.sh -j 20 -s summary      # one step
#   bash code/scripts/run_responses_pipeline.sh -M my_models.txt -x "--overwrite"
#   bash code/scripts/run_responses_pipeline.sh -r tmp/smoke -l tmp/smoke/logs -x "--limit_prompts 5"
#
# Options (same letters as in run_gt_pipeline.sh / run_matching_pipeline.sh):
#   -j N      parallel jobs (default 10)
#   -M FILE   models file (default data/models/models.txt)
#   -s STEP   summary | recommendations | all (default all)
#   -p ENV    python executable (default: python of the active environment)
#   -r DIR    results root (default <project>/results): DIR/responses_metadata, DIR/recommendations_metadata
#   -l DIR    logs folder (default <project>/logs)
#   -x ARGS   extra arguments passed to the per-model scripts (e.g. "--overwrite --limit_prompts 5")
# =============================================================================
set -euo pipefail

# --- Locations (independent of the working directory) ------------------------
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPTS_DIR}/../.." && pwd)"
LOGS_DIR="${PROJECT_ROOT}/logs"
RESULTS_DIR="${PROJECT_ROOT}/results"

# --- Options ----------------------------------------------------------------
JOBS=10
MODELS_FILE="${PROJECT_ROOT}/data/models/models.txt"
STEP="all"
PYTHON="python"
EXTRA_ARGS=""
while getopts "j:M:s:p:r:l:x:h" option; do
    case "${option}" in
        j) JOBS="${OPTARG}" ;;
        M) MODELS_FILE="${OPTARG}" ;;
        s) STEP="${OPTARG}" ;;
        p) PYTHON="${OPTARG}" ;;
        r) RESULTS_DIR="${OPTARG}" ;;
        l) LOGS_DIR="${OPTARG}" ;;
        x) EXTRA_ARGS="${OPTARG}" ;;
        h|*) sed -n '2,27p' "$0"; exit 0 ;;
    esac
done
mkdir -p "${LOGS_DIR}"
PARAMS="$(basename "${MODELS_FILE%.*}")"   # e.g. 'models' for models.txt
RESPONSES_METADATA_DIR="${RESULTS_DIR}/responses_metadata"
RECOMMENDATIONS_METADATA_DIR="${RESULTS_DIR}/recommendations_metadata"

# --- Helpers ----------------------------------------------------------------
# run_per_model <step name> <script> <script args>: one GNU parallel job per model, timestamped joblog
run_per_model() {
    local name="$1" script="$2" args="$3 --log_dir ${LOGS_DIR} ${EXTRA_ARGS}"
    local joblog="${LOGS_DIR}/$(date +%Y-%m-%d-%H:%M)_${name}_${PARAMS}.joblog"
    echo "[$(date +%H:%M:%S)] ${name}: ${JOBS} jobs, joblog ${joblog}"
    if ! parallel -j "${JOBS}" --nice 10 --joblog "${joblog}" \
            "${PYTHON}" "${SCRIPTS_DIR}/${script}" --models {} ${args} '>' /dev/null \
            :::: "${MODELS_FILE}"; then
        echo "[$(date +%H:%M:%S)] ${name}: some models FAILED (see ${joblog}, column Exitval, and their logs in ${LOGS_DIR})"
        echo "Re-run only the failed ones with:"
        echo "  parallel --resume-failed -j ${JOBS} --nice 10 --joblog ${joblog} ${PYTHON} ${SCRIPTS_DIR}/${script} --models {} ${args} '>' /dev/null :::: ${MODELS_FILE}"
        exit 1
    fi
    echo "[$(date +%H:%M:%S)] ${name}: done"
}

# --- Steps --------------------------------------------------------------------
if [[ "${STEP}" == "all" || "${STEP}" == "summary" ]]; then
    run_per_model "responses_summary" "batch_eval_responses.py" "--output_dir ${RESPONSES_METADATA_DIR}"
fi
if [[ "${STEP}" == "all" || "${STEP}" == "recommendations" ]]; then
    run_per_model "recommendations" "batch_eval_recommendations.py" \
        "--responses_metadata_dir ${RESPONSES_METADATA_DIR} --output_dir ${RECOMMENDATIONS_METADATA_DIR}"
fi
