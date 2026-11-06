#!/usr/bin/env bash
# =============================================================================
# Benchmark metrics (plan §2.3). Works from any working directory.
#   technical   steps 7-8, per model (GNU parallel): configuration metrics (validity, refusals, behaviour /
#               recommendation consistency) and response metrics (duplicates, k ratio, factuality)
#   social      step 9, per model (GNU parallel): parity / diversity / unknown share per attribute, popularity,
#               connectedness, similarity on the matched authors (needs the GT: -g, default DIR/groundtruth)
#   rq1         RQ1, one process per metric (GNU parallel): partial omega^2 + Shapley R^2 of the model and of every
#               configuration variable, cluster-bootstrap CIs over models -> DIR/metrics/<-V>/rq1/<metric>.parquet
#   rq2         RQ2, one process per metric (GNU parallel): EMM and deviation from the grand mean of every value of
#               every configuration variable, cluster-bootstrap CIs over models -> DIR/metrics/<-V>/rq2/<metric>.parquet
# Needs B.1-B.3 and B.6 (social needs the GT; rq1 and rq2 need technical and social). Valid tag sets come from the analysis config (-y). Folders are derived from the results
# root (-r): DIR/{responses,recommendations}_metadata, DIR/matching/authors/<-w>, output DIR/metrics/<-V>.
#
# Usage:
#   bash code/scripts/run_metrics_pipeline.sh
#   bash code/scripts/run_metrics_pipeline.sh -V robustness/with_duplicates -a "--keep_duplicates"
#   bash code/scripts/run_metrics_pipeline.sh -r tmp/smoke -M tmp/smoke/models.txt -y config/analysis_smoke.yaml -l tmp/smoke/logs
#
# Options (same letters as in the other runners):
#   -s STEPS  comma-separated: technical,social,rq1,rq2, or all (default)
#   -j N      parallel jobs (models; metrics for rq1 / rq2) (default 10)
#   -c N      math-library threads per rq1 / rq2 job (default 2)
#   -M FILE   models file (default data/models/models.txt)
#   -r DIR    results root (default <project>/results)
#   -V NAME   metrics variant: output DIR/metrics/NAME (default main; robustness checks: robustness/<check>)
#   -w NAME   author-matching variant read: DIR/matching/authors/NAME (default main)
#   -g DIR    ground-truth folder for the social metrics (default DIR/groundtruth)
#   -y FILE   analysis config (default config/analysis.yaml; smoke: config/analysis_smoke.yaml). Missing: in a
#             terminal the wizard (configure_analysis.py --phase early) creates it, otherwise the pipeline stops
#   -p ENV    python executable (default: python of the active environment)
#   -l DIR    logs folder (default <project>/logs)
#   -x ARGS   extra arguments for ALL scripts (only common ones, e.g. "--overwrite")
#   -a ARGS   extra arguments for the metrics script, e.g. "--keep_duplicates" / "--order_sensitive_keys"
#   -q ARGS   extra arguments for the research-question scripts, e.g. "--n_boot 200" / "--metrics factuality"
# =============================================================================
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPTS_DIR}/../.." && pwd)"
LOGS_DIR="${PROJECT_ROOT}/logs"
RESULTS_DIR="${PROJECT_ROOT}/results"
MODELS_FILE="${PROJECT_ROOT}/data/models/models.txt"
CONFIG="${PROJECT_ROOT}/config/analysis.yaml"

GT_DIR=""; STEP="all"; JOBS=10; PY_THREADS=2; VARIANT="main"; MATCHING_VARIANT="main"; PYTHON="python"; EXTRA_ARGS=""; SINGLE_ARGS=""
RQ_ARGS=""
while getopts "s:j:c:M:r:V:w:g:y:p:l:x:a:q:h" option; do
    case "${option}" in
        s) STEP="${OPTARG}" ;;
        j) JOBS="${OPTARG}" ;;
        c) PY_THREADS="${OPTARG}" ;;
        q) RQ_ARGS="${OPTARG}" ;;
        M) MODELS_FILE="${OPTARG}" ;;
        r) RESULTS_DIR="${OPTARG}" ;;
        V) VARIANT="${OPTARG}" ;;
        w) MATCHING_VARIANT="${OPTARG}" ;;
        g) GT_DIR="${OPTARG}" ;;
        y) CONFIG="${OPTARG}" ;;
        p) PYTHON="${OPTARG}" ;;
        l) LOGS_DIR="${OPTARG}" ;;
        x) EXTRA_ARGS="${OPTARG}" ;;
        a) SINGLE_ARGS="${OPTARG}" ;;
        h|*) sed -n '2,36p' "$0"; exit 0 ;;
    esac
done
ALL_STEPS="technical social rq1 rq2"
for s in ${STEP//,/ }; do
    if [[ "${s}" != "all" && " ${ALL_STEPS} " != *" ${s} "* ]]; then
        echo "ERROR: unknown step '${s}' in -s ${STEP} (steps: ${ALL_STEPS// /,} or all)"; exit 1
    fi
done
selected() { [[ ",${STEP}," == *",all,"* || ",${STEP}," == *",$1,"* ]]; }
mkdir -p "${LOGS_DIR}"
METRICS_DIR="${RESULTS_DIR}/metrics/${VARIANT}"
GT_DIR="${GT_DIR:-${RESULTS_DIR}/groundtruth}"

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

ensure_config
if selected technical; then
    joblog="${LOGS_DIR}/$(date +%Y-%m-%d-%H:%M)_metrics_models.joblog"
    echo "[$(date +%H:%M:%S)] metrics (${VARIANT}): models of ${MODELS_FILE}, ${JOBS} job(s) at a time, -> ${METRICS_DIR}, joblog ${joblog}"
    if ! parallel -j "${JOBS}" --nice 10 --joblog "${joblog}" \
            "${PYTHON}" "${SCRIPTS_DIR}/batch_metrics.py" --config "${CONFIG}" --models {} \
            --responses_metadata_dir "${RESULTS_DIR}/responses_metadata" \
            --recommendations_metadata_dir "${RESULTS_DIR}/recommendations_metadata" \
            --matching_dir "${RESULTS_DIR}/matching/authors/${MATCHING_VARIANT}" --output_dir "${METRICS_DIR}" \
            --log_dir "${LOGS_DIR}" ${EXTRA_ARGS} ${SINGLE_ARGS} '>' /dev/null :::: "${MODELS_FILE}"; then
        echo "[$(date +%H:%M:%S)] metrics: some models FAILED (see ${joblog}); re-run the same command (finished models are skipped)"
        exit 1
    fi
    echo "[$(date +%H:%M:%S)] metrics: done"
fi
if selected social; then
    joblog="${LOGS_DIR}/$(date +%Y-%m-%d-%H:%M)_social_metrics_models.joblog"
    echo "[$(date +%H:%M:%S)] social metrics (${VARIANT}): models of ${MODELS_FILE}, ${JOBS} job(s), GT ${GT_DIR}, joblog ${joblog}"
    if ! parallel -j "${JOBS}" --nice 10 --joblog "${joblog}" \
            "${PYTHON}" "${SCRIPTS_DIR}/batch_social_metrics.py" --config "${CONFIG}" --models {} \
            --responses_metadata_dir "${RESULTS_DIR}/responses_metadata" \
            --recommendations_metadata_dir "${RESULTS_DIR}/recommendations_metadata" \
            --matching_dir "${RESULTS_DIR}/matching/authors/${MATCHING_VARIANT}" --gt_dir "${GT_DIR}" \
            --output_dir "${METRICS_DIR}" --log_dir "${LOGS_DIR}" ${EXTRA_ARGS} ${SINGLE_ARGS} '>' /dev/null :::: "${MODELS_FILE}"; then
        echo "[$(date +%H:%M:%S)] social metrics: some models FAILED (see ${joblog}); re-run the same command (finished models are skipped)"
        exit 1
    fi
    echo "[$(date +%H:%M:%S)] social metrics: done"
fi
if selected rq1 || selected rq2; then   # the RQs need the late settings (valid tags, thresholds) reviewed
    if ! problem="$("${PYTHON}" "${SCRIPTS_DIR}/configure_analysis.py" --config "${CONFIG}" --check --phase late)"; then
        echo "STOP before the research questions: ${problem}"
        exit 1
    fi
fi
for rq in rq1 rq2; do
    selected "${rq}" || continue
    read -r -a RQ_METRICS <<< "$("${PYTHON}" "${SCRIPTS_DIR}/batch_${rq}.py" --list_metrics ${RQ_ARGS})"
    joblog="${LOGS_DIR}/$(date +%Y-%m-%d-%H:%M)_${rq}_metrics.joblog"
    echo "[$(date +%H:%M:%S)] ${rq} (${VARIANT}): ${#RQ_METRICS[@]} metrics, ${JOBS} job(s) x ${PY_THREADS} threads, -> ${METRICS_DIR}/${rq}, joblog ${joblog}"
    if ! OMP_NUM_THREADS="${PY_THREADS}" OPENBLAS_NUM_THREADS="${PY_THREADS}" MKL_NUM_THREADS="${PY_THREADS}" \
            parallel -j "${JOBS}" --nice 10 --joblog "${joblog}" \
            "${PYTHON}" "${SCRIPTS_DIR}/batch_${rq}.py" --metrics_dir "${METRICS_DIR}" ${RQ_ARGS} --metrics {} \
            --log_dir "${LOGS_DIR}" ${EXTRA_ARGS} '>' /dev/null ::: "${RQ_METRICS[@]}"; then
        echo "[$(date +%H:%M:%S)] ${rq}: some metrics FAILED (see ${joblog}); re-run the same command (finished metrics are skipped)"
        exit 1
    fi
    echo "[$(date +%H:%M:%S)] ${rq}: done"
done
