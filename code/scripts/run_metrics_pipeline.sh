#!/usr/bin/env bash
# =============================================================================
# Benchmark metrics (plan §2.3). Works from any working directory.
#   technical   steps 7-8, per model (GNU parallel): configuration metrics (validity, refusals, behaviour /
#               recommendation consistency) and response metrics (duplicates, k balance, factuality)
#   expected    step 9a, one process: expected social metrics under a random recommender per field x seniority x
#               list size (-> DIR/metrics/<-V>/social_expected/), for the size-adjusted metrics <metric>_adj
#   social      step 9, per model (GNU parallel): parity / diversity / unknown share per attribute, popularity,
#               connectedness, similarity on the matched authors (needs the GT: -g, default DIR/groundtruth/population/main)
#   flows       per model (GNU parallel): the people table of the flow figures (valid response x factual person with
#               the prompt variables and the person's GT attributes) -> DIR/metrics/<-V>/flows/<model>.parquet
#   rq1         RQ1, one process per metric (GNU parallel): partial omega^2 + Shapley R^2 of the model and of every
#               configuration variable, cluster-bootstrap CIs over models -> DIR/metrics/<-V>/rq1/<metric>.parquet
#   rq2         RQ2, one process per metric (GNU parallel): EMM and deviation from the grand mean of every value of
#               every configuration variable, cluster-bootstrap CIs over models -> DIR/metrics/<-V>/rq2/<metric>.parquet
#   interactions  batch_rq1_interactions.py, one process per metric (GNU parallel): two-way blocks added to the additive RQ1 model (screening,
#               pre-specified, size / origin decompositions, global check), cluster-bootstrap CIs over models
#               -> DIR/metrics/<-V>/interactions/<metric>.parquet (+ cells/<metric>.parquet)
# Needs B.1-B.3 and B.6 (social and flows need the GT; rq1, rq2 and interactions need technical and social). Valid tag sets come from the analysis config (-y). Folders are derived from the results
# root (-r): DIR/{responses,recommendations}_metadata, DIR/matching/authors/<-w>, output DIR/metrics/<-V>.
#
# Usage:
#   bash code/scripts/run_metrics_pipeline.sh
#   bash code/scripts/run_metrics_pipeline.sh -V robustness/with_duplicates -a "--keep_duplicates"
#   bash code/scripts/run_metrics_pipeline.sh -r tmp/smoke -M tmp/smoke/models.txt -y config/analysis_smoke.yaml -l tmp/smoke/logs
#
# Options (same letters as in the other runners):
#   -s STEPS  comma-separated: technical,expected,social,flows,rq1,rq2,interactions, or all (default)
#   -j N      parallel jobs (models; metrics for rq1 / rq2 / interactions) (default 10)
#   -c N      math-library threads per rq1 / rq2 / interactions job (default 2)
#   -M FILE   models file (default data/models/models.txt)
#   -r DIR    results root (default <project>/results)
#   -V NAME   metrics variant: output DIR/metrics/NAME (default main; robustness checks: robustness/<check>)
#   -w NAME   author-matching variant read: DIR/matching/authors/NAME (default main)
#   -g DIR    A.4 outputs for the social metrics and flows (default DIR/groundtruth/population/main)
#   -v NAME   analysis variant (configure_variant.py): sets -y, -V (metrics/NAME), -w and -g (the variant's own
#             matching / GT population, or main's if it does not change their settings)
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
source "${SCRIPTS_DIR}/_pipeline_lib.sh"      # log, selected, ensure_config
PROJECT_ROOT="$(cd "${SCRIPTS_DIR}/../.." && pwd)"
LOGS_DIR="${PROJECT_ROOT}/logs"
RESULTS_DIR="${PROJECT_ROOT}/results"
MODELS_FILE="${PROJECT_ROOT}/data/models/models.txt"
CONFIG="${PROJECT_ROOT}/config/analysis.yaml"

GT_DIR=""; STEP="all"; JOBS=10; PY_THREADS=2; VARIANT="main"; MATCHING_VARIANT="main"; PYTHON="python"; EXTRA_ARGS=""; SINGLE_ARGS=""
RQ_ARGS=""
VARIANT_NAME=""; CONFIG_GIVEN=""
while getopts "s:j:c:M:r:V:w:g:y:v:p:l:x:a:q:h" option; do
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
        y) CONFIG="${OPTARG}"; CONFIG_GIVEN=1 ;;
        v) VARIANT_NAME="${OPTARG}" ;;
        p) PYTHON="${OPTARG}" ;;
        l) LOGS_DIR="${OPTARG}" ;;
        x) EXTRA_ARGS="${OPTARG}" ;;
        a) SINGLE_ARGS="${OPTARG}" ;;
        h|*) sed -n '2,45p' "$0"; exit 0 ;;
    esac
done
ALL_STEPS="technical expected social flows rq1 rq2 interactions"
for s in ${STEP//,/ }; do
    if [[ "${s}" != "all" && " ${ALL_STEPS} " != *" ${s} "* ]]; then
        echo "ERROR: unknown step '${s}' in -s ${STEP} (steps: ${ALL_STEPS// /,} or all)"; exit 1
    fi
done
mkdir -p "${LOGS_DIR}"
# -v NAME: analysis variant (configure_variant.py): its config, and the variants whose GT population / author
# matching it reads (its own if it changes their settings, else main's)
VARIANT_POPULATION="main"; VARIANT_AUTHORS="main"; VARIANT_METRICS="main"
if [[ -n "${VARIANT_NAME}" ]]; then
    variant_lines="$("${PYTHON}" "${SCRIPTS_DIR}/configure_variant.py" paths "${VARIANT_NAME}" --results_dir "${RESULTS_DIR}" ${CONFIG_GIVEN:+--config "${CONFIG}"})" || exit 1
    eval "${variant_lines}"
    CONFIG="${VARIANT_CONFIG}"
    log "variant ${VARIANT_NAME}: config ${CONFIG} | GT population: ${VARIANT_POPULATION} | author matching: ${VARIANT_AUTHORS}"
fi
if [[ -n "${VARIANT_NAME}" ]]; then VARIANT="${VARIANT_METRICS}"; MATCHING_VARIANT="${VARIANT_AUTHORS}"; fi
METRICS_DIR="${RESULTS_DIR}/metrics/${VARIANT}"
GT_DIR="${GT_DIR:-${RESULTS_DIR}/groundtruth/population/${VARIANT_POPULATION}}"


per_model() {   # per_model <label> <script> [args]: the script once per model of MODELS_FILE (JOBS at a time) on
                # the per-model inputs, -> METRICS_DIR; stops the pipeline if a model fails
    local label="$1" script="$2"; shift 2
    local stem="${script%.py}"
    local joblog="${LOGS_DIR}/$(date +%Y-%m-%d-%H:%M)_${stem#batch_}_models.joblog"   # e.g. ..._social_metrics_models
    log "${label} (${VARIANT}): models of ${MODELS_FILE}, ${JOBS} job(s) at a time, -> ${METRICS_DIR}, joblog ${joblog}"
    if ! parallel -j "${JOBS}" --nice 10 --joblog "${joblog}" \
            "${PYTHON}" "${SCRIPTS_DIR}/${script}" --config "${CONFIG}" --models {} \
            --responses_metadata_dir "${RESULTS_DIR}/responses_metadata" \
            --recommendations_metadata_dir "${RESULTS_DIR}/recommendations_metadata" \
            --matching_dir "${RESULTS_DIR}/matching/authors/${MATCHING_VARIANT}" "$@" \
            --output_dir "${METRICS_DIR}" --log_dir "${LOGS_DIR}" ${EXTRA_ARGS} ${SINGLE_ARGS} '>' /dev/null :::: "${MODELS_FILE}"; then
        log "${label}: some models FAILED (see ${joblog}); re-run the same command (finished models are skipped)"
        exit 1
    fi
    log "${label}: done"
}

ensure_config
if selected technical; then per_model metrics batch_metrics.py; fi
if selected expected; then
    log "social_expected (${VARIANT}): GT ${GT_DIR} -> ${METRICS_DIR}/social_expected"
    if ! nice -n 10 "${PYTHON}" "${SCRIPTS_DIR}/batch_social_expected.py" --config "${CONFIG}" --gt_dir "${GT_DIR}" \
            --output_dir "${METRICS_DIR}" --log_dir "${LOGS_DIR}" ${EXTRA_ARGS} > /dev/null; then
        log "social_expected: FAILED (see its log in ${LOGS_DIR})"
        exit 1
    fi
    log "social_expected: done"
fi
if selected social; then per_model "social metrics" batch_social_metrics.py --gt_dir "${GT_DIR}"; fi
if selected flows; then per_model flows batch_flows.py --gt_dir "${GT_DIR}"; fi
if selected rq1 || selected rq2 || selected interactions; then   # the RQs need the late settings (valid tags, thresholds) reviewed
    if ! problem="$("${PYTHON}" "${SCRIPTS_DIR}/configure_analysis.py" --config "${CONFIG}" --check --phase late)"; then
        echo "STOP before the research questions: ${problem}"
        exit 1
    fi
fi
for step_script in rq1:batch_rq1.py rq2:batch_rq2.py interactions:batch_rq1_interactions.py; do   # step:script
    rq="${step_script%%:*}"; script="${step_script#*:}"
    selected "${rq}" || continue
    read -r -a RQ_METRICS <<< "$("${PYTHON}" "${SCRIPTS_DIR}/${script}" --list_metrics --config "${CONFIG}" ${RQ_ARGS})"   # metric sets: study.ethnicity_attribute
    joblog="${LOGS_DIR}/$(date +%Y-%m-%d-%H:%M)_${rq}_metrics.joblog"
    log "${rq} (${VARIANT}): ${#RQ_METRICS[@]} metrics, ${JOBS} job(s) x ${PY_THREADS} threads, -> ${METRICS_DIR}/${rq}, joblog ${joblog}"
    if ! OMP_NUM_THREADS="${PY_THREADS}" OPENBLAS_NUM_THREADS="${PY_THREADS}" MKL_NUM_THREADS="${PY_THREADS}" \
            parallel -j "${JOBS}" --nice 10 --joblog "${joblog}" \
            "${PYTHON}" "${SCRIPTS_DIR}/${script}" --metrics_dir "${METRICS_DIR}" ${RQ_ARGS} --metrics {} \
            --log_dir "${LOGS_DIR}" ${EXTRA_ARGS} '>' /dev/null ::: "${RQ_METRICS[@]}"; then
        log "${rq}: some metrics FAILED (see ${joblog}); re-run the same command (finished metrics are skipped)"
        exit 1
    fi
    log "${rq}: done"
done
