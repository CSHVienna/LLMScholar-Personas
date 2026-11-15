#!/usr/bin/env bash
# =============================================================================
# Everything after the LLM data collection, step by step: ground truth (A.0-A.4) -> response tags (B.1-B.3) ->
# author matching (B.6) -> metrics -> research questions -> notebooks. Each step is one runner call wrapped in
# notify.sh (an email per step: OK / FAILED, duration, last output lines). Stops at the first failed step: the later
# steps do not run. Completed steps are listed in results/_run_all_done.txt and skipped by a later call (no rerun, no
# email); a failed step is not listed, so the next call starts with it. Inside a step, every runner also skips its
# finished outputs (models, buckets, metrics).
#
# Usage (on the shared server: always inside the capped scope):
#   systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=0 -p CPUQuota=1400% nice -n 10 \
#       bash code/scripts/run_all.sh
#   bash code/scripts/run_all.sh                          # again after a failure: continues where it stopped
#   bash code/scripts/run_all.sh -n                       # print the steps (done / to run)
#   bash code/scripts/run_all.sh -s technical -e flows    # only a range of steps
#   bash code/scripts/run_all.sh -f -s rq1 -e rq1         # run a step again although it is listed as done
#
# Options:
#   -s STEP   first step (default: the first)       -e STEP   last step (default: the last)
#   -y FILE   analysis config (default config/analysis.yaml); must be complete (early and late settings)
#   -C DIRS   demographics: reuse raceBERT probabilities of earlier runs (passed to run_gt_pipeline.sh -C)
#   -f        force: run the selected steps even if listed as done
#   -n        dry run: print the steps and whether each is done
# Emails: notify.sh (setup: code/scripts/README.md, "Email when a step ends"); without it the steps still run.
# =============================================================================
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPTS_DIR}/../.." && pwd)"
source "${SCRIPTS_DIR}/_pipeline_lib.sh"      # log
cd "${PROJECT_ROOT}"

CONFIG="config/analysis.yaml"; FIRST=""; LAST=""; CACHE=""; DRY=""; FORCE=""
DONE_FILE="results/_run_all_done.txt"      # one line per completed step: <step> <date time>
while getopts "s:e:y:C:fnh" option; do
    case "${option}" in
        s) FIRST="${OPTARG}" ;;
        e) LAST="${OPTARG}" ;;
        y) CONFIG="${OPTARG}" ;;
        C) CACHE="${OPTARG}" ;;
        f) FORCE=1 ;;
        n) DRY=1 ;;
        h|*) sed -n '2,22p' "$0"; exit 0 ;;
    esac
done

GT="bash code/scripts/run_gt_pipeline.sh"
RESPONSES="bash code/scripts/run_responses_pipeline.sh"
MATCHING="bash code/scripts/run_matching_pipeline.sh -y ${CONFIG}"
METRICS="bash code/scripts/run_metrics_pipeline.sh -y ${CONFIG}"
NOTEBOOK="jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1"
# step name | command (in run order)
STEPS=(
    "gt_authorships|${GT} -s authorships"
    "gt_pairs|${GT} -s pairs"
    "gt_edges|${GT} -s edges"
    "gt_bibliometrics|${GT} -s bibliometrics"
    "gt_names|${GT} -s names"
    "gt_demographics|${GT} -s demographics${CACHE:+ -C ${CACHE}}"
    "gt_population|${GT} -s population"
    "responses_summary|${RESPONSES} -s summary"
    "responses_recommendations|${RESPONSES} -s recommendations"
    "matching_candidates|${MATCHING} -s candidates"
    "matching_authors|${MATCHING} -s authors"
    "technical|${METRICS} -s technical"
    "expected|${METRICS} -s expected"
    "social|${METRICS} -s social"
    "flows|${METRICS} -s flows"
    "rq1|${METRICS} -s rq1"
    "rq2|${METRICS} -s rq2"
    "interactions|${METRICS} -s interactions -j 7 -c 2"
    "notebook_0_gt|${NOTEBOOK} code/notebooks/0_gt.ipynb"
    "notebook_1_tags|${NOTEBOOK} code/notebooks/1_responses_tags.ipynb"
    "notebook_2_matching|${NOTEBOOK} code/notebooks/2_matching_stats.ipynb"
    "notebook_3_metrics|${NOTEBOOK} code/notebooks/3_metrics.ipynb"
    "notebook_4_rqs|${NOTEBOOK} code/notebooks/4_research_questions.ipynb"
    "notebook_5_flows|${NOTEBOOK} code/notebooks/5_flows.ipynb"
)
NAMES=(); for s in "${STEPS[@]}"; do NAMES+=("${s%%|*}"); done
for name in "${FIRST}" "${LAST}"; do
    if [[ -n "${name}" && " ${NAMES[*]} " != *" ${name} "* ]]; then
        echo "ERROR: unknown step '${name}'. Steps: ${NAMES[*]}"; exit 1
    fi
done

if [[ -z "${DRY}" ]] && ! problem="$(python code/scripts/configure_analysis.py --config "${CONFIG}" --check --phase late)"; then
    echo "ERROR: ${CONFIG} is not complete: ${problem}"
    echo "Complete it first: python code/scripts/configure_analysis.py --config ${CONFIG} --phase early (then --phase late)"
    exit 1
fi

is_done() { [[ -f "${DONE_FILE}" ]] && grep -q "^$1 " "${DONE_FILE}"; }

started=""
for s in "${STEPS[@]}"; do
    name="${s%%|*}"; command="${s#*|}"
    [[ -z "${started}" && -n "${FIRST}" && "${name}" != "${FIRST}" ]] && continue
    started=1
    if [[ -n "${DRY}" ]]; then
        printf '%-6s %-26s %s\n' "$(is_done "${name}" && echo done || echo run)" "${name}" "${command}"
    elif is_done "${name}" && [[ -z "${FORCE}" ]]; then
        log "run_all: ${name} done before ($(grep "^${name} " "${DONE_FILE}" | tail -1 | cut -d' ' -f2-)): skipped"
    else
        log "run_all: ${name}"
        if ! bash code/scripts/notify.sh ${command}; then
            log "run_all: ${name} FAILED (later steps not run). Fix it, then run bash code/scripts/run_all.sh again"
            exit 1
        fi
        mkdir -p "$(dirname "${DONE_FILE}")"
        echo "${name} $(date '+%Y-%m-%d %H:%M:%S')" >> "${DONE_FILE}"
    fi
    [[ "${name}" == "${LAST}" ]] && break
done
[[ -n "${DRY}" ]] || log "run_all: done"
