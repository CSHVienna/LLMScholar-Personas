#!/usr/bin/env bash
# =============================================================================
# Check the latest runs of batch_eval_responses.py (B.2) and batch_eval_recommendations.py (B.3).
# Read-only. Works from any working directory.
#
# For every model of the models file it takes the NEWEST log of each step (only files directly in
# the logs folder; sub-folders such as logs/2025/ are archives and ignored; newest = latest timestamp
# in the file name) and reports:
#   1. status per step:  OK | FAILED | UNFINISHED (no "Done" line: still running or killed) | MISSING
#   2. order: the recommendations log must not be older than the responses log of the same model
#      (otherwise B.3 used outdated response tags)
#   3. stored tags: every tag in results/{responses,recommendations}_metadata/<model>.parquet must be
#      a current tag name (catches files written before a tag was renamed or removed)
# Exits with code 1 if anything is wrong, and prints the models to re-run.
#
# Usage:
#   bash code/scripts/check_eval_runs.sh
#   bash code/scripts/check_eval_runs.sh -M my_models.txt -l logs -r results -p python
#   bash code/scripts/check_eval_runs.sh -v          # one line per model (default: problems only)
# =============================================================================
set -uo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPTS_DIR}/../.." && pwd)"
MODELS_FILE="${PROJECT_ROOT}/data/models/models.txt"
LOGS_DIR="${PROJECT_ROOT}/logs"
RESULTS_DIR="${PROJECT_ROOT}/results"
PYTHON="python"
VERBOSE=0
while getopts "M:l:r:p:vh" option; do
    case "${option}" in
        M) MODELS_FILE="${OPTARG}" ;;
        l) LOGS_DIR="${OPTARG}" ;;
        r) RESULTS_DIR="${OPTARG}" ;;
        p) PYTHON="${OPTARG}" ;;
        v) VERBOSE=1 ;;
        h|*) sed -n '2,21p' "$0"; exit 0 ;;
    esac
done

# Log names written by the scripts: <YYYY-mm-dd-HH:MM>_<name>_<model>.log
STEP_RESPONSES="responses_summary"     # batch_eval_responses.py
STEP_RECOMMENDATIONS="recommendations" # batch_eval_recommendations.py

# newest_log <name> <model>: path of the newest log (file names start with a sortable timestamp); also logs of runs
# with a parameter suffix, e.g. <time>_<name>_<model>_limit20.log (--limit_prompts) or ..._english.log (--languages)
newest_log() {
    find "${LOGS_DIR}" -maxdepth 1 -type f \( -name "*_${1}_${2}.log" -o -name "*_${1}_${2}_*.log" \) -printf '%f\t%p\n' \
        2>/dev/null | sort | tail -n 1 | cut -f2
}
# status <log>: OK | FAILED | UNFINISHED | MISSING
status() {
    [[ -z "$1" ]] && { echo MISSING; return; }
    local done_line; done_line="$(grep -a 'Done in' "$1" | tail -n 1)"
    if [[ -z "${done_line}" ]]; then echo UNFINISHED
    elif [[ "${done_line}" == *"failed: none"* ]]; then echo OK
    else echo FAILED; fi
}
stamp() { [[ -n "$1" ]] && basename "$1" | cut -c1-16 || echo "-"; }

# --- 1 + 2. Logs ---------------------------------------------------------------
mapfile -t MODELS < <(grep -v '^\s*$' "${MODELS_FILE}" | grep -v '^\s*#' | tr -d '\r')
echo "Latest runs of ${#MODELS[@]} models (logs: ${LOGS_DIR})"
declare -a RERUN_RESPONSES=() RERUN_RECOMMENDATIONS=()
newest_resp="" newest_rec=""
for model in "${MODELS[@]}"; do
    resp_log="$(newest_log "${STEP_RESPONSES}" "${model}")"
    rec_log="$(newest_log "${STEP_RECOMMENDATIONS}" "${model}")"
    resp_status="$(status "${resp_log}")"; rec_status="$(status "${rec_log}")"
    resp_time="$(stamp "${resp_log}")"; rec_time="$(stamp "${rec_log}")"
    [[ "${resp_time}" > "${newest_resp}" && "${resp_time}" != "-" ]] && newest_resp="${resp_time}"
    [[ "${rec_time}" > "${newest_rec}" && "${rec_time}" != "-" ]] && newest_rec="${rec_time}"
    problem=""
    [[ "${resp_status}" != OK ]] && { problem+=" responses ${resp_status};"; RERUN_RESPONSES+=("${model}"); }
    if [[ "${rec_status}" != OK ]]; then
        problem+=" recommendations ${rec_status};"; RERUN_RECOMMENDATIONS+=("${model}")
    elif [[ "${resp_status}" == OK && "${rec_time}" < "${resp_time}" ]]; then
        problem+=" recommendations older than responses;"; RERUN_RECOMMENDATIONS+=("${model}")
    fi
    if [[ -n "${problem}" ]]; then
        printf '  PROBLEM  %-45s responses %s %-10s recommendations %s %-10s ->%s\n' \
            "${model}" "${resp_time}" "${resp_status}" "${rec_time}" "${rec_status}" "${problem}"
    elif (( VERBOSE )); then
        printf '  ok       %-45s responses %s  recommendations %s\n' "${model}" "${resp_time}" "${rec_time}"
    fi
done
echo "Newest logs: responses ${newest_resp:--}, recommendations ${newest_rec:--}"

# --- 3. Stored tags are current tag names -----------------------------------------
echo "Stored tags (${RESULTS_DIR}):"
STALE="$(PYTHONPATH="${PROJECT_ROOT}/code/libs${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" - "${RESULTS_DIR}" "${MODELS[@]}" <<'EOF'
import sys
from pathlib import Path
import pyarrow.parquet as pq
from utils import constants as cons

results, models = Path(sys.argv[1]), sys.argv[2:]
checks = [('responses', 'responses_metadata', cons.COL_RESPONSE_TAGS, set(cons.RESPONSE_TAGS)),
          ('recommendations', 'recommendations_metadata', cons.COL_REC_TAGS, set(cons.REC_TAGS))]
for step, folder, column, vocabulary in checks:
    bad = 0
    for model in models:
        path = results / folder / f"{model}.parquet"
        if not path.exists():
            print(f"{step}\t{model}\tno output file"); bad += 1; continue
        values = pq.read_table(path, columns=[column]).column(column).unique().to_pylist()
        unknown = sorted({t for v in values if v for t in v.split(cons.TAG_SEPARATOR)} - vocabulary)
        if unknown:
            print(f"{step}\t{model}\toutdated tags: {', '.join(unknown)}"); bad += 1
    print(f"#\t{step}: {len(models) - bad} of {len(models)} files with current tags", file=sys.stderr)
EOF
)"
while IFS=$'\t' read -r step model reason; do
    [[ -z "${step}" ]] && continue
    printf '  PROBLEM  %-45s %s %s\n' "${model}" "${step}:" "${reason}"
    if [[ "${step}" == responses ]]; then RERUN_RESPONSES+=("${model}"); else RERUN_RECOMMENDATIONS+=("${model}"); fi
done <<< "${STALE}"

# --- Summary ---------------------------------------------------------------------
unique() { printf '%s\n' "$@" | awk 'NF && !seen[$0]++'; }
mapfile -t RERUN_RESPONSES < <(unique "${RERUN_RESPONSES[@]+"${RERUN_RESPONSES[@]}"}")
mapfile -t RERUN_RECOMMENDATIONS < <(unique "${RERUN_RECOMMENDATIONS[@]+"${RERUN_RECOMMENDATIONS[@]}"}" "${RERUN_RESPONSES[@]+"${RERUN_RESPONSES[@]}"}")
if (( ${#RERUN_RESPONSES[@]} == 0 && ${#RERUN_RECOMMENDATIONS[@]} == 0 )); then
    echo "ALL OK: both steps finished for every model, in the right order, with current tags."
    exit 0
fi
echo
echo "To fix (re-running responses means re-running recommendations for the same models afterwards):"
if (( ${#RERUN_RESPONSES[@]} )); then
    echo "  responses (${#RERUN_RESPONSES[@]} models):"
    echo "    parallel -j 10 --nice 10 python code/scripts/batch_eval_responses.py --models {} --overwrite ::: ${RERUN_RESPONSES[*]}"
fi
if (( ${#RERUN_RECOMMENDATIONS[@]} )); then
    echo "  recommendations (${#RERUN_RECOMMENDATIONS[@]} models), after the responses:"
    echo "    parallel -j 10 --nice 10 python code/scripts/batch_eval_recommendations.py --models {} --overwrite ::: ${RERUN_RECOMMENDATIONS[*]}"
fi
exit 1
