# Shared by the run_*_pipeline.sh runners (sourced right after SCRIPTS_DIR is set). The functions read the runner's
# variables when called: STEP (-s), PYTHON (-p), SCRIPTS_DIR, CONFIG (-y).

log() { echo "[$(date +%H:%M:%S)] $*"; }                                   # a timestamped progress line

selected() { [[ ",${STEP}," == *",all,"* || ",${STEP}," == *",$1,"* ]]; }   # selected <step>: is it in -s?

ensure_config() {   # the config must exist and be complete (settings added later): else the wizard, in a terminal
    local problem
    if problem="$("${PYTHON}" "${SCRIPTS_DIR}/configure_analysis.py" --config "${CONFIG}" --check)"; then
        log "analysis config: ${CONFIG}"
    elif [[ -t 0 && -t 1 ]]; then
        log "${problem}"
        log "starting the wizard"
        "${PYTHON}" "${SCRIPTS_DIR}/configure_analysis.py" --config "${CONFIG}" --phase early || { echo "No config written: stopping."; exit 1; }
    else
        echo "ERROR: ${problem}"
        exit 1
    fi
}
