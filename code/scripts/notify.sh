#!/usr/bin/env bash
# =============================================================================
# Run any command and email when it ends (OK / FAILED, duration, last lines of its output). The command runs as
# usual: same output on screen, same exit code. Nothing else in the pipeline changes.
#
# Usage:
#   bash code/scripts/notify.sh <command ...>
#   bash code/scripts/notify.sh bash code/scripts/run_metrics_pipeline.sh -s technical
#   bash code/scripts/notify.sh python code/scripts/batch_eval_responses.py --models_file data/models/models.txt
#
# Setup (once; address and password stay in the conda env, like the API keys): code/scripts/README.md,
# section "Email when a step ends".
#
# Environment:
#   NOTIFY_EMAIL          recipient (default NOTIFY_SMTP_USER; neither set: the command still runs, no email)
#   NOTIFY_SMTP_HOST      mail server to send through (e.g. smtp.gmail.com); unset: the system `mail` command
#   NOTIFY_SMTP_PORT      default 587 (STARTTLS)
#   NOTIFY_SMTP_USER      login and sender address
#   NOTIFY_SMTP_PASSWORD  app password (read from the environment only, never put on a command line)
#   NOTIFY_LINES          last output lines in the email (default 30)
#   NOTIFY_DRY_RUN        1: print the email instead of sending it (to test)
# A failed send never changes the command's exit code.
# =============================================================================
set -uo pipefail

if [[ $# -eq 0 ]]; then
    sed -n '2,21p' "$0"; exit 2
fi

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPTS_DIR}/../.." && pwd)"
LINES="${NOTIFY_LINES:-30}"
mkdir -p "${PROJECT_ROOT}/tmp"
OUTPUT="$(mktemp "${PROJECT_ROOT}/tmp/notify.XXXXXX.log")"

start=$(date +%s)
started="$(date '+%Y-%m-%d %H:%M:%S')"
"$@" 2>&1 | tee "${OUTPUT}"
status=${PIPESTATUS[0]}
elapsed=$(( $(date +%s) - start ))
duration="$(printf '%dh %02dm %02ds' $((elapsed / 3600)) $((elapsed % 3600 / 60)) $((elapsed % 60)))"

state=$([[ ${status} -eq 0 ]] && echo OK || echo "FAILED (exit ${status})")
command_text="$*"
subject="[LLMScholar] ${state}: ${command_text:0:80} (${duration})"
body="$(printf 'Command:  %s\nHost:     %s\nFolder:   %s\nStarted:  %s\nDuration: %s\nResult:   %s\n\nLast %s lines of output:\n\n%s\n' \
    "${command_text}" "$(hostname)" "$(pwd)" "${started}" "${duration}" "${state}" "${LINES}" "$(tail -n "${LINES}" "${OUTPUT}")")"

recipient="${NOTIFY_EMAIL:-${NOTIFY_SMTP_USER:-}}"
send_smtp() {   # body on stdin; subject / recipient / login from the environment (no secret on the command line)
    NOTIFY_SUBJECT="${subject}" NOTIFY_TO="${recipient}" python3 -c '
import os, smtplib, sys
from email.message import EmailMessage
message = EmailMessage()
message["Subject"], message["From"], message["To"] = os.environ["NOTIFY_SUBJECT"], os.environ["NOTIFY_SMTP_USER"], os.environ["NOTIFY_TO"]
message.set_content(sys.stdin.read())
try:
    with smtplib.SMTP(os.environ["NOTIFY_SMTP_HOST"], int(os.environ.get("NOTIFY_SMTP_PORT", "587")), timeout=30) as server:
        server.starttls()
        server.login(os.environ["NOTIFY_SMTP_USER"], os.environ["NOTIFY_SMTP_PASSWORD"])
        server.send_message(message)
except Exception as error:
    sys.exit(f"[notify] {type(error).__name__}: {error}")
'
}

if [[ "${NOTIFY_DRY_RUN:-0}" == "1" ]]; then
    printf '\n----- email (dry run) -----\nTo: %s\nVia: %s\nSubject: %s\n\n%s\n' "${recipient:-<NOTIFY_EMAIL not set>}" \
        "${NOTIFY_SMTP_HOST:-system mail}" "${subject}" "${body}"
elif [[ -z "${recipient}" ]]; then
    echo "[notify] NOTIFY_EMAIL is not set: no email sent" >&2
elif [[ -n "${NOTIFY_SMTP_HOST:-}" ]]; then
    if printf '%s\n' "${body}" | send_smtp; then
        echo "[notify] email sent to ${recipient} via ${NOTIFY_SMTP_HOST}" >&2
    else
        echo "[notify] sending the email via ${NOTIFY_SMTP_HOST} failed (the command's result is unchanged)" >&2
    fi
elif printf '%s\n' "${body}" | mail -s "${subject}" "${recipient}"; then
    echo "[notify] handed to the system mail (this server delivers locally only: set NOTIFY_SMTP_HOST, see README)" >&2
else
    echo "[notify] sending the email failed (the command's result is unchanged)" >&2
fi
rm -f "${OUTPUT}"
exit "${status}"
