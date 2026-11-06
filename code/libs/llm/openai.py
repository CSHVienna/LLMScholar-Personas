"""
OpenAI: single calls (``prompt_gpt``, used by batch_input_params.py) and the Batch API (batch_collect_cloud_*.py).

Batch API: one JSONL file of requests -> uploaded -> one batch job -> one output file, one JSON line per
answer, ``custom_id`` = prompt id. The stored answer is that whole line (read by ``responses.sources.GptAdapter``).
"""

from pathlib import Path
from typing import List, Tuple

from openai import OpenAI

BATCH_ENDPOINT = "/v1/chat/completions"
BATCH_WINDOW = "24h"
BATCH_MAX_REQUESTS = 50_000                    # OpenAI limit per batch file
PROMPT_ID_KEY = 'custom_id'

# batch.status -> collection state
DONE_STATES = {'completed'}
FAILED_STATES = {'failed', 'cancelled', 'expired'}   # 'expired' with an output file still counts as done (partial)


def prompt_gpt(api_key, instructions, input, model="gpt-4.1-nano", temperature=0):
    client = OpenAI(
        # This is the default and can be omitted
        api_key=api_key,
    )

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=input,
        temperature=temperature,
    )

    return response


# -----------------------------------------------------------------------------
# Batch API
# -----------------------------------------------------------------------------
def make_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def batch_request_line(api_name: str, prompt_id: int | str, prompt: str) -> dict:
    """One request of the batch file (JSON mode, as in the original collection)."""
    return {
        "custom_id": str(prompt_id),
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": {
            "model": api_name,
            "messages": [{"role": "user", "content": prompt.strip()}],
            "response_format": {"type": "json_object"},
        },
    }


def submit_batch(client: OpenAI, requests_path: Path, api_name: str, description: str) -> dict:
    """Upload the requests file and start the batch job. Returns what is needed to find it again."""
    with open(requests_path, "rb") as f:
        input_file = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(input_file_id=input_file.id, endpoint=BATCH_ENDPOINT,
                                  completion_window=BATCH_WINDOW, metadata={"description": description})
    return {'job_id': batch.id, 'input_file_id': input_file.id}


def batch_status(client: OpenAI, job_id: str) -> Tuple[str, str]:
    """('done' | 'failed' | 'running', details)."""
    batch = client.batches.retrieve(job_id)
    details = f"status={batch.status} counts={batch.request_counts}"
    if batch.errors:
        details += f" errors={batch.errors}"
    if batch.status in DONE_STATES or (batch.status == 'expired' and batch.output_file_id):
        return 'done', details
    if batch.status in FAILED_STATES:
        return 'failed', details
    return 'running', details


def download_batch(client: OpenAI, job_id: str) -> str:
    """The output file (JSONL text; empty if no request succeeded)."""
    batch = client.batches.retrieve(job_id)
    if not batch.output_file_id:
        return ''
    return client.files.content(batch.output_file_id).text


def finish_batch(client: OpenAI, job_id: str) -> None:
    """Nothing to clean up (OpenAI batches cannot be deleted)."""
    return None
