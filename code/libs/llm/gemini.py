"""
Gemini Batch API (batch_collect_cloud_*.py): one JSONL file of requests -> uploaded -> one batch job ->
one output file, one JSON line per answer, ``key`` = prompt id. The stored answer is that whole line
(read by ``responses.sources.GeminiAdapter``).
"""

from pathlib import Path
from typing import Tuple

from google import genai
from google.genai import types

PROMPT_ID_KEY = 'key'
DONE_STATES = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_PARTIALLY_SUCCEEDED'}
FAILED_STATES = {'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}


def make_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def batch_request_line(api_name: str, prompt_id: int | str, prompt: str) -> dict:
    """One request of the batch file (JSON output, as in the original collection; the model is set per job)."""
    return {
        "key": str(prompt_id),
        "request": {
            "contents": [{"parts": [{"text": prompt.strip()}]}],
            "generation_config": {"responseMimeType": "application/json"},
        },
    }


def submit_batch(client: genai.Client, requests_path: Path, api_name: str, description: str) -> dict:
    """Upload the requests file and start the batch job. Returns what is needed to find it again."""
    uploaded = client.files.upload(file=str(requests_path),
                                   config=types.UploadFileConfig(display_name=description, mime_type='application/json'))
    job = client.batches.create(model=api_name, src=uploaded.name, config={'display_name': description})
    return {'job_id': job.name, 'input_file_id': uploaded.name}


def batch_status(client: genai.Client, job_id: str) -> Tuple[str, str]:
    """('done' | 'failed' | 'running', details)."""
    job = client.batches.get(name=job_id)
    state = job.state.name if job.state else 'UNKNOWN'
    details = f"state={state}" + (f" error={job.error}" if job.error else '')
    if state in DONE_STATES:
        return 'done', details
    if state in FAILED_STATES:
        return 'failed', details
    return 'running', details


def download_batch(client: genai.Client, job_id: str) -> str:
    """The output file (JSONL text)."""
    job = client.batches.get(name=job_id)
    if not job.dest or not job.dest.file_name:
        raise ValueError(f"Batch {job_id} has no output file (not a batch from a file?)")
    return client.files.download(file=job.dest.file_name).decode("utf-8")


def finish_batch(client: genai.Client, job_id: str) -> None:
    """Delete the finished job (as in the original collection)."""
    client.batches.delete(name=job_id)
