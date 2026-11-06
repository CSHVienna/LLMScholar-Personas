"""
The raw response file of one model in one language, as read by the audit (``responses.reader``):

    <responses_dir>/results_<api>_<language>/<api>_<language>_<model>.json
    {"<prompt id>": {id, parameters, prompt, language, model (api name), responses: [raw answer per run]}}

Run ``r`` of a prompt is done when the prompt has >= r responses, so every collection step can be stopped
and started again. Answers arriving one by one (Ollama workers) are first written as one small JSON each
into ``_temp_<model>/`` and merged into the response file by :meth:`ResponseStore.merge_temp`.
"""

import os
import time
from pathlib import Path
from typing import Any, List, Optional

from llm.models import ModelSpec
from utils import constants as cons
from utils import ios


class ResponseStore:
    """Raw responses of one model in one language (see the module docstring)."""

    def __init__(self, responses_dir: Path | str, model: ModelSpec, language: str):
        self.model = model
        self.language = language
        self.folder = Path(responses_dir) / cons.RESPONSES_SOURCE_DIR_TEMPLATE.format(source=model.api, language=language)
        self.path = self.folder / cons.RESPONSES_FILE_TEMPLATE.format(source=model.api, language=language, model=model.slug)
        self.temp_dir = self.folder / cons.TEMP_RESPONSES_DIR.format(model=model.slug)
        self.records: Optional[dict] = None          # loaded lazily: workers only write temp files

    # -------------------------------------------------------------------------
    # Response file
    # -------------------------------------------------------------------------
    def load(self) -> 'ResponseStore':
        """Read the response file (empty if absent). A broken file raises: it is never overwritten."""
        self.records = ios.load_json(self.path) if self.path.exists() else {}
        return self

    def save(self) -> None:
        ios.save_json(self._records(), self.path)

    def add_prompts(self, prompts: List[dict]) -> None:
        """Add a record for every prompt not in the file yet. A stored prompt with a different text raises
        (the prompt ids would not mean the same prompts any more)."""
        records = self._records()
        for item in prompts:
            key = str(item['id'])
            if key not in records:
                records[key] = {**item, 'language': self.language, 'model': self.model.api_name, 'responses': []}
            elif records[key].get('prompt', '').strip() != item['prompt'].strip():
                raise ValueError(f"{self.path}: prompt {key} differs from the prompt file")

    def n_runs(self, prompt_id: int | str) -> int:
        return len(self._records().get(str(prompt_id), {}).get('responses', []))

    def missing(self, prompts: List[dict], run: int) -> List[dict]:
        """Prompts without run ``run`` yet (fewer than ``run`` responses)."""
        return [item for item in prompts if self.n_runs(item['id']) < run]

    def add_response(self, prompt_id: int | str, response: Any) -> None:
        key = str(prompt_id)
        records = self._records()
        if key not in records:
            raise KeyError(f"{self.path}: response for unknown prompt {key}")
        records[key]['responses'].append(response)

    # -------------------------------------------------------------------------
    # Temporary files (one per answer)
    # -------------------------------------------------------------------------
    def write_temp(self, prompt_id: int | str, response: Any) -> Path:
        """Store one answer as its own file (safe from parallel workers; needs no loaded records)."""
        path = self.temp_dir / f"{time.time_ns()}_{os.getpid()}_{prompt_id}.json"
        ios.save_json({'prompt_id': str(prompt_id), 'language': self.language, 'model': self.model.api_name,
                       'response': response}, path)
        return path

    def merge_temp(self) -> int:
        """Append the temp answers (in arrival order) to the records, save, then delete the temp files.

        Returns the number of answers merged. A temp file of another model / language raises before anything
        is changed.
        """
        files = sorted(p for p in self.temp_dir.glob('*.json') if not p.name.startswith('.')) \
            if self.temp_dir.exists() else []
        if not files:
            return 0
        items = [ios.load_json(p) for p in files]
        for path, item in zip(files, items):
            if item.get('language') != self.language or item.get('model') != self.model.api_name:
                raise ValueError(f"{path}: belongs to {item.get('model')} / {item.get('language')}")
        for item in items:
            self.add_response(item['prompt_id'], item['response'])
        self.save()                                   # save first: a crash now repeats an answer, never loses one
        for path in files:
            path.unlink()
        return len(items)

    # -------------------------------------------------------------------------
    def _records(self) -> dict:
        if self.records is None:
            self.load()
        return self.records


def load_prompts(context_dir: Path | str, language: str, limit: Optional[int] = None) -> List[dict]:
    """The prompts of a language: ``<context_dir>/<language>/all_params_and_prompts.json`` (first ``limit`` only)."""
    path = Path(context_dir) / language / cons.PROMPTS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}: run batch_input_prompts.py first")
    prompts = ios.load_json(path)
    return prompts if limit is None else prompts[:limit]
