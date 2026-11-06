"""
Cloud batch collection (GPT, Gemini): the batch files of one model x language x run, and the provider
module that handles each API.

Files of one batch, in ``results_<api>_<language>/_batches/``::

    <api>__<language>__<model>__run<NN>.requests.jsonl    written by prepare
    <api>__<language>__<model>__run<NN>.submitted.json    written by submit (job id); its presence = submitted
    <api>__<language>__<model>__run<NN>.responses.jsonl   written by retrieve (provider output, as downloaded)

When a batch is merged, its files move to ``_batches/done/``; when it failed, to ``_batches/failed/``
(then prepare writes it again for the prompts still missing). A batch file in ``_batches/`` blocks a new one
for the same run, and a prompt is queued only (repetitions - responses - open batches with it) more times,
so every prompt ends with exactly ``--repetitions`` answers, whatever order the batches finish in.
"""

import importlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List, Tuple

from llm.store import ResponseStore
from utils import constants as cons
from utils import ios

PROVIDER_MODULES = {cons.SOURCE_GPT: 'llm.openai', cons.SOURCE_GEMINI: 'llm.gemini'}


def provider(api: str) -> ModuleType:
    """The module of a cloud API (imported on first use, so only its client library is needed)."""
    if api not in PROVIDER_MODULES:
        raise ValueError(f"Not a cloud API: {api!r} (cloud APIs: {cons.CLOUD_APIS})")
    return importlib.import_module(PROVIDER_MODULES[api])


@dataclass(frozen=True)
class BatchFiles:
    folder: Path        # .../_batches (or a sub-folder)
    name: str           # <api>__<language>__<model>__run<NN>

    @property
    def requests(self) -> Path:
        return self.folder / f"{self.name}{cons.BATCH_REQUESTS_SUFFIX}"

    @property
    def submitted(self) -> Path:
        return self.folder / f"{self.name}{cons.BATCH_SUBMITTED_SUFFIX}"

    @property
    def responses(self) -> Path:
        return self.folder / f"{self.name}{cons.BATCH_RESPONSES_SUFFIX}"

    @property
    def run(self) -> int:
        return int(self.name.rsplit('__run', 1)[1][:2])          # archived copies may end in _2, _3, ...

    def existing(self) -> List[Path]:
        return [p for p in (self.requests, self.submitted, self.responses) if p.exists()]


def batches_dir(store: ResponseStore) -> Path:
    return store.folder / cons.BATCHES_DIR


def batch_files(store: ResponseStore, run: int) -> BatchFiles:
    return BatchFiles(batches_dir(store), cons.BATCH_FILE_TEMPLATE.format(
        api=store.model.api, language=store.language, model=store.model.slug, run=run))


def open_batches(store: ResponseStore) -> List[BatchFiles]:
    """Batches of this model x language not merged / failed yet (requests file in ``_batches/``), by run."""
    prefix = cons.BATCH_FILE_TEMPLATE.format(api=store.model.api, language=store.language,
                                             model=store.model.slug, run=0)[:-2]          # ...__run
    folder = batches_dir(store)
    found = [BatchFiles(folder, p.name[:-len(cons.BATCH_REQUESTS_SUFFIX)])
             for p in folder.glob(f"{prefix}[0-9][0-9]{cons.BATCH_REQUESTS_SUFFIX}")] if folder.exists() else []
    return sorted(found, key=lambda files: files.run)


def open_request_counts(store: ResponseStore, batches: List[BatchFiles]) -> Counter:
    """How many open batches (requests written, not merged yet) contain each prompt id."""
    key = provider(store.model.api).PROMPT_ID_KEY
    counts = Counter()
    for files in batches:
        with open(files.requests, encoding='utf-8') as f:
            counts.update(str(json.loads(line)[key]) for line in f if line.strip())
    return counts


def write_requests(files: BatchFiles, lines: Iterable[dict]) -> int:
    files.folder.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(files.requests, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
            n += 1
    return n


def read_submitted(files: BatchFiles) -> dict:
    return ios.load_json(files.submitted)


def parse_output(api: str, text: str) -> List[Tuple[str, dict]]:
    """(prompt id, answer line) for every non-empty line of a provider output file."""
    key = provider(api).PROMPT_ID_KEY
    answers = []
    for line in text.splitlines():
        if line.strip():
            item = json.loads(line)
            answers.append((str(item[key]), item))
    return answers


def archive(files: BatchFiles, sub: str) -> BatchFiles:
    """Move the files of a batch into ``_batches/<sub>/`` (``..._2``, ``..._3`` if the name is taken)."""
    target_dir = files.folder / sub
    target_dir.mkdir(parents=True, exist_ok=True)
    name, index = files.name, 1
    while BatchFiles(target_dir, name).existing():
        index += 1
        name = f"{files.name}_{index}"
    target = BatchFiles(target_dir, name)
    for source, dest in ((files.requests, target.requests), (files.submitted, target.submitted),
                         (files.responses, target.responses)):
        if source.exists():
            source.rename(dest)
    return target


# -----------------------------------------------------------------------------
# Steps (the scripts only add the CLI, the API client and the logging)
# -----------------------------------------------------------------------------
def prepare_batches(store: ResponseStore, prompts: List[dict], repetitions: int,
                    overwrite: bool = False) -> Dict[int, int]:
    """Write a requests file for every run 1..``repetitions`` that has prompts left and no open batch.

    A prompt goes into (repetitions - its responses - open batches with it) of the free runs, lowest first.
    ``overwrite`` rewrites open
    batches that are not submitted yet. Returns {run: number of requests written}.
    """
    module = provider(store.model.api)
    store.add_prompts(prompts)
    store.save()

    open_runs = {files.run: files for files in open_batches(store)}
    if overwrite:
        for run, files in list(open_runs.items()):
            if not files.submitted.exists():
                files.requests.unlink()
                del open_runs[run]
    queued = open_request_counts(store, list(open_runs.values()))

    # A prompt still needs repetitions - answers - open batches with it; it goes into that many free run slots
    # (the run number only names the file: what counts is how often each prompt is asked).
    free_runs = [run for run in range(1, repetitions + 1) if run not in open_runs]
    deficit = {str(p['id']): repetitions - store.n_runs(p['id']) - queued[str(p['id'])] for p in prompts}
    written = {}
    for slot, run in enumerate(free_runs):
        todo = [p for p in prompts if deficit[str(p['id'])] > slot]
        if not todo:
            continue
        if len(todo) > getattr(module, 'BATCH_MAX_REQUESTS', len(todo)):
            raise ValueError(f"{len(todo)} requests > {module.BATCH_MAX_REQUESTS} per batch file")
        written[run] = write_requests(batch_files(store, run),
                                      (module.batch_request_line(store.model.api_name, p['id'], p['prompt'])
                                       for p in todo))
    return written


def merge_output(store: ResponseStore, files: BatchFiles, text: str) -> int:
    """Keep the downloaded output next to the requests, add its answers to the response file, archive the
    batch in ``done/``. Returns the number of answers merged."""
    files.responses.write_text(text, encoding='utf-8')
    answers = parse_output(store.model.api, text)
    for prompt_id, answer in answers:
        store.add_response(prompt_id, answer)
    store.save()
    archive(files, cons.BATCHES_DONE_DIR)
    return len(answers)
