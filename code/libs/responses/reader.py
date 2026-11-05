"""
Reading raw response files and turning them into parsed, tagged responses.

:class:`ResponseReader` is the single entry point used by every step that needs
the raw responses (response summary, recommendation tagging), so that all steps
parse and classify the output in exactly the same way.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from utils import constants as cons
from utils import ios
from responses.content_parser import ContentParser, ParsedContent
from responses.context import PromptContext
from responses.sources import RawResponse, get_adapter
from responses.taxonomy import classify, has_recommendations


@dataclass(frozen=True)
class ParsedResponse:
    """One response (model x prompt x run) after extraction, decoding and classification."""
    ids: dict                 # model, source, date, config variables, prompt_id, run_id
    raw: RawResponse
    parsed: Optional[ParsedContent]
    tags: List[str]           # response tags, in tag order

    @property
    def has_recommendations(self) -> bool:
        return has_recommendations(self.tags)

    @property
    def persons(self) -> List[dict]:
        return self.parsed.persons if self.parsed else []


class ResponseReader:
    """Iterate the parsed responses of a model across languages.

    Parameters
    ----------
    responses_dir    : folder with ``results_<source>_<language>/`` sub-folders.
    context          : translator of prompt parameters into canonical values.
    collection_dates : ``{model: {language: 'YYYY-MM-DD'}}`` used when responses carry no timestamp.
    limit_prompts    : only the first N prompts per file (smoke tests).
    """

    def __init__(self, responses_dir: Path | str, context: PromptContext,
                 collection_dates: Optional[Dict[str, Dict[str, str]]] = None,
                 limit_prompts: Optional[int] = None):
        self.responses_dir = Path(responses_dir)
        self.context = context
        self.collection_dates = collection_dates or {}
        self.limit_prompts = limit_prompts
        self.parser = ContentParser()

    @classmethod
    def from_args(cls, args) -> 'ResponseReader':
        """Build a reader from the CLI arguments added by ``utils.batch.add_raw_response_arguments``."""
        return cls(args.responses_dir, PromptContext(args.context_dir, args.languages),
                   read_collection_dates(args.collection_dates_file), args.limit_prompts)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def locate(self, model: str, language: str) -> Optional[Tuple[str, Path]]:
        """Return (source, path) of the raw file of ``model`` in ``language``, or None if absent."""
        for source in cons.LLM_SOURCES:
            path = (self.responses_dir
                    / cons.RESPONSES_SOURCE_DIR_TEMPLATE.format(source=source, language=language)
                    / cons.RESPONSES_FILE_TEMPLATE.format(source=source, language=language, model=model))
            if path.exists():
                return source, path
        return None

    def iter_model(self, model: str, languages: List[str]) -> Iterator[ParsedResponse]:
        """Yield all responses of ``model`` in ``languages`` (missing files are reported and skipped)."""
        for language in languages:
            located = self.locate(model, language)
            if located is None:
                ios.printf(f"[{model}] missing file for language={language}")
                continue
            source, path = located
            ios.printf(f"[{model}] reading {path}")
            yield from self.iter_file(path, source, model, language)

    def raw_texts(self, model: str, language: str, keys: List[Tuple[int, int]]) -> Dict[Tuple[int, int], Tuple[str, str]]:
        """Prompt and raw generated text of specific responses: {(prompt_id, run_id): (prompt, text)}."""
        located = self.locate(model, language)
        if located is None:
            return {}
        source, path = located
        adapter, records = get_adapter(source), ios.load_json(path)
        found = {}
        for prompt_id, run_id in keys:
            record = records[str(prompt_id)]
            raw = adapter.extract(record['responses'][run_id - 1])
            found[(prompt_id, run_id)] = (record.get('prompt', ''), raw.text or '')
        return found

    def iter_file(self, path: Path, source: str, model: str, language: str) -> Iterator[ParsedResponse]:
        """Yield every response of one raw file."""
        adapter = get_adapter(source)
        fallback_date = self.collection_dates.get(model, {}).get(language, cons.DATE_UNKNOWN)

        for position, (prompt_key, record) in enumerate(ios.load_json(path).items()):
            if self.limit_prompts is not None and position >= self.limit_prompts:
                break

            # 1. Prompt-level identifiers (canonical English values)
            raws = [adapter.extract(response) for response in record.get('responses', [])]
            prompt_ids = {cons.COL_MODEL: model, cons.COL_SOURCE: source,
                          cons.COL_DATE: collection_date(raws, fallback_date),
                          **self.context.canonical(language, record.get('parameters', {})),
                          cons.COL_PROMPT_ID: int(record.get('id', prompt_key))}

            # 2. One parsed response per run (run_id starts at 1)
            for run_id, raw in enumerate(raws, start=1):
                parsed = self.parser.parse(raw.text) if raw.text else None
                yield ParsedResponse(ids={**prompt_ids, cons.COL_RUN_ID: run_id},
                                     raw=raw, parsed=parsed, tags=classify(raw, parsed))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def collection_date(raws: List[RawResponse], fallback: str = cons.DATE_UNKNOWN) -> str:
    """Date of the first timestamped run of a prompt, else ``fallback``.

    One date per prompt, so the runs of a configuration are never split across dates.
    """
    timestamps = sorted(raw.created_at for raw in raws if raw.created_at)
    return timestamps[0][:10] if timestamps else fallback


def read_collection_dates(path: Optional[Path | str]) -> Dict[str, Dict[str, str]]:
    """Read ``{model: {language: date}}``; keys starting with '_' are comments. Missing file -> {}."""
    if path is None or not Path(path).exists():
        return {}
    return {model: dates for model, dates in ios.load_json(path).items() if not model.startswith('_')}
