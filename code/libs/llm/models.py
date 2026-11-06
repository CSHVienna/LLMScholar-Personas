"""
Model names: the slug used everywhere in the project (``models.txt``, file names, the audit) and the
name the API expects (``api_name`` in ``metadata.json``).

The slug is ``model_slug(api_name)`` ('.' -> '_', ':' -> '-'); it cannot be turned back reliably, hence
``api_name``. ``api`` ('gpt' / 'gemini' / 'ollama') picks the caller and the response folder.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from utils import constants as cons
from utils import ios


@dataclass(frozen=True)
class ModelSpec:
    slug: str           # e.g. gpt-4_1-nano-2025-04-14 (models.txt, file names)
    api: str            # gpt | gemini | ollama
    api_name: str       # e.g. gpt-4.1-nano-2025-04-14 (what the API expects)


def model_slug(api_name: str) -> str:
    """File-name-safe model name: '.' -> '_', ':' -> '-' (e.g. 'llama3.2:3b' -> 'llama3_2-3b')."""
    return api_name.replace('.', '_').replace(':', '-')


def load_metadata(path: Path | str = cons.DEFAULT_MODELS_METADATA_FILE) -> Dict[str, dict]:
    """``{slug: metadata}``; keys starting with '_' are comments and are dropped."""
    return {key: value for key, value in ios.load_json(path).items() if not key.startswith('_')}


def validate_metadata(metadata: Dict[str, dict]) -> List[str]:
    """Problems found (empty = fine): missing / unknown ``api``, missing ``api_name``, key != slug of ``api_name``."""
    problems = []
    for slug, meta in metadata.items():
        api, api_name = meta.get(cons.META_API), meta.get(cons.META_API_NAME)
        if api not in cons.LLM_SOURCES:
            problems.append(f"{slug}: {cons.META_API}={api!r} not in {cons.LLM_SOURCES}")
        if not api_name:
            problems.append(f"{slug}: missing {cons.META_API_NAME}")
        elif model_slug(api_name) != slug:
            problems.append(f"{slug}: key != model_slug({api_name!r}) = {model_slug(api_name)!r}")
    return problems


def spec(metadata: Dict[str, dict], slug: str) -> ModelSpec:
    """The :class:`ModelSpec` of one model; raises if it is unknown or incomplete."""
    if slug not in metadata:
        raise KeyError(f"Model {slug!r} not in the models metadata")
    meta = metadata[slug]
    problems = validate_metadata({slug: meta})
    if problems:
        raise ValueError('; '.join(problems))
    return ModelSpec(slug=slug, api=meta[cons.META_API], api_name=meta[cons.META_API_NAME])


def slugs(metadata: Dict[str, dict], apis: Optional[List[str]] = None) -> List[str]:
    """Sorted slugs, only of ``apis`` if given."""
    return sorted(slug for slug, meta in metadata.items() if apis is None or meta.get(cons.META_API) in apis)
