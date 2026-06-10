import hashlib
import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List

import pandas as pd

PROMPT_LINE_SEPARATOR = "\n\n####################\n\n"


def path_join(*args: str) -> Path:
    return Path(os.path.join(*args))


def printf(msg: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


def read_text(file_path: Path | str) -> str:
    p = Path(file_path)
    with p.open("r", encoding="utf-8") as f:
        return f.read()


def path_exists(p: Path | str) -> bool:
    return os.path.exists(p)


def path(p: Path | str) -> Path:
    return Path(p)


def validate_path(path: Path | str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=False)


def write_list_to_file(
    data_list: List[str],
    file_path: Path | str,
    line_separator: str = PROMPT_LINE_SEPARATOR,
) -> None:
    with open(file_path, "w") as f:
        f.write(line_separator.join(data_list))


def read_list_from_file_llm_prompt(
    file_path: Path | str, line_separator: str = PROMPT_LINE_SEPARATOR
) -> List[List[str]]:
    content = []
    with open(file_path, "r") as f:
        item = []
        for line in f.read().splitlines():

            if line_separator == line:
                content.append(item)
                item = []
            else:
                item.append(line.strip())

        return content


def read_list_from_file(file_path: Path | str) -> List[List[str]]:
    item = []

    with open(file_path, "r") as f:

        for line in f.read().splitlines():
            item.append(line.strip())

    return item


def load_json(path: Path | str) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Any, path: Path | str) -> None:
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        

def list_files_in_folder(folder_path: Path | str, pattern: str = "*") -> List[Path]:
    p = Path(folder_path)
    return list(p.glob(pattern))


def to_csv(df: pd.DataFrame, file_path: Path | str, **kwargs) -> None:
    p = Path(file_path)
    df.to_csv(p, index=False, **kwargs)


def load_csv(file_path: Path | str, **kwargs) -> pd.DataFrame:
    p = Path(file_path)
    return pd.read_csv(p, **kwargs)


def ensure_parent_dir(path: Path | str) -> Path:
    """Create the parent directory of `path` if missing. Returns Path(path)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_input_csv(input_path: Path | str, logger=None, **kwargs) -> pd.DataFrame:
    """Standard input-CSV loader for pipeline scripts.

    Logs the input path and the resulting row count. `low_memory=False` by
    default to avoid pandas mixed-dtype warnings on the wide factuality CSVs.
    """
    kwargs.setdefault("low_memory", False)
    if logger:
        logger.info("Loading: %s", input_path)
    df = pd.read_csv(input_path, **kwargs)
    if logger:
        logger.info("Rows: %d", len(df))
    return df


def write_output_csv(df: pd.DataFrame, output_path: Path | str, logger=None) -> None:
    """Standard output-CSV writer for pipeline scripts.

    Creates any missing parent directories, writes the CSV without an index,
    and logs the row count and destination.
    """
    p = ensure_parent_dir(output_path)
    df.to_csv(p, index=False)
    if logger:
        logger.info("Saved %d rows → %s", len(df), p)


def load_pickle(path: Path | str | None, *, logger: logging.Logger | None = None):
    """Load a pickle if it exists, return None otherwise. Never raises — on any
    read error logs a warning and returns None."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if logger:
            logger.info("Pickle: loaded %s", path)
        return obj
    except Exception as exc:
        if logger:
            logger.warning("Pickle: failed to load %s (%s) — ignoring", path, exc)
        return None


def save_pickle(obj, path: Path | str | None, *, logger: logging.Logger | None = None) -> None:
    """Save `obj` as a pickle. Creates missing parent directories. Never raises
    — on any write error logs a warning and returns."""
    if not path:
        return
    try:
        ensure_parent_dir(path)
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        if logger:
            logger.info("Pickle: saved → %s", path)
    except Exception as exc:
        if logger:
            logger.warning("Pickle: failed to save %s (%s)", path, exc)


def cached_pickle(
    cache_path: Path | str,
    compute: Callable[[], Any],
    *,
    logger: logging.Logger | None = None,
):
    """Return `compute()` cached as a pickle at `cache_path`.

    On cache hit: read and return the pickle. On cache miss (or unreadable):
    call `compute()`, persist its result, return it.
    """
    cached = load_pickle(cache_path, logger=logger)
    if cached is not None:
        return cached
    value = compute()
    save_pickle(value, cache_path, logger=logger)
    return value


def file_hash(*paths) -> str:
    """MD5 of mtime+size for each path — changes when any source file is updated."""
    parts = []
    for p in paths:
        p = Path(p)
        if p.exists():
            s = p.stat()
            parts.append(f"{p}:{s.st_size}:{int(s.st_mtime)}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:10]
