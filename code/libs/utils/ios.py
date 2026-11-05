import os
import logging
import glob
import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Any, Generator
from datetime import datetime
import re

import pandas as pd

_SURROGATES = re.compile('[\ud800-\udfff]')

PROMPT_LINE_SEPARATOR = "\n\n####################\n\n"

def path_join(*args: str) -> Path:
    return Path(os.path.join(*args))

def printf(msg: str) -> None:
    """Log through the pipeline logger if a batch script configured it (utils.logs), else print."""
    logger = logging.getLogger('llmscholar')
    if logger.handlers:
        logger.info(msg)
    else:
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

def write_list_to_file(data_list: List[str], file_path: Path | str, line_separator: str = PROMPT_LINE_SEPARATOR) -> None:
    with open(file_path, 'w') as f:
        f.write(line_separator.join(data_list))

def read_list_from_file(file_path: Path | str, line_separator: str = PROMPT_LINE_SEPARATOR) -> List[List[str]]:
    content = []
    with open(file_path, 'r') as f:
        item = []
        for line in f.read().splitlines():
            
            if line_separator == line:
                content.append(item)
                item = []
            else:
                item.append(line.strip())

        return content
    
def load_json(path: Path | str) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
    
def list_files_in_folder(folder_path: Path | str, pattern: str = "*") -> List[Path]:
    p = Path(folder_path)
    return list(p.glob(pattern))

def to_csv(df: pd.DataFrame, file_path: Path | str, **kwargs) -> None:
    p = Path(file_path)
    df.to_csv(p, index=False, **kwargs)

def load_csv(file_path: Path | str, **kwargs) -> pd.DataFrame:
    p = Path(file_path)
    return pd.read_csv(p, **kwargs)

def to_parquet(df: pd.DataFrame, file_path: Path | str, **kwargs) -> None:
    """Write a DataFrame to Parquet (creating the parent folder if needed).

    Lone UTF-16 surrogates (e.g. '\\udcaa' decoded from LLM output) cannot be encoded as UTF-8;
    they are replaced by U+FFFD in every text column before writing.
    """
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    replace_surrogates(df).to_parquet(p, index=False, **kwargs)


def replace_surrogates(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``df`` with lone surrogates replaced by U+FFFD in object (text) columns."""
    text_columns = df.select_dtypes(include='object').columns
    if len(text_columns) == 0:
        return df
    df = df.copy()
    for column in text_columns:
        df[column] = df[column].map(lambda v: _SURROGATES.sub('\ufffd', v) if isinstance(v, str) else v)
    return df


def load_parquet(file_path: Path | str, **kwargs) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    return pd.read_parquet(Path(file_path), **kwargs)


def load_parquet_folder(folder: Path | str, columns: List[str] = None, models: List[str] = None) -> pd.DataFrame:
    """Concatenate the per-model files ``<folder>/<model>.parquet`` (only ``columns``; only ``models`` if given)."""
    files = sorted(Path(folder).glob("*.parquet"))
    if models:
        files = [f for f in files if f.stem in models]
    if not files:
        raise FileNotFoundError(f"No <model>.parquet files in {folder}")
    return pd.concat([load_parquet(f, columns=columns) for f in files], ignore_index=True)
