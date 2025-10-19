import os
import glob
import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Any, Generator


PROMPT_LINE_SEPARATOR = "\n\n####################\n\n"

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