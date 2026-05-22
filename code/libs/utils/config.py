# config.py
import os
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path


def _expand(p: str) -> Path:
    return Path(p).expanduser().resolve()


def load_config(path: Path | str = "config.ini") -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}. Create it first.")

    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read(cfg_path, encoding="utf-8")

    # --- required field(s) ---
    keys_dir = os.environ.get("LLM_KEYS_DIR") or parser.get(
        "secrets", "keys_dir", fallback=""
    )
    if not keys_dir:
        raise ValueError(
            "Keys Dir key is missing. Set env var LLM_KEYS_DIR or fill [secrets].keys_dir in config.ini"
        )

    # --- typed reads with defaults ---
    openai_api_dir = parser.get("openai", "data_dir", fallback="")
    namsor_api_dir = parser.get("namsor", "data_dir", fallback="")

    return {
        "LLM_KEYS_DIR": keys_dir,
        "OPENAI_API_DIR": openai_api_dir,
        "NAMSOR_API_DIR": namsor_api_dir,
    }
