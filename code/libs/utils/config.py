# config.py
import os
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path


def _expand(p: str) -> Path:
    return Path(p).expanduser().resolve()


def _find_config(start: Path | None = None) -> Path:
    """Walk up from `start` (or cwd) looking for a config.ini file."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / "config.ini"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "config.ini not found. Copy config.ini.example to config.ini at the repo root "
        "and fill in the paths for your environment."
    )


def _read_parser(path: Path | str | None = None) -> ConfigParser:
    cfg_path = Path(path) if path is not None else _find_config()
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read(cfg_path, encoding="utf-8")
    return parser


def load_config(path: Path | str | None = None) -> dict:
    # If no explicit path, walk up from cwd to find config.ini.
    cfg_path = Path(path) if path is not None else _find_config()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}. Create it first.")

    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read(cfg_path, encoding="utf-8")

    keys_dir = os.environ.get("LLM_KEYS_DIR") or parser.get(
        "secrets", "keys_dir", fallback=""
    )
    if not keys_dir:
        raise ValueError(
            "Keys Dir key is missing. Set env var LLM_KEYS_DIR or fill [secrets].keys_dir in config.ini"
        )

    openai_api_dir = parser.get("openai", "data_dir", fallback="")
    namsor_api_dir = parser.get("namsor", "data_dir", fallback="")

    return {
        "LLM_KEYS_DIR": keys_dir,
        "OPENAI_API_DIR": openai_api_dir,
        "NAMSOR_API_DIR": namsor_api_dir,
    }


def get_data_path(key: str, *, path: Path | str | None = None) -> str:
    """Return [data].<key> from config.ini. Raises if missing or unresolved placeholder."""
    parser = _read_parser(path)
    value = parser.get("data", key, fallback="").strip()
    if not value:
        raise KeyError(f"Missing [data].{key} in config.ini")
    if value.startswith("<") and value.endswith(">"):
        raise ValueError(
            f"[data].{key} still has the placeholder value {value!r}. "
            f"Edit config.ini and set it to the real path."
        )
    return value


def config_default(key: str) -> str | None:
    """Like get_data_path, but returns None if config.ini is missing or the
    key is unset. Convenient as an argparse `default=` so `--help` keeps
    working when the user has not configured [data] yet."""
    try:
        return get_data_path(key)
    except (FileNotFoundError, KeyError, ValueError):
        return None


def get_results_path(*, path: Path | str | None = None) -> Path:
    """Return [data].results_dir as an absolute Path.

    Falls back to ``<repo_root>/results`` if config.ini is missing or
    ``[data].results_dir`` is not set / still has its placeholder value.
    """
    try:
        return _expand(get_data_path("results_dir", path=path))
    except (FileNotFoundError, KeyError, ValueError):
        # Fall back to <repo_root>/results — repo_root is two levels up from
        # this file (code/libs/utils/config.py → code/libs/utils → code/libs → code → repo).
        return Path(__file__).resolve().parents[3] / "results"
