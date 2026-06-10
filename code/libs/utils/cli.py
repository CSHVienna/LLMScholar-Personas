"""CLI helpers shared by every script under code/scripts/."""

import argparse
import os

# ── ANSI colour constants ─────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
GRAY = "\033[90m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"


def colorize(text: str, color: str) -> str:
    """Wrap `text` with the given ANSI colour code (and `RESET`)."""
    return f"{color}{text}{RESET}"


def print_separator(char: str = "─", width: int = 80) -> None:
    """Print a single-line separator in `GRAY`."""
    print(colorize(char * width, GRAY))


def clear_screen() -> None:
    """Clear the terminal — `clear` on POSIX, `cls` on Windows."""
    os.system("clear" if os.name == "posix" else "cls")


def add_io_args(
    parser: argparse.ArgumentParser,
    *,
    input_help: str = "Input CSV path",
    output_help: str = "Output CSV path",
) -> argparse.ArgumentParser:
    """Add the standard `--input`/`--output` required arguments to a parser."""
    parser.add_argument("--input", required=True, help=input_help)
    parser.add_argument("--output", required=True, help=output_help)
    return parser
