#!/usr/bin/env python3
"""
Create or edit the analysis config (libs/utils/analysis_config.py): every analysis decision in one YAML file.

Interactive (a terminal is needed): one section at a time; Enter keeps the value shown, '?' explains, 'b' goes
back, 'q' quits without saving. At the end all values are shown for review before anything is written; when an
existing config is edited, the steps to re-run because of the changes are listed.

Phases: --phase early asks only what the ground truth and the matching candidates need (seniority, field share,
L/M/H bins, candidate settings); the late settings (valid tags, gender / ethnicity thresholds, social settings,
Jaro-Winkler thresholds) keep PROVISIONAL defaults until --phase late, after the metrics. The RQs need them reviewed.

Usage (from the project root):
    python code/scripts/configure_analysis.py --phase early                     # before the GT (config/analysis.yaml)
    python code/scripts/configure_analysis.py --phase late                      # after the metrics, before the RQs
    python code/scripts/configure_analysis.py                                   # every setting
    python code/scripts/configure_analysis.py --config config/analysis_smoke.yaml
    python code/scripts/configure_analysis.py --show                            # print the current values
    python code/scripts/configure_analysis.py --defaults                        # write the defaults, no questions
    python code/scripts/configure_analysis.py --check [--phase late]            # exit 1 if missing / invalid (/ late not reviewed)
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from utils import analysis_config as ac

BOLD, DIM, RED, GREEN, RESET = ('\033[1m', '\033[2m', '\033[31m', '\033[32m', '\033[0m') if sys.stdout.isatty() \
    else ('', '', '', '', '')
BACK, QUIT, HELP = 'b', 'q', '?'


class Quit(Exception):
    pass


class Back(Exception):
    pass


def ask_setting(s: ac.Setting, current: Any, ask: Callable[[str], str], say: Callable[[str], None]) -> Any:
    """Ask one setting until a valid answer (Enter keeps the current value)."""
    if s.kind in ('tags', 'choice'):
        chosen = current if isinstance(current, list) else [current]
        say('    ' + '   '.join(f"{i}. {c}{' ✓' if c in chosen else ''}" for i, c in enumerate(s.choices, 1)))
    hint = {'tags': 'numbers or names', 'choice': 'number or name', 'bool': 'y / n', 'floats': 'numbers',
            'strings': 'words, or comma-separated'}.get(s.kind, s.kind)
    while True:
        answer = ask(f"  {BOLD}{s.key}{RESET} — {s.help} {DIM}({hint}){RESET} [{ac.format_value(current)}]: ").strip()
        if answer == '':
            return current
        if answer.lower() == QUIT:
            raise Quit()
        if answer.lower() == BACK:
            raise Back()
        if answer == HELP:
            say(f"    {s.details or s.help}")
            continue
        try:
            return ac.parse_value(s, answer)
        except (ValueError, IndexError) as e:
            say(f"    {RED}✗ {e}{RESET} — try again ('{HELP}' explains, Enter keeps [{ac.format_value(current)}])")


def ask_section(index: int, section: ac.Section, values: Dict[str, Any], ask, say, total: int = 0) -> Dict[str, Any]:
    """All settings of one section, then its cross-check; 'b' on the first setting goes to the previous section."""
    say(f"\n{BOLD}── {index}/{total or len(ac.SCHEMA)}  {section.title} ──{RESET}")
    result = dict(values)
    i = 0
    while i < len(section.settings):
        s = section.settings[i]
        try:
            result[s.key] = ask_setting(s, result[s.key], ask, say)
            i += 1
        except Back:
            if i == 0:
                raise
            i -= 1
        if i == len(section.settings) and section.check:
            message = section.check(result)
            if message:
                say(f"  {RED}✗ {message}{RESET} — please revise this section")
                i = 0
    return result


def review(values, start, say, new=(), sections=None, provisional=False) -> None:
    say(f"\n{BOLD}── Review ──{RESET}   {DIM}(* = changed, + = new setting"
        f"{', [late] = provisional until --phase late' if provisional else ''}){RESET}")
    for n, section in enumerate(sections or ac.SCHEMA, 1):
        say(f"  {BOLD}{n}. {section.title}{RESET}")
        for s in section.settings:
            mark = '+' if f"{section.key}.{s.key}" in new else '*' if values[section.key][s.key] != start[section.key][s.key] else ' '
            late = f"  {DIM}[late]{RESET}" if provisional and s.phase == ac.LATE else ''
            say(f"   {mark} {s.key:<22} {ac.format_value(values[section.key][s.key])}{late}")


def wizard(path: Path, ask: Callable[[str], str] = input, say: Callable[[str], None] = print,
           phase: Optional[str] = None) -> Optional[Dict]:
    """Interactive creation / edition of every setting, or of one phase (early / late); returns the saved values
    (None if quit without saving). Early: the late settings keep their (provisional) values; late / all: they are
    marked reviewed."""
    if phase == ac.LATE and not path.exists():
        raise SystemExit(f"{path} does not exist: create it first with --phase early")
    sections = ac.phase_settings(phase)
    existing, missing = ac.load_partial(path) if path.exists() else (None, [])
    reviewed = ac.late_reviewed(path) if existing else False
    reviewed_after = True if phase in (None, ac.LATE) else reviewed
    start = existing or ac.defaults()
    values = copy.deepcopy(start)
    what = {None: 'every setting', 'early': 'EARLY settings (ground truth, matching candidates)',
            ac.LATE: 'LATE settings (decided after the metrics)'}[phase]
    say(f"{BOLD}Analysis config{RESET}  →  {path}  ({'editing' if existing else 'new; values shown are the defaults'}; "
        f"{what})")
    if phase == 'early' and not reviewed:
        say(f"{DIM}Late settings ({', '.join(ac.late_keys())}) keep provisional defaults until --phase late.{RESET}")
    if missing:
        say(f"{BOLD}New settings since this file was written (marked + in the review): {', '.join(missing)}{RESET}")
    say(f"{DIM}Enter = keep the value in [ ]   {HELP} = explain   {BACK} = back   {QUIT} = quit without saving{RESET}")
    try:
        n = 0
        while n < len(sections):
            section = sections[n]
            try:
                values[section.key] = ask_section(n + 1, section, values[section.key], ask, say, len(sections))
                n += 1
            except Back:
                n = max(n - 1, 0)
        while True:
            review(values, start, say, missing, sections, provisional=not reviewed_after)
            if existing:
                steps = ac.changed_steps(existing, values) if not missing else ac.changed_steps(
                    {s: {k: (None if f"{s}.{k}" in missing else v) for k, v in vs.items()} for s, vs in existing.items()}, values)
                if steps:
                    say(f"\n  {BOLD}Re-run because of the changes:{RESET}")
                    for step, keys in steps.items():
                        say(f"   • {step}: {', '.join(keys)}")
                    say(f"  {DIM}{ac.RERUN_NOTE}{RESET}")
                else:
                    say("\n  No changes.")
            answer = ask(f"\n  {BOLD}s{RESET} save   {BOLD}1-{len(sections)}{RESET} edit a section   "
                         f"{BOLD}q{RESET} quit without saving: ").strip().lower()
            if answer == 's':
                ac.save(path, values, reviewed=reviewed_after)
                say(f"{GREEN}✓ saved {path}{RESET}")
                return values
            if answer == QUIT:
                raise Quit()
            if answer.isdigit() and 1 <= int(answer) <= len(sections):
                section = sections[int(answer) - 1]
                try:
                    values[section.key] = ask_section(int(answer), section, values[section.key], ask, say, len(sections))
                except Back:
                    pass
    except (Quit, EOFError, KeyboardInterrupt):
        say(f"\n{DIM}Quit: nothing written.{RESET}")
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=ac.DEFAULT_CONFIG_FILE, help='Config file to create / edit')
    parser.add_argument('--show', action='store_true', help='Print the current values and exit')
    parser.add_argument('--phase', choices=ac.PHASES, default=None,
                        help='Ask only the early settings (before the GT) or the late ones (after the metrics); default: all')
    parser.add_argument('--check', action='store_true',
                        help='Exit 0 if the config exists and is complete and valid, else 1 with the reason (for the runners)')
    parser.add_argument('--defaults', action='store_true', help='Write the defaults without asking (not over an existing file)')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = Path(args.config)
    if args.check:
        try:
            ac.require_late_reviewed(path) if args.phase == ac.LATE else ac.load(path)
        except (FileNotFoundError, ValueError) as e:
            print(e)
            raise SystemExit(1)
        return
    if args.show:
        values = ac.load(path)
        review(values, values, print, sections=ac.phase_settings(args.phase), provisional=not ac.late_reviewed(path))
        return
    if args.defaults:
        if path.exists():
            raise SystemExit(f"{path} exists: edit it interactively, or delete it first")
        ac.save(path, ac.defaults(), reviewed=False)
        print(f"✓ defaults written to {path} (late settings provisional: review them with --phase late after the metrics)")
        return
    if not sys.stdin.isatty():
        raise SystemExit(f"configure_analysis.py needs a terminal (or use --defaults) to create {path}")
    if wizard(path, phase=args.phase) is None:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
