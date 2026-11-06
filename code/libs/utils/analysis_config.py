"""
Analysis config: every analysis decision (thresholds, valid tag sets, label schemes) in one YAML file, created and
edited with the wizard (scripts/configure_analysis.py) and read by the batch scripts (--config).

    config/analysis.yaml          full runs
    config/analysis_smoke.yaml    smoke tests

Two phases (Setting.phase):
    early   needed to build the ground truth and the matching candidates (seniority, field share, L/M/H bins,
            candidate settings): asked first (configure_analysis.py --phase early)
    late    decided after looking at the data and the metrics (valid tags, perceived gender / ethnicity thresholds,
            social settings, Jaro-Winkler thresholds): until reviewed (--phase late) they hold PROVISIONAL defaults,
            the pipeline runs with them, and the research questions (rq1, rq2, notebook 4) refuse to run.
The file records whether the late settings were reviewed (late_reviewed; files without it count as reviewed).

Once a file exists it is loaded as it is; the wizard only runs again if the file is deleted (or to edit it).
In a script, the config sets the defaults of the matching options; an option given on the command line wins
(robustness variants). Every output records the values it used (_params.json).
"""

import argparse
import datetime
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import yaml

from utils import constants as cons

DEFAULT_CONFIG_FILE = str(cons.PROJECT_ROOT / 'config' / 'analysis.yaml')
SMOKE_CONFIG_FILE = str(cons.PROJECT_ROOT / 'config' / 'analysis_smoke.yaml')

STEP_POPULATION = 'A.4 population (run_gt_pipeline.sh -s population)'
STEP_CANDIDATES = 'B.6a candidates (run_matching_pipeline.sh -s candidates,authors)'
STEP_AUTHORS = 'B.6b author matching (run_matching_pipeline.sh -s authors)'
STEP_METRICS = 'metrics (run_metrics_pipeline.sh)'
PHASES = ('early', 'late')
LATE = 'late'
RERUN_NOTE = ('Re-run from the first step listed, in pipeline order (A.4 population -> B.6 matching -> metrics -> '
              'social metrics -> RQs): the later steps read its outputs.')


@dataclass(frozen=True)
class Setting:
    key: str                       # YAML key within its section
    kind: str                      # int | float | bool | choice | floats | strings | tags
    default: Any
    help: str                      # one line, shown with the question
    details: str = ''              # shown on '?'
    arg: Optional[str] = None      # script option it sets (default: key)
    choices: Sequence[str] = ()
    low: Optional[float] = None    # inclusive bounds for numbers
    high: Optional[float] = None
    used_by: Sequence[str] = ()
    phase: str = 'early'           # early (before the GT / matching) | late (after the metrics, before the RQs)

    @property
    def option(self) -> str:
        return self.arg or self.key


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    settings: List[Setting]
    check: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None   # cross-setting check -> error message


def _check_seniority(v: Dict[str, Any]) -> Optional[str]:
    # 'three' (junior / mid / senior) and 'single' (junior / senior) are separate definitions: single_cutoff is free
    if not v['junior_max'] < v['senior_min']:
        return f"need junior_max < senior_min (got {v['junior_max']}, {v['senior_min']})"
    return None


def _check_scheme(edges: List[float], labels: List[str]) -> Optional[str]:
    """Cut points + one label per interval (gender, prominence)."""
    if edges != sorted(set(edges)):
        return f"cut points must be increasing: {edges}"
    if len(labels) != len(edges) + 1:
        return f"{len(edges)} cut points need {len(edges) + 1} labels (got {len(labels)}: {labels})"
    if len(set(labels)) != len(labels) or cons.GENDER_UNKNOWN in labels:
        return f"labels must be distinct and not '{cons.GENDER_UNKNOWN}': {labels}"
    return None


def _check_gender(v: Dict[str, Any]) -> Optional[str]:
    return _check_scheme(v['p_female_edges'], v['labels'])


def _check_prominence(v: Dict[str, Any]) -> Optional[str]:
    return _check_scheme(v['bin_edges'], v['bin_labels'])


def _check_matching(v: Dict[str, Any]) -> Optional[str]:
    if v['jw_full'] < v['candidates_floor']:
        return f"jw_full {v['jw_full']} < candidates_floor {v['candidates_floor']} (pairs below the floor are not stored)"
    return None


SCHEMA: List[Section] = [
    Section('validity', 'Valid responses and recommendations', [
        Setting('valid_response_tags', 'tags', list(cons.DEFAULT_VALID_RESPONSE_TAGS),
                'a response is valid if ALL its tags are in this set', choices=cons.RESPONSE_TAGS, used_by=[STEP_METRICS], phase=LATE,
                details='Tags of B.1-B.2 (see the codebook). Valid responses enter validity, consistency, duplicates, '
                        'k ratio and factuality.'),
        Setting('valid_rec_tags', 'tags', list(cons.DEFAULT_VALID_REC_TAGS),
                'a recommendation is valid if ALL its tags are in this set', choices=cons.REC_TAGS, used_by=[STEP_METRICS], phase=LATE,
                details='Tags of B.3. Valid recommendations enter duplicates and factuality (not recommendation '
                        'consistency, which uses every recommendation).'),
    ]),
    Section('seniority', 'Seniority (academic age = reference year - first publication year)', [
        Setting('reference_year', 'int', cons.GT_REFERENCE_YEAR, 'year academic age is measured at',
                low=1950, high=2100, used_by=[STEP_POPULATION, STEP_AUTHORS]),
        Setting('junior_max', 'int', cons.DEFAULT_SENIORITY_JUNIOR_MAX, 'junior: academic age <= this',
                low=1, high=60, used_by=[STEP_POPULATION, STEP_AUTHORS]),
        Setting('senior_min', 'int', cons.DEFAULT_SENIORITY_SENIOR_MIN, 'senior: academic age >= this (between: mid)',
                low=1, high=80, used_by=[STEP_POPULATION, STEP_AUTHORS]),
        Setting('single_cutoff', 'int', cons.DEFAULT_SENIORITY_SINGLE_CUTOFF, 'two-class definition (single): junior <= this < senior; independent of junior_max / senior_min',
                low=1, high=80, used_by=[STEP_POPULATION, STEP_AUTHORS]),
        Setting('matching_definition', 'choice', cons.MATCH_SENIORITY_THREE,
                'which definition author matching uses for the requested seniority', arg='seniority_definition',
                choices=[cons.MATCH_SENIORITY_THREE, cons.MATCH_SENIORITY_SINGLE], used_by=[STEP_AUTHORS],
                details="'three': junior / mid / senior (a mid-career person matches neither junior nor senior); "
                        "'single': the two-class cut-off."),
    ], _check_seniority),
    Section('field', 'Field membership', [
        Setting('min_field_share', 'float', cons.DEFAULT_GT_MIN_FIELD_SHARE,
                "minimum share of a scholar's works in a field to belong to it (0 = any work)",
                low=0, high=0.99, used_by=[STEP_POPULATION, STEP_AUTHORS],
                details='Works counted by their primary topic (A.1). Used for the population (A.4) and for the '
                        'name + field level of author matching.'),
    ]),
    Section('gender', 'Perceived gender (from p_female of the first name)', [
        Setting('p_female_edges', 'floats', list(cons.DEFAULT_GENDER_EDGES),
                'cut points on p_female, increasing, in (0, 1)', arg='gender_edges', low=0.001, high=0.999,
                used_by=[STEP_POPULATION], phase=LATE,
                details='Intervals [0, e1), [e1, e2), ..., [ek, 1]. 3 classes: 0.1 0.9; 5 classes: 0.1 0.4 0.6 0.9. '
                        "Without p_female (initials, unknown first name): 'unknown'."),
        Setting('labels', 'strings', list(cons.DEFAULT_GENDER_LABELS),
                'one label per interval, from p_female = 0 to 1', arg='gender_labels', used_by=[STEP_POPULATION], phase=LATE,
                details='e.g. male neutral female, or male, mostly male, neutral, mostly female, female.'),
    ], _check_gender),
    Section('ethnicity', 'Perceived ethnicity (raceBERT on the full name)', [
        Setting('min_prob', 'float', cons.DEFAULT_ETHNICITY_MIN_PROB,
                "label = top class if its probability >= this, else 'unknown'", arg='ethnicity_min_prob',
                low=0.2, high=1.0, used_by=[STEP_POPULATION], phase=LATE),
    ]),
    Section('prominence', 'Prominence classes: productivity, influence, collaboration (within field x seniority)', [
        Setting('bin_edges', 'floats', [round(e, 4) for e in cons.DEFAULT_GT_BIN_EDGES],
                'cut points on the percent rank, increasing, in (0, 1)', low=0.01, high=0.99, used_by=[STEP_POPULATION],
                details='One scheme for # works (productivity), # citations (influence) and # coauthors (collaboration), '
                        'ranked within field x seniority; intervals [0, e1), [e1, e2), ..., [ek, 1]. 3 classes: '
                        '0.3333 0.6667 (L M H); 5 classes: 0.2 0.4 0.6 0.8.'),
        Setting('bin_labels', 'strings', list(cons.DEFAULT_GT_BIN_LABELS), 'one label per interval, from low to high',
                used_by=[STEP_POPULATION], details='e.g. L M H, or L ML M MH H.'),
        Setting('top_percentile', 'float', cons.DEFAULT_GT_TOP_PERCENTILE,
                'popularity: top group = top x % by works and by citations (within field x seniority)',
                low=0.1, high=50, used_by=[STEP_POPULATION]),
    ], _check_prominence),
    Section('social', 'Social metrics (on the matched authors of the valid responses)', [
        Setting('include_unknown', 'bool', False,
                "count 'unknown' (no usable name / country) as a class in parity and diversity",
                used_by=[STEP_METRICS], phase=LATE,
                details="No (default): both the recommended people and the GT target are renormalised over the known "
                        "classes, and the unknown share is stored separately. Yes: 'unknown' is one more class."),
        Setting('parity_target', 'choice', 'field_x_seniority', 'GT target of parity (scope of the GT shares)',
                choices=['field_x_seniority', 'field', 'seniority', 'all'], used_by=[STEP_METRICS], phase=LATE,
                details='field_x_seniority: the requested field and seniority (main); the others are robustness checks.'),
        Setting('top_country_share', 'float', 0.5,
                'popularity by country: top countries = the fewest that together hold this share of the GT scholars',
                low=0.05, high=0.95, used_by=[STEP_METRICS], phase=LATE,
                details='Countries ranked by their share of the GT scholars in the target (requested field x seniority); '
                        'the smallest set reaching the share is the top group (known countries).'),
    ]),
    Section('matching', 'Author matching (recommended name -> OpenAlex person)', [
        Setting('jw_first', 'float', cons.DEFAULT_MATCH_JW_FIRST, 'fuzzy rule: minimum Jaro-Winkler of the first names',
                low=0, high=1, used_by=[STEP_AUTHORS], phase=LATE,
                details='Fuzzy is the last-resort rule (after exact, alternative, short, initials). Both parts must be '
                        'close: the full-name score alone accepts any "Alexander X" for "Neville Alexander". '
                        'Provisional: check notebook 2_matching_stats.ipynb on the full data. 0 = not used.'),
        Setting('jw_last', 'float', cons.DEFAULT_MATCH_JW_LAST, 'fuzzy rule: minimum Jaro-Winkler of the last names',
                low=0, high=1, used_by=[STEP_AUTHORS], phase=LATE),
        Setting('jw_full', 'float', cons.DEFAULT_MATCH_JW_FULL, 'fuzzy rule: minimum Jaro-Winkler of the full names',
                low=0, high=1, used_by=[STEP_AUTHORS], phase=LATE),
        Setting('seed', 'int', cons.DEFAULT_SEED, 'seed of the random pick among homonyms', low=0, high=2 ** 31,
                used_by=[STEP_AUTHORS]),
        Setting('candidates_floor', 'float', cons.DEFAULT_MATCH_JW_FLOOR,
                'fuzzy pairs stored by the candidate step if jw_full >= this', arg='jw_floor', low=0.5, high=1,
                used_by=[STEP_CANDIDATES],
                details='The lowest jw_full usable later: the Jaro-Winkler thresholds are decided late (after the metrics) but '
                        'can never go below this floor, so set it at or below the lowest value you may want. A lower '
                        'floor stores more pairs (slower candidate step). Changing it means re-running the (heavy) '
                        'candidate step.'),
        Setting('fuzzy_all', 'bool', False, 'score every name by the fuzzy rule (not only names without a stricter match)',
                used_by=[STEP_CANDIDATES]),
    ], _check_matching),
]
SETTINGS = {(section.key, s.key): s for section in SCHEMA for s in section.settings}


# --- values ------------------------------------------------------------------------------------------------------
def defaults() -> Dict[str, Dict[str, Any]]:
    return {section.key: {s.key: s.default for s in section.settings} for section in SCHEMA}


def parse_value(setting: Setting, text: str) -> Any:
    """Typed value from the text typed by the user (raises ValueError with a readable message)."""
    text = text.strip()
    if setting.kind == 'bool':
        if text.lower() in ('y', 'yes', 'true', '1'):
            return True
        if text.lower() in ('n', 'no', 'false', '0'):
            return False
        raise ValueError("answer y or n")
    if setting.kind == 'int':
        value = int(text)
    elif setting.kind == 'float':
        value = float(text)
    elif setting.kind == 'choice':
        options = list(setting.choices)
        value = options[int(text) - 1] if text.isdigit() and 1 <= int(text) <= len(options) else text
    elif setting.kind == 'floats':
        value = [float(x) for x in text.replace(',', ' ').split()]
    elif setting.kind == 'strings':
        value = [x.strip() for x in text.split(',') if x.strip()] if ',' in text else text.split()
    elif setting.kind == 'tags':
        options = list(setting.choices)
        value = []
        for item in text.replace(',', ' ').split():
            value.append(options[int(item) - 1] if item.isdigit() and 1 <= int(item) <= len(options) else item)
    else:
        raise ValueError(f"unknown kind {setting.kind}")
    check_value(setting, value)
    return value


def check_value(setting: Setting, value: Any) -> None:
    numbers = value if setting.kind == 'floats' else [value] if setting.kind in ('int', 'float') else []
    for number in numbers:
        if (setting.low is not None and number < setting.low) or (setting.high is not None and number > setting.high):
            raise ValueError(f"{number} is outside [{setting.low}, {setting.high}]")
    if setting.kind == 'choice' and value not in setting.choices:
        raise ValueError(f"choose one of {', '.join(setting.choices)}")
    if setting.kind == 'tags':
        unknown = [v for v in value if v not in setting.choices]
        if unknown or not value:
            raise ValueError(f"unknown tags {unknown}; choose from {', '.join(setting.choices)}" if unknown else "choose at least one tag")
    if setting.kind in ('floats', 'strings') and not value:
        raise ValueError("give at least one value")
    if setting.kind == 'bool' and not isinstance(value, bool):
        raise ValueError("answer y or n")


def validate(values: Dict[str, Dict[str, Any]]) -> List[str]:
    """All problems of a config (empty list = valid)."""
    problems = []
    for section in SCHEMA:
        given = values.get(section.key) or {}
        for s in section.settings:
            if s.key not in given:
                problems.append(f"{section.key}.{s.key}: missing")
                continue
            try:
                check_value(s, given[s.key])
            except (ValueError, TypeError) as e:
                problems.append(f"{section.key}.{s.key}: {e}")
        if section.check and not any(p.startswith(f"{section.key}.") for p in problems):
            message = section.check(given)
            if message:
                problems.append(f"{section.key}: {message}")
    return problems


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, list):
        return ', '.join(str(v) for v in value)
    return str(value)


def phase_settings(phase: Optional[str]) -> List[Section]:
    """The schema restricted to one phase (sections without settings of the phase left out); None = everything."""
    if phase is None:
        return list(SCHEMA)
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    out = []
    for section in SCHEMA:
        settings = [s for s in section.settings if s.phase == phase]
        if settings:
            out.append(Section(section.key, section.title, settings, section.check))
    return out


def late_keys() -> List[str]:
    return [f"{section}.{key}" for (section, key), s in SETTINGS.items() if s.phase == LATE]


def changed_steps(old: Dict[str, Dict[str, Any]], new: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Step -> changed settings ('section.key') between two configs."""
    steps: Dict[str, List[str]] = {}
    for (section, key), s in SETTINGS.items():
        if (old.get(section) or {}).get(key) != (new.get(section) or {}).get(key):
            for step in s.used_by:
                steps.setdefault(step, []).append(f"{section}.{key}")
    return steps


# --- files -------------------------------------------------------------------------------------------------------
def load(path: Path | str) -> Dict[str, Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No analysis config {path}: create it with  python code/scripts/configure_analysis.py "
                                f"--config {path}")
    values, missing = load_partial(path)
    if missing:
        raise ValueError(f"{path} lacks settings added since it was written: {', '.join(missing)}. Add them with  "
                         f"python code/scripts/configure_analysis.py --config {path}  (your other answers are kept)")
    problems = validate(values)
    if problems:
        raise ValueError(f"{path} is not valid (fix it or re-run configure_analysis.py):\n  " + '\n  '.join(problems))
    return values


def load_partial(path: Path | str):
    """(values, missing): the file's values, with settings it does not have yet filled in with their defaults."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    values, missing = defaults(), []
    for (section, key), _ in SETTINGS.items():
        if key in (raw.get(section) or {}):
            values[section][key] = raw[section][key]
        else:
            missing.append(f"{section}.{key}")
    return values, missing


def late_reviewed(path: Path | str) -> bool:
    """True if the late settings were reviewed (files written before the phases existed count as reviewed)."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return bool(raw.get('late_reviewed', True))


def require_late_reviewed(path: Path | str) -> None:
    """For the research questions: the config is complete and valid, and its late settings were reviewed."""
    load(path)
    if not late_reviewed(path):
        raise ValueError(f"The late settings of {path} are still the provisional defaults ({', '.join(late_keys())}). "
                         f"Look at the metrics, then review them:  python code/scripts/configure_analysis.py --config "
                         f"{path} --phase late  (and re-run the steps it lists) before the research questions.")


def save(path: Path | str, values: Dict[str, Dict[str, Any]], reviewed: bool = True) -> None:
    """YAML with the help text of every setting as a comment (late settings marked while provisional); atomic."""
    problems = validate(values)
    if problems:
        raise ValueError('\n'.join(problems))
    lines = [f"# Analysis config, written by configure_analysis.py on {datetime.datetime.now():%Y-%m-%d %H:%M}.",
             "# Edit with:  python code/scripts/configure_analysis.py --config <this file> [--phase early|late]", "",
             "# late settings (marked [late]) reviewed after the metrics? (no = provisional defaults; RQs refuse to run)",
             f"late_reviewed: {'true' if reviewed else 'false'}", ""]
    for section in SCHEMA:
        lines.append(f"# {section.title}")
        lines.append(f"{section.key}:")
        for s in section.settings:
            dumped = yaml.safe_dump(values[section.key][s.key], default_flow_style=True).strip()
            dumped = dumped[:-4].strip() if dumped.endswith('...') else dumped
            tag = ('[late, PROVISIONAL] ' if not reviewed else '[late] ') if s.phase == LATE else ''
            lines.append(f"  {s.key}: {dumped}    # {tag}{s.help}")
        lines.append("")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text('\n'.join(lines))
    tmp.replace(path)


# --- scripts -----------------------------------------------------------------------------------------------------
def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--config', default=None,
                        help='Analysis config (YAML, see configure_analysis.py): sets the defaults of the options below; '
                             'options given on the command line win')


def parse_with_config(parser: argparse.ArgumentParser, argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """parse_args where --config (if given) sets the defaults of every option the parser has."""
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        values = load(known.config)
        options = {action.dest for action in parser._actions}
        parser.set_defaults(**{s.option: values[section][key] for (section, key), s in SETTINGS.items()
                               if s.option in options})
    return parser.parse_args(argv)
