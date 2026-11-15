"""
Shared command line of the research-question scripts (batch_rq1.py, batch_rq2.py, batch_rq1_interactions.py): the
common options, the study attribute of the metric sets, the metric keys of --metrics, --list_metrics, the output
folder and the log.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

from analysis import influence as inf
from metrics import summary
from utils import analysis_config as ac
from utils import constants as cons
from utils import logs

DEFAULT_METRICS = ['all', 'social_adj']        # raw metrics + the size-adjusted social ones


def add_rq_arguments(parser: argparse.ArgumentParser, name: str, variables_help: str,
                     n_boot_help: str = 'Bootstrap replicates') -> None:
    """--metrics_dir, --metrics, --variables, --n_boot, --seed, --output_dir (default <metrics_dir>/<name>),
    --list_metrics, --config / --study_ethnicity, --overwrite and the log options."""
    parser.add_argument('--metrics_dir', default=cons.DEFAULT_METRICS_DIR, help='Output of the metrics (a variant folder)')
    parser.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS,
                        help="Metric keys, or a set / subgroup ('all', 'technical', 'factuality', 'social_adj', ...)")
    parser.add_argument('--variables', nargs='+', default=inf.DEFAULT_FACTORS, help=variables_help)
    parser.add_argument('--n_boot', type=int, default=cons.DEFAULT_BOOTSTRAP_SAMPLES, help=n_boot_help)
    parser.add_argument('--seed', type=int, default=cons.DEFAULT_SEED)
    parser.add_argument('--output_dir', default=None, help=f'Default: <metrics_dir>/{name}')
    parser.add_argument('--list_metrics', action='store_true', help='Print the metric keys of --metrics and exit')
    ac.add_config_argument(parser)
    parser.add_argument('--study_ethnicity', choices=cons.STUDY_ETHNICITY_CHOICES, default=cons.DEFAULT_STUDY_ETHNICITY,
                        help="*_ethnicity metrics of the sets in --metrics: ethnicity | race | both (default: the config)")
    parser.add_argument('--overwrite', action='store_true', help='Recompute existing outputs')
    logs.add_log_arguments(parser)


def metric_keys(selection: Sequence[str]) -> List[str]:
    """Metric keys of every item (a key, a set or a subgroup), in order, without repeats."""
    return list(dict.fromkeys(m.key for item in selection for m in summary.select_metrics(item)))


def parse(parser: argparse.ArgumentParser, name: str) -> Tuple[argparse.Namespace, List[str]]:
    """(args, metric keys). --study_ethnicity defaults to the config's (only that setting: e.g. --seed stays the
    RQ's own) and the metric sets point at it; --list_metrics prints the keys and exits; the output folder defaults to
    <metrics_dir>/<name>; the log is set up."""
    args = parser.parse_args()
    if args.config and not any(a.startswith('--study_ethnicity') for a in sys.argv[1:]):
        args.study_ethnicity = ac.load(args.config)['study']['ethnicity_attribute']
    summary.use_study_ethnicity(args.study_ethnicity)
    keys = metric_keys(args.metrics)
    if args.list_metrics:
        print(' '.join(keys))
        sys.exit(0)
    args.output_dir = args.output_dir or str(Path(args.metrics_dir) / name)
    logs.setup_logging(name, args, keys[0] if len(keys) == 1 else f"{len(keys)}metrics")
    return args, keys
