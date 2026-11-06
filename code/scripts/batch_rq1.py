#!/usr/bin/env python3
"""
RQ1: global influence of each configuration variable (and of the model) on each metric (libs/analysis/influence.py):
partial omega^2 (type-II ANOVA) and Shapley R^2, with cluster-bootstrap CIs over models. Needs the metrics (steps 7-8).

Writes <output_dir>/<metric>.parquet (default <metrics_dir>/rq1/), one row per variable:
    metric, variable, family (model / persona / context), omega2 (+ _ci_low, _ci_high, _std, _n_boot),
    shapley_r2 (+ the same), r2_full, n_configs, n_models
The parameters are recorded in <output_dir>/_params.json (other parameters: use another --output_dir).

Usage (from any folder; run_metrics_pipeline.sh -s rq1 runs one process per metric):
    python batch_rq1.py --metrics_dir <smoke>/metrics/main --metrics validity --n_boot 200
    python batch_rq1.py --list_metrics
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
from pathlib import Path

from analysis import influence as inf
from groundtruth.authorships import check_params
from metrics import summary
from utils import batch
from utils import constants as cons
from utils import ios
from utils import logs


def run_metric(metric: str, args: argparse.Namespace) -> str:
    output_file = Path(args.output_dir) / f"{metric}.parquet"
    if output_file.exists() and not args.overwrite:
        return f"[{metric}] skipped (exists)"
    table = inf.influence(args.configs, metric, args.variables, args.n_boot, args.n_boot_shapley, seed=args.seed)
    tmp = output_file.with_name(output_file.name + '.tmp')
    ios.to_parquet(table, tmp)
    tmp.replace(output_file)
    top = table.sort_values('omega2', ascending=False).iloc[0]
    return (f"[{metric}] {int(top['n_configs'])} configurations, {int(top['n_models'])} models, R2 {top['r2_full']:.3f} "
            f"| largest omega2: {top['variable']} {top['omega2']:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--metrics_dir', default=cons.DEFAULT_METRICS_DIR, help='Output of the metrics (a variant folder)')
    parser.add_argument('--metrics', nargs='+', default=['all'],
                        help="Metric keys, or a family / subgroup ('all', 'technical', 'factuality', 'consistency')")
    parser.add_argument('--variables', nargs='+', default=inf.DEFAULT_FACTORS,
                        help='Factors of the model (model + configuration variables); subfield is coded nested in field')
    parser.add_argument('--n_boot', type=int, default=cons.DEFAULT_BOOTSTRAP_SAMPLES, help='Bootstrap replicates (omega2)')
    parser.add_argument('--n_boot_shapley', type=int, default=None, help='Bootstrap replicates (Shapley; default --n_boot)')
    parser.add_argument('--seed', type=int, default=cons.DEFAULT_SEED)
    parser.add_argument('--output_dir', default=None, help='Default: <metrics_dir>/rq1')
    parser.add_argument('--list_metrics', action='store_true', help='Print the metric keys of --metrics and exit')
    parser.add_argument('--overwrite', action='store_true', help='Recompute existing outputs')
    logs.add_log_arguments(parser)
    return parser


def metric_keys(selection) -> list:
    """Metric keys of every item (a key, a family or a subgroup), in order, without repeats."""
    return list(dict.fromkeys(m.key for item in selection for m in summary.select_metrics(item)))


def main() -> None:
    args = build_parser().parse_args()
    keys = metric_keys(args.metrics)
    if args.list_metrics:
        print(' '.join(keys))
        return
    args.output_dir = args.output_dir or str(Path(args.metrics_dir) / 'rq1')
    logs.setup_logging('rq1', args, keys[0] if len(keys) == 1 else f"{len(keys)}metrics")
    check_params(args.output_dir, {'variables': list(args.variables), 'n_boot': args.n_boot,
                                   'n_boot_shapley': args.n_boot_shapley or args.n_boot, 'seed': args.seed,
                                   'metrics_dir': str(Path(args.metrics_dir).resolve())})
    args.configs = summary.configuration_table(args.metrics_dir)
    batch.run_jobs(keys, run_metric, args)


if __name__ == '__main__':
    main()
