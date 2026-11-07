#!/usr/bin/env python3
"""
RQ2: local influence of each value of the configuration variables on each metric (libs/analysis/deviations.py):
weighted effect coding, i.e. every level vs the grand mean, adjusted for the model and the other variables; estimated
marginal means (EMM) and deviations with cluster-bootstrap CIs over models. Needs the metrics (steps 7-9).

Writes <output_dir>/<metric>.parquet (default <metrics_dir>/rq2/), one row per level of every configuration variable:
    metric, variable, family (persona / context), parent (the field of a subfield), level, n, share, grand_mean,
    emm, deviation, reference, contrast (+ _ci_low, _ci_high, _std, _n_boot for the three), r2, r2_adj,
    delta_r2_model, n_configs, n_models
The parameters are recorded in <output_dir>/_params.json (other parameters: use another --output_dir).

Usage (from any folder; run_metrics_pipeline.sh -s rq2 runs one process per metric):
    python batch_rq2.py --metrics_dir <smoke>/metrics/main --metrics validity --n_boot 200
    python batch_rq2.py --list_metrics
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
from pathlib import Path

from analysis import deviations as dv
from analysis import influence as inf
from batch_rq1 import metric_keys
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
    table = dv.deviations(args.configs, metric, args.variables, args.n_boot, seed=args.seed)
    tmp = output_file.with_name(output_file.name + '.tmp')
    ios.to_parquet(table, tmp)
    tmp.replace(output_file)
    top = table.loc[table['deviation'].abs().idxmax()]
    return (f"[{metric}] {int(top['n_configs'])} configurations, {int(top['n_models'])} models, grand mean "
            f"{top['grand_mean']:.3f} | largest deviation: {top['variable']}={top['level']} {top['deviation']:+.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--metrics_dir', default=cons.DEFAULT_METRICS_DIR, help='Output of the metrics (a variant folder)')
    parser.add_argument('--metrics', nargs='+', default=['all', 'social_adj'],
                        help="Metric keys, or a family / subgroup ('all', 'technical', 'factuality', 'consistency')")
    parser.add_argument('--variables', nargs='+', default=inf.DEFAULT_FACTORS,
                        help='Factors of the model (model + configuration variables); subfield is coded nested in field')
    parser.add_argument('--n_boot', type=int, default=cons.DEFAULT_BOOTSTRAP_SAMPLES, help='Bootstrap replicates')
    parser.add_argument('--seed', type=int, default=cons.DEFAULT_SEED)
    parser.add_argument('--output_dir', default=None, help='Default: <metrics_dir>/rq2')
    parser.add_argument('--list_metrics', action='store_true', help='Print the metric keys of --metrics and exit')
    parser.add_argument('--overwrite', action='store_true', help='Recompute existing outputs')
    logs.add_log_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    keys = metric_keys(args.metrics)
    if args.list_metrics:
        print(' '.join(keys))
        return
    args.output_dir = args.output_dir or str(Path(args.metrics_dir) / 'rq2')
    logs.setup_logging('rq2', args, keys[0] if len(keys) == 1 else f"{len(keys)}metrics")
    check_params(args.output_dir, {'variables': list(args.variables), 'n_boot': args.n_boot, 'seed': args.seed,
                                   'metrics_dir': str(Path(args.metrics_dir).resolve())},
                 overwrite=args.overwrite)
    args.configs = summary.configuration_table(args.metrics_dir)
    batch.run_jobs(keys, run_metric, args)


if __name__ == '__main__':
    main()
