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
    python batch_rq1.py --list_metrics --config config/analysis.yaml    # sets follow study.ethnicity_attribute
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import _rq
import argparse
from pathlib import Path

from analysis import influence as inf
from utils.batch import check_params
from metrics import summary
from utils import batch
from utils import ios


def run_metric(metric: str, args: argparse.Namespace) -> str:
    output_file = Path(args.output_dir) / f"{metric}.parquet"
    if output_file.exists() and not args.overwrite:
        return f"[{metric}] skipped (exists)"
    table = inf.influence(args.configs, metric, args.variables, args.n_boot, args.n_boot_shapley, seed=args.seed)
    ios.write_parquet_atomic(table, output_file)
    top = table.sort_values('omega2', ascending=False).iloc[0]
    return (f"[{metric}] {int(top['n_configs'])} configurations, {int(top['n_models'])} models, R2 {top['r2_full']:.3f} "
            f"| largest omega2: {top['variable']} {top['omega2']:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _rq.add_rq_arguments(parser, 'rq1', 'Factors of the model (model + configuration variables); subfield is coded '
                         'nested in field', n_boot_help='Bootstrap replicates (omega2)')
    parser.add_argument('--n_boot_shapley', type=int, default=None, help='Bootstrap replicates (Shapley; default --n_boot)')
    return parser


def main() -> None:
    args, keys = _rq.parse(build_parser(), 'rq1')
    check_params(args.output_dir, {'variables': list(args.variables), 'n_boot': args.n_boot,
                                   'n_boot_shapley': args.n_boot_shapley or args.n_boot, 'seed': args.seed,
                                   'metrics_dir': str(Path(args.metrics_dir).resolve())},
                 overwrite=args.overwrite)
    args.configs = summary.configuration_table(args.metrics_dir)
    batch.run_jobs(keys, run_metric, args)


if __name__ == '__main__':
    main()
