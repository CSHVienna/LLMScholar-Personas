#!/usr/bin/env python3
"""
Interactions (plan D.7.3; libs/analysis/interactions.py): two-way blocks added to the additive RQ1 model, one at a
time (15 prompt pairs + LLM x each prompt variable; model characteristic x variable as decompositions) and all at
once (global check). Delta R^2, local f^2, partial omega^2, max |cell deviation|, cluster-bootstrap CIs over models.
Needs the metrics (steps 7-9) and the model metadata (size, origin).

Writes <output_dir>/<metric>.parquet (one row per block: metric, block, factor_a, factor_b, role, parent, df,
delta_r2, delta_r2_adj, f2, omega2, max_abs_deviation, share (+ _ci_low, _ci_high, _std, _n_boot), p_value, r2_base,
n_configs, n_models) and <output_dir>/cells/<metric>.parquet (one row per cell: level_a, level_b, n, observed,
additive, deviation (+ CI)). Multiple-testing correction and flags: interactions.load() (over all metrics).
The parameters are recorded in <output_dir>/_params.json (other parameters: use another --output_dir).

Usage (from any folder; run_metrics_pipeline.sh -s interactions runs one process per metric):
    python batch_interactions.py --metrics_dir <smoke>/metrics/main --metrics validity --n_boot 200
    python batch_interactions.py --list_metrics
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
from pathlib import Path

from analysis import influence as inf
from analysis import interactions as ix
from batch_rq1 import add_study_arguments, parse_study_args
from groundtruth.authorships import check_params
from metrics import summary
from utils import batch
from utils import constants as cons
from utils import ios
from utils import logs


def run_metric(metric: str, args: argparse.Namespace) -> str:
    output_file = Path(args.output_dir) / f"{metric}.parquet"
    cells_file = Path(args.output_dir) / 'cells' / f"{metric}.parquet"
    if output_file.exists() and cells_file.exists() and not args.overwrite:
        return f"[{metric}] skipped (exists)"
    if metric not in args.configs:
        raise ValueError(f"[{metric}] not in the configuration table of {args.metrics_dir}: the metrics are older than "
                         f"this metric; re-run the step that writes it (run_metrics_pipeline.sh -s expected,social)")
    table, cells = ix.interactions(args.configs, metric, args.variables, n_boot=args.n_boot, seed=args.seed)
    for frame, path in ((cells, cells_file), (table, output_file)):     # the block table last: it marks "done"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp')
        ios.to_parquet(frame, tmp)
        tmp.replace(path)
    glob = table[table['role'] == ix.GLOBAL_ROLE].iloc[0]
    blocks = table[table['role'].isin([ix.PRESPECIFIED, ix.SCREENING])]
    top = blocks.sort_values('omega2', ascending=False).iloc[0]
    return (f"[{metric}] {int(glob['n_configs'])} configurations, {int(glob['n_models'])} models, R2 base "
            f"{glob['r2_base']:.3f}, all interactions +{glob['delta_r2']:.3f} | largest omega2: {top['block']} "
            f"{top['omega2']:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--metrics_dir', default=cons.DEFAULT_METRICS_DIR, help='Output of the metrics (a variant folder)')
    parser.add_argument('--metrics', nargs='+', default=['all', 'social_adj'],
                        help="Metric keys, or a family / subgroup ('all', 'technical', 'factuality', 'social_adj')")
    parser.add_argument('--variables', nargs='+', default=inf.DEFAULT_FACTORS,
                        help='Factors of the additive base model (as RQ1); the blocks: cons.INTERACTION_*')
    parser.add_argument('--n_boot', type=int, default=cons.DEFAULT_BOOTSTRAP_SAMPLES, help='Bootstrap replicates')
    parser.add_argument('--seed', type=int, default=cons.DEFAULT_SEED)
    parser.add_argument('--models_metadata', default=cons.DEFAULT_MODELS_METADATA_FILE,
                        help='Model characteristics (size, origin) for the decompositions')
    parser.add_argument('--output_dir', default=None, help='Default: <metrics_dir>/interactions')
    parser.add_argument('--list_metrics', action='store_true', help='Print the metric keys of --metrics and exit')
    add_study_arguments(parser)
    parser.add_argument('--overwrite', action='store_true', help='Recompute existing outputs')
    logs.add_log_arguments(parser)
    return parser


def metric_keys(selection) -> list:
    """Metric keys of every item (a key, a family or a subgroup), in order, without repeats."""
    return list(dict.fromkeys(m.key for item in selection for m in summary.select_metrics(item)))


def main() -> None:
    args = parse_study_args(build_parser())
    keys = metric_keys(args.metrics)
    if args.list_metrics:
        print(' '.join(keys))
        return
    args.output_dir = args.output_dir or str(Path(args.metrics_dir) / 'interactions')
    logs.setup_logging('interactions', args, keys[0] if len(keys) == 1 else f"{len(keys)}metrics")
    check_params(args.output_dir, {'variables': list(args.variables), 'n_boot': args.n_boot, 'seed': args.seed,
                                   'screening': [list(b) for b in ix.screening_blocks()],
                                   'prespecified': [list(b) for b in cons.INTERACTION_PRESPECIFIED],
                                   'decompositions': [list(b) for b in cons.INTERACTION_DECOMPOSITIONS],
                                   'metrics_dir': str(Path(args.metrics_dir).resolve())},
                 overwrite=args.overwrite)
    configs = summary.configuration_table(args.metrics_dir)
    args.configs = summary.with_model_attributes(configs, summary.load_model_metadata(args.models_metadata))
    batch.run_jobs(keys, run_metric, args)


if __name__ == '__main__':
    main()
