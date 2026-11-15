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
    python batch_rq1_interactions.py --metrics_dir <smoke>/metrics/main --metrics validity --n_boot 200
    python batch_rq1_interactions.py --list_metrics
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import _rq
import argparse
from pathlib import Path

from analysis import interactions as ix
from utils.batch import check_params
from metrics import summary
from utils import batch
from utils import constants as cons
from utils import ios


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
        ios.write_parquet_atomic(frame, path)
    glob = table[table['role'] == ix.GLOBAL_ROLE].iloc[0]
    blocks = table[table['role'].isin([ix.PRESPECIFIED, ix.SCREENING])]
    top = blocks.sort_values('omega2', ascending=False).iloc[0]
    return (f"[{metric}] {int(glob['n_configs'])} configurations, {int(glob['n_models'])} models, R2 base "
            f"{glob['r2_base']:.3f}, all interactions +{glob['delta_r2']:.3f} | largest omega2: {top['block']} "
            f"{top['omega2']:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _rq.add_rq_arguments(parser, 'interactions', 'Factors of the additive base model (as RQ1); the blocks: cons.INTERACTION_*')
    parser.add_argument('--models_metadata', default=cons.DEFAULT_MODELS_METADATA_FILE,
                        help='Model characteristics (size, origin) for the decompositions')
    return parser


def main() -> None:
    args, keys = _rq.parse(build_parser(), 'interactions')
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
