#!/usr/bin/env python3
"""
Step B.6a: OpenAlex candidates of every distinct recommended name, with the result of every name-match rule
(libs/matching/candidates.py). Needs B.3 (recommendations_metadata) and A.2 (names index). Heavy, run once;
thresholds are applied later (B.6b, batch_match_authors.py). Stages, each resumable, in this order:

    --stage rec_names    one process: distinct recommended names and their keys      -> rec_names.parquet
    --stage exact        per OpenAlex bucket (--buckets): exact / short / initials     -> exact/bucket-<bb>.parquet
    --stage fuzzy_recs   one process (needs ALL exact buckets): names for the fuzzy rule -> fuzzy_recs.parquet
    --stage oa_tokens    per OpenAlex bucket: OpenAlex names with those surname tokens  -> oa_tokens/bucket=<b>/
    --stage fuzzy        per shard (--shards; needs ALL oa_tokens): Jaro-Winkler pairs  -> fuzzy/shard-<ss>.parquet
    --stage merge        per OpenAlex bucket (needs ALL fuzzy shards): the candidates   -> bucket-<bb>.parquet

Usage (from any folder; run_matching_pipeline.sh runs the stages):
    python batch_match_candidates.py --stage rec_names --recommendations_metadata_dir <smoke>/recs --output_dir <smoke>/candidates
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
from pathlib import Path

from matching import candidates as mc
from groundtruth import db
from groundtruth.authorships import check_params
from utils import batch
from utils import analysis_config as ac
from utils import constants as cons
from utils import ios
from utils import logs


def out(args: argparse.Namespace, *parts) -> Path:
    return Path(args.output_dir).joinpath(*parts)


def index_file(args: argparse.Namespace, bucket: int) -> Path:
    path = db.bucket_file(args.names_dir, bucket)
    if not path.exists():
        raise FileNotFoundError(f"No A.2 name index: {path}")
    return path


def run_exact(bucket: int, args: argparse.Namespace) -> str:
    output_file = db.bucket_file(out(args, mc.EXACT_DIR), bucket)
    if output_file.exists() and not args.overwrite:
        return f"[bucket {bucket}] skipped (exists)"
    con = db.connect(args, ':memory:')
    mc.check_index(con, index_file(args, bucket))
    n = mc.write_exact_bucket(con, out(args, mc.REC_NAMES_FILE), index_file(args, bucket), output_file)
    con.close()
    return f"[bucket {bucket}] {n} exact / short / initials candidates"


def run_oa_tokens(bucket: int, args: argparse.Namespace) -> str:
    if mc.oa_tokens_done(out(args, mc.OA_TOKENS_DIR), bucket) and not args.overwrite:
        return f"[bucket {bucket}] skipped (exists)"
    con = db.connect(args, ':memory:')
    n = mc.write_oa_tokens_bucket(con, index_file(args, bucket), out(args, mc.FUZZY_RECS_FILE),
                                  out(args, mc.OA_TOKENS_DIR), bucket)
    con.close()
    return f"[bucket {bucket}] {n} OpenAlex (name, surname token) rows"


def run_fuzzy(shard: int, args: argparse.Namespace) -> str:
    output_file = mc.shard_file(out(args, mc.FUZZY_DIR), shard)
    if output_file.exists() and not args.overwrite:
        return f"[shard {shard}] skipped (exists)"
    con = db.connect(args, ':memory:')
    n = mc.write_fuzzy_shard(con, out(args, mc.FUZZY_RECS_FILE), out(args, mc.OA_TOKENS_DIR), output_file, shard,
                             args.jw_floor)
    con.close()
    return f"[shard {shard}] {n} fuzzy pairs (jw_full >= {args.jw_floor})"


def run_merge(bucket: int, args: argparse.Namespace) -> str:
    output_file = db.bucket_file(args.output_dir, bucket)
    if output_file.exists() and not args.overwrite:
        return f"[bucket {bucket}] skipped (exists)"
    con = db.connect(args, ':memory:')
    n = mc.write_merged_bucket(con, db.bucket_file(out(args, mc.EXACT_DIR), bucket), out(args, mc.FUZZY_DIR),
                               index_file(args, bucket), output_file)
    con.close()
    return f"[bucket {bucket}] {n} candidates"


def require(paths, what: str) -> None:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{what} incomplete: {len(missing)} missing (e.g. {missing[:2]})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', required=True, choices=['rec_names', 'exact', 'fuzzy_recs', 'oa_tokens', 'fuzzy', 'merge'])
    db.add_bucket_arguments(parser)
    db.add_resource_arguments(parser)
    parser.add_argument('--shards', nargs='+', type=int, default=None, help='Stage fuzzy: shards (default all); GNU parallel: --shards {}')
    parser.add_argument('--n_shards', type=int, default=cons.DEFAULT_MATCH_N_SHARDS)
    parser.add_argument('--jw_floor', type=float, default=cons.DEFAULT_MATCH_JW_FLOOR,
                        help='Fuzzy pairs stored if jw_full >= this (the lowest threshold usable in B.6b)')
    parser.add_argument('--fuzzy_all', action='store_true',
                        help='Score every name by the fuzzy rule (default: only names without exact / short / initials match)')
    parser.add_argument('--recommendations_metadata_dir', default=cons.DEFAULT_RECOMMENDATIONS_METADATA_DIR, help='Output of B.3')
    parser.add_argument('--names_dir', default=cons.DEFAULT_GT_NAMES_DIR, help='Output of A.2')
    parser.add_argument('--output_dir', default=cons.DEFAULT_MATCH_CANDIDATES_DIR)
    logs.add_log_arguments(parser)
    ac.add_config_argument(parser)
    return parser


def main() -> None:
    args = ac.parse_with_config(build_parser())
    if args.stage == 'fuzzy':
        units = args.shards if args.shards is not None else list(range(args.n_shards))
        tag = 'all' if len(units) == args.n_shards else f"shard{units[0]:02d}" if len(units) == 1 else f"{len(units)}shards"
    else:
        units, tag = db.buckets_of(args), db.bucket_params(args)
    logs.setup_logging(f"match_candidates_{args.stage}", args, tag if args.stage not in ('rec_names', 'fuzzy_recs') else 'all')
    check_params(args.output_dir, {'n_buckets': args.n_buckets, 'n_shards': args.n_shards, 'jw_floor': args.jw_floor,
                                   'fuzzy_all': args.fuzzy_all})
    all_exact = [db.bucket_file(out(args, mc.EXACT_DIR), b) for b in range(args.n_buckets)]

    if args.stage == 'rec_names':
        output_file = out(args, mc.REC_NAMES_FILE)
        if output_file.exists() and not args.overwrite:
            ios.printf(f"Skipped (exists): {output_file}")
            return
        con = db.connect(args, ':memory:')
        df = mc.write_rec_names(con, str(Path(args.recommendations_metadata_dir) / '*.parquet'), output_file, args.n_shards)
        ios.printf(f"{len(df)} (first, last) pairs -> {df['rec_name'].nunique()} distinct names -> {output_file}")
    elif args.stage == 'exact':
        require([out(args, mc.REC_NAMES_FILE)], 'rec_names')
        batch.run_jobs(units, run_exact, args)
    elif args.stage == 'fuzzy_recs':
        require(all_exact, 'Stage exact')
        output_file = out(args, mc.FUZZY_RECS_FILE)
        if output_file.exists() and not args.overwrite:
            ios.printf(f"Skipped (exists): {output_file}")
            return
        con = db.connect(args, ':memory:')
        n = mc.write_fuzzy_recs(con, out(args, mc.REC_NAMES_FILE), out(args, mc.EXACT_DIR), output_file, args.fuzzy_all)
        ios.printf(f"{n} names for the fuzzy rule ({'all' if args.fuzzy_all else 'without exact / short / initials match'})")
    elif args.stage == 'oa_tokens':
        require([out(args, mc.FUZZY_RECS_FILE)], 'Stage fuzzy_recs')
        batch.run_jobs(units, run_oa_tokens, args)
    elif args.stage == 'fuzzy':
        missing = [b for b in range(args.n_buckets) if not mc.oa_tokens_done(out(args, mc.OA_TOKENS_DIR), b)]
        if missing:
            raise SystemExit(f"Stage oa_tokens incomplete: buckets {missing[:5]} ... missing")
        batch.run_jobs(units, run_fuzzy, args)
    else:
        require(all_exact + [mc.shard_file(out(args, mc.FUZZY_DIR), s) for s in range(args.n_shards)], 'Stages exact / fuzzy')
        batch.run_jobs(units, run_merge, args)
    ios.printf(f"Stage {args.stage} finished")


if __name__ == '__main__':
    main()
