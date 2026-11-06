#!/usr/bin/env python3
"""
Step A.3: perceived gender (nomquamgender, first name) and perceived ethnicity (raceBERT, full name) of the authors
who are people with >= 1 work in at least one prompt field (the broadest population: any later field threshold only
selects a subset). Needs step A.1. Four stages (libs/demographics/stages.py), each resumable, run in this order;
both models run ONCE per distinct name over all buckets:

    --stage keys       per bucket (--buckets): first name + normalised raceBERT name -> keys/bucket-<bb>.parquet
    --stage gender     one process (needs ALL keys buckets): nomquamgender once per distinct first name (~1 min)
                                                                                     -> gender/first_names.parquet
    --stage ethnicity  per shard (--shards; needs ALL keys buckets): raceBERT once per distinct full name, sorted
                       by length, checkpointed every --chunk_names names             -> ethnicity/shard-<ss>.parquet
    --stage merge      per bucket (needs gender + ALL shards): keys + probabilities  -> bucket-<bb>.parquet

Final output <output_dir>/bucket-<bb>.parquet (default results/groundtruth/demographics/) holds PROBABILITIES only:
    author_id, gender_first_name, p_female, p_asian, p_black, p_hispanic, p_white, p_other
Labels are assigned later with thresholds (A.4, batch_gt_population.py), so changing a threshold never re-runs a model.
keys/, gender/ and ethnicity/ are kept (gender/ and ethnicity/ are name -> probabilities lookups, e.g. for the
LLM-recommended names).

--cache_dir DIR [DIR ...] (stage ethnicity): reuse raceBERT probabilities of earlier ethnicity shards / demographics outputs
(bucket-<bb>.parquet and unfinished bucket-<bb>.parts/part-*.parquet, with author_id + p_asian ... p_other);
only the names not found there are inferred.

Usage (from any folder; run_gt_pipeline.sh -s demographics runs the four stages):
    python batch_gt_demographics.py --stage keys --limit_authors 500 --authors_dir <smoke>/authors --output_dir <smoke>/demographics
    python batch_gt_demographics.py --stage gender --output_dir <smoke>/demographics
    python batch_gt_demographics.py --stage ethnicity --output_dir <smoke>/demographics --torch_threads 3 [--cache_dir DIR]
    python batch_gt_demographics.py --stage merge --output_dir <smoke>/demographics
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
from pathlib import Path

from demographics import stages
from groundtruth import db
from groundtruth.fields import load_field_mapping
from utils import batch
from utils import constants as cons
from utils import ios
from utils import logs


def keys_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / stages.KEYS_DIR


def ethnicity_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / stages.ETHNICITY_DIR


def gender_file(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / stages.GENDER_DIR / stages.GENDER_FILE


def run_keys(bucket: int, args: argparse.Namespace) -> str:
    output_file = db.bucket_file(keys_dir(args), bucket)
    if output_file.exists() and not args.overwrite:
        return f"[bucket {bucket}] skipped (exists): {output_file}"
    keys = stages.write_keys_bucket(db.bucket_file(args.authors_dir, bucket), output_file, args.fields,
                                    args.n_shards, args.limit_authors)
    return (f"[bucket {bucket}] {len(keys)} authors -> {output_file.name} | no first name: "
            f"{int(keys['gender_name'].isna().sum())} | no raceBERT name: {int(keys['racebert_name'].isna().sum())}")


def run_gender(args: argparse.Namespace) -> None:
    output_file = gender_file(args)
    if output_file.exists() and not args.overwrite:
        ios.printf(f"Skipped (exists): {output_file}")
        return
    con = db.connect(args, ':memory:')
    names = stages.gender_names(con, keys_dir(args))
    con.close()
    from demographics.gender import GenderInferrer
    df = stages.write_gender_table(names, output_file, GenderInferrer().p_female)
    ios.printf(f"{len(df)} distinct first names -> {output_file} | not in the reference: {int(df['p_female'].isna().sum())}")


def run_ethnicity(shard: int, args: argparse.Namespace) -> str:
    output_file = stages.shard_file(ethnicity_dir(args), shard)
    if output_file.exists() and not args.overwrite:
        return f"[shard {shard}] skipped (exists): {output_file}"
    con = db.connect(args, ':memory:')
    names = stages.shard_names(con, keys_dir(args), shard)
    cached = stages.cached_probabilities(con, keys_dir(args), shard, args.cache_files)
    con.close()
    df = stages.infer_shard(names, output_file, lambda chunk: ethnicity_model(args).probabilities(chunk),
                            args.chunk_names, args.overwrite, cached=cached)
    return f"[shard {shard}] {len(df)} distinct names ({len(cached)} from the cache) -> {output_file.name}"


def ethnicity_model(args: argparse.Namespace):
    """Loaded on first use (not at all if every name is cached or every shard is done)."""
    if not hasattr(args, 'ethnicity'):
        from demographics.ethnicity import EthnicityInferrer
        args.ethnicity = EthnicityInferrer(batch_size=args.batch_size, torch_threads=args.torch_threads)
    return args.ethnicity


def run_merge(bucket: int, args: argparse.Namespace) -> str:
    output_file = db.bucket_file(args.output_dir, bucket)
    if output_file.exists() and not args.overwrite:
        return f"[bucket {bucket}] skipped (exists): {output_file}"
    con = db.connect(args, ':memory:')
    n = stages.write_merged_bucket(con, db.bucket_file(keys_dir(args), bucket), gender_file(args), ethnicity_dir(args),
                                   output_file)
    con.close()
    return f"[bucket {bucket}] {n} authors -> {output_file.name}"


def require_complete(paths, what: str, step: str) -> None:
    missing = stages.missing_files(paths)
    if missing:
        raise SystemExit(f"{what} incomplete: {len(missing)} missing (e.g. {missing[:3]}); run {step} first")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', required=True, choices=['keys', 'gender', 'ethnicity', 'merge'])
    db.add_bucket_arguments(parser)
    db.add_resource_arguments(parser)
    parser.add_argument('--n_shards', type=int, default=cons.DEFAULT_GT_NAME_SHARDS,
                        help='Distinct names are split into shards by crc32(name) (must be the same in every stage)')
    parser.add_argument('--shards', nargs='+', type=int, default=None,
                        help='Shards to process in --stage ethnicity (default: all); GNU parallel: --shards {}')
    parser.add_argument('--authors_dir', default=cons.DEFAULT_GT_AUTHORS_DIR, help='Output of step A.1')
    parser.add_argument('--field_mapping_file', default=cons.DEFAULT_FIELD_MAPPING_FILE)
    parser.add_argument('--batch_size', type=int, default=cons.DEFAULT_RACEBERT_BATCH_SIZE)
    parser.add_argument('--torch_threads', type=int, default=3, help='CPU threads for raceBERT per process')
    parser.add_argument('--chunk_names', type=int, default=cons.DEFAULT_GT_DEMOGRAPHICS_CHUNK,
                        help='Names per checkpoint part (a crashed run resumes after the last stored part)')
    parser.add_argument('--cache_dir', nargs='+', default=None,
                        help='Stage ethnicity: folders of earlier outputs whose probabilities are reused: ethnicity shards '
                             '(by name) and / or demographics buckets (by author_id -> name)')
    parser.add_argument('--limit_authors', type=int, default=None, help='Only the first N authors per bucket (smoke tests)')
    parser.add_argument('--output_dir', default=cons.DEFAULT_GT_DEMOGRAPHICS_DIR)
    logs.add_log_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == 'ethnicity':
        shards = args.shards if args.shards is not None else list(range(args.n_shards))
        bad = [s for s in shards if not 0 <= s < args.n_shards]
        if bad:
            raise SystemExit(f"Shards out of range [0, {args.n_shards}): {bad}")
        tag = 'all' if len(shards) == args.n_shards else f"shard{shards[0]:02d}" if len(shards) == 1 else f"{len(shards)}shards"
    else:
        tag = db.bucket_params(args)
    logs.setup_logging(f"gt_demographics_{args.stage}", args, tag)
    stages.check_layout(args.output_dir, args.n_buckets, args.n_shards, args.limit_authors)

    all_keys = [db.bucket_file(keys_dir(args), b) for b in range(args.n_buckets)]
    if args.stage == 'keys':
        args.fields = load_field_mapping(args.field_mapping_file)
        batch.run_jobs(db.buckets_of(args), run_keys, args)
    elif args.stage == 'gender':
        require_complete(all_keys, 'Keys (--stage keys)', '--stage keys')
        run_gender(args)
    elif args.stage == 'ethnicity':
        require_complete(all_keys, 'Keys (--stage keys)', '--stage keys')
        args.cache_files = stages.cache_files(args.cache_dir)
        if args.cache_dir:
            n_names = sum(stages.is_name_level(f) for f in args.cache_files)
            ios.printf(f"Cache: {n_names} name-level + {len(args.cache_files) - n_names} author-level files in "
                       f"{', '.join(args.cache_dir)}")
        batch.run_jobs(shards, run_ethnicity, args)
    else:
        require_complete(all_keys, 'Keys (--stage keys)', '--stage keys')
        require_complete([gender_file(args)], 'Gender table (--stage gender)', '--stage gender')
        require_complete([stages.shard_file(ethnicity_dir(args), s) for s in range(args.n_shards)],
                         'Ethnicity shards (--stage ethnicity)', '--stage ethnicity')
        batch.run_jobs(db.buckets_of(args), run_merge, args)
    ios.printf(f"Stage {args.stage} finished")


if __name__ == '__main__':
    main()
