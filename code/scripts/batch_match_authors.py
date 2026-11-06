#!/usr/bin/env python3
"""
Step B.6b: author matching of every recommendation (libs/matching/decide.py), from the B.6a candidates and the
A.1 bibliometrics, with the thresholds given here (cheap: re-run whenever a threshold changes; the thresholds are
recorded in <output_dir>/_params.json, and outputs with other thresholds are refused: use another --output_dir).

Writes <output_dir>/<model>.parquet (default results/matching/authors/main/; other thresholds: authors/<variant>/), one row per recommendation:
    response identifiers + rec_index, first_name, last_name, rec_name,
    match_rule (strictest rule with a match), n_rule_<rule> (candidates of every rule), jw_full_max, jw_parts_max,
    n_name, n_name_field, n_name_seniority, n_author, matched_name, matched_name_field, matched_name_seniority, matched_author,
    author_id (random among the n_author people), matched_author_field / author_id_field,
    matched_author_seniority / author_id_seniority

Usage (from any folder):
    python batch_match_authors.py --models gpt-4_1-nano-2025-04-14 --candidates_dir <smoke>/candidates --output_dir <smoke>/matching
    parallel -j 10 python batch_match_authors.py --models {} :::: data/models/models.txt
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
import json
from pathlib import Path

from matching import candidates as mc
from matching.decide import write_matches
from groundtruth import db
from groundtruth.authorships import check_params
from groundtruth.fields import load_field_mapping
from utils import batch
from utils import analysis_config as ac
from utils import constants as cons


def run_model(model: str, args: argparse.Namespace) -> str:
    output_file = batch.model_file(args.output_dir, model)
    if output_file.exists() and not args.overwrite:
        return f"[{model}] skipped (exists)"
    recommendations_file = batch.model_file(args.recommendations_metadata_dir, model)
    if not recommendations_file.exists():
        return f"[{model}] no recommendations file: {recommendations_file}"
    con = db.connect(args, ':memory:')
    n = write_matches(con, output_file, recommendations_file=recommendations_file,
                         rec_names_file=Path(args.candidates_dir) / mc.REC_NAMES_FILE,
                         candidates_glob=str(Path(args.candidates_dir) / 'bucket-*.parquet'),
                         authors_glob=str(Path(args.authors_dir) / 'bucket-*.parquet'), fields=args.fields,
                         seed=args.seed, jw_full_min=args.jw_full, jw_first_min=args.jw_first, jw_last_min=args.jw_last,
                         min_field_share=args.min_field_share, seniority_definition=args.seniority_definition,
                         reference_year=args.reference_year, junior_max=args.junior_max, senior_min=args.senior_min,
                         single_cutoff=args.single_cutoff)
    stats = con.execute(f"SELECT avg(matched_name::INT), avg(matched_author::INT) FROM read_parquet({db.sql_path(output_file)})").fetchone()
    con.close()
    return f"[{model}] {n} recommendations | matched_name {stats[0]:.3f} | matched_author {stats[1]:.3f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    batch.add_model_arguments(parser)
    db.add_resource_arguments(parser)
    parser.add_argument('--candidates_dir', default=cons.DEFAULT_MATCH_CANDIDATES_DIR, help='Output of B.6a')
    parser.add_argument('--recommendations_metadata_dir', default=cons.DEFAULT_RECOMMENDATIONS_METADATA_DIR, help='Output of B.3')
    parser.add_argument('--authors_dir', default=cons.DEFAULT_GT_AUTHORS_DIR, help='Output of A.1')
    parser.add_argument('--field_mapping_file', default=cons.DEFAULT_FIELD_MAPPING_FILE)
    parser.add_argument('--output_dir', default=cons.DEFAULT_MATCH_AUTHORS_DIR,
                        help='One folder per threshold variant: results/matching/authors/<variant>/')
    parser.add_argument('--seed', type=int, default=cons.DEFAULT_SEED, help='Seed of the random pick among homonyms')
    parser.add_argument('--jw_full', type=float, default=cons.DEFAULT_MATCH_JW_FULL, help='Fuzzy rule: minimum jw_full')
    parser.add_argument('--jw_first', type=float, default=cons.DEFAULT_MATCH_JW_FIRST,
                        help='Fuzzy rule: minimum jw_first (provisional default; 0 = not used)')
    parser.add_argument('--jw_last', type=float, default=cons.DEFAULT_MATCH_JW_LAST,
                        help='Fuzzy rule: minimum jw_last (provisional default; 0 = not used)')
    parser.add_argument('--min_field_share', type=float, default=cons.DEFAULT_GT_MIN_FIELD_SHARE,
                        help='Field level: minimum share of the candidate\'s works in the requested field')
    parser.add_argument('--seniority_definition', choices=[cons.MATCH_SENIORITY_THREE, cons.MATCH_SENIORITY_SINGLE],
                        default=cons.MATCH_SENIORITY_THREE, help='Seniority definition (as in A.4)')
    parser.add_argument('--reference_year', type=int, default=cons.GT_REFERENCE_YEAR, help='academic_age = year - first_pub_year')
    parser.add_argument('--junior_max', type=int, default=cons.DEFAULT_SENIORITY_JUNIOR_MAX)
    parser.add_argument('--senior_min', type=int, default=cons.DEFAULT_SENIORITY_SENIOR_MIN)
    parser.add_argument('--single_cutoff', type=int, default=cons.DEFAULT_SENIORITY_SINGLE_CUTOFF)
    ac.add_config_argument(parser)
    return parser


def main() -> None:
    args = ac.parse_with_config(build_parser())
    params_file = Path(args.candidates_dir) / '_params.json'
    if not params_file.exists():
        raise SystemExit(f"No B.6a output in {args.candidates_dir} (run batch_match_candidates.py first)")
    b6a = json.loads(params_file.read_text())
    if args.jw_full < b6a['jw_floor']:
        raise SystemExit(f"--jw_full {args.jw_full} < the B.6a jw_floor {b6a['jw_floor']}: re-run B.6a with a lower floor")
    missing = [b for b in range(b6a['n_buckets']) if not db.bucket_file(args.candidates_dir, b).exists()]
    if missing:
        raise SystemExit(f"B.6a incomplete: candidate buckets {missing[:5]} ... missing")
    check_params(args.output_dir, {key: getattr(args, key) for key in
                                   ['seed', 'jw_full', 'jw_first', 'jw_last', 'min_field_share', 'seniority_definition', 'reference_year',
                                    'junior_max', 'senior_min', 'single_cutoff']})
    args.fields = load_field_mapping(args.field_mapping_file)
    batch.run_per_model('match_authors', run_model, args)


if __name__ == '__main__':
    main()
