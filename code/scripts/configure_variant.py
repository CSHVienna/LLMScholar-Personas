#!/usr/bin/env python3
"""
Analysis variants: a name you choose + a full analysis config.

    main            config/analysis.yaml
    <name>          config/variants/<name>.yaml   (the base config with some settings changed; header: variant, base,
                                                   differs = the changed settings, e.g. "jwfull=0.97, pf=0.2-0.8")

Outputs of a variant go to folders named after it (groundtruth/population/<name>, matching/authors/<name>,
metrics/<name>, plots/<name>). A stage whose settings equal the base's is not re-run: the variant reads main's
GT population (no A.4 setting changed) or main's author matching (no B.6b setting changed). The runners take
``-v <name>``; notebooks take ``VARIANT = '<name>'``.

Usage (from any folder):
    python configure_variant.py create jw097 --set matching.jw_full=0.97
    python configure_variant.py create pf02 --set pf=0.2,0.8 glabels=male,neutral,female    # codes work too
    python configure_variant.py list
    python configure_variant.py show jw097
    python configure_variant.py paths jw097 --results_dir tmp/smoke      # shell lines, used by the runners
"""

import _bootstrap  # noqa: F401  (adds code/libs to sys.path)
import argparse
import copy
import shlex
import sys
from pathlib import Path

from utils import analysis_config as ac
from utils import constants as cons

BY_CODE = {code: key for key, code in ac.CODES.items()}


def setting_key(text: str) -> tuple:
    """'section.key' or a code ('jwfull') -> (section, key)."""
    if text in BY_CODE:
        return BY_CODE[text]
    key = tuple(text.split('.', 1))
    if key not in ac.SETTINGS:
        raise SystemExit(f"Unknown setting {text!r}: use section.key (e.g. matching.jw_full) or a code "
                         f"({', '.join(sorted(BY_CODE))})")
    return key


def relative(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(cons.PROJECT_ROOT))
    except ValueError:
        return str(path)


def create(args: argparse.Namespace) -> None:
    name = ac.check_variant_name(args.name)
    if name == ac.MAIN:
        raise SystemExit("'main' is config/analysis.yaml: edit it with configure_analysis.py")
    base_path = ac.resolve_project_path(args.base)
    base = ac.load(base_path)
    values = copy.deepcopy(base)
    for item in args.set:
        if '=' not in item:
            raise SystemExit(f"--set {item!r}: use setting=value")
        text, value = item.split('=', 1)
        section, key = setting_key(text.strip())
        try:
            values[section][key] = ac.parse_value(ac.SETTINGS[(section, key)], value.replace(',', ' ')
                                                  if ac.SETTINGS[(section, key)].kind != 'strings' else value)
        except ValueError as error:
            raise SystemExit(f"{section}.{key}: {error}")
    problems = ac.validate(values)
    if problems:
        raise SystemExit("Not a valid config:\n  " + "\n  ".join(problems))
    changes = ac.differences(base, values)
    if not changes:
        raise SystemExit("No setting differs from the base: that is the base itself")
    shared = [k for step, keys in ac.changed_steps(base, values).items() if step in ac.SHARED_STEPS for k in keys]
    if shared:
        raise SystemExit(f"{shared} must equal the base: the candidate step is shared by every variant")

    path = ac.variant_file(name)
    if path.exists():
        if ac.load(path) == values:
            print(f"{path} exists with these settings: kept")
            show_variant(name)
            return
        if not args.overwrite:
            raise SystemExit(f"{path} exists with other settings ({ac.variant_info(name)['differs']}). Choose another "
                             f"name, or --overwrite (its outputs would then mix settings: delete them first)")
    for other in ac.list_variants():
        if other not in (ac.MAIN, name) and ac.load(ac.variant_file(other)) == values:
            print(f"WARNING: variant {other!r} already has exactly these settings")
    ac.save(path, values, reviewed=ac.late_reviewed(base_path), variant={'name': name, 'base': relative(base_path)})
    print(f"Wrote {path}")
    show_variant(name)


def show_variant(name: str, results_dir: str = cons.DEFAULT_RESULTS_DIR) -> None:
    info = ac.variant_info(name, results_dir)
    print(f"variant  {name}\nconfig   {info['config']}\nbase     {info['base'] or '-'}\ndiffers  {info['differs'] or '-'}")
    for key, old, new in info['changes']:
        print(f"           {key}: {ac.format_value(old)} -> {ac.format_value(new)}")
    own = lambda stage: 'own' if info[stage] == name else f"reuses {info[stage]}"
    print(f"GT population (A.4)    {own('population'):12} {info['gt_dir']}")
    print(f"author matching (B.6b) {own('authors'):12} {info['matching_dir']}")
    print(f"metrics                {'own':12} {info['metrics_dir']}")
    print(f"plots                  {'own':12} {info['plots_dir']}")
    steps = (["bash code/scripts/run_gt_pipeline.sh -s population -v " + name] if info['population'] == name else []) \
        + (["bash code/scripts/run_matching_pipeline.sh -s authors -v " + name] if info['authors'] == name else []) \
        + ["bash code/scripts/run_metrics_pipeline.sh -v " + name]
    print("run, in this order:\n  " + "\n  ".join(steps))


def list_all(args: argparse.Namespace) -> None:
    for name in ac.list_variants():
        info = ac.variant_info(name)
        own = [stage for stage in ('population', 'authors') if info[stage] == name and name != ac.MAIN]
        print(f"{name:24} {info['differs'] or '(base)':50} own: {', '.join(own + ['metrics'])}")


def paths(args: argparse.Namespace) -> None:
    """Shell assignments for the runners (eval)."""
    try:
        info = ac.variant_info(args.name, args.results_dir, args.config if args.name == ac.MAIN else None)
    except (ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    if args.config and args.name != ac.MAIN:
        print(f"ERROR: -y only replaces the config of main; variant {args.name} has its own ({info['config']})",
              file=sys.stderr)
        raise SystemExit(1)
    for key, value in [('VARIANT_CONFIG', info['config']), ('VARIANT_POPULATION', info['population']),
                       ('VARIANT_AUTHORS', info['authors']), ('VARIANT_METRICS', info['name'])]:
        print(f"{key}={shlex.quote(value)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('create', help='Write config/variants/<name>.yaml')
    p.add_argument('name')
    p.add_argument('--set', nargs='+', required=True, metavar='SETTING=VALUE',
                   help='section.key=value or code=value; lists comma-separated (pf=0.2,0.8)')
    p.add_argument('--base', default=ac.MAIN_CONFIG_REL, help='Config the variant starts from')
    p.add_argument('--overwrite', action='store_true', help='Replace a variant with other settings')
    p.set_defaults(func=create)
    p = sub.add_parser('list', help='All variants and what differs')
    p.set_defaults(func=list_all)
    p = sub.add_parser('show', help='One variant: settings, folders, commands')
    p.add_argument('name')
    p.add_argument('--results_dir', default=cons.DEFAULT_RESULTS_DIR)
    p.set_defaults(func=lambda a: show_variant(a.name, a.results_dir))
    p = sub.add_parser('paths', help='Shell lines for the runners')
    p.add_argument('name')
    p.add_argument('--results_dir', default=cons.DEFAULT_RESULTS_DIR)
    p.add_argument('--config', default=None, help='Config file of main (e.g. the smoke config)')
    p.set_defaults(func=paths)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"ERROR: {error}")


if __name__ == '__main__':
    main()
