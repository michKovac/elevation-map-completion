#!/usr/bin/env python3
"""
Ablation-study runner: one full 5-fold CV experiment per config variant.

Variants are declared in a TOML plan (default: configs/ablations.toml) — each
[[variant]] carries a name and a set of configs/cv_resnet34.toml overrides. The runner
creates the experiments on first use and records them in
runs/ablations/<plan>/registry.json; re-running the same plan CONTINUES the
existing experiments (skipping finished folds), so the plan can be executed
incrementally and split across GPUs like any single experiment.

Usage:
    python run_ablations.py --list                      # show plan + status
    python run_ablations.py                             # run everything, sequentially
    python run_ablations.py --variants aug_none aug_geo # subset of variants
    CUDA_VISIBLE_DEVICES=0 python run_ablations.py --folds 0 1 2
    CUDA_VISIBLE_DEVICES=1 python run_ablations.py --folds 3 4
    python run_ablations.py --stage aggregate           # summaries only

Compare finished variants:
    python scripts/eval/compare_experiments.py runs/cv_abl_* --out reports/ablations
"""
import argparse
import json
import sys
import tomllib
from pathlib import Path
from elevcomp.paths import CONFIG_DIR

from elevcomp.cv import HPARAM_OVERRIDES, create_experiment, run_experiment


def load_plan(path: Path) -> dict:
    with open(path, 'rb') as f:
        plan = tomllib.load(f)
    name = plan.get('plan', {}).get('name', path.stem)
    defaults = plan.get('plan', {}).get('defaults', {})
    variants = plan.get('variant', [])
    if not variants:
        sys.exit(f'No [[variant]] entries in {path}')
    seen = set()
    for v in variants:
        if 'name' not in v:
            sys.exit('Every [[variant]] needs a name.')
        if v['name'] in seen:
            sys.exit(f'Duplicate variant name: {v["name"]}')
        seen.add(v['name'])
        overrides = {**defaults, **v.get('overrides', {})}
        bad = [k for k in overrides if k not in HPARAM_OVERRIDES]
        if bad:
            sys.exit(f'Variant {v["name"]}: overrides {bad} are not in '
                     f'HPARAM_OVERRIDES (core/cv.py).')
        v['resolved_overrides'] = overrides
    return {'name': name, 'variants': variants}


def fold_status(exp_dir: Path) -> str:
    folds = json.load(open(exp_dir / 'folds.json'))
    trained = sum((exp_dir / f'fold_{f["fold"]}_{f["test_env"]}'
                   / 'train_done.json').exists() for f in folds)
    evald = sum((exp_dir / f'fold_{f["fold"]}_{f["test_env"]}'
                 / 'eval_done.json').exists() for f in folds)
    return f'{trained}/{len(folds)} trained, {evald}/{len(folds)} evaluated'


def main():
    p = argparse.ArgumentParser(description='Run an ablation plan as CV experiments.')
    p.add_argument('--plan', default=str(CONFIG_DIR / 'ablations.toml'))
    p.add_argument('--variants', nargs='+', help='subset of variant names')
    p.add_argument('--folds', type=int, nargs='+', help='subset of folds')
    p.add_argument('--stage', default='all',
                   choices=['all', 'train', 'eval', 'aggregate'])
    p.add_argument('--list', action='store_true', help='show plan + status, exit')
    p.add_argument('--data_root',
                   default=str(Path(__file__).parent / '../datasets/elevation_dataset'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--debug', type=int, default=0, metavar='N',
                   help='N files per trajectory (smoke test, marks _DEBUG)')
    p.add_argument('--num_workers', type=int)
    args = p.parse_args()

    plan = load_plan(Path(args.plan))
    reg_dir = Path(__file__).parent / 'runs' / 'ablations' / plan['name']
    reg_dir.mkdir(parents=True, exist_ok=True)
    reg_path = reg_dir / 'registry.json'
    registry = json.load(open(reg_path)) if reg_path.exists() else {}

    sel = args.variants or [v['name'] for v in plan['variants']]
    unknown = set(sel) - {v['name'] for v in plan['variants']}
    if unknown:
        sys.exit(f'Unknown variants: {sorted(unknown)}')

    if args.list:
        print(f'Plan: {plan["name"]}  ({args.plan})')
        for v in plan['variants']:
            exp = registry.get(v['name'])
            status = (fold_status(Path(exp)) if exp and Path(exp).exists()
                      else 'not created')
            print(f'  {v["name"]:<16} {status:<32} {v.get("description", "")}')
            print(f'  {"":<16} overrides: {v["resolved_overrides"]}')
        return

    for v in plan['variants']:
        if v['name'] not in sel:
            continue
        print(f'\n{"█" * 70}\nVARIANT: {v["name"]} — {v.get("description", "")}'
              f'\n{"█" * 70}')

        exp_dir = Path(registry[v['name']]) if v['name'] in registry else None
        if exp_dir is None or not exp_dir.exists():
            exp_dir = create_experiment(
                name=f'{plan["name"]}_{v["name"]}',
                overrides=v['resolved_overrides'],
                data_root=args.data_root, seed=args.seed,
                debug_limit=args.debug,
                command=' '.join(sys.argv))
            registry[v['name']] = str(exp_dir)
            with open(reg_path, 'w') as f:
                json.dump(registry, f, indent=2)

        run_experiment(exp_dir, folds_sel=args.folds, stage=args.stage,
                       num_workers=args.num_workers)

    print(f'\nRegistry: {reg_path}')
    print('Compare finished variants with:\n'
          f'  python scripts/eval/compare_experiments.py '
          f'{" ".join(registry.get(n, "?") for n in sel)} --out reports/ablations')


if __name__ == '__main__':
    main()
