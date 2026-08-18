#!/usr/bin/env python3
"""
Export the auto-generated tables and figures of a finished experiment into
the paper (MDPI___Raimoc/tables/ and MDPI___Raimoc/figures/), where
results.tex picks them up via \\InputIfFileExists on the next compile.

Mapping (source → paper asset):
    <exp>/summary/table_main.tex               → tables/table_main.tex
    <exp>/summary/table_uncertainty.tex        → tables/table_uncertainty.tex
    <exp>/summary/table_domain_gap.tex         → tables/table_domain_gap.tex
    <exp>/summary/test_rmse_by_env.png         → figures/fig_results_rmse_by_env.png
    <exp>/baselines/summary/table_baselines_rmse_hole.tex
                                               → tables/table_baselines_rmse_hole.tex
    <exp>/baselines/summary/table_baselines_rmse.tex
                                               → tables/table_baselines_rmse.tex
    <exp>/fold_<k>_<Env>/eval/calibration_test.png
                                               → figures/fig_results_calibration.png
    reports/table_compute_<model>.tex          → tables/table_compute.tex

Optional (comparison tables produced by scripts/eval/compare_experiments.py):
    --arch_table  <path/compare_test_nll.tex>  → tables/table_architectures.tex
    --aug_table   <path/compare_test_nll.tex>  → tables/table_abl_augmentation.tex
    --loss_table  <path/compare_test_nll.tex>  → tables/table_abl_loss.tex

Usage:
    python tools/export_paper_assets.py --exp_dir runs/cv_unet_beta05_x
    python tools/export_paper_assets.py --exp_dir runs/cv_x --calib_fold 2 \\
        --aug_table reports/abl_aug/compare_test_nll.tex
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
from elevcomp.paths import project_root

ROOT = project_root()
PAPER = ROOT / 'results'


def copy(src: Path, dst: Path) -> bool:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        print(f'  ✓ {src.relative_to(ROOT)}  →  {dst.relative_to(PAPER)}')
        return True
    print(f'  ✗ missing: {src.relative_to(ROOT)}')
    return False


def main():
    p = argparse.ArgumentParser(
        description='Copy experiment tables/figures into the paper.')
    p.add_argument('--exp_dir', required=True,
                   help='finished CV experiment directory')
    p.add_argument('--calib_fold', type=int, default=0,
                   help='fold whose calibration figure is shown in the paper')
    p.add_argument('--arch_table', help='compare_experiments output for the '
                   'architecture comparison')
    p.add_argument('--aug_table', help='compare_experiments output for the '
                   'augmentation ablation')
    p.add_argument('--loss_table', help='compare_experiments output for the '
                   'loss ablation')
    args = p.parse_args()

    exp = Path(args.exp_dir)
    if not exp.is_absolute():
        exp = ROOT / exp
    if not (exp / 'experiment.json').exists():
        sys.exit(f'{exp} is not an experiment directory (no experiment.json).')
    model = json.load(open(exp / 'experiment.json'))['config'].get('model', 'unet')

    tables = PAPER / 'tables'
    figures = PAPER / 'figures'
    n_ok = 0

    print(f'Experiment: {exp.name}')
    print('── cross-fold summary ──')
    n_ok += copy(exp / 'summary/table_main.tex', tables / 'table_main.tex')
    n_ok += copy(exp / 'summary/table_uncertainty.tex',
                 tables / 'table_uncertainty.tex')
    n_ok += copy(exp / 'summary/table_domain_gap.tex',
                 tables / 'table_domain_gap.tex')
    n_ok += copy(exp / 'summary/test_rmse_by_env.png',
                 figures / 'fig_results_rmse_by_env.png')

    print('── baselines ──')
    n_ok += copy(exp / 'baselines/summary/table_baselines_rmse_hole.tex',
                 tables / 'table_baselines_rmse_hole.tex')
    n_ok += copy(exp / 'baselines/summary/table_baselines_rmse.tex',
                 tables / 'table_baselines_rmse.tex')

    print('── calibration figure ──')
    folds = json.load(open(exp / 'folds.json'))
    fd = next(f for f in folds if f['fold'] == args.calib_fold)
    n_ok += copy(exp / f'fold_{fd["fold"]}_{fd["test_env"]}/eval/calibration_test.png',
                 figures / 'fig_results_calibration.png')

    print('── computational cost ──')
    n_ok += copy(ROOT / f'reports/table_compute_{model}.tex',
                 tables / 'table_compute.tex')

    print('── comparison tables (optional) ──')
    for arg, name in ((args.arch_table, 'table_architectures.tex'),
                      (args.aug_table, 'table_abl_augmentation.tex'),
                      (args.loss_table, 'table_abl_loss.tex')):
        if arg:
            n_ok += copy(Path(arg) if Path(arg).is_absolute() else ROOT / arg,
                         tables / name)
        else:
            print(f'  – {name}: not requested (pass the matching --*_table)')

    print(f'\n{n_ok} assets exported. Recompile the paper:')
    print('  cd MDPI___Raimoc && ~/.local/bin/tectonic -X compile template.tex')
    print('Then replace the remaining red \\pending{...} markers in results.tex')
    print(f'with numbers from {exp.name}/summary/summary.json.')


if __name__ == '__main__':
    main()
