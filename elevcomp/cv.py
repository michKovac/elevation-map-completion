"""
Cross-environment 5-fold experiment driver (library part).

Driven by scripts/train.py.

Public API:
    load_base_config()   — base TOML + override dict → frozen experiment config
    create_experiment()  — new experiment directory with config/folds snapshot
    run_experiment()     — train/eval selected folds + cross-fold aggregation
    aggregate()          — (re)build summary tables and figures

Fold design and persistence layout are documented in core/folds.py and in the
auto-generated README.md inside every experiment directory.
"""
import json
import time
import tomllib
from datetime import datetime
from pathlib import Path
from elevcomp.paths import CONFIG_DIR

import numpy as np
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .calibration import (MIN_HOLE_PX, PIX_SUBSAMPLE, SPARS_FRACS, StreamingCorr,
                          calib_sample)
from .figures import (fig_calibration, fig_learning_curves, fig_test_by_env,
                      save_val_images, visualize_uncertainty)
from .folds import build_folds, discover_environments, eval_loader, make_datasets
from .inference import nll_predict, tta_predict
from .losses import ElevationLoss
from .metrics import compute_metrics
from .model import build_model, count_parameters
from .training import evaluate, make_state, save_checkpoint, train_one_epoch
from .utils import (aggregate_rows, environment_info, nanmean, nanstd,
                    seed_everything, worker_init, write_csv)

# Config keys that may be overridden when CREATING an experiment (CLI or
# ablation plan). Afterwards the config is frozen in experiment.json.
HPARAM_OVERRIDES = [
    # optimization
    'epochs', 'batch_size', 'lr', 'weight_decay', 'grad_clip', 'amp', 'amp_dtype',
    # loss
    'w_valid', 'w_hole', 'uncertainty', 'uncertainty_beta', 'uncertainty_warmup',
    # augmentation
    'aug_hflip_p', 'aug_vflip_p', 'aug_rot90_p',
    'aug_ray_p', 'aug_ray_n_max', 'aug_ray_angle_deg',
]

MAIN_METRICS = ['rmse', 'mae', 'rmse_hole', 'mae_hole', 'rmse_comp',
                'mae_comp', 'abs_rel', 'ssim']
UNC_METRICS = ['pearson', 'cov68', 'mean_sigma']
GAP_METRICS = ['rmse', 'rmse_hole', 'mae', 'mae_hole']   # domain-gap table


# ─────────────────────────────────────────────────────────────────────────────
# Experiment setup
# ─────────────────────────────────────────────────────────────────────────────

def load_base_config(overrides: dict = None) -> dict:
    """configs/default.toml + overrides → experiment config."""
    cfg_path = CONFIG_DIR / 'cv_resnet34.toml'
    if not cfg_path.exists():
        raise FileNotFoundError(
            f'{cfg_path} not found.')
    with open(cfg_path, 'rb') as f:
        cfg = tomllib.load(f)
    # Defensive: these single-run keys have no meaning in the CV experiment
    # (data comes from fold file lists, folds always start from scratch).
    cfg['resume'] = None
    cfg.pop('data_dir', None)
    for k, v in (overrides or {}).items():
        if k not in cfg and k not in HPARAM_OVERRIDES and k != 'num_workers':
            raise KeyError(f'Unknown config override: {k!r}')
        cfg[k] = v
    return cfg


README_TMPL = """# Experiment: {name}

Created: {created}
5-fold leave-one-environment-out cross-validation for elevation map completion.

## Fold design
- Fold k tests on environment k (all its trajectories) — never seen in training.
- Validation = last (sorted) {vt} trajectory/ies of EACH training environment.
- Normalization stats computed from TRAIN files only (see fold_*/stats.json).
- Exact file lists: folds.json (experiment level) and fold_*/split_files.json.
- Seeds: base {seed}; fold seed = base + 1000·fold (weights, loaders, subsampling).

## Layout
- experiment.json — resolved config + CLI + library versions + GPU info
- fold_k_<Env>/
  - config.json, stats.json, split_files.json, history.json (per-epoch), tb/
  - best.pth (lowest val RMSE), last.pth, epoch_*.pth
  - vis/ — training-progress grids
  - eval/
    - metrics_{{train,val,test}}.json — mean/std/n over samples
    - per_sample_{{train,val,test}}.csv — one row per sample (test: both methods,
      calibration columns, path of the saved prediction)
    - calibration_test.json — pooled Pearson/coverage, mean sparsification curves
    - calibration_pixels_test.npz — subsampled (σ, |err|) hole pixels [m], float16
    - learning_curves.png, calibration_test.png, test_samples.png
  - predictions/test/<traj>/<sample>.npz — pred_nll, sigma_nll, pred_tta, sigma_tta
    (normalized float16, aligned to the source NPZ; see predictions/meta.json)
- baselines/ — classical baseline results (created by run_baselines.py)
- summary/ — summary.json, summary_folds.csv, table_main.tex,
  table_uncertainty.tex, table_domain_gap.tex, test_rmse_by_env.png

## Reload a test prediction
```python
import json, numpy as np
fold = 'fold_0_ForestEnv'
meta  = json.load(open(f'{{fold}}/predictions/meta.json'))
m, s  = meta['stats']['mean'], meta['stats']['std']
p     = np.load(f'{{fold}}/predictions/test/P2001/P2001_sample_000000.npz')
src   = np.load('<data_root>/ForestEnv/P2001/P2001_sample_000000.npz')
pred_m  = p['pred_nll'].astype(np.float32) * s + m   # metres, same grid as src
sigma_m = p['sigma_nll'].astype(np.float32) * s
```

## Metrics conventions
- All metrics in denormalized metres; aggregate = mean over samples (± std).
- 'hole' = gt_mask=1 & partial_mask=0; 'comp' = stereo kept where measured.
- SSIM is masked: averaged over pixels whose full 11×11 window is GT-valid.
- Calibration on hole pixels: Pearson(σ,|e|), coverage σ>|e| (→68 %), AUSE
  (mean gap between σ- and oracle-sorted normalized sparsification curves).
- history.json val numbers are batch-averaged (training-time); the eval/
  numbers are the canonical ones for the paper.
{debug_note}"""


def create_experiment(name: str, overrides: dict, data_root: Path,
                      seed: int = 42, val_traj_per_env: int = 1,
                      debug_limit: int = 0, command: str = '') -> Path:
    """Create a new experiment directory with config snapshot + fold definitions."""
    cfg = load_base_config(overrides)

    data_root = Path(data_root).resolve()
    envs = discover_environments(data_root)
    folds = build_folds(envs, val_traj_per_env, debug_limit)

    name = name + ('_DEBUG' if debug_limit else '')
    save_dir = Path(cfg['save_dir'])
    if not save_dir.is_absolute():
        save_dir = Path(__file__).parent.parent / save_dir
    exp_dir = save_dir / f'cv_{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    exp_dir.mkdir(parents=True)

    cv = {
        'name': name,
        'data_root': str(data_root),
        'environments': {e: sorted(t) for e, t in envs.items()},
        'n_files': {e: sum(len(v) for v in t.values()) for e, t in envs.items()},
        'val_traj_per_env': val_traj_per_env,
        'val_rule': 'last sorted trajectory/ies of each training environment',
        'seed': seed,
        'debug_limit': debug_limit,
        'created': datetime.now().isoformat(timespec='seconds'),
        'command': command,
        'overrides': overrides,
    }
    with open(exp_dir / 'experiment.json', 'w') as f:
        json.dump({'name': name, 'config': cfg, 'cv': cv,
                   'environment_info': environment_info()}, f, indent=2)
    with open(exp_dir / 'folds.json', 'w') as f:
        json.dump(folds, f)

    debug_note = ('\n**DEBUG RUN** — only {} files per trajectory; '
                  'not for the paper.\n'.format(debug_limit) if debug_limit else '')
    (exp_dir / 'README.md').write_text(README_TMPL.format(
        name=name, created=cv['created'], vt=val_traj_per_env,
        seed=seed, debug_note=debug_note))

    print(f'Created experiment: {exp_dir}')
    for fd in folds:
        print(f'  fold {fd["fold"]}: test={fd["test_env"]:<22} '
              f'train {fd["n_train"]}, val {fd["n_val"]}, test {fd["n_test"]}')
    return exp_dir


# ─────────────────────────────────────────────────────────────────────────────
# Training one fold  (single GPU, no DDP — parallelize folds across GPUs)
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(cfg: dict, cv: dict, fold: dict, fold_dir: Path, device):
    fold_seed = cv['seed'] + 1000 * fold['fold']
    seed_everything(fold_seed)
    torch.backends.cudnn.benchmark = True

    fold_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = fold_dir / 'vis'

    # Persist the exact split of this fold for standalone reproducibility.
    with open(fold_dir / 'split_files.json', 'w') as f:
        json.dump({k: fold[k] for k in
                   ('fold', 'test_env', 'val_trajectories',
                    'train_files', 'val_files', 'test_files')}, f)

    # Normalisation stats are expensive (reads every train GT) — cache them.
    stats_path = fold_dir / 'stats.json'
    stats = json.load(open(stats_path)) if stats_path.exists() else None

    t_data = time.time()
    train_ds, val_ds, _, stats = make_datasets(cfg, fold, Path(cv['data_root']), stats)
    if not stats_path.exists():
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    print(f'  data ready in {time.time() - t_data:.0f}s — '
          f'train {len(train_ds)}, val {len(val_ds)} | '
          f'stats: mean {stats["mean"]:.3f}, std {stats["std"]:.3f}')

    gen = torch.Generator().manual_seed(fold_seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=cfg['num_workers'], pin_memory=True,
        persistent_workers=(cfg['num_workers'] > 0),
        drop_last=(len(train_ds) >= cfg['batch_size']),
        generator=gen, worker_init_fn=worker_init,
    )
    val_loader = eval_loader(val_ds, cfg)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                                  weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'], eta_min=cfg['lr'] * 0.01)

    # AMP precision. fp16 + GradScaler is the default; bf16 (no scaler, full
    # exponent range) is available for numerically unstable configurations.
    amp_on = cfg.get('amp', True)
    amp_dtype_str = str(cfg.get('amp_dtype', 'fp16')).lower()
    if not amp_on:
        autocast_dtype, scaler = None, None
    elif amp_dtype_str in ('bf16', 'bfloat16'):
        autocast_dtype, scaler = torch.bfloat16, None
    else:
        autocast_dtype, scaler = torch.float16, GradScaler('cuda')
    uncertainty = cfg.get('uncertainty', False)
    unc_warmup = cfg.get('uncertainty_warmup', 0) if uncertainty else 0
    loss_fn = ElevationLoss(cfg['w_valid'], cfg['w_hole'],
                            beta=cfg.get('uncertainty_beta', 0.5))

    fold_cfg = {**cfg, 'fold': fold['fold'], 'test_env': fold['test_env'],
                'fold_seed': fold_seed}
    with open(fold_dir / 'config.json', 'w') as f:
        json.dump(fold_cfg, f, indent=2)

    # Auto-resume an interrupted fold.
    start_epoch, best_rmse, best_epoch, history = 0, float('inf'), -1, []
    last_ckpt = fold_dir / 'last.pth'
    if last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if scaler is not None and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_rmse = ckpt.get('best_rmse', float('inf'))
        hist_path = fold_dir / 'history.json'
        if hist_path.exists():
            history = json.load(open(hist_path))[:start_epoch]
            best_epoch = min(
                (r['epoch'] for r in history if r['rmse'] <= best_rmse),
                default=-1,
            )
        print(f'  resumed from epoch {start_epoch} (best RMSE {best_rmse:.4f} m)')

    amp_mode = 'off' if autocast_dtype is None else str(autocast_dtype).split('.')[-1]
    print(f'  model {cfg["model"]} — {count_parameters(model):,} params | '
          f'{cfg["epochs"]} epochs | uncertainty: {uncertainty} (warmup {unc_warmup}) '
          f'| amp: {amp_mode}')

    writer = SummaryWriter(fold_dir / 'tb')
    t_fold = time.time()

    for epoch in range(start_epoch, cfg['epochs']):
        t0 = time.time()
        use_unc = uncertainty and (epoch >= unc_warmup)

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn,
                                     scaler, device, None, epoch,
                                     cfg['grad_clip'], use_unc, autocast_dtype)
        val_m = evaluate(model, val_loader, loss_fn, device, stats, use_unc)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'lr': lr_now, 'time_s': elapsed, **val_m})

        print(f"  ep {epoch+1:03d}/{cfg['epochs']} | train {train_loss:.4f} | "
              f"val rmse {val_m['rmse']:.3f}m mae {val_m['mae']:.3f}m"
              f"{'  rmse_hole %.3fm' % val_m['rmse_hole'] if 'rmse_hole' in val_m else ''}"
              f" | lr {lr_now:.2e} | {elapsed:.1f}s")

        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/lr', lr_now, epoch)
        for k, v in val_m.items():
            writer.add_scalar(f'val/{k}', v, epoch)

        new_best = val_m['rmse'] < best_rmse
        if new_best:
            best_rmse, best_epoch = val_m['rmse'], epoch

        state = make_state(epoch, model, optimizer, scheduler, scaler,
                           best_rmse, stats, fold_cfg)
        save_checkpoint(state, fold_dir / 'last.pth')
        if new_best:
            save_checkpoint(state, fold_dir / 'best.pth')
        if (epoch + 1) % cfg['save_every'] == 0:
            save_checkpoint(state, fold_dir / f'epoch_{epoch+1:03d}.pth')

        if (epoch + 1) % cfg.get('vis_every', 25) == 0 or epoch == 0:
            save_val_images(model, val_loader, device, stats, epoch,
                            vis_dir, cfg.get('n_vis', 8), use_unc, writer)

        with open(fold_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    writer.close()
    done = {
        'best_val_rmse': best_rmse,
        'best_epoch': best_epoch,
        'epochs': cfg['epochs'],
        'wall_time_s': time.time() - t_fold,
        'finished_at': datetime.now().isoformat(timespec='seconds'),
    }
    with open(fold_dir / 'train_done.json', 'w') as f:
        json.dump(done, f, indent=2)
    print(f'  fold {fold["fold"]} training done — best val RMSE {best_rmse:.4f} m '
          f'@ epoch {best_epoch + 1}')


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation of one trained fold
# ─────────────────────────────────────────────────────────────────────────────

def per_sample_metrics(pred, gt, gm, pm, pe, stats) -> list:
    """compute_metrics on every sample of a batch individually."""
    rows = []
    for i in range(pred.shape[0]):
        rows.append(compute_metrics(
            pred[i:i + 1], gt[i:i + 1], gm[i:i + 1], pm[i:i + 1], pe[i:i + 1],
            denorm_std=stats['std'], denorm_mean=stats['mean']))
    return rows


@torch.no_grad()
def eval_split_simple(model, ds, cfg, stats, device, uncertainty, data_root):
    """Single-forward-pass per-sample metrics for train/val splits."""
    loader = eval_loader(ds, cfg)
    rows, idx = [], 0
    for inp, gt, gm in loader:
        inp = inp.to(device, non_blocking=True)
        gt, gm = gt.to(device), gm.to(device)
        pred, _ = nll_predict(model, inp, uncertainty)
        pm, pe = inp[:, 1:2], inp[:, :1]
        for m in per_sample_metrics(pred, gt, gm, pm, pe, stats):
            src = Path(ds.files[idx]).relative_to(data_root)
            rows.append({'sample_id': str(src),
                         'env': src.parts[0], 'trajectory': src.parts[1], **m})
            idx += 1
    return rows


@torch.no_grad()
def eval_test_full(model, ds, cfg, stats, device, uncertainty,
                   data_root, pred_dir, rng):
    """
    Full test-environment evaluation:
      per-sample metrics for β-NLL prediction and TTA-mean prediction,
      per-sample calibration (Pearson, coverage@1σ, AUSE) for both σ methods,
      fold-level streaming calibration, mean sparsification curves,
      a calibration pixel subsample, and per-sample prediction NPZs.
    """
    loader = eval_loader(ds, cfg, batch_size=max(cfg['batch_size'] // 2, 8))
    std_m = stats['std']

    # Crop offsets: predictions are saved aligned to the original (unpadded)
    # source arrays so they overlay directly on the dataset NPZs.
    src0 = np.load(ds.files[0])['gt_mask']
    H0, W0 = src0.shape
    ph, pw = (cfg['pad_to'] - H0) // 2, (cfg['pad_to'] - W0) // 2

    corr = {'nll': StreamingCorr(), 'tta': StreamingCorr()}
    curves = {k: [] for k in ('nll', 'tta', 'oracle_nll', 'oracle_tta')}
    pix = {k: [] for k in ('err_nll', 'sig_nll', 'err_tta', 'sig_tta')}
    rows, idx = [], 0

    for inp, gt, gm in loader:
        inp = inp.to(device, non_blocking=True)
        gt, gm = gt.to(device), gm.to(device)
        pm, pe = inp[:, 1:2], inp[:, :1]

        pred_nll, sigma_nll = nll_predict(model, inp, uncertainty)
        pred_tta, sigma_tta = tta_predict(model, inp)

        m_nll = per_sample_metrics(pred_nll, gt, gm, pm, pe, stats)
        m_tta = per_sample_metrics(pred_tta, gt, gm, pm, pe, stats)

        for i in range(inp.shape[0]):
            src = Path(ds.files[idx]).relative_to(data_root)
            traj = src.parts[1]

            gm_np = gm[i, 0].cpu().numpy().astype(bool)
            pm_np = (pm[i, 0].cpu().numpy() > 0.5)
            hole = gm_np & ~pm_np
            gt_np = gt[i, 0].cpu().numpy()

            pr_nll = pred_nll[i, 0].cpu().numpy()
            pr_tta = pred_tta[i, 0].cpu().numpy()
            sg_tta = sigma_tta[i, 0].cpu().numpy()
            sg_nll = sigma_nll[i, 0].cpu().numpy() if uncertainty else None

            err_nll = np.abs(pr_nll - gt_np)[hole] * std_m
            err_tta = np.abs(pr_tta - gt_np)[hole] * std_m
            s_tta = sg_tta[hole] * std_m
            s_nll = sg_nll[hole] * std_m if uncertainty else None

            cal_tta = calib_sample(err_tta, s_tta)
            cal_nll = (calib_sample(err_nll, s_nll) if uncertainty
                       else {'mean_sigma': np.nan, 'pearson': np.nan,
                             'cov68': np.nan, 'ause': np.nan,
                             'curve_unc': None, 'curve_oracle': None})

            if cal_tta['curve_unc'] is not None:
                curves['tta'].append(cal_tta['curve_unc'])
                curves['oracle_tta'].append(cal_tta['curve_oracle'])
            if cal_nll['curve_unc'] is not None:
                curves['nll'].append(cal_nll['curve_unc'])
                curves['oracle_nll'].append(cal_nll['curve_oracle'])

            if hole.sum() >= MIN_HOLE_PX:
                corr['tta'].add(s_tta, err_tta)
                if uncertainty:
                    corr['nll'].add(s_nll, err_nll)
                take = rng.choice(hole.sum(),
                                  size=min(PIX_SUBSAMPLE, int(hole.sum())),
                                  replace=False)
                pix['err_tta'].append(err_tta[take].astype(np.float16))
                pix['sig_tta'].append(s_tta[take].astype(np.float16))
                if uncertainty:
                    pix['err_nll'].append(err_nll[take].astype(np.float16))
                    pix['sig_nll'].append(s_nll[take].astype(np.float16))

            # Save prediction (normalized float16, cropped to source geometry).
            out_file = pred_dir / traj / (src.stem + '.npz')
            out_file.parent.mkdir(parents=True, exist_ok=True)
            arrays = {
                'pred_nll': pr_nll[ph:ph + H0, pw:pw + W0].astype(np.float16),
                'pred_tta': pr_tta[ph:ph + H0, pw:pw + W0].astype(np.float16),
                'sigma_tta': sg_tta[ph:ph + H0, pw:pw + W0].astype(np.float16),
            }
            if uncertainty:
                arrays['sigma_nll'] = sg_nll[ph:ph + H0, pw:pw + W0].astype(np.float16)
            np.savez_compressed(out_file, **arrays)

            row = {'sample_id': str(src), 'env': src.parts[0], 'trajectory': traj,
                   'pred_file': str(out_file.relative_to(pred_dir.parent)),
                   'coverage': float(pm_np[gm_np].mean()) if gm_np.any() else np.nan}
            row.update({f'nll_{k}': v for k, v in m_nll[i].items()})
            row.update({f'tta_{k}': v for k, v in m_tta[i].items()})
            for meth, cal in (('nll', cal_nll), ('tta', cal_tta)):
                for k in ('mean_sigma', 'pearson', 'cov68', 'ause'):
                    row[f'unc_{meth}_{k}'] = cal[k]
            rows.append(row)
            idx += 1

        if idx % 1000 < inp.shape[0]:
            print(f'    test eval {idx}/{len(ds)}')

    calibration = {
        'region': 'hole pixels (gt_mask=1 & partial_mask=0), metres',
        'sparsification_fractions': SPARS_FRACS.tolist(),
        'nll': {'global': corr['nll'].result(),
                'mean_curve': (np.mean(curves['nll'], axis=0).tolist()
                               if curves['nll'] else None),
                'oracle_mean_curve': (np.mean(curves['oracle_nll'], axis=0).tolist()
                                      if curves['oracle_nll'] else None)},
        'tta': {'global': corr['tta'].result(),
                'mean_curve': (np.mean(curves['tta'], axis=0).tolist()
                               if curves['tta'] else None),
                'oracle_mean_curve': (np.mean(curves['oracle_tta'], axis=0).tolist()
                                      if curves['oracle_tta'] else None)},
    }
    pix_arrays = {k: (np.concatenate(v) if v else np.array([], dtype=np.float16))
                  for k, v in pix.items()}

    pred_meta = {
        'units': 'normalized — denormalize with x * std + mean from stats.json',
        'stats': stats,
        'dtype': 'float16',
        'shape': [int(H0), int(W0)],
        'pad_to': cfg['pad_to'],
        'crop_offset': [int(ph), int(pw)],
        'contents': {
            'pred_nll': 'single-pass prediction (channel 0)',
            'sigma_nll': 'aleatoric σ = exp(0.5·log_var) from the β-NLL head',
            'pred_tta': 'mean prediction over the 8 D4 TTA transforms',
            'sigma_tta': 'std over the 8 D4 TTA transforms',
        },
        'aligned_to': 'original dataset NPZ arrays (same H×W, same orientation)',
    }
    return rows, calibration, pix_arrays, pred_meta


def eval_fold(cfg: dict, cv: dict, fold: dict, fold_dir: Path, device):
    fold_seed = cv['seed'] + 1000 * fold['fold']
    data_root = Path(cv['data_root'])
    uncertainty = cfg.get('uncertainty', False)

    ckpt_path = fold_dir / 'best.pth'
    ckpt = torch.load(ckpt_path, map_location=device)
    stats = ckpt['stats']
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'  loaded {ckpt_path.name} (epoch {ckpt["epoch"] + 1}, '
          f'best val RMSE {ckpt.get("best_rmse", float("nan")):.4f} m)')

    train_ds, val_ds, test_ds, _ = make_datasets(cfg, fold, data_root, stats)
    # Evaluation must be augmentation-free everywhere, including train.
    train_ds.augment = False
    train_ds.aug_ray_p = 0.0

    eval_dir = fold_dir / 'eval'
    eval_dir.mkdir(exist_ok=True)
    metrics_all = {}

    for name, ds in (('train', train_ds), ('val', val_ds)):
        t0 = time.time()
        rows = eval_split_simple(model, ds, cfg, stats, device,
                                 uncertainty, data_root)
        write_csv(eval_dir / f'per_sample_{name}.csv', rows)
        agg = aggregate_rows([{k: v for k, v in r.items()
                               if isinstance(v, float)} for r in rows])
        metrics_all[name] = agg
        with open(eval_dir / f'metrics_{name}.json', 'w') as f:
            json.dump(agg, f, indent=2)
        print(f'  {name}: {len(rows)} samples in {time.time() - t0:.0f}s — '
              f'rmse {agg["rmse"]["mean"]:.3f}±{agg["rmse"]["std"]:.3f} m')

    # Test — full protocol with both uncertainty methods + saved predictions.
    t0 = time.time()
    pred_dir = fold_dir / 'predictions' / 'test'
    pred_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(fold_seed)
    rows, calibration, pix, pred_meta = eval_test_full(
        model, test_ds, cfg, stats, device, uncertainty, data_root,
        pred_dir, rng)

    write_csv(eval_dir / 'per_sample_test.csv', rows)
    with open(fold_dir / 'predictions' / 'meta.json', 'w') as f:
        json.dump(pred_meta, f, indent=2)

    test_agg = {}
    for meth in ('nll', 'tta'):
        sub = [{k[len(meth) + 1:]: v for k, v in r.items()
                if k.startswith(meth + '_') and isinstance(v, float)}
               for r in rows]
        test_agg[meth] = aggregate_rows(sub)
    unc_rows = [{k: v for k, v in r.items()
                 if k.startswith('unc_') and isinstance(v, float)} for r in rows]
    test_agg['uncertainty_per_sample'] = aggregate_rows(unc_rows)
    metrics_all['test'] = test_agg

    with open(eval_dir / 'metrics_test.json', 'w') as f:
        json.dump(test_agg, f, indent=2)
    with open(eval_dir / 'calibration_test.json', 'w') as f:
        json.dump(calibration, f, indent=2)
    np.savez_compressed(eval_dir / 'calibration_pixels_test.npz', **pix)

    print(f'  test: {len(rows)} samples in {time.time() - t0:.0f}s — '
          f'NLL rmse {test_agg["nll"]["rmse"]["mean"]:.3f} m | '
          f'TTA rmse {test_agg["tta"]["rmse"]["mean"]:.3f} m')

    # Figures — informative, not paper-final (paper figures rebuild from NPZs).
    fig_learning_curves(fold_dir, eval_dir / 'learning_curves.png')
    fig_calibration(calibration, pix, uncertainty, eval_dir / 'calibration_test.png')
    vis_rng = np.random.default_rng(42)
    vis_idx = vis_rng.choice(len(test_ds),
                             size=min(cfg.get('n_vis', 8), len(test_ds)),
                             replace=False).tolist()
    visualize_uncertainty(model, test_ds, stats, vis_idx, device,
                          uncertainty, eval_dir / 'test_samples.png')

    with open(fold_dir / 'eval_done.json', 'w') as f:
        json.dump({'finished_at': datetime.now().isoformat(timespec='seconds'),
                   'checkpoint_epoch': ckpt['epoch'],
                   'n_test_predictions': len(rows)}, f, indent=2)
    return metrics_all


# ─────────────────────────────────────────────────────────────────────────────
# Cross-fold aggregation (summary tables + figures)
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(m, digits=3):
    return 'n/a' if m is None or not np.isfinite(m) else f'{m:.{digits}f}'


def write_tables(summary: dict, out_dir: Path):
    folds = summary['folds']

    # table_main.tex — test metrics of the deployed (β-NLL single-pass) prediction
    lines = [
        '% Auto-generated by scripts/train.py — test metrics per held-out environment',
        '% Prediction: single forward pass (channel 0). Mean over test samples.',
        r'\begin{tabular}{l' + 'c' * len(MAIN_METRICS) + '}',
        r'\toprule',
        'Test environment & ' + ' & '.join(
            m.replace('_', r'\_').upper() for m in MAIN_METRICS) + r' \\',
        r'\midrule',
    ]
    cols = {m: [] for m in MAIN_METRICS}
    for k, fd in folds.items():
        vals = []
        for m in MAIN_METRICS:
            mu = fd['test']['nll'].get(m, {}).get('mean', float('nan'))
            cols[m].append(mu)
            vals.append(_fmt(mu))
        lines.append(f'{fd["test_env"]} & ' + ' & '.join(vals) + r' \\')
    lines += [r'\midrule',
              'Mean $\\pm$ std & ' + ' & '.join(
                  f'{_fmt(nanmean(cols[m]))} $\\pm$ {_fmt(nanstd(cols[m]))}'
                  for m in MAIN_METRICS) + r' \\',
              r'\bottomrule', r'\end{tabular}']
    (out_dir / 'table_main.tex').write_text('\n'.join(lines) + '\n')

    # table_uncertainty.tex — calibration of both σ methods (hole pixels)
    lines = [
        '% Auto-generated — uncertainty calibration on hole pixels of the test environment',
        '% Global = pooled over all hole pixels of the fold; AUSE = mean per-sample.',
        r'\begin{tabular}{lcccc|cccc}',
        r'\toprule',
        r' & \multicolumn{4}{c|}{$\beta$-NLL head} & \multicolumn{4}{c}{TTA ensemble} \\',
        r'Test environment & $r(\sigma,|e|)$ & Cov@1$\sigma$ & $\bar\sigma$ [m] & AUSE'
        r' & $r(\sigma,|e|)$ & Cov@1$\sigma$ & $\bar\sigma$ [m] & AUSE \\',
        r'\midrule',
    ]
    acc = {meth: {m: [] for m in UNC_METRICS + ['ause']} for meth in ('nll', 'tta')}
    for k, fd in folds.items():
        vals = []
        for meth in ('nll', 'tta'):
            g = fd['calibration'][meth]['global'] or {}
            ause = fd['test']['uncertainty_per_sample'].get(
                f'unc_{meth}_ause', {}).get('mean', float('nan'))
            for m in UNC_METRICS:
                acc[meth][m].append(g.get(m) if g.get(m) is not None else float('nan'))
            acc[meth]['ause'].append(ause)
            vals += [_fmt(g.get('pearson')), _fmt(g.get('cov68'), 2),
                     _fmt(g.get('mean_sigma')), _fmt(ause)]
        lines.append(f'{fd["test_env"]} & ' + ' & '.join(vals) + r' \\')
    mean_vals = []
    for meth in ('nll', 'tta'):
        for m in ['pearson', 'cov68', 'mean_sigma', 'ause']:
            v = acc[meth][m]
            mean_vals.append(f'{_fmt(nanmean(v))} $\\pm$ {_fmt(nanstd(v))}')
    lines += [r'\midrule',
              'Mean $\\pm$ std & ' + ' & '.join(mean_vals) + r' \\',
              r'\bottomrule', r'\end{tabular}']
    (out_dir / 'table_uncertainty.tex').write_text('\n'.join(lines) + '\n')

    # table_domain_gap.tex — in-domain (val: unseen trajectories of TRAINING
    # environments) vs cross-domain (test: fully unseen environment).
    lines = [
        '% Auto-generated — domain gap: val = unseen trajectories of seen',
        '% environments (in-domain), test = fully unseen environment (cross-domain).',
        r'\begin{tabular}{l' + 'ccc' * len(GAP_METRICS[:2]) + '}',
        r'\toprule',
        ' & ' + ' & '.join(
            r'\multicolumn{3}{c}{' + m.replace("_", r"\_").upper() + ' [m]}'
            for m in GAP_METRICS[:2]) + r' \\',
        'Test environment & ' + ' & '.join(
            ['val & test & $\\Delta$'] * len(GAP_METRICS[:2])) + r' \\',
        r'\midrule',
    ]
    gap_acc = {m: {'val': [], 'test': [], 'delta': []} for m in GAP_METRICS[:2]}
    for k, fd in folds.items():
        vals = []
        for m in GAP_METRICS[:2]:
            v = fd['val'].get(m, {}).get('mean', float('nan'))
            t = fd['test']['nll'].get(m, {}).get('mean', float('nan'))
            d = t - v
            gap_acc[m]['val'].append(v)
            gap_acc[m]['test'].append(t)
            gap_acc[m]['delta'].append(d)
            vals += [_fmt(v), _fmt(t), f'{d:+.3f}']
        lines.append(f'{fd["test_env"]} & ' + ' & '.join(vals) + r' \\')
    mean_vals = []
    for m in GAP_METRICS[:2]:
        mean_vals += [_fmt(nanmean(gap_acc[m]['val'])),
                      _fmt(nanmean(gap_acc[m]['test'])),
                      f'{nanmean(gap_acc[m]["delta"]):+.3f}']
    lines += [r'\midrule',
              'Mean & ' + ' & '.join(mean_vals) + r' \\',
              r'\bottomrule', r'\end{tabular}']
    (out_dir / 'table_domain_gap.tex').write_text('\n'.join(lines) + '\n')


def aggregate(exp_dir: Path):
    exp = json.load(open(exp_dir / 'experiment.json'))
    folds_def = json.load(open(exp_dir / 'folds.json'))
    out_dir = exp_dir / 'summary'
    out_dir.mkdir(exist_ok=True)

    folds, missing = {}, []
    for fd in folds_def:
        fdir = exp_dir / f'fold_{fd["fold"]}_{fd["test_env"]}'
        if not (fdir / 'eval_done.json').exists():
            missing.append(fdir.name)
            continue
        ev = fdir / 'eval'
        entry = {
            'test_env': fd['test_env'],
            'train_done': json.load(open(fdir / 'train_done.json')),
            'calibration': json.load(open(ev / 'calibration_test.json')),
            'test': json.load(open(ev / 'metrics_test.json')),
        }
        for s in ('train', 'val'):
            entry[s] = json.load(open(ev / f'metrics_{s}.json'))
        folds[str(fd['fold'])] = entry
    if missing:
        print(f'  WARNING: summary is PARTIAL — missing evaluated folds: {missing}')
    if not folds:
        print('  nothing to aggregate yet.')
        return

    # Cross-fold mean ± std (over folds) of the per-fold sample means.
    cross = {}
    for split, get in (
            ('train', lambda f: f['train']),
            ('val', lambda f: f['val']),
            ('test_nll', lambda f: f['test']['nll']),
            ('test_tta', lambda f: f['test']['tta'])):
        keys = sorted({k for f in folds.values() for k in get(f)})
        cross[split] = {}
        for k in keys:
            v = np.array([get(f).get(k, {}).get('mean', np.nan)
                          for f in folds.values()])
            cross[split][k] = {'mean_over_folds': nanmean(v),
                               'std_over_folds': nanstd(v),
                               'per_fold': {fk: (None if not np.isfinite(x) else float(x))
                                            for fk, x in zip(folds, v)}}

    # Domain gap: in-domain (val) vs cross-domain (test) for the key metrics.
    domain_gap = {}
    for m in GAP_METRICS:
        per_fold = {}
        for fk, fd in folds.items():
            v = fd['val'].get(m, {}).get('mean', float('nan'))
            t = fd['test']['nll'].get(m, {}).get('mean', float('nan'))
            per_fold[fk] = {'val': v, 'test': t, 'delta': t - v,
                            'rel': (t - v) / v if v and np.isfinite(v) else None}
        domain_gap[m] = {
            'per_fold': per_fold,
            'mean_val': nanmean([p['val'] for p in per_fold.values()]),
            'mean_test': nanmean([p['test'] for p in per_fold.values()]),
            'mean_delta': nanmean([p['delta'] for p in per_fold.values()]),
        }

    summary = {
        'experiment': exp['name'],
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'n_folds_included': len(folds),
        'missing_folds': missing,
        'folds': folds,
        'cross_fold': cross,
        'domain_gap': domain_gap,
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Flat CSV: one row per fold × split.
    rows = []
    for fk, fd in folds.items():
        for split, data in (('train', fd['train']), ('val', fd['val']),
                            ('test_nll', fd['test']['nll']),
                            ('test_tta', fd['test']['tta'])):
            row = {'fold': fk, 'test_env': fd['test_env'], 'split': split}
            for m, st in data.items():
                row[m] = st['mean']
                row[f'{m}_std'] = st['std']
            rows.append(row)
    write_csv(out_dir / 'summary_folds.csv', rows)

    per_env = {fd['test_env']: fd['test'] for fd in folds.values()}
    fig_test_by_env(per_env, out_dir / 'test_rmse_by_env.png')
    write_tables(summary, out_dir)

    print(f'  summary written to {out_dir}')
    print(f'  test RMSE (β-NLL pred): '
          f'{cross["test_nll"]["rmse"]["mean_over_folds"]:.3f} ± '
          f'{cross["test_nll"]["rmse"]["std_over_folds"]:.3f} m over '
          f'{len(folds)} folds')
    gap = domain_gap['rmse']
    print(f'  domain gap (RMSE): val {gap["mean_val"]:.3f} m → '
          f'test {gap["mean_test"]:.3f} m  (Δ {gap["mean_delta"]:+.3f} m)')


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(exp_dir: Path, folds_sel: list = None, stage: str = 'all',
                   num_workers: int = None):
    """Train/evaluate the selected folds of an experiment, then aggregate."""
    exp = json.load(open(exp_dir / 'experiment.json'))
    cfg, cv = exp['config'], exp['cv']
    if num_workers is not None:
        cfg['num_workers'] = num_workers

    folds = json.load(open(exp_dir / 'folds.json'))
    sel = folds_sel if folds_sel is not None else [f['fold'] for f in folds]
    bad = [k for k in sel if k not in {f['fold'] for f in folds}]
    if bad:
        raise ValueError(f'Unknown folds {bad}; available: {[f["fold"] for f in folds]}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if stage in ('all', 'train', 'eval'):
        for fd in folds:
            if fd['fold'] not in sel:
                continue
            fold_dir = exp_dir / f'fold_{fd["fold"]}_{fd["test_env"]}'
            print(f'\n{"═" * 70}\nFOLD {fd["fold"]} — test environment: '
                  f'{fd["test_env"]}\n{"═" * 70}')

            if stage in ('all', 'train'):
                if (fold_dir / 'train_done.json').exists():
                    print('  training already complete — skipping')
                else:
                    train_fold(cfg, cv, fd, fold_dir, device)

            if stage in ('all', 'eval'):
                if not (fold_dir / 'best.pth').exists():
                    print('  no best.pth yet — skipping evaluation')
                    continue
                if stage == 'all' and (fold_dir / 'eval_done.json').exists():
                    print('  evaluation already complete — skipping')
                else:
                    eval_fold(cfg, cv, fd, fold_dir, device)

    if stage in ('all', 'aggregate'):
        print(f'\n{"═" * 70}\nCROSS-FOLD SUMMARY\n{"═" * 70}')
        aggregate(exp_dir)
