#!/usr/bin/env python3
"""
Figures for the Materials and Methods section of the paper.

Generates into MDPI___Raimoc/figures/:
    fig_pipeline.png         methodology overview schema — one real sample
                             flowing sensing → rasterization → network →
                             composite output, with a training-only strip
    fig_data_generation.png  raw multi-camera depth → merged point cloud →
                             partial DEM + observability mask + ground truth
                             (real frame; DEMs read from the actual dataset NPZ)
    fig_composite.png        inference: partial input → prediction + σ →
                             composite output (Equation "composite" in the paper)
    fig_environments.png     one sample per environment (partial / ground truth)
    fig_ray_augmentation.png copied from reports/ (tools/preview_ray_augmentation.py)

The point-cloud reconstruction mirrors Create_dataset.ipynb exactly
(back-projection, pose transforms, 'yxz_neg' axis mapping), so the
intermediate panels are faithful to the dataset-generation pipeline.

Usage:
    python tools/make_method_figures.py
    python tools/make_method_figures.py --frame 400 --checkpoint runs/<run>/best.pth
"""
import argparse
import json
import shutil
import sys
from pathlib import Path


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from elevcomp.paths import project_root
from elevcomp.dataset import ElevationDataset
from elevcomp.inference import nll_predict, tta_predict
from elevcomp.model import build_model

ROOT = project_root()
RAW_ROOT = ROOT.parent / 'datasets' / 'tartanair'      # raw TartanGround
DS_ROOT = ROOT.parent / 'datasets' / 'elevation_dataset'
FIG_DIR = ROOT / 'results' / 'figures'

CAMS = ('front', 'left', 'right', 'back')
BOUNDS = np.array([-25, 25, -25, 25, -25, 25], dtype=np.float32)
RESOLUTION = 0.2

CMAP_ELEV, CMAP_SIGMA, CMAP_DEPTH = 'terrain', 'plasma', 'viridis'


# ── Exact re-implementation of the dataset-generation transforms ──────────────

def load_depth(path):
    import cv2
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return d.view('<f4').squeeze().astype(np.float32)


def depth_to_points(depth, f=320.0, c=320.0, dmin=0.1, dmax=100.0):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32) + 0.5,
                       np.arange(h, dtype=np.float32) + 0.5)
    valid = (depth > dmin) & (depth < dmax)
    d = depth[valid]
    x = (u[valid] - c) * d / f
    y = (v[valid] - c) * d / f
    return np.stack([d, x, y], axis=1)          # camera frame: [depth, x, y]


def pose_to_T(pose):
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rotation.from_quat(pose[3:7] / np.linalg.norm(pose[3:7])
                                   ).as_matrix().astype(np.float32)
    return T


def merged_cloud(traj_path: Path, frame: int):
    """All four cameras of one frame in the front-camera reference frame,
    cropped to the grid bounds and remapped to x-right/y-forward/z-up."""
    poses = {c: np.loadtxt(traj_path / f'pose_lcam_{c}.txt', dtype=np.float32)
             for c in CAMS}
    T_ref_inv = np.linalg.inv(pose_to_T(poses['front'][frame]))

    pts_all, per_cam = [], {}
    for cam in CAMS:
        files = sorted((traj_path / f'depth_lcam_{cam}').glob('*.png'))
        depth = load_depth(files[frame])
        pts = depth_to_points(depth)
        T = T_ref_inv @ pose_to_T(poses[cam][frame])
        pts = pts @ T[:3, :3].T + T[:3, 3]
        m = np.all((pts >= BOUNDS[::2]) & (pts <= BOUNDS[1::2]), axis=1)
        pts_all.append(pts[m])
        per_cam[cam] = depth
    pts = np.concatenate(pts_all)
    pts = np.stack([pts[:, 1], pts[:, 0], -pts[:, 2]], axis=1)   # 'yxz_neg'
    return per_cam, pts


# ── Small plotting helpers ─────────────────────────────────────────────────────

def _elev_panel(ax, data, vmin, vmax, title, cmap=CMAP_ELEV, fs=9):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title(title, fontsize=fs)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def _cbar(fig, im, ax, label, fs=7):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=fs)
    cb.ax.tick_params(labelsize=fs)


# ── Figure 1: data generation ─────────────────────────────────────────────────

def fig_data_generation(env: str, traj: str, frame: int, out: Path):
    per_cam, pts = merged_cloud(RAW_ROOT / env / 'Data_anymal' / traj, frame)
    npz = np.load(DS_ROOT / env / traj / f'{traj}_sample_{frame:06d}.npz')
    pe, pm = npz['partial_elevation'], npz['partial_mask'].astype(bool)
    gt, gm = npz['gt_elevation'], npz['gt_mask'].astype(bool)

    fig = plt.figure(figsize=(13.2, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)

    dmax = max(float(np.percentile(d[np.isfinite(d) & (d < 100)], 98))
               for d in per_cam.values())
    for i, cam in enumerate(CAMS):
        ax = fig.add_subplot(gs[0, i])
        d = per_cam[cam].copy()
        d[d >= 100] = np.nan
        im = ax.imshow(d, cmap=CMAP_DEPTH, vmin=0, vmax=dmax)
        ax.set_title(f'depth, {cam}', fontsize=16)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 3:
            _cbar(fig, im, ax, 'depth [m]', fs=14)

    vals = gt[gm]
    vmin, vmax = np.percentile(vals, 1), np.percentile(vals, 99)

    ax = fig.add_subplot(gs[1, 0])
    sub = pts[np.random.default_rng(0).choice(len(pts),
                                              min(200_000, len(pts)),
                                              replace=False)]
    ax.scatter(sub[:, 1], sub[:, 0], c=sub[:, 2], s=0.15,
               cmap=CMAP_ELEV, vmin=vmin, vmax=vmax, rasterized=True)
    ax.plot(0, 0, 'k^', ms=7)
    ax.set(xlim=(-25, 25), ylim=(-25, 25), aspect='equal')
    ax.set_title('merged point cloud', fontsize=16)
    ax.tick_params(labelsize=13)
    ax.set_xlabel('[m]', fontsize=13)

    im = _elev_panel(fig.add_subplot(gs[1, 1]),
                     np.where(pm, pe, np.nan), vmin, vmax,
                     'partial map $\\mathbf{E} \\odot \\mathbf{M}$', fs=16)
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(pm, cmap='gray', origin='lower')
    cov = pm[gm].mean()
    ax.set_title('observability mask $\\mathbf{M}$', fontsize=16)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 3])
    im = _elev_panel(ax, np.where(gm, gt, np.nan), vmin, vmax,
                     'ground truth $\\mathbf{E}^{\\mathrm{gt}}$', fs=16)
    _cbar(fig, im, ax, 'elevation [m]', fs=14)

    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}  ({env}/{traj} frame {frame})')


# ── Figure 2: inference + composite output ────────────────────────────────────

def fig_composite(checkpoint: Path, sample_idx: int, out: Path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(checkpoint, map_location=device)
    cfg, stats = ck['config'], ck['stats']
    model = build_model(cfg).to(device)
    model.load_state_dict(ck['model'])
    model.eval()

    if 'data_dir' in cfg:
        # single-run checkpoint (train.py): fractional test split of one dir
        ds = ElevationDataset(str(ROOT / cfg['data_dir']), split='test',
                              augment=False, stats=stats, pad_to=cfg['pad_to'])
    else:
        # CV checkpoint (run_cv_experiment.py): use this fold's held-out test set
        exp = json.load(open(checkpoint.parent.parent / 'experiment.json'))
        data_root = Path(exp['cv']['data_root'])
        split = json.load(open(checkpoint.parent / 'split_files.json'))
        files = [data_root / f for f in split['test_files']]
        ds = ElevationDataset(split='test', augment=False, stats=stats,
                              pad_to=cfg['pad_to'], files=files)
    inp, gt, gm = ds[sample_idx]
    pred, sigma = nll_predict(model, inp[None].to(device),
                              cfg.get('uncertainty', False))

    m, s = stats['mean'], stats['std']
    c0 = (cfg['pad_to'] - 251) // 2
    crop = np.s_[c0:c0 + 251, c0:c0 + 251]

    pm = inp[1].numpy()[crop] > 0.5
    pe = (inp[0].numpy() * s + m)[crop]
    pr = (pred[0, 0].cpu().numpy() * s + m)[crop]
    sg = (sigma[0, 0].cpu().numpy() * s)[crop]
    gte = (gt[0].numpy() * s + m)[crop]
    gmc = gm[0].numpy()[crop].astype(bool)

    comp = np.where(pm, pe, pr)
    vals = gte[gmc]
    vmin, vmax = np.percentile(vals, 1), np.percentile(vals, 99)
    smax = np.percentile(sg[gmc & ~pm], 95)

    fig, axes = plt.subplots(1, 5, figsize=(15.2, 3.4), constrained_layout=True)
    im0 = _elev_panel(axes[0], np.where(pm, pe, np.nan), vmin, vmax,
                      'input $\\mathbf{E} \\odot \\mathbf{M}$', fs=22)
    _elev_panel(axes[1], pr, vmin, vmax, 'prediction $\\hat{\\mathbf{E}}$', fs=22)
    im2 = axes[2].imshow(np.where(gmc, sg, np.nan), cmap=CMAP_SIGMA,
                         vmin=0, vmax=smax, origin='lower')
    axes[2].set_title('uncertainty $\\hat{\\sigma}$', fontsize=22)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    _elev_panel(axes[3], comp, vmin, vmax, 'composite $\\hat{\\mathbf{E}}^{\\mathrm{comp}}$', fs=22)
    im4 = _elev_panel(axes[4], np.where(gmc, gte, np.nan), vmin, vmax,
                      'ground truth $\\mathbf{E}^{\\mathrm{gt}}$', fs=22)
    _cbar(fig, im2, axes[2], '$\\sigma$ [m]', fs=16)
    _cbar(fig, im4, axes[4], 'elevation [m]', fs=16)

    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}  (checkpoint {checkpoint}, test sample {sample_idx})')


# ── Figure 3: one sample per environment ─────────────────────────────────────

def fig_environments(frame_frac: float, out: Path):
    short = {'ForestEnv': 'Forest', 'Gascola': 'Gascola',
             'ModularNeighborhood': 'Modular', 'OldTownSummer': 'Old Town',
             'SeasonalForestWinter': 'Winter'}
    envs = sorted(p.name for p in DS_ROOT.iterdir()
                  if p.is_dir() and any(q.is_dir() for q in p.iterdir()))
    fig, axes = plt.subplots(2, len(envs), figsize=(2.7 * len(envs), 5.6),
                             constrained_layout=True)
    for col, env in enumerate(envs):
        traj = sorted(p.name for p in (DS_ROOT / env).iterdir() if p.is_dir())[0]
        files = sorted((DS_ROOT / env / traj).glob('*.npz'))
        npz = np.load(files[int(frame_frac * len(files))])
        pe, pm = npz['partial_elevation'], npz['partial_mask'].astype(bool)
        gt, gm = npz['gt_elevation'], npz['gt_mask'].astype(bool)
        vals = gt[gm]
        vmin, vmax = np.percentile(vals, 1), np.percentile(vals, 99)

        _elev_panel(axes[0, col], np.where(pm, pe, np.nan), vmin, vmax,
                    f'{short.get(env, env)}\n({pm[gm].mean():.0%} obs.)', fs=17)
        _elev_panel(axes[1, col], np.where(gm, gt, np.nan), vmin, vmax, '')
    axes[0, 0].set_ylabel('sensor input', fontsize=18)
    axes[1, 0].set_ylabel('ground truth', fontsize=18)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')


# ── Figure 0: methodology overview schema ────────────────────────────────────
#
# One real sample flows through the whole diagram. Visual semantics:
#   solid arrows  = data flow at deployment
#   dashed arrows = training-time only (augmentation, loss, fold protocol)
#   colormaps     = same as all other paper figures (viridis depth,
#                   terrain elevation, plasma σ)
#   blue / orange = β-NLL head vs TTA ensemble (matches calibration figures)

INK = '#333333'          # flow arrows, borders, text
TRAIN = '#8A8A8A'        # training-only (dashed)
RULE = '#BBBBBB'         # stage-group rules
C_NLL, C_TTA = '#0072B2', '#E69F00'

W, H = 14.6, 8.45        # canvas [inch]


def _panel(fig, x, y, w, h, arr=None, cmap=None, vmin=None, vmax=None,
           border=INK, lw=1.0, zorder=3):
    ax = fig.add_axes([x / W, y / H, w / W, h / H], zorder=zorder)
    if arr is not None:
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(border); s.set_linewidth(lw)
    return ax


def _arrow(bg, x0, y0, x1, y1, style='-', color=INK, lw=2.2, rad=0.0):
    bg.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                linestyle=style, shrinkA=2, shrinkB=2,
                                mutation_scale=20,
                                connectionstyle=f'arc3,rad={rad}'))


def _caption(bg, x, y, text, size=14, color=INK, weight='normal', ha='center'):
    bg.text(x, y, text, fontsize=size, color=color, ha=ha, va='top',
            weight=weight)


def fig_pipeline(env, traj, frame, checkpoint: Path, out: Path):
    # ── data for the panels: one sample end to end ────────────────────────────
    per_cam, pts = merged_cloud(RAW_ROOT / env / 'Data_anymal' / traj, frame)
    npz = np.load(DS_ROOT / env / traj / f'{traj}_sample_{frame:06d}.npz')
    pe = npz['partial_elevation']
    pm = npz['partial_mask'].astype(bool)
    gt, gm = npz['gt_elevation'], npz['gt_mask'].astype(bool)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(checkpoint, map_location=device)
    cfg, stats = ck['config'], ck['stats']
    model = build_model(cfg).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    m, s = stats['mean'], stats['std']

    pad, c0 = cfg['pad_to'], (cfg['pad_to'] - pe.shape[0]) // 2
    pe_n = np.zeros((pad, pad), np.float32)
    pm_p = np.zeros((pad, pad), np.float32)
    pe_n[c0:c0 + pe.shape[0], c0:c0 + pe.shape[1]] = \
        np.where(np.isfinite(pe) & pm, (pe - m) / s, 0.0)
    pm_p[c0:c0 + pe.shape[0], c0:c0 + pe.shape[1]] = pm
    inp = torch.from_numpy(np.stack([pe_n, pm_p]))[None].to(device)
    pred_t, sigma_t = nll_predict(model, inp, cfg.get('uncertainty', False))
    crop = np.s_[c0:c0 + pe.shape[0], c0:c0 + pe.shape[1]]
    pred = (pred_t[0, 0].cpu().numpy() * s + m)[crop]
    sig = (sigma_t[0, 0].cpu().numpy() * s)[crop]
    comp = np.where(pm, pe, pred)

    vals = gt[gm]
    vmin, vmax = np.percentile(vals, 1), np.percentile(vals, 99)
    smax = np.percentile(sig[gm & ~pm], 95)

    # ray-cone mini-panel: the actual training transform on this sample
    from elevcomp.dataset import _apply_ray_augmentation
    import random as _rnd
    _rnd.seed(5)
    pe_ray, pm_ray = _apply_ray_augmentation(
        np.where(pm, pe, np.nan), pm.astype(np.float32), 2, (25, 40))

    # thumbnails for the leave-one-environment-out glyph
    env_thumbs = []
    for e in sorted(p.name for p in DS_ROOT.iterdir()
                    if p.is_dir() and any(q.is_dir() for q in p.iterdir())):
        t0 = sorted(p.name for p in (DS_ROOT / e).iterdir() if p.is_dir())[0]
        f0 = sorted((DS_ROOT / e / t0).glob('*.npz'))
        d = np.load(f0[len(f0) // 2])
        g, gmk = d['gt_elevation'], d['gt_mask'].astype(bool)
        env_thumbs.append((e, np.where(gmk, g, np.nan)))

    # ── canvas ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(W, H))
    bg = fig.add_axes([0, 0, 1, 1], zorder=1)
    bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis('off')

    py, ps = 5.85, 1.85            # main-row panel bottom y and size
    cy = py + ps / 2               # arrow y

    # stage-group headers
    for x0, x1, label in ((0.50, 6.80, 'data generation'),
                          (7.15, 8.85, 'completion network'),
                          (9.20, 14.15, 'output with uncertainty')):
        bg.plot([x0, x1], [py + ps + 0.22] * 2, color=RULE, lw=1.2)
        bg.text((x0 + x1) / 2, py + ps + 0.30, label, fontsize=18,
                color=INK, ha='center', va='bottom', weight='bold')

    # ① four depth cameras (2×2 grid)
    gx = 0.50
    dmax = max(float(np.percentile(d[d < 100], 98)) for d in per_cam.values())
    for i, cam in enumerate(CAMS):
        d = per_cam[cam][::4, ::4].copy(); d[d >= 100] = np.nan
        _panel(fig, gx + (i % 2) * 0.95, py + 0.95 - (i // 2) * 0.95,
               0.90, 0.90, d[::-1], CMAP_DEPTH, 0, dmax, lw=0.8)
    _caption(bg, gx + 0.93, py - 0.13,
             'four depth cameras\n(360°, rendered)')

    # ② merged point cloud (top view)
    ax = _panel(fig, 2.70, py, ps, ps, lw=0.8)
    sub = pts[np.random.default_rng(0).choice(len(pts), 120_000, replace=False)]
    ax.scatter(sub[:, 1], sub[:, 0], c=sub[:, 2], s=0.08, cmap=CMAP_ELEV,
               vmin=vmin, vmax=vmax, rasterized=True)
    ax.plot(0, 0, 'k^', ms=5)
    ax.set(xlim=(-25, 25), ylim=(-25, 25), aspect='equal')
    _caption(bg, 2.70 + ps / 2, py - 0.13,
             'robot-centric\npoint cloud')

    # ③ partial DEM + mask inset
    x3 = 4.78
    _panel(fig, x3, py, ps, ps, np.where(pm, pe, np.nan),
           CMAP_ELEV, vmin, vmax)
    _panel(fig, x3 + ps - 0.60, py + 0.05, 0.55, 0.55, pm,
           'gray', 0, 1, lw=0.7, zorder=4)
    _caption(bg, x3 + ps / 2, py - 0.13,
             'partial map $\\mathbf{E}\\odot \\mathbf{M}$\n+ mask $\\mathbf{M}$ ($P_{20}$, 0.2 m)')

    # ④ U-Net glyph (encoder–decoder hourglass, skips arc above)
    ux, uw = 7.20, 1.60
    from matplotlib.patches import Polygon
    bg.add_patch(Polygon([(ux, py + ps), (ux + 0.66, py + ps - 0.58),
                          (ux + 0.66, py + 0.58), (ux, py)],
                         closed=True, facecolor='#DCE4EC', edgecolor=INK, lw=1.0))
    bg.add_patch(Polygon([(ux + uw, py + ps), (ux + uw - 0.66, py + ps - 0.58),
                          (ux + uw - 0.66, py + 0.58), (ux + uw, py)],
                         closed=True, facecolor='#DCE4EC', edgecolor=INK, lw=1.0))
    bg.add_patch(plt.Rectangle((ux + 0.66, py + 0.58), uw - 1.32, ps - 1.16,
                               facecolor='#C3D0DE', edgecolor=INK, lw=1.0))
    for dy in (0.30, 0.54):                          # skip connections
        bg.annotate('', xy=(ux + uw - 0.46, py + ps - dy),
                    xytext=(ux + 0.46, py + ps - dy),
                    arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.8,
                                    mutation_scale=7,
                                    connectionstyle='arc3,rad=-0.12'))
    _caption(bg, ux + uw / 2, py - 0.13,
             'U-Net encoder–decoder\n2 ch in → $\\hat{\\mathbf{E}}$, $\\log\\hat{\\sigma}^2$')

    # ⑤ two output channels (stacked squares)
    ox, oh = 9.20, 0.88
    _panel(fig, ox, py + ps - oh, oh, oh, pred, CMAP_ELEV, vmin, vmax, lw=0.9)
    _panel(fig, ox, py, oh, oh, sig, CMAP_SIGMA, 0, smax,
           border=C_NLL, lw=1.6)
    bg.text(ox + oh + 0.12, py + ps - oh / 2, 'prediction $\\hat{\\mathbf{E}}$',
            fontsize=14, color=INK, va='center')
    bg.text(ox + oh + 0.12, py + oh / 2, 'uncertainty $\\hat{\\sigma}$',
            fontsize=14, color=C_NLL, va='center')

    # ⑥ composite output (the deliverable — emphasized border)
    cx2 = 12.30
    _panel(fig, cx2, py, ps, ps, comp, CMAP_ELEV, vmin, vmax, lw=2.2)
    _caption(bg, cx2 + ps / 2 + 0.20, py - 0.13,
             'composite map\n$\\mathbf{M}\\odot\\mathbf{E} + (\\mathbf{1}-\\mathbf{M})\\odot\\hat{\\mathbf{E}}$')

    # main-flow arrows
    _arrow(bg, 2.40, cy, 2.66, cy)
    _arrow(bg, 4.59, cy, 4.74, cy)
    _arrow(bg, 6.69, cy, 7.16, cy)
    _arrow(bg, ux + uw + 0.05, cy, ox - 0.05, cy)
    _arrow(bg, 11.55, py + 1.05, cx2 - 0.05, py + 0.98, rad=-0.04)
    # measurements bypass the network (Equation "composite")
    _arrow(bg, 5.92, py - 0.62, cx2 + 0.16, py - 0.04, rad=0.15, lw=1.3)
    bg.text(9.75, 4.72, 'observed cells are copied unchanged',
            fontsize=14, color=INK, ha='center', style='italic')

    # ── training-only strip ───────────────────────────────────────────────────
    # Panels are the content — size them to fill the box, captions inside with
    # clear padding. The held-out fold pops out: larger, thick black border.
    bx0, bx1 = 2.85, 12.90          # box left/right
    by0, by1 = 0.60, 3.95           # box bottom/top
    ty = 1.42                        # ray-cone panel bottom
    th = 2.00                        # ray-cone panel size
    bg.add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False,
                               edgecolor=TRAIN, lw=1.2, linestyle=(0, (4, 3))))
    bg.text(3.85, by1, 'training only', fontsize=16, color=TRAIN,
            ha='center', va='center', style='italic',
            bbox=dict(facecolor='white', edgecolor='none', pad=3))

    _panel(fig, 3.25, ty, th, th, np.where(pm_ray > 0.5, pe_ray, np.nan),
           CMAP_ELEV, vmin, vmax, border=TRAIN, lw=1.2)
    _caption(bg, 3.25 + th / 2, ty - 0.16,
             'ray-cone + geometric\naugmentation', color=TRAIN, size=14)

    bg.text(6.52, ty + th / 2 + 0.10,
            'masked loss\n$w_v{=}0.1$ on $V$\n$w_h{=}1.0$ on $H$\n'
            '$\\beta$-NLL after L1 warmup',
            fontsize=14.5, color=INK, ha='center', va='center')

    tx0, ts, tg = 7.80, 0.92, 0.12   # leave-one-environment-out glyph
    trow = ty + (th - ts) / 2        # thumb row aligned with ray-panel centre
    for i, (e, thumb) in enumerate(env_thumbs):
        held = (i == 3)
        # the held-out environment pops out: larger panel, thick black border
        grow = 0.10 if held else 0.0
        _panel(fig, tx0 + i * (ts + tg) - grow, trow - grow,
               ts + 2 * grow, ts + 2 * grow,
               thumb[::2, ::2], CMAP_ELEV, None, None,
               border=INK if held else TRAIN, lw=3.0 if held else 0.8)
        if held:
            bg.text(tx0 + i * (ts + tg) + ts / 2, trow - grow - 0.12,
                    'held-out test', fontsize=14, ha='center', va='top',
                    color=INK, weight='bold')
    _caption(bg, tx0 + (5 * ts + 4 * tg) / 2, ty - 0.16,
             '5× leave-one-environment-out', color=TRAIN, size=14)

    _arrow(bg, 6.9, by1, ux + 0.45, py + 0.04, style='--',
           color=TRAIN, lw=1.4, rad=-0.06)

    # ── legend line ───────────────────────────────────────────────────────────
    ly = 0.26
    bg.plot([0.55, 1.10], [ly] * 2, color=INK, lw=2.4)
    bg.text(1.24, ly, 'inference data flow', fontsize=14.5, va='center')
    bg.plot([3.95, 4.50], [ly] * 2, color=TRAIN, lw=2.0, ls=(0, (4, 3)))
    bg.text(4.64, ly, 'training only', fontsize=14.5, va='center')
    bg.text(6.55, ly, '$\\hat{\\sigma}$:', fontsize=14.5, va='center')
    bg.text(6.95, ly, 'β-NLL head (1 pass)', fontsize=14.5, va='center',
            color=C_NLL)
    bg.text(9.35, ly, 'or TTA ensemble (8 passes)', fontsize=14.5,
            va='center', color=C_TTA)

    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)
    print(f'saved {out}  ({env}/{traj} frame {frame})')


# ── Figure: pipeline v2 (airier top row; training strip + legend unchanged) ──
#
# Same content as fig_pipeline. The output labels move below their panels
# (caption style, like every other stage), which frees ~1.4 in of width; that
# width is redistributed into uniform ~0.7 in gaps between the stages.

def fig_pipeline_v2(env, traj, frame, checkpoint: Path, out: Path):
    # ── data for the panels: one sample end to end (same as fig_pipeline) ─────
    per_cam, pts = merged_cloud(RAW_ROOT / env / 'Data_anymal' / traj, frame)
    npz = np.load(DS_ROOT / env / traj / f'{traj}_sample_{frame:06d}.npz')
    pe = npz['partial_elevation']
    pm = npz['partial_mask'].astype(bool)
    gt, gm = npz['gt_elevation'], npz['gt_mask'].astype(bool)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(checkpoint, map_location=device)
    cfg, stats = ck['config'], ck['stats']
    model = build_model(cfg).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    m, s = stats['mean'], stats['std']

    pad, c0 = cfg['pad_to'], (cfg['pad_to'] - pe.shape[0]) // 2
    pe_n = np.zeros((pad, pad), np.float32)
    pm_p = np.zeros((pad, pad), np.float32)
    pe_n[c0:c0 + pe.shape[0], c0:c0 + pe.shape[1]] = \
        np.where(np.isfinite(pe) & pm, (pe - m) / s, 0.0)
    pm_p[c0:c0 + pe.shape[0], c0:c0 + pe.shape[1]] = pm
    inp = torch.from_numpy(np.stack([pe_n, pm_p]))[None].to(device)
    pred_t, sigma_t = nll_predict(model, inp, cfg.get('uncertainty', False))
    crop = np.s_[c0:c0 + pe.shape[0], c0:c0 + pe.shape[1]]
    pred = (pred_t[0, 0].cpu().numpy() * s + m)[crop]
    sig = (sigma_t[0, 0].cpu().numpy() * s)[crop]
    comp = np.where(pm, pe, pred)

    vals = gt[gm]
    vmin, vmax = np.percentile(vals, 1), np.percentile(vals, 99)
    smax = np.percentile(sig[gm & ~pm], 95)

    # TTA σ (8-pass D4 ensemble) — shown next to the β-NLL σ so both
    # uncertainty options are visible in the schema itself.
    _, sig_tta_t = tta_predict(model, inp)
    sig_tta = (sig_tta_t[0, 0].cpu().numpy() * s)[crop]
    smax_tta = np.percentile(sig_tta[gm & ~pm], 95)

    from elevcomp.dataset import _apply_ray_augmentation
    import random as _rnd
    _rnd.seed(5)
    pe_ray, pm_ray = _apply_ray_augmentation(
        np.where(pm, pe, np.nan), pm.astype(np.float32), 2, (25, 40))

    # Three samples per environment: the fold glyph shows each environment as a
    # small stack of maps (a subset), not a single image.
    env_thumbs = []
    for e in sorted(p.name for p in DS_ROOT.iterdir()
                    if p.is_dir() and any(q.is_dir() for q in p.iterdir())):
        t0 = sorted(p.name for p in (DS_ROOT / e).iterdir() if p.is_dir())[0]
        f0 = sorted((DS_ROOT / e / t0).glob('*.npz'))
        stack = []
        for frac in (0.5, 0.25, 0.75):          # front, middle, back
            d = np.load(f0[int(frac * len(f0))])
            g, gmk = d['gt_elevation'], d['gt_mask'].astype(bool)
            stack.append(np.where(gmk, g, np.nan))
        env_thumbs.append((e, stack))

    # ── canvas (own dims: no bottom legend row, slim side margins → bigger
    # panels; the legend lives in the empty corner left of the training box) ──
    W2, H2 = 14.6, 8.15
    fig = plt.figure(figsize=(W2, H2))
    bg = fig.add_axes([0, 0, 1, 1], zorder=1)
    bg.set_xlim(0, W2); bg.set_ylim(0, H2); bg.axis('off')

    def pan(x, y, w_, h_, arr=None, cmap=None, vmin_=None, vmax_=None,
            border=INK, lw=1.0, zorder=3):
        ax = fig.add_axes([x / W2, y / H2, w_ / W2, h_ / H2], zorder=zorder)
        if arr is not None:
            ax.imshow(arr, cmap=cmap, vmin=vmin_, vmax=vmax_, origin='lower')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(border); sp.set_linewidth(lw)
        return ax

    py, ps = 5.55, 1.95
    cy = py + ps / 2

    # stage-group headers (uniform 0.54 in gaps between groups)
    for x0, x1, label in ((0.35, 7.28, 'data generation'),
                          (7.82, 9.47, 'completion network'),
                          (10.01, 14.26, 'output with uncertainty')):
        bg.plot([x0, x1], [py + ps + 0.22] * 2, color=RULE, lw=1.2)
        bg.text((x0 + x1) / 2, py + ps + 0.30, label, fontsize=17,
                color=INK, ha='center', va='bottom', weight='bold')

    # ① four depth cameras (2×2 grid, 1.95 in total)
    gx = 0.35
    dmax = max(float(np.percentile(d[d < 100], 98)) for d in per_cam.values())
    for i, cam in enumerate(CAMS):
        d = per_cam[cam][::4, ::4].copy(); d[d >= 100] = np.nan
        pan(gx + (i % 2) * 1.00, py + 1.00 - (i // 2) * 1.00,
            0.95, 0.95, d[::-1], CMAP_DEPTH, 0, dmax, lw=0.8)
    _caption(bg, gx + 0.975, py - 0.13,
             'four depth cameras\n(360°, rendered)')

    # ② merged point cloud (top view)
    ax = pan(2.84, py, ps, ps, lw=0.8)
    sub = pts[np.random.default_rng(0).choice(len(pts), 120_000, replace=False)]
    ax.scatter(sub[:, 1], sub[:, 0], c=sub[:, 2], s=0.08, cmap=CMAP_ELEV,
               vmin=vmin, vmax=vmax, rasterized=True)
    ax.plot(0, 0, 'k^', ms=5)
    ax.set(xlim=(-25, 25), ylim=(-25, 25), aspect='equal')
    _caption(bg, 2.84 + ps / 2, py - 0.13, 'robot-centric\npoint cloud')

    # ③ partial DEM + mask inset
    x3 = 5.33
    pan(x3, py, ps, ps, np.where(pm, pe, np.nan), CMAP_ELEV, vmin, vmax)
    pan(x3 + ps - 0.62, py + 0.05, 0.57, 0.57, pm,
        'gray', 0, 1, lw=0.7, zorder=4)
    _caption(bg, x3 + ps / 2, py - 0.13,
             'partial map $\\mathbf{E}\\odot \\mathbf{M}$\n+ mask $\\mathbf{M}$ ($P_{20}$, 0.2 m)')

    # ④ U-Net glyph
    ux, uw = 7.82, 1.65
    from matplotlib.patches import Polygon
    bg.add_patch(Polygon([(ux, py + ps), (ux + 0.68, py + ps - 0.58),
                          (ux + 0.68, py + 0.58), (ux, py)],
                         closed=True, facecolor='#DCE4EC', edgecolor=INK, lw=1.0))
    bg.add_patch(Polygon([(ux + uw, py + ps), (ux + uw - 0.68, py + ps - 0.58),
                          (ux + uw - 0.68, py + 0.58), (ux + uw, py)],
                         closed=True, facecolor='#DCE4EC', edgecolor=INK, lw=1.0))
    bg.add_patch(plt.Rectangle((ux + 0.68, py + 0.58), uw - 1.36, ps - 1.16,
                               facecolor='#C3D0DE', edgecolor=INK, lw=1.0))
    for dy in (0.30, 0.54):
        bg.annotate('', xy=(ux + uw - 0.48, py + ps - dy),
                    xytext=(ux + 0.48, py + ps - dy),
                    arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.8,
                                    mutation_scale=7,
                                    connectionstyle='arc3,rad=-0.12'))
    _caption(bg, ux + uw / 2, py - 0.13,
             'U-Net encoder–decoder\n2 ch in → $\\hat{\\mathbf{E}}$, $\\log\\hat{\\sigma}^2$')

    # ⑤ outputs: prediction on top, BOTH σ options side by side below.
    # Blue = β-NLL head, orange = TTA ensemble — the two uncertainty choices
    # are visible in the schema itself, not only in a legend.
    sx0, sw = 10.01, 0.84                      # σ pair: left x, square size
    pred_w = 0.86
    px0 = sx0 + (2 * sw + 0.08 - pred_w) / 2   # centred over the σ pair
    pan(px0, py + ps - pred_w, pred_w, pred_w, pred,
        CMAP_ELEV, vmin, vmax, lw=0.9)
    pan(sx0, py, sw, sw, sig, CMAP_SIGMA, 0, smax, border=C_NLL, lw=1.8)
    pan(sx0 + sw + 0.08, py, sw, sw, sig_tta, CMAP_SIGMA, 0, smax_tta,
        border=C_TTA, lw=1.8)
    bg.text(sx0 + sw + 0.04, py - 0.13, 'prediction $\\hat{\\mathbf{E}}$',
            fontsize=14, color=INK, ha='center', va='top')
    bg.text(sx0 + sw / 2 + 0.14, py - 0.40, '$\\hat{\\sigma}$: β-NLL (1 pass)',
            fontsize=12.5, color=C_NLL, ha='center', va='top')
    bg.text(sx0 + sw + 0.08 + sw / 2 + 0.24, py - 0.64, 'or TTA (8 passes)',
            fontsize=12.5, color=C_TTA, ha='center', va='top')

    # ⑥ composite output (the deliverable)
    cx2 = 12.31
    pan(cx2, py, ps, ps, comp, CMAP_ELEV, vmin, vmax, lw=2.2)
    _caption(bg, cx2 + ps / 2, py - 0.13,
             'composite map\n$\\mathbf{M}\\odot\\mathbf{E} + (\\mathbf{1}-\\mathbf{M})\\odot\\hat{\\mathbf{E}}$')

    # main-flow arrows (uniform gaps)
    _arrow(bg, 2.40, cy, 2.78, cy)
    _arrow(bg, 4.89, cy, 5.27, cy)
    _arrow(bg, 7.38, cy, 7.76, cy)
    _arrow(bg, 9.57, cy, 9.95, cy)
    _arrow(bg, 11.87, cy, 12.25, cy)
    # measurements bypass the network (Equation "composite").
    # The arc stays below every caption and ends just UNDER the composite
    # caption (pointing at it), so the arrow never crosses text.
    _arrow(bg, x3 + ps / 2, py - 0.70, cx2 + 0.50, py - 0.65, rad=0.18, lw=1.3)
    bg.text(9.60, 3.98, 'observed cells are copied unchanged',
            fontsize=14, color=INK, ha='center', style='italic')

    # ── training-only strip (content as in fig_pipeline, shifted down: the
    # bottom legend row is gone) ──────────────────────────────────────────────
    bx0, bx1 = 2.85, 12.90
    by0, by1 = 0.30, 3.65
    ty = 1.12
    th = 2.00
    bg.add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False,
                               edgecolor=TRAIN, lw=1.2, linestyle=(0, (4, 3))))
    bg.text(3.85, by1, 'training only', fontsize=16, color=TRAIN,
            ha='center', va='center', style='italic',
            bbox=dict(facecolor='white', edgecolor='none', pad=3))

    pan(3.25, ty, th, th, np.where(pm_ray > 0.5, pe_ray, np.nan),
        CMAP_ELEV, vmin, vmax, border=TRAIN, lw=1.2)
    _caption(bg, 3.25 + th / 2, ty - 0.16,
             'ray-cone + geometric\naugmentation', color=TRAIN, size=14)

    bg.text(6.52, ty + th / 2 + 0.10,
            'masked loss\n$w_v{=}0.1$ on $V$\n$w_h{=}1.0$ on $H$\n'
            '$\\beta$-NLL after L1 warmup',
            fontsize=14.5, color=INK, ha='center', va='center')

    # Each environment = a stack of three real samples (deck look), so the
    # glyph reads as five SUBSETS, not five single images.
    tx0, ts, tg, sd = 7.74, 0.88, 0.16, 0.05   # sd = stack offset per layer
    env_short = {'ForestEnv': 'Forest', 'Gascola': 'Gascola',
                 'ModularNeighborhood': 'Modular', 'OldTownSummer': 'Old Town',
                 'SeasonalForestWinter': 'Winter'}
    trow = ty + (th - ts) / 2
    for i, (e, stack) in enumerate(env_thumbs):
        held = (i == 3)
        x = tx0 + i * (ts + tg)
        grow = 0.08 if held else 0.0
        for layer in (2, 1):                    # back layers first
            off = grow + layer * sd
            pan(x + off, trow + off, ts, ts,
                stack[layer][::2, ::2], CMAP_ELEV, None, None,
                border=TRAIN, lw=0.7)
        pan(x - grow, trow - grow, ts + 2 * grow, ts + 2 * grow,
            stack[0][::2, ::2], CMAP_ELEV, None, None,
            border=INK if held else TRAIN, lw=3.0 if held else 0.8)
        bg.text(x + ts / 2, trow - grow - 0.10, env_short.get(e, e),
                fontsize=12.5, ha='center', va='top',
                color=INK if held else TRAIN)
        if held:
            bg.text(x + ts / 2, trow - grow - 0.33, 'held-out test',
                    fontsize=13.5, ha='center', va='top',
                    color=INK, weight='bold')
    _caption(bg, tx0 + (5 * ts + 4 * tg) / 2, ty - 0.16,
             '5× leave-one-environment-out', color=TRAIN, size=14)

    # Training feeds the network: a subtle dashed connector ending just UNDER
    # the U-Net caption (pointing at it), so it never crosses text and stays
    # visually quiet.
    _arrow(bg, 6.9, by1, 8.645, py - 0.67, style='--',
           color='#B4B4B4', lw=1.1, rad=-0.06)

    # ── legend: stacked in the empty corner left of the training box ──────────
    for ly, colr, lsty, lwd, lab in (
            (2.45, INK, '-', 2.4, 'inference data flow'),
            (2.05, TRAIN, (0, (4, 3)), 2.0, 'training only')):
        bg.plot([0.40, 0.85], [ly] * 2, color=colr, ls=lsty, lw=lwd)
        bg.text(0.97, ly, lab, fontsize=13.5, va='center')

    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)
    print(f'saved {out}  ({env}/{traj} frame {frame})')


# ── Figure: loss masking (companion schema to fig_pipeline) ──────────────────
#
# Same visual language as fig_pipeline (fonts, ink colors, caption style),
# same sample. Story: the two masks define the supervision regions, the
# region weights concentrate the per-cell loss in the holes.

C_EXCL, C_VALID, C_HOLE = '#EBEBEB', '#9BC4E2', '#E66101'


def fig_loss_masking(env, traj, frame, checkpoint: Path, out: Path):
    npz = np.load(DS_ROOT / env / traj / f'{traj}_sample_{frame:06d}.npz')
    pe = npz['partial_elevation']
    pm = npz['partial_mask'].astype(bool)
    gt, gm = npz['gt_elevation'], npz['gt_mask'].astype(bool)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(checkpoint, map_location=device)
    cfg, stats = ck['config'], ck['stats']
    model = build_model(cfg).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    m, s = stats['mean'], stats['std']
    pad, c0 = cfg['pad_to'], (cfg['pad_to'] - pe.shape[0]) // 2
    pe_n = np.zeros((pad, pad), np.float32)
    pm_p = np.zeros((pad, pad), np.float32)
    pe_n[c0:c0 + pe.shape[0], c0:c0 + pe.shape[1]] = \
        np.where(np.isfinite(pe) & pm, (pe - m) / s, 0.0)
    pm_p[c0:c0 + pe.shape[0], c0:c0 + pe.shape[1]] = pm
    inp = torch.from_numpy(np.stack([pe_n, pm_p]))[None].to(device)
    pred_t, _ = nll_predict(model, inp, cfg.get('uncertainty', False))
    crop = np.s_[c0:c0 + pe.shape[0], c0:c0 + pe.shape[1]]
    pred = (pred_t[0, 0].cpu().numpy() * s + m)[crop]

    gt_f = np.where(np.isfinite(gt), gt, 0.0)
    err = np.abs(pred - gt_f)

    # supervision regions: 0 = no GT, 1 = valid & observed, 2 = hole
    regions = np.zeros(gm.shape, np.uint8)
    regions[gm & pm] = 1
    regions[gm & ~pm] = 2
    wmap = np.where(gm & ~pm, 1.0, np.where(gm & pm, 0.1, 0.0))
    weighted = wmap * err

    from matplotlib.colors import ListedColormap
    reg_cmap = ListedColormap([C_EXCL, C_VALID, C_HOLE])

    W2, H2 = 12.2, 4.35
    fig = plt.figure(figsize=(W2, H2))
    bg = fig.add_axes([0, 0, 1, 1], zorder=1)
    bg.set_xlim(0, W2); bg.set_ylim(0, H2); bg.axis('off')

    def pan(x, y, w_, h_, arr, cmap, vmin=None, vmax=None, lw=1.0):
        ax = fig.add_axes([x / W2, y / H2, w_ / W2, h_ / H2], zorder=3)
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap).copy()
            cmap.set_bad(C_EXCL)          # NaN (no GT) = same gray as regions
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(INK); sp.set_linewidth(lw)
        return ax

    py2, ps2 = 1.95, 1.90
    cy2 = py2 + ps2 / 2

    pan(0.35, py2, ps2, ps2, pm, 'gray', 0, 1)
    _caption(bg, 0.35 + ps2 / 2, py2 - 0.13,
             'observability mask $\\mathbf{M}$\n(white = observed)')
    bg.text(2.60, cy2, '+', fontsize=26, color=INK, ha='center', va='center')
    pan(2.95, py2, ps2, ps2, gm, 'gray', 0, 1)
    _caption(bg, 2.95 + ps2 / 2, py2 - 0.13,
             'ground-truth mask $\\mathbf{M}^{\\mathrm{gt}}$\n(white = valid)')

    _arrow(bg, 4.92, cy2, 6.08, cy2)
    pan(6.15, py2, ps2, ps2, regions, reg_cmap, 0, 2)
    _caption(bg, 6.15 + ps2 / 2, py2 - 0.13, 'supervision regions')

    _arrow(bg, 8.12, cy2, 9.88, cy2)
    bg.text(9.00, cy2 + 0.22, '$\\times\\;|\\hat{\\mathbf{E}}-\\mathbf{E}^{\\mathrm{gt}}|$',
            fontsize=14.5, color=INK, ha='center', va='bottom')
    pan(9.95, py2, ps2, ps2, np.where(gm, weighted, np.nan), 'magma',
        0, float(np.percentile(weighted[gm & ~pm], 90)))
    _caption(bg, 9.95 + ps2 / 2, py2 - 0.13,
             'weighted per-cell loss\n$\\mathbf{w} \\odot |\\hat{\\mathbf{E}}-\\mathbf{E}^{\\mathrm{gt}}|$')

    # equation + region legend
    bg.text(W2 / 2, 0.98,
            '$\\mathcal{L} \\;=\\; \\frac{0.1}{|V|}\\sum_{(i,j)\\in V}'
            '\\ell_{ij} \\;+\\; \\frac{1.0}{|H|}\\sum_{(i,j)\\in H}\\ell_{ij}$'
            '$\\qquad\\ell$: L1 (warmup) $\\rightarrow$ $\\beta$-NLL '
            '($\\beta{=}0.5$)',
            fontsize=15.5, color=INK, ha='center', va='center')
    ly2 = 0.34
    for x, c, lab in ((1.05, C_EXCL, 'no ground truth (excluded)'),
                      (5.25, C_VALID, 'valid and observed ($w_v{=}0.1$)'),
                      (9.45, C_HOLE, 'hole $H$ ($w_h{=}1.0$)')):
        bg.add_patch(plt.Rectangle((x, ly2 - 0.10), 0.30, 0.20,
                                   facecolor=c, edgecolor=INK, lw=0.6))
        bg.text(x + 0.42, ly2, lab, fontsize=14, color=INK, va='center')

    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)
    print(f'saved {out}  ({env}/{traj} frame {frame})')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--env', default='OldTownSummer')
    p.add_argument('--traj', default='P2000')
    p.add_argument('--frame', type=int, default=400)
    p.add_argument('--checkpoint',
                   default=str(ROOT / 'runs/unet_20260701_195832/best.pth'))
    p.add_argument('--sample_idx', type=int, default=25,
                   help='test-split sample for the composite figure')
    p.add_argument('--frame_frac', type=float, default=0.5,
                   help='relative frame position for the environments figure')
    args = p.parse_args()

    FIG_DIR.mkdir(exist_ok=True)
    fig_pipeline('OldTownSummer', 'P2004', 1100, Path(args.checkpoint),
                 FIG_DIR / 'fig_pipeline.png')
    fig_pipeline_v2('OldTownSummer', 'P2004', 1100, Path(args.checkpoint),
                    FIG_DIR / 'fig_pipeline_v2.png')
    fig_loss_masking('OldTownSummer', 'P2004', 1100, Path(args.checkpoint),
                     FIG_DIR / 'fig_loss_masking.png')
    fig_data_generation(args.env, args.traj, args.frame,
                        FIG_DIR / 'fig_data_generation.png')
    fig_composite(Path(args.checkpoint), args.sample_idx,
                  FIG_DIR / 'fig_composite.png')
    fig_environments(args.frame_frac, FIG_DIR / 'fig_environments.png')

    ray_src = ROOT / 'reports' / 'preview_ray_augmentation.png'
    if ray_src.exists():
        shutil.copy(ray_src, FIG_DIR / 'fig_ray_augmentation.png')
        print(f'copied {ray_src.name} → figures/fig_ray_augmentation.png')


if __name__ == '__main__':
    main()
