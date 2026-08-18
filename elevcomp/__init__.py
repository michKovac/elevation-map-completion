"""
Core library for elevation map completion.

Importable modules (no entry points here — see repository root for runnable
scripts and tools/ for standalone analyses):

    dataset      — ElevationDataset, dataloaders, ray-cone augmentation
    model        — PConvUNet / SimpleUNet / AttentionUNet + uncertainty heads
    losses       — masked L1 / β-NLL elevation loss
    metrics      — MAE / RMSE / AbsRel / masked SSIM (denormalized metres)
    training     — train/eval epoch loops, checkpoint helpers
    inference    — single-pass (β-NLL) and D4 TTA ensemble prediction
    baselines    — classical hole-filling baselines (nearest, linear, IDW, …)
    folds        — environment discovery + leave-one-environment-out folds
    calibration  — uncertainty calibration (sparsification, AUSE, coverage)
    figures      — shared matplotlib figures (training vis, calibration, summary)
    cv           — cross-validation experiment driver (train/eval/aggregate)
    utils        — seeding, environment info, CSV/statistics helpers
"""
