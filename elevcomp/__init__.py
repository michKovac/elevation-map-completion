"""
Elevation map completion — reference implementation.

    data          building our dataset from TartanGround depth and scene clouds
    dataset       sample loading, normalisation and the ray-cone augmentation
    model         U-Net with a ResNet-34 encoder and a log-variance head
    losses        masked L1 and beta-NLL with hole weighting
    training      training and validation epoch loops, checkpointing
    inference     single-pass beta-NLL and D4 test-time-augmentation prediction
    metrics       MAE / RMSE / AbsRel / masked SSIM, in metres
    calibration   sparsification, AUSE, coverage@1-sigma
    traversability  slope, traversability, uncertainty gate, false-safe rate
    paths         where data, runs and configuration live

Runnable entry points are scripts/ and examples/.
"""
