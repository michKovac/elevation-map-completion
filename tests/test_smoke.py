"""Smoke tests that run on synthetic data, without the generated dataset."""
import numpy as np
import torch

from elevcomp.dataset import _apply_ray_augmentation
from elevcomp.inference import D4_TRANSFORMS, d4_apply, d4_invert, nll_predict
from elevcomp.losses import ElevationLoss
from elevcomp.model import build_model, count_parameters

CFG = {'model': 'unet_pytorch', 'uncertainty': True, 'encoder_weights': 'none'}


def test_resnet34_unet_shapes_and_size():
    net = build_model(CFG)
    assert 24e6 < count_parameters(net) < 25e6
    out = net(torch.zeros(2, 2, 256, 256))
    assert out.shape == (2, 2, 256, 256)   # prediction + log_var


def test_nll_predict_returns_positive_sigma():
    net = build_model(CFG).eval()
    pred, sigma = nll_predict(net, torch.zeros(1, 2, 256, 256), has_uncertainty=True)
    assert pred.shape == sigma.shape == (1, 1, 256, 256)
    assert torch.all(sigma > 0)


def test_d4_transforms_are_invertible():
    x = torch.randn(1, 1, 16, 16)
    for k, hflip in D4_TRANSFORMS:
        assert torch.allclose(d4_invert(d4_apply(x, k, hflip), k, hflip), x)


def test_ray_augmentation_only_removes_observations():
    rng = np.random.default_rng(0)
    pe = rng.normal(size=(251, 251)).astype(np.float32)
    pm = np.ones((251, 251), dtype=np.float32)

    pe_aug, pm_aug = _apply_ray_augmentation(pe, pm, n_cones_max=3, angle_range=(5, 25))

    assert pm_aug.sum() < pm.sum()                      # something was removed
    assert np.all(pm_aug[pm_aug > 0] == 1)              # mask stays binary
    kept = pm_aug > 0
    assert np.array_equal(pe_aug[kept], pe[kept])       # kept cells are untouched
    assert np.all(np.isnan(pe_aug[~kept]))              # removed cells become NaN


def test_hole_weighting_dominates_the_loss():
    loss = ElevationLoss(w_valid=0.1, w_hole=1.0)
    gt = torch.zeros(1, 1, 8, 8)
    gt_mask = torch.ones(1, 1, 8, 8)
    partial_mask = torch.ones(1, 1, 8, 8)
    partial_mask[..., :4] = 0.0            # left half is a hole

    err_in_hole = torch.zeros(1, 1, 8, 8); err_in_hole[..., :4] = 1.0
    err_in_obs = torch.zeros(1, 1, 8, 8); err_in_obs[..., 4:] = 1.0

    assert (loss(err_in_hole, gt, gt_mask, partial_mask)
            > loss(err_in_obs, gt, gt_mask, partial_mask))
