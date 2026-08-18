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


def test_traversability_gate_is_conservative():
    """The uncertainty gate may only remove safe cells, never create them."""
    import numpy as np

    from elevcomp.traversability import (NON_TRAVERSABLE, TRAVERSABLE, apply_sigma_gate,
                                         slope_deg, traversability)

    flat = np.zeros((20, 20), np.float32)
    valid = np.ones((20, 20), bool)
    trav = traversability(slope_deg(flat, valid), 25.0)
    assert (trav[1:-1, 1:-1] == TRAVERSABLE).all()

    gated = apply_sigma_gate(trav, np.full((20, 20), 2.0, np.float32), tau=1.0)
    assert (gated[1:-1, 1:-1] == NON_TRAVERSABLE).all()
    assert ((gated == TRAVERSABLE) <= (trav == TRAVERSABLE)).all()


def test_elevation_rasterises_to_the_paper_grid():
    """50 x 50 m at 0.2 m/cell is the 251 x 251 grid the model expects."""
    import numpy as np

    from elevcomp.data.raster import apply_axis_mapping, grid_shape, rasterize_elevation

    bounds = [-25, 25, -25, 25, -25, 25]
    assert grid_shape(bounds, 0.2) == (251, 251)

    # a single point at the origin lands in the centre cell and nowhere else
    elev, mask = rasterize_elevation(np.array([[0.0, 0.0, 1.5]], np.float32), bounds, 0.2)
    assert mask.sum() == 1
    assert np.isclose(elev[mask][0], 1.5)

    pts = np.array([[1.0, 2.0, 3.0]], np.float32)
    assert np.array_equal(apply_axis_mapping(pts, 'yxz_neg'),
                          np.array([[2.0, 1.0, -3.0]], np.float32))
