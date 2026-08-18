"""
Completion network: U-Net with a ResNet-34 encoder and a heteroscedastic head.

The network takes the partial elevation map and its observation mask and
predicts the completed elevation. With uncertainty enabled it emits a second
channel holding the log-variance, so one forward pass yields both the
completion and its per-cell confidence.

References
    U-Net
        Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image
        Segmentation", MICCAI 2015. https://arxiv.org/abs/1505.04597
    ResNet-34 encoder (ImageNet-pretrained, ILSVRC 2012)
        He et al., "Deep Residual Learning for Image Recognition",
        CVPR 2016. https://arxiv.org/abs/1512.03385
    Heteroscedastic uncertainty head
        Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning
        for Computer Vision?", NeurIPS 2017. https://arxiv.org/abs/1703.04977
    Implementation
        Iakubovskii, "Segmentation Models Pytorch", 2019.
        https://github.com/qubvel/segmentation-models-pytorch
"""
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def _init_logvar_channel(net: nn.Module, num_classes: int) -> None:
    """
    Start the log-variance channel at high certainty.

    Zero weights and bias -5.0 mean the variance term contributes almost nothing
    at first, so the prediction channel receives full gradients from epoch 1.
    The head is located as the last Conv2d with `num_classes` outputs.
    """
    target = None
    for mod in net.modules():
        if isinstance(mod, nn.Conv2d) and mod.out_channels == num_classes:
            target = mod
    if target is None:
        raise RuntimeError('could not locate final prediction conv for log_var init')

    nn.init.zeros_(target.weight[1:])
    if target.bias is not None:
        nn.init.constant_(target.bias[1], -5.0)


class ElevationUNet(nn.Module):
    """
    Input : (B, 2, H, W)  [normalised elevation | observation mask]
    Output: (B, 1, H, W)  [prediction]  or  (B, 2, H, W)  [prediction | log_var]
    """

    def __init__(self, encoder_name: str = 'resnet34',
                 encoder_weights: str = 'imagenet', uncertainty: bool = False):
        super().__init__()
        self.uncertainty = uncertainty
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=2,
            classes=2 if uncertainty else 1,
            activation=None,
        )
        if uncertainty:
            _init_logvar_channel(self.net, num_classes=2)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.net(inp)


def build_model(cfg) -> nn.Module:
    """Build the network from a config dict (or a checkpoint's stored config)."""
    model = cfg.get('model', 'unet_pytorch')
    if model != 'unet_pytorch':
        raise ValueError(
            f'Unknown model: {model!r}. This repository implements the U-Net with a '
            'ResNet-34 encoder used in the paper; the other architectures it compares '
            'against are standard and not reproduced here.')

    # TOML has no None literal, so from-scratch training is spelled as a string.
    enc_w = cfg.get('encoder_weights', 'imagenet')
    if isinstance(enc_w, str) and enc_w.lower() in ('none', 'null', ''):
        enc_w = None

    return ElevationUNet(encoder_name=cfg.get('encoder_name', 'resnet34'),
                         encoder_weights=enc_w,
                         uncertainty=cfg.get('uncertainty', False))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
