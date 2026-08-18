"""
Neural network architectures for elevation map completion.

PConv / PConv-UNet
    PartialConv layer and PConvUNet adapted from:
        naoto0804/pytorch-inpainting-with-partial-conv
        https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/net.py
        (MIT License)
    Original paper:
        Liu et al., "Image Inpainting for Irregular Holes Using Partial Convolutions",
        ECCV 2018. https://arxiv.org/abs/1804.07723

U-Net backbone
    Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image
    Segmentation", MICCAI 2015. https://arxiv.org/abs/1505.04597

ResNet-34 encoder (AttentionUNet)
    He et al., "Deep Residual Learning for Image Recognition",
    CVPR 2016. https://arxiv.org/abs/1512.03385

scSE decoder attention (AttentionUNet)
    Roy et al., "Concurrent Spatial and Channel 'Squeeze & Excitation' in
    Fully Convolutional Networks", MICCAI 2018. https://arxiv.org/abs/1803.02579

segmentation_models_pytorch library
    Iakubovskii, "Segmentation Models Pytorch", 2019.
    https://github.com/qubvel/segmentation-models-pytorch

Heteroscedastic uncertainty heads (log_var_head)
    Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning
    for Computer Vision?", NeurIPS 2017. https://arxiv.org/abs/1703.04977

Architecture adapted for single-channel (elevation) input/output on 256×256 maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# ─────────────────────────────────────────────────────────────────────────────
# Partial Convolution  (direct port from naoto0804, corrected bias formula)
# ─────────────────────────────────────────────────────────────────────────────

class PartialConv(nn.Module):
    """
    Partial convolution layer.

    Equation from Liu et al. (eq. 1):
        W^T * (M ⊙ X) / sum(M)  +  b     if sum(M) > 0
        0                                  otherwise

    Equivalent form used here (avoids recomputing bias):
        [C(M ⊙ X) - C(0)] / D(M)  +  C(0)

    where C(·) = W^T·(·) + b  (the raw convolution),
          C(0) = b,
          D(M) = mask_conv(M)  (counts valid entries per output position).

    Mask update (eq. 2):
        M_out[p] = 1   if sum(M in kernel) > 0
                   0   otherwise

    Key implementation detail: mask_conv has the SAME in_channels as input_conv
    so that the mask tracks per-channel validity correctly when multi-channel
    masks are concatenated in the decoder (Liu et al., Sec. 3).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.input_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias,
        )
        # Mask update conv: same shape as input_conv but weights fixed to 1, no bias.
        # Computes sum of valid entries in each receptive field (per output channel).
        self.mask_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias=False,
        )
        nn.init.constant_(self.mask_conv.weight, 1.0)
        for p in self.mask_conv.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x    : (B, C_in, H, W)  feature values (0 where mask == 0)
        mask : (B, C_in, H, W)  binary validity mask (same shape as x)
        Returns: (output, new_mask)  both (B, C_out, H', W')
        """
        # Masked convolution
        output = self.input_conv(x * mask)

        # Bias term C(0) = b
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            # D(M): count of valid input entries per output position & channel
            output_mask = self.mask_conv(mask)

        # Positions where ALL input mask entries in the kernel are 0
        no_update_holes = output_mask == 0

        # Avoid division by zero: replace 0 with 1 (will be masked out after)
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        # Apply re-normalisation with bias correction
        output_pre = (output - output_bias) / mask_sum + output_bias
        # Force 0 at fully-masked output positions
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        # New mask: 1 where any input was valid, 0 where all inputs were holes
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


# ─────────────────────────────────────────────────────────────────────────────
# Building block: PartialConv → BN → Activation
# ─────────────────────────────────────────────────────────────────────────────

class _PCBActiv(nn.Module):
    """PartialConv + optional BatchNorm + optional activation."""

    _SAMPLE = {
        'down-7': (7, 2, 3),   # (kernel, stride, padding)
        'down-5': (5, 2, 2),
        'down-3': (3, 2, 1),
        'none-3': (3, 1, 1),
    }

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        bn: bool = True,
        sample: str = 'none-3',
        activ: str = 'relu',   # 'relu' | 'leaky' | None
        conv_bias: bool = False,
    ):
        super().__init__()
        k, s, p = self._SAMPLE[sample]
        self.conv = PartialConv(in_ch, out_ch, k, s, p, bias=conv_bias)
        self.bn   = nn.BatchNorm2d(out_ch) if bn else None
        if activ == 'relu':
            self.activ = nn.ReLU()
        elif activ == 'leaky':
            self.activ = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.activ = None

    def forward(self, x, mask):
        x, mask = self.conv(x, mask)
        if self.bn is not None:
            x = self.bn(x)
        if self.activ is not None:
            x = self.activ(x)
        return x, mask


# ─────────────────────────────────────────────────────────────────────────────
# PConv-UNet
# ─────────────────────────────────────────────────────────────────────────────

class PConvUNet(nn.Module):
    """
    PConv-UNet for elevation map completion.

    Architecture follows naoto0804/pytorch-inpainting-with-partial-conv
    with layer_size=6 (adapted for 256×256 input) and single-channel I/O.

    Encoder kernel sizes: 7, 5, 5, 3, 3, 3  (all stride 2)
    Decoder: PConv 3×3 stride 1, upsample nearest
    No BN in enc_1 and dec_1; no activation in dec_1.

    Input convention (matches dataset.py):
        inp  : (B, 2, H, W) — ch 0: normalized elevation, ch 1: partial_mask
    The mask is extracted from ch 1 and used as the PConv mask.
    Internally the network operates on 1-channel elevation features.

    Output:
        pred : (B, 1, H, W) — predicted complete elevation (normalized)
    """

    def __init__(self, layer_size: int = 6, upsampling_mode: str = 'nearest',
                 uncertainty: bool = False):
        super().__init__()
        self.layer_size      = layer_size
        self.upsampling_mode = upsampling_mode
        self.uncertainty     = uncertainty
        input_channels = 1

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc_1 = _PCBActiv(input_channels, 64,  bn=False, sample='down-7')
        self.enc_2 = _PCBActiv(64,  128, sample='down-5')
        self.enc_3 = _PCBActiv(128, 256, sample='down-5')
        self.enc_4 = _PCBActiv(256, 512, sample='down-3')
        for i in range(4, layer_size):
            setattr(self, f'enc_{i+1}', _PCBActiv(512, 512, sample='down-3'))

        # ── Decoder ───────────────────────────────────────────────────────────
        for i in range(4, layer_size):
            setattr(self, f'dec_{i+1}', _PCBActiv(512 + 512, 512, activ='leaky'))
        self.dec_4 = _PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = _PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = _PCBActiv(128 + 64,  64,  activ='leaky')
        # dec_1 always outputs 1 channel (prediction)
        self.dec_1 = _PCBActiv(
            64 + input_channels, 1,
            bn=False, activ=None, conv_bias=True,
        )
        # Separate log_var head: branches off the 65-ch pre-dec_1 feature so it
        # has access to rich spatial features rather than the scalar prediction.
        self.log_var_head = None
        if uncertainty:
            self.log_var_head = nn.Conv2d(64 + input_channels, 1, kernel_size=1, bias=True)
            nn.init.zeros_(self.log_var_head.weight)
            nn.init.constant_(self.log_var_head.bias, -5.0)

    def forward(self, inp: torch.Tensor, input_mask: torch.Tensor = None):
        """
        inp        : (B, 2, H, W)  dataset format [elevation | partial_mask]
        input_mask : (B, 1, H, W)  override mask (optional)
        Returns    : (B, 1, H, W)  predicted elevation
        """
        x    = inp[:, :1]                   # elevation channel only
        mask = inp[:, 1:2] if input_mask is None else input_mask
        # Expand mask to match the single input channel
        mask = mask.expand_as(x)            # (B, 1, H, W)

        # ── Encoder forward ───────────────────────────────────────────────────
        h      = {'h_0': x,    }
        h_mask = {'h_0': mask, }

        h_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            h_key  = f'h_{i}'
            h[h_key], h_mask[h_key] = getattr(self, f'enc_{i}')(
                h[h_prev], h_mask[h_prev]
            )
            h_prev = h_key

        # ── Decoder forward ───────────────────────────────────────────────────
        # Start from bottleneck
        feat = h[f'h_{self.layer_size}']
        mask_d = h_mask[f'h_{self.layer_size}']

        for i in range(self.layer_size, 1, -1):
            skip_key = f'h_{i-1}'

            # Upsample features and mask
            feat   = F.interpolate(feat,   scale_factor=2, mode=self.upsampling_mode)
            mask_d = F.interpolate(mask_d, scale_factor=2, mode='nearest')

            # Concatenate with skip connection (features AND mask)
            feat   = torch.cat([feat,   h[skip_key]],      dim=1)
            mask_d = torch.cat([mask_d, h_mask[skip_key]], dim=1)

            feat, mask_d = getattr(self, f'dec_{i}')(feat, mask_d)

        # Last decoder step (dec_1): run separately so log_var_head can branch
        # off the rich pre-dec_1 feature rather than the scalar prediction output.
        feat   = F.interpolate(feat,   scale_factor=2, mode=self.upsampling_mode)
        mask_d = F.interpolate(mask_d, scale_factor=2, mode='nearest')
        pre_dec1 = torch.cat([feat,   h['h_0']],      dim=1)
        mask_d   = torch.cat([mask_d, h_mask['h_0']], dim=1)
        feat, mask_d = self.dec_1(pre_dec1, mask_d)   # (B, 1, H, W)

        if self.log_var_head is not None:
            return torch.cat([feat, self.log_var_head(pre_dec1)], dim=1)
        return feat   # (B, 1, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Plain U-Net baseline  (no partial conv — ablation)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleUNet(nn.Module):
    """
    Standard encoder-decoder U-Net without partial convolutions.
    Elevation and mask are concatenated as a 2-channel input.
    Used as the ablation baseline against PConvUNet.

    Architecture follows the U-Net encoder-decoder design:
        Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image
        Segmentation", MICCAI 2015. https://arxiv.org/abs/1505.04597
    """

    def __init__(self, base_ch: int = 64, uncertainty: bool = False):
        super().__init__()
        c = base_ch

        def _enc(ic, oc, ks=3, stride=1):
            pad = ks // 2
            return nn.Sequential(
                nn.Conv2d(ic, oc, ks, stride, pad),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
            )

        def _dec(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, 1, 1),
                nn.BatchNorm2d(oc),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.enc1 = _enc(2,   c,    ks=7)
        self.enc2 = _enc(c,   c*2,  stride=2)
        self.enc3 = _enc(c*2, c*4,  stride=2)
        self.enc4 = _enc(c*4, c*8,  stride=2)
        self.enc5 = _enc(c*8, c*8,  stride=2)

        self.dec4 = _dec(c*8 + c*8, c*8)
        self.dec3 = _dec(c*8 + c*4, c*4)
        self.dec2 = _dec(c*4 + c*2, c*2)
        self.dec1 = _dec(c*2 + c,   c)
        self.out  = nn.Conv2d(c, 1, 1)
        # Separate log_var head initialized to -5: starts with high certainty so
        # the prediction head receives strong L1 gradients from the first epoch.
        self.log_var_head = None
        if uncertainty:
            self.log_var_head = nn.Conv2d(c, 1, 1, bias=True)
            nn.init.zeros_(self.log_var_head.weight)
            nn.init.constant_(self.log_var_head.bias, -5.0)

    def forward(self, inp: torch.Tensor):
        e1 = self.enc1(inp)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        def _up(x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
            return torch.cat([x, skip], dim=1)

        d4 = self.dec4(_up(e5, e4))
        d3 = self.dec3(_up(d4, e3))
        d2 = self.dec2(_up(d3, e2))
        d1 = self.dec1(_up(d2, e1))
        pred = self.out(d1)
        if self.log_var_head is not None:
            return torch.cat([pred, self.log_var_head(d1)], dim=1)
        return pred


# ─────────────────────────────────────────────────────────────────────────────

class AttentionUNet(nn.Module):
    """
    U-Net with ResNet34 encoder and scSE decoder attention, built via
    segmentation_models_pytorch (https://github.com/qubvel/segmentation-models-pytorch).

    References:
        U-Net:    Ronneberger et al., MICCAI 2015. https://arxiv.org/abs/1505.04597
        ResNet34: He et al., CVPR 2016. https://arxiv.org/abs/1512.03385
                  Encoder weights pretrained on ImageNet (ILSVRC 2012).
        scSE:     Roy et al., "Concurrent Spatial and Channel 'Squeeze & Excitation'
                  in Fully Convolutional Networks", MICCAI 2018.
                  https://arxiv.org/abs/1803.02579

    Input : (B, 2, H, W)  [elevation | partial_mask]
    Output: (B, 1, H, W)  predicted elevation
    """

    def __init__(self, uncertainty: bool = False):
        super().__init__()
        self.uncertainty = uncertainty
        self.net = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=2,
            classes=2 if uncertainty else 1,
            activation=None,
            decoder_attention_type="scse",
        )
        if uncertainty:
            # Initialize log_var output channel (ch 1) to -5: high certainty at start
            # so the prediction channel receives full L1 gradients from epoch 1.
            seg_conv = self.net.segmentation_head[0]
            nn.init.zeros_(seg_conv.weight[1:])
            nn.init.constant_(seg_conv.bias[1], -5.0)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.net(inp)


# ─────────────────────────────────────────────────────────────────────────────
# Library SOTA wrappers (segmentation_models_pytorch), adapted to our task:
#   - 2-channel input  [elevation | partial_mask]  (in_channels=2)
#   - regression output, no activation             (activation=None)
#   - β-NLL uncertainty via a 2nd output channel   (classes=2 → [pred | log_var])
# Same I/O contract and log_var init (-5.0) as AttentionUNet, so training,
# losses and inference.split_output work unchanged.  One architecture per
# experiment, selected by cfg['model']; encoder is overridable via
# cfg['encoder_name'] and cfg['encoder_weights'] (default: ImageNet, matching
# AttentionUNet).
# ─────────────────────────────────────────────────────────────────────────────

def _init_logvar_channel(net: nn.Module, num_classes: int) -> None:
    """
    Initialize the log_var output channel (index 1) of the final prediction conv
    to zero weights / bias -5.0 — high initial certainty so the prediction channel
    receives full L1 gradients from epoch 1 (matches AttentionUNet / SimpleUNet).

    Locates the final head conv robustly as the last nn.Conv2d whose out_channels
    equals num_classes (works across smp heads: SegmentationHead, DPT head, ...).
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


class SegformerUNet(nn.Module):
    """
    SegFormer (MiT hierarchical transformer encoder + lightweight all-MLP decoder),
    built via segmentation_models_pytorch.  Transformer inductive bias baseline;
    MiT-B2 (~24.7M) is capacity-matched to AttentionUNet (~24.6M).

    Reference:
        Xie et al., "SegFormer: Simple and Efficient Design for Semantic
        Segmentation with Transformers", NeurIPS 2021.
        https://arxiv.org/abs/2105.15203

    Input : (B, 2, H, W)  [elevation | partial_mask]
    Output: (B, 1|2, H, W)  [pred] or [pred | log_var]
    """

    def __init__(self, encoder_name: str = 'mit_b2',
                 encoder_weights: str = 'imagenet', uncertainty: bool = False):
        super().__init__()
        self.uncertainty = uncertainty
        self.net = smp.Segformer(
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


class UNetPytorch(nn.Module):
    """
    Classic U-Net with a ResNet-34 encoder, built via segmentation_models_pytorch
    — a fully referenceable library implementation of the plain-U-Net baseline
    (the library counterpart to the custom SimpleUNet).  Identical to
    AttentionUNet but WITHOUT scSE decoder attention, so the pair
    unet_pytorch ↔ attunet isolates the contribution of the attention mechanism
    (same ResNet-34 backbone, ImageNet-pretrained).

    References:
        U-Net:    Ronneberger et al., MICCAI 2015. https://arxiv.org/abs/1505.04597
        ResNet34: He et al., CVPR 2016. https://arxiv.org/abs/1512.03385
                  Encoder weights pretrained on ImageNet (ILSVRC 2012).
        smp:      Iakubovskii, "Segmentation Models Pytorch", 2019.
                  https://github.com/qubvel/segmentation-models-pytorch

    Input : (B, 2, H, W)  [elevation | partial_mask]
    Output: (B, 1|2, H, W)  [pred] or [pred | log_var]
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
    unc = cfg.get('uncertainty', False)
    # encoder_weights: 'imagenet' (default, like attunet) or 'none'/'' for
    # from-scratch training (TOML has no None literal → accept a string).
    enc_w = cfg.get('encoder_weights', 'imagenet')
    if isinstance(enc_w, str) and enc_w.lower() in ('none', 'null', ''):
        enc_w = None
    if cfg['model'] == 'pconv':
        return PConvUNet(layer_size=cfg.get('layer_size', 6), uncertainty=unc)
    elif cfg['model'] == 'unet':
        return SimpleUNet(base_ch=cfg.get('base_channels', 64), uncertainty=unc)
    elif cfg['model'] == 'attunet':
        return AttentionUNet(uncertainty=unc)
    elif cfg['model'] == 'unet_pytorch':
        return UNetPytorch(encoder_name=cfg.get('encoder_name', 'resnet34'),
                           encoder_weights=enc_w, uncertainty=unc)
    elif cfg['model'] == 'segformer':
        return SegformerUNet(encoder_name=cfg.get('encoder_name', 'mit_b2'),
                             encoder_weights=enc_w, uncertainty=unc)
    else:
        raise ValueError(f'Unknown model: {cfg["model"]}')


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
