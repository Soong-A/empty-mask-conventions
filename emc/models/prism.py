"""PRISM: a lightweight encoder-pyramid-decoder network (Appendix A.2, Table A.1).

    stem        space-to-depth rearrangement (3 x 256^2 -> 12 x 128^2), conv, GroupNorm, GELU
    encoder 1-3 depthwise residual blocks with dilation 1, 2 and a local divisive
                modulation block (DivisiveNorm) at 128^2, 64^2 and 32^2
    encoder 4   ConvNeXt blocks (depthwise 7 x 7, 4x expansion) at 16^2
    encoder 5   grouped channel-mixing blocks (4 groups) at 8^2
    pyramid     top-down feature pyramid, every level mapped to 16 channels
    decoder     progressive refinement m1..m5 (8^2 ... 128^2) and a full-resolution
                head m6 that sees the finest pyramid feature, the upsampled m5 and a
                non-learned image-gradient cue

forward() returns (gt_pre, key_points, prob): the six deep-supervision masks
m1..m6, two boundary bands (kept for interface compatibility, unused by the loss)
and the full-resolution probability map.

Ablation arms (Appendix B.4) are variants of one class:
    prism_dn     complete model (M); 'pgc_no_gain' is the name under which three of
                 its eight training runs were logged
    ng_no_div    A1, divisive modulation removed
    ng_sub       A2, subtractive control matched in learned quantities
    ng_no_hires  A3, full-resolution head removed (m6 = upsampled m5)
    ng_no_edge   A4, image-gradient cue removed from the full-resolution head

Every arm constructs the same set of modules in the same order, active or not, so
that for a given seed all arms start from identical weights; inactive modules are
"dormant" (registered, never called). ``count_parameters(model, active=True)``
excludes them: 40,401 active and 44,465 registered parameters for prism_dn.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConvBlock(nn.Module):
    """Depthwise 3x3 + pointwise conv, GroupNorm, GELU."""

    def __init__(self, in_ch, out_ch, groups=4, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))


class ConvNeXtBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 7, padding=3, groups=ch)
        self.norm = nn.GroupNorm(min(4, ch), ch)
        self.pw1 = nn.Conv2d(ch, ch * 4, 1)
        self.pw2 = nn.Conv2d(ch * 4, ch, 1)
        self.act = nn.GELU()

    def forward(self, x):
        r = x
        x = self.act(self.pw1(self.norm(self.dw(x))))
        return r + self.pw2(x)


class GroupShuffleBlock(nn.Module):
    def __init__(self, ch, groups=4):
        super().__init__()
        self.groups = groups
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.norm = nn.GroupNorm(min(groups, ch), ch)
        self.pw = nn.Conv2d(ch, ch, 1, groups=groups)
        self.act = nn.GELU()

    def channel_shuffle(self, x):
        b, c, h, w = x.shape
        x = x.view(b, self.groups, c // self.groups, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)

    def forward(self, x):
        r = x
        x = self.pw(self.norm(self.dw(x)))
        return self.act(r + self.channel_shuffle(x))


class SpaceToDepthStem(nn.Module):
    def __init__(self, c1: int = 8, kernel: int = 3):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.proj = nn.Conv2d(12, c1, kernel, padding=kernel // 2, bias=False)
        self.norm = nn.GroupNorm(min(4, c1), c1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.proj(self.unshuffle(x))))


class Stride2Stem(nn.Module):
    """Strided-conv stem; dormant in every variant used in the paper."""

    def __init__(self, out_ch: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(3, out_ch, kernel, stride=2, padding=kernel // 2, bias=False)
        self.norm = nn.GroupNorm(min(4, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DetailBlock(nn.Module):
    """Depthwise residual block with dilation."""

    def __init__(self, ch: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, kernel, padding=dilation * (kernel // 2),
                            dilation=dilation, groups=ch, bias=False)
        self.norm = nn.GroupNorm(min(4, ch), ch)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.pw(self.act(self.norm(self.dw(x))))


class DivisiveNorm(nn.Module):
    """Local divisive modulation (33 parameters per block).

    divisive:  E = AvgPool5(|x|); D = softplus(sigma) + softplus(w) E;
               G = clip((1/D) / mean(1/D), 0.25, 4); out = x (1 + gamma (G - 1))
    matched subtractive control (ng_sub): out = x + gamma (softplus(w) / softplus(sigma)) (x - AvgPool5(x)),
               the same learned quantities, local support and initialisation.
    gamma starts at 0, so every arm is the identity at initialisation. ``v`` is a
    dormant parameter kept for checkpoint compatibility.
    """

    def __init__(self, ch: int, surround: int = 5, divisive: bool = True):
        super().__init__()
        self.surround = surround
        self.divisive = divisive
        self.w = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.sigma = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.v = nn.Parameter(torch.full((1, ch, 1, 1), -1.0))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.surround // 2
        if not self.divisive:
            sur = F.avg_pool2d(x, self.surround, 1, p, count_include_pad=False)
            strength = F.softplus(self.w) / F.softplus(self.sigma).clamp_min(1e-4)
            return x + self.gamma * strength * (x - sur)
        e = F.avg_pool2d(x.abs(), self.surround, 1, p, count_include_pad=False)
        denom = F.softplus(self.sigma) + F.softplus(self.w) * e
        gain = 1.0 / denom.clamp_min(1e-4)
        gain = gain / gain.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        gain = gain.clamp(0.25, 4.0)
        return x * (1.0 + self.gamma * (gain - 1.0))


class GatedLateral(nn.Module):
    """Dormant in every variant used in the paper (registered for paired initialisation)."""

    def __init__(self, ch: int, kernel: int = 5):
        super().__init__()
        self.lateral = nn.Conv2d(ch, ch, kernel, padding=kernel // 2, groups=ch, bias=False)
        self.norm = nn.GroupNorm(min(4, ch), ch)
        self.proj = nn.Conv2d(ch, ch, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gate_dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.gate_pw = nn.Conv2d(ch, 1, 1)
        self.a = nn.Parameter(torch.tensor(2.0))
        self.b = nn.Parameter(torch.tensor(0.0))


class NeuromodGain(nn.Module):
    """Dormant in every variant used in the paper (registered for paired initialisation)."""

    def __init__(self, ch: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(4, ch // 2)
        self.fc1 = nn.Linear(ch, hidden)
        self.fc2 = nn.Linear(hidden, ch * 2)
        self.act = nn.GELU()
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)


class PRISMLiteEncoder(nn.Module):
    def __init__(self, c1=8, c2=16, c3=24, c4=32, c5=48, depths=(2, 2, 2, 2, 3),
                 stem_kernel=3, divnorm=True, divisive=True):
        super().__init__()
        n1, n2, n3, n4, n5 = depths
        self.stem = SpaceToDepthStem(c1, kernel=stem_kernel)
        self.stem_s2 = Stride2Stem(c1)                          # dormant
        self.stage1 = self._detail_stage(c1, n1)
        self.down1 = DSConvBlock(c1, c2, stride=2)
        self.stage2 = self._detail_stage(c2, n2)
        self.down2 = DSConvBlock(c2, c3, stride=2)
        self.stage3 = self._detail_stage(c3, n3)
        self.down3 = DSConvBlock(c3, c4, stride=2)
        self.stage4 = nn.Sequential(*[ConvNeXtBlock(c4) for _ in range(n4)])
        self.down4 = DSConvBlock(c4, c5, stride=2)
        self.stage5 = nn.Sequential(*[GroupShuffleBlock(c5) for _ in range(n5)])
        hi = [c1, c2, c3]                                      # 128^2, 64^2, 32^2
        self.use_divnorm = bool(divnorm)
        self.divnorm = nn.ModuleList([DivisiveNorm(c, divisive=divisive) for c in hi])
        self.lateral = nn.ModuleDict({str(i): GatedLateral(hi[i]) for i in (1, 2)})   # dormant
        self.gain = nn.ModuleList([NeuromodGain(c) for c in hi])                     # dormant

    @staticmethod
    def _detail_stage(ch, n):
        return nn.Sequential(*[DetailBlock(ch, kernel=3, dilation=(i % 3) + 1) for i in range(n)])

    def _modulate(self, x, idx):
        return self.divnorm[idx](x) if self.use_divnorm else x

    def forward(self, x):
        h = self.stem(x)
        f1 = self._modulate(self.stage1(h), 0)                   # [c1, 128^2]
        f2 = self._modulate(self.stage2(self.down1(f1)), 1)      # [c2, 64^2]
        f3 = self._modulate(self.stage3(self.down2(f2)), 2)      # [c3, 32^2]
        f4 = self.stage4(self.down3(f3))                         # [c4, 16^2]
        f5 = self.stage5(self.down4(f4))                         # [c5, 8^2]
        return f1, f2, f3, f4, f5


def generate_boundary_band(mask: torch.Tensor, width: int = 3) -> torch.Tensor:
    k = 2 * width + 1
    dilated = F.max_pool2d(mask, k, stride=1, padding=width)
    eroded = -F.max_pool2d(-mask, k, stride=1, padding=width)
    return dilated - eroded


def image_edge_map(image: torch.Tensor) -> torch.Tensor:
    """Non-learned image-gradient magnitude, normalised per image."""
    gray = image.mean(dim=1, keepdim=True)
    dx = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
    grad = torch.sqrt(dx * dx + dy * dy + 1e-8)
    return grad / (grad.amax(dim=(2, 3), keepdim=True) + 1e-3)


class RefineStage(nn.Module):
    def __init__(self, feat_channels: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(DSConvBlock(feat_channels + 1, hidden), DSConvBlock(hidden, hidden),
                                 nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat, prev_mask):
        return torch.sigmoid(self.net(torch.cat([feat, prev_mask], dim=1)))


class ImageGuidedRefiner(nn.Module):
    def __init__(self, feat_channels: int = 16, hidden: int = 8):
        super().__init__()
        in_ch = feat_channels + 2
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.GroupNorm(min(4, hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, m5_up, feat, img_edge):
        return torch.sigmoid(self.refine(torch.cat([feat, m5_up, img_edge], dim=1)))


class LightFPNNeck(nn.Module):
    def __init__(self, c1=8, c2=16, c3=24, c4=32, c5=48, fpn_dim=16):
        super().__init__()
        self.lat1 = nn.Conv2d(c1, fpn_dim, 1)
        self.lat2 = nn.Conv2d(c2, fpn_dim, 1)
        self.lat3 = nn.Conv2d(c3, fpn_dim, 1)
        self.lat4 = nn.Conv2d(c4, fpn_dim, 1)
        self.lat5 = nn.Conv2d(c5, fpn_dim, 1)
        self.smooth1 = DSConvBlock(fpn_dim, fpn_dim)
        self.smooth2 = DSConvBlock(fpn_dim, fpn_dim)
        self.smooth3 = DSConvBlock(fpn_dim, fpn_dim)
        self.smooth4 = DSConvBlock(fpn_dim, fpn_dim)
        self.smooth5 = DSConvBlock(fpn_dim, fpn_dim)

    def forward(self, f1, f2, f3, f4, f5):
        up = lambda t: F.interpolate(t, scale_factor=2, mode="nearest")  # noqa: E731
        p5 = self.smooth5(self.lat5(f5))
        p4 = self.smooth4(self.lat4(f4) + up(p5))
        p3 = self.smooth3(self.lat3(f3) + up(p4))
        p2 = self.smooth2(self.lat2(f2) + up(p3))
        p1 = self.smooth1(self.lat1(f1) + up(p2))
        return p1, p2, p3, p4, p5


VARIANTS = {
    #               divnorm, divisive, hires_refine, img_guide
    "prism_dn":    (True, True, True, True),
    "pgc_no_gain": (True, True, True, True),      # earlier name of prism_dn (3 of 8 runs)
    "ng_no_div":   (False, True, True, True),
    "ng_sub":      (True, False, True, True),
    "ng_no_hires": (True, True, False, True),
    "ng_no_edge":  (True, True, True, False),
}


class PRISMNet(nn.Module):
    def __init__(self, num_classes: int = 1, input_channels: int = 3, variant: str = "prism_dn",
                 c1=8, c2=16, c3=24, c4=32, c5=48, fpn_dim: int = 16, refine_hidden: int = 16,
                 stage6_hidden: int = 8, depths=(2, 2, 2, 2, 3), stem_kernel: int = 3):
        super().__init__()
        assert num_classes == 1 and input_channels == 3
        if variant not in VARIANTS:
            raise ValueError(f"unknown PRISM variant {variant!r}; choose from {sorted(VARIANTS)}")
        divnorm, divisive, hires_refine, img_guide = VARIANTS[variant]
        self.variant = variant
        self.hires_refine, self.img_guide = hires_refine, img_guide
        self.encoder = PRISMLiteEncoder(c1, c2, c3, c4, c5, depths=depths, stem_kernel=stem_kernel,
                                        divnorm=divnorm, divisive=divisive)
        self.fpn = LightFPNNeck(c1, c2, c3, c4, c5, fpn_dim=fpn_dim)
        self.stage1_init_head = nn.Conv2d(fpn_dim, 1, 1)
        nn.init.zeros_(self.stage1_init_head.weight)
        nn.init.zeros_(self.stage1_init_head.bias)
        self.stage1 = RefineStage(fpn_dim, refine_hidden)
        self.stage2 = RefineStage(fpn_dim, refine_hidden)
        self.stage3 = RefineStage(fpn_dim, refine_hidden)
        self.stage4 = RefineStage(fpn_dim, refine_hidden)
        self.stage5 = RefineStage(fpn_dim, refine_hidden)
        self.stage6 = ImageGuidedRefiner(feat_channels=fpn_dim, hidden=stage6_hidden)

    @staticmethod
    def _interp(x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x):
        f1, f2, f3, f4, f5 = self.encoder(x)
        p1, p2, p3, p4, p5 = self.fpn(f1, f2, f3, f4, f5)
        init_mask = torch.sigmoid(self.stage1_init_head(p5))
        m1 = self.stage1(p5, init_mask)
        m2 = self.stage2(p4, self._interp(m1, (16, 16)))
        m3 = self.stage3(p3, self._interp(m2, (32, 32)))
        m4 = self.stage4(p2, self._interp(m3, (64, 64)))
        m5 = self.stage5(p1, self._interp(m4, (128, 128)))
        m5_up = self._interp(m5, (256, 256))
        if self.hires_refine:
            edge = image_edge_map(x) if self.img_guide else torch.zeros_like(x[:, :1])
            m6 = self.stage6(m5_up, self._interp(p1, (256, 256)), edge)
        else:
            m6 = m5_up
        gt_pre = (m1, m2, m3, m4, m5, m6)
        key_points = (generate_boundary_band(m3), generate_boundary_band(m6))
        return gt_pre, key_points, m6


def dormant_parameters(model: PRISMNet) -> set:
    """Names of registered parameters that the forward pass never uses."""
    enc = model.encoder
    prefixes = ["encoder.lateral.", "encoder.gain.", "encoder.stem_s2."]
    if not enc.use_divnorm:
        prefixes.append("encoder.divnorm.")
    if not model.hires_refine:
        prefixes.append("stage6.")
    names = {n for n, _ in model.named_parameters() if n.startswith(tuple(prefixes))}
    names |= {n for n, _ in model.named_parameters() if n.startswith("encoder.divnorm.") and n.endswith(".v")}
    return names


def count_parameters(model: PRISMNet, active: bool = False) -> int:
    skip = dormant_parameters(model) if active else set()
    return sum(p.numel() for n, p in model.named_parameters() if n not in skip)
