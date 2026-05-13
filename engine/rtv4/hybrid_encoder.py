"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation

from ..core import register

import logging
_logger = logging.getLogger(__name__)

__all__ = ['HybridEncoder']


class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = \
            ch_in, ch_out, kernel_size, stride, g, padding, bias

    def forward(self, x):
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv_bn_fused'):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('norm')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# TODO, add activation for cv1 following YOLOv10
# self.cv1 = Conv(c1, c2, 1, 1)
# self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)
class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s, act=None):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu",
                 bottletype=VGGBlock):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_2 = self.conv2(x)
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        return self.conv3(x_1 + x_2)

class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=3,
                 bias=False,
                 act="silu"):
        super().__init__()
        self.c = c3//2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(CSPLayer(c3//2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv3 = nn.Sequential(CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv4 = ConvNormLayer_fuse(c3+(2*c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class PrewittEdgeGuidedEnhance(nn.Module):
    """Prewitt edge prior for shallow feature enhancement.

    The fixed kernels follow the Prewitt operator templates:
        Gx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
        Gy = [[ 1, 1, 1], [ 0, 0, 0], [-1,-1,-1]]
    Gradient magnitude follows the paper description with infinity norm,
    and the adaptive threshold is T = k * Gmax.
    """

    def __init__(self, channels, alpha=0.15, threshold_ratio=0.2, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.threshold_ratio = threshold_ratio
        self.eps = eps
        kernel_x = torch.tensor(
            [[-1., 0., 1.],
             [-1., 0., 1.],
             [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[1., 1., 1.],
             [0., 0., 0.],
             [-1., -1., -1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('prewitt_kernel_x', kernel_x, persistent=False)
        self.register_buffer('prewitt_kernel_y', kernel_y, persistent=False)
        self.edge_proj = nn.Conv2d(1, channels, kernel_size=1, bias=True)

    def forward(self, x):
        intensity = x.mean(dim=1, keepdim=True)
        kernel_x = self.prewitt_kernel_x.to(device=x.device, dtype=x.dtype)
        kernel_y = self.prewitt_kernel_y.to(device=x.device, dtype=x.dtype)
        gx = F.conv2d(intensity, kernel_x, padding=1)
        gy = F.conv2d(intensity, kernel_y, padding=1)
        grad = torch.maximum(gx.abs(), gy.abs())
        gmax = grad.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
        edge = grad * (grad >= self.threshold_ratio * gmax).to(grad.dtype)
        edge = edge / (gmax + self.eps)
        edge = torch.nan_to_num(edge, nan=0.0, posinf=1.0, neginf=0.0).detach()
        gate = torch.sigmoid(self.edge_proj(edge))
        return x * (1.0 + self.alpha * gate)


class PrewittFranklinEdgeGuidedEnhance(nn.Module):
    """Prewitt-Franklin edge gate following the paper's full decision flow.

    Implemented steps:
        1. Prewitt pixel-level coarse edge extraction with infinity norm.
        2. T = k * Gmax with k = 0.2.
        3. 7x7 Franklin moment templates M00, M11, M20, M31 and M40.
        4. phi, l1, l2, l and k from Eq. (15), (17)-(20).
        5. Edge decision kt < k, lt < l and |l1-l2| < lt from Eq. (23).

    The Franklin templates are sampled from the Franklin radial basis functions
    phi_0 ... phi_4 given in Eq. (3)-(7), combined with the polar moment form
    in Eq. (9). The module uses the resulting decision map as a fixed
    image-processing prior and learns only the 1x1 projection into feature
    channels.
    """

    def __init__(self, channels, alpha=0.15, threshold_ratio=0.2, kernel_size=7, eps=1e-6):
        super().__init__()
        if kernel_size != 7:
            raise ValueError("The paper-selected Franklin template size is 7x7.")
        self.alpha = alpha
        self.threshold_ratio = threshold_ratio
        self.eps = eps
        kernel_x = torch.tensor(
            [[-1., 0., 1.],
             [-1., 0., 1.],
             [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[1., 1., 1.],
             [0., 0., 0.],
             [-1., -1., -1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('prewitt_kernel_x', kernel_x, persistent=False)
        self.register_buffer('prewitt_kernel_y', kernel_y, persistent=False)
        self.register_buffer('franklin_kernels', self._build_franklin_kernels(kernel_size), persistent=False)
        self.edge_proj = nn.Conv2d(4, channels, kernel_size=1, bias=True)

    @staticmethod
    def _franklin_phi(order, r):
        if order == 0:
            return torch.ones_like(r)
        if order == 1:
            return (3.0 ** 0.5) * (2.0 * r - 1.0)
        if order == 2:
            return torch.where(
                r < 0.5,
                (3.0 ** 0.5) * (1.0 - 4.0 * r),
                (3.0 ** 0.5) * (4.0 * r - 3.0),
            )
        if order == 3:
            scale = (33.0 ** 0.5) / 11.0
            return torch.where(
                r < 0.25,
                scale * (5.0 - 38.0 * r),
                torch.where(
                    r < 0.5,
                    scale * (26.0 * r - 11.0),
                    scale * (5.0 - 6.0 * r),
                ),
            )
        if order == 4:
            scale = (231.0 ** 0.5) / 77.0
            return torch.where(
                r < 0.25,
                scale * (1.0 - 12.0 * r),
                torch.where(
                    r < 0.5,
                    scale * (36.0 * r - 11.0),
                    torch.where(
                        r < 0.75,
                        scale * (45.0 - 76.0 * r),
                        scale * (100.0 * r - 87.0),
                    ),
                ),
            )
        raise ValueError(f"Unsupported Franklin order: {order}")

    @classmethod
    def _build_franklin_kernels(cls, kernel_size):
        coords = torch.linspace(-1.0, 1.0, kernel_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        rr = torch.sqrt(xx * xx + yy * yy).clamp(max=1.0)
        theta = torch.atan2(yy, xx)
        disk = (xx * xx + yy * yy <= 1.0).to(torch.float32)
        area = (2.0 / (kernel_size - 1)) ** 2

        def moment_kernel(order, angular_order, part='real'):
            radial = cls._franklin_phi(order, rr)
            scale = (order + 1.0) / torch.pi
            if angular_order == 0:
                angular = torch.ones_like(theta)
            elif part == 'real':
                angular = torch.cos(angular_order * theta)
            else:
                angular = -torch.sin(angular_order * theta)
            return scale * radial * angular * disk * area

        kernels = torch.stack([
            moment_kernel(0, 0),
            moment_kernel(1, 1, 'real'),
            moment_kernel(1, 1, 'imag'),
            moment_kernel(2, 0),
            moment_kernel(3, 1, 'real'),
            moment_kernel(3, 1, 'imag'),
            moment_kernel(4, 0),
        ], dim=0).unsqueeze(1)
        return kernels

    def _safe_div(self, num, den):
        den_safe = torch.where(den >= 0, torch.full_like(den, self.eps), torch.full_like(den, -self.eps))
        den = torch.where(den.abs() < self.eps, den_safe, den)
        return num / den

    def forward(self, x):
        intensity = x.mean(dim=1, keepdim=True)

        kernel_x = self.prewitt_kernel_x.to(device=x.device, dtype=x.dtype)
        kernel_y = self.prewitt_kernel_y.to(device=x.device, dtype=x.dtype)
        gx = F.conv2d(intensity, kernel_x, padding=1)
        gy = F.conv2d(intensity, kernel_y, padding=1)
        grad = torch.maximum(gx.abs(), gy.abs())
        gmax = grad.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
        prewitt_edge = (grad >= self.threshold_ratio * gmax).to(grad.dtype)
        grad_norm = grad / (gmax + self.eps)

        kernels = self.franklin_kernels.to(device=x.device, dtype=x.dtype)
        moments = F.conv2d(intensity, kernels, padding=3)
        f00, f11_re, f11_im, f20, f31_re, f31_im, f40 = moments.chunk(7, dim=1)
        phi = torch.atan2(f11_im, f11_re + self.eps)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        f11_rot = f11_re * cos_phi + f11_im * sin_phi
        f31_rot = f31_re * cos_phi + f31_im * sin_phi
        f20_rot = f20
        f40_rot = f40

        l1_arg = self._safe_div(5.0 * f40_rot + 3.0 * f20_rot, 8.0 * f20_rot)
        l2_arg = self._safe_div(5.0 * f31_rot + 3.0 * f20_rot, 8.0 * f20_rot)
        l1 = l1_arg.clamp(min=self.eps, max=1.0).sqrt()
        l2 = l2_arg.clamp(min=self.eps, max=1.0).sqrt()
        l = ((l1 + l2) * 0.5).clamp(min=0.0, max=1.0 - self.eps)

        denom = 2.0 * (1.0 - l * l).clamp(min=self.eps).pow(1.5)
        k = (3.0 * f11_rot.abs()) / (denom + self.eps)
        lt = l.flatten(2).mean(dim=2).view(-1, 1, 1, 1)
        kt = k.flatten(2).mean(dim=2).view(-1, 1, 1, 1)
        franklin_edge = ((kt < k) & (lt < l) & ((l1 - l2).abs() < lt)).to(x.dtype)
        franklin_edge = franklin_edge * prewitt_edge
        k_norm = k / (k.flatten(1).amax(dim=1).view(-1, 1, 1, 1) + self.eps)

        edge_features = torch.cat([grad_norm, franklin_edge, l, k_norm], dim=1)
        edge_features = torch.nan_to_num(edge_features, nan=0.0, posinf=1.0, neginf=0.0).detach()
        gate = torch.sigmoid(self.edge_proj(edge_features))
        return x * (1.0 + self.alpha * gate)


class SmallLesionCrossScaleEnhance(nn.Module):
    """Lightweight cross-scale enhancement for high-resolution disease spots."""

    def __init__(self, channels, alpha=0.15, act='silu'):
        super().__init__()
        self.alpha = alpha
        self.dw3 = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
        self.dw5 = ConvNormLayer_fuse(channels, channels, 5, 1, g=channels, act=act)
        self.dw7 = ConvNormLayer_fuse(channels, channels, 7, 1, g=channels, act=act)
        self.mix = ConvNormLayer_fuse(channels * 3, channels, 1, 1, act=act)
        self.gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        context = self.mix(torch.cat([self.dw3(x), self.dw5(x), self.dw7(x)], dim=1))
        gate = torch.sigmoid(self.gate(context))
        return x + self.alpha * context * gate


class EfficientChannelSpatialAttention(nn.Module):
    """Lightweight channel-spatial attention for weak texture and background suppression."""

    def __init__(self, channels, alpha=0.15, spatial_kernel_size=7):
        super().__init__()
        self.alpha = alpha
        padding = (spatial_kernel_size - 1) // 2
        self.channel_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel_size, padding=padding, bias=False)

    def forward(self, x):
        channel_context = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(1, 2)
        channel_gate = torch.sigmoid(self.channel_conv(channel_context).transpose(1, 2).unsqueeze(-1))

        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max = torch.amax(x, dim=1, keepdim=True)
        spatial_gate = torch.sigmoid(self.spatial_conv(torch.cat([spatial_avg, spatial_max], dim=1)))

        return x + self.alpha * x * channel_gate * spatial_gate


class BackgroundSuppressionGate(nn.Module):
    """Foreground gate that weakens clutter influence through residual reweighting."""

    def __init__(self, channels, alpha=0.10, act='silu'):
        super().__init__()
        self.alpha = alpha
        self.local = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
        self.gate = nn.Conv2d(channels + 2, 1, kernel_size=1, bias=True)

    def forward(self, x):
        local = self.local(x)
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        fg_gate = torch.sigmoid(self.gate(torch.cat([local, avg_map, max_map], dim=1)))
        return x * (1.0 + self.alpha * fg_gate)


class ResidualContextRefine(nn.Module):
    """Residual multi-receptive-field refinement for each output scale."""

    def __init__(self, channels, alpha=0.10, act='silu'):
        super().__init__()
        self.alpha = alpha
        self.local = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
        self.context = ConvNormLayer_fuse(channels, channels, 9, 1, g=channels, act=act)
        self.mix = ConvNormLayer_fuse(channels * 2, channels, 1, 1, act=act)
        self.gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        refined = self.mix(torch.cat([self.local(x), self.context(x)], dim=1))
        gate = torch.sigmoid(self.gate(refined))
        return x + self.alpha * refined * gate


class LearnableScaleFusion(nn.Module):
    """Learnable two-input fusion weights for top-down FPN and bottom-up PAN."""

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32))

    def forward(self, x0, x1):
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        return x0 * weights[0], x1 * weights[1]


class CARAFEUpsample(nn.Module):
    """Content-Aware ReAssembly of FEatures (CARAFE).

    This follows the CARAFE operator: channel compression, content encoder,
    kernel normalization with pixel shuffle, and content-aware reassembly.
    The implementation is pure PyTorch so it does not depend on MMCV CUDA ops.
    """

    def __init__(self, channels, scale_factor=2, compressed_channels=64,
                 encoder_kernel=3, up_kernel=5, up_group=1, act='silu'):
        super().__init__()
        if up_group != 1:
            raise ValueError("Pure PyTorch CARAFE currently uses up_group=1.")
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.up_group = up_group
        self.channel_compressor = ConvNormLayer(
            channels, compressed_channels, 1, 1, padding=0, act=act
        )
        self.content_encoder = nn.Conv2d(
            compressed_channels,
            (up_kernel * up_kernel) * up_group * scale_factor * scale_factor,
            kernel_size=encoder_kernel,
            padding=encoder_kernel // 2,
            bias=True,
        )

    def forward(self, x, output_size=None):
        b, c, h, w = x.shape
        sf = self.scale_factor
        masks = self.content_encoder(self.channel_compressor(x))
        masks = F.pixel_shuffle(masks, sf)
        h_up, w_up = h * sf, w * sf

        if output_size is not None and (h_up, w_up) != tuple(output_size):
            masks = F.interpolate(masks, size=output_size, mode='bilinear', align_corners=False)
            h_up, w_up = output_size

        masks = masks.view(b, self.up_group, self.up_kernel * self.up_kernel, h_up, w_up)
        masks = F.softmax(masks, dim=2)

        x_up = F.interpolate(x, size=(h_up, w_up), mode='nearest')
        patches = F.unfold(
            x_up,
            kernel_size=self.up_kernel,
            padding=self.up_kernel // 2,
        ).view(b, c, self.up_kernel * self.up_kernel, h_up, w_up)
        return (patches * masks[:, 0:1]).sum(dim=2)


class BiFPNFusionBlock(nn.Module):
    """EfficientDet BiFPN normalized weighted fusion plus separable conv."""

    def __init__(self, channels, num_inputs=2, eps=1e-4, act='silu'):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.dwconv = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=None)
        self.pwconv = ConvNormLayer_fuse(channels, channels, 1, 1, act=act)

    def forward(self, *inputs):
        if len(inputs) != self.weights.numel():
            raise ValueError(f"BiFPNFusionBlock expects {self.weights.numel()} inputs, got {len(inputs)}.")
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        fused = sum(x * weights[i] for i, x in enumerate(inputs))
        return self.pwconv(self.dwconv(fused))


class ECALayer(nn.Module):
    """ECA-Net channel attention with adaptive 1D kernel size."""

    def __init__(self, channels, gamma=2, b=1, alpha=0.10):
        super().__init__()
        kernel_size = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        kernel_size = max(3, kernel_size)
        self.alpha = alpha
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(1, 2)
        y = torch.sigmoid(self.conv(y).transpose(1, 2).unsqueeze(-1))
        return x + self.alpha * x * y


class CBAMLayer(nn.Module):
    """CBAM channel and spatial attention."""

    def __init__(self, channels, reduction=16, spatial_kernel=7, alpha=0.10):
        super().__init__()
        self.alpha = alpha
        mid_channels = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(
            2, 1, spatial_kernel, padding=spatial_kernel // 2, bias=False
        )

    def forward(self, x):
        avg_attn = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_attn = self.mlp(F.adaptive_max_pool2d(x, 1))
        channel_gate = torch.sigmoid(avg_attn + max_attn)
        y = x * channel_gate
        avg_out = torch.mean(y, dim=1, keepdim=True)
        max_out = torch.amax(y, dim=1, keepdim=True)
        spatial_gate = torch.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        y = y * spatial_gate
        return x + self.alpha * y


class CoordinateAttentionExact(nn.Module):
    """Coordinate Attention with separate height and width encodings."""

    def __init__(self, channels, reduction=32, alpha=0.12):
        super().__init__()
        self.alpha = alpha
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

    @staticmethod
    def _h_sigmoid(x):
        return F.relu6(x + 3.0, inplace=True) / 6.0

    def _h_swish(self, x):
        return x * self._h_sigmoid(x)

    def forward(self, x):
        identity = x
        _, _, h, w = x.size()
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self._h_swish(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return identity + self.alpha * identity * a_h * a_w


class LSKBlock(nn.Module):
    """Large Selective Kernel block from LSKNet."""

    def __init__(self, channels, alpha=0.10):
        super().__init__()
        self.alpha = alpha
        self.conv0 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.conv_spatial = nn.Conv2d(channels, channels, 7, stride=1,
                                      padding=9, groups=channels, dilation=3)
        hidden = max(1, channels // 2)
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.conv2 = nn.Conv2d(channels, hidden, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn = torch.amax(attn, dim=1, keepdim=True)
        sig = torch.sigmoid(self.conv_squeeze(torch.cat([avg_attn, max_attn], dim=1)))
        attn = attn1 * sig[:, 0:1] + attn2 * sig[:, 1:2]
        attn = self.conv(attn)
        return x + self.alpha * x * attn


class PartialConv2d(nn.Module):
    """FasterNet partial convolution on a channel subset."""

    def __init__(self, channels, n_div=4):
        super().__init__()
        self.dim_conv = channels // n_div
        self.dim_untouched = channels - self.dim_conv
        self.partial_conv3 = nn.Conv2d(
            self.dim_conv, self.dim_conv, 3, 1, 1, bias=False
        )

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        return torch.cat([self.partial_conv3(x1), x2], dim=1)


class FasterNetBlock(nn.Module):
    """FasterNet block with partial convolution and pointwise MLP."""

    def __init__(self, channels, mlp_ratio=2.0, n_div=4, alpha=0.10, act='silu'):
        super().__init__()
        self.alpha = alpha
        hidden = int(channels * mlp_ratio)
        self.spatial_mixing = PartialConv2d(channels, n_div=n_div)
        self.mlp = nn.Sequential(
            ConvNormLayer_fuse(channels, hidden, 1, 1, act=act),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.alpha * self.mlp(self.spatial_mixing(x))


class RepVGGEnhanceBlock(nn.Module):
    """RepVGG training-time 3x3 + 1x1 + identity block."""

    def __init__(self, channels, alpha=0.10, act='silu'):
        super().__init__()
        self.alpha = alpha
        self.branch_3x3 = ConvNormLayer(channels, channels, 3, 1, padding=1, act=None)
        self.branch_1x1 = ConvNormLayer(channels, channels, 1, 1, padding=0, act=None)
        self.branch_identity = nn.BatchNorm2d(channels)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        y = self.branch_3x3(x) + self.branch_1x1(x) + self.branch_identity(x)
        return x + self.alpha * self.act(y)


class GatherDistributeContext(nn.Module):
    """Multi-scale gather-distribute context injection for neck outputs.

    The module gathers all neck outputs at each target scale, fuses them with a
    1x1 projection, and injects the distributed context residually.
    """

    def __init__(self, channels, num_levels, alpha=0.10, act='silu'):
        super().__init__()
        self.alpha = alpha
        self.fuse = nn.ModuleList([
            ConvNormLayer_fuse(channels * num_levels, channels, 1, 1, act=act)
            for _ in range(num_levels)
        ])

    def forward(self, feats):
        outs = []
        for i, feat in enumerate(feats):
            size = feat.shape[-2:]
            aligned = [
                src if src.shape[-2:] == size
                else F.interpolate(src, size=size, mode='bilinear', align_corners=False)
                for src in feats
            ]
            context = self.fuse[i](torch.cat(aligned, dim=1))
            outs.append(feat + self.alpha * context)
        return outs


class CoordinateAttentionEnhance(nn.Module):
    """Coordinate attention keeps row/column position cues for leaf and boll structure."""

    def __init__(self, channels, reduction=32, alpha=0.12, act='silu'):
        super().__init__()
        self.alpha = alpha
        mid_channels = max(8, channels // reduction)
        self.reduce = ConvNormLayer(channels, mid_channels, 1, 1, padding=0, act=act)
        self.conv_h = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        _, _, h, w = x.shape
        context_h = x.mean(dim=3, keepdim=True)
        context_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        context = self.reduce(torch.cat([context_h, context_w], dim=2))
        context_h, context_w = torch.split(context, [h, w], dim=2)
        gate_h = torch.sigmoid(self.conv_h(context_h))
        gate_w = torch.sigmoid(self.conv_w(context_w.transpose(2, 3)))
        return x + self.alpha * x * gate_h * gate_w


class HighFrequencyResidualEnhance(nn.Module):
    """High-frequency residual enhancement for tiny leaf spots and pest edges."""

    def __init__(self, channels, alpha=0.10, act='silu'):
        super().__init__()
        self.alpha = alpha
        self.proj = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
        self.gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low
        enhanced = self.proj(high)
        gate = torch.sigmoid(self.gate(high))
        return x + self.alpha * enhanced * gate


class GlobalContextCalibration(nn.Module):
    """SE-style global context calibration for field-background robustness."""

    def __init__(self, channels, reduction=16, alpha=0.10, act='silu'):
        super().__init__()
        self.alpha = alpha
        mid_channels = max(8, channels // reduction)
        self.fc1 = ConvNormLayer(channels, mid_channels, 1, 1, padding=0, act=act)
        self.fc2 = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        context = F.adaptive_avg_pool2d(x, 1)
        gate = torch.sigmoid(self.fc2(self.fc1(context)))
        return x + self.alpha * x * gate


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='dfine',
                 distill_teacher_dim=0,
                 ifpn_upsample_mode='nearest',
                 edge_enhance=False,
                 edge_enhance_alpha=0.15,
                 edge_threshold_ratio=0.2,
                 prewitt_franklin_enhance=False,
                 prewitt_franklin_alpha=0.15,
                 small_lesion_enhance=False,
                 small_lesion_alpha=0.15,
                 attention_enhance=False,
                 attention_alpha=0.15,
                 background_suppression=False,
                 background_alpha=0.10,
                 output_refine=False,
                 output_refine_alpha=0.10,
                 weighted_fusion=False,
                 coordinate_attention=False,
                 coordinate_attention_alpha=0.12,
                 high_frequency_enhance=False,
                 high_frequency_alpha=0.10,
                 global_context=False,
                 global_context_alpha=0.10,
                 carafe_upsample=False,
                 carafe_compressed_channels=64,
                 carafe_kernel_size=5,
                 bifpn_fusion=False,
                 eca_attention=False,
                 eca_alpha=0.10,
                 cbam_attention=False,
                 cbam_alpha=0.10,
                 coord_attention_exact=False,
                 lsk_attention=False,
                 lsk_alpha=0.10,
                 fasternet_pconv=False,
                 fasternet_alpha=0.10,
                 repvgg_enhance=False,
                 repvgg_alpha=0.10,
                 gather_distribute=False,
                 gather_distribute_alpha=0.10,
                 safe_module_init_scale=1e-3,
                 safe_module_trainable_scale=False,
                 safe_module_max_scale=0.02,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.distill_teacher_dim = distill_teacher_dim
        self.ifpn_upsample_mode = ifpn_upsample_mode
        self.carafe_upsample = carafe_upsample
        self.bifpn_fusion = bifpn_fusion
        self.safe_module_init_scale = safe_module_init_scale
        self.safe_module_trainable_scale = safe_module_trainable_scale
        self.safe_module_max_scale = safe_module_max_scale

        assert len(use_encoder_idx) > 0, "use_encoder_idx must specify at least one encoder output"
        # target AIFI output F5 for distillation
        self.encoder_idx_for_distillation = use_encoder_idx[-1]

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))

            self.input_proj.append(proj)

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act
            )

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # feature_projector
        self.feature_projector = None
        if self.distill_teacher_dim > 0:
            self.feature_projector = nn.Sequential(
                    nn.Linear(hidden_dim, self.distill_teacher_dim),
                    # nn.GELU(),
                )

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            # TODO, add activation for those lateral convs
            if version == 'dfine':
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
            else:
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
                if version == 'dfine' else CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2, act=act)) \
                if version == 'dfine' else ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
                if version == 'dfine' else CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )

        self.top_down_fusion_weights = nn.ModuleList([
            LearnableScaleFusion() for _ in range(len(in_channels) - 1)
        ]) if weighted_fusion else None
        self.bottom_up_fusion_weights = nn.ModuleList([
            LearnableScaleFusion() for _ in range(len(in_channels) - 1)
        ]) if weighted_fusion else None
        self.carafe_upsamplers = nn.ModuleList([
            CARAFEUpsample(
                hidden_dim,
                scale_factor=2,
                compressed_channels=carafe_compressed_channels,
                up_kernel=carafe_kernel_size,
                act=act,
            )
            for _ in range(len(in_channels) - 1)
        ]) if carafe_upsample else None
        self.bifpn_top_down = nn.ModuleList([
            BiFPNFusionBlock(hidden_dim, num_inputs=2, act=act)
            for _ in range(len(in_channels) - 1)
        ]) if bifpn_fusion else None
        self.bifpn_bottom_up = nn.ModuleList([
            BiFPNFusionBlock(hidden_dim, num_inputs=2, act=act)
            for _ in range(len(in_channels) - 1)
        ]) if bifpn_fusion else None
        self.edge_enhancer = PrewittEdgeGuidedEnhance(
            hidden_dim, alpha=edge_enhance_alpha, threshold_ratio=edge_threshold_ratio
        ) if edge_enhance else None
        self.prewitt_franklin_enhancer = PrewittFranklinEdgeGuidedEnhance(
            hidden_dim, alpha=prewitt_franklin_alpha, threshold_ratio=edge_threshold_ratio
        ) if prewitt_franklin_enhance else None
        self.small_lesion_enhancer = SmallLesionCrossScaleEnhance(
            hidden_dim, alpha=small_lesion_alpha, act=act
        ) if small_lesion_enhance else None
        self.attention_enhancer = EfficientChannelSpatialAttention(
            hidden_dim, alpha=attention_alpha
        ) if attention_enhance else None
        self.background_suppressor = BackgroundSuppressionGate(
            hidden_dim, alpha=background_alpha, act=act
        ) if background_suppression else None
        self.output_refines = nn.ModuleList([
            ResidualContextRefine(hidden_dim, alpha=output_refine_alpha, act=act)
            for _ in range(len(in_channels))
        ]) if output_refine else None
        self.high_frequency_enhancer = HighFrequencyResidualEnhance(
            hidden_dim, alpha=high_frequency_alpha, act=act
        ) if high_frequency_enhance else None
        self.coordinate_attentions = nn.ModuleList([
            CoordinateAttentionEnhance(hidden_dim, alpha=coordinate_attention_alpha, act=act)
            for _ in range(len(in_channels))
        ]) if coordinate_attention else None
        self.coordinate_attentions_exact = nn.ModuleList([
            CoordinateAttentionExact(hidden_dim, alpha=coordinate_attention_alpha)
            for _ in range(len(in_channels))
        ]) if coord_attention_exact else None
        self.eca_attentions = nn.ModuleList([
            ECALayer(hidden_dim, alpha=eca_alpha)
            for _ in range(len(in_channels))
        ]) if eca_attention else None
        self.cbam_attentions = nn.ModuleList([
            CBAMLayer(hidden_dim, alpha=cbam_alpha)
            for _ in range(len(in_channels))
        ]) if cbam_attention else None
        self.lsk_attentions = nn.ModuleList([
            LSKBlock(hidden_dim, alpha=lsk_alpha)
            for _ in range(len(in_channels))
        ]) if lsk_attention else None
        self.fasternet_blocks = nn.ModuleList([
            FasterNetBlock(hidden_dim, alpha=fasternet_alpha, act=act)
            for _ in range(len(in_channels))
        ]) if fasternet_pconv else None
        self.repvgg_blocks = nn.ModuleList([
            RepVGGEnhanceBlock(hidden_dim, alpha=repvgg_alpha, act=act)
            for _ in range(len(in_channels))
        ]) if repvgg_enhance else None
        self.gather_distribute = GatherDistributeContext(
            hidden_dim, len(in_channels), alpha=gather_distribute_alpha, act=act
        ) if gather_distribute else None
        self.global_contexts = nn.ModuleList([
            GlobalContextCalibration(hidden_dim, alpha=global_context_alpha, act=act)
            for _ in range(len(in_channels))
        ]) if global_context else None

        self.carafe_scales = nn.ParameterList([
            self._make_safe_scale(safe_module_init_scale)
            for _ in range(len(in_channels) - 1)
        ]) if carafe_upsample else None
        self.bifpn_top_down_scales = nn.ParameterList([
            self._make_safe_scale(safe_module_init_scale)
            for _ in range(len(in_channels) - 1)
        ]) if bifpn_fusion else None
        self.bifpn_bottom_up_scales = nn.ParameterList([
            self._make_safe_scale(safe_module_init_scale)
            for _ in range(len(in_channels) - 1)
        ]) if bifpn_fusion else None
        self.edge_enhancer_scale = self._make_safe_scale(safe_module_init_scale) if edge_enhance else None
        self.prewitt_franklin_scale = self._make_safe_scale(safe_module_init_scale) if prewitt_franklin_enhance else None
        self.small_lesion_scale = self._make_safe_scale(safe_module_init_scale) if small_lesion_enhance else None
        self.high_frequency_scale = self._make_safe_scale(safe_module_init_scale) if high_frequency_enhance else None
        self.attention_scale = self._make_safe_scale(safe_module_init_scale) if attention_enhance else None
        self.background_scale = self._make_safe_scale(safe_module_init_scale) if background_suppression else None
        self.gather_distribute_scale = self._make_safe_scale(safe_module_init_scale) if gather_distribute else None
        self.output_refine_scales = nn.ParameterList([
            self._make_safe_scale(safe_module_init_scale)
            for _ in range(len(in_channels))
        ]) if output_refine else None
        self.coordinate_attention_scales = nn.ParameterList([
            self._make_safe_scale(safe_module_init_scale)
            for _ in range(len(in_channels))
        ]) if coordinate_attention else None
        self.coordinate_attention_exact_scales = nn.ParameterList([
            self._make_safe_scale(safe_module_init_scale)
            for _ in range(len(in_channels))
        ]) if coord_attention_exact else None
        self.eca_attention_scales = nn.ParameterList([
            self._make_safe_scale(safe_module_init_scale)
            for _ in range(len(in_channels))
        ]) if eca_attention else None
        self.cbam_attention_scales = nn.ParameterList([
            self._make_safe_scale(safe_module_init_scale)
            for _ in range(len(in_channels))
        ]) if cbam_attention else None
        self.lsk_attention_scales = nn.ParameterList([
            self._make_safe_scale(safe_module_init_scale)
            for _ in range(len(in_channels))
        ]) if lsk_attention else None
        self.fasternet_scales = nn.ParameterList([
            self._make_safe_scale(safe_module_init_scale)
            for _ in range(len(in_channels))
        ]) if fasternet_pconv else None
        self.repvgg_scales = nn.ParameterList([
            self._make_safe_scale(safe_module_init_scale)
            for _ in range(len(in_channels))
        ]) if repvgg_enhance else None
        self.global_context_scales = nn.ParameterList([
            self._make_safe_scale(safe_module_init_scale)
            for _ in range(len(in_channels))
        ]) if global_context else None

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def _upsample_like(self, x, ref):
        if self.ifpn_upsample_mode == 'nearest':
            return F.interpolate(x, size=ref.shape[-2:], mode='nearest')
        return F.interpolate(x, size=ref.shape[-2:], mode=self.ifpn_upsample_mode, align_corners=False)

    def _make_safe_scale(self, init_value):
        return nn.Parameter(
            torch.tensor(float(init_value), dtype=torch.float32),
            requires_grad=bool(self.safe_module_trainable_scale),
        )

    def _scale_like(self, scale, ref):
        value = scale.to(device=ref.device, dtype=ref.dtype)
        if self.safe_module_max_scale is not None:
            max_scale = float(self.safe_module_max_scale)
            value = value.clamp(min=-max_scale, max=max_scale)
        return value

    def _safe_residual(self, module, x, scale):
        enhanced = module(x)
        return x + self._scale_like(scale, x) * (enhanced - x)

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        distill_student_output = None

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory :torch.Tensor = self.encoder[i](src_flatten, src_mask=None, pos_embed=pos_embed)

                # Reshape back to [B, C, H, W] for subsequent FPN/PAN layers
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

                # Apply feature projector to F5
                if self.training and self.feature_projector is not None and enc_ind == self.encoder_idx_for_distillation:
                    # _logger.info(f"[HybridEncoder] Feature size: {h}x{w}")
                    distill_student_output = self.feature_projector(proj_feats[enc_ind].permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # [B, distill_teacher_dim, H, W]


        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            fusion_idx = len(self.in_channels) - 1 - idx
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[fusion_idx](feat_heigh)
            inner_outs[0] = feat_heigh
            base_upsample_feat = self._upsample_like(feat_heigh, feat_low)
            if self.carafe_upsamplers is not None:
                carafe_feat = self.carafe_upsamplers[fusion_idx](feat_heigh, feat_low.shape[-2:])
                upsample_feat = base_upsample_feat + self._scale_like(
                    self.carafe_scales[fusion_idx], base_upsample_feat
                ) * (carafe_feat - base_upsample_feat)
            else:
                upsample_feat = base_upsample_feat
            if self.top_down_fusion_weights is not None:
                upsample_feat, feat_low = self.top_down_fusion_weights[fusion_idx](
                    upsample_feat, feat_low
                )
            fpn_out = self.fpn_blocks[fusion_idx](torch.concat([upsample_feat, feat_low], dim=1))
            if self.bifpn_top_down is not None:
                bifpn_out = self.bifpn_top_down[fusion_idx](upsample_feat, feat_low)
                inner_out = fpn_out + self._scale_like(
                    self.bifpn_top_down_scales[fusion_idx], fpn_out
                ) * (bifpn_out - fpn_out)
            else:
                inner_out = fpn_out
            inner_outs.insert(0, inner_out)

        if self.edge_enhancer is not None:
            inner_outs[0] = self._safe_residual(self.edge_enhancer, inner_outs[0], self.edge_enhancer_scale)
        if self.prewitt_franklin_enhancer is not None:
            inner_outs[0] = self._safe_residual(
                self.prewitt_franklin_enhancer, inner_outs[0], self.prewitt_franklin_scale
            )
        if self.small_lesion_enhancer is not None:
            inner_outs[0] = self._safe_residual(self.small_lesion_enhancer, inner_outs[0], self.small_lesion_scale)
        if self.high_frequency_enhancer is not None:
            inner_outs[0] = self._safe_residual(
                self.high_frequency_enhancer, inner_outs[0], self.high_frequency_scale
            )
        if self.attention_enhancer is not None:
            inner_outs[0] = self._safe_residual(self.attention_enhancer, inner_outs[0], self.attention_scale)
        if self.background_suppressor is not None:
            inner_outs[0] = self._safe_residual(self.background_suppressor, inner_outs[0], self.background_scale)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            if self.bottom_up_fusion_weights is not None:
                downsample_feat, feat_height = self.bottom_up_fusion_weights[idx](
                    downsample_feat, feat_height
                )
            pan_out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            if self.bifpn_bottom_up is not None:
                bifpn_out = self.bifpn_bottom_up[idx](downsample_feat, feat_height)
                out = pan_out + self._scale_like(
                    self.bifpn_bottom_up_scales[idx], pan_out
                ) * (bifpn_out - pan_out)
            else:
                out = pan_out
            outs.append(out)

        if self.gather_distribute is not None:
            gathered = self.gather_distribute(outs)
            outs = [
                out + self._scale_like(self.gather_distribute_scale, out) * (enhanced - out)
                for out, enhanced in zip(outs, gathered)
            ]
        if self.output_refines is not None:
            outs = [
                out + self._scale_like(scale, out) * (refine(out) - out)
                for refine, scale, out in zip(self.output_refines, self.output_refine_scales, outs)
            ]
        if self.coordinate_attentions is not None:
            outs = [
                out + self._scale_like(scale, out) * (attn(out) - out)
                for attn, scale, out in zip(self.coordinate_attentions, self.coordinate_attention_scales, outs)
            ]
        if self.coordinate_attentions_exact is not None:
            outs = [
                out + self._scale_like(scale, out) * (attn(out) - out)
                for attn, scale, out in zip(
                    self.coordinate_attentions_exact, self.coordinate_attention_exact_scales, outs
                )
            ]
        if self.eca_attentions is not None:
            outs = [
                out + self._scale_like(scale, out) * (attn(out) - out)
                for attn, scale, out in zip(self.eca_attentions, self.eca_attention_scales, outs)
            ]
        if self.cbam_attentions is not None:
            outs = [
                out + self._scale_like(scale, out) * (attn(out) - out)
                for attn, scale, out in zip(self.cbam_attentions, self.cbam_attention_scales, outs)
            ]
        if self.lsk_attentions is not None:
            outs = [
                out + self._scale_like(scale, out) * (attn(out) - out)
                for attn, scale, out in zip(self.lsk_attentions, self.lsk_attention_scales, outs)
            ]
        if self.fasternet_blocks is not None:
            outs = [
                out + self._scale_like(scale, out) * (block(out) - out)
                for block, scale, out in zip(self.fasternet_blocks, self.fasternet_scales, outs)
            ]
        if self.repvgg_blocks is not None:
            outs = [
                out + self._scale_like(scale, out) * (block(out) - out)
                for block, scale, out in zip(self.repvgg_blocks, self.repvgg_scales, outs)
            ]
        if self.global_contexts is not None:
            outs = [
                out + self._scale_like(scale, out) * (context(out) - out)
                for context, scale, out in zip(self.global_contexts, self.global_context_scales, outs)
            ]

        if self.training and distill_student_output is not None:
            return outs, distill_student_output
        return outs
