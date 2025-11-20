import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

import numpy as np

from basicsr.utils.registry import ARCH_REGISTRY
from functools import partial

import math


import collections.abc
import torchvision
import warnings
from distutils.version import LooseVersion
from itertools import repeat
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from basicsr.utils import get_root_logger


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    ``Paper: Delving Deep into Deformable Alignment in Video Super-Resolution``
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# From PyTorch
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def make_coord(shape, ranges=None):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    ret = ret.flip(-1)
    return ret



class IGConv(nn.Module):
    def __init__(
            self, dim,
            # Own parameters
            kernel_size, implicit_dim: int = 256, latent_layers: int = 4,
            geo_ensemble: bool = True, max_s=4
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        assert implicit_dim % 2 == 0
        self.implicit_dim = implicit_dim
        self.latent_layers = latent_layers
        self.geo_ensemble = geo_ensemble
        self.max_s = max_s

        self.phase = nn.Conv2d(1, implicit_dim // 2, 1, 1)
        self.freq = nn.Parameter(
            torch.randn((dim * kernel_size ** 2), implicit_dim, 1, 1) * 0.02, requires_grad=True)
        self.amplitude = nn.Parameter(
            torch.randn((dim * kernel_size ** 2), implicit_dim, 1, 1) * 0.02, requires_grad=True)
        query_kernel_layers = []
        for _ in range(latent_layers):
            query_kernel_layers.append(
                nn.Conv2d(implicit_dim, implicit_dim, 1, 1, 0)
            )
            query_kernel_layers.append(nn.ReLU())

        query_kernel_layers.append(
            nn.Conv2d(implicit_dim, 3, 1, 1, 0)
        )
        self.query_kernel = nn.Sequential(*query_kernel_layers)
        self.resize = self._implicit_representation_latent

    def forward(self, x, scale, add_val=None):
        k_interp = self.resize(scale)
        if self.geo_ensemble:
            rgb = self._geo_ensemble(x, k_interp)
        else:
            rgb = F.conv2d(x, k_interp, bias=None, stride=1, padding=self.kernel_size // 2)
        if add_val is not None:
            rgb += add_val
        rgb = F.pixel_shuffle(rgb, scale)
        return rgb

    @staticmethod
    def _geo_ensemble(x, k_interp):
        k = k_interp
        k_hflip = k.flip([3])
        k_vflip = k.flip([2])
        k_hvflip = k.flip([2, 3])
        k_rot90 = torch.rot90(k, -1, [2, 3])
        k_rot90_hflip = k_rot90.flip([3])
        k_rot90_vflip = k_rot90.flip([2])
        k_rot90_hvflip = k_rot90.flip([2, 3])
        k = torch.cat([k, k_hflip, k_vflip, k_hvflip, k_rot90, k_rot90_hflip, k_rot90_vflip, k_rot90_hvflip], dim=0)
        ks = k.shape[-1]
        x = F.conv2d(x, k, bias=None, stride=1, padding=ks // 2)
        x = x.reshape(x.shape[0], 8, -1, x.shape[-2], x.shape[-1])
        x = x.mean(dim=1)
        return x

    def _implicit_representation_latent(self, scale):
        # print(scale, self.max_s, 'check ############################')
        scale_phase = min(scale, self.max_s)
        r = torch.ones(1, 1, scale, scale).to(
            self.query_kernel[0].weight.device) / scale_phase * 2  # 2 / r following LIIF/LTE
        coords = make_coord((scale, scale)).unsqueeze(0).to(self.query_kernel[0].weight.device)
        freq = self.freq.repeat(1, 1, scale, scale)  # RGB RGB
        amplitude = self.amplitude.repeat(1, 1, scale, scale)
        coords = coords.permute(0, 3, 1, 2).contiguous()

        # Fourier basis
        coords = coords.repeat(freq.shape[0], 1, 1, 1)
        freq_1, freq_2 = freq.chunk(2, dim=1)
        freq = freq_1 * coords[:, :1] + freq_2 * coords[:, 1:]  # RGB
        phase = self.phase(r)  # To RGB
        freq = freq + phase  # RGB
        freq = torch.cat([torch.cos(np.pi * freq), torch.sin(np.pi * freq)],
                         dim=1)  # cos(R)cos(G)cos(B) sin(R)sin(G)sin(B)

        # 4. R(F_theta(.))
        k_interp = self.query_kernel(freq * amplitude)
        k_interp = rearrange(
            k_interp, '(Cin Kh Kw) RGB rh rw -> (RGB rh rw) Cin Kh Kw', Kh=self.kernel_size, Kw=self.kernel_size,
            Cin=self.dim
        )
        return k_interp

    def extra_repr(self):
        return f'dim={self.dim}, kernel_size={self.kernel_size}' + \
            f', \nimplicit_dim={self.implicit_dim}, latent_layers={self.latent_layers}, geo_ensemble={self.geo_ensemble}'

    def instantiate(self, scale):
        k = self._implicit_representation_latent(scale)
        device = k.device
        kernel_size = k.shape[-1]
        c_in = k.shape[1]
        c_out = k.shape[0]

        if self.geo_ensemble:
            c = k.shape[0]
            k_hflip = k.flip([3])
            k_vflip = k.flip([2])
            k_hvflip = k.flip([2, 3])
            k_rot90 = torch.rot90(k, -1, [2, 3])
            k_rot90_hflip = k_rot90.flip([3])
            k_rot90_vflip = k_rot90.flip([2])
            k_rot90_hvflip = k_rot90.flip([2, 3])
            k = (k + k_hflip + k_vflip + k_hvflip + k_rot90 + k_rot90_hflip + k_rot90_vflip + k_rot90_hvflip) / 8.

        self.__class__ = InstantiatedIGConv
        self.__init__(c_in, c_out, kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.weight.data = k
        self.to(device)
        self.scale = scale


class InstantiatedIGConv(nn.Conv2d):
    def forward(self, x, scale=None, add_val=None):
        x = F.conv2d(x, self.weight, bias=None, stride=1, padding=self.padding)
        if add_val is not None:
            x += add_val
        x = F.pixel_shuffle(x, self.scale)
        return x


class IGConvDSSepRGB(nn.Module):
    def __init__(
            self, dim,
            # Own parameters
            kernel_size: int = 3,
            implicit_dim: int = 256,
            latent_layers: int = 4,
            geo_ensemble: bool = True,
            max_s=4,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.implicit_dim = implicit_dim
        self.latent_layers = latent_layers
        self.geo_ensemble = geo_ensemble
        self.max_s = max_s

        group = 2
        self.group = group
        implicit_dim = implicit_dim * 2
        self.phase = nn.Conv2d(1, implicit_dim // 2, 1, 1)
        self.freq = nn.Parameter(
            torch.randn((dim * kernel_size ** 2), implicit_dim, 1, 1) * 0.02, requires_grad=True)
        self.amplitude = nn.Parameter(
            torch.randn((dim * kernel_size ** 2), implicit_dim, 1, 1) * 0.02, requires_grad=True)
        query_kernel_layers = []
        for _ in range(latent_layers):
            query_kernel_layers.append(
                nn.Conv2d(implicit_dim, implicit_dim, 1, 1, 0, groups=group)
            )
            query_kernel_layers.append(nn.ReLU())
        query_kernel_layers.append(
            nn.Conv2d(implicit_dim, 12, 1, 1, 0, groups=group, bias=False)
        )
        self.query_kernel = nn.Sequential(*query_kernel_layers)
        for p in self.query_kernel.parameters():
            torch.nn.init.trunc_normal_(p, std=0.01)
        # to initialize RGB delta same
        with torch.no_grad():
            kernel_offset = torch.nn.init.trunc_normal_(torch.zeros(2, implicit_dim // 2, 1, 1), std=0.01)
            kernel_offset_rgb = kernel_offset.repeat(3, 1, 1, 1)
            kernel_scope = torch.nn.init.zeros_(torch.zeros(2, implicit_dim // 2, 1, 1))
            kernel_scope_rgb = kernel_scope.repeat(3, 1, 1, 1)
            kernel = torch.cat([kernel_offset_rgb, kernel_scope_rgb], dim=0)
            self.query_kernel[-1].weight.data = kernel

        self.resize = self._implicit_representation_latent

    def forward(self, x, scale):
        k_interp = self.resize(scale)
        if self.geo_ensemble:
            xyxy = self._geo_ensemble(x, k_interp)
        else:
            xyxy = F.conv2d(x, k_interp, bias=None, stride=1, padding=self.kernel_size // 2)
        xy_offset, xy_scope = xyxy.chunk(2, dim=1)
        return xy_offset, xy_scope

    @staticmethod
    def _geo_ensemble(x, k_interp):
        k = k_interp
        k_hflip = k.flip([3])
        k_vflip = k.flip([2])
        k_hvflip = k.flip([2, 3])
        k_rot90 = torch.rot90(k, -1, [2, 3])
        k_rot90_hflip = k_rot90.flip([3])
        k_rot90_vflip = k_rot90.flip([2])
        k_rot90_hvflip = k_rot90.flip([2, 3])
        k = (k + k_hflip + k_vflip + k_hvflip + k_rot90 + k_rot90_hflip + k_rot90_vflip + k_rot90_hvflip) / 8
        ks = k.shape[-1]
        x = F.conv2d(x, k, bias=None, stride=1, padding=ks // 2)
        return x

    def _implicit_representation_latent(self, scale):
        scale_phase = min(scale, self.max_s)
        r = torch.ones(1, 1, scale, scale).to(
            self.query_kernel[0].weight.device) / scale_phase * 2  # 2 / r following LIIF/LTE
        coords = make_coord((scale, scale)).unsqueeze(0).to(self.query_kernel[0].weight.device)
        freq = self.freq.repeat(1, 1, scale, scale)  # OffsetScopeOffsetScope
        amplitude = self.amplitude.repeat(1, 1, scale, scale)
        coords = coords.permute(0, 3, 1, 2).contiguous()

        coords = coords.repeat(freq.shape[0], 1, 1, 1)
        freq_1, freq_2 = freq.chunk(2, dim=1)
        freq = freq_1 * coords[:, :1] + freq_2 * coords[:, 1:]  # OffsetScope
        phase = self.phase(r)
        freq = freq + phase
        freq = torch.cat([torch.cos(np.pi * freq), torch.sin(np.pi * freq)],
                         dim=1)  # cos(Offset)cos(Scope) sin(Offset)sin(Scope)
        freq = rearrange(freq, 'b (g d) h w -> b (d g) h w', d=2)  # cos(Offset)sin(Offset) cos(Scope)sin(Scope)

        k_interp = self.query_kernel(freq * amplitude)
        k_interp = rearrange(
            k_interp, '(Cin Kh Kw) RGB rh rw -> (RGB rh rw) Cin Kh Kw', Kh=self.kernel_size, Kw=self.kernel_size,
            Cin=self.dim
        )
        return k_interp


class IGSample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    https://github.com/tiny-smart/dysample

    We implemented IGSample based on NeoSR's DySample implementation:
    https://github.com/muslll/neosr/blob/master/neosr/archs/arch_util.py
    """

    def __init__(self, dim, kernel_size=3, implicit_dim=128, latent_layers=2, geo_ensemble=True, max_s=4):
        super().__init__()

        self.convs = IGConvDSSepRGB(
            dim, kernel_size, implicit_dim=implicit_dim, latent_layers=latent_layers,
            geo_ensemble=geo_ensemble, max_s=max_s
        )

    def pos(self, scale):
        h = torch.arange((-scale + 1) / 2, (scale - 1) / 2 + 1) / scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, 3, 1)
            .reshape(1, -1, 1, 1)
        )

    def forward(self, x: torch.Tensor, feat: torch.Tensor, scale: int):
        offset, scope = self.convs(feat, scale)
        pos = self.pos(scale).to(x.device)

        offset = offset * scope.sigmoid() * 0.5 + pos

        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device, non_blocking=True)
        )
        normalizer = torch.tensor(
            [W, H], dtype=x.dtype, device=x.device, pin_memory=True
        ).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), scale)
            .view(B, 2, -1, scale * H, scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(B * 3, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, scale * H, scale * W)

        return output

    def extra_repr(self) -> str:
        return f'dim={self.convs.dim}, kernel_size={self.convs.kernel_size}' + \
            f', \nimplicit_dim={self.convs.implicit_dim}, latent_layers={self.convs.latent_layers}, geo_ensemble={self.convs.geo_ensemble}'

def test_direct_metrics(model, input_shape, scale, n_repeat=100, use_float16=True):
    from torch.backends import cudnn
    import tqdm
    import numpy as np
    from contextlib import nullcontext

    cudnn.benchmark = True

    print(f'CUDNN Benchmark: {cudnn.benchmark}')
    if use_float16:
        context = torch.cuda.amp.autocast
        print('Using AMP(FP16) for testing ...')
    else:
        context = nullcontext
        print('Using FP32 for testing ...')

    x = torch.FloatTensor(*input_shape).uniform_(0., 1.)
    x = x.cuda()
    print(f'Input shape: {x.shape}')
    model = model.cuda()
    model.eval()

    with context():
        with torch.inference_mode():
            print('warmup ...')
            for _ in tqdm.tqdm(range(100)):
                model(x, scale)  # Make sure CUDNN to find proper algorithms, especially for convolutions.
                torch.cuda.synchronize()

            print('testing ...')
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            timings = np.zeros((n_repeat, 1))

            for rep in tqdm.tqdm(range(n_repeat)):
                starter.record()
                model(x, scale)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

    avg = np.sum(timings) / n_repeat
    med = np.median(timings)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('------------ Results ------------')
    print(f'Average time: {avg:.5f} ms')
    print(f'Median time: {med:.5f} ms')
    print(f'Maximum GPU memory Occupancy: {torch.cuda.max_memory_allocated() / 1024 ** 2:.5f} MB')
    print(f'Params: {params / 1000}K')  # For convenience and sanity check.
    print('---------------------------------')
