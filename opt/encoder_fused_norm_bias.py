"""Inference-only Conv3d bias + temporal GN + SiLU + padding fusion.

The convolution itself remains ATen/cuDNN. Its bias is added in registers at
both statistical and output reads, including the original FP16 rounding.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from encoder_fused_temporal_norm import _merge
from encoder_fused_norm_opaque import OpaqueFusedNormPadConv


@triton.jit
def _partial_bias(X, PreBias, P, Q, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                  SD: tl.constexpr, SC: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
                  G: tl.constexpr, NP: tl.constexpr, BS: tl.constexpr, BC: tl.constexpr):
    part = tl.program_id(0); cb = tl.program_id(1); t = tl.program_id(2)
    s = part * BS + tl.arange(0, BS)
    c = cb * BC + tl.arange(0, BC)
    cg: tl.constexpr = C // G
    bg: tl.constexpr = BC // cg
    off = t*SD + (s[:, None]//W)*SH + (s[:, None]%W)*SW + c[None, :]*SC
    valid = (s[:, None] < H*W) & (c[None, :] < C)
    v = tl.load(X + off, valid, other=0).to(tl.float32)
    pre_bias = tl.load(PreBias + c, c < C, other=0).to(tl.float32)
    v = (v + pre_bias[None, :]).to(tl.float16).to(tl.float32)
    v = tl.reshape(tl.where(valid, v, 0.), (BS, bg, cg))
    count = tl.minimum(BS, H*W-part*BS) * cg
    avg = tl.div_rn(tl.sum(tl.sum(v, 2), 0), count.to(tl.float32))
    centered = v - avg[None, :, None]
    mask = tl.reshape(valid, (BS, bg, cg))
    m2 = tl.sum(tl.sum(tl.where(mask, centered*centered, 0.), 2), 0)
    g = cb*bg + tl.arange(0, bg)
    dest = (t*NP + part)*G + g
    tl.store(P + dest, avg, g < G); tl.store(Q + dest, m2, g < G)


@triton.jit
def _normalize_pack_bias(X, PreBias, Weight, Bias, S, Y,
                         C: tl.constexpr, D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                         SD: tl.constexpr, SC: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
                         G: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    total: tl.constexpr = C*(D+2)*(H+2)*(W+2)
    c = idx % C; q = idx // C
    t = q // ((H+2)*(W+2)) - 2
    h = q // (W+2) % (H+2) - 1; w = q % (W+2) - 1
    h = tl.where(h < 0, -h, tl.where(h >= H, 2*H-2-h, h))
    w = tl.where(w < 0, -w, tl.where(w >= W, 2*W-2-w, w))
    valid = (idx < total) & (t >= 0) & (t < D)
    v = tl.load(X+t*SD+c*SC+h*SH+w*SW, valid, other=0).to(tl.float32)
    pre_bias = tl.load(PreBias+c).to(tl.float32)
    v = (v + pre_bias).to(tl.float16).to(tl.float32)
    stat = (t*G + c//(C//G))*2
    mean = tl.load(S+stat, valid, other=0)
    rstd = tl.load(S+stat+1, valid, other=0)
    gamma = tl.load(Weight+c).to(tl.float32); beta = tl.load(Bias+c).to(tl.float32)
    z = (((v-mean)*rstd)*gamma+beta).to(tl.float16).to(tl.float32)
    z = (z/(1.+tl.exp(-z))).to(tl.float16)
    tl.store(Y+idx, tl.where(valid, z, 0.), idx < total)


def fused_bias_temporal_norm_pad(x, pre_bias, weight, bias, eps):
    b, c, d, h, w = x.shape
    if b != 1 or x.dtype != torch.float16 or c not in (128, 256) or min(h, w) < 2:
        raise ValueError('Validated scope: batch-one FP16 C=128/256, spatial >=2')
    if c*(d+2)*(h+2)*(w+2) >= 2**31:
        raise ValueError('Output exceeds int32 index range')
    for parameter in (pre_bias, weight, bias):
        if parameter.shape != (c,) or parameter.dtype != x.dtype or parameter.device != x.device or not parameter.is_contiguous():
            raise ValueError('Expected contiguous FP16 channel vectors on input device')
    bs, bc, warps = 128, 64, 8
    nparts = triton.cdiv(h*w, bs)
    p = torch.empty((d, nparts, 32), device=x.device, dtype=torch.float32)
    q = torch.empty_like(p)
    stats = torch.empty((d, 32, 2), device=x.device, dtype=torch.float32)
    y = torch.empty((b, c, d+2, h+2, w+2), device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
    _partial_bias[(nparts, triton.cdiv(c, bc), d)](x, pre_bias, p, q, c, h, w, x.stride(2), x.stride(1), x.stride(3), x.stride(4), 32, nparts, bs, bc, num_warps=warps)
    _merge[(32, d)](p, q, stats, h, w, c, 32, nparts, bs, eps, triton.next_power_of_2(nparts), num_warps=4)
    _normalize_pack_bias[(triton.cdiv(y.numel(), 1024),)](x, pre_bias, weight, bias, stats, y, c, d, h, w, x.stride(2), x.stride(1), x.stride(3), x.stride(4), 32, 1024, num_warps=4, enable_fp_fusion=False)
    return y


@torch.library.custom_op('h3vae_encoder::bias_temporal_norm_pack', mutates_args=(), device_types='cuda')
def bias_temporal_norm_pack(x: torch.Tensor, pre_bias: torch.Tensor, weight: torch.Tensor,
                           bias: torch.Tensor, eps: float) -> torch.Tensor:
    return fused_bias_temporal_norm_pad(x, pre_bias, weight, bias, eps)


@bias_temporal_norm_pack.register_fake
def _fake(x, pre_bias, weight, bias, eps):
    b, c, d, h, w = x.shape
    return torch.empty((b, c, d+2, h+2, w+2), device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)


class BiasFusedResidualBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        if block.use_fused_norm or block.conv1.bias is None:
            raise ValueError('Expected original residual block with conv1 bias')
        self.first = OpaqueFusedNormPadConv(block.norm1, block.conv1, with_conv=False)
        self.second = OpaqueFusedNormPadConv(block.norm2, block.conv2, with_conv=False)
        self.shortcut = getattr(block, 'nin_shortcut', None)
        # Optional production INT8 wrappers are installed by
        # encoder_int8_integration.  None preserves the existing FP16 path.
        self.int8_first = None
        self.int8_second = None

    def forward(self, x, zq=None):
        if zq is not None:
            raise ValueError('Unconditional inference only')
        first_input = self.first(x)
        h = (self.int8_first(first_input) if self.int8_first is not None else
             F.conv3d(first_input, self.first.weight, None, stride=self.first.stride))
        h = bias_temporal_norm_pack(h, self.first.bias, self.second.norm_weight, self.second.norm_bias, self.second.eps)
        second_input = h
        h = (self.int8_second(second_input) if self.int8_second is not None else
             F.conv3d(second_input, self.second.weight, self.second.bias, stride=self.second.stride))
        return (self.shortcut(x) if self.shortcut is not None else x) + h
