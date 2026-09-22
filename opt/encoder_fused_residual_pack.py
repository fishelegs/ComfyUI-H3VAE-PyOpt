"""Stage1 residual/bias/right-bottom-reflect/causal-zero packing.

Matches inspected compiled control: FP32 adds followed by one FP16 store.
This is not the same as forcing eager FP16 rounding after each addition.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from encoder_fused_norm_bias import BiasFusedResidualBlock, bias_temporal_norm_pack

@triton.jit
def _residual_pack(X, R, Bias, Y, C: tl.constexpr, D: tl.constexpr,
                   H: tl.constexpr, W: tl.constexpr,
                   XD: tl.constexpr, XC: tl.constexpr, XH: tl.constexpr, XW: tl.constexpr,
                   RD: tl.constexpr, RC: tl.constexpr, RH: tl.constexpr, RW: tl.constexpr,
                   BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    total: tl.constexpr = C*(D+2)*(H+1)*(W+1)
    c = i % C; q = i // C
    t = q // ((H+1)*(W+1)) - 2
    h = q // (W+1) % (H+1); w = q % (W+1)
    h = tl.where(h == H, H-2, h); w = tl.where(w == W, W-2, w)
    valid = (i < total) & (t >= 0) & (t < D)
    x = tl.load(X+t*XD+c*XC+h*XH+w*XW, valid, other=0).to(tl.float32)
    r = tl.load(R+t*RD+c*RC+h*RH+w*RW, valid, other=0).to(tl.float32)
    bias = tl.load(Bias+c).to(tl.float32)
    y = r + (x + bias)
    tl.store(Y+i, tl.where(valid, y, 0.).to(tl.float16), i < total)

@torch.library.custom_op('h3vae_encoder::residual_downsample_pack', mutates_args=(), device_types='cuda')
def residual_downsample_pack(x: torch.Tensor, residual: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    b, c, d, h, w = x.shape
    if b != 1 or c != 256 or min(h, w) < 2 or x.dtype != torch.float16:
        raise ValueError('Validated scope: batch-one FP16 C=256 H/W>=2')
    if residual.shape != x.shape or residual.dtype != x.dtype or residual.device != x.device:
        raise ValueError('Residual must match input')
    if bias.shape != (c,) or bias.dtype != x.dtype or bias.device != x.device or not bias.is_contiguous():
        raise ValueError('Expected contiguous FP16 channel bias')
    if c*(d+2)*(h+1)*(w+1) >= 2**31:
        raise ValueError('Output exceeds validated int32 index range')
    y = torch.empty((b, c, d+2, h+1, w+1), device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
    _residual_pack[(triton.cdiv(y.numel(), 1024),)](x, residual, bias, y, c, d, h, w,
        x.stride(2), x.stride(1), x.stride(3), x.stride(4),
        residual.stride(2), residual.stride(1), residual.stride(3), residual.stride(4),
        1024, num_warps=4, enable_fp_fusion=False)
    return y

@residual_downsample_pack.register_fake
def _fake(x, residual, bias):
    b, c, d, h, w = x.shape
    return torch.empty((b, c, d+2, h+1, w+1), device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)

class ResidualDownsampleBlock(BiasFusedResidualBlock):
    def __init__(self, block, downsample):
        super().__init__(block)
        conv = downsample.conv
        if self.shortcut is not None or conv.padding != (1, 0, 0) or conv.stride != (2, 2, 2) or conv.pad_mode != 'reflect' or conv.pad_mode_t != 'constant' or not conv.causal or conv.spatial_parallel:
            raise ValueError('Expected stage1 final identity-shortcut block and causal reflect downsample')
        self.down_weight = conv.weight; self.down_bias = conv.bias

    def forward(self, x, zq=None):
        if zq is not None: raise ValueError('Unconditional inference only')
        first_input = self.first(x)
        h = (self.int8_first(first_input) if self.int8_first is not None else
             F.conv3d(first_input, self.first.weight, None, stride=self.first.stride))
        h = bias_temporal_norm_pack(h, self.first.bias, self.second.norm_weight, self.second.norm_bias, self.second.eps)
        second_input = h
        h = (self.int8_second(second_input) if self.int8_second is not None else
             F.conv3d(second_input, self.second.weight, None, stride=self.second.stride))
        h = residual_downsample_pack(h, x, self.second.bias)
        return F.conv3d(h, self.down_weight, self.down_bias, stride=(2, 2, 2))
