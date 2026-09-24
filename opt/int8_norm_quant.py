"""Opt-in residual + FP32 RMS statistics + INT8 row quantization.

CUDA FP16 inference only; version-pinned CK 0.2.34 GEMM and
quantized weights. Changed fusion boundaries require end-to-end quality tests.
"""
from functools import lru_cache

import torch
from torch import nn
from .int8_swiglu_fused import (
    clone_fused_decoder, prequantized_linear, swiglu_int8_linear,
)


@lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def kernel(X, A, G, W, H, Q, S, EPS: tl.constexpr, K: tl.constexpr,
               ROUND_STATS: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, K)
        x = tl.load(X + row*K + col).to(tl.float32)
        a = tl.load(A + row*K + col).to(tl.float32)
        g = tl.load(G + col).to(tl.float32)
        w = tl.load(W + col).to(tl.float32)
        raw = tl.fma(a, g, x)
        h = raw.to(tl.float16).to(tl.float32)
        stat = h if ROUND_STATS else raw
        inv = libdevice.rsqrt(tl.sum(stat*stat, 0) * (1.0/K) + EPS)
        act = (h * inv * w).to(tl.float16).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(act), 0) * (1.0/127.0), 1.e-30)
        v = tl.div_rn(act, scale.to(tl.float16).to(tl.float32)).to(tl.float16).to(tl.float32)
        v = libdevice.nearbyint(v)
        v = tl.where(v != v, -128., v)
        v = tl.minimum(127., tl.maximum(-128., v))
        tl.store(H + row*K + col, h)
        tl.store(Q + row*K + col, v.to(tl.int8))
        tl.store(S + row, scale)
    return kernel


def validate(x, a, g, w, eps, warps):
    if x.ndim < 2 or x.shape[-1] != 2048 or x.numel() == 0:
        raise ValueError('nonempty residual rows of width 2048 required')
    if a.shape != x.shape or g.shape != (2048,) or w.shape != (2048,):
        raise ValueError('residual/scale/RMS weight shape mismatch')
    if any(t.dtype != torch.float16 or not t.is_contiguous() or t.device != x.device
           for t in (x, a, g, w)):
        raise ValueError('contiguous FP16 tensors on the same device required')
    if not 0 < eps < 1 or warps not in (4, 8, 16):
        raise ValueError('invalid epsilon or warp count')


def residual_quant(x, a, g, w, eps, warps=8, round_stats=False):
    validate(x, a, g, w, eps, warps)
    if x.device.type != 'cuda' or torch.version.hip is not None:
        raise ValueError('NVIDIA CUDA required')
    m, k = x.numel()//2048, 2048
    h = torch.empty_like(x)
    q = torch.empty((m,k), device=x.device, dtype=torch.int8)
    s = torch.empty((m,1), device=x.device, dtype=torch.float32)
    _kernel()[(m,)](x, a, g, w, h, q, s, eps, k, round_stats,
                    num_warps=warps, enable_fp_fusion=False)
    return h, q, s


@torch.library.custom_op('h3vae_norm_exp::residual_rms_linear', mutates_args=())
def residual_rms_linear(x: torch.Tensor, a: torch.Tensor, g: torch.Tensor,
                        w: torch.Tensor, qw: torch.Tensor, ws: torch.Tensor,
                        bias: torch.Tensor | None, eps: float, warps: int
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    h,q,s = residual_quant(x,a,g,w,eps,warps)
    out = prequantized_linear(q,s,qw,ws,bias)
    return h,out.reshape(*x.shape[:-1],qw.shape[0])


@residual_rms_linear.register_fake
def _fake(x,a,g,w,qw,ws,bias,eps,warps):
    return torch.empty_like(x), x.new_empty((*x.shape[:-1],qw.shape[0]))


class NormQuantBlock(nn.Module):
    def __init__(self, block, warps=8):
        super().__init__()
        if not isinstance(block.norm2,nn.RMSNorm) or not block.use_scale:
            raise ValueError('scaled residual block with RMSNorm required')
        if block.norm2.normalized_shape != (2048,) or block.norm2.weight is None:
            raise ValueError('affine RMSNorm width 2048 required')
        self.norm1,self.norm2 = block.norm1,block.norm2
        self.attn,self.ff = block.attn,block.ff
        self.scale1,self.scale2 = block.scale1,block.scale2
        self.warps = warps
        self.train(block.training)

    def forward(self, hidden_states, rotary_pos_emb=None, pack_info=None):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('eval + inference/no_grad required')
        if pack_info is None:
            pack_info = {}
        normed = self.norm1(hidden_states.float()).to(hidden_states.dtype)
        attn = self.attn(normed,rotary_pos_emb,pack_info)
        w1,w2 = self.ff.w1,self.ff.w2
        h,hidden = residual_rms_linear(hidden_states,attn,self.scale1,self.norm2.weight,
                                       w1.qweight,w1.weight_scale,w1.bias,
                                       self.norm2.eps,self.warps)
        out = swiglu_int8_linear(hidden,w2.qweight,w2.weight_scale,w2.bias,16)
        return h + out*self.scale2


def clone_norm_quant_decoder(raw, warps=8):
    import os
    if os.environ.get('MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM','1').lower() not in ('1','true','yes','on'):
        raise ValueError('experimental fusion requires FP32 norm enabled')
    if warps not in (4,8,16):
        raise ValueError('unsupported warp count')
    result = clone_fused_decoder(raw,16)
    result.transformer_blocks = nn.ModuleList(
        [NormQuantBlock(block,warps).eval() for block in result.transformer_blocks])
    return result
