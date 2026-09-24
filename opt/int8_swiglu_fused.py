"""Opt-in no-rotation SwiGLU + rowwise INT8 quantization.

Uses the version-pinned private CK 0.2.34 CUTLASS entry point;
unsupported inputs fail rather than silently falling back or re-quantizing.
The original checkpoint, quantized weights and FP16 defaults are unchanged.
"""
import copy
from functools import lru_cache
import importlib.metadata

import torch
from torch import nn


@lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def quant(X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        gate = tl.load(X + row * (2*K) + col, col < K, other=0).to(tl.float32)
        up = tl.load(X + row * (2*K) + K + col, col < K, other=0).to(tl.float32)
        # Match the compiled baseline: no intermediate FP16 SiLU rounding.
        act = ((gate / (1.0 + libdevice.exp(-gate))) * up).to(tl.float16).to(tl.float32)
        amax = tl.max(tl.abs(act), axis=0)
        scale = tl.maximum(amax * (1.0 / 127.0), 1.0e-30)
        # CK CUDA FP16 quantization divides in source dtype, then rounds.
        denom = scale.to(tl.float16).to(tl.float32)
        value = tl.div_rn(act, denom).to(tl.float16).to(tl.float32)
        value = libdevice.nearbyint(value)
        value = tl.where(value != value, -128.0, value)
        value = tl.minimum(127.0, tl.maximum(-128.0, value))
        tl.store(Q + row*K + col, value.to(tl.int8), col < K)
        tl.store(S + row, scale)

    return quant


def validate_input(x):
    if x.dtype != torch.float16 or x.ndim < 2 or not x.is_contiguous():
        raise ValueError('contiguous FP16 input with at least two dimensions required')
    if x.shape[-1] <= 0 or x.shape[-1] % 2 or x.numel() == 0:
        raise ValueError('nonempty [gate | up] rows of even width required')
    if x.shape[-1] // 2 > 16384:
        raise ValueError('experimental quantizer K exceeds 16384')


def quantize_swiglu(x, *, num_warps=8):
    validate_input(x)
    if x.device.type != 'cuda' or torch.version.hip is not None:
        raise ValueError('NVIDIA CUDA required')
    if num_warps not in (4, 8, 16):
        raise ValueError('unsupported warp count')
    import triton
    k = x.shape[-1] // 2
    rows = x.numel() // (2*k)
    q = torch.empty((rows, k), device=x.device, dtype=torch.int8)
    scale = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    _kernel()[(rows,)](x, q, scale, k, triton.next_power_of_2(k),
                      num_warps=num_warps, enable_fp_fusion=False)
    return q, scale


@lru_cache(maxsize=1)
def require_backend():
    if importlib.metadata.version('comfy-kitchen') != '0.2.34':
        raise RuntimeError('experiment requires exactly comfy-kitchen==0.2.34')
    from comfy_kitchen.backends import cuda
    if not hasattr(cuda._C, 'cutlass_int8_dequant'):
        raise RuntimeError('CK CUTLASS INT8 entry point unavailable')
    return cuda


def prequantized_linear(q, scale, weight, weight_scale, bias=None):
    """Same CK CUTLASS GEMM, consuming already quantized data exactly once."""
    backend = require_backend()
    m, k = q.shape
    n = weight.shape[0]
    if q.dtype != torch.int8 or weight.dtype != torch.int8 or weight.shape[1] != k:
        raise ValueError('INT8 matrix dimensions/types do not match')
    if scale.dtype != torch.float32 or weight_scale.dtype != torch.float32:
        raise ValueError('FP32 scales required')
    if scale.numel() != m or weight_scale.numel() != n:
        raise ValueError('one scale per activation/weight row required')
    tensors = (q, scale, weight, weight_scale) + (() if bias is None else (bias,))
    if any(t.device != q.device or not t.is_contiguous() for t in tensors):
        raise ValueError('contiguous tensors on the same CUDA device required')
    if q.device.type != 'cuda' or torch.version.hip is not None:
        raise ValueError('NVIDIA CUDA required')
    if bias is not None and (bias.dtype != torch.float16 or bias.numel() != n):
        raise ValueError('FP16 bias matching output channels required')
    out = torch.empty((m, n), device=q.device, dtype=torch.float16)
    bias_arg = bias if bias is not None else torch.empty(0, device=q.device, dtype=torch.float16)
    wrap = backend._wrap_for_dlpack
    with torch.cuda.device(q.device):
        used = backend._C.cutlass_int8_dequant(
            wrap(q), wrap(weight), wrap(scale.reshape(m, 1)),
            wrap(weight_scale), wrap(bias_arg), wrap(out),
            backend.DTYPE_TO_CODE[torch.float16],
            torch.cuda.current_stream(q.device).cuda_stream)
    if not used:
        raise RuntimeError('experimental CUTLASS GEMM rejected the shape; no fallback')
    return out


@torch.library.custom_op('h3vae_exp::swiglu_int8_linear', mutates_args=())
def swiglu_int8_linear(x: torch.Tensor, weight: torch.Tensor,
                      weight_scale: torch.Tensor, bias: torch.Tensor | None,
                      warps: int) -> torch.Tensor:
    q, scale = quantize_swiglu(x, num_warps=warps)
    out = prequantized_linear(q, scale, weight, weight_scale, bias)
    return out.reshape(*x.shape[:-1], weight.shape[0])


@swiglu_int8_linear.register_fake
def _fake(x, weight, weight_scale, bias, warps):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


class FusedFFN(nn.Module):
    def __init__(self, original, warps=8):
        super().__init__()
        if getattr(original, 'mode', None) != 'INT8both':
            raise ValueError('requires an existing INT8both FFN')
        self.w1, self.w2 = original.w1, original.w2
        self.warps = warps
        self.train(original.training)

    def forward(self, x):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('eval + no_grad/inference_mode required')
        hidden = self.w1(x)
        return swiglu_int8_linear(hidden, self.w2.qweight,
                                 self.w2.weight_scale, self.w2.bias, self.warps)


def clone_fused_decoder(raw, warps=8):
    require_backend()
    if raw.training:
        raise ValueError('decoder.eval() required')
    def clone(module):
        new = copy.copy(module)
        new._parameters = module._parameters.copy()
        new._buffers = module._buffers.copy()
        new._modules = {key: None if child is None else clone(child)
                        for key, child in module._modules.items()}
        return new
    result = clone(raw)
    for block in result.transformer_blocks:
        block.ff = FusedFFN(block.ff, warps).eval()
    return result
