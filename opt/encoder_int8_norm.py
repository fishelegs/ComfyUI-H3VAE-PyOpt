"""INT8 producer for the validated batch-one H3 encoder prefix.

GroupNorm, FP16 rounding, SiLU and causal/reflect padding are unchanged.
Two normalization passes trade repeated arithmetic for less memory traffic:
the first writes only partial absmax, the second writes INT8 directly.
No full-sized normalized FP16 intermediate is allocated.
CUDA/Triton imports remain lazy for CPU-only contract tests.
"""
from __future__ import annotations

import torch

from opt.encoder_int8 import _quantization_kernels


def _norm_statistics(x, weight, bias, eps, pre_bias=None):
    if x.ndim != 5 or x.device.type != "cuda":
        raise ValueError("INT8 norm producer requires a CUDA rank-5 input")
    b, c, d, h, w = x.shape
    if b != 1 or x.dtype != torch.float16 or c not in (128, 256) or d < 1 or min(h, w) < 2:
        raise ValueError("INT8 norm producer requires batch-one FP16 C=128/256, D>=1, H/W>=2")
    if c * (d + 2) * (h + 2) * (w + 2) >= 2**31:
        raise ValueError("Padded output exceeds int32 index range")
    for parameter in (weight, bias) + (() if pre_bias is None else (pre_bias,)):
        if (parameter.shape != (c,) or parameter.dtype != x.dtype
                or parameter.device != x.device or not parameter.is_contiguous()):
            raise ValueError("Expected contiguous FP16 channel vectors on input device")

    import triton
    from encoder_fused_temporal_norm import _partial, _merge
    from encoder_fused_norm_bias import _partial_bias

    # Match the existing opaque norm's reduction order exactly.
    bs, bc, warps = 128, 64, 8
    nparts = triton.cdiv(h * w, bs)
    p = torch.empty((d, nparts, 32), device=x.device, dtype=torch.float32)
    q = torch.empty_like(p)
    stats = torch.empty((d, 32, 2), device=x.device, dtype=torch.float32)
    grid = (nparts, triton.cdiv(c, bc), d)
    strides = (x.stride(2), x.stride(1), x.stride(3), x.stride(4))
    args = (p, q, c, h, w, *strides, 32, nparts, bs, bc)
    if pre_bias is None:
        _partial[grid](x, *args, num_warps=warps)
    else:
        _partial_bias[grid](x, pre_bias, *args, num_warps=warps)
    _merge[(32, d)](
        p, q, stats, h, w, c, 32, nparts, bs, eps,
        triton.next_power_of_2(nparts), num_warps=4,
    )
    return stats


def _norm_pack_with_maxima(x, weight, bias, eps, pre_bias=None):
    """Materialized reference producer retained for exact kernel validation."""
    stats = _norm_statistics(x, weight, bias, eps, pre_bias)
    import triton
    from encoder_fused_temporal_norm import _normalize_pack
    from encoder_fused_norm_bias import _normalize_pack_bias

    b, c, d, h, w = x.shape
    strides = (x.stride(2), x.stride(1), x.stride(3), x.stride(4))
    y = torch.empty(
        (b, c, d + 2, h + 2, w + 2), device=x.device,
        dtype=x.dtype, memory_format=torch.channels_last_3d,
    )
    count = triton.cdiv(y.numel(), 1024)
    maxima = torch.empty((count,), device=x.device, dtype=torch.float32)
    args = (weight, bias, stats, y, c, d, h, w, *strides, 32, 1024)
    options = dict(Maxima=maxima, WRITE_MAX=True, num_warps=4, enable_fp_fusion=False)
    if pre_bias is None:
        _normalize_pack[(count,)](x, *args, **options)
    else:
        _normalize_pack_bias[(count,)](x, pre_bias, *args, **options)
    return y, maxima


def quantized_temporal_norm_pad(x, weight, bias, eps, *, pre_bias=None):
    """Recompute rounded norm/SiLU values and write padded INT8 directly."""
    stats = _norm_statistics(x, weight, bias, eps, pre_bias)
    import triton
    from opt.encoder_int8_norm_recompute import recompute_pack_kernel

    b, c, d, h, w = x.shape
    qy = torch.empty(
        (b, c, d + 2, h + 2, w + 2), device=x.device,
        dtype=torch.int8, memory_format=torch.channels_last_3d,
    )
    block = 4096
    count = triton.cdiv(qy.numel(), block)
    maxima = torch.empty((count,), device=x.device, dtype=torch.float32)
    activation_scale = torch.empty((), device=x.device, dtype=torch.float32)
    kernel = recompute_pack_kernel()
    args = (
        x, pre_bias if pre_bias is not None else bias, weight, bias, stats,
        qy, maxima, activation_scale, c, d, h, w,
        x.stride(2), x.stride(1), x.stride(3), x.stride(4), pre_bias is not None,
    )
    # Both passes preserve the original FP16 rounding points. Disable FMA
    # contraction just as in the existing norm/pack implementation.
    kernel[(count,)](*args, False, block, num_warps=4, enable_fp_fusion=False)
    reduce, scale, _ = _quantization_kernels()
    nparts = triton.cdiv(count, 8192)
    partials = torch.empty((nparts,), device=x.device, dtype=torch.float32)
    reduce[(nparts,)](maxima, partials, count, GRID=nparts, BLOCK=8192, num_warps=4)
    scale[(1,)](
        partials, activation_scale, N=nparts,
        BLOCK=triton.next_power_of_2(nparts), num_warps=4,
    )
    kernel[(count,)](*args, True, block, num_warps=4, enable_fp_fusion=False)
    return qy, activation_scale
