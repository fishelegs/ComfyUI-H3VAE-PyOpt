"""SM120 INT8 FFN-up GEMM/SwiGLU fusion preserving the CK 0.2.34 contract.

INT32 accumulation, FP32 scales/FMA bias, FP16 linear and activation rounding,
and CK-compatible row quantization are retained. Triton compiles lazily.
Other CUDA architectures retain the existing decoder fusion implementation.
"""
from functools import lru_cache

import torch


CONFIG = (128, 64, 64, 4, 4)


def supports_fused_ffn_up(device):
    """Only select the architecture with paired full-decoder validation."""
    device = torch.device(device)
    return (device.type == 'cuda' and torch.version.hip is None
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability(device) == (12, 0))


def validate_prequantized(q, scale, weight, weight_scale, bias):
    if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] != 2048:
        raise ValueError('nonempty INT8 activation matrix with K2048 required')
    # Both output kernels form row * 8192 using signed 32-bit indices.
    if q.shape[0] > (1 << 31) // 8192:
        raise ValueError('INT8 FFN-up M exceeds 32-bit output indexing bound')
    if weight.shape != (16384, 2048):
        raise ValueError('INT8 FFN-up weight must have shape 16384x2048')
    if q.dtype != torch.int8 or weight.dtype != torch.int8:
        raise ValueError('INT8 activations and weights required')
    if scale.dtype != torch.float32 or weight_scale.dtype != torch.float32:
        raise ValueError('FP32 activation and weight scales required')
    if scale.shape != (q.shape[0], 1) or weight_scale.shape != (16384,):
        raise ValueError('one activation/weight scale per row required')
    if bias is not None and (bias.shape != (16384,) or bias.dtype != torch.float16):
        raise ValueError('FP16 bias with 16384 elements required')
    values = (q, scale, weight, weight_scale) + (() if bias is None else (bias,))
    if any(not value.is_contiguous() or value.device != q.device for value in values):
        raise ValueError('contiguous tensors on the same device required')


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def up_swiglu(Q, W, S, WS, B, ACT,
                   M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   HAS_BIAS: tl.constexpr):
        pid = tl.program_id(0)
        nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
        group = pid // (8 * nn)
        first = group * 8
        size = tl.minimum(nm - first, 8)
        pm = first + (pid % (8 * nn)) % size
        pn = (pid % (8 * nn)) // size
        rows = pm * BM + tl.arange(0, BM)
        pair = tl.arange(0, 2 * BN)
        cols = pn * BN + pair // 2
        wrows = cols + (pair % 2) * N
        ks = tl.arange(0, BK)
        qp = Q + rows[:, None] * K + ks[None, :]
        wp = W + wrows[None, :] * K + ks[:, None]
        acc = tl.zeros((BM, 2 * BN), tl.int32)
        for step in range(tl.cdiv(K, BK)):
            a = tl.load(qp, (rows[:, None] < M) & (ks[None, :] + step * BK < K), other=0)
            w = tl.load(wp, (cols[None, :] < N) & (ks[:, None] + step * BK < K), other=0)
            acc = tl.dot(a, w, acc, out_dtype=tl.int32)
            qp += BK
            wp += BK
        sx = tl.load(S + rows, rows < M, other=0)
        sw = tl.load(WS + wrows, cols < N, other=0)
        value = acc.to(tl.float32) * sx[:, None]
        if HAS_BIAS:
            bias = tl.load(B + wrows, cols < N, other=0).to(tl.float32)
            value = tl.fma(value, sw[None, :], bias[None, :])
        else:
            value = value * sw[None, :]
        rounded = value.to(tl.float16).to(tl.float32).reshape(BM, BN, 2)
        gate, up = tl.split(rounded)
        act = ((gate / (1.0 + libdevice.exp(-gate))) * up).to(tl.float16)
        out_cols = pn * BN + tl.arange(0, BN)
        tl.store(ACT + rows[:, None] * N + out_cols[None, :], act,
                 (rows[:, None] < M) & (out_cols[None, :] < N))

    @triton.jit
    def quantize_act(ACT, Q, S, N: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        act = tl.load(ACT + row * N + cols, cols < N, other=0).to(tl.float32)
        amax = tl.max(tl.abs(act), axis=0)
        scale = tl.maximum(amax * (1.0 / 127.0), 1.0e-30)
        denom = scale.to(tl.float16).to(tl.float32)
        value = tl.div_rn(act, denom).to(tl.float16).to(tl.float32)
        value = libdevice.nearbyint(value)
        value = tl.where(value != value, -128.0, value)
        value = tl.minimum(127.0, tl.maximum(-128.0, value))
        tl.store(Q + row * N + cols, value.to(tl.int8), cols < N)
        tl.store(S + row, scale)
    return up_swiglu, quantize_act


def quantize_ffn_up(q, scale, weight, weight_scale, bias=None):
    """Consume prequantized norm output and return exact INT8 SwiGLU rows/scales."""
    validate_prequantized(q, scale, weight, weight_scale, bias)
    if not supports_fused_ffn_up(q.device):
        raise ValueError('fused INT8 FFN-up requires NVIDIA CUDA SM120')
    if torch.is_grad_enabled():
        raise RuntimeError('fused INT8 FFN-up requires inference/no_grad')
    # The configured K=2048 bounds |sum(q*w)| by 2048*128*128 < 2**31.
    import triton
    m, k = q.shape
    n = weight.shape[0] // 2
    bm, bn, bk, warps, stages = CONFIG
    act = torch.empty((m, n), device=q.device, dtype=torch.float16)
    qout = torch.empty((m, n), device=q.device, dtype=torch.int8)
    sout = torch.empty((m, 1), device=q.device, dtype=torch.float32)
    producer, quantizer = _kernels()
    producer[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        q, weight, scale, weight_scale, bias if bias is not None else q, act,
        m, n, k, bm, bn, bk, bias is not None,
        num_warps=warps, num_stages=stages, enable_fp_fusion=False)
    quantizer[(m,)](act, qout, sout, n, triton.next_power_of_2(n),
                    num_warps=8, enable_fp_fusion=False)
    return qout, sout


def _residual_ffn_impl(x: torch.Tensor, a: torch.Tensor, g: torch.Tensor,
                       norm_weight: torch.Tensor, up_weight: torch.Tensor,
                       up_scale: torch.Tensor, up_bias: torch.Tensor | None,
                       down_weight: torch.Tensor, down_scale: torch.Tensor,
                       down_bias: torch.Tensor | None, eps: float, warps: int
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    from .int8_norm_quant import residual_quant
    from .int8_swiglu_fused import prequantized_linear, require_backend
    require_backend()  # Missing/wrong CK fails before any replacement arithmetic.
    if down_weight.shape != (2048, 8192):
        raise ValueError('INT8 FFN-down weight must have shape 2048x8192')
    h, q, scale = residual_quant(x, a, g, norm_weight, eps, warps)
    qout, sout = quantize_ffn_up(q, scale, up_weight, up_scale, up_bias)
    del q, scale
    out = prequantized_linear(qout, sout, down_weight, down_scale, down_bias)
    return h, out.reshape(*x.shape[:-1], down_weight.shape[0])


@torch.library.custom_op('h3vae_int8_ffn::residual_ffn', mutates_args=())
def fused_residual_ffn(x: torch.Tensor, a: torch.Tensor, g: torch.Tensor,
                       norm_weight: torch.Tensor, up_weight: torch.Tensor,
                       up_scale: torch.Tensor, up_bias: torch.Tensor | None,
                       down_weight: torch.Tensor, down_scale: torch.Tensor,
                       down_bias: torch.Tensor | None, eps: float, warps: int
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    return _residual_ffn_impl(x, a, g, norm_weight, up_weight, up_scale, up_bias,
                              down_weight, down_scale, down_bias, eps, warps)


@fused_residual_ffn.register_fake
def _fake(x, a, g, norm_weight, up_weight, up_scale, up_bias,
          down_weight, down_scale, down_bias, eps, warps):
    return torch.empty_like(x), x.new_empty((*x.shape[:-1], down_weight.shape[0]))
