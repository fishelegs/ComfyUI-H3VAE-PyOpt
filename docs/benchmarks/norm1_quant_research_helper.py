"""Single-layout Gluon correction for the private retained-B4 S1 prototype."""

import triton.experimental.gluon as triton
import triton.experimental.gluon.language as tl
from triton.experimental.gluon.language.extra import libdevice


@triton.jit
def reference_local_square_sum(raw):
    # The observed baseline sums eight consecutive values per lane, using
    # FMA for even positions and separately rounded squares for odd positions.
    # Return one partial and seven zeros; the explicit blocked layout then
    # keeps the baseline inter-lane and inter-warp butterfly tree.
    partial = tl.inline_asm_elementwise(
        """{
            .reg .f32 odd, acc;
            mul.rn.f32 odd, $9, $9;
            fma.rn.f32 acc, $8, $8, odd;
            fma.rn.f32 acc, $10, $10, acc;
            mul.rn.f32 odd, $11, $11;
            add.rn.f32 acc, odd, acc;
            fma.rn.f32 acc, $12, $12, acc;
            mul.rn.f32 odd, $13, $13;
            add.rn.f32 acc, odd, acc;
            fma.rn.f32 acc, $14, $14, acc;
            mul.rn.f32 odd, $15, $15;
            add.rn.f32 $0, odd, acc;
            mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0;
            mov.b32 $4, 0; mov.b32 $5, 0; mov.b32 $6, 0; mov.b32 $7, 0;
        }""",
        constraints="=f,=f,=f,=f,=f,=f,=f,=f,f,f,f,f,f,f,f,f",
        args=[raw], dtype=tl.float32, is_pure=True, pack=8,
    )
    return tl.sum(partial, axis=1)


@triton.jit
def residual_norm_quant(H, O, G, W, Q, S, NORM, M: tl.constexpr, DEBUG: tl.constexpr):
    width: tl.constexpr = 2048
    layout: tl.constexpr = tl.BlockedLayout([1, 8], [1, 32], [1, 8], [1, 0])
    rows = tl.program_id(0) * 2 + tl.arange(0, 2, layout=tl.SliceLayout(1, layout))
    cols = tl.arange(0, width, layout=tl.SliceLayout(0, layout))
    mask = rows[:, None] < M
    h = tl.load(H + rows[:, None] * width + cols[None, :], mask, other=0).to(tl.float32)
    o = tl.load(O + rows[:, None] * width + cols[None, :], mask, other=0).to(tl.float32)
    g = tl.load(G + cols).to(tl.float32)
    w = tl.load(W + cols).to(tl.float32)

    raw = tl.fma(o, g[None, :], h)
    sum_sq = reference_local_square_sum(raw)
    residual_half = raw.to(tl.float16)
    tl.store(H + rows[:, None] * width + cols[None, :], residual_half, mask)
    residual = residual_half.to(tl.float32)
    inv_rms = libdevice.rsqrt(sum_sq / 2048.0 + 1.0e-5)
    norm_fp32 = residual * inv_rms[:, None]
    norm_fp32 = norm_fp32 * w[None, :]
    norm_half = norm_fp32.to(tl.float16)
    if DEBUG:
        tl.store(NORM + rows[:, None] * width + cols[None, :], norm_half, mask)

    act = norm_half.to(tl.float32)
    amax = tl.max(tl.abs(act), axis=1)
    scale = tl.maximum(amax * (1.0 / 127.0), 1.0e-30)
    denom = scale.to(tl.float16).to(tl.float32)
    value = tl.div_rn(act, denom[:, None]).to(tl.float16).to(tl.float32)
    value = libdevice.nearbyint(value)
    value = tl.where(value != value, -128.0, value)
    value = tl.minimum(127.0, tl.maximum(-128.0, value))
    tl.store(Q + rows[:, None] * width + cols[None, :], value.to(tl.int8), mask)
    tl.store(S + rows, scale, rows < M)
