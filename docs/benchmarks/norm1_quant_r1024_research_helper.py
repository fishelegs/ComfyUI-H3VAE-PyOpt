"""Private exact W2 residual/norm/quant helper; not runtime-wired."""

import triton.experimental.gluon as triton
import triton.experimental.gluon.language as tl
from triton.experimental.gluon.language.extra import libdevice


@triton.jit
def sequential_16_slot_sum(values):
    # Archived X1/R1024/W2 PTX reduces each thread's 16 FP32 accumulator
    # slots in register order with a left fold, then uses warp offsets
    # 16/8/4/2/1 and a two-warp shared-memory offset-1 fold.
    folded = tl.inline_asm_elementwise(
        """{
            .reg .f32 acc;
            add.rn.f32 acc, $16, $17;
            add.rn.f32 acc, acc, $18;
            add.rn.f32 acc, acc, $19;
            add.rn.f32 acc, acc, $20;
            add.rn.f32 acc, acc, $21;
            add.rn.f32 acc, acc, $22;
            add.rn.f32 acc, acc, $23;
            add.rn.f32 acc, acc, $24;
            add.rn.f32 acc, acc, $25;
            add.rn.f32 acc, acc, $26;
            add.rn.f32 acc, acc, $27;
            add.rn.f32 acc, acc, $28;
            add.rn.f32 acc, acc, $29;
            add.rn.f32 acc, acc, $30;
            add.rn.f32 acc, acc, $31;
            mov.b32 $0, acc;
            mov.b32 $1, 0;  mov.b32 $2, 0;  mov.b32 $3, 0;
            mov.b32 $4, 0;  mov.b32 $5, 0;  mov.b32 $6, 0;
            mov.b32 $7, 0;  mov.b32 $8, 0;  mov.b32 $9, 0;
            mov.b32 $10, 0; mov.b32 $11, 0; mov.b32 $12, 0;
            mov.b32 $13, 0; mov.b32 $14, 0; mov.b32 $15, 0;
        }""",
        constraints="=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f",
        args=[values], dtype=tl.float32, is_pure=True, pack=16,
    )
    return tl.sum(folded, axis=1)


@triton.jit
def residual_norm_quant(H, O, G, W, Q, S, NORM,
                        M: tl.constexpr, DEBUG: tl.constexpr):
    # This blocked layout is the exact archived R1024/W2 oracle layout.
    width: tl.constexpr = 2048
    half_width: tl.constexpr = 1024
    layout: tl.constexpr = tl.BlockedLayout(
        [1, 8], [1, 32], [1, 2], [1, 0]
    )
    rows = tl.program_id(0) + tl.arange(
        0, 1, layout=tl.SliceLayout(1, layout)
    )
    cols = tl.arange(0, half_width, layout=tl.SliceLayout(0, layout))
    mask = rows[:, None] < M

    h0 = tl.load(H + rows[:, None] * width + cols[None, :], mask, other=0).to(tl.float32)
    h1 = tl.load(H + rows[:, None] * width + half_width + cols[None, :], mask, other=0).to(tl.float32)
    o0 = tl.load(O + rows[:, None] * width + cols[None, :], mask, other=0).to(tl.float32)
    o1 = tl.load(O + rows[:, None] * width + half_width + cols[None, :], mask, other=0).to(tl.float32)
    g0 = tl.load(G + cols).to(tl.float32)
    g1 = tl.load(G + half_width + cols).to(tl.float32)
    w0 = tl.load(W + cols).to(tl.float32)
    w1 = tl.load(W + half_width + cols).to(tl.float32)

    # The native source loops over R0_BLOCK=1024 twice. The first square is
    # rounded, then the second segment is fused into that accumulator.
    raw0 = tl.fma(o0, g0[None, :], h0)
    raw1 = tl.fma(o1, g1[None, :], h1)
    square_acc = tl.fma(raw1, raw1, raw0 * raw0)
    sum_sq = sequential_16_slot_sum(square_acc)

    residual0 = raw0.to(tl.float16)
    residual1 = raw1.to(tl.float16)
    tl.store(H + rows[:, None] * width + cols[None, :], residual0, mask)
    tl.store(H + rows[:, None] * width + half_width + cols[None, :], residual1, mask)

    inv_rms = libdevice.rsqrt(sum_sq / 2048.0 + 1.0e-5)
    norm0 = residual0.to(tl.float32) * inv_rms[:, None]
    norm1 = residual1.to(tl.float32) * inv_rms[:, None]
    norm0 = norm0 * w0[None, :]
    norm1 = norm1 * w1[None, :]
    norm0 = norm0.to(tl.float16)
    norm1 = norm1.to(tl.float16)
    if DEBUG:
        tl.store(NORM + rows[:, None] * width + cols[None, :], norm0, mask)
        tl.store(NORM + rows[:, None] * width + half_width + cols[None, :], norm1, mask)

    act0 = norm0.to(tl.float32)
    act1 = norm1.to(tl.float32)
    amax0 = tl.max(tl.abs(act0), axis=1)
    amax1 = tl.max(tl.abs(act1), axis=1)
    amax = tl.maximum(amax0, amax1)
    scale = tl.maximum(amax * (1.0 / 127.0), 1.0e-30)
    denom = scale.to(tl.float16).to(tl.float32)
    value0 = tl.div_rn(act0, denom[:, None]).to(tl.float16).to(tl.float32)
    value1 = tl.div_rn(act1, denom[:, None]).to(tl.float16).to(tl.float32)
    value0 = libdevice.nearbyint(value0)
    value1 = libdevice.nearbyint(value1)
    value0 = tl.where(value0 != value0, -128.0, value0)
    value1 = tl.where(value1 != value1, -128.0, value1)
    value0 = tl.minimum(127.0, tl.maximum(-128.0, value0))
    value1 = tl.minimum(127.0, tl.maximum(-128.0, value1))
    tl.store(Q + rows[:, None] * width + cols[None, :], value0.to(tl.int8), mask)
    tl.store(Q + rows[:, None] * width + half_width + cols[None, :], value1.to(tl.int8), mask)
    tl.store(S + rows, scale, rows < M)
