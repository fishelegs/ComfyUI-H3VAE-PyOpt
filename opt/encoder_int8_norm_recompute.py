"""Lazy two-pass norm/SiLU/pad producer with no FP16 activation scratch."""
from functools import lru_cache


@lru_cache(maxsize=1)
def recompute_pack_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def recompute_pack(
        X, PreBias, Weight, Bias, Stats, Q, Maxima, Scale,
        C: tl.constexpr, D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
        SD: tl.constexpr, SC: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
        HAS_PRE_BIAS: tl.constexpr, WRITE_QUANTIZED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        total: tl.constexpr = C * (D + 2) * (H + 2) * (W + 2)
        c = idx % C
        position = idx // C
        t = position // ((H + 2) * (W + 2)) - 2
        h = position // (W + 2) % (H + 2) - 1
        w = position % (W + 2) - 1
        h = tl.where(h < 0, -h, tl.where(h >= H, 2 * H - 2 - h, h))
        w = tl.where(w < 0, -w, tl.where(w >= W, 2 * W - 2 - w, w))
        valid = (idx < total) & (t >= 0) & (t < D)
        v = tl.load(X + t * SD + c * SC + h * SH + w * SW, valid, other=0).to(tl.float32)
        if HAS_PRE_BIAS:
            pre_bias = tl.load(PreBias + c).to(tl.float32)
            v = (v + pre_bias).to(tl.float16).to(tl.float32)
        stat = (t * 32 + c // (C // 32)) * 2
        mean = tl.load(Stats + stat, valid, other=0)
        rstd = tl.load(Stats + stat + 1, valid, other=0)
        gamma = tl.load(Weight + c).to(tl.float32)
        beta = tl.load(Bias + c).to(tl.float32)
        z = (((v - mean) * rstd) * gamma + beta).to(tl.float16).to(tl.float32)
        z = (z / (1. + tl.exp(-z))).to(tl.float16)
        values = tl.where(valid, z, 0.).to(tl.float32)
        if WRITE_QUANTIZED:
            scaled = values / tl.load(Scale)
            rounded = tl.where(
                scaled >= 0., tl.floor(scaled + 0.5), -tl.floor(-scaled + 0.5),
            )
            rounded = tl.maximum(tl.minimum(rounded, 127.), -127.)
            tl.store(Q + idx, rounded.to(tl.int8), idx < total)
        else:
            tl.store(Maxima + tl.program_id(0), tl.max(tl.abs(values), 0))

    return recompute_pack
