"""Fixed-contract final residual/LayerNorm kernel for the SM120 fusion.

No captured compiled kernel supplies this operation's output. Torch Inductor's
Welford routine is a lazy dependency, not copied implementation code.
"""
from functools import lru_cache

import torch

SHAPE = (4, 1797, 2048)
ROWS = 7188
WIDTH = 2048
EPS = 9.999999747378752e-06


def validate_metadata(h, o, gate, gamma, beta, *, require_cuda):
    tensors = (h, o, gate, gamma, beta)
    if (tuple(h.shape) != SHAPE or tuple(o.shape) != SHAPE or
            any(tuple(v.shape) != (WIDTH,) for v in tensors[2:]) or
            any(v.dtype != torch.float16 or not v.is_contiguous() for v in tensors) or
            any(v.device != h.device for v in tensors)):
        raise ValueError("expected contiguous FP16 B4 H/O and width-2048 vectors on one device")
    if require_cuda and (h.device.type != "cuda" or torch.version.hip is not None):
        raise ValueError("fixed final LayerNorm requires NVIDIA CUDA")


@lru_cache(maxsize=1)
def kernel_for_inspection():
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
    from torch._inductor.runtime import triton_helpers

    @triton.jit
    def final_residual_ln_w8_kernel(
        H, O, GATE, GAMMA, BETA, Y,
        M: tl.constexpr, K: tl.constexpr, EPSILON: tl.constexpr,
    ):
        row = tl.program_id(0) + tl.arange(0, 1)[:, None]
        col = tl.arange(0, K)[None, :]
        index = row * K + col
        mask = row < M
        h = tl.load(H + index, mask, other=0.0).to(tl.float32)
        o = tl.load(O + index, mask, other=0.0).to(tl.float32)
        gate = tl.load(GATE + col).to(tl.float32)
        raw = libdevice.fma(o, gate, h)
        zero = tl.full((1, K), 0.0, tl.float32)
        one = tl.full((1, K), 1.0, tl.float32)
        mean, m2, _weight = triton_helpers.welford(raw, zero, one, 1)
        variance = m2[:, None] / 2048.0
        inv = libdevice.rsqrt(variance + EPSILON)
        normalized = (raw - mean[:, None]) * inv
        gamma = tl.load(GAMMA + col).to(tl.float32)
        beta = tl.load(BETA + col).to(tl.float32)
        result = libdevice.fma(normalized, gamma, beta)
        tl.store(Y + index, result, mask)

    return final_residual_ln_w8_kernel


def final_residual_ln_w8_impl(h, o, gate, gamma, beta):
    validate_metadata(h, o, gate, gamma, beta, require_cuda=True)
    if torch.is_grad_enabled():
        raise RuntimeError("fixed final LayerNorm requires inference/no_grad")
    y = torch.empty_like(h, memory_format=torch.contiguous_format)
    if any(y.data_ptr() == v.data_ptr() for v in (h, o, gate, gamma, beta)):
        raise RuntimeError("final LayerNorm output must own separate storage")
    kernel_for_inspection()[(ROWS,)](
        h, o, gate, gamma, beta, y, ROWS, WIDTH, EPS,
        num_warps=8, num_stages=1, enable_fp_fusion=True,
    )
    return y


@torch.library.custom_op("h3vae_final_ln::residual_ln_w8", mutates_args=())
def final_residual_ln_w8(
    h: torch.Tensor, o: torch.Tensor, gate: torch.Tensor,
    gamma: torch.Tensor, beta: torch.Tensor,
) -> torch.Tensor:
    return final_residual_ln_w8_impl(h, o, gate, gamma, beta)


@final_residual_ln_w8.register_fake
def _fake(h, o, gate, gamma, beta):
    validate_metadata(h, o, gate, gamma, beta, require_cuda=False)
    return torch.empty_like(h, memory_format=torch.contiguous_format)
