"""Private block0 FP16 RMSNorm helper used only by the isolated candidate fork.

Implements only the B4 H3 block0 RMS operation. Triton is imported and
compiled on first real CUDA invocation; importing this module registers the
custom-op schema and fake implementation without importing Triton.
"""
from functools import lru_cache

import torch

_SHAPE = (4, 1797, 2048)
_WIDTH = 2048
_ROWS = 7188
_XBLOCK = 2
_R0_BLOCK = 2048
_EPS_F32 = 9.999999747378752e-06


def _validate_metadata(x, gamma, *, require_cuda):
    if (tuple(x.shape) != _SHAPE or tuple(gamma.shape) != (_WIDTH,) or
            x.dtype != torch.float16 or gamma.dtype != torch.float16 or
            not x.is_contiguous() or not gamma.is_contiguous() or
            x.device != gamma.device):
        raise ValueError(
            "expected contiguous B4 [4,1797,2048] and gamma [2048], "
            "both FP16 on one device"
        )
    if require_cuda and (x.device.type != "cuda" or torch.version.hip is not None):
        raise ValueError("block0 RMS X2/W16 requires NVIDIA CUDA")


@lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def rms_fp16_x2_w16_kernel(
        X, GAMMA, Y, M: tl.constexpr,
        XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr, EPS: tl.constexpr,
    ):
        rows = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
        cols = tl.arange(0, R0_BLOCK)
        mask = rows[:, None] < M

        x = tl.load(X + rows[:, None] * R0_BLOCK + cols[None, :],
                    mask, other=0.0).to(tl.float32)
        gamma = tl.load(GAMMA + cols).to(tl.float32)

        square_acc = tl.full((XBLOCK, R0_BLOCK), 0.0, tl.float32)
        square_acc = tl.where(mask, square_acc + x * x, square_acc)
        sum_sq = tl.sum(square_acc, axis=1)
        mean_sq = sum_sq / 2048.0
        mean_sq_eps = mean_sq + EPS
        inv_rms = libdevice.rsqrt(mean_sq_eps)
        normalized = x * inv_rms[:, None]
        affine = normalized * gamma[None, :]
        tl.store(Y + rows[:, None] * R0_BLOCK + cols[None, :], affine, mask)

    return rms_fp16_x2_w16_kernel


def kernel_for_inspection():
    """Return the lazy JIT function; after a launch its device cache is inspectable."""
    return _kernel()


def rms_fp16_x2_w16_impl(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Return an independent FP16 RMS output for the fixed block0 B4 shape."""
    _validate_metadata(x, gamma, require_cuda=True)
    if torch.is_grad_enabled():
        raise RuntimeError("block0 RMS X2/W16 requires inference/no_grad")

    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    if output.data_ptr() == x.data_ptr():
        raise RuntimeError("block0 RMS output must own separate storage")
    kernel_for_inspection()[(_ROWS // _XBLOCK,)](
        x, gamma, output, _ROWS, _XBLOCK, _R0_BLOCK, _EPS_F32,
        num_warps=16, num_stages=1, enable_fp_fusion=True,
    )
    return output


@torch.library.custom_op(
    "h3vae_norm1_block0::rms_fp16_x2_w16", mutates_args=()
)
def rms_fp16_x2_w16(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    return rms_fp16_x2_w16_impl(x, gamma)


@rms_fp16_x2_w16.register_fake
def _fake_rms_fp16_x2_w16(x, gamma):
    _validate_metadata(x, gamma, require_cuda=False)
    return torch.empty_like(x, memory_format=torch.contiguous_format)
