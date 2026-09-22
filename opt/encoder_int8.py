"""INT8 encoder convolution kernel used by the optional integration.

It implements a real, bounded implicit-GEMM 3-D convolution for already
padded H3 encoder tensors.  The production runtime imports it only when the
explicit ``int8_encode`` option is enabled:

* activations are quantized once per call with one FP32 tensor scale;
* weights are prepared once with one FP32 scale per output channel;
* the Triton kernel gathers convolution windows directly from channels-last
  storage and uses ``tl.dot(int8, int8, out_dtype=int32)``;
* dequantization, bias and FP16 output are fused in the kernel;
* no im2col/unfold activation matrix is materialized and there is no FP16
  fallback when Triton/CUDA is unavailable.

The optional integration calls ``int8_valid_conv3d`` after padding outside the
module.  This mirrors the optimized H3 prefix call sites, which pass already
padded tensors and keep causal/reflect padding in their existing wrappers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


_TRITON_KERNELS: tuple[Any, Any] | None = None
_TILE_VARIANTS = {
    "64x64x64": (64, 64, 64),
    "128x64x64": (128, 64, 64),
    "128x64x128": (128, 64, 128),
}


@dataclass(frozen=True)
class Int8Conv3DConfig:
    """Static configuration for the supported valid convolution contract."""

    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int, int]
    stride: tuple[int, int, int]
    dilation: tuple[int, int, int] = (1, 1, 1)
    groups: int = 1


def int32_accumulator_bound(config: Int8Conv3DConfig) -> int:
    """Worst-case absolute INT32 dot sum for symmetric int8 operands."""

    reduction = config.in_channels * math.prod(config.kernel_size)
    return reduction * 127 * 127


def validate_int8_weight_buffers(
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    config: Int8Conv3DConfig,
) -> None:
    """Validate static buffers once before a timed loop.

    The finite/positive checks intentionally live outside
    ``int8_valid_conv3d`` so a production caller does not introduce a hidden
    device synchronization on every convolution call.
    """

    expected_qshape = (
        config.out_channels,
        config.kernel_size[0] * config.kernel_size[1]
        * config.kernel_size[2] * config.in_channels,
    )
    if tuple(qweight.shape) != expected_qshape:
        raise ValueError(
            f"qweight shape {tuple(qweight.shape)} != packed {expected_qshape}"
        )
    if qweight.dtype != torch.int8 or not qweight.is_contiguous():
        raise ValueError("qweight must be contiguous torch.int8")
    if tuple(weight_scale.shape) != (config.out_channels,):
        raise ValueError("weight_scale must be [out_channels]")
    if weight_scale.dtype != torch.float32 or not weight_scale.is_contiguous():
        raise ValueError("weight_scale must be contiguous FP32")
    if not bool(torch.isfinite(weight_scale).all().item()) or not bool(
        (weight_scale > 0).all().item()
    ):
        raise ValueError("weight_scale must be finite and positive")


def _triton_kernels() -> tuple[Any, Any]:
    """Lazily compile Triton function objects, keeping CPU import safe."""

    global _TRITON_KERNELS
    if _TRITON_KERNELS is not None:
        return _TRITON_KERNELS
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:  # pragma: no cover - CPU-only environments
        raise RuntimeError(
            "INT8 encoder requires Triton; no FP16 fallback is available"
        ) from exc

    @triton.jit
    def quantize_tensor_kernel(
        x_ptr,
        q_ptr,
        n_elements,
        channels,
        depth,
        height,
        width,
        sx0,
        sx1,
        sx2,
        sx3,
        sx4,
        scale_ptr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements

        # The destination is channels-last-3d, so linear offsets are physical
        # NDHWC: C is the fastest logical index.  This makes q writes fully
        # contiguous while source loads still honor arbitrary x strides.
        oc = offsets % channels
        rem = offsets // channels
        ow = rem % width
        rem = rem // width
        oh = rem % height
        rem = rem // height
        od = rem % depth
        ob = rem // depth

        x_offsets = (
            ob * sx0 + oc * sx1 + od * sx2 + oh * sx3 + ow * sx4
        )
        q_offsets = offsets
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr).to(tl.float32)
        scaled = x / scale
        rounded = tl.where(
            scaled >= 0.0,
            tl.math.floor(scaled + 0.5),
            -tl.math.floor(-scaled + 0.5),
        )
        rounded = tl.maximum(tl.minimum(rounded, 127.0), -127.0)
        tl.store(q_ptr + q_offsets, rounded.to(tl.int8), mask=mask)

    @triton.jit
    def implicit_conv3d_kernel(
        qx_ptr,
        qw_ptr,
        x_scale_ptr,
        w_scale_ptr,
        bias_ptr,
        out_ptr,
        batch,
        channels,
        depth,
        height,
        width,
        out_channels,
        out_depth,
        out_height,
        out_width,
        sx0,
        sx1,
        sx2,
        sx3,
        sx4,
        so0,
        so1,
        so2,
        so3,
        so4,
        stride_d,
        stride_h,
        stride_w,
        KERNEL_D: tl.constexpr,
        KERNEL_H: tl.constexpr,
        KERNEL_W: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        output_m = out_depth * out_height * out_width
        total_m = batch * output_m
        m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = m_offsets < total_m
        n_mask = n_offsets < out_channels

        ow = m_offsets % out_width
        rem = m_offsets // out_width
        oh = rem % out_height
        rem = rem // out_height
        od = rem % out_depth
        ob = rem // out_depth

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        k_total = KERNEL_D * KERNEL_H * KERNEL_W * channels
        for kd in range(KERNEL_D):
            input_d = od * stride_d + kd
            for kh in range(KERNEL_H):
                input_h = oh * stride_h + kh
                for kw in range(KERNEL_W):
                    input_w = ow * stride_w + kw
                    tap = (kd * KERNEL_H + kh) * KERNEL_W + kw
                    for c_start in range(0, channels, BLOCK_K):
                        c_offsets = c_start + tl.arange(0, BLOCK_K)
                        c_mask = c_offsets < channels
                        x_offsets = (
                            ob[:, None] * sx0
                            + c_offsets[None, :] * sx1
                            + input_d[:, None] * sx2
                            + input_h[:, None] * sx3
                            + input_w[:, None] * sx4
                        )
                        w_offsets = (
                            tap * channels
                            + c_offsets[:, None]
                            + n_offsets[None, :] * k_total
                        )
                        x_values = tl.load(
                            qx_ptr + x_offsets,
                            mask=m_mask[:, None] & c_mask[None, :],
                            other=0,
                        )
                        w_values = tl.load(
                            qw_ptr + w_offsets,
                            mask=c_mask[:, None] & n_mask[None, :],
                            other=0,
                        )
                        accumulator = tl.dot(
                            x_values,
                            w_values,
                            acc=accumulator,
                            out_dtype=tl.int32,
                        )

        x_scale = tl.load(x_scale_ptr).to(tl.float32)
        w_scale = tl.load(w_scale_ptr + n_offsets, mask=n_mask, other=0.0)
        values = accumulator.to(tl.float32) * x_scale * w_scale[None, :]
        if HAS_BIAS:
            bias = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0)
            values += bias[None, :].to(tl.float32)

        out_offsets = (
            ob[:, None] * so0
            + n_offsets[None, :] * so1
            + od[:, None] * so2
            + oh[:, None] * so3
            + ow[:, None] * so4
        )
        tl.store(
            out_ptr + out_offsets,
            values.to(tl.float16),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    _TRITON_KERNELS = (quantize_tensor_kernel, implicit_conv3d_kernel)
    return _TRITON_KERNELS


def _validate_conv_config(
    weight: torch.Tensor,
    stride: tuple[int, int, int],
    dilation: tuple[int, int, int],
    groups: int,
) -> Int8Conv3DConfig:
    if weight.ndim != 5:
        raise ValueError("weight must be [out_channels, in_channels, kd, kh, kw]")
    if groups != 1:
        raise ValueError("INT8 implicit conv currently supports groups=1 only")
    if tuple(dilation) != (1, 1, 1):
        raise ValueError("INT8 implicit conv currently supports dilation=1 only")
    stride = tuple(int(v) for v in stride)
    if any(v not in (1, 2) for v in stride):
        raise ValueError("INT8 implicit conv supports stride values 1 or 2")
    kernel_size = tuple(int(v) for v in weight.shape[-3:])
    if any(v not in (1, 3) for v in kernel_size):
        raise ValueError("INT8 implicit conv supports kernel dimensions 1 or 3")
    return Int8Conv3DConfig(
        in_channels=int(weight.shape[1]),
        out_channels=int(weight.shape[0]),
        kernel_size=kernel_size,
        stride=stride,
        dilation=(1, 1, 1),
        groups=1,
    )


def prepare_int8_weight(
    weight: torch.Tensor,
    *,
    stride: tuple[int, int, int] = (1, 1, 1),
    dilation: tuple[int, int, int] = (1, 1, 1),
    groups: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, Int8Conv3DConfig]:
    """Prepare packed ``[O, Kd, Kh, Kw, C]`` INT8 weights once."""

    if weight.numel() == 0 or not weight.is_floating_point():
        raise ValueError("source weight must be a non-empty floating tensor")
    if not bool(torch.isfinite(weight.detach()).all().item()):
        raise ValueError("source weight contains non-finite values")
    config = _validate_conv_config(weight, stride, dilation, groups)
    bound = int32_accumulator_bound(config)
    if bound >= 2**31:
        raise ValueError(
            f"INT32 accumulator bound {bound} exceeds signed INT32 range"
        )
    source = weight.detach().float().contiguous()
    packed = source.permute(0, 2, 3, 4, 1).reshape(config.out_channels, -1)
    scale = packed.abs().amax(dim=1).div(127.0)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    qweight = torch.round(packed / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return qweight.contiguous(), scale.float().contiguous(), config


@torch.no_grad()
def quantize_activation_tensor(
    x: torch.Tensor,
    *,
    check_finite: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one already-padded FP16 channels-last tensor once per call."""

    if x.ndim != 5:
        raise ValueError("activation must be [B,C,D,H,W]")
    if x.dtype != torch.float16:
        raise ValueError("INT8 implicit conv currently requires FP16 activations")
    if not x.is_cuda:
        raise RuntimeError("INT8 activation quantization requires CUDA")
    if check_finite and not bool(torch.isfinite(x).all().item()):
        raise ValueError("INT8 activation quantization rejects non-finite input")
    x = x.contiguous(memory_format=torch.channels_last_3d)
    scale = x.abs().amax().float().div(127.0)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    qx = torch.empty(
        x.shape,
        dtype=torch.int8,
        device=x.device,
        memory_format=torch.channels_last_3d,
    )
    quant_kernel, _ = _triton_kernels()
    block = 256
    grid = (triton_cdiv(x.numel(), block),)
    quant_kernel[grid](
        x,
        qx,
        x.numel(),
        x.shape[1],
        x.shape[2],
        x.shape[3],
        x.shape[4],
        *x.stride(),
        scale,
        BLOCK=block,
        num_warps=4,
    )
    return qx, scale


def triton_cdiv(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _valid_output_shape(
    x_shape: tuple[int, int, int, int, int],
    config: Int8Conv3DConfig,
) -> tuple[int, int, int, int, int]:
    b, _, d, h, w = x_shape
    kd, kh, kw = config.kernel_size
    sd, sh, sw = config.stride
    output = tuple(
        (length - kernel) // step + 1
        for length, kernel, step in (
            (d, kd, sd),
            (h, kh, sh),
            (w, kw, sw),
        )
    )
    if any(v <= 0 for v in output):
        raise ValueError(f"input shape {x_shape} is too small for valid conv")
    return (b, config.out_channels, *output)


@torch.no_grad()
def int8_valid_conv3d(
    x: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    config: Int8Conv3DConfig,
    bias: torch.Tensor | None = None,
    tile_variant: str = "64x64x64",
) -> torch.Tensor:
    """Run the real Triton INT8 valid convolution with no FP16 fallback."""

    if tile_variant not in _TILE_VARIANTS:
        raise ValueError(f"unknown tile variant {tile_variant!r}")
    if x.ndim != 5 or x.device.type != "cuda":
        raise RuntimeError("INT8 implicit conv requires a CUDA 5-D activation")
    if x.dtype != torch.float16:
        raise ValueError("INT8 implicit conv currently requires FP16 activations")
    if qweight.dtype != torch.int8 or qweight.device != x.device:
        raise ValueError("qweight must be CUDA torch.int8")
    if not qweight.is_contiguous():
        raise ValueError("qweight must use packed contiguous layout")
    if weight_scale.dtype != torch.float32 or weight_scale.device != x.device:
        raise ValueError("weight_scale must be CUDA FP32")
    if tuple(weight_scale.shape) != (config.out_channels,):
        raise ValueError("weight_scale must be [out_channels]")
    if not weight_scale.is_contiguous():
        raise ValueError("weight_scale must be contiguous")
    expected_qshape = (
        config.out_channels,
        config.kernel_size[0] * config.kernel_size[1]
        * config.kernel_size[2] * config.in_channels,
    )
    if tuple(qweight.shape) != expected_qshape:
        raise ValueError(
            f"qweight shape {tuple(qweight.shape)} != packed {expected_qshape}"
        )
    if x.shape[1] != config.in_channels:
        raise ValueError("activation channel count does not match qweight")
    if bias is not None and (
        bias.device != x.device
        or bias.dtype != torch.float16
        or bias.ndim != 1
        or bias.numel() != config.out_channels
        or not bias.is_contiguous()
    ):
        raise ValueError("bias must be contiguous CUDA FP16 [out_channels]")

    x = x.contiguous(memory_format=torch.channels_last_3d)
    qx, activation_scale = quantize_activation_tensor(x)
    output_shape = _valid_output_shape(tuple(x.shape), config)
    output = torch.empty(
        output_shape,
        dtype=torch.float16,
        device=x.device,
        memory_format=torch.channels_last_3d,
    )
    _, conv_kernel = _triton_kernels()
    block_m, block_n, block_k = _TILE_VARIANTS[tile_variant]
    grid = (
        triton_cdiv(output_shape[0] * output_shape[2] * output_shape[3] * output_shape[4], block_m),
        triton_cdiv(output_shape[1], block_n),
    )
    has_bias = bias is not None
    if bias is None:
        bias = torch.empty((1,), dtype=torch.float16, device=x.device)
    conv_kernel[grid](
        qx,
        qweight,
        activation_scale,
        weight_scale,
        bias,
        output,
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.shape[3],
        x.shape[4],
        output.shape[1],
        output.shape[2],
        output.shape[3],
        output.shape[4],
        *x.stride(),
        *output.stride(),
        config.stride[0],
        config.stride[1],
        config.stride[2],
        KERNEL_D=config.kernel_size[0],
        KERNEL_H=config.kernel_size[1],
        KERNEL_W=config.kernel_size[2],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        HAS_BIAS=has_bias,
        num_warps=4,
        num_stages=2,
    )
    return output


def int8_valid_conv3d_cpu_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: tuple[int, int, int] = (1, 1, 1),
) -> torch.Tensor:
    """Tiny CPU oracle for contracts; never used by the CUDA implementation."""

    qweight, weight_scale, config = prepare_int8_weight(weight, stride=stride)
    scale = x.float().abs().amax().div(127.0)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    scaled = x.float() / scale
    rounded = torch.where(
        scaled >= 0,
        torch.floor(scaled + 0.5),
        -torch.floor(-scaled + 0.5),
    )
    qx = rounded.clamp(-127, 127).to(torch.int8)
    deq_x = qx.float() * scale
    packed = qweight.float() * weight_scale[:, None]
    unpacked = packed.reshape(
        config.out_channels,
        config.kernel_size[0],
        config.kernel_size[1],
        config.kernel_size[2],
        config.in_channels,
    ).permute(0, 4, 1, 2, 3)
    return F.conv3d(deq_x, unpacked, bias, stride=config.stride, padding=0)


def _source_pad(
    x: torch.Tensor,
    padding: tuple[int, int, int],
    kernel_size: tuple[int, int, int],
    *,
    causal: bool,
    spatial_mode: str,
    temporal_mode: str,
) -> torch.Tensor:
    """Match BaseConv3d's explicit spatial/temporal padding for experiments."""

    pt, ph, pw = padding
    if not (pt or ph or pw):
        return x
    if ph or pw:
        x = F.pad(x, (pw, pw, ph, ph, 0, 0), mode=spatial_mode)
    depth = x.shape[2]
    if depth > 1:
        left = 2 * pt if causal else pt
        right = 0 if causal else pt
        return F.pad(x, (0, 0, 0, 0, left, right), mode=temporal_mode)
    if temporal_mode == "constant":
        if not causal:
            raise ValueError("constant temporal padding requires causal mode")
        zeros = torch.zeros_like(x[:, :, :1]).expand(
            -1, -1, kernel_size[0] - 1, -1, -1
        )
        return torch.cat([zeros, x], dim=2)
    return x.expand(-1, -1, x.shape[2] + 2 * pt, -1, -1)


class TritonInt8Conv3d(nn.Module):
    """Inference-only wrapper around one source 3-D convolution."""

    def __init__(self, original: nn.Module, *, assume_padded: bool = False):
        super().__init__()
        if not isinstance(original, nn.Conv3d):
            raise ValueError("expected an nn.Conv3d-compatible source module")
        if not assume_padded and not hasattr(original, "causal"):
            raise ValueError(
                "unpadded wrapper requires a SpatialParallelConv3d-like source; "
                "use assume_padded=True for ordinary nn.Conv3d"
            )
        config = _validate_conv_config(
            original.weight,
            tuple(original.stride),
            tuple(original.dilation),
            int(original.groups),
        )
        qweight, scale, _ = prepare_int8_weight(
            original.weight,
            stride=config.stride,
            dilation=config.dilation,
            groups=config.groups,
        )
        validate_int8_weight_buffers(qweight, scale, config)
        # Keep the packed integer tensor visible to state_dict/load_list-style
        # module walkers.  Integral Parameters never receive gradients and
        # remain INT8 when a caller moves the module to FP16.
        self.qweight = nn.Parameter(qweight, requires_grad=False)
        self.register_buffer("weight_scale", scale, persistent=True)
        if original.bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            if original.bias.ndim != 1 or original.bias.shape[0] != config.out_channels:
                raise ValueError("source bias shape does not match output channels")
            if not bool(torch.isfinite(original.bias.detach()).all().item()):
                raise ValueError("source bias contains non-finite values")
            self.bias = nn.Parameter(
                original.bias.detach().clone().contiguous(), requires_grad=False
            )
        self.config = config
        self.padding = tuple(int(v) for v in original.padding)
        self.pad_mode = getattr(original, "pad_mode", "constant")
        self.pad_mode_t = getattr(original, "pad_mode_t", "constant")
        self.causal = bool(getattr(original, "causal", False))
        self.assume_padded = bool(assume_padded)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.assume_padded:
            x = _source_pad(
                x,
                self.padding,
                self.config.kernel_size,
                causal=self.causal,
                spatial_mode=self.pad_mode,
                temporal_mode=self.pad_mode_t,
            )
        return int8_valid_conv3d(
            x,
            self.qweight,
            self.weight_scale,
            config=self.config,
            bias=self.bias,
        )


__all__ = [
    "Int8Conv3DConfig",
    "TritonInt8Conv3d",
    "int8_valid_conv3d",
    "int8_valid_conv3d_cpu_reference",
    "prepare_int8_weight",
    "quantize_activation_tensor",
    "int32_accumulator_bound",
    "validate_int8_weight_buffers",
]
