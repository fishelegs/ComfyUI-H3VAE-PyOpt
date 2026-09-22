"""Optional INT8 wiring for the validated H3 encoder prefix.

The production policy in this module is intentionally narrow: only the two
residual blocks in encoder prefix stages 0 and 1 are replaced, covering eight
3x3x3 convolutions.  Their norm/padding wrappers remain in place and still
produce already-padded tensors; the INT8 kernel therefore receives a valid
convolution input and cannot silently change causal or reflect-padding
semantics.  Downsample convolutions, shortcuts, ``conv_in``/``conv_out``, and
the suffix retain their original H3 precision and execution paths.

This module is opt-in and is not imported by the default runtime.  The kernel
implementation lives in ``encoder_int8.py``; this file only owns the module
state, custom-op boundary, and installation into the existing fused prefix.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


from opt.encoder_int8 import (
    Int8Conv3DConfig,
    int8_valid_conv3d,
    prepare_int8_weight,
)


INT8_ENCODER_POLICY = {
    "stages": (0, 1),
    "blocks_per_stage": 2,
    "convolutions_per_block": ("first", "second"),
    "quantized_conv_count": 8,
    "kernel": (3, 3, 3),
    "tile_variants_by_input_channels": {
        128: "128x64x64",
        256: "128x64x128",
    },
    "excluded": (
        "conv_in",
        "conv_out",
        "downsample",
        "nin_shortcut",
        "encoder suffix",
        "quant_conv",
    ),
}


def _resolve_tile_variant(
    input_channels: int,
    requested: str | None,
) -> str:
    """Select the validated Triton tile for a source convolution.

    The two variants were measured on the real H3 prefix shapes. Keeping the
    selection here (rather than making the kernel guess from a dynamic tensor)
    makes it a compile-time constant for each installed module.
    """

    if requested not in (None, "auto"):
        return str(requested)
    try:
        return INT8_ENCODER_POLICY["tile_variants_by_input_channels"][
            int(input_channels)
        ]
    except KeyError as exc:
        raise ValueError(
            "INT8 encoder stage-aware tile policy only supports input channels "
            f"128 or 256, got {input_channels}"
        ) from exc


def _output_shape(
    x: torch.Tensor,
    qweight: torch.Tensor,
    kernel: tuple[int, int, int],
    stride: tuple[int, int, int],
) -> tuple[int, int, int, int, int]:
    if x.ndim != 5:
        raise ValueError("INT8 encoder activation must be rank 5")
    if qweight.ndim != 2:
        raise ValueError("INT8 encoder qweight must be rank 2")
    output_dims = tuple(
        (length - kernel_size) // step + 1
        for length, kernel_size, step in zip(x.shape[-3:], kernel, stride)
    )
    if any(value <= 0 for value in output_dims):
        raise ValueError(
            f"activation shape {tuple(x.shape)} is too small for valid conv"
        )
    return (x.shape[0], qweight.shape[0], *output_dims)


@torch.library.custom_op(
    "h3vae_encoder::int8_valid_conv3d_integration",
    mutates_args=(),
    device_types="cuda",
)
def int8_valid_conv3d_integration(
    x: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
    kernel_d: int,
    kernel_h: int,
    kernel_w: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    apply_bias: bool,
    tile_variant: str,
) -> torch.Tensor:
    """Opaque fullgraph boundary around the Triton INT8 valid convolution."""

    config = Int8Conv3DConfig(
        in_channels=int(qweight.shape[1]) // (int(kernel_d) * int(kernel_h) * int(kernel_w)),
        out_channels=int(qweight.shape[0]),
        kernel_size=(int(kernel_d), int(kernel_h), int(kernel_w)),
        stride=(int(stride_d), int(stride_h), int(stride_w)),
    )
    return int8_valid_conv3d(
        x,
        qweight,
        weight_scale,
        config=config,
        bias=bias if apply_bias else None,
        tile_variant=tile_variant,
    )


@int8_valid_conv3d_integration.register_fake
def _int8_valid_conv3d_integration_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
    kernel_d: int,
    kernel_h: int,
    kernel_w: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    apply_bias: bool,
    tile_variant: str,
) -> torch.Tensor:
    del weight_scale, bias, apply_bias, tile_variant
    shape = _output_shape(
        x,
        qweight,
        (int(kernel_d), int(kernel_h), int(kernel_w)),
        (int(stride_d), int(stride_h), int(stride_w)),
    )
    return torch.empty(
        shape,
        device=x.device,
        dtype=x.dtype,
        memory_format=torch.channels_last_3d,
    )


class Int8FusedValidConv3d(nn.Module):
    """Static-weight INT8 wrapper for an already-padded fused Conv3d call."""

    def __init__(
        self,
        source: nn.Module,
        *,
        apply_bias: bool,
        tile_variant: str = "128x64x64",
    ) -> None:
        super().__init__()
        source_weight = getattr(source, "weight", None)
        source_bias = getattr(source, "bias", None)
        if not isinstance(source_weight, torch.Tensor) or source_weight.ndim != 5:
            raise ValueError("INT8 encoder source must expose a rank-5 weight")
        if source_bias is None or not isinstance(source_bias, torch.Tensor):
            raise ValueError("INT8 encoder integration requires a source bias")
        if source_bias.ndim != 1 or source_bias.shape[0] != source_weight.shape[0]:
            raise ValueError("source bias must match the convolution output channels")
        if source_bias.device != source_weight.device:
            raise ValueError("source weight and bias must share a device")
        if source_weight.dtype != torch.float16:
            raise ValueError("INT8 encoder integration requires FP16 source weights")
        if source_bias.dtype != torch.float16:
            raise ValueError("INT8 encoder integration requires an FP16 source bias")
        if not bool(torch.isfinite(source_bias.detach()).all().item()):
            raise ValueError("source bias contains non-finite values")
        stride = tuple(int(value) for value in getattr(source, "stride", (1, 1, 1)))
        if tuple(source_weight.shape[-3:]) != (3, 3, 3):
            raise ValueError("the initial INT8 encoder policy only supports 3x3x3")
        qweight, weight_scale, config = prepare_int8_weight(
            source_weight,
            stride=stride,
            dilation=(1, 1, 1),
            groups=1,
        )
        if config.kernel_size != (3, 3, 3):
            raise ValueError("unexpected INT8 encoder kernel configuration")
        # qweight is an immutable state tensor, but a Parameter makes ComfyUI's
        # ModelPatcher include this leaf during CPU offload/reload.  It is not
        # trainable and remains INT8 under every _apply call.
        self.qweight = nn.Parameter(qweight.contiguous(), requires_grad=False)
        self.register_buffer(
            "weight_scale", weight_scale.float().contiguous(), persistent=True
        )
        # Keep a private FP16 bias Parameter.  The original fused wrapper's
        # bias remains available to its norm/residual epilogue; this copy is
        # the bias consumed only by the ordinary second convolution.
        self.bias = nn.Parameter(
            source_bias.detach().clone().contiguous(), requires_grad=False
        )
        self.config = config
        self.apply_bias = bool(apply_bias)
        self.tile_variant = str(tile_variant)

    def _apply(self, fn, recurse=True):
        # ComfyUI may call module.to(device, dtype).  FP32 scales are part of
        # the quantization contract and must not be rounded to FP16 first.
        original_scale = self.weight_scale.detach().clone()
        result = super()._apply(fn, recurse=recurse)
        self.qweight.data = self.qweight.data.to(
            device=self.qweight.device, dtype=torch.int8
        )
        self.weight_scale.data = original_scale.to(
            device=self.weight_scale.device, dtype=torch.float32
        )
        self.bias.data = self.bias.data.to(
            device=self.bias.device, dtype=torch.float16
        )
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float16:
            raise RuntimeError("INT8 encoder convolution requires FP16 activations")
        if x.device != self.qweight.device:
            raise RuntimeError(
                "INT8 encoder activation and qweight must be on the same device"
            )
        return int8_valid_conv3d_integration(
            x,
            self.qweight,
            self.weight_scale,
            self.bias,
            int(self.config.kernel_size[0]),
            int(self.config.kernel_size[1]),
            int(self.config.kernel_size[2]),
            int(self.config.stride[0]),
            int(self.config.stride[1]),
            int(self.config.stride[2]),
            self.apply_bias,
            self.tile_variant,
        )


def _check_encoder_platform(prefix: nn.Module) -> None:
    parameter_devices = {parameter.device for parameter in prefix.parameters()}
    if not parameter_devices:
        raise RuntimeError("int8_encode requires encoder prefix parameters")
    if any(device.type != "cuda" for device in parameter_devices):
        devices = ", ".join(sorted(str(device) for device in parameter_devices))
        raise RuntimeError(
            "int8_encode requires all encoder prefix parameters on CUDA; "
            f"got {devices}"
        )
    if len(parameter_devices) != 1:
        devices = ", ".join(sorted(str(device) for device in parameter_devices))
        raise RuntimeError(
            "int8_encode requires all encoder prefix parameters on one CUDA "
            f"device; got {devices}"
        )
    if torch.version.hip is not None:
        raise RuntimeError("int8_encode is CUDA-only and does not support ROCm")
    device = next(iter(parameter_devices))
    if not torch.cuda.is_available():
        raise RuntimeError("int8_encode requires an available CUDA GPU")
    capability = torch.cuda.get_device_capability(device)
    if tuple(capability) < (8, 0):
        raise RuntimeError(
            "int8_encode requires CUDA compute capability SM80 or newer; "
            f"got SM{capability[0]}{capability[1]}"
        )


def install_int8_encoder_prefix(
    prefix: nn.Module,
    *,
    require_cuda: bool = True,
    tile_variant: str | None = "auto",
) -> dict[str, Any]:
    """Install exactly the eight approved prefix residual convolutions."""

    if require_cuda:
        _check_encoder_platform(prefix)
    stages = getattr(prefix, "stages", None)
    if not isinstance(stages, nn.ModuleList) or len(stages) < 2:
        raise ValueError("INT8 encoder prefix must expose at least two stages")

    pending = []
    installed = []
    for stage_index in INT8_ENCODER_POLICY["stages"]:
        stage = stages[stage_index]
        blocks = getattr(stage, "block", None)
        if not isinstance(blocks, nn.ModuleList) or len(blocks) != 2:
            raise ValueError(
                f"INT8 encoder policy expected two blocks at stage {stage_index}"
            )
        for block_index, block in enumerate(blocks):
            block_type = type(block).__name__
            expected_types = {"BiasFusedResidualBlock", "ResidualDownsampleBlock"}
            if block_type not in expected_types:
                raise ValueError(
                    f"stage {stage_index} block {block_index} has unsupported "
                    f"type {block_type}"
                )
            is_residual_downsample = block_type == "ResidualDownsampleBlock"
            if stage_index == 1 and block_index == 1:
                if not is_residual_downsample or not hasattr(block, "down_weight"):
                    raise ValueError(
                        "stage 1 final block must be ResidualDownsampleBlock"
                    )
            elif is_residual_downsample:
                raise ValueError(
                    f"unexpected ResidualDownsampleBlock at stage {stage_index} "
                    f"block {block_index}"
                )
            for field in ("first", "second"):
                source = getattr(block, field, None)
                if source is None:
                    raise ValueError(
                        f"stage {stage_index} block {block_index} lacks {field}"
                    )
                if getattr(block, f"int8_{field}", None) is not None:
                    raise ValueError(
                        f"stage {stage_index} block {block_index} {field} "
                        "already has an INT8 wrapper"
                    )
                # The final stage-1 residual block uses residual_downsample_pack,
                # which adds second.bias after the convolution.  Do not add it
                # a second time in the INT8 epilogue.
                apply_bias = not (
                    field == "first"
                    or (stage_index == 1 and block_index == 1 and field == "second")
                )
                name = f"stages.{stage_index}.block.{block_index}.{field}"
                selected_tile = _resolve_tile_variant(
                    int(source.weight.shape[1]), tile_variant
                )
                wrapper = Int8FusedValidConv3d(
                    source,
                    apply_bias=apply_bias,
                    tile_variant=selected_tile,
                )
                pending.append(
                    (block, field, wrapper)
                )
                installed_item = {
                    "path": name,
                    "weight_shape": list(source.weight.shape),
                    "input_channels": int(source.weight.shape[1]),
                    "stride": list(wrapper.config.stride),
                    "apply_bias": apply_bias,
                    "tile_variant": selected_tile,
                }
                # Keep metadata separate from module mutation so a failed
                # validation cannot leave a partially installed prefix.
                installed.append(
                    installed_item
                )

    expected = int(INT8_ENCODER_POLICY["quantized_conv_count"])
    if len(installed) != expected:
        raise RuntimeError(
            f"INT8 encoder policy installed {len(installed)} convolutions, "
            f"expected {expected}"
        )
    for block, field, wrapper in pending:
        setattr(block, f"int8_{field}", wrapper)
    return {
        "enabled": True,
        "count": len(installed),
        "policy": dict(INT8_ENCODER_POLICY),
        "tile_variant": "stage-aware-input-channels",
        "tile_variants": sorted({item["tile_variant"] for item in installed}),
        "convolutions": installed,
    }


__all__ = [
    "INT8_ENCODER_POLICY",
    "Int8FusedValidConv3d",
    "install_int8_encoder_prefix",
    "int8_valid_conv3d_integration",
]
