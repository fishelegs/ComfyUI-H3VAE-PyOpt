"""Opt-in CK INT8 linear/FFN implementation for the H3 decoder.

The candidate uses comfy-kitchen 0.2.34's row-wise/per-output INT8 contract:
weights are quantized once per output row, while ``int8_linear`` dynamically
quantizes each activation row on every call.  The original FP16 Parameters
remain untouched and are not registered as children of the replacement.

The runtime imports this module only for the explicit INT8 decoder option.
It does not enable input-act fusion, ConvRot or residual epilogues, and does
not alter the default path. The CUDA implementation is inference-only; CPU
tests may inject a small fake CK module to exercise the container and scale
validation without pretending to validate CUDA kernels.
"""
from __future__ import annotations

import torch
from torch import nn


@torch.library.custom_op("h3vae_pyopt::int8_linear", mutates_args=())
def _int8_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Keep CK's Python registry dispatch outside the Dynamo graph."""
    from comfy_kitchen import int8_linear

    return int8_linear(
        x,
        qweight,
        weight_scale,
        bias,
        out_dtype=x.dtype,
        convrot=False,
    )


@_int8_linear.register_fake
def _(x, qweight, weight_scale, bias):
    del weight_scale, bias
    return x.new_empty((*x.shape[:-1], qweight.shape[0]))


def _require_int8_api() -> None:
    try:
        import comfy_kitchen as ck
    except ImportError as exc:
        raise RuntimeError(
            "kitchen INT8 experiment requires comfy-kitchen 0.2.34"
        ) from exc
    if not callable(getattr(ck, "quantize_int8_rowwise", None)):
        raise RuntimeError(
            "kitchen INT8 experiment requires quantize_int8_rowwise"
        )
    if not callable(getattr(ck, "int8_linear", None)):
        raise RuntimeError("kitchen INT8 experiment requires int8_linear")


def _validate_source_linear(original: nn.Module, *, require_cuda: bool) -> nn.Linear:
    if not isinstance(original, nn.Linear):
        raise ValueError("kitchen INT8 replacement requires nn.Linear")
    weight = original.weight
    if weight.ndim != 2:
        raise ValueError("kitchen INT8 weight must be rank 2")
    if not bool(torch.isfinite(weight.detach()).all()):
        raise ValueError("kitchen INT8 rejects non-finite source weights")
    if original.bias is not None:
        bias = original.bias
        if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
            raise ValueError("kitchen INT8 bias shape does not match weight")
        if not bool(torch.isfinite(bias.detach()).all()):
            raise ValueError("kitchen INT8 rejects non-finite source bias")
    if require_cuda:
        if weight.device.type != "cuda" or weight.dtype != torch.float16:
            raise ValueError("kitchen INT8 decoder requires CUDA FP16 weights")
        if torch.version.hip is not None:
            raise ValueError("kitchen INT8 decoder requires a CUDA PyTorch build")
        if original.bias is not None and (
            original.bias.device != weight.device
            or original.bias.dtype != torch.float16
        ):
            raise ValueError("kitchen INT8 bias must be CUDA FP16")
    return original


def quantize_static_weight(
    original: nn.Linear,
    *,
    quantize_fn=None,
    require_cuda: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one source weight and validate CK's static rowwise contract.

    ``quantize_fn`` is only for CPU tests or an explicitly injected experiment
    implementation.  The real path resolves CK's exact public API and uses
    deterministic rounding (``stochastic_rounding=0``).
    """
    _validate_source_linear(original, require_cuda=require_cuda)
    if quantize_fn is None:
        _require_int8_api()
        import comfy_kitchen as ck

        quantize_fn = ck.quantize_int8_rowwise
    qweight, weight_scale = quantize_fn(
        original.weight.detach(), stochastic_rounding=0
    )
    expected_shape = tuple(original.weight.shape)
    if tuple(qweight.shape) != expected_shape or qweight.dtype != torch.int8:
        raise ValueError(
            "INT8 quantizer returned unexpected weight: "
            f"shape={tuple(qweight.shape)}, dtype={qweight.dtype}, "
            f"expected shape={expected_shape}, dtype=torch.int8"
        )
    if qweight.device != original.weight.device:
        raise ValueError("INT8 quantized weight changed device")
    qweight = qweight.contiguous()
    scale = weight_scale.reshape(-1).contiguous()
    if scale.numel() != original.weight.shape[0]:
        raise ValueError(
            "INT8 weight scale must have one value per output row, got "
            f"{scale.numel()} for N={original.weight.shape[0]}"
        )
    if scale.dtype != torch.float32:
        raise ValueError(
            f"INT8 weight scale must be float32, got {scale.dtype}"
        )
    if scale.device != original.weight.device:
        raise ValueError("INT8 weight scale changed device")
    # CK 0.2.34's CUDA quantizer currently emits -128 for an exactly zero
    # source row (with a tiny positive scale).  Normalize only that safe case;
    # an invalid scale on any nonzero row is a hard failure, never a silent
    # fallback that could hide corrupt weights.
    zero_rows = torch.all(original.weight.detach() == 0, dim=1)
    invalid_scales = ~torch.isfinite(scale) | (scale <= 0)
    invalid_rows = invalid_scales & ~zero_rows
    if bool(invalid_rows.any()):
        raise ValueError("INT8 weight scales must be finite and strictly positive")
    if bool(zero_rows.any()):
        qweight = qweight.clone()
        scale = scale.clone()
        qweight[zero_rows] = 0
        scale[zero_rows] = 1.0
    return qweight, scale


class KitchenInt8Linear(nn.Module):
    """Inference-only linear with static rowwise INT8 weights."""

    def __init__(
        self,
        original: nn.Linear,
        *,
        quantize_fn=None,
        require_cuda: bool = False,
    ):
        super().__init__()
        _require_int8_api() if quantize_fn is None else None
        source = _validate_source_linear(original, require_cuda=require_cuda)
        qweight, weight_scale = quantize_static_weight(
            source,
            quantize_fn=quantize_fn,
            require_cuda=require_cuda,
        )
        # Keep both buffers in state_dict so ComfyUI's module_size and
        # ModelPatcher account for the additional INT8 storage.  They are
        # generated from the source checkpoint at runtime and never mutate
        # that checkpoint.
        self.register_buffer("qweight", qweight)
        self.register_buffer("weight_scale", weight_scale)
        self.bias = source.bias
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.source_dtype = source.weight.dtype
        self.source_device = source.weight.device
        self.extra_weight_bytes = (
            qweight.numel() * qweight.element_size()
            + weight_scale.numel() * weight_scale.element_size()
        )
        self.train(source.training)

    def _apply(self, fn, recurse=True):
        """Move the candidate without narrowing CK's FP32 weight scales.

        Module.to(dtype=...) applies the requested dtype to floating buffers.
        CK's INT8 GEMM contract deliberately requires one FP32 scale per
        output row, so a VAE offload/dtype move must not turn weight_scale
        into FP16/BF16 (or round it and cast it back).  The int8 weights
        still follow normal device moves, and parameters such as the source
        bias are handled by the regular Module implementation.
        """
        original_scale = self.weight_scale
        result = super()._apply(fn, recurse=recurse)
        self.weight_scale = original_scale.to(
            device=self.qweight.device, dtype=torch.float32
        )
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError("kitchen INT8 candidate is inference-only; call eval()")
        if torch.is_grad_enabled():
            raise RuntimeError("kitchen INT8 candidate requires torch.no_grad()")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"INT8 input K={x.shape[-1]} does not match K={self.in_features}"
            )
        if x.dtype != self.source_dtype:
            raise ValueError(
                "INT8 input dtype must match source dtype "
                f"{self.source_dtype}, got {x.dtype}"
            )
        if x.device != self.qweight.device:
            raise ValueError("INT8 input and quantized weight must share a device")
        # Do not check isfinite here: activation quantization is part of the
        # timed CK call.  Benchmarks validate captured activations and outputs
        # outside the timed region so NaNs cannot be hidden by quantization.
        return _int8_linear(x, self.qweight, self.weight_scale, self.bias)


class KitchenInt8FFN(nn.Module):
    """Source-shaped gated SiLU FFN with selectable INT8 GEMMs."""

    MODES = frozenset(("INT8both", "INT8w1", "INT8w2"))

    def __init__(
        self,
        original: nn.Module,
        mode: str = "INT8both",
        *,
        quantize_fn=None,
        require_cuda: bool = False,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unsupported INT8 FFN mode: {mode}")
        if not bool(getattr(original, "use_gated", False)):
            raise ValueError("kitchen INT8 experiment requires a gated FFN")
        if not isinstance(getattr(original, "act_fn", None), nn.SiLU):
            raise ValueError("kitchen INT8 experiment requires a SiLU FFN")
        if not isinstance(getattr(original, "w1", None), nn.Linear):
            raise ValueError("kitchen INT8 FFN requires an nn.Linear w1")
        if not isinstance(getattr(original, "w2", None), nn.Linear):
            raise ValueError("kitchen INT8 FFN requires an nn.Linear w2")
        self.mode = mode
        self.use_gated = True
        self.act_fn = original.act_fn
        self.w1 = (
            KitchenInt8Linear(
                original.w1,
                quantize_fn=quantize_fn,
                require_cuda=require_cuda,
            )
            if mode in ("INT8both", "INT8w1")
            else original.w1
        )
        self.w2 = (
            KitchenInt8Linear(
                original.w2,
                quantize_fn=quantize_fn,
                require_cuda=require_cuda,
            )
            if mode in ("INT8both", "INT8w2")
            else original.w2
        )
        self.train(original.training)

    @property
    def extra_weight_bytes(self) -> int:
        return sum(
            getattr(module, "extra_weight_bytes", 0)
            for module in (self.w1, self.w2)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ff_hidden_states = self.w1(hidden_states)
        gate, up = ff_hidden_states.chunk(2, dim=-1)
        ff_hidden_states = self.act_fn(gate) * up
        return self.w2(ff_hidden_states)


def install_kitchen_int8_ffn(
    decoder: nn.Module,
    mode: str = "INT8both",
    *,
    require_cuda: bool = True,
) -> int:
    """Opt-in replace all H3 decoder FFNs; caller owns clone isolation.

    Validation and quantization finish for every block before any ``block.ff``
    is assigned, so an unsupported block leaves the caller's decoder unchanged.
    """
    if decoder.training:
        raise ValueError("kitchen INT8 candidate requires decoder.eval()")
    blocks = getattr(decoder, "transformer_blocks", None)
    if blocks is None:
        raise ValueError("decoder has no transformer_blocks")
    _require_int8_api()
    replacements = [
        KitchenInt8FFN(
            block.ff,
            mode,
            require_cuda=require_cuda,
        )
        for block in blocks
    ]
    for block, replacement in zip(blocks, replacements):
        block.ff = replacement
    decoder._h3vae_kitchen_int8_ffn_mode = mode
    return len(replacements)


__all__ = [
    "KitchenInt8FFN",
    "KitchenInt8Linear",
    "install_kitchen_int8_ffn",
    "quantize_static_weight",
]
