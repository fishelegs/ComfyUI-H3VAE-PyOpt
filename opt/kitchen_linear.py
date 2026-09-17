"""Opt-in comfy-kitchen FP16 linear path for the compiled H3 decoder.

The custom op keeps comfy-kitchen's Python CUDA-stream lookup out of Dynamo
tracing, while preserving the reference linear weights and module names.
"""
from __future__ import annotations

import torch
from torch import nn


@torch.library.custom_op("h3vae_pyopt::fp16_linear", mutates_args=())
def _fp16_linear(x: torch.Tensor, weight: torch.Tensor,
                 bias: torch.Tensor | None) -> torch.Tensor:
    from comfy_kitchen import fp16_linear
    return fp16_linear(x, weight, bias)


@_fp16_linear.register_fake
def _(x, weight, bias):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


class KitchenLinear(nn.Module):
    def __init__(self, original: nn.Linear):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias

    def forward(self, x):
        return _fp16_linear(x, self.weight, self.bias)


def install_kitchen_linears(decoder: nn.Module) -> int:
    """Replace only decoder linears; requires CUDA FP16 and CK 0.2.34 API."""
    try:
        import comfy_kitchen as ck
    except ImportError as exc:
        raise RuntimeError("fast_linear requires comfy-kitchen >= 0.2.34") from exc
    if not hasattr(ck, "fp16_linear"):
        raise RuntimeError("fast_linear requires comfy-kitchen >= 0.2.34")

    def replace(module):
        count = 0
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                if child.weight.device.type != "cuda" or child.weight.dtype != torch.float16:
                    raise ValueError("fast_linear requires a CUDA FP16 decoder")
                setattr(module, name, KitchenLinear(child))
                count += 1
            else:
                count += replace(child)
        return count

    return replace(decoder)
