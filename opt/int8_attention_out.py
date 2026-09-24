"""Opt-in INT8 attention output projections after QKV INT8.

Container cloning shares immutable tensors, not module registration dictionaries.
Only the returned decoder is intended for independent device offload/reload.
"""
import copy
import torch
from torch import nn
from .int8_qkv import selected_indices, clone_qkv_int8_decoder
from .kitchen_int8 import KitchenInt8Linear


def clone_containers(module, memo=None):
    """Preserve module aliases and share read-only parameters/buffers."""
    if memo is None:
        memo = {}
    if id(module) in memo:
        return memo[id(module)]
    result = copy.copy(module)
    memo[id(module)] = result
    result._parameters = module._parameters.copy()
    result._buffers = module._buffers.copy()
    result._modules = {name: None if child is None else clone_containers(child, memo)
                       for name, child in module._modules.items()}
    return result


def out_linear(block):
    attention = getattr(block.attn, 'original', block.attn)
    source = getattr(attention, 'to_out', None)
    if not isinstance(source, nn.Linear):
        raise ValueError('expected original nn.Linear output projection')
    if (source.in_features, source.out_features) != (2048, 2048):
        raise ValueError('output projection experiment requires K2048/N2048')
    if source.training:
        raise ValueError('output projection inference experiment requires eval()')
    return attention, source


def clone_attention_out_decoder(baseline, *, indices=None):
    chosen = selected_indices(len(baseline.transformer_blocks), indices)
    for i in chosen:
        out_linear(baseline.transformer_blocks[i])
    result = clone_containers(baseline)
    for i in chosen:
        attention, source = out_linear(result.transformer_blocks[i])
        attention.to_out = KitchenInt8Linear(source, require_cuda=True).eval()
    return result


def from_raw_decoder(raw, *, warps=8):
    return clone_attention_out_decoder(clone_qkv_int8_decoder(raw, warps=warps))


def out_inventory(decoder):
    rows = []
    for i, block in enumerate(decoder.transformer_blocks):
        attention = getattr(block.attn, 'original', block.attn)
        linear = attention.to_out
        if isinstance(linear, KitchenInt8Linear):
            if linear.qweight.dtype != torch.int8 or linear.weight_scale.dtype != torch.float32:
                raise RuntimeError('output projection quantized state has invalid dtype')
            rows.append({'block': i, 'shape': list(linear.qweight.shape),
                         'qweight_dtype': str(linear.qweight.dtype),
                         'scale_dtype': str(linear.weight_scale.dtype),
                         'device': str(linear.qweight.device)})
    return rows
