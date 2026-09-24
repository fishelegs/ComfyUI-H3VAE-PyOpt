"""Opt-in INT8 QKV on the fused FFN decoder.

Attention, QK norm/RoPE, output projections and encoder are unchanged.
This changes precision, not just scheduling/fusion. Original weights stay intact.
"""
import torch
from torch import nn
from .int8_norm_quant import clone_norm_quant_decoder
from .kitchen_int8 import KitchenInt8Linear


def qkv_linear(block):
    attention = getattr(block.attn, 'original', block.attn)
    source = getattr(attention, 'to_qkv', None)
    if not isinstance(source, nn.Linear):
        raise ValueError('expected original nn.Linear QKV projection')
    if (source.in_features, source.out_features) != (2048, 6144):
        raise ValueError('QKV experiment requires K2048/N6144')
    if source.training:
        raise ValueError('QKV inference experiment requires eval()')
    return attention, source


def selected_indices(count, indices=None):
    result = tuple(range(count)) if indices is None else tuple(indices)
    if not result or len(set(result)) != len(result):
        raise ValueError('nonempty unique block indices required')
    if any(type(i) is not int or i < 0 or i >= count for i in result):
        raise ValueError('block index out of range or not an integer')
    return result


def clone_qkv_int8_decoder(raw, *, indices=None, warps=8):
    chosen = selected_indices(len(raw.transformer_blocks), indices)
    for i in chosen:
        qkv_linear(raw.transformer_blocks[i])
    result = clone_norm_quant_decoder(raw, warps)
    for i in chosen:
        attention, source = qkv_linear(result.transformer_blocks[i])
        attention.to_qkv = KitchenInt8Linear(source, require_cuda=True).eval()
    return result


def qkv_inventory(decoder):
    rows = []
    for i, block in enumerate(decoder.transformer_blocks):
        attention = getattr(block.attn, 'original', block.attn)
        linear = attention.to_qkv
        if isinstance(linear, KitchenInt8Linear):
            if linear.qweight.dtype != torch.int8 or linear.weight_scale.dtype != torch.float32:
                raise RuntimeError('QKV quantized state has invalid dtype')
            rows.append({'block': i, 'shape': list(linear.qweight.shape),
                         'qweight_dtype': str(linear.qweight.dtype),
                         'scale_dtype': str(linear.weight_scale.dtype),
                         'device': str(linear.qweight.device)})
    return rows
