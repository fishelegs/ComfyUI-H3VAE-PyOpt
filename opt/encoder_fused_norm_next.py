"""Opt-in follow-up candidates; originals and prior winner stay unchanged."""
import torch
from encoder_staged_batch import EncoderPrefix
from encoder_fused_norm_blocks import FusedNormResidualBlock


def next_fused_prefix(encoder, variant='extend'):
    if variant not in ('extend', 'bias', 'extend_bias', 'extend_bias_pack'):
        raise ValueError(variant)
    prefix = EncoderPrefix(encoder, 2)
    stages = (0, 1) if variant.startswith('extend') else (0,)
    for idx in stages:
        stage = torch.nn.Module()
        if 'bias' in variant:
            from encoder_fused_norm_bias import BiasFusedResidualBlock
            block_type = BiasFusedResidualBlock
        else:
            block_type = lambda b: FusedNormResidualBlock(b, opaque=True)
        stage.block = torch.nn.ModuleList([block_type(b) for b in encoder.down[idx].block])
        if hasattr(encoder.down[idx], 'downsample'):
            stage.downsample = encoder.down[idx].downsample
        prefix.stages[idx] = stage
    if variant == 'extend_bias_pack':
        from encoder_fused_residual_pack import ResidualDownsampleBlock
        prefix.stages[1].block[-1] = ResidualDownsampleBlock(encoder.down[1].block[-1], encoder.down[1].downsample)
        del prefix.stages[1].downsample
    return prefix
