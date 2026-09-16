"""Opt-in fused GroupNorm for the high-resolution residual blocks only."""
import torch
from encoder_fused_temporal_norm import FusedNormPadConv
from encoder_staged_batch import EncoderPrefix
from encoder_fused_norm_opaque import OpaqueFusedNormPadConv

class FusedNormResidualBlock(torch.nn.Module):
    def __init__(self,block,config=(128,64,8),opaque=False):
        super().__init__()
        if block.use_fused_norm:raise ValueError('Unexpected pre-existing fused norm wrapper')
        cls=OpaqueFusedNormPadConv if opaque else FusedNormPadConv
        self.first=cls(block.norm1,block.conv1,config)
        self.second=cls(block.norm2,block.conv2,config)
        self.shortcut=getattr(block,'nin_shortcut',None)
    def forward(self,x,zq=None):
        if zq is not None:raise ValueError('Unconditional encoder only')
        h=self.second(self.first(x))
        return (self.shortcut(x) if self.shortcut is not None else x)+h

def fused_norm_prefix(encoder,cut=2,config=(128,64,8),opaque=False):
    prefix=EncoderPrefix(encoder,cut)
    # New containers, shared original parameter objects. Do not mutate encoder.
    stage=torch.nn.Module()
    stage.block=torch.nn.ModuleList([FusedNormResidualBlock(b,config,opaque) for b in encoder.down[0].block])
    if hasattr(encoder.down[0],'downsample'):stage.downsample=encoder.down[0].downsample
    prefix.stages[0]=stage
    return prefix
