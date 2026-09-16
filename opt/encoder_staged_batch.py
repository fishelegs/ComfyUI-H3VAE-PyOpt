"""Opt-in staged MiniMax-H3 encoder batching; preserve independent clips."""
import torch
import torch.nn.functional as F


class EncoderPrefix(torch.nn.Module):
    def __init__(self, encoder, cut):
        super().__init__()
        if cut not in (1, 2):
            raise ValueError('Only validated split locations 1 and 2 supported')
        if encoder.gradient_checkpointing or encoder.use_fused_norm:
            raise ValueError('Expected inference encoder without fused-norm flag')
        self.conv_in=encoder.conv_in
        self.stages=torch.nn.ModuleList(list(encoder.down[:cut]))

    def forward(self,x):
        h=self.conv_in(x.contiguous(memory_format=torch.channels_last_3d))
        for stage in self.stages:
            for block in stage.block:h=block(h)
            if hasattr(stage,'downsample'):h=stage.downsample(h)
        return h


class EncoderSuffix(torch.nn.Module):
    def __init__(self,encoder,quant_conv,cut):
        super().__init__()
        self.stages=torch.nn.ModuleList(list(encoder.down[cut:]))
        self.norm_out=encoder.norm_out
        self.conv_out=encoder.conv_out
        self.quant_conv=quant_conv

    def forward(self,h):
        for stage in self.stages:
            for block in stage.block:h=block(h)
            if hasattr(stage,'downsample'):h=stage.downsample(h)
        return self.quant_conv(self.conv_out(F.silu(self.norm_out(h))))


def encode_staged(normalized_video,prefix,suffix,*,clip_length,token_drop,batch_size,cut):
    if normalized_video.ndim!=5 or normalized_video.shape[0]!=1:
        raise ValueError('Expected batch-one video')
    if (cut,batch_size) not in [(1,1),(1,2),(2,1),(2,2),(2,4)]:
        raise ValueError('Unsupported combination; avoid large int32-indexed tensors')
    x=normalized_video
    pad=(-x.shape[2])%clip_length
    if pad:x=torch.cat([x,x[:,:,-1:].repeat(1,1,pad,1,1)],dim=2)
    x=x.contiguous(memory_format=torch.channels_last_3d)
    outputs=[]
    for start in range(0,x.shape[2],clip_length*batch_size):
        end=min(start+clip_length*batch_size,x.shape[2])
        features=[prefix(x[:,:,t:t+clip_length]) for t in range(start,end,clip_length)]
        batch=(features[0] if len(features)==1 else torch.cat(features,dim=0)).contiguous(memory_format=torch.channels_last_3d)
        y=suffix(batch)
        outputs.extend(y[i:i+1] for i in range(len(features)))
        del features,batch,y
    out=torch.cat(outputs,dim=2)
    return out[:,:,:-token_drop] if token_drop else out
