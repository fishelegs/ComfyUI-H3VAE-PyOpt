"""Opt-in, inference-only QK RMSNorm/RoPE fusion; no model source edits.

Keep PyTorch Linear and Flash SDPA. Stats are FP32; normalized Q/K and
each FP16 rotary product retain the eager rounding boundaries.
"""
import copy
import os
import torch
import triton
import triton.language as tl

@triton.jit
def _qk_rope(X,C,S,Q,K,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,R:tl.constexpr,
             EPS:tl.constexpr,ROWS:tl.constexpr):
    row=tl.program_id(0)*ROWS+tl.arange(0,ROWS)
    d=tl.arange(0,D)
    token=row//H
    offsets=row[:,None]*(3*D)+d[None,:]
    q=tl.load(X+offsets,row[:,None]<N*H,other=0).to(tl.float32)
    k=tl.load(X+offsets+D,row[:,None]<N*H,other=0).to(tl.float32)
    q=q*tl.rsqrt(tl.sum(q*q,1)[:,None]/D+EPS)
    k=k*tl.rsqrt(tl.sum(k*k,1)[:,None]/D+EPS)
    q=q.to(tl.float16).to(tl.float32)
    k=k.to(tl.float16).to(tl.float32)
    rd=tl.where(d<R,(d+R//2)%R,d)
    idx=tl.broadcast_to(rd[None,:],(ROWS,D))
    qr=tl.gather(q,idx,1)*tl.where(d[None,:]<R//2,-1.,1.)
    kr=tl.gather(k,idx,1)*tl.where(d[None,:]<R//2,-1.,1.)
    mask=(row[:,None]<N*H)&(d[None,:]<R)
    c=tl.load(C+token[:,None]*R+d[None,:],mask,other=1).to(tl.float32)
    s=tl.load(S+token[:,None]*R+d[None,:],mask,other=0).to(tl.float32)
    qo=(q*c).to(tl.float16).to(tl.float32)+(qr*s).to(tl.float16).to(tl.float32)
    ko=(k*c).to(tl.float16).to(tl.float32)+(kr*s).to(tl.float16).to(tl.float32)
    tl.store(Q+row[:,None]*D+d[None,:],tl.where(d[None,:]<R,qo,q),row[:,None]<N*H)
    tl.store(K+row[:,None]*D+d[None,:],tl.where(d[None,:]<R,ko,k),row[:,None]<N*H)

@torch.library.custom_op('h3vae_decoder::qk_rope',mutates_args=(),device_types='cuda')
def qk_rope(x:torch.Tensor,cos:torch.Tensor,sin:torch.Tensor,heads:int,eps:float,rows:int)->tuple[torch.Tensor,torch.Tensor]:
    if rows not in (4,8,16,32):
        raise ValueError('rows must be one of the tested configurations: 4, 8, 16, 32')
    b,n,_=x.shape
    d=x.shape[-1]//(3*heads)
    assert x.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()
    assert x.dtype==torch.float16 and cos.dtype==torch.float16 and sin.dtype==torch.float16
    assert d==64 and cos.shape==(b,n,1,48) and sin.shape==cos.shape
    q=torch.empty((b,n,heads,d),device=x.device,dtype=x.dtype)
    k=torch.empty_like(q)
    _qk_rope[(triton.cdiv(b*n*heads,rows),)](x,cos,sin,q,k,b*n,heads,d,cos.shape[-1],eps,rows,num_warps=4,enable_fp_fusion=False)
    return q,k

@qk_rope.register_fake
def _fake(x,cos,sin,heads,eps,rows):
    shape=(*x.shape[:2],heads,x.shape[-1]//(3*heads))
    return x.new_empty(shape),x.new_empty(shape)

class FusedQKAttention(torch.nn.Module):
    def __init__(self,original,rows=8):
        super().__init__()
        if original.spatial_parallel or not isinstance(original.norm_q,torch.nn.RMSNorm) or not isinstance(original.norm_k,torch.nn.RMSNorm):
            raise ValueError('single-device RMSNorm attention required')
        if original.norm_q.weight is not None or original.norm_k.weight is not None or original.dim_head!=64 or original.norm_q.eps!=original.norm_k.eps:
            raise ValueError('expected head=64 and unaffined equal-eps QK norms')
        self.original=original
        self.rows=rows
    def forward(self,x,rotary_pos_emb=None,pack_info={}):
        if rotary_pos_emb is None:
            return self.original(x,rotary_pos_emb,pack_info)
        b,n,_=x.shape
        a=self.original
        qkv=a.to_qkv(x)
        q,k=qk_rope(qkv,*rotary_pos_emb,a.heads,a.norm_q.eps,self.rows)
        v=qkv.view(b,n,a.heads,3*a.dim_head)[...,2*a.dim_head:]
        y=a.perform_attention(q,k,v,pack_info)
        return a.to_out(y.reshape(b,n,-1))

def clone_with_qk_fusion(raw,rows=8):
    """Copy module containers, share read-only weights; original stays intact."""
    if raw.training or os.environ.get('MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM','1').lower() in ('0','false','no','off'):
        raise ValueError('QK fusion requires eval inference with FP32 norm enabled')
    decoder=copy.copy(raw)
    decoder._modules=raw._modules.copy()
    blocks=[]
    for block in raw.transformer_blocks:
        new=copy.copy(block)
        new._modules=block._modules.copy()
        new.attn=FusedQKAttention(block.attn,rows).eval()
        blocks.append(new)
    decoder.transformer_blocks=torch.nn.ModuleList(blocks)
    return decoder


def enable_qk_rope_decoder(model,rows=8,mode="max-autotune-no-cudagraphs"):
    """Opt-in installer; call before warmup, then infer under torch.no_grad().

    Returns the previous decoder for restoration. Compatible with the
    existing optional paired-tile scheduler; no overlap/scheduler changes.
    Do not pass the separate StaticCUDAGraphDecoder wrapper.
    """
    from pytorch_decoder_optim import enable_decoder_attention_in_graph
    core=getattr(model,'model',model)
    previous=core.decoder
    raw=getattr(previous,'_orig_mod',previous)
    if not hasattr(raw,'transformer_blocks'):
        raise ValueError('expected raw or whole-compiled ViT decoder')
    enable_decoder_attention_in_graph(raw)
    core.decoder=torch.compile(clone_with_qk_fusion(raw,rows),mode=mode,dynamic=False,fullgraph=False)
    return previous
