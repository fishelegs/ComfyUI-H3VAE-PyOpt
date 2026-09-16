"""Experimental FP16 temporal GroupNorm + SiLU + reflection/causal padding.

Three Triton kernels: stable local moments, merge moments, normalize/pack.
Only batch-one, 32 groups, causal left=2, spatial reflect=1 are supported.
No convolution mainloop fusion; convolution keeps the original padded shape.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _partial(X,P,Q,C:tl.constexpr,H:tl.constexpr,W:tl.constexpr,
             SD:tl.constexpr,SC:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
             G:tl.constexpr,NP:tl.constexpr,BS:tl.constexpr,BC:tl.constexpr):
    part=tl.program_id(0);cb=tl.program_id(1);t=tl.program_id(2)
    s=part*BS+tl.arange(0,BS)
    c=cb*BC+tl.arange(0,BC)
    cg:tl.constexpr=C//G
    bg:tl.constexpr=BC//cg
    off=t*SD+(s[:,None]//W)*SH+(s[:,None]%W)*SW+c[None,:]*SC
    valid=(s[:,None]<H*W)&(c[None,:]<C)
    v=tl.load(X+off,valid,other=0).to(tl.float32)
    v=tl.reshape(v,(BS,bg,cg))
    count=tl.minimum(BS,H*W-part*BS)*cg
    avg=tl.div_rn(tl.sum(tl.sum(v,2),0),count.to(tl.float32))
    centered=v-avg[None,:,None]
    mask=tl.reshape(valid,(BS,bg,cg))
    m2=tl.sum(tl.sum(tl.where(mask,centered*centered,0.),2),0)
    g=cb*bg+tl.arange(0,bg)
    dest=(t*NP+part)*G+g
    tl.store(P+dest,avg,g<G);tl.store(Q+dest,m2,g<G)

@triton.jit
def _merge(P,Q,S,H:tl.constexpr,W:tl.constexpr,C:tl.constexpr,G:tl.constexpr,
           NP:tl.constexpr,BS:tl.constexpr,EPS:tl.constexpr,BP:tl.constexpr):
    g=tl.program_id(0);t=tl.program_id(1)
    p=tl.arange(0,BP)
    count=tl.minimum(BS,tl.maximum(0,H*W-p*BS))*(C//G)
    avg=tl.load(P+(t*NP+p)*G+g,p<NP,other=0)
    m2=tl.load(Q+(t*NP+p)*G+g,p<NP,other=0)
    total:tl.constexpr=H*W*(C//G)
    mean=tl.div_rn(tl.sum(avg*count,0),total*1.0)
    delta=avg-mean
    var=tl.div_rn(tl.sum(m2+count*delta*delta,0),total*1.0)
    tl.store(S+(t*G+g)*2,mean)
    tl.store(S+(t*G+g)*2+1,tl.rsqrt(tl.maximum(var,0.)+EPS))

@triton.jit
def _normalize_pack(X,Weight,Bias,S,Y,C:tl.constexpr,D:tl.constexpr,H:tl.constexpr,W:tl.constexpr,
                    SD:tl.constexpr,SC:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
                    G:tl.constexpr,BLOCK:tl.constexpr):
    idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    total:tl.constexpr=C*(D+2)*(H+2)*(W+2)
    c=idx%C;q=idx//C
    t=q//((H+2)*(W+2))-2
    h=q//(W+2)%(H+2)-1;w=q%(W+2)-1
    h=tl.where(h<0,-h,tl.where(h>=H,2*H-2-h,h))
    w=tl.where(w<0,-w,tl.where(w>=W,2*W-2-w,w))
    valid=(idx<total)&(t>=0)&(t<D)
    v=tl.load(X+t*SD+c*SC+h*SH+w*SW,valid,other=0).to(tl.float32)
    stat=(t*G+c//(C//G))*2
    mean=tl.load(S+stat,valid,other=0)
    rstd=tl.load(S+stat+1,valid,other=0)
    gamma=tl.load(Weight+c).to(tl.float32);beta=tl.load(Bias+c).to(tl.float32)
    z=(((v-mean)*rstd)*gamma+beta).to(tl.float16).to(tl.float32)
    z=(z/(1.+tl.exp(-z))).to(tl.float16)
    tl.store(Y+idx,tl.where(valid,z,0.),idx<total)

def fused_temporal_norm_pad(x,weight,bias,eps,config=(128,32,4)):
    b,c,d,h,w=x.shape
    if b!=1 or x.dtype!=torch.float16 or c not in (128,256,512,1024) or min(h,w)<2:
        raise ValueError('Only tested batch-one FP16 channel counts and H/W >=2 supported')
    if c*(d+2)*(h+2)*(w+2)>=2**31:
        raise ValueError('Padded output exceeds validated int32 index range')
    bs,bc,warps=config
    if bc%(c//32):raise ValueError('Channel tile must contain complete groups')
    nparts=triton.cdiv(h*w,bs)
    p=torch.empty((d,nparts,32),device=x.device,dtype=torch.float32)
    q=torch.empty_like(p)
    stats=torch.empty((d,32,2),device=x.device,dtype=torch.float32)
    y=torch.empty((b,c,d+2,h+2,w+2),device=x.device,dtype=x.dtype,memory_format=torch.channels_last_3d)
    _partial[(nparts,triton.cdiv(c,bc),d)](x,p,q,c,h,w,x.stride(2),x.stride(1),x.stride(3),x.stride(4),32,nparts,bs,bc,num_warps=warps)
    _merge[(32,d)](p,q,stats,h,w,c,32,nparts,bs,eps,triton.next_power_of_2(nparts),num_warps=4)
    _normalize_pack[(triton.cdiv(y.numel(),1024),)](x,weight,bias,stats,y,c,d,h,w,x.stride(2),x.stride(1),x.stride(3),x.stride(4),32,1024,num_warps=4,enable_fp_fusion=False)
    return y

class FusedNormPadConv(torch.nn.Module):
    def __init__(self,norm,conv,config=(128,32,4),with_conv=True):
        super().__init__()
        if type(norm).__name__!='TemporalIsolatedSpatialParallelGroupNorm' or norm.num_groups!=32:
            raise ValueError('Expected temporal-isolated GroupNorm with 32 groups')
        if conv.padding!=(1,1,1) or conv.kernel_size!=(3,3,3) or not conv.causal or conv.pad_mode!='reflect' or conv.pad_mode_t!='constant' or conv.spatial_parallel:
            raise ValueError('Expected single-GPU causal 3x3x3 reflect-padded conv')
        self.norm_weight=norm.weight;self.norm_bias=norm.bias;self.eps=norm.eps
        self.weight=conv.weight;self.bias=conv.bias;self.stride=conv.stride
        self.config=config;self.with_conv=with_conv
    def forward(self,x):
        y=fused_temporal_norm_pad(x,self.norm_weight,self.norm_bias,self.eps,self.config)
        return F.conv3d(y,self.weight,self.bias,stride=self.stride) if self.with_conv else y
