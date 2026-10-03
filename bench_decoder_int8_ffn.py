"""Same-graph SM120 INT8 decoder FFN-up A/B and real-video exactness checks.

Baseline invokes the original two registered custom ops. Candidate uses the
production fused FFN. Only the new opaque op's private eager implementation
switches; compiled graph, static weights, input, tile and batch stay fixed.
Media/model loading, FP16 encoding, compilation and validation are not timed.
"""
from __future__ import annotations

import argparse
import collections
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
from unittest.mock import patch

import torch
import bench_int8_vae as base
from h3vae_runtime import H3VAEPyOptRuntime, DECODER_COMPILE_MODE
from opt import int8_ffn_up_fused as fused
from opt import int8_norm_quant as norm
from opt.int8_attention_out import clone_containers
from opt.int8_swiglu_fused import prequantized_linear, quantize_swiglu, swiglu_int8_linear
from scripts.benchmark_reduction_control import ReferenceReductionBindings

REPO = Path(__file__).resolve().parent


@contextmanager
def _record_reduction_schedules():
    """Read-only evidence of actual normalization schedules outside timing."""
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    original_run = CachingAutotuner.run
    schedules = {}

    def observed(self, *args, **kwargs):
        value = original_run(self, *args, **kwargs)
        name = self.inductor_meta.get('kernel_name', '')
        if ('rsqrt' in name or 'native_layer_norm' in name) and self.launchers:
            config = self.launchers[0].config
            key = hashlib.sha256(self.fn.src.encode()).hexdigest()
            row = {'kernel': name, 'source_sha256': key,
                   'config': dict(config.kwargs), 'num_warps': config.num_warps,
                   'num_stages': config.num_stages,
                   'found_by_coordesc': bool(getattr(config, 'found_by_coordesc', False))}
            if key in schedules and schedules[key] != row:
                raise RuntimeError('Reduction schedule changed during validation')
            schedules[key] = row
        return value

    with patch.object(CachingAutotuner, 'run', observed):
        yield schedules


def _controlled_schedules(control):
    return {key: {'kernel': entry['kernel'].inductor_meta['kernel_name'],
                  'source_sha256': key, 'config': entry['schedule']['kwargs'],
                  'num_warps': entry['schedule']['num_warps'],
                  'num_stages': entry['schedule']['num_stages'],
                  'found_by_coordesc': True}
            for key, entry in control.reference.items()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True,help='primary timed video')
    parser.add_argument('--videos-dir',type=Path,help='additional MP4s for exactness only')
    parser.add_argument('--model-code-dir',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--blocks',type=int,default=3)
    parser.add_argument('--seed',type=int,default=20261002)
    parser.add_argument('--tile-batch',type=int,default=4)
    parser.add_argument('--reuse-reference-reductions',action='store_true',
                        help='explicitly reuse same-source production norm autotuners in both A/B variants')
    args = parser.parse_args(argv)
    if args.warmup < 2 or args.blocks < 1 or args.tile_batch < 1:
        parser.error('warmup >= 2, blocks >= 1 and tile-batch >= 1 required')
    if args.output.exists():
        parser.error('choose a fresh output path')
    return args


def _canonical_baseline(x,a,g,nw,uw,us,ub,dw,ds,db,eps,warps):
    # Preserve registered-op dispatch and temporary lifetimes, as in production.
    h, hidden = norm.residual_rms_linear(x,a,g,nw,uw,us,ub,eps,warps)
    out = swiglu_int8_linear(hidden,dw,ds,db,16)
    return h,out


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = 5
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    if not fused.supports_fused_ffn_up('cuda'):
        raise RuntimeError('this experiment requires SM120')
    environment = base._environment()
    environment.pop('comfy_kitchen_file',None)
    # Explicit original configuration: newer SM120 auto selection is disabled
    # only during this reference runtime's construction, not by changing kernels.
    with patch.object(norm,'supports_fused_ffn_up',return_value=False):
        runtime = H3VAEPyOptRuntime(
            model_code_dir=args.model_code_dir,weights_path=args.weights,
            encoder_tile_size=256,decoder_tile_size=256,encoder_staged_batch=4,
            tile_batch=args.tile_batch,int8_encode=False,int8_decode=True,
            decode_fusions=True,log_calls=False).eval()
    old_decoder = runtime.core.decoder
    old_raw = getattr(old_decoder,'_orig_mod',old_decoder)
    if len(old_raw.transformer_blocks)!=36 or any(b.fuse_ffn_up for b in old_raw.transformer_blocks):
        raise RuntimeError('original decoder reference was not selected')
    new_raw = clone_containers(old_raw)
    for block in new_raw.transformer_blocks:
        if not isinstance(block,norm.NormQuantBlock):
            raise RuntimeError('expected NormQuantBlock')
        block.fuse_ffn_up = True
    new_decoder = None
    mode='baseline'
    verify=False
    counts=collections.Counter()
    seen_weights=set()
    seen_shapes=collections.Counter()
    checked=set()
    expected_calls=None
    reduction_control = ReferenceReductionBindings() if args.reuse_reference_reductions else None
    original_impl=fused._residual_ffn_impl
    sources=[Path(__file__),REPO/'opt/int8_ffn_up_fused.py',REPO/'opt/int8_norm_quant.py',
             REPO/'opt/int8_swiglu_fused.py',REPO/'h3vae_runtime.py',
             REPO/'scripts/benchmark_reduction_control.py']
    report={'scope':__doc__,'status':'running','environment':environment,
            'baseline_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
            'weights_sha256':base._sha256(args.weights),
            'source_sha256':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            'settings':{'seed':args.seed,'warmup':args.warmup,'blocks':args.blocks,
                        'tile_batch':args.tile_batch,'decoder_tile':256,'encoder_tile':256,
                        'encoder_staged_batch':4,'int8_encode':False,'int8_decode':True,
                        'decode_fusions':True,'dtype':'float16','compile_mode':DECODER_COMPILE_MODE,
                        'sampling':'alternating ABBA / BAAB blocks','config':list(fused.CONFIG),
                        'reuse_reference_reductions':args.reuse_reference_reductions,
                        'normalization_schedule_control': 'reference_generated_globals' if reduction_control else 'none'},
            'videos':[],'samples':[],'q_scale_checks':[],
            'notes':['Only runtime.decode is timed; FP16 encode is shared and excluded.',
                     'Original production decoder, canonical nested baseline and candidate RGB must be exactly equal.',
                     'Every checked/timed decode must execute all 36 independent weights and the expected opaque call count.',
                     'Same-input q/scale checks run outside the timer on every independent weight/input shape.',
                     'FP16 defaults and non-SM120 fusion path are unchanged; original INT8 loss remains.',
                     'Peaks include the resident benchmark model/graph buffers, not isolated model VRAM.',
                     'Optional reference-generated-globals control fixes norm schedules in both variants; no run observer is active during timing.',
                     'Independent default recompilations can choose different floating-point reduction orders; exactness is scoped to the recorded schedules.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

    def dispatch(x,a,g,nw,uw,us,ub,dw,ds,db,eps,warps):
        counts[mode]+=1
        seen_weights.add(uw.data_ptr())
        key=(uw.data_ptr(),tuple(x.shape))
        shape=(x.numel()//x.shape[-1],uw.shape[0],x.shape[-1])
        seen_shapes[(mode,str(shape))]+=1
        if mode=='baseline':
            return _canonical_baseline(x,a,g,nw,uw,us,ub,dw,ds,db,eps,warps)
        if mode!='candidate':
            raise RuntimeError('unknown benchmark variant')
        if verify and key not in checked:
            _,q,s=norm.residual_quant(x,a,g,nw,eps,warps)
            hidden=prequantized_linear(q,s,uw,us,ub)
            rq,rs=quantize_swiglu(hidden,num_warps=16)
            cq,cs=fused.quantize_ffn_up(q,s,uw,us,ub)
            row={'check_index':len(checked),'input_shape':list(x.shape),
                 'gemm_mnk':list(shape),'q_equal':torch.equal(cq,rq),
                 'scale_equal':torch.equal(cs,rs)}
            report['q_scale_checks'].append(row)
            if not row['q_equal'] or not row['scale_equal']:
                save()
                raise RuntimeError(f'real FFN quantization changed: {row}')
            checked.add(key)
            del q,s,hidden,rq,rs,cq,cs
        return original_impl(x,a,g,nw,uw,us,ub,dw,ds,db,eps,warps)

    def checked_decode(latent,stage):
        nonlocal expected_calls
        before=counts[mode]
        seen_weights.clear()
        value=runtime.decode(latent)
        calls=counts[mode]-before
        weights=len(seen_weights)
        if calls<=0 or calls%36 or weights!=36:
            raise RuntimeError(f'{stage}: invalid execution coverage calls={calls}, weights={weights}')
        if expected_calls is None:
            expected_calls=calls
        elif calls!=expected_calls:
            raise RuntimeError(f'{stage}: call count {calls} differs from {expected_calls}')
        if runtime.core.decoder is not new_decoder:
            raise RuntimeError('compiled decoder changed during paired comparison')
        return value,{'opaque_calls':calls,'independent_weights':weights}

    paths=[args.video]
    if args.videos_dir:
        paths += [p for p in sorted(args.videos_dir.glob('*.mp4')) if p.resolve()!=args.video.resolve()]
    fused._residual_ffn_impl=dispatch
    save()
    try:
        for index,path in enumerate(paths):
            reference_rgb,x,_=base._read_rgb_video(path)
            del reference_rgb
            x=x.cuda()
            latent=runtime.encode(x).half()
            torch.cuda.synchronize()
            item={'sample':path.stem,'video_sha256':base._sha256(path),
                  'input_shape':list(x.shape),'input_stride':list(x.stride()),
                  'latent_shape':list(latent.shape),'latent_stride':list(latent.stride())}
            del x
            runtime.core.decoder=old_decoder
            original_before=sum(counts.values())
            with (reduction_control.capture_reference() if reduction_control
                  else _record_reduction_schedules()) as schedules:
                production=runtime.decode(latent)
            if reduction_control:
                schedules=_controlled_schedules(reduction_control)
            item['production_reduction_schedules']=list(schedules.values())
            torch.cuda.synchronize()
            item['production_finite']=bool(torch.isfinite(production).all())
            if not item['production_finite']:
                report['videos'].append(item); save()
                raise RuntimeError('original production RGB is not finite')
            if sum(counts.values())!=original_before:
                raise RuntimeError('Original production reference unexpectedly executed the new opaque op')
            item['original_new_op_calls']=0
            if new_decoder is None:
                # Reset is required: mutating a class method alone reused the old
                # graph in an earlier rejected harness. Container flags are separate.
                torch._dynamo.reset()
                new_decoder=torch.compile(new_raw,mode=DECODER_COMPILE_MODE,dynamic=False)
            runtime.core.decoder=new_decoder
            expected_calls=None
            for mode in ('baseline','candidate'):
                verify=mode=='candidate'
                with (reduction_control.bind_candidate() if reduction_control
                      else _record_reduction_schedules()) as schedules:
                    result,coverage=checked_decode(latent,'quality')
                if reduction_control:
                    reduction_control.assert_bound()
                    schedules=_controlled_schedules(reduction_control)
                exact=torch.equal(result,production)
                finite=bool(torch.isfinite(result).all())
                item[mode]={'equal_to_production':exact,'finite':finite,
                            'reduction_schedules':list(schedules.values()),**coverage}
                if not exact or not finite:
                    item[mode]['error']=base._tensor_error(result,production)
                    report['videos'].append(item); save()
                    raise RuntimeError(f'{mode} RGB differs from original production')
                del result
            verify=False
            if reduction_control:
                reduction_control.assert_bound()
                report['normalization_bindings']=list(reduction_control.bindings)
            if len({key[0] for key in checked})!=36:
                raise RuntimeError('Did not shadow-check every independent FFN-up weight')
            del production
            report['videos'].append(item)
            save()
            print(path.stem,'full RGB exact; calls',expected_calls,flush=True)
            if index==0:
                for mode in ('baseline','candidate'):
                    for _ in range(args.warmup):
                        result,_=checked_decode(latent,'warmup')
                        del result
                torch.cuda.synchronize()
                orders=(('baseline','candidate','candidate','baseline'),
                        ('candidate','baseline','baseline','candidate'))
                for block in range(args.blocks):
                    for mode in orders[block%2]:
                        torch.cuda.reset_peak_memory_stats()
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        wall=time.monotonic(); start.record()
                        result,coverage=checked_decode(latent,'timing')
                        end.record(); end.synchronize()
                        row={'block':block,'mode':mode,'seconds':start.elapsed_time(end)/1000,
                             'wall_seconds':time.monotonic()-wall,
                             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),**coverage}
                        report['samples'].append(row); del result
                        save(); print(row,flush=True)
            if reduction_control:
                reduction_control.assert_bound()
            del latent
        report['means_seconds']={mode:statistics.mean(row['seconds'] for row in report['samples'] if row['mode']==mode)
                                 for mode in ('baseline','candidate')}
        report['saved_seconds']=report['means_seconds']['baseline']-report['means_seconds']['candidate']
        report['reduction_percent']=100*report['saved_seconds']/report['means_seconds']['baseline']
        report['peak_allocated_bytes']={mode:max(row['peak_allocated_bytes'] for row in report['samples'] if row['mode']==mode)
                                       for mode in ('baseline','candidate')}
        report['call_counts']=dict(counts)
        report['shape_counts']=[{'mode':mode,'shape_mnk':shape,'count':count}
                                for (mode,shape),count in seen_shapes.items()]
        report['video_count']=len(report['videos'])
        report['status']='passed'
        save(); print(json.dumps(report,indent=2),flush=True)
    except Exception as exc:
        report['status']='failed'; report['failure']=f'{type(exc).__name__}: {exc}'
        save(); raise
    finally:
        if reduction_control:
            reduction_control.restore()
        fused._residual_ffn_impl=original_impl
        runtime.core.decoder=old_decoder


if __name__=='__main__': main()
