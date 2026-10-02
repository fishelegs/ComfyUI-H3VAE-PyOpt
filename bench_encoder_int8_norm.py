"""Paired full-video validation of INT8 encoder norm/absmax producer fusion.

Both prefixes share immutable parameters, use the same INT8 convolution tile
and compile options, and differ only in the norm producer boundary. With --baseline-producer-file,
both variants share one compiled graph and dispatch inside its opaque op. Loading,
compilation and media I/O are excluded. Original checkpoints are read-only.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import statistics

import torch

import bench_int8_vae as base
from h3vae_runtime import H3VAEPyOptRuntime, ENCODER_COMPILE_OPTIONS
from opt import encoder_int8_integration as integration
from opt import encoder_int8_norm as producer


def _clone_prefix(module):
    result = copy.copy(module)
    result._parameters = module._parameters.copy()
    result._buffers = module._buffers.copy()
    result._modules = {
        key: None if child is None else _clone_prefix(child)
        for key, child in module._modules.items()
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True, help='Timed sample')
    parser.add_argument('--videos-dir', type=Path, help='Additional quality samples')
    parser.add_argument('--model-code-dir', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--blocks', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--baseline-producer-file', type=Path, help='Compare producer implementations inside the same compiled graph')
    parser.add_argument('--check-rgb', action='store_true')
    args = parser.parse_args()
    if args.warmup < 2 or args.blocks < 1:
        parser.error('warmup>=2 and blocks>=1 are required')
    if args.output.exists():
        parser.error('output exists; choose a fresh path')
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = 5
    runtime = H3VAEPyOptRuntime(
        model_code_dir=args.model_code_dir, weights_path=args.weights,
        encoder_tile_size=256, decoder_tile_size=256,
        encoder_staged_batch=4, tile_batch=2, int8_encode=True,
        log_calls=False,
    ).eval()
    raw = runtime.prefix._orig_mod
    modules = [m for m in raw.modules() if isinstance(m, integration.Int8FusedValidConv3d)]
    if len(modules) != 8 or not all(m.fuse_norm for m in modules):
        raise RuntimeError('Expected eight SM120 norm-fused INT8 convolutions')
    candidate_producer = producer.quantized_temporal_norm_pad
    mode = 'candidate'
    verify = False
    checked = set()
    if args.baseline_producer_file:
        spec = importlib.util.spec_from_file_location('h3vae_baseline_producer', args.baseline_producer_file)
        baseline_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline_module)
        old_producer = baseline_module.quantized_temporal_norm_pad
        prefixes = {'baseline': runtime.prefix, 'candidate': runtime.prefix}

        def dispatch(x, weight, bias, eps, *, pre_bias=None):
            if mode == 'baseline':
                return old_producer(x, weight, bias, eps, pre_bias=pre_bias)
            q, scale = candidate_producer(x, weight, bias, eps, pre_bias=pre_bias)
            key = (weight.data_ptr(), tuple(x.shape), pre_bias is not None)
            if verify and key not in checked:
                old_q, old_scale = old_producer(x, weight, bias, eps, pre_bias=pre_bias)
                if not torch.equal(q, old_q) or not torch.equal(scale, old_scale):
                    raise RuntimeError('Real-layer quantization changed')
                checked.add(key)
            return q, scale

        producer.quantized_temporal_norm_pad = dispatch
    else:
        baseline = _clone_prefix(raw)
        for module in baseline.modules():
            if isinstance(module, integration.Int8FusedValidConv3d):
                module.fuse_norm = False
        prefixes = {
            'baseline': torch.compile(baseline, options=ENCODER_COMPILE_OPTIONS, fullgraph=True, dynamic=False),
            'candidate': runtime.prefix,
        }
    hits = {'calls': 0, 'weights': set()}
    original = integration._conv3d_from_quantized

    def counted(qx, scale, qw, ws, **kwargs):
        hits['calls'] += 1
        hits['weights'].add(qw.data_ptr())
        return original(qx, scale, qw, ws, **kwargs)

    integration._conv3d_from_quantized = counted
    report = {
        'environment': base._environment(),
        'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'weights_sha256': base._sha256(args.weights),
        'input_contract': {'dtype': 'float16', 'encoder_tile': 256, 'encoder_staged_batch': 4,
                           'decoder_tile': 256, 'tile_batch': 2, 'int8_decode': False},
        'sources': {str(p): base._sha256(p) for p in [
            Path('opt/encoder_int8.py'), Path('opt/encoder_int8_norm.py'),
            Path('opt/encoder_int8_norm_recompute.py'),
            Path('opt/encoder_int8_pipeline.py'), Path('opt/encoder_int8_integration.py'),
            Path('opt/encoder_fused_temporal_norm.py'), Path('opt/encoder_fused_norm_bias.py'),
            Path('opt/encoder_fused_residual_pack.py'), Path('bench_encoder_int8_norm.py'),
        ]},
        'samples': [],
        'notes': ['Prefixes share immutable weights; both graphs use identical compile options.',
                  'Peak allocated memory includes retained validation outputs, not isolated model VRAM.',
                  'Other GPU work was not stopped. Raw timing samples are retained.'],
    }
    if args.baseline_producer_file:
        report['baseline_producer_sha256'] = base._sha256(args.baseline_producer_file)
        report['notes'][0] = 'Same compiled graph and immutable weights; only the opaque producer implementation changes.'
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    paths = [args.video]
    if args.videos_dir:
        paths += [p for p in sorted(args.videos_dir.glob('*.mp4')) if p.resolve() != args.video.resolve()]
    try:
        for index, path in enumerate(paths):
            reference, x, metadata = base._read_rgb_video(path)
            del reference
            x = x.cuda()
            latents = {}
            item = {'sample': path.stem, 'shape': list(x.shape), 'stride': list(x.stride()), 'blocks': []}
            for mode in prefixes:
                runtime.prefix = prefixes[mode]
                verify = index == 0 and mode == 'candidate'
                before = hits['calls']
                latents[mode] = runtime.encode(x)
                torch.cuda.synchronize()
                item[mode + '_producer_calls'] = hits['calls'] - before
            verify = False
            item['latent_error'] = base._tensor_error(latents['baseline'], latents['candidate'])
            item['latents_equal'] = torch.equal(latents['baseline'], latents['candidate'])
            if not item['latents_equal'] or item['candidate_producer_calls'] == 0:
                raise RuntimeError(f'Producer validation failed: {item}')
            print(path.stem, 'latent exact, producer calls', item['candidate_producer_calls'], flush=True)
            report['samples'].append(item)
            if index == 0:
                for _ in range(args.warmup):
                    for mode in prefixes:
                        runtime.prefix = prefixes[mode]
                        y = runtime.encode(x)
                        torch.cuda.synchronize()
                for block_index in range(args.blocks):
                    block = []
                    order = ['baseline', 'candidate', 'candidate', 'baseline']
                    if block_index % 2:
                        order = ['candidate', 'baseline', 'baseline', 'candidate']
                    for mode in order:
                        runtime.prefix = prefixes[mode]
                        torch.cuda.reset_peak_memory_stats()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        y = runtime.encode(x)
                        end.record()
                        end.synchronize()
                        result = {'mode': mode, 'ms': start.elapsed_time(end),
                                  'peak_allocated': torch.cuda.max_memory_allocated()}
                        block.append(result)
                        print(block_index, result, flush=True)
                    item['blocks'].append(block)
                    save()
                item['mean_ms'] = {
                    mode: statistics.mean(r['ms'] for block in item['blocks'] for r in block if r['mode'] == mode)
                    for mode in prefixes
                }
                item['latency_reduction_percent'] = 100 * (1 - item['mean_ms']['candidate'] / item['mean_ms']['baseline'])
                if args.check_rgb:
                    rgb0 = runtime.decode(latents['baseline'].half()).cpu()
                    rgb1 = runtime.decode(latents['candidate'].half()).cpu()
                    item['rgb_equal'] = torch.equal(rgb0, rgb1)
                    item['rgb_shape'] = list(rgb0.shape)
                    if not item['rgb_equal']:
                        raise RuntimeError('Decoded RGB changed')
                    del rgb0, rgb1
                del y
            report['producer_weight_count'] = len(hits['weights'])
            report['exact_producer_checks'] = len(checked)
            save()
            del latents, x
        if len(hits['weights']) != 8:
            raise RuntimeError('Not all eight producer paths executed')
        report['status'] = 'ok'
        save()
    finally:
        integration._conv3d_from_quantized = original
        producer.quantized_temporal_norm_pad = candidate_producer


if __name__ == '__main__':
    main()
