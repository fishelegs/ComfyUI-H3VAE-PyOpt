"""Full-video same-compiled-graph A/B for causal-zero INT8 conv skipping.

Only the opaque fused norm's quantized-convolution dispatch varies. The
producer, quantization, norm, static weights, compiler graph and runtime
outputs are common. Compilation, model loading, media I/O and verification
are excluded from timing; dynamic quantization and allocations are included.
Private paths and raw media never enter the repository from this experiment.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import bench_int8_vae as base
from h3vae_runtime import H3VAEPyOptRuntime, ENCODER_COMPILE_OPTIONS
from opt import encoder_int8_integration as integration


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--videos-dir', type=Path)
    parser.add_argument('--model-code-dir', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--blocks', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--check-rgb', action='store_true')
    args = parser.parse_args()
    if args.warmup < 2 or args.blocks < 1:
        parser.error('warmup >= 2 and blocks >= 1 required')
    if args.output.exists():
        parser.error('choose a fresh output path')
    return args


def eligible(qx, config, tile_variant):
    # This dispatcher is called only by int8_norm_conv3d, whose producer writes
    # known-zero temporal planes. Never install this on generic conv entrypoints.
    if (tile_variant != '128x128x64_pipeline4'
            or tuple(config.kernel_size) != (3, 3, 3)
            or tuple(config.stride) != (1, 1, 1)
            or qx.shape[2] < 3 or qx.shape[3] < 3 or qx.shape[4] < 3):
        return False
    out_h, out_w = qx.shape[3] - 2, qx.shape[4] - 2
    return out_h * out_w % 128 == 0 and 9 * qx.shape[1] % 64 == 0


@torch.inference_mode()
def main():
    args = parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = 5
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError('experiment is scoped to SM120')
    environment = base._environment()
    environment.pop('comfy_kitchen_file', None)
    runtime = H3VAEPyOptRuntime(
        model_code_dir=args.model_code_dir, weights_path=args.weights,
        encoder_tile_size=256, decoder_tile_size=256,
        encoder_staged_batch=4, tile_batch=2,
        int8_encode=True, int8_decode=False, decode_fusions=False,
        log_calls=False,
    ).eval()
    raw = runtime.prefix._orig_mod
    modules = [m for m in raw.modules() if isinstance(m, integration.Int8FusedValidConv3d)]
    if len(modules) != 8 or not all(m.fuse_norm for m in modules):
        raise RuntimeError('expected eight SM120 norm-fused INT8 convolutions')
    prefix_identity = id(runtime.prefix)
    sources = [REPO/'opt/encoder_int8.py', REPO/'opt/encoder_int8_pipeline.py',
               REPO/'opt/encoder_int8_integration.py', REPO/'opt/encoder_int8_norm.py',
               REPO/'opt/encoder_int8_norm_recompute.py', REPO/'h3vae_runtime.py',
               Path(__file__)]
    report = {
        'scope': __doc__, 'environment': environment,
        'baseline_commit': subprocess.check_output(
            ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        'settings': {'warmup': args.warmup, 'blocks': args.blocks, 'seed': args.seed,
                     'check_rgb': args.check_rgb,
                     'input_dtype': 'float16', 'encoder_tile': 256,
                     'encoder_staged_batch': 4, 'decoder_tile': 256,
                     'decoder_tile_batch': 2, 'int8_encode': True,
                     'int8_decode': False, 'decode_fusions': False,
                     'compile_encoder': True, 'compile_decoder': True,
                     'encoder_compile_options': ENCODER_COMPILE_OPTIONS,
                     'cudnn_benchmark': True, 'cudnn_benchmark_limit': 5,
                     'sampling': 'ABBA / BAAB alternating blocks'},
        'weights_sha256': base._sha256(args.weights),
        'source_sha256': {str(p.relative_to(REPO)) if p.is_relative_to(REPO)
                          else p.name: sha256(p) for p in sources},
        'notes': [
            'Same compiled graph and immutable weights; only opaque quantized conv kernel differs.',
            'Norm producer is unchanged; both candidates see the same quantization contract.',
            'Peak allocated bytes include retained validation outputs, not isolated model VRAM.',
            'No validation synchronization inside timed hot paths.',
            'Other GPU work was not stopped; raw timing samples are retained.',
            'Synthetic replay and shadow checks are not substitutes for full video quality.',
        ],
        'samples': [], 'exact_kernel_checks': [], 'status': 'running',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    mode = 'baseline'
    verify = False
    checked = set()
    weight_ids = {}
    counts = collections.Counter()
    shape_counts = collections.Counter()
    fallback_counts = collections.Counter()
    original = integration._conv3d_from_quantized

    def dispatch(qx, scale, qw, ws, *, config, bias, tile_variant, causal_prefix_zero=False):
        if not causal_prefix_zero:
            raise RuntimeError('Benchmark dispatch requires the known-zero causal norm producer')
        key_id = weight_ids.setdefault(qw.data_ptr(), len(weight_ids))
        shape_key = f'weight{key_id}:in{tuple(qx.shape)}:out{config.out_channels}:bias{bias is not None}'
        counts[mode] += 1
        shape_counts[(mode, shape_key)] += 1
        can_skip = eligible(qx, config, tile_variant)
        kwargs = {'config': config, 'bias': bias, 'tile_variant': tile_variant}
        if mode == 'baseline' or not can_skip:
            if mode == 'candidate' and not can_skip:
                fallback_counts[shape_key] += 1
            return original(qx, scale, qw, ws, **kwargs, causal_prefix_zero=False)
        y = original(qx, scale, qw, ws, **kwargs, causal_prefix_zero=True)
        check_key = (key_id, tuple(qx.shape), bias is not None)
        if verify and check_key not in checked:
            if bool(torch.count_nonzero(qx[:, :, :2])):
                raise RuntimeError('causal-zero precondition failed on real producer')
            reference = original(qx, scale, qw, ws, **kwargs, causal_prefix_zero=False)
            exact = torch.equal(reference, y)
            row = {'weight_id': key_id, 'input_shape': list(qx.shape),
                   'output_shape': list(y.shape), 'qweight_shape': list(qw.shape),
                   'has_bias': bias is not None, 'zero_prefix': True, 'equal': exact,
                   'max_abs': 0.0 if exact else base._tensor_error(reference, y)['max_abs']}
            report['exact_kernel_checks'].append(row)
            if not exact:
                raise RuntimeError(f'real-layer convolution changed: {row}')
            checked.add(check_key)
        return y

    integration._conv3d_from_quantized = dispatch
    paths = [args.video]
    if args.videos_dir:
        paths += [p for p in sorted(args.videos_dir.glob('*.mp4'))
                  if p.resolve() != args.video.resolve()]
    save()
    try:
        for index, path in enumerate(paths):
            reference_rgb, x, metadata = base._read_rgb_video(path)
            del reference_rgb, metadata
            x = x.cuda()
            latents = {}
            item = {'sample': path.stem, 'input_shape': list(x.shape),
                    'input_stride': list(x.stride()), 'blocks': []}
            for mode in ('baseline', 'candidate'):
                verify = mode == 'candidate'
                before = counts[mode]
                latents[mode] = runtime.encode(x)
                torch.cuda.synchronize()
                item[mode + '_conv_calls'] = counts[mode] - before
            verify = False
            if id(runtime.prefix) != prefix_identity:
                raise RuntimeError('compiled prefix changed during experiment')
            item['latents_equal'] = bool(torch.equal(latents['baseline'], latents['candidate']))
            item['latent_error'] = base._tensor_error(latents['baseline'], latents['candidate'])
            if not item['latents_equal'] or item['candidate_conv_calls'] == 0:
                raise RuntimeError(f'full latent validation failed: {item}')
            report['samples'].append(item)
            print(path.stem, 'latent exact, conv calls', item['candidate_conv_calls'], flush=True)
            save()
            if index == 0:
                for _ in range(args.warmup):
                    for mode in ('baseline', 'candidate'):
                        y = runtime.encode(x)
                        torch.cuda.synchronize()
                for block_index in range(args.blocks):
                    block = []
                    order = ['baseline', 'candidate', 'candidate', 'baseline']
                    if block_index % 2:
                        order = ['candidate', 'baseline', 'baseline', 'candidate']
                    for mode in order:
                        before = counts[mode]
                        torch.cuda.reset_peak_memory_stats()
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        y = runtime.encode(x)
                        end.record()
                        end.synchronize()
                        sample = {'mode': mode, 'ms': start.elapsed_time(end),
                                  'peak_allocated': torch.cuda.max_memory_allocated(),
                                  'conv_calls': counts[mode] - before}
                        block.append(sample)
                        print(block_index, sample, flush=True)
                    item['blocks'].append(block)
                    save()
                item['mean_ms'] = {
                    label: statistics.mean(row['ms'] for block in item['blocks']
                                           for row in block if row['mode'] == label)
                    for label in ('baseline', 'candidate')}
                item['latency_reduction_percent'] = 100 * (
                    1-item['mean_ms']['candidate']/item['mean_ms']['baseline'])
                if args.check_rgb:
                    rgb0 = runtime.decode(latents['baseline'].half()).cpu()
                    rgb1 = runtime.decode(latents['candidate'].half()).cpu()
                    item['rgb_equal'] = bool(torch.equal(rgb0, rgb1))
                    item['rgb_shape'] = list(rgb0.shape)
                    if not item['rgb_equal']:
                        raise RuntimeError('decoded RGB changed')
                    del rgb0, rgb1
                del y
            report['weight_count'] = len(weight_ids)
            report['checked_real_kernel_count'] = len(checked)
            save()
            del latents, x
        report['calls_total'] = dict(counts)
        report['calls_by_real_shape'] = [
            {'mode': label, 'shape': shape, 'calls': calls}
            for (label, shape), calls in sorted(shape_counts.items())]
        report['candidate_fallback_calls'] = dict(fallback_counts)
        if len(weight_ids) != 8 or not checked:
            raise RuntimeError('not all eight expected convolutions executed')
        report['status'] = 'ok'
        save()
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        save()
        raise
    finally:
        integration._conv3d_from_quantized = original


if __name__ == '__main__':
    main()
