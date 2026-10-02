"""Controlled full-video FP16 / previous INT8 / optimized INT8 encoder A/B.

The previous kernel is supplied explicitly from a Git snapshot. Both INT8
variants share the exact same compiled runtime and immutable weights; only
its opaque custom-op implementation is dispatched differently in this isolated
benchmark process. CUDA graphs are disabled by the production encoder options.
No model, runtime defaults, or original checkpoint is modified by this tool.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

import bench_int8_roundtrip as rt
import bench_int8_vae as base
from opt import encoder_int8 as current
from opt import encoder_int8_integration as integration


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("h3vae_encoder_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class Dispatch:
    def __init__(self, baseline):
        self.baseline = baseline
        self.mode = "optimized"
        self.calls = Counter()
        self.checked = {}
        self.verify = False
        baseline._TILE_VARIANTS["128x128x64"] = (128, 128, 64)

    def old_call(self, x, qw, ws, kwargs, *, tile_only=False):
        kwargs = dict(kwargs)
        kwargs["tile_variant"] = (
            "128x128x64"
            if tile_only
            else {128: "128x64x64", 256: "128x64x128"}[kwargs["config"].in_channels]
        )
        return self.baseline.int8_valid_conv3d(x, qw, ws, **kwargs)

    def __call__(self, x, qw, ws, **kwargs):
        self.calls[self.mode] += 1
        if self.mode in ("previous", "tile_only"):
            return self.old_call(x, qw, ws, kwargs, tile_only=self.mode == "tile_only")
        result = current.int8_valid_conv3d(x, qw, ws, **kwargs)
        if not self.verify:
            return result
        key = (qw.data_ptr(), tuple(x.shape), kwargs["bias"] is not None)
        if key not in self.checked:
            old = self.old_call(x, qw, ws, kwargs)
            metrics = base._tensor_error(old, result)
            self.checked[key] = {
                "input_shape": list(x.shape),
                "weight_shape": list(qw.shape),
                "bias": kwargs["bias"] is not None,
                **metrics,
            }
            if metrics["max_abs"] != 0:
                raise RuntimeError(f"real convolution changed: {self.checked[key]}")
        return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path)
    group.add_argument("--videos-dir", type=Path)
    parser.add_argument("--model-code-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--baseline-kernel-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--include-tile-only", action="store_true")
    parser.add_argument("--encoder-tile", type=int, default=256)
    parser.add_argument("--decoder-tile", type=int, default=256)
    parser.add_argument("--encoder-staged-batch", type=int, default=4)
    parser.add_argument("--tile-batch", type=int, default=2)
    args = parser.parse_args()
    if not args.quality_only and (args.warmup < 2 or args.runs < 3):
        parser.error("timing requires warmup>=2 and runs>=3")
    args.no_compile_encoder = False
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    paths = [args.video] if args.video else sorted(args.videos_dir.glob("*.mp4"))
    if not paths:
        raise ValueError("no input videos")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = 5
    baseline = load_baseline(args.baseline_kernel_file)
    dispatch = Dispatch(baseline)
    old_entry = integration.int8_valid_conv3d
    environment = base._environment()
    report = {
        "environment": environment,
        "settings": vars(args).copy(),
        "baseline_sha256": base._sha256(args.baseline_kernel_file),
        "weights_sha256": base._sha256(args.weights),
        "implementation_sources": rt._implementation_source_hashes(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "notes": [
            "Both runtimes resident; peak allocation is not isolated model VRAM.",
            "Dynamic quantization included; loading/compilation/media IO/metrics excluded.",
            "Only the first video is timed; all videos receive quality checks.",
            "Same compiled INT8 graph, same weights and input layout; opaque op dispatch varies.",
        ],
        "samples": [],
    }
    report["settings"] = {
        k: str(v) if isinstance(v, Path) else v for k, v in report["settings"].items()
    }
    print("Constructing compiled FP16 and INT8 runtimes", flush=True)
    fp16 = rt._make_runtime(args, int8_encode=False, int8_decode=False)
    int8 = rt._make_runtime(args, int8_encode=True, int8_decode=False)
    report["int8_modules"] = int8.int8_encoder_metadata
    modes = ["fp16", "previous", "optimized"]
    if args.include_tile_only:
        modes.insert(2, "tile_only")
    integration.int8_valid_conv3d = dispatch
    try:
        for index, path in enumerate(paths):
            print(f"Reading {path.name}", flush=True)
            reference, prepared, source = base._read_rgb_video(path)
            x = prepared.to("cuda")
            latents = {}

            def encode(mode, video=x):
                dispatch.mode = mode
                return (fp16 if mode == "fp16" else int8).encode(video)

            for mode in modes:
                dispatch.verify = mode == "optimized"
                latent = encode(mode)
                torch.cuda.synchronize()
                rt._validate_latent_shape(latent, reference, name=mode)
                rt._tensor_is_finite(latent, mode)
                latents[mode] = latent.cpu()
                del latent
                print(f"{path.name}: initial {mode} encode ready", flush=True)
            dispatch.verify = False
            errors = {
                mode: rt._latent_error(latents["previous"], latents[mode])
                for mode in modes
                if mode != "previous"
            }
            if errors["optimized"]["max_abs"] != 0:
                raise RuntimeError(f"optimized latent differs from previous: {errors}")
            sample = {
                "source_name": path.name,
                "source_sha256": base._sha256(path),
                "source": source,
                "input_stride": list(x.stride()),
                "latent_errors_vs_previous": errors,
            }
            if index == 0 and not args.quality_only:
                for warm in range(args.warmup):
                    for mode in modes[warm % len(modes) :] + modes[: warm % len(modes)]:
                        latent = encode(mode)
                        torch.cuda.synchronize()
                        del latent
                    print(f"warmup {warm + 1}/{args.warmup} complete", flush=True)
                samples = {mode: [] for mode in modes}
                orders = []
                for run in range(args.runs):
                    order = modes[run % len(modes) :] + modes[: run % len(modes)]
                    orders.append(order)
                    for pos, mode in enumerate(order):
                        samples[mode].append(
                            rt._timed_runtime_call(
                                lambda mode=mode: encode(mode),
                                device=torch.device("cuda"),
                                expected_shape=tuple(latents[mode].shape),
                                label=mode,
                                run_index=run,
                                order_position=pos,
                                timed=True,
                            )
                        )
                        print(
                            f"timed {run + 1} {mode}: {samples[mode][-1]['cuda_event_ms']:.3f} ms",
                            flush=True,
                        )
                sample["timing"] = rt._summarize_timings(samples)
                for mode, values in samples.items():
                    sample["timing"][mode]["median_cuda_event_ms"] = statistics.median(
                        v["cuda_event_ms"] for v in values
                    )
                sample["timed_order"] = orders
            # Decode every public latent through the same unchanged FP16 decoder.
            outputs = {}
            sample["quality_vs_source"] = {}
            sample["decode_wall_seconds"] = {}
            for mode in ("fp16", "previous", "optimized"):
                latent = latents[mode].to(device="cuda", dtype=torch.float16)
                start = time.perf_counter()
                output = fp16.decode(latent)
                torch.cuda.synchronize()
                sample["decode_wall_seconds"][mode] = time.perf_counter() - start
                rt._tensor_is_finite(output, f"{mode} decode")
                if output.shape != reference.shape:
                    raise RuntimeError("decoded shape differs from source")
                outputs[mode] = output.cpu()
                del output, latent
                sample["quality_vs_source"][mode] = base._quality_metrics(
                    reference, outputs[mode]
                )
                print(
                    f"{path.name}: {mode} source PSNR {sample['quality_vs_source'][mode]['mean_frame_psnr_db']:.6f}",
                    flush=True,
                )
            sample["decoded_error_vs_previous"] = base._tensor_error(
                outputs["previous"], outputs["optimized"]
            )
            sample["quality_optimized_vs_fp16"] = base._quality_metrics(
                outputs["fp16"], outputs["optimized"]
            )
            if sample["decoded_error_vs_previous"]["max_abs"] != 0:
                raise RuntimeError(
                    "identical latents produced differing decoded output"
                )
            report["samples"].append(sample)
            report["real_convolution_checks"] = list(dispatch.checked.values())
            report["dispatch_calls"] = dict(dispatch.calls)
            write_json(args.output_dir / "summary.json", report)
            print(f"{path.name}: exact latent and RGB equality passed", flush=True)
            del x, prepared, reference, latents, outputs
    finally:
        integration.int8_valid_conv3d = old_entry
    print(
        f"Completed {len(paths)} videos: {args.output_dir / 'summary.json'}", flush=True
    )


if __name__ == "__main__":
    main()
