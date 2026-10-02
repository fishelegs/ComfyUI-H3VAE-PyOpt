"""Bounded INT8 encoder tile investigation; does not change runtime policy.

Synthetic tensors match two captured T17 hotspot geometries. Uses the current
quantizer with legacy versus BN128 tiles; this is not a historical full-kernel
baseline after the 2026-10-02 optimization. Includes dynamic
quantization in full-operator timings; excludes padding and weight preparation.
The additional tile is registered only in this benchmark process. Results do
not validate full encoder performance, real-video quality, or memory savings.
"""

import argparse
import json, sys, platform, statistics, subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
import triton

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from opt.encoder_int8 import (
    prepare_int8_weight,
    quantize_activation_tensor,
    int8_valid_conv3d,
    _triton_kernels,
    _TILE_VARIANTS,
)

_TILE_VARIANTS["probe_128x128x64"] = (128, 128, 64)


def measure(functions):
    for fn in functions.values():
        for _ in range(2):
            fn()
    torch.cuda.synchronize()
    raw = {key: [] for key in functions}
    keys = list(functions)
    for r in range(7):
        for key in keys[r % len(keys) :] + keys[: r % len(keys)]:
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            value = functions[key]()
            end.record()
            end.synchronize()
            raw[key].append(start.elapsed_time(end))
    return {
        k: {
            "mean_ms": statistics.mean(v),
            "median_ms": statistics.median(v),
            "samples_ms": v,
        }
        for k, v in raw.items()
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261002)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = 5
    result = {
        "scope": "synthetic inputs, real hotspot geometries; not full encode/video quality",
        "commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "environment": {
            "python": platform.python_version(),
            "os": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
            "gpu": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "settings": {
            "seed": 20261002,
            "warmup": 2,
            "runs": 7,
            "layout": "channels_last_3d",
            "dtype": "fp16",
            "cudnn_benchmark": True,
            "benchmark_limit": 5,
            "padding": "synthetic constant prepared outside timing",
            "static_weight_prepare": "outside timing",
            "background_gpu_work": "not controlled; inspect GPU separately",
        },
        "cases": [],
    }
    for c, h, tile in [(128, 256, "128x64x64"), (256, 128, "128x64x128")]:
        x = torch.randn(
            (1, c, 17, h, h), device="cuda", dtype=torch.float16
        ).contiguous(memory_format=torch.channels_last_3d)
        x = F.pad(x, (1, 1, 1, 1, 2, 0)).contiguous(
            memory_format=torch.channels_last_3d
        )
        w = (
            torch.randn((c, c, 3, 3, 3), device="cuda", dtype=torch.float16)
            / (27 * c) ** 0.5
        ).contiguous(memory_format=torch.channels_last_3d)
        bias = torch.randn(c, device="cuda", dtype=torch.float16) * 0.1
        qw, ws, config = prepare_int8_weight(w)
        qx, xs = quantize_activation_tensor(x)
        out = torch.empty(
            (1, c, 17, h, h),
            device="cuda",
            dtype=torch.float16,
            memory_format=torch.channels_last_3d,
        )
        _, kernel = _triton_kernels()

        def launch(bm, bn, bk, warps, stages):
            kernel[(triton.cdiv(17 * h * h, bm), triton.cdiv(c, bn))](
                qx,
                qw,
                xs,
                ws,
                bias,
                out,
                *x.shape,
                c,
                17,
                h,
                h,
                *qx.stride(),
                *out.stride(),
                1,
                1,
                1,
                KERNEL_D=3,
                KERNEL_H=3,
                KERNEL_W=3,
                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_K=bk,
                HAS_BIAS=True,
                num_warps=warps,
                num_stages=stages,
            )
            return out

        configs = [
            (128, 64, 64, 4, 2),
            (128, 64, 128, 4, 2),
            (128, 128, 64, 4, 2),
            (128, 128, 128, 4, 2),
            (256, 64, 64, 8, 2),
            (128, 64, 128, 4, 3),
        ]
        funcs = {
            "fp16_conv": lambda: F.conv3d(x, w, bias).contiguous(
                memory_format=torch.channels_last_3d
            ),
            "int8_full": lambda: int8_valid_conv3d(
                x, qw, ws, config=config, bias=bias, tile_variant=tile
            ),
            "quantization": lambda: quantize_activation_tensor(x),
        }
        funcs["int8_full_bn128"] = lambda: int8_valid_conv3d(
            x, qw, ws, config=config, bias=bias, tile_variant="probe_128x128x64"
        )
        original = funcs["int8_full"]()
        correctness = {
            "int8_full_bn128": {
                "max_abs_vs_stock_int8": float(
                    (funcs["int8_full_bn128"]() - original).abs().max()
                )
            }
        }
        for cfg in configs:
            name = "conv_" + "_".join(map(str, cfg))
            funcs[name] = lambda cfg=cfg: launch(*cfg)
            y = funcs[name]()
            correctness[name] = {
                "max_abs_vs_stock_int8": float((y - original).abs().max()),
                "finite": bool(torch.isfinite(y).all()),
            }
        metrics = measure(funcs)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            funcs["int8_full"]()
            torch.cuda.synchronize()
        profile = [
            {
                "name": e.key,
                "count": e.count,
                "self_device_us": e.self_device_time_total,
            }
            for e in prof.key_averages()
            if e.self_device_time_total > 0
        ]
        ref = funcs["fp16_conv"]()
        case = {
            "channels": c,
            "input_shape": list(x.shape),
            "reference_tile": tile,
            "timing": metrics,
            "candidate_correctness": correctness,
            "rmse_int8_vs_fp16": float(
                (original.float() - ref.float()).square().mean().sqrt()
            ),
            "profile": sorted(profile, key=lambda e: -e["self_device_us"]),
        }
        result["cases"].append(case)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "channels": c,
                    "timing_ms": {
                        k: round(v["median_ms"], 4) for k, v in metrics.items()
                    },
                    "correctness": correctness,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
