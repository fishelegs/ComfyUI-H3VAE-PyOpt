"""Capture a warmed, complete INT8 VAE stage with Nsight Systems.

Example (paths supplied by caller):
  nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
    --capture-range-end=stop --sample=none --cpuctxsw=none --kill=none \
    -o encode python bench_int8_profile.py --mode encode --video VIDEO \
    --model-code-dir CODE --weights WEIGHTS --output report.json

Encode uses INT8 prefix / ordinary decoder settings; decode uses FP16 encode
and the 144-linear INT8 fused decoder. These are separate runtime profiles.
Compilation, loading, video I/O and finite checks are outside the capture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import torch

from bench_int8_vae import _environment, _read_rgb_video
from h3vae_runtime import H3VAEPyOptRuntime


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("encode", "decode"), required=True)
    p.add_argument("--model-code-dir", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261002)
    p.add_argument("--encoder-tile", type=int, default=256)
    p.add_argument("--decoder-tile", type=int, default=256)
    p.add_argument("--encoder-staged-batch", type=int, default=4)
    p.add_argument("--tile-batch", type=int)
    args = p.parse_args()
    if args.warmup < 2:
        p.error("at least two warmups are required")
    if args.tile_batch is None:
        args.tile_batch = 4 if args.mode == "decode" else 2
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = 5
    environment = _environment()
    runtime = H3VAEPyOptRuntime(
        model_code_dir=str(args.model_code_dir), weights_path=str(args.weights),
        encoder_tile_size=args.encoder_tile, decoder_tile_size=args.decoder_tile,
        encoder_staged_batch=args.encoder_staged_batch, tile_batch=args.tile_batch,
        int8_encode=args.mode == "encode", int8_decode=args.mode == "decode",
        decode_fusions=args.mode == "decode", log_calls=False,
    ).eval()
    reference, pixels, _ = _read_rgb_video(args.video)
    del reference
    pixels = pixels.cuda()
    input_shape, input_stride = list(pixels.shape), list(pixels.stride())
    if args.mode == "decode":
        # runtime.encode returns normalized FP32 latents; INT8 decode requires FP16.
        latent = runtime.encode(pixels).half()
        torch.cuda.synchronize()
        del pixels
        operation = lambda: runtime.decode(latent)
    else:
        operation = lambda: runtime.encode(pixels)
    samples = []
    for i in range(args.warmup + 1):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin = time.monotonic()
        start.record()
        result = operation()
        end.record()
        end.synchronize()
        samples.append({"cuda_ms": start.elapsed_time(end), "wall_s": time.monotonic() - begin})
        del result
        print(f"{args.mode} warm/reference {i}: {samples[-1]}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.cudart().cudaProfilerStart()
    try:
        with torch.cuda.nvtx.range("vae_" + args.mode):
            begin = time.monotonic()
            result = operation()
            torch.cuda.synchronize()
            profile_wall_s = time.monotonic() - begin
    finally:
        torch.cuda.cudart().cudaProfilerStop()
    finite = bool(torch.isfinite(result).all())
    if not finite:
        raise RuntimeError("profile output is non-finite")
    report = {
        "mode": args.mode, "environment": environment,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "input_shape": input_shape, "input_stride": input_stride,
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "warmups_and_reference": samples, "profile_wall_s": profile_wall_s,
        "output_shape": list(result.shape), "finite": finite,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
