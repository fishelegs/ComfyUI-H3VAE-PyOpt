#!/usr/bin/env python3
"""Full-video A/B: optimized PyTorch ComfyUI runtime versus legacy TRT VAE.

No model code, weights, or engines are bundled. Run on an otherwise idle GPU.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import torch


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def time_cuda(fn, warmup: int, runs: int):
    with torch.inference_mode():
        for _ in range(warmup):
            last = fn()
            torch.cuda.synchronize()
            del last
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(runs):
            start.record()
            last = fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        torch.cuda.synchronize()
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "samples_ms": times,
    }, last.detach().cpu()


def errors(reference: torch.Tensor, candidate: torch.Tensor, *, pixels: bool) -> dict:
    if reference.shape != candidate.shape:
        return {"shape_match": False, "reference": list(reference.shape),
                "candidate": list(candidate.shape)}
    # Chunk over temporal dimension to avoid materializing another full video.
    sum_abs = sum_sq = max_abs = count = 0
    signal_sq = 0.0
    for ref_part, cand_part in zip(reference.split(4, 2), candidate.split(4, 2)):
        ref = ref_part.float()
        diff = ref - cand_part.float()
        sum_abs += diff.abs().sum().item()
        sum_sq += diff.square().sum().item()
        signal_sq += ref.square().sum().item()
        max_abs = max(max_abs, diff.abs().max().item())
        count += diff.numel()
    mse = sum_sq / count
    return {"shape_match": True, "mae": sum_abs / count,
            "rmse": mse ** 0.5, "max_abs": max_abs,
            "psnr_db": (None if not pixels or mse == 0 else
                        10 * math.log10(1 / mse)),
            "reference_rms": (signal_sq / count) ** 0.5}


def run_pyopt(args, z, x01):
    from h3vae_runtime import H3VAEPyOptRuntime

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = 5
    runtime = H3VAEPyOptRuntime(
        model_code_dir=str(args.model_code_dir), weights_path=str(args.weights),
        decoder_tile_size=args.decoder_tile, encoder_tile_size=args.encoder_tile,
        tile_batch=args.tile_batch,
        encoder_staged_batch=args.staged_batch, compile_decoder=not args.no_compile,
        compile_encoder=not args.no_compile, log_calls=False,
    ).eval()
    x_comfy = x01.float().mul(2).sub(1).half()
    decode, decoded = time_cuda(lambda: runtime.decode(z), args.warmup, args.runs)
    encode, encoded = time_cuda(lambda: runtime.encode(x_comfy), args.warmup, args.runs)
    result = {"decode": decode, "encode": encode,
              "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "decoded_shape": list(decoded.shape), "encoded_shape": list(encoded.shape)}
    del runtime, x_comfy
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return result, decoded, encoded


def run_trt(args, z, x01):
    try:
        from bench_h3vae_trt import MiniMaxH3TRTVideoVAE, TensorRTRunner
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError("TensorRT Python bindings are missing; use the compatible "
                           "environment that built the engines") from exc

    vae = MiniMaxH3TRTVideoVAE(
        TensorRTRunner(args.decoder_engine, "latent_tile", "pixel_tile"),
        TensorRTRunner(args.encoder_engine, "pixel_tile", "moments_tile"),
        args.model_code_dir, args.decoder_tile, args.encoder_tile,
    )
    decode, decoded = time_cuda(lambda: vae.decode(z), args.warmup, args.runs)
    encode, encoded = time_cuda(lambda: vae.encode(x01), args.warmup, args.runs)
    result = {"decode": decode, "encode": encode, "tensorrt": trt.__version__,
              "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "decoded_shape": list(decoded.shape), "encoded_shape": list(encoded.shape)}
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return result, decoded, encoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-code-dir", type=Path, required=True,
                        help="external MiniMax FL2VA/video_vae directory")
    parser.add_argument("--weights", type=Path, required=True,
                        help="external FP16 VAE weights; must match the TRT engines")
    parser.add_argument("--decoder-engine", type=Path)
    parser.add_argument("--encoder-engine", type=Path)
    parser.add_argument("--pyopt-only", action="store_true")
    parser.add_argument("--height", type=int, default=672)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--decoder-tile", type=int, default=368)
    parser.add_argument("--encoder-tile", type=int, default=672)
    parser.add_argument("--tile-batch", type=int, default=2,
                        help="decoder tiles per call; 0 selects adaptive batching")
    parser.add_argument("--staged-batch", type=int, default=4)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path, default=Path("results/pyopt_vs_trt.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.height % 16 or args.width % 16 or args.frames < 1:
        parser.error("height/width must be multiples of 16 and frames must be positive")
    if args.frames != 1 and (args.frames - 5) % 17:
        parser.error("video frames must be 17k+5 (e.g. 124 or 243)")
    if args.warmup < 1 or args.runs < 2:
        parser.error("use at least one warmup and two measured runs")
    required = [args.model_code_dir / "config.json", args.weights]
    if not args.pyopt_only:
        if not args.decoder_engine or not args.encoder_engine:
            parser.error("both TRT engine paths are required unless --pyopt-only")
        required += [args.decoder_engine, args.encoder_engine]
    for path in required:
        if not path.is_file():
            parser.error(f"file not found: {path}")

    torch.manual_seed(args.seed)
    tokens = 1 if args.frames == 1 else (args.frames - 5) // 17 * 5 + 2
    z = torch.randn((1, 24, tokens, args.height // 16, args.width // 16),
                    device="cuda", dtype=torch.float16)
    x01 = torch.rand((1, 3, args.frames, args.height, args.width),
                     device="cuda", dtype=torch.float16)
    report = {"environment": {"python": sys.version.split()[0],
                               "torch": torch.__version__,
                               "cuda": torch.version.cuda,
                               "gpu": torch.cuda.get_device_name()},
              "config": {"height": args.height, "width": args.width,
                         "frames": args.frames, "decoder_tile": args.decoder_tile,
                         "encoder_tile": args.encoder_tile,
                         "tile_batch": args.tile_batch,
                         "staged_batch": args.staged_batch,
                         "warmup": args.warmup, "runs": args.runs,
                         "seed": args.seed, "compile": not args.no_compile},
              "artifacts_sha256": {str(path): digest(path) for path in required[1:]},
              "note": "PyOpt accepts [-1,1] pixels; TRT accepts [0,1]. Both derive from x01."}
    torch.cuda.reset_peak_memory_stats()
    pyopt, py_dec, py_enc = run_pyopt(args, z, x01)
    report["pyopt"] = pyopt
    if not args.pyopt_only:
        trt_result, trt_dec, trt_enc = run_trt(args, z, x01)
        report["trt"] = trt_result
        report["quality"] = {"decode": errors(trt_dec, py_dec, pixels=True),
                             "encode": errors(trt_enc, py_enc, pixels=False)}
        report["speedup_percent"] = {
            kind: 100 * (trt_result[kind]["median_ms"] - pyopt[kind]["median_ms"])
            / trt_result[kind]["median_ms"] for kind in ("decode", "encode")}
        report["speedup_percent"]["total"] = 100 * (
            sum(trt_result[k]["median_ms"] for k in ("decode", "encode"))
            - sum(pyopt[k]["median_ms"] for k in ("decode", "encode"))) / sum(
                trt_result[k]["median_ms"] for k in ("decode", "encode"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
