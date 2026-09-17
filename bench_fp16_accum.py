#!/usr/bin/env python3
"""Test PyTorch matmul FP16 accumulation for PyOpt (not comfy-kitchen --fast)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bench_pyopt_vs_trt import errors, time_cuda
from h3vae_runtime import H3VAEPyOptRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-code-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--height", type=int, default=672)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/fp16_accum_ab.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if not hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
        parser.error("this PyTorch build has no allow_fp16_accumulation setting")
    if args.height % 16 or args.width % 16 or (args.frames != 1 and (args.frames - 5) % 17):
        parser.error("height/width must be multiples of 16; frames must be 1 or 17k+5")

    torch.manual_seed(20260917)
    tokens = 1 if args.frames == 1 else (args.frames - 5) // 17 * 5 + 2
    z = torch.randn((1, 24, tokens, args.height // 16, args.width // 16),
                    device="cuda", dtype=torch.float16)
    x = torch.rand((1, 3, args.frames, args.height, args.width),
                   device="cuda", dtype=torch.float16).mul_(2).sub_(1)
    original_flag = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    try:
        runtime = H3VAEPyOptRuntime(
            model_code_dir=str(args.model_code_dir), weights_path=str(args.weights),
            log_calls=False).eval()
        report = {"environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name()},
                  "config": {"height": args.height, "width": args.width,
                             "frames": args.frames, "warmup": args.warmup,
                             "runs": args.runs}, "measurements": []}
        baseline_dec = baseline_enc = None
        for label, enabled in (("off", False), ("on", True), ("off_repeat", False)):
            torch.backends.cuda.matmul.allow_fp16_accumulation = enabled
            dec_time, dec = time_cuda(lambda: runtime.decode(z), args.warmup, args.runs)
            enc_time, enc = time_cuda(lambda: runtime.encode(x), args.warmup, args.runs)
            item = {"setting": label, "decode": dec_time, "encode": enc_time}
            if baseline_dec is None:
                baseline_dec, baseline_enc = dec, enc
            else:
                item["decode_vs_off"] = errors(baseline_dec, dec, pixels=True)
                item["encode_vs_off"] = errors(baseline_enc, enc, pixels=False)
            report["measurements"].append(item)
            print(f"{label}: decode {dec_time['median_ms'] / 1000:.3f}s, "
                  f"encode {enc_time['median_ms'] / 1000:.3f}s", flush=True)
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = original_flag
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
