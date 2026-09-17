#!/usr/bin/env python3
"""Profile one warmed PyOpt encode or decode call by CUDA operator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from h3vae_runtime import H3VAEPyOptRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-code-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--mode", choices=("encode", "decode"), required=True)
    parser.add_argument("--height", type=int, default=672)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--output", type=Path, default=Path("results/operator_profile.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.height % 16 or args.width % 16 or (args.frames != 1 and (args.frames - 5) % 17):
        parser.error("height/width must be multiples of 16; frames must be 1 or 17k+5")

    torch.manual_seed(20260917)
    runtime = H3VAEPyOptRuntime(
        model_code_dir=str(args.model_code_dir), weights_path=str(args.weights),
        log_calls=False).eval()
    if args.mode == "encode":
        data = torch.rand((1, 3, args.frames, args.height, args.width),
                          device="cuda", dtype=torch.float16).mul_(2).sub_(1)
        fn = lambda: runtime.encode(data)
    else:
        tokens = 1 if args.frames == 1 else (args.frames - 5) // 17 * 5 + 2
        data = torch.randn((1, 24, tokens, args.height // 16, args.width // 16),
                           device="cuda", dtype=torch.float16)
        fn = lambda: runtime.decode(data)
    with torch.inference_mode():
        fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
    rows = sorted(prof.key_averages(), key=lambda row: row.self_device_time_total, reverse=True)
    result = {"mode": args.mode, "shape": [args.height, args.width, args.frames],
              "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name()},
              "top_cuda_operators": [
                  {"name": row.key, "self_cuda_ms": row.self_device_time_total / 1000,
                   "total_cuda_ms": row.device_time_total / 1000, "calls": row.count}
                  for row in rows if row.self_device_time_total > 0
              ][:30]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
