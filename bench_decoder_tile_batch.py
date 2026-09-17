#!/usr/bin/env python3
"""Same-process decoder tile batch A/B with output checks."""
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
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--decoder-tile", type=int, default=256)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 0])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/decoder_tile_batch.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if any(batch < 0 or batch > 8 for batch in args.batches):
        parser.error("batch must be 0 (adaptive) or between 1 and 8")
    if args.height % 16 or args.width % 16 or (args.frames != 1 and (args.frames - 5) % 17):
        parser.error("height/width must be multiples of 16; frames must be 1 or 17k+5")

    torch.manual_seed(20260917)
    tokens = 1 if args.frames == 1 else (args.frames - 5) // 17 * 5 + 2
    z = torch.randn((1, 24, tokens, args.height // 16, args.width // 16),
                    device="cuda", dtype=torch.float16)
    runtime = H3VAEPyOptRuntime(
        model_code_dir=str(args.model_code_dir), weights_path=str(args.weights),
        decoder_tile_size=args.decoder_tile, tile_batch=1, log_calls=False).eval()
    core = runtime.core
    serial_run_tiles = core._run_tile_tasks
    report = {"environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name()},
              "config": {"height": args.height, "width": args.width,
                         "frames": args.frames, "decoder_tile": args.decoder_tile,
                         "warmup": args.warmup, "runs": args.runs}, "batches": []}
    reference = None
    for batch in args.batches:
        core._run_tile_tasks = serial_run_tiles
        core._last_tile_batch = 1
        if batch != 1:
            runtime._install_batched_tiles(core, batch)
        torch.cuda.reset_peak_memory_stats()
        timing, output = time_cuda(lambda: runtime.decode(z), args.warmup, args.runs)
        item = {"requested_batch": batch, "actual_batch": core._last_tile_batch,
                "decode": timing, "peak_allocated_mib":
                torch.cuda.max_memory_allocated() / 2**20}
        if reference is None:
            reference = output
        else:
            item["vs_serial"] = errors(reference, output, pixels=True)
        report["batches"].append(item)
        print(f"batch={batch} actual={item['actual_batch']} "
              f"median={timing['median_ms'] / 1000:.3f}s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
