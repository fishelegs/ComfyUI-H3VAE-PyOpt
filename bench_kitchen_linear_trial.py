#!/usr/bin/env python3
"""A/B an opt-in comfy-kitchen FP16 linear replacement in PyOpt's decoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from bench_pyopt_vs_trt import errors, time_cuda
from h3vae_runtime import H3VAEPyOptRuntime, DECODER_COMPILE_MODE


def register_kitchen_op(kitchen):
    # Keep the Python stream lookup outside Dynamo tracing. The custom op is
    # intentionally inference-only, matching this benchmark's scope.
    @torch.library.custom_op("h3pyopt_trial::fp16_linear", mutates_args=())
    def kitchen_linear(x: torch.Tensor, weight: torch.Tensor,
                       bias: torch.Tensor | None) -> torch.Tensor:
        return kitchen.fp16_linear(x, weight, bias)

    @kitchen_linear.register_fake
    def _(x, weight, bias):
        return x.new_empty((*x.shape[:-1], weight.shape[0]))

    return kitchen_linear


class KitchenLinear(nn.Module):
    """Share the existing linear weights; change only the inference GEMM backend."""

    def __init__(self, original: nn.Linear, op):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        self.op = op

    def forward(self, x):
        return self.op(x, self.weight, self.bias)


def replace_linears(module: nn.Module, op) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, KitchenLinear(child, op))
            count += 1
        else:
            count += replace_linears(child, op)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-code-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--decoder-tile", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/kitchen_linear_trial.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.height % 16 or args.width % 16 or (args.frames != 1 and (args.frames - 5) % 17):
        parser.error("height/width must be multiples of 16; frames must be 1 or 17k+5")
    if args.warmup < 1 or args.runs < 2:
        parser.error("use at least one warmup and two measured runs")
    try:
        import comfy_kitchen as ck
    except ImportError as exc:
        parser.error(f"comfy-kitchen is required: {exc}")
    if not hasattr(ck, "fp16_linear"):
        parser.error("comfy-kitchen with fp16_linear is required (tested with 0.2.34)")

    torch.manual_seed(20260917)
    tokens = 1 if args.frames == 1 else (args.frames - 5) // 17 * 5 + 2
    z = torch.randn((1, 24, tokens, args.height // 16, args.width // 16),
                    device="cuda", dtype=torch.float16)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = 5
    original_flag = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    try:
        runtime = H3VAEPyOptRuntime(
            model_code_dir=str(args.model_code_dir), weights_path=str(args.weights),
            decoder_tile_size=args.decoder_tile, tile_batch=2, log_calls=False).eval()
        off_time, off_pixels = time_cuda(lambda: runtime.decode(z), args.warmup, args.runs)
        print(f"baseline off: {off_time['mean_ms'] / 1000:.3f}s", flush=True)

        torch.backends.cuda.matmul.allow_fp16_accumulation = True
        plain_time, plain_pixels = time_cuda(lambda: runtime.decode(z), args.warmup, args.runs)
        print(f"plain flag on: {plain_time['mean_ms'] / 1000:.3f}s", flush=True)

        raw_decoder = runtime.core.decoder._orig_mod
        count = replace_linears(raw_decoder, register_kitchen_op(ck))
        runtime.core.decoder = torch.compile(raw_decoder, mode=DECODER_COMPILE_MODE,
                                             dynamic=False)
        kitchen_time, kitchen_pixels = time_cuda(
            lambda: runtime.decode(z), args.warmup, args.runs)
        print(f"kitchen on: {kitchen_time['mean_ms'] / 1000:.3f}s", flush=True)

        torch.backends.cuda.matmul.allow_fp16_accumulation = False
        kitchen_off_time, kitchen_off_pixels = time_cuda(
            lambda: runtime.decode(z), args.warmup, args.runs)
        print(f"kitchen with global flag off: {kitchen_off_time['mean_ms'] / 1000:.3f}s", flush=True)

        result = {
            "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name()},
            "config": {"height": args.height, "width": args.width, "frames": args.frames,
                       "decoder_tile": args.decoder_tile, "warmup": args.warmup,
                       "runs": args.runs, "replaced_linears": count},
            "baseline_off": off_time,
            "plain_flag_on": plain_time,
            "kitchen_flag_on": kitchen_time,
            "kitchen_flag_off": kitchen_off_time,
            "plain_vs_off": errors(off_pixels, plain_pixels, pixels=True),
            "kitchen_vs_off": errors(off_pixels, kitchen_pixels, pixels=True),
            "kitchen_vs_plain": errors(plain_pixels, kitchen_pixels, pixels=True),
            "kitchen_flag_off_vs_baseline": errors(off_pixels, kitchen_off_pixels, pixels=True),
        }
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = original_flag
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
