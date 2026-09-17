#!/usr/bin/env python3
"""Compare PyOpt spatial-tiled encode with a reference encoder tile plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bench_pyopt_vs_trt import errors
from h3vae_runtime import H3VAEPyOptRuntime, _load_klvae_core, _load_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-code-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--encoder-tile", type=int, default=672)
    parser.add_argument("--reference-tile", type=int, default=None,
                        help="Reference tile size; defaults to --encoder-tile")
    parser.add_argument("--output", type=Path, default=Path("results/encoder_quality.json"))
    args = parser.parse_args()
    selected_tile = args.encoder_tile or (
        672 if max(args.height, args.width) <= 672 else 256)
    reference_tile = selected_tile if args.reference_tile is None else args.reference_tile
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.height % 16 or args.width % 16 or args.frames < 1:
        parser.error("height and width must be divisible by 16; frames must be positive")

    torch.manual_seed(20260917)
    runtime = H3VAEPyOptRuntime(
        model_code_dir=str(args.model_code_dir), weights_path=str(args.weights),
        encoder_tile_size=args.encoder_tile, log_calls=False).eval()
    reference, _ = _load_klvae_core(str(args.model_code_dir))
    _load_weights(reference, str(args.weights))
    reference.to(device="cuda", dtype=torch.float16).eval()
    reference.tile_size = reference_tile
    reference.stack_tiling = False
    pixels = torch.rand((1, 3, args.frames, args.height, args.width),
                        device="cuda", dtype=torch.float16).mul_(2).sub_(1)

    with torch.inference_mode():
        candidate = runtime.encode(pixels).cpu()
        normalized = runtime._normalize_pixels(pixels)
        if args.frames == 1:
            moments = reference.tiled_encode(normalized)[:, :, -1:, :, :]
        else:
            moments = reference.encode_temporal(normalized)
        mean = moments.float()[:, :moments.shape[1] // 2]
        expected = ((mean - runtime.latents_mean.float()) /
                    runtime.latents_std.float()).cpu()
    result = {
        "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name()},
        "config": {"height": args.height, "width": args.width,
                   "frames": args.frames, "encoder_tile": args.encoder_tile,
                   "selected_encoder_tile": selected_tile,
                   "reference_tile": reference_tile},
        "reference_shape": list(expected.shape),
        "pyopt_shape": list(candidate.shape),
        "quality": errors(expected, candidate, pixels=False),
    }
    quality = result["quality"]
    if quality["shape_match"]:
        quality["relative_rmse"] = quality["rmse"] / max(quality["reference_rms"], 1e-12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
