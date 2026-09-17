#!/usr/bin/env python3
"""Compare official ComfyUI MiniMax-H3 VAE outputs with fp16 accumulation off/on."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=Path("results/comfy_fast_quality.json"))
    args = parser.parse_args()
    if args.height % 16 or args.width % 16 or (args.frames != 1 and (args.frames - 5) % 17):
        parser.error("height/width must be multiples of 16; frames must be 1 or 17k+5")

    sys.path.insert(0, str(args.comfy_root.resolve()))
    os.environ.setdefault("COMFY_H3_VAE_COMPILE", "1")
    os.environ["COMFY_H3_VAE_TILE_SIZE"] = str(args.tile_size)

    import torch
    from safetensors.torch import load_file
    import comfy.ops
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
    from bench_pyopt_vs_trt import errors

    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    original_flag = torch.backends.cuda.matmul.allow_fp16_accumulation
    try:
        model = MiniMaxH3VideoVAE(
            tile_size=args.tile_size, operations=comfy.ops.disable_weight_init).eval()
        missing, unexpected = model.load_state_dict(
            load_file(str(args.weights), device="cpu"), strict=False)
        if missing or unexpected:
            raise RuntimeError(f"weight mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
        model.to(device="cuda", dtype=torch.float16).eval()
        tokens = 1 if args.frames == 1 else (args.frames - 5) // 17 * 5 + 2
        torch.manual_seed(args.seed)
        z = torch.randn((1, 24, tokens, args.height // 16, args.width // 16),
                        device="cuda", dtype=torch.float16)
        torch.manual_seed(args.seed + 1)
        x = torch.rand((1, 3, args.frames, args.height, args.width),
                       device="cuda", dtype=torch.float16).mul_(2).sub_(1)
        outputs = {}
        with torch.inference_mode():
            for label, enabled in (("off", False), ("on", True), ("off_repeat", False)):
                torch.backends.cuda.matmul.allow_fp16_accumulation = enabled
                # The first call for each branch also absorbs per-shape compilation.
                model.decode(z)
                model.encode(x)
                torch.cuda.synchronize()
                outputs[label] = (model.decode(z).cpu(), model.encode(x).cpu())
        ref_dec, ref_enc = outputs["off"]
        result = {
            "source": str(args.comfy_root.resolve()),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "shape": [args.height, args.width, args.frames],
            "tile_size": args.tile_size,
            "seed": args.seed,
            "decode_fast_vs_default": errors(ref_dec, outputs["on"][0], pixels=True),
            "encode_fast_vs_default": errors(ref_enc, outputs["on"][1], pixels=False),
            "decode_repeat_vs_default": errors(ref_dec, outputs["off_repeat"][0], pixels=True),
            "encode_repeat_vs_default": errors(ref_enc, outputs["off_repeat"][1], pixels=False),
        }
        quality = result["encode_fast_vs_default"]
        if quality["shape_match"]:
            quality["relative_rmse"] = quality["rmse"] / max(quality["reference_rms"], 1e-12)
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = original_flag
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
