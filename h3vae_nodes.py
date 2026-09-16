"""ComfyUI node: load the MiniMax H3 video VAE with PyTorch optimizations.

Drop-in replacement for the stock ``VAE Loader`` in H3 workflows: connect the
output to the standard ``VAE Decode`` / ``VAE Encode`` nodes. Everything the
stock nodes do (tiling fallbacks, device handling, latent scaling) keeps
working because the runtime implements the same first_stage_model interface
as ``comfy.ldm.minimax.vae.MiniMaxH3VideoVAE``.
"""
from __future__ import annotations

import logging

import torch

import folder_paths
from h3vae_runtime import (DEFAULT_MODEL_CODE_DIR, DEFAULT_WEIGHTS_FALLBACK,
                           H3VAEPyOptRuntime, build_comfy_vae)

logger = logging.getLogger("ComfyUI_H3VAE_PyOpt")

# Cache built VAEs so re-queueing a workflow does not reload 5 GB of weights.
_VAE_CACHE: dict[tuple, object] = {}


def _resolve_weights(vae_name: str, weights_path: str) -> str:
    if weights_path:
        return weights_path
    if vae_name == "use default":
        if not DEFAULT_WEIGHTS_FALLBACK:
            raise ValueError("Set H3_VAE_WEIGHTS_PATH or select a VAE file")
        return DEFAULT_WEIGHTS_FALLBACK
    return folder_paths.get_full_path_or_raise("vae", vae_name)


class H3VAEPyOptLoader:
    """Load MiniMax H3 video VAE (optimized PyTorch runtime)."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            vae_files = folder_paths.get_filename_list("vae")
        except Exception:
            vae_files = []
        choices = ["use default"] + vae_files
        default = ("use default" if "use default" in choices
                   else (choices[0] if choices else "use default"))
        return {
            "required": {
                "vae_name": (choices, {"default": default,
                            "tooltip": "Same safetensors the stock VAELoader uses "
                                       "(key-compatible fp16/fp32 export). 'use default' "
                                       "requires H3_VAE_WEIGHTS_PATH."}),
                "dtype": (["fp16", "fp32"], {"default": "fp16"}),
                "decoder_tile_size": ("INT", {
                    "default": 256, "min": 128, "max": 1024, "step": 16,
                    "tooltip": "Decoder spatial tile. 256 = stock behaviour (use this "
                               "for an apples-to-apples A/B). 368 = the campaign's fast "
                               "configuration, but RoPE coords are tile-normalized so the "
                               "output shifts (~20% mean) -- verify quality."}),
                "tile_batch": ("INT", {
                    "default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Decoder tiles batched per forward call. 1 = serial "
                               "(stock). 2 = the validated pairing (+~1%)."}),
                "compile_decoder": ("BOOLEAN", {"default": True,
                    "tooltip": "Whole-decoder torch.compile + QK RMSNorm/RoPE Triton "
                               "fusion (max-autotune-no-cudagraphs)."}),
                "compile_encoder": ("BOOLEAN", {"default": True,
                    "tooltip": "Encoder GN/SiLU/padding Triton fusions + whole-graph "
                               "compile + staged clip batching."}),
                "encoder_staged_batch": ([1, 2, 4], {"default": 4,
                    "tooltip": "Encoder clips batched in the suffix stage (validated "
                               "combinations with cut=2)."}),
                "cudnn_benchmark": ("BOOLEAN", {"default": True,
                    "tooltip": "Process-wide torch.backends.cudnn.benchmark. ~4% faster "
                               "encoder steady state; first conv call per shape pays a "
                               "search spike (~2 GiB at 336px) -- warmup absorbs it."}),
                "warmup": (["none", "decode", "encode", "both"], {"default": "none",
                    "tooltip": "Run a dummy pass at load to absorb inductor compile + "
                               "cuDNN search before the first real request. Use with the "
                               "warmup shape parameters below."}),
                "warmup_frames": ("INT", {"default": 124, "min": 1, "max": 1000}),
                "warmup_width": ("INT", {"default": 1344, "min": 64, "max": 4096, "step": 16}),
                "warmup_height": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 16}),
                "log_calls": ("BOOLEAN", {"default": True,
                    "tooltip": "Log per-call decode/encode latency (running mean) for "
                               "A/B comparison."}),
            },
            "optional": {
                "model_code_dir": ("STRING", {"default": DEFAULT_MODEL_CODE_DIR,
                    "tooltip": "FL2VA video_vae bundle dir (klvae reference code + "
                               "source config)."}),
                "weights_path": ("STRING", {"default": "",
                    "tooltip": "Direct path override for the VAE safetensors."}),
            },
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = "MiniMax_H3/Acceleration"
    DESCRIPTION = (
        "Load the MiniMax H3 video VAE with the validated PyTorch optimization "
        "stack: decoder QK RMSNorm+RoPE Triton fusion + whole-decoder compile + "
        "batched tiles; encoder GN/SiLU/padding fusions + staged batching. "
        "Output feeds the stock VAEDecode/VAEEncode nodes."
    )

    def load(self, vae_name, dtype, decoder_tile_size, tile_batch,
             compile_decoder, compile_encoder, encoder_staged_batch,
             cudnn_benchmark, warmup, warmup_frames, warmup_width,
             warmup_height, log_calls, model_code_dir=DEFAULT_MODEL_CODE_DIR,
             weights_path=""):
        key = (vae_name, weights_path, dtype, int(decoder_tile_size),
               int(tile_batch), bool(compile_decoder), bool(compile_encoder),
               int(encoder_staged_batch), model_code_dir, bool(log_calls))
        vae = _VAE_CACHE.get(key)
        if vae is None:
            weights = _resolve_weights(vae_name, weights_path)
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
            torch.backends.cudnn.benchmark_limit = 5
            runtime = H3VAEPyOptRuntime(
                model_code_dir=model_code_dir,
                weights_path=weights,
                device="cuda",
                dtype=torch.float16 if dtype == "fp16" else torch.float32,
                decoder_tile_size=int(decoder_tile_size),
                tile_batch=int(tile_batch),
                compile_decoder=bool(compile_decoder),
                compile_encoder=bool(compile_encoder),
                encoder_staged_batch=int(encoder_staged_batch),
                log_calls=bool(log_calls),
            )
            vae = build_comfy_vae(runtime)
            _VAE_CACHE[key] = vae
            logger.info("[H3VAE-PyOpt] loaded %s (%s, tile=%d, batch=%d)",
                        weights, dtype, decoder_tile_size, tile_batch)

        runtime = vae.first_stage_model
        if warmup != "none" and not getattr(runtime, "_warmed_up", False):
            try:
                runtime.warmup(int(warmup_frames), int(warmup_width),
                               int(warmup_height), mode=warmup)
                runtime._warmed_up = True
            except Exception as exc:  # noqa: BLE001 - warmup is best effort
                logger.warning("[H3VAE-PyOpt] warmup failed (%s: %s); the first "
                               "request will pay the compile cost",
                               type(exc).__name__, exc)
        return (vae,)


NODE_CLASS_MAPPINGS = {"H3VAEPyOptLoader": H3VAEPyOptLoader}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3VAEPyOptLoader": "MiniMax H3 VAE Load (PyTorch Optimized)",
}
