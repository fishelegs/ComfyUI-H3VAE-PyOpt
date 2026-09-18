"""ComfyUI_H3VAE_PyOpt -- MiniMax H3 video VAE, optimized PyTorch runtime.

Replaces the stock VAE Loader for H3 workflows: feeds the standard
VAEDecode / VAEEncode nodes with a runtime that carries the 2026-09
optimization campaign's validated stack (decoder QK RMSNorm+RoPE Triton
fusion + whole-decoder compile + batched tiles; encoder GN/SiLU/padding
fusions + channels-last-3d + staged clip batching).

ComfyUI loads custom-node directories as flat modules (spec_from_file_location
on __init__.py), so this package puts itself on sys.path and imports its
modules under unique h3vae_* names.
"""
import os
import sys

__version__ = "0.1.0"

# Must happen before video_vae is imported (deferred to runtime load) and
# before any inductor compile computes its cache dir.
os.environ.setdefault("MINIMAX_H3_TORCH_SDPA_BACKEND", "flash")
os.environ.setdefault("MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM", "1")

_here = os.path.dirname(os.path.realpath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from h3vae_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
