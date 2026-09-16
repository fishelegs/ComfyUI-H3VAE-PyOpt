"""Pure-PyTorch inference optimizations for the MiniMax-H3 VAE decoder.

Apply these optimizations only after the model has been moved to its final
CUDA device/dtype and switched to eval mode.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


@dataclass
class DecoderOptimizationInfo:
    whole_compile: bool = False
    attention_in_graph: bool = False
    cuda_graph_requested: bool = False
    cuda_graph_captured: bool = False
    cuda_graph_fallback_reason: str | None = None
    expected_shape: tuple[int, ...] | None = None
    compile_backend: str = "inductor"
    compile_mode: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def enable_decoder_attention_in_graph(decoder: nn.Module) -> bool:
    """Remove the local ``torch.compiler.disable`` wrapper from Flash SDPA.

    The bundled attention module imports ``flash_attn`` into its own globals.
    Updating those globals is more reliable than importing another copy of the
    dynamically loaded module.
    """

    blocks = getattr(decoder, "transformer_blocks", None)
    if not blocks:
        raise ValueError("decoder has no transformer_blocks")

    attention = blocks[0].attn
    forward_globals = attention.__class__.forward.__globals__
    flash_attn = forward_globals.get("flash_attn")
    if flash_attn is None:
        raise RuntimeError("decoder attention globals do not contain flash_attn")

    original = getattr(flash_attn, "__wrapped__", None)
    if original is None:
        return not bool(getattr(flash_attn, "_torchdynamo_disable", False))

    forward_globals["flash_attn"] = original
    return True


class StaticCUDAGraphDecoder(nn.Module):
    """Replay one fixed-shape decoder tile through a CUDA Graph.

    The graph owns a static output tensor. ``forward`` clones that tensor after
    every replay because the VAE tiled path retains all four tile outputs until
    spatial blending starts. Returning the static tensor directly would alias
    all tiles and silently corrupt the decoded image.
    """

    def __init__(
        self,
        decoder: nn.Module,
        expected_shape: tuple[int, ...],
        warmup: int = 3,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.expected_shape = tuple(expected_shape)
        self.warmup = warmup
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_input: torch.Tensor | None = None
        self.static_output: torch.Tensor | None = None
        self.capture_error: str | None = None

    def _matches(self, x: torch.Tensor) -> bool:
        return (
            x.is_cuda
            and not self.training
            and tuple(x.shape) == self.expected_shape
            and not torch.is_grad_enabled()
        )

    @torch.no_grad()
    def _capture(self, x: torch.Tensor) -> None:
        self.static_input = torch.empty_like(x)
        self.static_input.copy_(x)

        current_stream = torch.cuda.current_stream(x.device)
        warmup_stream = torch.cuda.Stream(device=x.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(self.warmup):
                self.decoder(self.static_input)
        current_stream.wait_stream(warmup_stream)
        torch.cuda.synchronize(x.device)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.decoder(self.static_input)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._matches(x) or self.capture_error is not None:
            return self.decoder(x)

        if self.graph is None:
            try:
                self._capture(x)
            except Exception as exc:
                self.capture_error = f"{type(exc).__name__}: {exc}"
                self.graph = None
                self.static_input = None
                self.static_output = None
                logger.warning(
                    "MiniMax-H3 decoder CUDA Graph capture failed; using the "
                    "compiled decoder: %s",
                    self.capture_error,
                )
                return self.decoder(x)
        else:
            self.static_input.copy_(x)
            self.graph.replay()

        # The original tiled decoder keeps all tile results alive for blending.
        return self.static_output.clone()


def optimize_minimax_h3_decoder(
    model: nn.Module,
    *,
    expected_shape: tuple[int, ...],
    attention_in_graph: bool = True,
    cuda_graph: bool = False,
    backend: str = "inductor",
    mode: str = "max-autotune-no-cudagraphs",
) -> DecoderOptimizationInfo:
    """Compile the MiniMax-H3 decoder and optionally add fixed-shape replay.

    ``model`` may be the public ``MiniMaxH3VideoVAE`` wrapper or its underlying
    ``AutoencoderKL``. The function intentionally leaves the encoder untouched.
    """

    core = getattr(model, "model", model)
    decoder = core.decoder
    info = DecoderOptimizationInfo(
        attention_in_graph=False,
        cuda_graph_requested=cuda_graph,
        expected_shape=tuple(expected_shape),
        compile_backend=backend,
        compile_mode=mode,
    )

    if attention_in_graph:
        info.attention_in_graph = enable_decoder_attention_in_graph(decoder)

    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable in this PyTorch build")

    decoder = torch.compile(
        decoder,
        backend=backend,
        mode=mode,
        fullgraph=False,
        dynamic=False,
    )
    info.whole_compile = True

    if cuda_graph:
        decoder = StaticCUDAGraphDecoder(
            decoder, expected_shape=expected_shape
        ).eval()

    core.decoder = decoder
    # Keep metadata on the model for benchmark/reporting and production logs.
    core._decoder_optimization_info = info
    return info


def refresh_cuda_graph_info(model: nn.Module) -> dict[str, Any] | None:
    """Return optimization metadata after lazy CUDA Graph capture."""

    core = getattr(model, "model", model)
    info = getattr(core, "_decoder_optimization_info", None)
    if info is None:
        return None

    decoder = core.decoder
    if isinstance(decoder, StaticCUDAGraphDecoder):
        info.cuda_graph_captured = decoder.graph is not None
        info.cuda_graph_fallback_reason = decoder.capture_error
    return info.to_dict()
