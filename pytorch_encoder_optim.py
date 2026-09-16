"""Pure-PyTorch inference optimizations for the MiniMax-H3 VAE encoder."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


@dataclass
class EncoderOptimizationInfo:
    whole_compile: bool = False
    channels_last_3d: bool = False
    quant_conv_in_graph: bool = False
    cuda_graph_requested: bool = False
    cuda_graph_captured: bool = False
    cuda_graph_fallback_reason: str | None = None
    expected_shape: tuple[int, ...] | None = None
    compile_backend: str = "inductor"
    compile_mode: str = "max-autotune-no-cudagraphs"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChannelsLast3DEncoder(nn.Module):
    """Keep the CNN internally NDHWC while accepting ordinary NCDHW tiles."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous(memory_format=torch.channels_last_3d)
        return self.encoder(x)


class EncoderWithQuantConv(nn.Module):
    """Compile the encoder CNN and its trailing 1x1 quant conv together."""

    def __init__(
        self,
        encoder: nn.Module,
        quant_conv: nn.Module,
        channels_last_3d: bool,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv
        self.channels_last_3d = channels_last_3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_last_3d:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        return self.quant_conv(self.encoder(x))


class StaticCUDAGraphEncoder(nn.Module):
    """Replay a fixed-shape compiled encoder with safe output ownership."""

    def __init__(
        self,
        encoder: nn.Module,
        expected_shape: tuple[int, ...],
        warmup: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = encoder
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
        # Use a dense NCDHW staging tensor even when x is a non-contiguous
        # temporal slice. The compiled wrapper performs the NDHWC conversion.
        self.static_input = torch.empty(
            x.shape, device=x.device, dtype=x.dtype
        )
        self.static_input.copy_(x)

        current_stream = torch.cuda.current_stream(x.device)
        warmup_stream = torch.cuda.Stream(device=x.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(self.warmup):
                self.encoder(self.static_input)
        current_stream.wait_stream(warmup_stream)
        torch.cuda.synchronize(x.device)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.encoder(self.static_input)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._matches(x) or self.capture_error is not None:
            return self.encoder(x)

        if self.graph is None:
            try:
                self._capture(x)
            except Exception as exc:
                self.capture_error = f"{type(exc).__name__}: {exc}"
                self.graph = None
                self.static_input = None
                self.static_output = None
                logger.warning(
                    "MiniMax-H3 encoder CUDA Graph capture failed; using the "
                    "compiled encoder: %s",
                    self.capture_error,
                )
                return self.encoder(x)
        else:
            self.static_input.copy_(x)
            self.graph.replay()

        # encode_temporal retains each clip output until the final cat.
        return self.static_output.clone()


def optimize_minimax_h3_encoder(
    model: nn.Module,
    *,
    expected_shape: tuple[int, ...],
    channels_last_3d: bool = True,
    quant_conv_in_graph: bool = False,
    cuda_graph: bool = False,
    backend: str = "inductor",
    mode: str = "max-autotune-no-cudagraphs",
) -> EncoderOptimizationInfo:
    """Optimize only the MiniMax-H3 CNN encoder; leave decoder untouched."""

    core = getattr(model, "model", model)
    encoder = core.encoder
    info = EncoderOptimizationInfo(
        channels_last_3d=channels_last_3d,
        quant_conv_in_graph=quant_conv_in_graph,
        cuda_graph_requested=cuda_graph,
        expected_shape=tuple(expected_shape),
        compile_backend=backend,
        compile_mode=mode,
    )

    if channels_last_3d:
        encoder.to(memory_format=torch.channels_last_3d)
        core.quant_conv.to(memory_format=torch.channels_last_3d)

    if quant_conv_in_graph:
        encoder = EncoderWithQuantConv(
            encoder, core.quant_conv, channels_last_3d
        )
        core.quant_conv = nn.Identity()
    elif channels_last_3d:
        encoder = ChannelsLast3DEncoder(encoder)

    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable in this PyTorch build")
    encoder = torch.compile(
        encoder,
        backend=backend,
        mode=mode,
        fullgraph=False,
        dynamic=False,
    )
    info.whole_compile = True

    if cuda_graph:
        encoder = StaticCUDAGraphEncoder(
            encoder, expected_shape=expected_shape
        ).eval()

    core.encoder = encoder
    core._encoder_optimization_info = info
    return info


class FixedVideoPreprocessor(nn.Module):
    """Fuse temporal padding, pixel normalization, and NDHWC packing."""

    def __init__(self, clip_length: int) -> None:
        super().__init__()
        self.clip_length = clip_length

    def forward(
        self,
        x: torch.Tensor,
        pixel_mean: torch.Tensor,
        pixel_std: torch.Tensor,
    ) -> torch.Tensor:
        pad_size = (-x.shape[2]) % self.clip_length
        if pad_size:
            x = torch.cat(
                [x, x[:, :, -1:].repeat(1, 1, pad_size, 1, 1)], dim=2
            )
        x = (x - pixel_mean) / pixel_std
        return x.contiguous(memory_format=torch.channels_last_3d)


class FixedVideoPostprocessor(nn.Module):
    """Fuse temporal concat trimming, posterior mean, and normalization."""

    def __init__(self, token_drop: int) -> None:
        super().__init__()
        self.token_drop = token_drop

    def forward(
        self,
        moments: torch.Tensor,
        latent_mean: torch.Tensor,
        latent_std: torch.Tensor,
    ) -> torch.Tensor:
        if self.token_drop:
            moments = moments[:, :, : -self.token_drop]
        mean = moments[:, : moments.shape[1] // 2]
        return (mean - latent_mean) / latent_std


def build_compiled_encoder_video_prepost(
    model: nn.Module,
    *,
    backend: str = "inductor",
    mode: str = "default",
) -> tuple[nn.Module, nn.Module]:
    """Build small fixed-video pre/post graphs around the per-clip encoder."""

    core = getattr(model, "model", model)
    preprocessor = torch.compile(
        FixedVideoPreprocessor(core.clip_length).eval(),
        backend=backend,
        mode=mode,
        fullgraph=True,
        dynamic=False,
    )
    postprocessor = torch.compile(
        FixedVideoPostprocessor(core.token_drop).eval(),
        backend=backend,
        mode=mode,
        fullgraph=True,
        dynamic=False,
    )
    return preprocessor, postprocessor


def encode_video_mean_compiled_prepost(
    model: nn.Module,
    x: torch.Tensor,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    preprocessor: nn.Module,
    postprocessor: nn.Module,
) -> torch.Tensor:
    """Encode one fixed video with compiled pre/post and per-clip CNN graph."""

    core = getattr(model, "model", model)
    if x.ndim != 5 or x.shape[0] != 1:
        raise ValueError("compiled pre/post requires a 5D batch-1 video")
    if any(
        bool(getattr(core, name, False))
        for name in (
            "isolated_first_frame",
            "isolated_last_frame",
            "isolated_key_frame",
            "encoder_parallel",
        )
    ):
        raise ValueError("compiled pre/post does not support isolated/parallel modes")
    if core.tile_size < x.shape[-2] or core.tile_size < x.shape[-1]:
        raise ValueError("compiled pre/post requires one spatial tile")

    x = preprocessor(x, pixel_mean, pixel_std)
    outputs = []
    for start in range(0, x.shape[2], core.clip_length):
        outputs.append(core.encode(x[:, :, start : start + core.clip_length]))
    moments = torch.cat(outputs, dim=2)
    return postprocessor(moments, latent_mean, latent_std)


def encode_temporal_single_tile_channels_last(
    model: nn.Module, x: torch.Tensor
) -> torch.Tensor:
    """Fast equivalent of encode_temporal for one spatial tile per clip.

    The complete padded video is converted to channels-last once. Temporal
    slices keep a channels-last-compatible stride for batch size one, avoiding
    eight separate NCDHW-to-NDHWC materializations at 124 frames.
    """

    core = getattr(model, "model", model)
    if x.ndim != 5 or x.shape[0] != 1:
        raise ValueError("fast temporal encode requires a 5D batch-1 video")
    if any(
        bool(getattr(core, name, False))
        for name in (
            "isolated_first_frame",
            "isolated_last_frame",
            "isolated_key_frame",
            "encoder_parallel",
        )
    ):
        raise ValueError("fast temporal encode does not support isolated/parallel modes")
    if core.tile_size < x.shape[-2] or core.tile_size < x.shape[-1]:
        raise ValueError("fast temporal encode requires one spatial tile")

    clip_length = core.clip_length
    if x.shape[2] % clip_length:
        pad_size = (-x.shape[2]) % clip_length
        x = torch.cat(
            [x, x[:, :, -1:].repeat(1, 1, pad_size, 1, 1)], dim=2
        )
    x = x.contiguous(memory_format=torch.channels_last_3d)

    outputs = []
    for start in range(0, x.shape[2], clip_length):
        clip = x[:, :, start : start + clip_length]
        outputs.append(core.encode(clip))
    output = torch.cat(outputs, dim=2)
    if core.token_drop > 0:
        output = output[:, :, : -core.token_drop]
    return output



def encode_temporal_single_tile_microbatch(
    model: nn.Module, x: torch.Tensor, microbatch: int
) -> torch.Tensor:
    """Encode several independent temporal clips in one fixed-size batch."""

    core = getattr(model, "model", model)
    if microbatch < 2:
        return encode_temporal_single_tile_channels_last(model, x)
    if x.ndim != 5 or x.shape[0] != 1:
        raise ValueError("temporal microbatch encode requires a 5D batch-1 video")
    if any(
        bool(getattr(core, name, False))
        for name in (
            "isolated_first_frame",
            "isolated_last_frame",
            "isolated_key_frame",
            "encoder_parallel",
        )
    ):
        raise ValueError("temporal microbatch does not support isolated/parallel modes")
    if core.tile_size < x.shape[-2] or core.tile_size < x.shape[-1]:
        raise ValueError("temporal microbatch requires one spatial tile")

    clip_length = core.clip_length
    if x.shape[2] % clip_length:
        pad_size = (-x.shape[2]) % clip_length
        x = torch.cat(
            [x, x[:, :, -1:].repeat(1, 1, pad_size, 1, 1)], dim=2
        )
    clips = [
        x[:, :, start : start + clip_length]
        for start in range(0, x.shape[2], clip_length)
    ]

    outputs = []
    for start in range(0, len(clips), microbatch):
        group = clips[start : start + microbatch]
        valid = len(group)
        if valid < microbatch:
            group.extend([group[-1]] * (microbatch - valid))
        batch = torch.cat(group, dim=0).contiguous(
            memory_format=torch.channels_last_3d
        )
        encoded = core.encode(batch)[:valid]
        batch_size, channels, tokens, height, width = encoded.shape
        encoded = encoded.permute(1, 0, 2, 3, 4).reshape(
            1, channels, batch_size * tokens, height, width
        )
        outputs.append(encoded)

    output = torch.cat(outputs, dim=2)
    if core.token_drop > 0:
        output = output[:, :, : -core.token_drop]
    return output

def refresh_encoder_cuda_graph_info(model: nn.Module) -> dict[str, Any] | None:
    """Return optimization metadata after lazy CUDA Graph capture."""

    core = getattr(model, "model", model)
    info = getattr(core, "_encoder_optimization_info", None)
    if info is None:
        return None

    encoder = core.encoder
    if isinstance(encoder, StaticCUDAGraphEncoder):
        info.cuda_graph_captured = encoder.graph is not None
        info.cuda_graph_fallback_reason = encoder.capture_error
    return info.to_dict()
