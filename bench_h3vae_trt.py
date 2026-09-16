#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import shlex
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import tensorrt as trt

from pytorch_decoder_optim import (
    optimize_minimax_h3_decoder,
    refresh_cuda_graph_info,
)
from pytorch_encoder_optim import (
    build_compiled_encoder_video_prepost,
    encode_temporal_single_tile_channels_last,
    encode_temporal_single_tile_microbatch,
    encode_video_mean_compiled_prepost,
    optimize_minimax_h3_encoder,
    refresh_encoder_cuda_graph_info,
)


DECODER_SHAPE = (1, 24, 7, 16, 16)
ENCODER_SHAPE = (1, 3, 17, 256, 256)
DECODER_OUT_SHAPE = (1, 3, 28, 256, 256)
ENCODER_OUT_SHAPE = (1, 48, 5, 16, 16)
SPATIAL_RATIO = 16
TEMPORAL_RATIO = 4


class TensorRTRunner:
    def __init__(self, engine_path: Path, input_name: str, output_name: str):
        self.engine_path = engine_path
        self.input_name = input_name
        self.output_name = output_name
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = None
        self.context = None
        self.stream = None

    def load(self) -> None:
        with self.engine_path.open("rb") as f:
            engine_bytes = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"failed to create execution context for {self.engine_path}")

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, out_shape: tuple[int, ...]) -> torch.Tensor:
        if self.context is None:
            self.load()
        if self.stream is None or self.stream.device != x.device:
            self.stream = torch.cuda.Stream(device=x.device)
        x = x.contiguous()
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        if not self.context.set_input_shape(self.input_name, tuple(x.shape)):
            raise RuntimeError(
                f"failed to set {self.input_name} shape to {tuple(x.shape)} for {self.engine_path}"
            )
        actual_input_shape = tuple(self.context.get_tensor_shape(self.input_name))
        if actual_input_shape != tuple(x.shape):
            raise RuntimeError(
                f"{self.engine_path} input shape mismatch for {self.input_name}: "
                f"requested {tuple(x.shape)}, got {actual_input_shape}"
            )
        actual_output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        if actual_output_shape != tuple(out_shape):
            raise RuntimeError(
                f"{self.engine_path} output shape mismatch for {self.output_name}: "
                f"requested {tuple(out_shape)}, got {actual_output_shape}"
            )
        self.context.set_tensor_address(self.input_name, x.data_ptr())
        self.context.set_tensor_address(self.output_name, out.data_ptr())
        self.stream.wait_stream(torch.cuda.current_stream(x.device))
        ok = self.context.execute_async_v3(self.stream.cuda_stream)
        if not ok:
            raise RuntimeError(f"TensorRT execution failed for {self.engine_path}")
        torch.cuda.current_stream(x.device).wait_stream(self.stream)
        return out


@dataclass
class TimedResult:
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    runs: int


@dataclass(frozen=True)
class ShapeConfig:
    decoder_shape: tuple[int, int, int, int, int]
    encoder_shape: tuple[int, int, int, int, int]
    decoder_out_shape: tuple[int, int, int, int, int]
    encoder_out_shape: tuple[int, int, int, int, int]
    mode: str = "engine_tile"
    height: int | None = None
    width: int | None = None
    frames: int | None = None
    decoder_engine_calls_per_run: int | None = None
    encoder_engine_calls_per_run: int | None = None


def ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def shape_config_from_args(args) -> ShapeConfig:
    if args.height is None and args.width is None and args.frames is None:
        return ShapeConfig(DECODER_SHAPE, ENCODER_SHAPE, DECODER_OUT_SHAPE, ENCODER_OUT_SHAPE)

    if args.height is None or args.width is None or args.frames is None:
        raise ValueError("--height, --width, and --frames must be provided together")
    if args.height % SPATIAL_RATIO != 0 or args.width % SPATIAL_RATIO != 0:
        raise ValueError(f"height and width must be divisible by {SPATIAL_RATIO}")
    if not args.full_video and args.frames % TEMPORAL_RATIO != 0:
        raise ValueError(f"frames must be divisible by {TEMPORAL_RATIO} for decoder benchmarking")

    latent_h = args.height // SPATIAL_RATIO
    latent_w = args.width // SPATIAL_RATIO
    if args.full_video:
        planner = MiniMaxH3TRTVideoPlanner(args.decoder_tile_size, args.encoder_tile_size)
        decoder_latent_t = planner.latent_tokens_for_frames(args.frames)
        decoder_y_tiles = len(planner.split_tiles(args.height, planner.decoder_tile_size)[0])
        decoder_x_tiles = len(planner.split_tiles(args.width, planner.decoder_tile_size)[0])
        encoder_y_tiles = len(planner.split_tiles(args.height, planner.encoder_tile_size)[0])
        encoder_x_tiles = len(planner.split_tiles(args.width, planner.encoder_tile_size)[0])
        decoder_spatial_tiles = decoder_y_tiles * decoder_x_tiles
        encoder_spatial_tiles = encoder_y_tiles * encoder_x_tiles
        _, decoder_chunks = planner.decode_temporal_chunks(decoder_latent_t)
        encoder_clips = ceil_div(args.frames, planner.clip_length)
        return ShapeConfig(
            decoder_shape=(1, 24, decoder_latent_t, latent_h, latent_w),
            encoder_shape=(1, 3, args.frames, args.height, args.width),
            decoder_out_shape=(1, 3, args.frames, args.height, args.width),
            encoder_out_shape=(1, 24, decoder_latent_t, latent_h, latent_w),
            mode="project_tiled_video",
            height=args.height,
            width=args.width,
            frames=args.frames,
            decoder_engine_calls_per_run=decoder_chunks * decoder_spatial_tiles,
            encoder_engine_calls_per_run=encoder_clips * encoder_spatial_tiles,
        )

    decoder_latent_t = args.frames // TEMPORAL_RATIO
    encoder_latent_t = ceil_div(args.frames, TEMPORAL_RATIO)
    return ShapeConfig(
        decoder_shape=(1, 24, decoder_latent_t, latent_h, latent_w),
        encoder_shape=(1, 3, args.frames, args.height, args.width),
        decoder_out_shape=(1, 3, args.frames, args.height, args.width),
        encoder_out_shape=(1, 48, encoder_latent_t, latent_h, latent_w),
        height=args.height,
        width=args.width,
        frames=args.frames,
    )


class MiniMaxH3TRTVideoPlanner:
    def __init__(self, decoder_tile_size: int = 256, encoder_tile_size: int = 256) -> None:
        self.vae_ratio = 16
        self.vae_ratio_t = 4
        self.clip_length = 17
        self.token_drop = 3
        self.frame_pre_padding = (-self.clip_length) % self.vae_ratio_t
        self.tokens_chunk_size = math.ceil(self.clip_length / self.vae_ratio_t)
        self.token_overlap = (-self.token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(self.token_overlap * self.vae_ratio_t - self.frame_pre_padding, 0)
        self.decoder_tile_size = decoder_tile_size
        self.encoder_tile_size = encoder_tile_size
        self.tile_overlap_min = 64

    def decode_temporal_chunks(self, z_len: int) -> tuple[int, int]:
        pseudo_total_tokens = z_len + self.token_drop
        pad_tokens = (-pseudo_total_tokens) % self.tokens_chunk_size
        pseudo_total_tokens += pad_tokens
        num_chunks = pseudo_total_tokens // self.tokens_chunk_size - int(self.token_drop > 0)
        if num_chunks < 1:
            pad_tokens += self.tokens_chunk_size
            num_chunks += 1
        return pad_tokens, num_chunks

    def decode_temporal_pad_frames(self, z_len: int, pad_tokens: int) -> int:
        if pad_tokens <= 0:
            return 0
        intra_tail = self.clip_length % self.vae_ratio_t
        if intra_tail == 0:
            return pad_tokens * self.vae_ratio_t
        z_len_before_pad = z_len - pad_tokens
        return sum(
            intra_tail if (z_len_before_pad + k) % self.tokens_chunk_size == 0 else self.vae_ratio_t
            for k in range(pad_tokens)
        )

    def decode_temporal_frame_plan(self, z_len: int, num_chunks: int, pad_tokens: int) -> int:
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1
        total_frames = 0
        final_overlap_frames = 0

        for i in range(num_chunks):
            t_start_idx = i * self.tokens_chunk_size
            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
            clip_token_len = max(0, min(t_end_idx, z_len) - min(t_start_idx, z_len))
            clip_frame_len = clip_token_len * self.vae_ratio_t

            for j in range(split_count):
                f_start_idx = j * chunk_dec
                f_end_idx = min(f_start_idx + chunk_dec, clip_frame_len)
                chunk_frames = max(0, f_end_idx - f_start_idx - self.frame_pre_padding)
                if j == 0:
                    total_frames += chunk_frames
                else:
                    final_overlap_frames = chunk_frames

        total_frames += final_overlap_frames
        return total_frames - self.decode_temporal_pad_frames(z_len, pad_tokens)

    def output_frames_for_latent_tokens(self, z_len: int) -> int:
        pad_tokens, num_chunks = self.decode_temporal_chunks(z_len)
        return self.decode_temporal_frame_plan(z_len + pad_tokens, num_chunks, pad_tokens)

    def latent_tokens_for_frames(self, frames: int) -> int:
        for z_len in range(1, max(8, frames) + 32):
            if self.output_frames_for_latent_tokens(z_len) == frames:
                return z_len
        raise ValueError(f"could not derive decoder latent tokens for {frames} frames")

    def split_tiles(self, input_len: int, tile_size: int) -> tuple[list[int], list[int], list[int]]:
        if tile_size >= input_len:
            return [0], [input_len], []
        n_tiles = math.ceil(input_len / tile_size)
        while True:
            overlaps = [self.tile_overlap_min] * (n_tiles - 1)
            remaining = tile_size * n_tiles - sum(overlaps) - input_len
            if remaining < 0:
                n_tiles += 1
            else:
                break
        for i in range(remaining // self.vae_ratio):
            overlaps[i % (n_tiles - 1)] += self.vae_ratio
        tile_start_idx = [0]
        for i in range(n_tiles - 1):
            tile_start_idx.append(tile_start_idx[-1] + tile_size - overlaps[i])
        return tile_start_idx, [tile_size] * n_tiles, overlaps


class MiniMaxH3TRTVideoVAE(MiniMaxH3TRTVideoPlanner):
    def __init__(
        self,
        decoder_runner: TensorRTRunner,
        encoder_runner: TensorRTRunner,
        model_root: Path,
        decoder_tile_size: int = 256,
        encoder_tile_size: int = 256,
    ) -> None:
        super().__init__(decoder_tile_size, encoder_tile_size)
        self.decoder_runner = decoder_runner
        self.encoder_runner = encoder_runner
        with (model_root / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        self.latents_mean_values = config["latents_mean"]
        self.latents_std_values = config["latents_std"]

    def latents_mean(self, target: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.latents_mean_values, device=target.device, dtype=target.dtype).view(1, -1, 1, 1, 1)

    def latents_std(self, target: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.latents_std_values, device=target.device, dtype=target.dtype).view(1, -1, 1, 1, 1)

    def pixel_mean(self, target: torch.Tensor) -> torch.Tensor:
        return torch.tensor((0.485, 0.456, 0.406), device=target.device, dtype=target.dtype).view(1, 3, 1, 1, 1)

    def pixel_std(self, target: torch.Tensor) -> torch.Tensor:
        return torch.tensor((0.229, 0.224, 0.225), device=target.device, dtype=target.dtype).view(1, 3, 1, 1, 1)

    def _decode_pixels(self, z: torch.Tensor) -> torch.Tensor:
        b, _, t, h, w = z.shape
        out_shape = (b, 3, t * self.vae_ratio_t, h * self.vae_ratio, w * self.vae_ratio)
        return self.decoder_runner(z, out_shape)

    def _encode_moments(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t, h, w = x.shape
        target_h, target_w = self.encoder_tile_size, self.encoder_tile_size
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        if pad_h > 0 or pad_w > 0:
            x_in = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, 0), mode="constant", value=0.0)
        else:
            x_in = x
        out_shape = (b, 48, math.ceil(t / self.vae_ratio_t), target_h // self.vae_ratio, target_w // self.vae_ratio)
        moments = self.encoder_runner(x_in, out_shape)
        out_h = math.ceil(h / self.vae_ratio)
        out_w = math.ceil(w / self.vae_ratio)
        return moments[..., :out_h, :out_w]

    def _normalize_pixels(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.pixel_mean(x)) / self.pixel_std(x)

    def _finalize_pixels(self, part: torch.Tensor) -> torch.Tensor:
        return (part * self.pixel_std(part) + self.pixel_mean(part)).clamp(0.0, 1.0)

    def blend(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        if blend_extent <= 0:
            return b

        weight = torch.arange(blend_extent, device=b.device, dtype=b.dtype) / blend_extent
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight = weight.view(shape)

        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        blended = torch.lerp(a[tuple(slice_a)], b[tuple(slice_b)], weight)
        if blend_extent < b.shape[dim]:
            slice_b_rest = [slice(None)] * b.ndim
            slice_b_rest[dim] = slice(blend_extent, None)
            return torch.cat([blended, b[tuple(slice_b_rest)]], dim=dim)
        return blended

    def tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        height, width = z.shape[-2] * self.vae_ratio, z.shape[-1] * self.vae_ratio
        y_idx, y_len, y_overlap = self.split_tiles(height, self.decoder_tile_size)
        x_idx, x_len, x_overlap = self.split_tiles(width, self.decoder_tile_size)

        canvas, row_tails, out_y = None, [], 0
        for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
            zi, zl = i_pos // self.vae_ratio, i_len // self.vae_ratio
            new_tails, left_tail, out_x = [], None, 0
            for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len)):
                zj, zw = j_pos // self.vae_ratio, j_len // self.vae_ratio
                tile = self._decode_pixels(z[..., zi : zi + zl, zj : zj + zw])

                if i < len(y_idx) - 1:
                    new_tails.append(tile[..., -y_overlap[i] :, :].clone())
                next_left_tail = tile[..., :, -x_overlap[j] :].clone() if j < len(x_idx) - 1 else None

                if i > 0:
                    tile = self.blend(row_tails[j], tile, y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(left_tail, tile, x_overlap[j - 1], dim=-1)
                left_tail = next_left_tail

                if i < len(y_idx) - 1:
                    tile = tile[..., : -y_overlap[i], :]
                if j < len(x_idx) - 1:
                    tile = tile[..., :, : -x_overlap[j]]

                if canvas is None:
                    canvas = torch.zeros(*tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device)
                canvas[..., out_y : out_y + tile.shape[-2], out_x : out_x + tile.shape[-1]].copy_(tile)
                out_x += tile.shape[-1]
            row_tails = new_tails
            out_y += tile.shape[-2]
        return canvas

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2], x.shape[-1]
        y_idx, y_len, y_overlap = self.split_tiles(height, self.encoder_tile_size)
        x_idx, x_len, x_overlap = self.split_tiles(width, self.encoder_tile_size)

        rows = []
        for i_pos, i_len in zip(y_idx, y_len):
            row = []
            for j_pos, j_len in zip(x_idx, x_len):
                tile = x[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len]
                row.append(self._encode_moments(tile))
            rows.append(row)

        latent_y_overlap = [overlap // self.vae_ratio for overlap in y_overlap]
        latent_x_overlap = [overlap // self.vae_ratio for overlap in x_overlap]

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend(rows[i - 1][j], tile, latent_y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(row[j - 1], tile, latent_x_overlap[j - 1], dim=-1)
                if i < len(rows) - 1:
                    tile = tile[..., : -latent_y_overlap[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, : -latent_x_overlap[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def decode_temporal(self, z: torch.Tensor) -> torch.Tensor:
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1

        pad_tokens, num_chunks = self.decode_temporal_chunks(z.shape[2])
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

        dec_chunks, dec_overlap = [], None
        for i in range(num_chunks):
            t_start = i * self.tokens_chunk_size
            t_end = t_start + self.tokens_chunk_size + self.token_overlap
            clip_z = z[:, :, t_start:t_end, :, :]
            clip_dec = self.tiled_decode(clip_z)

            for j in range(split_count):
                f_start = j * chunk_dec
                f_end = min(f_start + chunk_dec, clip_dec.shape[2])
                chunk = clip_dec[:, :, f_start:f_end, :, :]
                chunk = chunk[:, :, self.frame_pre_padding :, :, :]

                if j == 0:
                    if dec_overlap is not None:
                        chunk = self.blend(dec_overlap, chunk, self.frame_overlap, dim=-3)
                        dec_overlap = None
                    dec_chunks.append(self._finalize_pixels(chunk))
                else:
                    dec_overlap = chunk.contiguous()

            if i == num_chunks - 1 and dec_overlap is not None:
                dec_chunks.append(self._finalize_pixels(dec_overlap))

        dec = torch.cat(dec_chunks, dim=2)
        pad_frames = self.decode_temporal_pad_frames(z.shape[2], pad_tokens)
        if pad_frames > 0:
            dec = dec[:, :, :-pad_frames, :, :]
        return dec

    def encode_temporal(self, x: torch.Tensor) -> torch.Tensor:
        z_list = []
        num_clips = math.ceil(x.shape[2] / self.clip_length)
        for i in range(num_clips):
            clip_x = x[:, :, i * self.clip_length : (i + 1) * self.clip_length, :, :]
            if clip_x.shape[2] < self.clip_length:
                pad_frames = clip_x[:, :, -1:].repeat(1, 1, self.clip_length - clip_x.shape[2], 1, 1)
                clip_x = torch.cat([clip_x, pad_frames], dim=2)
            z_list.append(self.tiled_encode(self._normalize_pixels(clip_x)))

        z = torch.cat(z_list, dim=2)
        if self.token_drop > 0:
            z = z[:, :, : -self.token_drop]
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z * self.latents_std(z) + self.latents_mean(z)
        if z.shape[2] == 1:
            z_pad = z.repeat(1, 1, 7, 1, 1)
            return self._finalize_pixels(self.tiled_decode(z_pad)[:, :, -1:, :, :])
        return self.decode_temporal(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(2)
        if x.shape[2] == 1:
            moments = self.tiled_encode(self._normalize_pixels(x))[:, :, -1:, :, :]
        else:
            moments = self.encode_temporal(x)
        mean = torch.chunk(moments, 2, dim=1)[0]
        return (mean - self.latents_mean(mean)) / self.latents_std(mean)


def cuda_timed(fn, warmup: int, runs: int, stream_fn=None) -> tuple[TimedResult, torch.Tensor]:
    last = None
    for _ in range(warmup):
        last = fn()
    stream = stream_fn() if stream_fn is not None else None
    stream.synchronize() if stream is not None else torch.cuda.synchronize()

    elapsed = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(runs):
        starter.record(stream)
        last = fn()
        ender.record(stream)
        stream.synchronize() if stream is not None else torch.cuda.synchronize()
        elapsed.append(starter.elapsed_time(ender))

    return (
        TimedResult(
            mean_ms=float(statistics.mean(elapsed)),
            median_ms=float(statistics.median(elapsed)),
            min_ms=float(min(elapsed)),
            max_ms=float(max(elapsed)),
            runs=runs,
        ),
        last,
    )


def cleanup_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def load_original_vae(model_root: Path) -> torch.nn.Module:
    package_root = model_root.parent
    sys.path.insert(0, str(package_root))
    from video_vae.minimax_h3_video_vae import MiniMaxH3VideoVAE

    model = MiniMaxH3VideoVAE.from_pretrained(str(model_root))
    return model.eval().to(device="cuda", dtype=torch.float16)


def benchmark_trt(args, shapes: ShapeConfig, z_cpu: torch.Tensor, x_cpu: torch.Tensor) -> dict:
    cleanup_cuda()
    decoder = TensorRTRunner(args.decoder_engine, "latent_tile", "pixel_tile")
    encoder = TensorRTRunner(args.encoder_engine, "pixel_tile", "moments_tile")
    z = z_cpu.to("cuda", non_blocking=True)
    x = x_cpu.to("cuda", non_blocking=True)

    decoder_result, decoder_out = cuda_timed(
        lambda: decoder(z, shapes.decoder_out_shape), args.warmup, args.runs, lambda: decoder.stream
    )
    encoder_result, encoder_out = cuda_timed(
        lambda: encoder(x, shapes.encoder_out_shape), args.warmup, args.runs, lambda: encoder.stream
    )
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    result = {
        "decoder": decoder_result.__dict__,
        "encoder": encoder_result.__dict__,
        "peak_memory_bytes": int(peak),
        "decoder_out": decoder_out.detach().cpu(),
        "encoder_out": encoder_out.detach().cpu(),
    }
    del decoder, encoder, z, x, decoder_out, encoder_out
    cleanup_cuda()
    return result


def benchmark_trt_project(args, shapes: ShapeConfig, z_cpu: torch.Tensor, x_cpu: torch.Tensor) -> dict:
    cleanup_cuda()
    vae = MiniMaxH3TRTVideoVAE(
        TensorRTRunner(args.decoder_engine, "latent_tile", "pixel_tile"),
        TensorRTRunner(args.encoder_engine, "pixel_tile", "moments_tile"),
        args.original_vae,
        args.decoder_tile_size,
        args.encoder_tile_size,
    )
    z = z_cpu.to("cuda", non_blocking=True)
    x = x_cpu.to("cuda", non_blocking=True)

    decoder_result, decoder_out = cuda_timed(lambda: vae.decode(z), args.warmup, args.runs)
    encoder_result, encoder_out = cuda_timed(lambda: vae.encode(x), args.warmup, args.runs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    result = {
        "decoder": decoder_result.__dict__,
        "encoder": encoder_result.__dict__,
        "peak_memory_bytes": int(peak),
        "decoder_output_shape": tuple(decoder_out.shape),
        "encoder_output_shape": tuple(encoder_out.shape),
    }
    del vae, z, x, decoder_out, encoder_out
    cleanup_cuda()
    return result


def benchmark_original(args, z_cpu: torch.Tensor, x_cpu: torch.Tensor) -> dict:
    cleanup_cuda()
    model = load_original_vae(args.original_vae)
    z = z_cpu.to("cuda", non_blocking=True)
    x = x_cpu.to("cuda", non_blocking=True)

    decoder_result, decoder_out = cuda_timed(lambda: model.decode(z), args.warmup, args.runs)
    encoder_result, encoder_out = cuda_timed(lambda: model.encode(x), args.warmup, args.runs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    result = {
        "decoder": decoder_result.__dict__,
        "encoder": encoder_result.__dict__,
        "peak_memory_bytes": int(peak),
        "decoder_out": decoder_out.detach().cpu(),
        "encoder_out": encoder_out.detach().cpu(),
    }
    del model, z, x, decoder_out, encoder_out
    cleanup_cuda()
    return result


def benchmark_original_project(args, shapes: ShapeConfig, z_cpu: torch.Tensor, x_cpu: torch.Tensor) -> dict:
    cleanup_cuda()
    torch.backends.cudnn.benchmark = args.pytorch_cudnn_benchmark
    if hasattr(torch.backends.cudnn, "benchmark_limit"):
        torch.backends.cudnn.benchmark_limit = args.pytorch_cudnn_benchmark_limit
    model = load_original_vae(args.original_vae)
    pytorch_decoder_tile_size = (
        args.pytorch_decoder_tile_size or args.decoder_tile_size
    )
    pytorch_encoder_tile_size = (
        args.pytorch_encoder_tile_size or args.encoder_tile_size
    )
    model.model.decoder_tile_size = pytorch_decoder_tile_size
    model.model.tile_size = pytorch_encoder_tile_size
    model.model.stack_tiling = args.pytorch_stack_tiling
    decoder_compile_batch = 1
    if args.pytorch_stack_tiling and shapes.height is not None:
        decoder_compile_batch = (
            len(model.model.split_tiles(shapes.height, True)[0])
            * len(model.model.split_tiles(shapes.width, True)[0])
        )
    if args.pytorch_whole_decoder_compile:
        os.environ["MINIMAX_H3_TORCH_SDPA_BACKEND"] = args.pytorch_sdpa_backend
        os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")
        optimize_minimax_h3_decoder(
            model,
            expected_shape=(
                decoder_compile_batch,
                24,
                7,
                pytorch_decoder_tile_size // SPATIAL_RATIO,
                pytorch_decoder_tile_size // SPATIAL_RATIO,
            ),
            attention_in_graph=args.pytorch_attention_in_graph,
            cuda_graph=args.pytorch_decoder_cuda_graph,
            backend=args.pytorch_compile_backend,
            mode=args.pytorch_compile_mode,
        )
    if args.pytorch_whole_encoder_compile:
        os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")
        optimize_minimax_h3_encoder(
            model,
            expected_shape=(
                args.pytorch_encoder_microbatch,
                3,
                17,
                pytorch_encoder_tile_size,
                pytorch_encoder_tile_size,
            ),
            channels_last_3d=args.pytorch_encoder_channels_last_3d,
            quant_conv_in_graph=args.pytorch_encoder_quant_conv_in_graph,
            cuda_graph=args.pytorch_encoder_cuda_graph,
            backend=args.pytorch_encoder_compile_backend,
            mode=args.pytorch_encoder_compile_mode,
        )
    encoder_video_prepost = None
    if args.pytorch_encoder_compiled_prepost:
        encoder_video_prepost = build_compiled_encoder_video_prepost(
            model,
            backend=args.pytorch_encoder_compile_backend,
            mode="default",
        )
    z = z_cpu.to("cuda", non_blocking=True)
    x = x_cpu.to("cuda", non_blocking=True)
    latents_mean = torch.tensor(
        json.loads((args.original_vae / "config.json").read_text(encoding="utf-8"))["latents_mean"],
        device=z.device,
        dtype=z.dtype,
    ).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(
        json.loads((args.original_vae / "config.json").read_text(encoding="utf-8"))["latents_std"],
        device=z.device,
        dtype=z.dtype,
    ).view(1, -1, 1, 1, 1)
    pixel_mean = torch.tensor((0.485, 0.456, 0.406), device=x.device, dtype=x.dtype).view(1, 3, 1, 1, 1)
    pixel_std = torch.tensor((0.229, 0.224, 0.225), device=x.device, dtype=x.dtype).view(1, 3, 1, 1, 1)

    decoder_result, decoder_out = cuda_timed(
        lambda: model.decode_base(z * latents_std + latents_mean, frame_num=shapes.frames),
        args.warmup,
        args.runs,
    )
    def encode_pytorch_impl():
        if args.pytorch_encoder_compiled_prepost:
            preprocessor, postprocessor = encoder_video_prepost
            return encode_video_mean_compiled_prepost(
                model, x, pixel_mean, pixel_std, latents_mean,
                latents_std, preprocessor, postprocessor
            )
        normalized = (x - pixel_mean) / pixel_std
        if not args.pytorch_encoder_use_mean:
            return model.encode_base(normalized)
        if args.pytorch_encoder_fast_temporal:
            if args.pytorch_encoder_microbatch > 1:
                moments = encode_temporal_single_tile_microbatch(
                    model, normalized, args.pytorch_encoder_microbatch
                )
            else:
                moments = encode_temporal_single_tile_channels_last(
                    model, normalized
                )
        else:
            moments = model.model.encode_temporal(normalized)
        mean = torch.chunk(moments, 2, dim=1)[0]
        return (mean - latents_mean) / latents_std

    def encode_pytorch():
        if args.pytorch_encoder_inference_mode:
            with torch.inference_mode():
                return encode_pytorch_impl()
        return encode_pytorch_impl()

    encoder_result, encoder_out = cuda_timed(
        encode_pytorch,
        args.warmup,
        args.runs,
    )
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    result = {
        "decoder": decoder_result.__dict__,
        "encoder": encoder_result.__dict__,
        "peak_memory_bytes": int(peak),
        "decoder_output_shape": tuple(decoder_out.shape),
        "encoder_output_shape": tuple(encoder_out.shape),
        "decoder_optimization": refresh_cuda_graph_info(model),
        "encoder_optimization": refresh_encoder_cuda_graph_info(model),
    }
    del model, z, x, decoder_out, encoder_out
    cleanup_cuda()
    return result


def summarize(args, shapes: ShapeConfig, trt_result: dict, original_result: dict | None, started_at: str) -> dict:
    env = {
        "started_at": started_at,
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "command": " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv]),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_toolkit_from_nvcc": os.popen("nvcc --version | grep 'release' | tail -n 1").read().strip(),
        "tensorrt": trt.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver_from_nvidia_smi": os.popen(
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1"
        ).read().strip(),
    }
    report = {
        "environment": env,
        "config": {
            "runs": args.runs,
            "warmup": args.warmup,
            "dtype": "float16",
            "mode": shapes.mode,
            "height": shapes.height,
            "width": shapes.width,
            "frames": shapes.frames,
            "decoder_tile_size": args.decoder_tile_size,
            "encoder_tile_size": args.encoder_tile_size,
            "pytorch_decoder_tile_size": (
                args.pytorch_decoder_tile_size or args.decoder_tile_size
            ),
            "pytorch_stack_tiling": args.pytorch_stack_tiling,
            "pytorch_encoder_tile_size": (
                args.pytorch_encoder_tile_size or args.encoder_tile_size
            ),
            "pytorch_whole_decoder_compile": args.pytorch_whole_decoder_compile,
            "pytorch_attention_in_graph": args.pytorch_attention_in_graph,
            "pytorch_decoder_cuda_graph": args.pytorch_decoder_cuda_graph,
            "pytorch_compile_backend": args.pytorch_compile_backend,
            "pytorch_compile_mode": args.pytorch_compile_mode,
            "pytorch_sdpa_backend": args.pytorch_sdpa_backend,
            "pytorch_whole_encoder_compile": args.pytorch_whole_encoder_compile,
            "pytorch_encoder_channels_last_3d": args.pytorch_encoder_channels_last_3d,
            "pytorch_encoder_quant_conv_in_graph": args.pytorch_encoder_quant_conv_in_graph,
            "pytorch_encoder_fast_temporal": args.pytorch_encoder_fast_temporal,
            "pytorch_encoder_microbatch": args.pytorch_encoder_microbatch,
            "pytorch_encoder_compiled_prepost": args.pytorch_encoder_compiled_prepost,
            "pytorch_encoder_inference_mode": args.pytorch_encoder_inference_mode,
            "pytorch_cudnn_benchmark": args.pytorch_cudnn_benchmark,
            "pytorch_cudnn_benchmark_limit": args.pytorch_cudnn_benchmark_limit,
            "pytorch_encoder_cuda_graph": args.pytorch_encoder_cuda_graph,
            "pytorch_encoder_compile_backend": args.pytorch_encoder_compile_backend,
            "pytorch_encoder_compile_mode": args.pytorch_encoder_compile_mode,
            "pytorch_encoder_use_mean": args.pytorch_encoder_use_mean,
            "decoder_input_shape": shapes.decoder_shape,
            "decoder_output_shape": shapes.decoder_out_shape,
            "encoder_input_shape": shapes.encoder_shape,
            "encoder_output_shape": shapes.encoder_out_shape,
            "decoder_engine_calls_per_run": shapes.decoder_engine_calls_per_run,
            "encoder_engine_calls_per_run": shapes.encoder_engine_calls_per_run,
            "decoder_engine": str(args.decoder_engine),
            "encoder_engine": str(args.encoder_engine),
            "original_vae": str(args.original_vae) if original_result else None,
        },
        "trt": {
            "decoder": trt_result["decoder"],
            "encoder": trt_result["encoder"],
            "peak_memory_bytes": trt_result["peak_memory_bytes"],
            "decoder_output_shape": trt_result.get("decoder_output_shape"),
            "encoder_output_shape": trt_result.get("encoder_output_shape"),
        },
    }
    if original_result is not None:
        report["original"] = {
            "decoder": original_result["decoder"],
            "encoder": original_result["encoder"],
            "peak_memory_bytes": original_result["peak_memory_bytes"],
            "decoder_output_shape": original_result.get("decoder_output_shape"),
            "encoder_output_shape": original_result.get("encoder_output_shape"),
            "decoder_optimization": original_result.get("decoder_optimization"),
            "encoder_optimization": original_result.get("encoder_optimization"),
        }
        report["speedup"] = {
            "decoder_mean": original_result["decoder"]["mean_ms"] / trt_result["decoder"]["mean_ms"],
            "encoder_mean": original_result["encoder"]["mean_ms"] / trt_result["encoder"]["mean_ms"],
            "decoder_median": original_result["decoder"]["median_ms"] / trt_result["decoder"]["median_ms"],
            "encoder_median": original_result["encoder"]["median_ms"] / trt_result["encoder"]["median_ms"],
        }
        if "decoder_out" in trt_result and "encoder_out" in trt_result:
            report["accuracy"] = {
                "decoder_max_abs_diff": float(
                    (trt_result["decoder_out"] - original_result["decoder_out"]).abs().max()
                ),
                "encoder_max_abs_diff": float(
                    (trt_result["encoder_out"] - original_result["encoder_out"]).abs().max()
                ),
            }
    return report


def write_markdown(report: dict, path: Path) -> None:
    speedup = report.get("speedup", {})
    accuracy = report.get("accuracy", {})
    original = report.get("original")
    lines = [
        "# ComfyUI-H3VAE_TRT local benchmark",
        "",
        "## Environment",
        f"- Python: `{report['environment']['python']}` (`{report['environment']['python_executable']}`)",
        f"- Python prefix: `{report['environment']['python_prefix']}`",
        f"- Conda env: `{report['environment']['conda_default_env']}`",
        f"- Platform: `{report['environment']['platform']}`",
        f"- GPU: `{report['environment']['gpu']}`, driver `{report['environment']['driver_from_nvidia_smi']}`",
        f"- PyTorch: `{report['environment']['torch']}`, torch CUDA: `{report['environment']['torch_cuda']}`",
        f"- CUDA toolkit: `{report['environment']['cuda_toolkit_from_nvcc']}`",
        f"- TensorRT: `{report['environment']['tensorrt']}`",
        "",
        "## Command/config",
        f"- Command: `{report['environment']['command']}`",
        f"- Mode: `{report['config']['mode']}`",
        f"- Runs: `{report['config']['runs']}`, warmup: `{report['config']['warmup']}`, dtype: `float16`",
        f"- Requested video shape: `{report['config']['width']}x{report['config']['height']}x{report['config']['frames']}`",
        f"- TensorRT tile sizes: decoder `{report['config']['decoder_tile_size']}`, encoder `{report['config']['encoder_tile_size']}`",
        f"- PyTorch tile sizes: decoder `{report['config']['pytorch_decoder_tile_size']}`, encoder `{report['config']['pytorch_encoder_tile_size']}`",
        f"- Decoder shape: `{tuple(report['config']['decoder_input_shape'])}` -> `{tuple(report['config']['decoder_output_shape'])}`",
        f"- Encoder shape: `{tuple(report['config']['encoder_input_shape'])}` -> `{tuple(report['config']['encoder_output_shape'])}`",
        f"- TensorRT engine calls per run: decoder `{report['config']['decoder_engine_calls_per_run']}`, encoder `{report['config']['encoder_engine_calls_per_run']}`",
        f"- Decoder engine: `{report['config']['decoder_engine']}`",
        f"- Encoder engine: `{report['config']['encoder_engine']}`",
    ]
    if original:
        lines.append(f"- Original VAE: `{report['config']['original_vae']}`")
        if original.get("decoder_optimization"):
            lines.append(
                f"- PyTorch decoder optimization: `{original['decoder_optimization']}`"
            )
        if original.get("encoder_optimization"):
            lines.append(
                f"- PyTorch encoder optimization: `{original['encoder_optimization']}`"
            )
    lines += [
        "",
        "## Results",
        "| Path | Decoder mean/median/min ms | Encoder mean/median/min ms | Peak torch CUDA alloc |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| TensorRT | {report['trt']['decoder']['mean_ms']:.3f} / "
            f"{report['trt']['decoder']['median_ms']:.3f} / {report['trt']['decoder']['min_ms']:.3f} | "
            f"{report['trt']['encoder']['mean_ms']:.3f} / {report['trt']['encoder']['median_ms']:.3f} / "
            f"{report['trt']['encoder']['min_ms']:.3f} | {report['trt']['peak_memory_bytes'] / 1024**3:.2f} GiB |"
        ),
    ]
    if original:
        lines.append(
            f"| Original PyTorch | {original['decoder']['mean_ms']:.3f} / "
            f"{original['decoder']['median_ms']:.3f} / {original['decoder']['min_ms']:.3f} | "
            f"{original['encoder']['mean_ms']:.3f} / {original['encoder']['median_ms']:.3f} / "
            f"{original['encoder']['min_ms']:.3f} | {original['peak_memory_bytes'] / 1024**3:.2f} GiB |"
        )
        lines += [
            "",
            "## Output shapes",
            f"- TensorRT decoder output: `{tuple(report['trt']['decoder_output_shape']) if report['trt']['decoder_output_shape'] else None}`",
            f"- TensorRT encoder output: `{tuple(report['trt']['encoder_output_shape']) if report['trt']['encoder_output_shape'] else None}`",
            f"- Original decoder output: `{tuple(original['decoder_output_shape']) if original['decoder_output_shape'] else None}`",
            f"- Original encoder output: `{tuple(original['encoder_output_shape']) if original['encoder_output_shape'] else None}`",
            "",
            "## Conclusion",
            (
                f"- Decoder mean speedup: `{speedup['decoder_mean']:.2f}x`; "
                f"encoder mean speedup: `{speedup['encoder_mean']:.2f}x`."
            ),
        ]
        if accuracy:
            lines.append(
                f"- Max abs diff vs original: decoder `{accuracy['decoder_max_abs_diff']:.6f}`, "
                f"encoder `{accuracy['encoder_max_abs_diff']:.6f}`."
            )
        else:
            lines.append("- Accuracy diff skipped for project_tiled_video mode; this run reports performance and output shapes.")
    else:
        lines += [
            "",
            "## Conclusion",
            "- Original PyTorch comparison was skipped; this run only verifies the TensorRT project path.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder-engine", type=Path, default=Path("bench_assets/engines/minimax_h3_vae_decoder.engine"))
    parser.add_argument("--encoder-engine", type=Path, default=Path("bench_assets/engines/minimax_h3_vae_encoder.engine"))
    parser.add_argument("--original-vae", type=Path,
                        default=Path(os.environ.get("H3_VAE_MODEL_CODE_DIR", "video_vae")))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--decoder-tile-size", type=int, default=256)
    parser.add_argument("--encoder-tile-size", type=int, default=256)
    parser.add_argument("--pytorch-decoder-tile-size", type=int)
    parser.add_argument("--pytorch-encoder-tile-size", type=int)
    parser.add_argument("--full-video", action="store_true")
    parser.add_argument("--pytorch-whole-decoder-compile", action="store_true")
    parser.add_argument("--pytorch-stack-tiling", action="store_true")
    parser.add_argument("--pytorch-attention-in-graph", action="store_true")
    parser.add_argument("--pytorch-decoder-cuda-graph", action="store_true")
    parser.add_argument("--pytorch-compile-backend", default="inductor")
    parser.add_argument(
        "--pytorch-compile-mode", default="max-autotune-no-cudagraphs"
    )
    parser.add_argument("--pytorch-sdpa-backend", default="flash")
    parser.add_argument("--pytorch-whole-encoder-compile", action="store_true")
    parser.add_argument(
        "--pytorch-encoder-channels-last-3d",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pytorch-encoder-quant-conv-in-graph", action="store_true"
    )
    parser.add_argument("--pytorch-encoder-fast-temporal", action="store_true")
    parser.add_argument("--pytorch-encoder-microbatch", type=int, default=1)
    parser.add_argument("--pytorch-encoder-compiled-prepost", action="store_true")
    parser.add_argument("--pytorch-encoder-inference-mode", action="store_true")
    parser.add_argument("--pytorch-cudnn-benchmark", action="store_true")
    parser.add_argument("--pytorch-cudnn-benchmark-limit", type=int, default=10)
    parser.add_argument("--pytorch-encoder-cuda-graph", action="store_true")
    parser.add_argument(
        "--pytorch-encoder-use-mean",
        action="store_true",
        help="match the TensorRT path by returning normalized posterior mean",
    )
    parser.add_argument("--pytorch-encoder-compile-backend", default="inductor")
    parser.add_argument(
        "--pytorch-encoder-compile-mode",
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--skip-original", action="store_true")
    parser.add_argument("--json-out", type=Path, default=Path("bench_h3vae_trt_results.json"))
    parser.add_argument("--md-out", type=Path, default=Path("bench_h3vae_trt_report.md"))
    args = parser.parse_args()

    if (
        args.pytorch_attention_in_graph or args.pytorch_decoder_cuda_graph
    ) and not args.pytorch_whole_decoder_compile:
        parser.error(
            "--pytorch-attention-in-graph and --pytorch-decoder-cuda-graph "
            "require --pytorch-whole-decoder-compile"
        )
    if args.pytorch_encoder_compiled_prepost and not (
        args.pytorch_whole_encoder_compile
        and args.pytorch_encoder_channels_last_3d
        and args.pytorch_encoder_use_mean
    ):
        parser.error(
            "--pytorch-encoder-compiled-prepost requires whole encoder "
            "compile, channels-last-3d, and --pytorch-encoder-use-mean"
        )
    if args.pytorch_encoder_fast_temporal and not (
        args.pytorch_whole_encoder_compile
        and args.pytorch_encoder_channels_last_3d
        and args.pytorch_encoder_use_mean
    ):
        parser.error(
            "--pytorch-encoder-fast-temporal requires whole encoder "
            "compile, channels-last-3d, and --pytorch-encoder-use-mean"
        )
    if args.pytorch_encoder_microbatch < 1:
        parser.error("--pytorch-encoder-microbatch must be >= 1")
    if args.pytorch_encoder_microbatch > 1 and not args.pytorch_encoder_fast_temporal:
        parser.error(
            "--pytorch-encoder-microbatch > 1 requires "
            "--pytorch-encoder-fast-temporal"
        )
    if args.pytorch_encoder_cuda_graph and not args.pytorch_whole_encoder_compile:
        parser.error(
            "--pytorch-encoder-cuda-graph requires "
            "--pytorch-whole-encoder-compile"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    shapes = shape_config_from_args(args)
    torch.manual_seed(args.seed)
    z_cpu = torch.randn(shapes.decoder_shape, dtype=torch.float16, pin_memory=True)
    x_cpu = torch.randn(shapes.encoder_shape, dtype=torch.float16, pin_memory=True)

    started_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
    if args.full_video:
        trt_result = benchmark_trt_project(args, shapes, z_cpu, x_cpu)
        original_result = None if args.skip_original else benchmark_original_project(args, shapes, z_cpu, x_cpu)
    else:
        trt_result = benchmark_trt(args, shapes, z_cpu, x_cpu)
        original_result = None if args.skip_original else benchmark_original(args, z_cpu, x_cpu)
    report = summarize(args, shapes, trt_result, original_result, started_at)

    json_report = json.loads(json.dumps(report, default=str))
    args.json_out.write_text(json.dumps(json_report, indent=2), encoding="utf-8")
    write_markdown(json_report, args.md_out)
    print(args.md_out)
    print(args.json_out)


if __name__ == "__main__":
    main()
