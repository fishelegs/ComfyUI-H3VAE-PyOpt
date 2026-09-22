#!/usr/bin/env python3
"""Production-path A/B validation for the optional INT8 H3 VAE decoder.

The script constructs two independent H3VAEPyOptRuntime instances: the
compiled default decoder and the explicit INT8 FFN decoder. Each source video
is decoded as exact RGB uint8, encoded once by the default runtime, and then
decoded from the same explicitly FP16-cast latent by both runtimes. Only
runtime.decode is included in the alternating CUDA-event timing.

This is an opt-in validation tool. It does not change the runtime defaults,
ComfyUI nodes, model source, weights, or shared environment. PyAV and numpy
are imported only when a video is actually read; Pillow is not required.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from h3vae_runtime import DECODER_COMPILE_MODE, H3VAEPyOptRuntime


VARIANTS = ("default", "INT8both")
PSNR_THRESHOLD_DB = 30.0
MSE_AT_30_DB = 10.0 ** (-PSNR_THRESHOLD_DB / 10.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_snapshot() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    device = torch.cuda.current_device()
    result = {
        "available": True,
        "device_index": device,
        "name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        completed = None
    if completed is not None:
        result["nvidia_smi"] = completed.stdout.strip()
        if completed.returncode != 0:
            result["nvidia_smi_error"] = completed.stderr.strip()
    return result


def _environment() -> dict:
    try:
        ck_version = importlib.metadata.version("comfy-kitchen")
    except importlib.metadata.PackageNotFoundError:
        ck_version = None
    try:
        import comfy_kitchen as ck
    except ImportError:
        ck = None
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        triton_version = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_hip": torch.version.hip,
        "triton": triton_version,
        "comfy_kitchen_version": ck_version,
        "comfy_kitchen_file": None if ck is None else getattr(ck, "__file__", None),
        "sdpa_backend_env": os.environ.get(
            "MINIMAX_H3_TORCH_SDPA_BACKEND", "auto"
        ),
        "fp32_norm_env": os.environ.get(
            "MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM", "1"
        ),
        "allow_fp16_accumulation": bool(
            getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)
        ),
        "gpu_before_runtime": _gpu_snapshot(),
    }


def _rgb_uint8_to_reference(frames: torch.Tensor) -> torch.Tensor:
    """Convert exact RGB uint8 [T,H,W,C] to FP32 [1,C,T,H,W] / 255."""
    if frames.ndim != 4 or tuple(frames.shape[-1:]) != (3,):
        raise RuntimeError(f"expected RGB [T,H,W,3], got {tuple(frames.shape)}")
    if frames.dtype != torch.uint8:
        raise RuntimeError(f"expected uint8 RGB frames, got {frames.dtype}")
    return frames.permute(3, 0, 1, 2).contiguous().float().div(255.0).unsqueeze(0)


def _read_rgb_video(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Read source RGB exactly; return FP32 reference and FP16 encoder input."""
    try:
        import av
        import numpy as np
    except ImportError as exc:  # pragma: no cover - media-only dependency
        raise RuntimeError("PyAV and numpy are required for MP4 input") from exc

    frames = []
    pts = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise RuntimeError(f"video contains no video stream: {path}")
        stream = container.streams.video[0]
        rate = stream.average_rate
        fps = (
            None
            if rate is None
            else {"num": int(rate.numerator), "den": int(rate.denominator)}
        )
        stream_meta = {
            "codec": str(getattr(stream.codec_context, "name", "")),
            "width": int(stream.width),
            "height": int(stream.height),
            "frames_reported": int(stream.frames or 0),
            "average_rate": str(rate),
            "fps": fps,
            "time_base": str(stream.time_base),
            "pixel_format": str(getattr(stream.codec_context, "pix_fmt", "")),
            "color_range": str(getattr(stream, "color_range", None)),
            "colorspace": str(getattr(stream, "colorspace", None)),
            "color_primaries": str(getattr(stream, "color_primaries", None)),
            "color_transfer": str(getattr(stream, "color_trc", None)),
            "stream_metadata": {
                str(key): str(value) for key, value in (stream.metadata or {}).items()
            },
        }
        time_base = float(stream.time_base) if stream.time_base is not None else None
        pts_time_s = []
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            frame_pts = None if frame.pts is None else int(frame.pts)
            pts.append(frame_pts)
            pts_time_s.append(
                None
                if frame_pts is None or time_base is None
                else frame_pts * time_base
            )
        duration = container.duration

    if not frames:
        raise RuntimeError(f"video contains no decoded frames: {path}")
    array = np.stack(frames, axis=0)
    if array.ndim != 4 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise RuntimeError(
            f"expected decoded uint8 RGB [T,H,W,3], got {array.shape} {array.dtype}"
        )
    raw = torch.from_numpy(array)
    reference = _rgb_uint8_to_reference(raw)
    # This half conversion is only for encoder input. The reference above is
    # never reconstructed from half values, so source metrics stay exact.
    encoder_input = reference.mul(2.0).sub(1.0).to(dtype=torch.float16)
    metadata = {
        **stream_meta,
        "decoded_frames": int(array.shape[0]),
        "decoded_height": int(array.shape[1]),
        "decoded_width": int(array.shape[2]),
        "reference_layout": "B,C,T,H,W",
        "reference_dtype": "torch.float32",
        "reference_conversion": "PyAV RGB uint8 / 255.0",
        "encoder_input_dtype": str(encoder_input.dtype),
        "encoder_input_range": "[-1,1]",
        "pts": pts,
        "pts_time_s": pts_time_s,
        "container_duration_s": (
            None if duration is None else float(duration) / 1_000_000.0
        ),
    }
    del raw, array, frames
    return reference, encoder_input, metadata


def _validate_shape(reference: torch.Tensor) -> dict:
    if reference.ndim != 5 or tuple(reference.shape[:2]) != (1, 3):
        raise RuntimeError(f"unsupported reference layout: {tuple(reference.shape)}")
    _, _, frames, height, width = reference.shape
    if height % 16 or width % 16:
        raise RuntimeError(
            f"source spatial shape {(height, width)} is not divisible by 16; "
            "the benchmark never resizes or crops"
        )
    if frames != 1 and (frames - 5) % 17:
        raise RuntimeError(
            f"source frame count {frames} is not an H3 shape (1 or 17*k+5); "
            "the benchmark never drops or pads frames"
        )
    return {
        "reference_shape": list(reference.shape),
        "frames": int(frames),
        "height": int(height),
        "width": int(width),
        "spatial_divisible_by_16": True,
        "temporal_shape_supported": True,
        "resized": False,
        "cropped": False,
        "frames_dropped": 0,
    }


def _psnr_record(mse: float) -> dict:
    if mse == 0.0:
        return {
            "mse": 0.0,
            "psnr_db": None,
            "psnr_infinite": True,
            "below_30db": False,
        }
    psnr = 10.0 * math.log10(1.0 / mse)
    return {
        "mse": mse,
        "psnr_db": psnr,
        "psnr_infinite": False,
        "below_30db": bool(mse > MSE_AT_30_DB),
    }


def _quality_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Compute JSON-safe pixel metrics, including every frame's PSNR."""
    shape_match = tuple(reference.shape) == tuple(candidate.shape)
    result = {
        "shape_match": shape_match,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "metric_space": "float32 RGB [0,1]",
        "peak_value": 1.0,
        "reference_finite": bool(torch.isfinite(reference).all()),
        "candidate_finite": bool(torch.isfinite(candidate).all()),
        "reference_range": None,
        "candidate_range": None,
        "range_ok": False,
        "finite": False,
        "global_mse": None,
        "global_psnr_db": None,
        "global_psnr_infinite": False,
        "mean_frame_psnr_db": None,
        "mean_frame_psnr_infinite": False,
        "per_frame": [],
    }
    if not shape_match:
        raise RuntimeError(
            f"decoder output shape mismatch: {tuple(reference.shape)} vs "
            f"{tuple(candidate.shape)}"
        )
    if not result["reference_finite"] or not result["candidate_finite"]:
        raise RuntimeError("non-finite reference or decoder output")
    result["reference_range"] = [
        float(reference.min().item()),
        float(reference.max().item()),
    ]
    result["candidate_range"] = [
        float(candidate.min().item()),
        float(candidate.max().item()),
    ]
    result["range_ok"] = all(
        0.0 <= value <= 1.0
        for pair in (result["reference_range"], result["candidate_range"])
        for value in pair
    )
    if not result["range_ok"]:
        raise RuntimeError(
            "quality metrics require finalized pixel values in [0,1], got "
            f"reference={result['reference_range']} candidate={result['candidate_range']}"
        )

    total_abs = 0.0
    total_sq = 0.0
    total_count = 0
    max_abs = 0.0
    frame_psnr = []
    for frame_index in range(reference.shape[2]):
        diff = (
            reference[:, :, frame_index].float()
            - candidate[:, :, frame_index].float()
        )
        if not bool(torch.isfinite(diff).all()):
            raise RuntimeError(f"non-finite pixel difference at frame {frame_index}")
        frame_mse = float(diff.square().mean().item())
        frame_result = {"frame_index": frame_index, **_psnr_record(frame_mse)}
        result["per_frame"].append(frame_result)
        frame_psnr.append(frame_result["psnr_db"])
        total_abs += float(diff.abs().sum().item())
        total_sq += float(diff.square().sum().item())
        total_count += diff.numel()
        max_abs = max(max_abs, float(diff.abs().max().item()))

    global_mse = total_sq / total_count
    global_psnr = _psnr_record(global_mse)
    result["global_mse"] = global_mse
    result["global_psnr_db"] = global_psnr["psnr_db"]
    result["global_psnr_infinite"] = global_psnr["psnr_infinite"]
    finite_frame_psnr = [value for value in frame_psnr if value is not None]
    # A zero-error frame makes the arithmetic mean +infinity. JSON uses an
    # explicit flag rather than emitting a non-standard Infinity literal.
    if len(finite_frame_psnr) == len(frame_psnr):
        result["mean_frame_psnr_db"] = statistics.fmean(finite_frame_psnr)
    else:
        result["mean_frame_psnr_infinite"] = True
    finite_for_worst = [
        value if value is not None else float("inf") for value in frame_psnr
    ]
    worst_index = int(min(range(len(finite_for_worst)), key=finite_for_worst.__getitem__))
    result["worst_frame_index"] = worst_index
    result["worst_frame_psnr_db"] = frame_psnr[worst_index]
    result["worst_frame_psnr_infinite"] = frame_psnr[worst_index] is None
    result["below_30db_count"] = sum(
        int(frame["below_30db"]) for frame in result["per_frame"]
    )
    result["below_30db_ratio"] = result["below_30db_count"] / len(frame_psnr)
    result["mean_frame_psnr_pass_30db"] = bool(
        result["mean_frame_psnr_infinite"]
        or result["mean_frame_psnr_db"] >= PSNR_THRESHOLD_DB
    )
    result["global_psnr_pass_30db"] = bool(
        result["global_psnr_infinite"]
        or result["global_psnr_db"] >= PSNR_THRESHOLD_DB
    )
    result["all_frames_pass_30db"] = result["below_30db_count"] == 0
    result["mae"] = total_abs / total_count
    result["max_abs"] = max_abs
    result["finite"] = True
    return result


def _tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Small latent/control error helper that never emits NaN or Infinity."""
    if tuple(reference.shape) != tuple(candidate.shape):
        raise RuntimeError(
            f"tensor shape mismatch: {tuple(reference.shape)} vs "
            f"{tuple(candidate.shape)}"
        )
    if not bool(torch.isfinite(reference).all()) or not bool(
        torch.isfinite(candidate).all()
    ):
        raise RuntimeError("tensor error input is non-finite")
    diff = reference.float() - candidate.float()
    return {
        "shape_match": True,
        "finite": True,
        "mae": float(diff.abs().mean().item()),
        "rmse": float(diff.square().mean().sqrt().item()),
        "max_abs": float(diff.abs().max().item()),
    }


def _decoder_for_hooks(runtime: H3VAEPyOptRuntime):
    decoder = runtime.core.decoder
    return getattr(decoder, "_orig_mod", decoder)


def _validate_int8_inputs(
    runtime: H3VAEPyOptRuntime, latent: torch.Tensor
) -> dict:
    """Check every INT8 w1/w2 pre-quantization input before timed calls."""
    decoder = _decoder_for_hooks(runtime)
    blocks = getattr(decoder, "transformer_blocks", None)
    if blocks is None:
        raise RuntimeError("INT8 validation could not find decoder transformer_blocks")
    seen: dict[str, dict] = {}
    counts: dict[str, int] = {}
    hooks = []

    def make_hook(layer: int, name: str):
        def hook(_module, inputs):
            x = inputs[0]
            key = f"layer{layer}.{name}"
            counts[key] = counts.get(key, 0) + 1
            finite = bool(torch.isfinite(x).all())
            if key not in seen:
                seen[key] = {
                    "shape": list(x.shape),
                    "stride": list(x.stride()),
                    "dtype": str(x.dtype),
                    "device": str(x.device),
                    "finite": finite,
                }
            if not finite:
                raise RuntimeError(
                    f"INT8 input is non-finite before quantization: {key}"
                )

        return hook

    for layer, block in enumerate(blocks):
        hooks.append(block.ff.w1.register_forward_pre_hook(make_hook(layer, "w1")))
        hooks.append(block.ff.w2.register_forward_pre_hook(make_hook(layer, "w2")))
    previous_decoder = runtime.core.decoder
    runtime.core.decoder = decoder
    try:
        with torch.inference_mode():
            output = runtime.decode(latent)
            torch.cuda.synchronize()
    finally:
        for hook in hooks:
            hook.remove()
        runtime.core.decoder = previous_decoder
    expected = len(blocks) * 2
    if len(seen) != expected or set(seen) != set(counts):
        raise RuntimeError(
            f"INT8 validation observed {len(seen)} of {expected} FFN linears"
        )
    if any(count <= 0 for count in counts.values()):
        raise RuntimeError("INT8 validation did not observe every FFN call")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("INT8 validation decoder output is non-finite")
    result = {
        "block_count": len(blocks),
        "linear_hook_count": len(seen),
        "total_linear_calls": sum(counts.values()),
        "all_inputs_finite": True,
        "output_finite": True,
        "output_shape": list(output.shape),
        "path": "runtime.decode with uncompiled decoder body hooks",
        "per_linear": {
            key: {**seen[key], "call_count": counts[key]} for key in sorted(seen)
        },
    }
    del output
    return result


def _make_runtime(args, *, int8_decode: bool) -> H3VAEPyOptRuntime:
    return H3VAEPyOptRuntime(
        model_code_dir=str(args.model_code_dir),
        weights_path=str(args.weights),
        device="cuda",
        dtype=torch.float16,
        decoder_tile_size=int(args.decoder_tile),
        tile_batch=int(args.tile_batch),
        compile_decoder=True,
        compile_encoder=not bool(args.no_compile_encoder),
        encoder_staged_batch=int(args.encoder_staged_batch),
        log_calls=False,
        int8_decode=int8_decode,
    ).eval()


def _encode_once(
    runtime: H3VAEPyOptRuntime, encoder_input: torch.Tensor
) -> tuple[torch.Tensor, float]:
    input_device = encoder_input.to(device=runtime._device())
    start = time.perf_counter()
    with torch.inference_mode():
        encoded = runtime.encode(input_device)
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if encoded.dtype != torch.float32:
        raise RuntimeError(f"runtime.encode returned {encoded.dtype}, expected FP32")
    if not bool(torch.isfinite(encoded).all()):
        raise RuntimeError("runtime.encode returned a non-finite latent")
    latent_cpu = encoded.detach().float().cpu().contiguous()
    del input_device, encoded
    return latent_cpu, elapsed_ms


def _verify_encoder(
    runtime: H3VAEPyOptRuntime,
    encoder_input: torch.Tensor,
    reference_latent: torch.Tensor,
) -> dict:
    candidate, elapsed_ms = _encode_once(runtime, encoder_input)
    result = _tensor_error(reference_latent, candidate)
    result.update(
        {
            "extra_encode_wall_ms": elapsed_ms,
            "extra_encode_not_public_latent": True,
            "reference_dtype": str(reference_latent.dtype),
            "candidate_dtype": str(candidate.dtype),
        }
    )
    del candidate
    return result


def _timed_decode_pair(
    runtimes: dict[str, H3VAEPyOptRuntime],
    latent_cpu: torch.Tensor,
    *,
    warmup: int,
    runs: int,
) -> tuple[dict, dict[str, torch.Tensor]]:
    latent = latent_cpu.to(device=runtimes["default"]._device(), dtype=torch.float16)
    timings = {variant: [] for variant in VARIANTS}
    warmup_order = []
    for index in range(warmup):
        offset = index % len(VARIANTS)
        order = VARIANTS[offset:] + VARIANTS[:offset]
        warmup_order.append(list(order))
        for variant in order:
            with torch.inference_mode():
                output = runtimes[variant].decode(latent)
                torch.cuda.synchronize()
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError(f"{variant} warmup output is non-finite")
            del output

    outputs: dict[str, torch.Tensor] = {}
    timed_order = []
    for run_index in range(runs):
        offset = run_index % len(VARIANTS)
        order = VARIANTS[offset:] + VARIANTS[:offset]
        timed_order.append(list(order))
        for position, variant in enumerate(order):
            runtime = runtimes[variant]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(runtime._device())
            start_wall = time.perf_counter()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.inference_mode():
                output = runtime.decode(latent)
            end_event.record()
            end_event.synchronize()
            wall_ms = (time.perf_counter() - start_wall) * 1000.0
            event_ms = start_event.elapsed_time(end_event)
            peak = torch.cuda.max_memory_allocated(runtime._device())
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError(f"{variant} timed output is non-finite")
            output_cpu = output.detach().float().cpu().contiguous()
            del output
            outputs[variant] = output_cpu
            timings[variant].append(
                {
                    "run_index": run_index,
                    "order_position": position,
                    "cuda_event_ms": event_ms,
                    "wall_sync_ms": wall_ms,
                    "peak_memory_allocated_bytes": int(peak),
                }
            )
    control_start = time.perf_counter()
    with torch.inference_mode():
        control_output = runtimes["default"].decode(latent)
        torch.cuda.synchronize()
    control_ms = (time.perf_counter() - control_start) * 1000.0
    control_cpu = control_output.detach().float().cpu().contiguous()
    control = _tensor_error(outputs["default"], control_cpu)
    control["wall_sync_ms"] = control_ms
    del control_output, control_cpu, latent

    timing_summary = {}
    for variant, samples in timings.items():
        timing_summary[variant] = {
            "runs": samples,
            "mean_cuda_event_ms": statistics.fmean(
                item["cuda_event_ms"] for item in samples
            ),
            "mean_wall_sync_ms": statistics.fmean(
                item["wall_sync_ms"] for item in samples
            ),
            "max_peak_memory_allocated_bytes": max(
                item["peak_memory_allocated_bytes"] for item in samples
            ),
        }
    return {
        "warmup_count_per_variant": warmup,
        "warmup_order": warmup_order,
        "timed_runs_per_variant": runs,
        "alternating_order": timed_order,
        "timing": timing_summary,
        "control_repeat_default": control,
    }, outputs


def _process_video(
    path: Path,
    args,
    runtimes: dict[str, H3VAEPyOptRuntime],
    *,
    verify_encoder: bool,
) -> dict:
    source_sha256 = _sha256(path)
    reference, encoder_input, source_meta = _read_rgb_video(path)
    shape_meta = _validate_shape(reference)
    latent_cpu, encode_ms = _encode_once(runtimes["default"], encoder_input)
    if latent_cpu.ndim != 5 or tuple(latent_cpu.shape[:2]) != (1, 24):
        raise RuntimeError(f"unexpected latent shape: {tuple(latent_cpu.shape)}")
    if not bool(torch.isfinite(latent_cpu).all()):
        raise RuntimeError("public latent is non-finite")
    encoder_verification = None
    if verify_encoder:
        encoder_verification = _verify_encoder(
            runtimes["INT8both"], encoder_input, latent_cpu
        )
    validation_latent = latent_cpu.to(
        device=runtimes["INT8both"]._device(), dtype=torch.float16
    )
    int8_validation = _validate_int8_inputs(
        runtimes["INT8both"], validation_latent
    )
    del validation_latent
    timing, outputs = _timed_decode_pair(
        runtimes,
        latent_cpu,
        warmup=int(args.warmup),
        runs=int(args.runs),
    )
    quality = {
        "default_vs_source": _quality_metrics(reference, outputs["default"]),
        "INT8both_vs_source": _quality_metrics(reference, outputs["INT8both"]),
        "INT8both_vs_default": _quality_metrics(
            outputs["default"], outputs["INT8both"]
        ),
    }
    report = {
        "status": "ok",
        "source": {
            "path": str(path),
            "name": path.name,
            "sha256": source_sha256,
            **source_meta,
            **shape_meta,
        },
        "config": {
            "model_code_dir": str(args.model_code_dir),
            "weights": str(args.weights),
            "weights_sha256": args.weights_sha256,
            "decoder_tile_size": int(args.decoder_tile),
            "tile_batch": int(args.tile_batch),
            "compile_decoder": True,
            "compile_encoder": not bool(args.no_compile_encoder),
            "decoder_compile_mode": DECODER_COMPILE_MODE,
            "seed": int(args.seed),
            "latent_cast": "FP32 encoder output -> FP16 for both decoders",
            "quality_reference": "exact PyAV RGB uint8 / 255 FP32",
            "quality_threshold_db": PSNR_THRESHOLD_DB,
        },
        "encode": {
            "once_for_public_latent": True,
            "wall_ms_outside_decode_timing": encode_ms,
            "input_shape": list(encoder_input.shape),
            "input_dtype": str(encoder_input.dtype),
            "input_range": "[-1,1]",
            "output_shape": list(latent_cpu.shape),
            "output_dtype": str(latent_cpu.dtype),
            "output_finite": True,
            "encoder_verification": encoder_verification,
        },
        "int8_input_validation": int8_validation,
        "timing": timing,
        "quality": quality,
        "notes": [
            "Each sample is encoded once by the default runtime; both decoder paths receive the same FP16 latent.",
            "Only runtime.decode is timed. Video I/O, encode, runtime construction, compile, validation, output copies, and quality metrics are excluded.",
            "The encoder is not quantized. --verify-encoder performs one additional INT8-runtime encode outside the public-latent path.",
            "No resizing, cropping, frame dropping, or temporal padding is performed.",
        ],
    }
    del reference, encoder_input, latent_cpu, outputs
    return report


def _video_paths(args) -> list[Path]:
    if args.video is not None:
        return [args.video]
    paths = sorted(
        path
        for path in args.videos_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    )
    if not paths:
        raise ValueError(f"no MP4 files found in {args.videos_dir}")
    return paths


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path)
    group.add_argument("--videos-dir", type=Path)
    parser.add_argument("--model-code-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--decoder-tile", type=int, default=256)
    parser.add_argument("--tile-batch", type=int, default=2)
    parser.add_argument("--encoder-staged-batch", type=int, default=4)
    parser.add_argument("--no-compile-encoder", action="store_true")
    parser.add_argument(
        "--verify-encoder",
        action="store_true",
        help="encode once more with INT8 runtime for one sample and compare latents",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing per-video JSON and summary.json outputs",
    )
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.runs < 1:
        parser.error("--warmup must be >=0 and --runs must be >=1")
    if args.decoder_tile <= 0 or args.decoder_tile % 16:
        parser.error("--decoder-tile must be a positive multiple of 16")
    if args.tile_batch < 0:
        parser.error("--tile-batch must be non-negative")
    if args.videos_dir is not None and not args.videos_dir.is_dir():
        parser.error(f"--videos-dir does not exist: {args.videos_dir}")
    if args.video is not None and not args.video.is_file():
        parser.error(f"--video does not exist: {args.video}")
    if not args.model_code_dir.is_dir():
        parser.error(f"--model-code-dir does not exist: {args.model_code_dir}")
    if not args.weights.is_file():
        parser.error(f"--weights does not exist: {args.weights}")
    return args, parser


def main(argv=None) -> None:
    args, parser = _parse_args(argv)
    torch.set_num_threads(4)
    try:
        paths = _video_paths(args)
    except ValueError as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_paths = [args.output_dir / f"{path.stem}.json" for path in paths]
    summary_path = args.output_dir / "summary.json"
    if not args.overwrite:
        collisions = [path for path in [*report_paths, summary_path] if path.exists()]
        if collisions:
            parser.error(
                "refusing to overwrite existing outputs; use a new output-dir or "
                f"--overwrite: {collisions}"
            )
    if not torch.cuda.is_available():
        parser.error("INT8 VAE validation requires an available CUDA GPU")
    if torch.version.hip is not None:
        parser.error("INT8 VAE validation requires a CUDA PyTorch build, not ROCm")

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    args.weights_sha256 = _sha256(args.weights)
    environment = _environment()
    runtimes = {
        "default": _make_runtime(args, int8_decode=False),
        "INT8both": _make_runtime(args, int8_decode=True),
    }
    reports = []
    verified = False
    for path, report_path in zip(paths, report_paths):
        report = _process_video(
            path,
            args,
            runtimes,
            verify_encoder=bool(args.verify_encoder and not verified),
        )
        verified = verified or bool(args.verify_encoder)
        report["environment"] = {
            **environment,
            "gpu_after_sample": _gpu_snapshot(),
        }
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        reports.append(report)
        print(
            json.dumps(
                {
                    "sample": path.name,
                    "default_ms": report["timing"]["timing"]["default"][
                        "mean_cuda_event_ms"
                    ],
                    "int8_ms": report["timing"]["timing"]["INT8both"][
                        "mean_cuda_event_ms"
                    ],
                    "int8_vs_source_mean_psnr_db": report["quality"][
                        "INT8both_vs_source"
                    ]["mean_frame_psnr_db"],
                },
                indent=2,
            )
        )
    summary = {
        "suite": "H3 VAE production-path default versus INT8 decode",
        "environment": {**environment, "gpu_after_suite": _gpu_snapshot()},
        "config": {
            "model_code_dir": str(args.model_code_dir),
            "weights": str(args.weights),
            "weights_sha256": args.weights_sha256,
            "warmup_per_variant": int(args.warmup),
            "timed_runs_per_variant": int(args.runs),
            "decoder_tile_size": int(args.decoder_tile),
            "tile_batch": int(args.tile_batch),
            "verify_encoder_first_sample_only": bool(args.verify_encoder),
            "sample_count": len(reports),
        },
        "samples": [
            {
                "name": report["source"]["name"],
                "sha256": report["source"]["sha256"],
                "shape": report["source"]["reference_shape"],
                "quality": {
                    pair: {
                        key: value
                        for key, value in metrics.items()
                        if key not in {"per_frame"}
                    }
                    for pair, metrics in report["quality"].items()
                },
                "json": str(path),
            }
            for report, path in zip(reports, report_paths)
        ],
        "notes": [
            "This suite measures decode-only runtime paths; encode is a fixed-input preparation step.",
            "INT8 is decoder-only; encoder remains the production FP16 path.",
            "GPU snapshots include unrelated background work and are not idle-GPU claims.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path), "samples": len(reports)}, indent=2))


if __name__ == "__main__":
    main()
