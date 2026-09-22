#!/usr/bin/env python3
"""Four-way video validation for the experimental INT8 H3 VAE paths.

The benchmark exercises the production runtime through two independent
instances: the default FP16 runtime (``E0/D0``) and the opt-in runtime with
both INT8 paths enabled (``E1/D1``).  Each source is encoded by both
instances from the same prepared FP16 input, and each latent is decoded by
both runtimes.  The resulting four paths are therefore:

``default`` (E0/D0), ``d_only`` (E0/D1), ``e_only`` (E1/D0), and
``both`` (E1/D1).

Media conversion and quality accounting are deliberately shared with
``bench_int8_vae.py``.  PyAV and numpy stay lazy because this module is also
imported by CPU-only tests.  This is an opt-in GPU experiment; it does not
alter runtime defaults, model files, or ComfyUI nodes.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import time
from pathlib import Path

import torch

import bench_int8_vae as _base


H3VAEPyOptRuntime = _base.H3VAEPyOptRuntime
DECODER_COMPILE_MODE = _base.DECODER_COMPILE_MODE

PSNR_THRESHOLD_DB = _base.PSNR_THRESHOLD_DB
MSE_AT_30_DB = _base.MSE_AT_30_DB

ENCODER_VARIANTS = ("E0", "E1")
DECODE_PATHS = ("default", "d_only", "e_only", "both")

# Each path is represented by a runtime name and latent name.  Keeping this
# mapping explicit makes it difficult to accidentally label an E1/D0 result
# as decoder-only or compare the wrong latent in the quality report.
PATH_CONFIG = {
    "default": {"encode": "E0", "decode": "D0", "runtime": "default"},
    "d_only": {"encode": "E0", "decode": "D1", "runtime": "both"},
    "e_only": {"encode": "E1", "decode": "D0", "runtime": "default"},
    "both": {"encode": "E1", "decode": "D1", "runtime": "both"},
}


def _synchronize(device: torch.device) -> None:
    """Synchronize a CUDA device while keeping helper tests CPU-safe."""

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_is_finite(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"{name} did not return a torch.Tensor")
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{name} is non-finite")


def _validate_latent_shape(
    latent: torch.Tensor,
    reference: torch.Tensor,
    *,
    name: str,
) -> None:
    """Reject malformed encoder outputs before they reach either decoder."""

    if reference.ndim != 5 or latent.ndim != 5:
        raise RuntimeError(
            f"{name} expected rank-5 latent/reference, got "
            f"{latent.ndim}/{reference.ndim}"
        )
    _, channels, frames, height, width = reference.shape
    expected_frames = 1 if frames == 1 else (frames - 5) // 17 * 5 + 2
    expected = (1, 24, expected_frames, height // 16, width // 16)
    if tuple(latent.shape) != expected:
        raise RuntimeError(
            f"{name} shape mismatch: expected {expected}, got {tuple(latent.shape)}"
        )
    del channels


def _latent_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Return latent error metrics plus an explicit relative RMSE."""

    result = _base._tensor_error(reference, candidate)
    reference_rms = float(reference.float().square().mean().sqrt().item())
    rmse = float(result["rmse"])
    if reference_rms == 0.0:
        relative_rmse = 0.0 if rmse == 0.0 else None
    else:
        relative_rmse = rmse / reference_rms
    result.update(
        {
            "reference_rms": reference_rms,
            "relative_rmse": relative_rmse,
        }
    )
    return result


def _quality_headline(metrics: dict) -> dict:
    """Keep the requested headline metrics easy to find in each JSON."""

    return {
        "mean_frame_psnr_db": metrics["mean_frame_psnr_db"],
        "mean_frame_psnr_infinite": metrics["mean_frame_psnr_infinite"],
        "global_psnr_db": metrics["global_psnr_db"],
        "global_psnr_infinite": metrics["global_psnr_infinite"],
        "min_frame_psnr_db": metrics["worst_frame_psnr_db"],
        "min_frame_psnr_infinite": metrics["worst_frame_psnr_infinite"],
        "below_30db_count": metrics["below_30db_count"],
        "frame_count": len(metrics["per_frame"]),
    }


def _summarize_timings(samples: dict[str, list[dict]]) -> dict:
    result = {}
    for name, values in samples.items():
        event_values = [
            float(item["cuda_event_ms"])
            for item in values
            if item.get("cuda_event_ms") is not None
        ]
        wall_values = [float(item["wall_sync_ms"]) for item in values]
        result[name] = {
            "runs": values,
            "mean_cuda_event_ms": (
                statistics.fmean(event_values) if event_values else None
            ),
            "mean_wall_sync_ms": (
                statistics.fmean(wall_values) if wall_values else None
            ),
            "min_cuda_event_ms": min(event_values) if event_values else None,
            "min_wall_sync_ms": min(wall_values) if wall_values else None,
            "max_peak_memory_allocated_bytes": (
                max(int(item["peak_memory_allocated_bytes"]) for item in values)
                if values and any(
                    item.get("peak_memory_allocated_bytes") is not None
                    for item in values
                )
                else None
            ),
        }
    return result


def _timed_runtime_call(
    call,
    *,
    device: torch.device,
    expected_shape: tuple[int, ...],
    label: str,
    run_index: int,
    order_position: int,
    timed: bool,
) -> dict:
    """Execute one already-prepared runtime call and validate its output.

    The event and wall clocks stop as soon as the runtime returns and its
    CUDA work has completed.  Finite/shape checks and any CPU copy happen
    afterward, so dynamic per-call quantization is included while reporting
    and I/O work are excluded.
    """

    device = torch.device(device)
    _synchronize(device)
    peak = None
    if device.type == "cuda" and timed:
        torch.cuda.reset_peak_memory_stats(device)
    start_wall = time.perf_counter()
    start_event = None
    end_event = None
    if device.type == "cuda" and timed:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    with torch.inference_mode():
        output = call()
    if end_event is not None:
        end_event.record()
        end_event.synchronize()
    else:
        _synchronize(device)
    if device.type == "cuda" and timed:
        # Read the allocation watermark before finite/shape validation or any
        # later reporting allocations can affect the measurement.
        peak = int(torch.cuda.max_memory_allocated(device))
    wall_ms = (time.perf_counter() - start_wall) * 1000.0
    event_ms = (
        None if start_event is None else float(start_event.elapsed_time(end_event))
    )

    if not isinstance(output, torch.Tensor):
        raise RuntimeError(f"{label} returned {type(output).__name__}, not a tensor")
    if tuple(output.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"{label} output shape mismatch: expected {expected_shape}, "
            f"got {tuple(output.shape)}"
        )
    _tensor_is_finite(output, f"{label} output")
    del output
    return {
        "run_index": int(run_index),
        "order_position": int(order_position),
        "cuda_event_ms": event_ms,
        "wall_sync_ms": wall_ms,
        "peak_memory_allocated_bytes": peak,
    }


def _rotate_order(names: tuple[str, ...], index: int) -> list[str]:
    offset = int(index) % len(names)
    return list(names[offset:] + names[:offset])


def _prepare_runtime_input(
    prepared_input: torch.Tensor,
    runtime: H3VAEPyOptRuntime,
) -> torch.Tensor:
    """Transfer a single exact FP16 preparation outside encode timing."""

    if prepared_input.dtype != torch.float16:
        raise RuntimeError(
            f"prepared encoder input must be FP16, got {prepared_input.dtype}"
        )
    _tensor_is_finite(prepared_input, "prepared encoder input")
    return prepared_input.to(device=runtime._device())


def _encode_quality_once(
    runtime: H3VAEPyOptRuntime,
    prepared_input: torch.Tensor,
    reference: torch.Tensor,
    *,
    name: str,
) -> tuple[torch.Tensor, float]:
    """Encode once for quality/control and return a CPU FP32 latent."""

    input_device = _prepare_runtime_input(prepared_input, runtime)
    start = time.perf_counter()
    with torch.inference_mode():
        latent = runtime.encode(input_device)
        _synchronize(runtime._device())
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if not isinstance(latent, torch.Tensor):
        raise RuntimeError(f"{name} encode did not return a tensor")
    if latent.dtype != torch.float32:
        raise RuntimeError(
            f"{name} encode returned {latent.dtype}; expected FP32 latent output"
        )
    _tensor_is_finite(latent, f"{name} latent")
    latent_cpu = latent.detach().float().cpu().contiguous()
    _validate_latent_shape(latent_cpu, reference, name=f"{name} latent")
    del input_device, latent
    return latent_cpu, elapsed_ms


def _int8_encoder_targets(runtime: H3VAEPyOptRuntime) -> list[tuple[str, torch.nn.Module]]:
    """Find the eight production INT8 encoder convolutions by stable names.

    The runtime may wrap the prefix in ``torch.compile``. The integration
    contract gives the quantized modules terminal names ``int8_first`` and
    ``int8_second``; looking for those names avoids depending on source
    encoder private block class names.
    """

    targets: list[tuple[str, torch.nn.Module]] = []
    seen: set[int] = set()
    for root_name in ("prefix", "suffix"):
        root = getattr(runtime, root_name, None)
        if root is None:
            continue
        while hasattr(root, "_orig_mod"):
            root = root._orig_mod
        if not isinstance(root, torch.nn.Module):
            continue
        for qualname, module in root.named_modules():
            if not qualname:
                continue
            terminal = qualname.rsplit(".", 1)[-1]
            if terminal not in {"int8_first", "int8_second"}:
                continue
            if id(module) in seen:
                continue
            seen.add(id(module))
            targets.append((f"{root_name}.{qualname}", module))
    if len(targets) != 8:
        raise RuntimeError(
            "INT8 encoder validation found "
            f"{len(targets)} named convolutions; expected exactly 8 "
            "int8_first/int8_second modules"
        )
    return targets


def _validate_int8_encoder_inputs(
    runtime: H3VAEPyOptRuntime,
    prepared_input: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, float, dict]:
    """Run E1 once with pre-quantization finite hooks on all eight modules."""

    metadata = getattr(runtime, "int8_encoder_metadata", None)
    if not isinstance(metadata, dict) or int(metadata.get("count", -1)) != 8:
        raise RuntimeError(
            "INT8 encoder runtime metadata must report exactly 8 installed "
            f"convolutions, got {metadata!r}"
        )
    targets = _int8_encoder_targets(runtime)
    original_modules = {}
    for root_name in ("prefix", "suffix"):
        original = getattr(runtime, root_name, None)
        raw = original
        while hasattr(raw, "_orig_mod"):
            raw = raw._orig_mod
        if original is not None and raw is not original:
            original_modules[root_name] = original
            setattr(runtime, root_name, raw)
    seen: dict[str, dict] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs):
            if not inputs:
                raise RuntimeError(f"INT8 encoder {name} received no input")
            value = inputs[0]
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"INT8 encoder {name} input is not a tensor")
            finite = bool(torch.isfinite(value).all())
            counts[name] = counts.get(name, 0) + 1
            if name not in seen:
                seen[name] = {
                    "shape": list(value.shape),
                    "stride": list(value.stride()),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "finite": finite,
                }
            if not finite:
                raise RuntimeError(
                    f"INT8 encoder input is non-finite before quantization: {name}"
                )

        return hook

    try:
        for name, module in targets:
            handles.append(module.register_forward_pre_hook(make_hook(name)))
        latent_cpu, elapsed_ms = _encode_quality_once(
            runtime, prepared_input, reference, name="E1 hooked"
        )
    finally:
        for handle in handles:
            handle.remove()
        for root_name, original in original_modules.items():
            setattr(runtime, root_name, original)

    expected = {name for name, _module in targets}
    if set(seen) != expected or any(counts.get(name, 0) <= 0 for name in expected):
        raise RuntimeError(
            "INT8 encoder validation did not observe every quantized convolution: "
            f"observed={sorted(seen)} expected={sorted(expected)}"
        )
    if not all(item["finite"] for item in seen.values()):
        raise RuntimeError("INT8 encoder validation observed a non-finite input")
    validation = {
        "module_count": len(targets),
        "module_hook_count": len(seen),
        "total_module_calls": sum(counts.values()),
        "all_inputs_finite": True,
        "output_finite": True,
        "output_shape": list(latent_cpu.shape),
        "path": "runtime.encode with uncompiled INT8 encoder prefix hooks",
        "runtime_metadata": metadata,
        "per_module": {
            name: {**seen[name], "call_count": counts[name]}
            for name in sorted(seen)
        },
    }
    return latent_cpu, elapsed_ms, validation


def _validate_int8_decoder_inputs(
    runtime: H3VAEPyOptRuntime,
    latents: dict[str, torch.Tensor],
) -> dict:
    """Use the production decoder hook helper once for each E0/E1 latent."""

    result = {}
    for name in ("E0", "E1"):
        latent_device = latents[name].to(
            device=runtime._device(), dtype=torch.float16
        )
        result[name] = _base._validate_int8_inputs(runtime, latent_device)
        del latent_device
    return result


def _time_encodes(
    runtimes: dict[str, H3VAEPyOptRuntime],
    prepared_input: torch.Tensor,
    reference: torch.Tensor,
    *,
    warmup: int,
    runs: int,
) -> dict:
    """Time E0/E1 with alternating AB/BA order and raw samples retained."""

    device = torch.device(runtimes["default"]._device())
    device_inputs = {
        name: _prepare_runtime_input(prepared_input, runtime)
        for name, runtime in runtimes.items()
    }
    expected_shape = (
        1,
        24,
        1 if reference.shape[2] == 1 else (reference.shape[2] - 5) // 17 * 5 + 2,
        reference.shape[3] // 16,
        reference.shape[4] // 16,
    )
    warmup_order = []
    for index in range(int(warmup)):
        order = _rotate_order(ENCODER_VARIANTS, index)
        warmup_order.append(order)
        for name in order:
            runtime_name = "default" if name == "E0" else "both"
            call = lambda runtime=runtimes[runtime_name], input_tensor=device_inputs[runtime_name]: runtime.encode(input_tensor)
            _timed_runtime_call(
                call,
                device=device,
                expected_shape=expected_shape,
                label=f"{name} warmup",
                run_index=index,
                order_position=order.index(name),
                timed=False,
            )

    samples = {name: [] for name in ENCODER_VARIANTS}
    timed_order = []
    for run_index in range(int(runs)):
        order = _rotate_order(ENCODER_VARIANTS, run_index)
        timed_order.append(order)
        for order_position, name in enumerate(order):
            runtime_name = "default" if name == "E0" else "both"
            call = lambda runtime=runtimes[runtime_name], input_tensor=device_inputs[runtime_name]: runtime.encode(input_tensor)
            samples[name].append(
                _timed_runtime_call(
                    call,
                    device=device,
                    expected_shape=expected_shape,
                    label=f"{name} timed",
                    run_index=run_index,
                    order_position=order_position,
                    timed=True,
                )
            )
    for value in device_inputs.values():
        del value
    return {
        "warmup_count_per_variant": int(warmup),
        "warmup_order": warmup_order,
        "timed_runs_per_variant": int(runs),
        "alternating_order": timed_order,
        "timing": _summarize_timings(samples),
        "input_dtype": str(prepared_input.dtype),
        "input_shape": list(prepared_input.shape),
        "dynamic_quantization_in_timed_call": True,
        "static_quantization_in_timed_call": False,
    }


def _decode_call(
    runtime: H3VAEPyOptRuntime,
    latent_cpu: torch.Tensor,
    *,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Prepare one decoder latent and return (device latent, expected output)."""

    _tensor_is_finite(latent_cpu, "decoder latent")
    latent_device = latent_cpu.to(device=runtime._device(), dtype=torch.float16)
    expected_shape = tuple(reference.shape)
    return latent_device, expected_shape


def _quality_decode_paths(
    runtimes: dict[str, H3VAEPyOptRuntime],
    latents: dict[str, torch.Tensor],
    reference: torch.Tensor,
) -> tuple[dict, dict]:
    """Decode all paths once for quality, retaining only the E0/D0 CPU output."""

    quality_vs_source = {}
    quality_vs_default = {}
    default_output = None
    output_shapes = {}
    latent_devices = {}
    for name, path_config in PATH_CONFIG.items():
        runtime = runtimes[path_config["runtime"]]
        latent_name = path_config["encode"]
        if latent_name not in latent_devices:
            latent_devices[latent_name], _ = _decode_call(
                runtime, latents[latent_name], reference=reference
            )
        latent_device = latent_devices[latent_name]
        with torch.inference_mode():
            output = runtime.decode(latent_device)
            _synchronize(runtime._device())
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"{name} decoder did not return a tensor")
        if tuple(output.shape) != tuple(reference.shape):
            raise RuntimeError(
                f"{name} output shape mismatch: expected {tuple(reference.shape)}, "
                f"got {tuple(output.shape)}"
            )
        _tensor_is_finite(output, f"{name} decoder output")
        output_shapes[name] = list(output.shape)
        output_cpu = output.detach().float().cpu().contiguous()
        del output
        quality_vs_source[name] = _base._quality_metrics(reference, output_cpu)
        if name == "default":
            default_output = output_cpu
        else:
            if default_output is None:
                raise RuntimeError("default decoder output must be measured first")
            quality_vs_default[name] = _base._quality_metrics(
                default_output, output_cpu
            )
            del output_cpu

    # Keep the baseline only while its metrics are being serialized by the
    # caller; all alternate outputs have already been released.
    if default_output is None:
        raise RuntimeError("default decoder output was not produced")
    del default_output
    for latent in latent_devices.values():
        del latent
    return (
        {
            "vs_source": quality_vs_source,
            "vs_default": quality_vs_default,
            "headlines_vs_source": {
                name: _quality_headline(metrics)
                for name, metrics in quality_vs_source.items()
            },
            "headlines_vs_default": {
                name: _quality_headline(metrics)
                for name, metrics in quality_vs_default.items()
            },
        },
        output_shapes,
    )


def _time_decodes(
    runtimes: dict[str, H3VAEPyOptRuntime],
    latents: dict[str, torch.Tensor],
    reference: torch.Tensor,
    *,
    warmup: int,
    runs: int,
) -> dict:
    """Time four decode paths with rotated order and raw samples retained."""

    device = torch.device(runtimes["default"]._device())
    latent_devices = {}
    for name, path_config in PATH_CONFIG.items():
        latent_name = path_config["encode"]
        if latent_name not in latent_devices:
            latent_devices[latent_name], _ = _decode_call(
                runtimes[path_config["runtime"]],
                latents[latent_name],
                reference=reference,
            )
    expected_shape = tuple(reference.shape)

    warmup_order = []
    for index in range(int(warmup)):
        order = _rotate_order(DECODE_PATHS, index)
        warmup_order.append(order)
        for order_position, name in enumerate(order):
            path_config = PATH_CONFIG[name]
            runtime = runtimes[path_config["runtime"]]
            latent = latent_devices[path_config["encode"]]
            call = lambda runtime=runtime, latent=latent: runtime.decode(latent)
            _timed_runtime_call(
                call,
                device=device,
                expected_shape=expected_shape,
                label=f"{name} warmup",
                run_index=index,
                order_position=order_position,
                timed=False,
            )

    samples = {name: [] for name in DECODE_PATHS}
    timed_order = []
    for run_index in range(int(runs)):
        order = _rotate_order(DECODE_PATHS, run_index)
        timed_order.append(order)
        for order_position, name in enumerate(order):
            path_config = PATH_CONFIG[name]
            runtime = runtimes[path_config["runtime"]]
            latent = latent_devices[path_config["encode"]]
            call = lambda runtime=runtime, latent=latent: runtime.decode(latent)
            samples[name].append(
                _timed_runtime_call(
                    call,
                    device=device,
                    expected_shape=expected_shape,
                    label=f"{name} timed",
                    run_index=run_index,
                    order_position=order_position,
                    timed=True,
                )
            )
    for latent in latent_devices.values():
        del latent
    return {
        "warmup_count_per_path": int(warmup),
        "warmup_order": warmup_order,
        "timed_runs_per_path": int(runs),
        "rotating_order": timed_order,
        "timing": _summarize_timings(samples),
        "dynamic_quantization_in_timed_call": True,
        "static_quantization_in_timed_call": False,
    }


def _deterministic_controls(
    runtimes: dict[str, H3VAEPyOptRuntime],
    prepared_input: torch.Tensor,
    reference: torch.Tensor,
    latent_e0: torch.Tensor,
) -> dict:
    """Record repeat errors for the default encoder and default decoder."""

    repeat_e0, repeat_encode_ms = _encode_quality_once(
        runtimes["default"],
        prepared_input,
        reference,
        name="E0 repeat",
    )
    encode_repeat = _latent_error(latent_e0, repeat_e0)

    latent_device = latent_e0.to(
        device=runtimes["default"]._device(), dtype=torch.float16
    )
    with torch.inference_mode():
        first = runtimes["default"].decode(latent_device)
        _synchronize(runtimes["default"]._device())
    if tuple(first.shape) != tuple(reference.shape):
        raise RuntimeError("default deterministic first decode shape mismatch")
    _tensor_is_finite(first, "default deterministic first output")
    first_cpu = first.detach().float().cpu().contiguous()
    del first
    with torch.inference_mode():
        second = runtimes["default"].decode(latent_device)
        _synchronize(runtimes["default"]._device())
    if tuple(second.shape) != tuple(reference.shape):
        raise RuntimeError("default deterministic second decode shape mismatch")
    _tensor_is_finite(second, "default deterministic second output")
    second_cpu = second.detach().float().cpu().contiguous()
    decode_repeat = _base._tensor_error(first_cpu, second_cpu)
    del latent_device, second, first_cpu, second_cpu
    return {
        "encode_default_repeat": encode_repeat,
        "encode_default_repeat_wall_ms": repeat_encode_ms,
        "decode_default_repeat": decode_repeat,
    }


def _make_runtime(args, *, int8_encode: bool, int8_decode: bool):
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
        encoder_tile_size=int(args.encoder_tile),
        log_calls=False,
        int8_encode=bool(int8_encode),
        int8_decode=bool(int8_decode),
    ).eval()


def _process_video(
    path: Path,
    args,
    runtimes: dict[str, H3VAEPyOptRuntime],
    *,
    quality_only: bool,
) -> dict:
    source_sha256 = _base._sha256(path)
    reference, prepared_input, source_meta = _base._read_rgb_video(path)
    shape_meta = _base._validate_shape(reference)
    _tensor_is_finite(reference, "source reference")
    _tensor_is_finite(prepared_input, "prepared encoder input")

    latent_e0, encode_e0_ms = _encode_quality_once(
        runtimes["default"], prepared_input, reference, name="E0"
    )
    hooked_latent_e1, hooked_encode_ms, encoder_validation = _validate_int8_encoder_inputs(
        runtimes["both"], prepared_input, reference
    )
    # The hook pass temporarily unwraps the compiled prefix/suffix so hooks
    # can observe the actual INT8 modules.  Its latent is diagnostic only:
    # produce the public E1 latent with the restored compiled wrappers, under
    # the same policy used by the default E0 quality/timing path.
    del hooked_latent_e1
    latent_e1, encode_e1_ms = _encode_quality_once(
        runtimes["both"], prepared_input, reference, name="E1"
    )
    encoder_validation["hook_encode_wall_ms"] = hooked_encode_ms
    encoder_validation["hooked_latent_discarded"] = True
    encoder_validation["public_latent_from_restored_compiled_runtime"] = True
    latents = {"E0": latent_e0, "E1": latent_e1}
    latent_quality = _latent_error(latent_e0, latent_e1)
    decoder_validation = _validate_int8_decoder_inputs(runtimes["both"], latents)
    if quality_only:
        # Quality-only suite runs still warm every real path so compile and
        # cuDNN search are exercised before the quality calls.  ``runs=0``
        # means no steady-state samples are collected or reported.
        _time_encodes(
            runtimes,
            prepared_input,
            reference,
            warmup=int(args.warmup),
            runs=0,
        )
        _time_decodes(
            runtimes,
            latents,
            reference,
            warmup=int(args.warmup),
            runs=0,
        )
    controls = _deterministic_controls(
        runtimes, prepared_input, reference, latent_e0
    )
    quality, output_shapes = _quality_decode_paths(runtimes, latents, reference)
    timing = None
    if not quality_only:
        encode_timing = _time_encodes(
            runtimes,
            prepared_input,
            reference,
            warmup=int(args.warmup),
            runs=int(args.runs),
        )
        decode_timing = _time_decodes(
            runtimes,
            latents,
            reference,
            warmup=int(args.warmup),
            runs=int(args.runs),
        )
        timing = {"encode": encode_timing, "decode": decode_timing}
    effective_encoder_tile = int(
        args.encoder_tile
        or (672 if max(int(reference.shape[3]), int(reference.shape[4])) <= 672 else 256)
    )

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
            "encoder_tile_size": int(args.encoder_tile),
            "encoder_tile_size_effective": effective_encoder_tile,
            "encoder_staged_batch": int(args.encoder_staged_batch),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_benchmark_limit": int(torch.backends.cudnn.benchmark_limit),
            "compile_decoder": True,
            "compile_encoder": not bool(args.no_compile_encoder),
            "decoder_compile_mode": DECODER_COMPILE_MODE,
            "seed": int(args.seed),
            "quality_only": bool(quality_only),
            "quality_only_warmup_paths": bool(quality_only),
            "quality_reference": "exact PyAV RGB uint8 / 255 FP32",
            "quality_threshold_db": PSNR_THRESHOLD_DB,
        },
        "runtime_variants": {
            "default": {"int8_encode": False, "int8_decode": False},
            "both": {
                "int8_encode": True,
                "int8_decode": True,
                "int8_encoder_metadata": getattr(
                    runtimes["both"], "int8_encoder_metadata", None
                ),
            },
        },
        "encode": {
            "input_shape": list(prepared_input.shape),
            "input_dtype": str(prepared_input.dtype),
            "input_range": "[-1,1]",
            "same_exact_fp16_input": True,
            "E0": {
                "output_shape": list(latent_e0.shape),
                "output_dtype": str(latent_e0.dtype),
                "output_finite": True,
                "quality_call_wall_ms": encode_e0_ms,
            },
            "E1": {
                "output_shape": list(latent_e1.shape),
                "output_dtype": str(latent_e1.dtype),
                "output_finite": True,
                "quality_call_wall_ms": encode_e1_ms,
            },
            "E1_vs_E0": latent_quality,
        },
        "decode_paths": {
            name: {
                **PATH_CONFIG[name],
                "output_shape": output_shapes[name],
                "output_finite": True,
            }
            for name in DECODE_PATHS
        },
        "int8_input_validation": {
            "encoder": encoder_validation,
            "decoder": decoder_validation,
        },
        "quality": quality,
        "deterministic_controls": controls,
        "timing": timing,
        "notes": [
            "E0 and E1 receive the same exact prepared FP16 input; E0/E1 latents are compared in FP32.",
            "Each decode path receives its corresponding FP16 latent: default=E0/D0, d_only=E0/D1, e_only=E1/D0, both=E1/D1.",
            "Steady-state timing excludes media I/O, quality metrics, runtime construction, compilation, static quantization, and output CPU copies.",
            "Per-call dynamic quantization remains inside each timed runtime call.",
            "Finite and shape checks run before results are retained; non-finite or malformed outputs fail the sample.",
            "PSNR is a lossy reconstruction metric; these measurements do not claim losslessness.",
        ],
    }
    del reference, prepared_input, latent_e0, latent_e1, latents
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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


def _implementation_source_hashes() -> dict[str, str]:
    """Hash the exact runtime/helpers used by a GPU run for provenance."""

    root = Path(__file__).resolve().parent
    candidates = (
        ("h3vae_runtime.py", root / "h3vae_runtime.py"),
        ("opt/kitchen_int8.py", root / "opt" / "kitchen_int8.py"),
        ("opt/encoder_int8.py", root / "opt" / "encoder_int8.py"),
        (
            "opt/encoder_int8_integration.py",
            root / "opt" / "encoder_int8_integration.py",
        ),
        ("bench_int8_vae.py", root / "bench_int8_vae.py"),
        ("bench_int8_roundtrip.py", root / "bench_int8_roundtrip.py"),
    )
    return {
        name: _base._sha256(path)
        for name, path in candidates
        if path.is_file()
    }


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
    parser.add_argument(
        "--encoder-tile",
        type=int,
        default=256,
        help="encoder tile size; 256 matches the supplied 768x1344/1376 suite",
    )
    parser.add_argument("--encoder-staged-batch", type=int, default=4)
    parser.add_argument("--no-compile-encoder", action="store_true")
    parser.add_argument(
        "--no-cudnn-benchmark",
        action="store_true",
        help="disable cuDNN autotuning (enabled by default for the baseline)",
    )
    parser.add_argument(
        "--quality-only",
        action="store_true",
        help="run encode/decode quality validation but skip steady-state timing",
    )
    parser.add_argument(
        "--per-frame-csv",
        "--csv",
        dest="per_frame_csv",
        action="store_true",
        help="also write per-frame PSNR rows to per_frame_metrics.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing per-video JSON, CSV, and summary outputs",
    )
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.runs < 1:
        parser.error("--warmup must be >=0 and --runs must be >=1")
    if args.decoder_tile <= 0 or args.decoder_tile % 16:
        parser.error("--decoder-tile must be a positive multiple of 16")
    if args.encoder_tile < 0 or args.encoder_tile % 16:
        parser.error("--encoder-tile must be 0 or a positive multiple of 16")
    if args.tile_batch < 0 or args.encoder_staged_batch < 1:
        parser.error("--tile-batch must be >=0 and --encoder-staged-batch must be >=1")
    if args.videos_dir is not None and not args.videos_dir.is_dir():
        parser.error(f"--videos-dir does not exist: {args.videos_dir}")
    if args.video is not None and not args.video.is_file():
        parser.error(f"--video does not exist: {args.video}")
    if not args.model_code_dir.is_dir():
        parser.error(f"--model-code-dir does not exist: {args.model_code_dir}")
    if not args.weights.is_file():
        parser.error(f"--weights does not exist: {args.weights}")
    return args, parser


def _write_per_frame_csv(path: Path, reports: list[dict]) -> None:
    fields = [
        "video",
        "video_sha256",
        "path",
        "comparison",
        "frame_index",
        "psnr_db",
        "psnr_infinite",
        "mse",
        "below_30db",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            source = report["source"]
            quality = report["quality"]
            for comparison, candidates in (
                ("vs_source", quality["vs_source"]),
                ("vs_default", quality["vs_default"]),
            ):
                for name, metrics in candidates.items():
                    for frame in metrics["per_frame"]:
                        writer.writerow(
                            {
                                "video": source["name"],
                                "video_sha256": source["sha256"],
                                "path": name,
                                "comparison": comparison,
                                **{
                                    key: frame[key]
                                    for key in (
                                        "frame_index",
                                        "psnr_db",
                                        "psnr_infinite",
                                        "mse",
                                        "below_30db",
                                    )
                                },
                            }
                        )


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
    csv_path = args.output_dir / "per_frame_metrics.csv"
    outputs = [*report_paths, summary_path]
    if args.per_frame_csv:
        outputs.append(csv_path)
    if not args.overwrite:
        collisions = [path for path in outputs if path.exists()]
        if collisions:
            parser.error(
                "refusing to overwrite existing outputs; use a new output-dir or "
                f"--overwrite: {collisions}"
            )
    if not torch.cuda.is_available():
        parser.error("INT8 VAE round-trip validation requires an available CUDA GPU")
    if torch.version.hip is not None:
        parser.error("INT8 VAE round-trip validation requires a CUDA PyTorch build, not ROCm")

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    # Match the normal Loader policy so FP16 and INT8 paths are compared after
    # the same cuDNN autotuning/search budget. Warmups below absorb the search.
    torch.backends.cudnn.benchmark = not bool(args.no_cudnn_benchmark)
    torch.backends.cudnn.benchmark_limit = 5
    args.weights_sha256 = _base._sha256(args.weights)
    implementation_sources = _implementation_source_hashes()
    environment = _base._environment()
    # Static INT8 weights are prepared once at construction.  Only these two
    # runtime instances are needed for the four paths, which bounds GPU memory.
    runtimes = {
        "default": _make_runtime(args, int8_encode=False, int8_decode=False),
        "both": _make_runtime(args, int8_encode=True, int8_decode=True),
    }
    reports = []
    for path, report_path in zip(paths, report_paths):
        report = _process_video(
            path,
            args,
            runtimes,
            quality_only=bool(args.quality_only),
        )
        report["environment"] = {
            **environment,
            "gpu_after_sample": _base._gpu_snapshot(),
        }
        report["implementation_sources"] = implementation_sources
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        reports.append(report)
        print(
            json.dumps(
                {
                    "sample": path.name,
                    "quality_only": bool(args.quality_only),
                    "latent_relative_rmse": report["encode"]["E1_vs_E0"][
                        "relative_rmse"
                    ],
                    "mean_frame_psnr_db": {
                        name: metrics["mean_frame_psnr_db"]
                        for name, metrics in report["quality"]["vs_source"].items()
                    },
                },
                indent=2,
            )
        )
    if args.per_frame_csv:
        _write_per_frame_csv(csv_path, reports)
    effective_tiles = {
        report["source"]["name"]: report["config"]["encoder_tile_size_effective"]
        for report in reports
    }
    unique_effective_tiles = sorted(set(effective_tiles.values()))

    summary = {
        "suite": "H3 VAE four-way INT8 encode/decode round-trip",
        "schema_version": 1,
        "environment": {**environment, "gpu_after_suite": _base._gpu_snapshot()},
        "implementation_sources": implementation_sources,
        "config": {
            "model_code_dir": str(args.model_code_dir),
            "weights": str(args.weights),
            "weights_sha256": args.weights_sha256,
            "warmup_per_variant": int(args.warmup),
            "timed_runs_per_variant": int(args.runs),
            "quality_only": bool(args.quality_only),
            "decoder_tile_size": int(args.decoder_tile),
            "tile_batch": int(args.tile_batch),
            "encoder_tile_size": int(args.encoder_tile),
            "encoder_tile_size_effective": (
                unique_effective_tiles[0] if len(unique_effective_tiles) == 1 else None
            ),
            "encoder_tile_size_effective_by_sample": effective_tiles,
            "encoder_staged_batch": int(args.encoder_staged_batch),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_benchmark_limit": int(torch.backends.cudnn.benchmark_limit),
            "sample_count": len(reports),
            "per_frame_csv": str(csv_path) if args.per_frame_csv else None,
        },
        "samples": [
            {
                "name": report["source"]["name"],
                "sha256": report["source"]["sha256"],
                "shape": report["source"]["reference_shape"],
                "latent_error": {
                    key: value
                    for key, value in report["encode"]["E1_vs_E0"].items()
                    if key != "per_frame"
                },
                "quality_headlines_vs_source": report["quality"][
                    "headlines_vs_source"
                ],
                "quality_headlines_vs_default": report["quality"][
                    "headlines_vs_default"
                ],
                "timing": None
                if report["timing"] is None
                else {
                    "encode": report["timing"]["encode"]["timing"],
                    "decode": report["timing"]["decode"]["timing"],
                },
                "json": str(report_path),
            }
            for report, report_path in zip(reports, report_paths)
        ],
        "notes": [
            "This suite validates all four real combinations: FP16/FP16, FP16 encode plus INT8 decode, INT8 encode plus FP16 decode, and INT8/INT8.",
            "The default and both-int8 runtimes are independent; e_only and d_only paths use the appropriate component from those runtimes.",
            "No source videos, model weights, or generated media are copied into the output directory.",
            "Quality metrics do not claim losslessness; PSNR is measured against exact decoded RGB uint8 source data.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path), "samples": len(reports)}, indent=2))


if __name__ == "__main__":
    main()
