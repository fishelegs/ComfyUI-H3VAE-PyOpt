"""ComfyUI-wrapper smoke test for the opt-in INT8 H3 VAE paths.

This script is deliberately separate from the production runtime and the
video quality/performance benches.  It constructs one compiled runtime with
``int8_encode=True`` and ``int8_decode=True``, wraps it with
``build_comfy_vae``, checks the eight encoder and 72 decoder INT8 leaves, and
exercises the Comfy ``CoreModelPatcher`` CPU offload/reload path.

The default input follows the standard Comfy IMAGE contract: a reproducible
random FP16 RGB tensor with shape ``[T,H,W,C] = [17,256,256,3]`` (batch one).
``--rgb-dtype fp32`` mirrors the usual LoadImage output.  It is not a quality
sample.  Raw runtime encode receives exactly the values that
``comfy.sd.VAE.encode`` creates after ``movedim(-1, 1) -> movedim(1, 0) ->
unsqueeze(0) -> (*2 - 1)`` in the requested RGB dtype and final FP16 cast;
raw decoder output is compared after converting the wrapper's
``[B,T,H,W,C]`` result back to ``[B,C,T,H,W]``.

Using 4-D IMAGE input is important: the generic Comfy crop helper interprets
all non-batch dimensions as spatial for this API.  ``--frames 1`` is
available for a minimal API probe.

No model code, checkpoint, runtime, node, or global environment is modified.
The CLI requires explicit paths and performs no model construction on import.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


EXPECTED_ENCODER_MODULES = 8
EXPECTED_DECODER_MODULES = 72


def _add_comfy_root(comfy_root: Path) -> None:
    """Make the caller-selected Comfy SDK importable without hardcoding it."""

    resolved = str(comfy_root.resolve())
    if not comfy_root.is_dir():
        raise FileNotFoundError(f"Comfy root is not a directory: {comfy_root}")
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _synchronize(device: torch.device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _device_type_matches(actual: torch.device, expected: torch.device) -> bool:
    """Compare type and, for explicitly indexed devices, the CUDA index."""

    actual = torch.device(actual)
    expected = torch.device(expected)
    if actual.type != expected.type:
        return False
    if expected.index is None:
        return True
    return actual.index == expected.index


def _tensor_bytes(value: torch.Tensor) -> int:
    return int(value.numel() * value.element_size())


def _state_dict_bytes(module: nn.Module) -> tuple[int, int]:
    state = module.state_dict()
    return sum(_tensor_bytes(value) for value in state.values()), len(state)


def _unwrap_compiled(module: Any) -> Any:
    """Walk torch.compile's ``_orig_mod`` wrappers without assuming a class."""

    seen: set[int] = set()
    while hasattr(module, "_orig_mod") and id(module) not in seen:
        seen.add(id(module))
        module = module._orig_mod
    return module


def _compiled_paths(runtime: nn.Module) -> dict[str, Any]:
    decoder = getattr(getattr(runtime, "core", None), "decoder", None)
    prefix = getattr(runtime, "prefix", None)
    suffix = getattr(runtime, "suffix", None)
    return {
        "encoder_prefix": {
            "compiled_wrapper": bool(hasattr(prefix, "_orig_mod")),
            "type": type(prefix).__name__,
        },
        "encoder_suffix": {
            "compiled_wrapper": bool(hasattr(suffix, "_orig_mod")),
            "type": type(suffix).__name__,
        },
        "decoder": {
            "compiled_wrapper": bool(hasattr(decoder, "_orig_mod")),
            "type": type(decoder).__name__,
        },
    }


def _quantized_module_roots(runtime: nn.Module) -> tuple[tuple[str, Any], ...]:
    return (
        ("encoder.prefix", getattr(runtime, "prefix", None)),
        ("encoder.suffix", getattr(runtime, "suffix", None)),
        ("decoder", getattr(getattr(runtime, "core", None), "decoder", None)),
    )


def _find_quantized_modules(runtime: nn.Module) -> list[dict[str, Any]]:
    """Collect qweight/weight_scale leaves from both compiled model paths."""

    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for root_name, root in _quantized_module_roots(runtime):
        root = _unwrap_compiled(root)
        if not isinstance(root, nn.Module):
            continue
        for qualname, module in root.named_modules():
            qweight = getattr(module, "qweight", None)
            scale = getattr(module, "weight_scale", None)
            if not isinstance(qweight, torch.Tensor) or not isinstance(
                scale, torch.Tensor
            ):
                continue
            if id(module) in seen:
                continue
            seen.add(id(module))
            module_bytes, state_count = _state_dict_bytes(module)
            path = root_name if not qualname else f"{root_name}.{qualname}"
            kind = "decoder" if root_name == "decoder" else "encoder"
            records.append(
                {
                    "kind": kind,
                    "path": path,
                    "module_type": type(module).__name__,
                    "qweight_dtype": str(qweight.dtype),
                    "qweight_device": str(qweight.device),
                    "qweight_shape": list(qweight.shape),
                    "qweight_bytes": _tensor_bytes(qweight),
                    "scale_dtype": str(scale.dtype),
                    "scale_device": str(scale.device),
                    "scale_shape": list(scale.shape),
                    "scale_bytes": _tensor_bytes(scale),
                    "module_size_including_qparams_bytes": module_bytes,
                    "module_state_dict_key_count": state_count,
                }
            )
    return sorted(records, key=lambda item: item["path"])


def _capture_state(
    runtime: nn.Module,
    *,
    label: str,
    expected_device: torch.device,
) -> dict[str, Any]:
    records = _find_quantized_modules(runtime)
    encoder = [item for item in records if item["kind"] == "encoder"]
    decoder = [item for item in records if item["kind"] == "decoder"]
    if len(encoder) != EXPECTED_ENCODER_MODULES or len(decoder) != EXPECTED_DECODER_MODULES:
        raise RuntimeError(
            "unexpected INT8 module count: "
            f"encoder={len(encoder)} (expected {EXPECTED_ENCODER_MODULES}), "
            f"decoder={len(decoder)} (expected {EXPECTED_DECODER_MODULES})"
        )

    expected_device = torch.device(expected_device)
    q_devices = sorted({item["qweight_device"] for item in records})
    scale_devices = sorted({item["scale_device"] for item in records})
    qweight_ok = all(item["qweight_dtype"] == "torch.int8" for item in records)
    scale_ok = all(item["scale_dtype"] == "torch.float32" for item in records)
    device_ok = all(
        _device_type_matches(torch.device(item["qweight_device"]), expected_device)
        and _device_type_matches(torch.device(item["scale_device"]), expected_device)
        for item in records
    )
    if not qweight_ok or not scale_ok or not device_ok:
        raise RuntimeError(
            f"invalid INT8 state at {label}: qweight_int8={qweight_ok}, "
            f"scale_fp32={scale_ok}, device_match={device_ok}"
        )
    runtime_state_bytes, runtime_state_keys = _state_dict_bytes(runtime)
    return {
        "label": label,
        "expected_device": str(expected_device),
        "modules": len(records),
        "encoder_modules": len(encoder),
        "decoder_modules": len(decoder),
        "qweight_int8": qweight_ok,
        "scale_fp32": scale_ok,
        "qweight_device": q_devices,
        "scale_device": scale_devices,
        "device_match": device_ok,
        "quantized_qweight_bytes": sum(item["qweight_bytes"] for item in records),
        "quantized_scale_bytes": sum(item["scale_bytes"] for item in records),
        "quantized_module_state_dict_bytes": sum(
            item["module_size_including_qparams_bytes"] for item in records
        ),
        "runtime_state_dict_bytes": runtime_state_bytes,
        "runtime_state_dict_key_count": runtime_state_keys,
        "modules_detail": records,
    }


def _finite(value: torch.Tensor, name: str) -> bool:
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"{name} is not a tensor: {type(value).__name__}")
    result = bool(torch.isfinite(value).all().item())
    if not result:
        raise RuntimeError(f"{name} is non-finite")
    return result


def _metrics(reference: torch.Tensor, candidate: torch.Tensor, *, name: str) -> dict[str, Any]:
    """Report actual differences without assuming they are exactly zero."""

    reference = reference.detach().float().cpu().contiguous()
    candidate = candidate.detach().float().cpu().contiguous()
    reference_finite = _finite(reference, f"{name} reference")
    candidate_finite = _finite(candidate, f"{name} candidate")
    result: dict[str, Any] = {
        "shape_match": tuple(reference.shape) == tuple(candidate.shape),
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_finite": reference_finite,
        "candidate_finite": candidate_finite,
        "finite": reference_finite and candidate_finite,
    }
    if not result["shape_match"]:
        return result
    diff = candidate - reference
    result.update(
        {
            "rmse": float(diff.square().mean().sqrt().item()),
            "max_abs": float(diff.abs().max().item()),
            "mae": float(diff.abs().mean().item()),
        }
    )
    return result


def _raw_normalized_input(rgb_thwc: torch.Tensor) -> torch.Tensor:
    """Mirror Comfy process_input followed by VAE FP16 conversion."""

    if rgb_thwc.dtype not in (torch.float16, torch.float32):
        raise ValueError("smoke RGB input must be FP16 or FP32")
    if rgb_thwc.ndim != 4 or rgb_thwc.shape[-1] != 3:
        raise ValueError("smoke RGB input must be standard Comfy [T,H,W,3]")
    moved = rgb_thwc.movedim(-1, 1)
    # This is the exact latent_dim=3 branch in comfy.sd.VAE.encode for its
    # 4-D IMAGE input: [T,C,H,W] -> [C,T,H,W] -> [1,C,T,H,W].
    # Preserve Comfy's channels-last-3d layout.  VAE.encode performs this
    # exact move/unsqueeze sequence and does not call contiguous() before
    # process_input; making the raw helper contiguous would benchmark a
    # different runtime input layout and can change layout-sensitive kernels.
    moved = moved.movedim(1, 0).unsqueeze(0)
    # Comfy applies process_input before .to(self.vae_dtype); INT8 mode then
    # requires the resulting runtime input to be FP16.
    return (moved * 2.0 - 1.0).to(torch.float16)


def _wrapper_decode_bcthw(wrapper_output: torch.Tensor) -> torch.Tensor:
    """Convert Comfy VAE.decode's BTHWC output to runtime BCTHW."""

    if wrapper_output.ndim != 5 or wrapper_output.shape[-1] != 3:
        raise RuntimeError(
            "Comfy wrapper decode must return [B,T,H,W,3], got "
            f"{tuple(wrapper_output.shape)}"
        )
    return wrapper_output.movedim(-1, 1).contiguous()


def _run_phase(
    runtime: nn.Module,
    wrapper: Any,
    rgb_thwc: torch.Tensor,
    latent_bcthw: torch.Tensor,
    *,
    label: str,
    encoded_roundtrip: bool = False,
) -> dict[str, Any]:
    """Compare raw runtime and Comfy wrapper calls for one loaded state."""

    raw_input = _raw_normalized_input(rgb_thwc)
    _finite(raw_input, f"{label} raw normalized input")
    _finite(latent_bcthw, f"{label} latent")
    with torch.inference_mode():
        raw_latent = runtime.encode(raw_input)
        _synchronize(runtime._device())
        wrapper_latent = wrapper.encode(rgb_thwc)
        _synchronize(runtime._device())
        raw_pixels = runtime.decode(latent_bcthw)
        _synchronize(runtime._device())
        wrapper_pixels_bthwc = wrapper.decode(latent_bcthw)
        _synchronize(runtime._device())

    _finite(raw_latent, f"{label} raw latent")
    _finite(wrapper_latent, f"{label} wrapper latent")
    _finite(raw_pixels, f"{label} raw pixels")
    _finite(wrapper_pixels_bthwc, f"{label} wrapper pixels")
    wrapper_pixels = _wrapper_decode_bcthw(wrapper_pixels_bthwc)
    result = {
        "label": label,
        "raw_input_shape_bcthw": list(raw_input.shape),
        "raw_input_dtype": str(raw_input.dtype),
        "raw_latent_shape": list(raw_latent.shape),
        "wrapper_latent_shape": list(wrapper_latent.shape),
        "raw_decode_shape_bcthw": list(raw_pixels.shape),
        "wrapper_decode_shape_bthwc": list(wrapper_pixels_bthwc.shape),
        "wrapper_decode_reshaped_shape_bcthw": list(wrapper_pixels.shape),
        "raw_vs_wrapper_encode": _metrics(
            raw_latent, wrapper_latent, name=f"{label} encode"
        ),
        "raw_vs_wrapper_decode": _metrics(
            raw_pixels, wrapper_pixels, name=f"{label} decode"
        ),
    }
    compatibility_metrics = [
        result["raw_vs_wrapper_encode"],
        result["raw_vs_wrapper_decode"],
    ]
    if encoded_roundtrip:
        # VAE.encode returns Comfy's intermediate dtype (normally FP32), while
        # INT8 decode explicitly requires FP16 latent input.  This checks the
        # actual wrapper-produced latent after the same FP16 cast used by the
        # regular Comfy decode path, rather than an independent random latent.
        encoded_latent_fp16 = wrapper_latent.to(torch.float16)
        with torch.inference_mode():
            encoded_raw_pixels = runtime.decode(encoded_latent_fp16)
            _synchronize(runtime._device())
            encoded_wrapper_pixels_bthwc = wrapper.decode(encoded_latent_fp16)
            _synchronize(runtime._device())
        _finite(encoded_raw_pixels, f"{label} encoded-latent raw pixels")
        _finite(
            encoded_wrapper_pixels_bthwc,
            f"{label} encoded-latent wrapper pixels",
        )
        encoded_wrapper_pixels = _wrapper_decode_bcthw(encoded_wrapper_pixels_bthwc)
        result["encoded_latent_roundtrip"] = {
            "latent_shape_bcthw": list(encoded_latent_fp16.shape),
            "latent_dtype": str(encoded_latent_fp16.dtype),
            "raw_decode_shape_bcthw": list(encoded_raw_pixels.shape),
            "wrapper_decode_shape_bthwc": list(encoded_wrapper_pixels_bthwc.shape),
            "wrapper_decode_reshaped_shape_bcthw": list(encoded_wrapper_pixels.shape),
            "raw_vs_wrapper_decode": _metrics(
                encoded_raw_pixels,
                encoded_wrapper_pixels,
                name=f"{label} encoded-latent decode",
            ),
        }
        compatibility_metrics.append(result["encoded_latent_roundtrip"]["raw_vs_wrapper_decode"])
    else:
        encoded_latent_fp16 = None
        encoded_raw_pixels = None
        encoded_wrapper_pixels_bthwc = None
        encoded_wrapper_pixels = None
    result["compatibility_matches"] = all(
        metric.get("shape_match")
        and metric.get("finite")
        and metric.get("max_abs") == 0.0
        for metric in compatibility_metrics
    )
    if not result["compatibility_matches"]:
        raise RuntimeError(
            f"{label} raw/wrapper compatibility mismatch: "
            f"encode={result['raw_vs_wrapper_encode']}, "
            f"decode={result['raw_vs_wrapper_decode']}, "
            f"encoded_latent={result.get('encoded_latent_roundtrip')}"
        )
    # Keep only CPU snapshots used for before/after metrics; GPU temporaries
    # must be released before the patcher offload step.
    result["_raw_latent_cpu"] = raw_latent.detach().float().cpu().contiguous()
    result["_wrapper_latent_cpu"] = wrapper_latent.detach().float().cpu().contiguous()
    result["_raw_pixels_cpu"] = raw_pixels.detach().float().cpu().contiguous()
    result["_wrapper_pixels_cpu"] = wrapper_pixels.detach().float().cpu().contiguous()
    del (
        raw_input,
        raw_latent,
        wrapper_latent,
        raw_pixels,
        wrapper_pixels_bthwc,
        wrapper_pixels,
        encoded_latent_fp16,
        encoded_raw_pixels,
        encoded_wrapper_pixels_bthwc,
        encoded_wrapper_pixels,
    )
    return result


def _strip_snapshots(phase: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in phase.items() if not key.startswith("_")}


def _cross_phase(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_encode_before_vs_after": _metrics(
            before["_raw_latent_cpu"],
            after["_raw_latent_cpu"],
            name="raw encode before/after offload",
        ),
        "wrapper_encode_before_vs_after": _metrics(
            before["_wrapper_latent_cpu"],
            after["_wrapper_latent_cpu"],
            name="wrapper encode before/after offload",
        ),
        "raw_decode_before_vs_after": _metrics(
            before["_raw_pixels_cpu"],
            after["_raw_pixels_cpu"],
            name="raw decode before/after offload",
        ),
        "wrapper_decode_before_vs_after": _metrics(
            before["_wrapper_pixels_cpu"],
            after["_wrapper_pixels_cpu"],
            name="wrapper decode before/after offload",
        ),
    }


def _environment(device: torch.device) -> dict[str, Any]:
    try:
        ck_version = importlib.metadata.version("comfy-kitchen")
    except importlib.metadata.PackageNotFoundError:
        ck_version = None
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        triton_version = None
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_hip": torch.version.hip,
        "triton": triton_version,
        "comfy_kitchen": ck_version,
        "device": str(device),
    }
    if device.type == "cuda":
        result.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--model-code-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--rgb-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument(
        "--encoded-roundtrip",
        action="store_true",
        help="decode the FP16-cast latent produced by wrapper.encode",
    )
    parser.add_argument("--decoder-tile", type=int, default=256)
    parser.add_argument("--tile-batch", type=int, default=2)
    parser.add_argument("--encoder-tile", type=int, default=256)
    parser.add_argument("--encoder-staged-batch", type=int, default=4)
    return parser.parse_args(argv)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Comfy INT8 smoke requires --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.frames < 1 or args.height < 16 or args.width < 16:
        raise ValueError("frames must be >=1 and height/width must be >=16")
    if args.height % 16 or args.width % 16:
        raise ValueError("height and width must be divisible by 16")
    if not args.model_code_dir.is_dir() or not args.weights.is_file():
        raise FileNotFoundError("model-code-dir and weights must exist")
    _add_comfy_root(args.comfy_root)

    # Import runtime/build only after CLI path setup.  Importing this module
    # itself remains CPU-safe and does not import Comfy or construct a model.
    from h3vae_runtime import H3VAEPyOptRuntime, build_comfy_vae

    torch.manual_seed(int(args.seed))
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    rgb_dtype = torch.float32 if args.rgb_dtype == "fp32" else torch.float16
    rgb_thwc = torch.rand(
        (int(args.frames), int(args.height), int(args.width), 3),
        generator=generator,
        dtype=rgb_dtype,
    )
    token_frames = 1 if args.frames == 1 else max(1, (args.frames - 5) // 17 * 5 + 2)
    latent_bcthw = torch.randn(
        (1, 24, token_frames, args.height // 16, args.width // 16),
        generator=generator,
        dtype=torch.float16,
    )

    runtime = H3VAEPyOptRuntime(
        model_code_dir=str(args.model_code_dir),
        weights_path=str(args.weights),
        device=str(device),
        dtype=torch.float16,
        decoder_tile_size=int(args.decoder_tile),
        tile_batch=int(args.tile_batch),
        compile_decoder=True,
        compile_encoder=True,
        encoder_staged_batch=int(args.encoder_staged_batch),
        encoder_tile_size=int(args.encoder_tile),
        log_calls=False,
        int8_decode=True,
        int8_encode=True,
    ).eval()
    wrapper = build_comfy_vae(runtime, dtype=torch.float16)
    if not bool(getattr(runtime, "int8_encode", False)) or not bool(
        getattr(runtime, "int8_decode", False)
    ):
        raise RuntimeError("runtime did not retain both explicit INT8 flags")
    paths = _compiled_paths(runtime)
    if not all(item["compiled_wrapper"] for item in paths.values()):
        raise RuntimeError(f"compiled path missing: {paths}")

    crop_probe = wrapper.vae_encode_crop_pixels(rgb_thwc)
    crop_changed = tuple(crop_probe.shape) != tuple(rgb_thwc.shape)
    if crop_changed:
        raise RuntimeError(
            "Comfy wrapper crop changed standard IMAGE input shape: "
            f"{tuple(rgb_thwc.shape)} -> {tuple(crop_probe.shape)}"
        )

    state_before = _capture_state(
        runtime, label="cuda_before_offload", expected_device=device
    )
    phase_before = _run_phase(
        runtime,
        wrapper,
        rgb_thwc,
        latent_bcthw,
        label="before_offload",
        encoded_roundtrip=bool(args.encoded_roundtrip),
    )

    # Exercise exactly the requested CoreModelPatcher transitions.  No output
    # is assumed bit-identical; phase metrics below report the actual result.
    wrapper.patcher.unpatch_model(torch.device("cpu"))
    state_cpu = _capture_state(
        runtime, label="cpu_after_unpatch", expected_device=torch.device("cpu")
    )
    wrapper.patcher.load(device_to=device, full_load=True)
    state_after = _capture_state(
        runtime, label="cuda_after_reload", expected_device=device
    )
    phase_after = _run_phase(
        runtime,
        wrapper,
        rgb_thwc,
        latent_bcthw,
        label="after_reload",
        encoded_roundtrip=bool(args.encoded_roundtrip),
    )

    report = {
        "status": "ok",
        "scope": "compiled Comfy wrapper smoke; random tensors, not quality",
        "environment": _environment(device),
        "config": {
            "comfy_root": str(args.comfy_root),
            "model_code_dir": str(args.model_code_dir),
            "weights": str(args.weights),
            "device": str(device),
            "frames": int(args.frames),
            "height": int(args.height),
            "width": int(args.width),
            "seed": int(args.seed),
            "rgb_dtype": args.rgb_dtype,
            "encoded_roundtrip": bool(args.encoded_roundtrip),
            "decoder_tile": int(args.decoder_tile),
            "tile_batch": int(args.tile_batch),
            "encoder_tile": int(args.encoder_tile),
            "encoder_staged_batch": int(args.encoder_staged_batch),
            "int8_encode": True,
            "int8_decode": True,
            "compile_encoder": True,
            "compile_decoder": True,
            "comfy_crop_changed_probe": crop_changed,
        },
        "compiled_paths": paths,
        "input": {
            "wrapper_input_shape_thwc": list(rgb_thwc.shape),
            "wrapper_input_dtype": str(rgb_thwc.dtype),
            "wrapper_input_range": [float(rgb_thwc.min()), float(rgb_thwc.max())],
            "raw_normalized_shape_bcthw": list(_raw_normalized_input(rgb_thwc).shape),
            "raw_normalized_dtype": str(_raw_normalized_input(rgb_thwc).dtype),
            "raw_normalized_contract": (
                f"same {rgb_thwc.dtype} Comfy IMAGE movedim(-1,1)->movedim(1,0)"
                "->unsqueeze(0), process_input (* 2 - 1), then cast FP16"
            ),
        },
        "latent": {
            "shape_bcthw": list(latent_bcthw.shape),
            "dtype": str(latent_bcthw.dtype),
            "finite": bool(torch.isfinite(latent_bcthw).all()),
        },
        "module_state": {
            "before_offload": state_before,
            "cpu_after_unpatch": state_cpu,
            "after_cuda_reload": state_after,
        },
        "before_offload": _strip_snapshots(phase_before),
        "after_reload": _strip_snapshots(phase_after),
        "compatibility_matches": bool(
            phase_before["compatibility_matches"]
            and phase_after["compatibility_matches"]
        ),
        "before_after_offload": _cross_phase(phase_before, phase_after),
        "notes": [
            "Eight encoder INT8 Conv3d leaves and 72 decoder INT8 linear leaves are required.",
            "qweight and FP32 scale device/dtype are checked before offload, on CPU, and after full CUDA reload.",
            "Comfy wrapper encode compares against raw runtime using identical FP16-normalized RGB values.",
            "Decoder comparison reshapes wrapper BTHWC output to raw runtime BCTHW.",
            "Raw/wrapper compatibility is required to be finite, shape-matched, and exact; before/after metrics are reported without a zero assertion because offload algorithms may change.",
            "Random inputs/latents prove wrapper plumbing only; they are not image/video quality evidence.",
            "When --encoded-roundtrip is set, the report also checks decode of the actual FP16-cast wrapper.encode latent; this is a finite shape smoke, not quality evidence.",
        ],
    }
    del phase_before, phase_after, crop_probe, rgb_thwc, latent_bcthw
    gc.collect()
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report: dict[str, Any]
    try:
        report = run_smoke(args)
    except Exception as exc:  # noqa: BLE001 - write reproducible failure JSON
        report = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "config": {
                "comfy_root": str(args.comfy_root),
                "model_code_dir": str(args.model_code_dir),
                "weights": str(args.weights),
                "device": args.device,
                "frames": args.frames,
                "height": args.height,
                "width": args.width,
            },
        }
        _write_report(args.output, report)
        print(json.dumps(report, indent=2))
        raise
    _write_report(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI only
    raise SystemExit(main())
