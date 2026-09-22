"""Optimized MiniMax H3 video VAE runtime for ComfyUI.

Loads the reference klvae implementation (FL2VA video_vae bundle) and wires in
the validated PyTorch optimizations from the 2026-09 optimization campaign:

  decoder: QK RMSNorm+RoPE Triton fusion + whole-decoder torch.compile
           (max-autotune-no-cudagraphs) + batched spatial tile scheduler
  encoder: GN/SiLU/padding Triton fusions (extend_bias_pack prefix +
           fused_suffix all_stages) + channels-last-3d + internal padding +
           staged clip batching, whole-graph compiled

The runtime implements the ``first_stage_model`` interface that ComfyUI's
``comfy.sd.VAE`` expects from ``comfy.ldm.minimax.vae.MiniMaxH3VideoVAE``
(decode / encode / decode_output_shape / comfy_has_chunked_io / *_tiled), so
the stock VAEDecode / VAEEncode nodes drive it unchanged.

Inputs/outputs follow the core implementation's conventions exactly:
  decode: z [B,24,T,H,W] normalized latents -> float32 pixels [B,3,F,16H,16W] in [0,1]
  encode: x [B,3,T,H,W] in [-1,1]          -> normalized latents [B,24,t,H/16,W/16]
"""
from __future__ import annotations

import json
import importlib.metadata
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch

logger = logging.getLogger("ComfyUI_H3VAE_PyOpt")

DEFAULT_MODEL_CODE_DIR = os.environ.get("H3_VAE_MODEL_CODE_DIR", "")
DEFAULT_WEIGHTS_FALLBACK = os.environ.get("H3_VAE_WEIGHTS_PATH", "")

# inductor compile options used by the validated encoder fusion campaign
ENCODER_COMPILE_OPTIONS = {
    "max_autotune": True,
    "triton.cudagraphs": False,
    "max_autotune_conv_backends": "ATEN",
}
DECODER_COMPILE_MODE = "max-autotune-no-cudagraphs"

_OPT_DIR = Path(__file__).resolve().parent / "opt"


def _validate_int8_decode_options(
    *,
    enabled: bool,
    fast_linear: bool,
    device,
    dtype: torch.dtype,
    cuda_available: bool | None = None,
    hip: str | None = None,
    capability: tuple[int, int] | None = None,
    ck_version: str | None = None,
    has_quantize_api: bool | None = None,
    has_linear_api: bool | None = None,
    check_dependencies: bool = True,
) -> None:
    """Validate the explicit INT8 decoder contract without loading a model.

    The pure arguments make the policy testable on CPU CI.  Production calls
    perform the CUDA and dependency probes before model construction, so an
    unsupported request fails instead of silently returning an FP16 decoder.
    """
    if not enabled:
        return
    if fast_linear:
        raise ValueError(
            "int8_decode and fast_linear are mutually exclusive; choose one"
        )
    try:
        device_type = torch.device(device).type
    except (TypeError, RuntimeError) as exc:
        raise RuntimeError(
            f"int8_decode requires a CUDA device, got {device!r}"
        ) from exc
    if device_type != "cuda":
        raise RuntimeError(
            f"int8_decode requires a CUDA device, got {device!r}"
        )
    if dtype != torch.float16:
        raise RuntimeError("int8_decode requires dtype=fp16")
    if hip is None:
        hip = torch.version.hip
    if hip is not None:
        raise RuntimeError("int8_decode is CUDA-only and does not support ROCm")
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    if not cuda_available:
        raise RuntimeError("int8_decode requires an available CUDA GPU")
    if capability is None:
        capability = torch.cuda.get_device_capability(torch.device(device))
    if tuple(capability) < (8, 0):
        raise RuntimeError(
            "int8_decode requires CUDA compute capability SM80 or newer; "
            f"got SM{capability[0]}{capability[1]}"
        )
    if not check_dependencies:
        return
    if ck_version != "0.2.34":
        raise RuntimeError(
            "int8_decode requires comfy-kitchen==0.2.34; "
            f"found {ck_version or 'not installed'}"
        )
    if has_quantize_api is not True or has_linear_api is not True:
        raise RuntimeError(
            "int8_decode requires comfy-kitchen quantize_int8_rowwise "
            "and int8_linear APIs"
        )


def _validate_int8_encode_options(
    *,
    enabled: bool,
    device,
    dtype: torch.dtype,
    cuda_available: bool | None = None,
    hip: str | None = None,
    capability: tuple[int, int] | None = None,
) -> None:
    """Validate the explicit CUDA/FP16 contract for mixed INT8 encoding."""

    if not enabled:
        return
    try:
        device_type = torch.device(device).type
    except (TypeError, RuntimeError) as exc:
        raise RuntimeError(
            f"int8_encode requires a CUDA device, got {device!r}"
        ) from exc
    if device_type != "cuda":
        raise RuntimeError(f"int8_encode requires a CUDA device, got {device!r}")
    if dtype != torch.float16:
        raise RuntimeError("int8_encode requires dtype=fp16")
    if hip is None:
        hip = torch.version.hip
    if hip is not None:
        raise RuntimeError("int8_encode is CUDA-only and does not support ROCm")
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    if not cuda_available:
        raise RuntimeError("int8_encode requires an available CUDA GPU")
    if capability is None:
        capability = torch.cuda.get_device_capability(torch.device(device))
    if tuple(capability) < (8, 0):
        raise RuntimeError(
            "int8_encode requires CUDA compute capability SM80 or newer; "
            f"got SM{capability[0]}{capability[1]}"
        )


def _int8_dependency_info() -> tuple[str | None, bool, bool]:
    """Import CK and inspect the exact APIs used by the production path."""
    try:
        import comfy_kitchen as ck
    except Exception as exc:  # noqa: BLE001 - surface an explicit option error
        raise RuntimeError(
            "int8_decode could not import comfy-kitchen 0.2.34"
        ) from exc
    try:
        version = importlib.metadata.version("comfy-kitchen")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return (
        version,
        callable(getattr(ck, "quantize_int8_rowwise", None)),
        callable(getattr(ck, "int8_linear", None)),
    )


def _import_opt_modules():
    """Make the vendored optimization modules importable (flat imports)."""
    p = str(_OPT_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    from decoder_fused_qk_rope import clone_with_qk_fusion
    from encoder_fused_batched_norm import fused_suffix
    from encoder_fused_norm_next import next_fused_prefix
    from encoder_internal_padding import replace_selected
    from encoder_staged_batch import encode_staged
    from pytorch_decoder_optim import enable_decoder_attention_in_graph
    return (clone_with_qk_fusion, fused_suffix, next_fused_prefix,
            replace_selected, encode_staged, enable_decoder_attention_in_graph)


def _load_klvae_core(model_code_dir: str):
    """Build the reference AutoencoderKLLegacy (klvae) from the FL2VA bundle."""
    if not model_code_dir:
        raise ValueError("Set H3_VAE_MODEL_CODE_DIR or pass model_code_dir")
    model_code_dir = Path(model_code_dir)
    if not (model_code_dir / "config.json").is_file():
        raise FileNotFoundError(f"MiniMax VAE config.json not found in {model_code_dir}")
    pkg_parent = str(model_code_dir.parent)
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)

    cfg = json.loads((model_code_dir / "config.json").read_text())
    # seed the bundled parallel state for single-process inference
    from video_vae.parallel import get_parallel_state
    state = get_parallel_state()
    if isinstance(state, dict) and not state:
        state.update({
            "group_size": 1, "group_rank": 0, "local_process_group": None,
            "sp_size": 1, "sp_rank": 0, "sp_enabled": False,
            "sp_process_group": None, "tp_size": 1, "tp_rank": 0,
        })

    from video_vae.klvae import AutoencoderKLLegacy
    load_kwargs = {
        "clip_length": int(cfg["vae_clip_length"]),
        "token_drop": int(cfg["vae_token_drop"]),
        "encoder_tiling": int(cfg["vae_encoder_tiling"]),
        "decoder_tiling": int(cfg["vae_decoder_tiling"]),
        "parallel_tiling": int(cfg["vae_parallel_tiling"]),
        "tile_size": int(cfg["vae_tile_size"]),
        "tile_overlap_min": int(cfg["vae_tile_overlap_min"]),
        "encoder_parallel": int(cfg["vae_encoder_parallel"]),
        "decoder_parallel": int(cfg["vae_decoder_parallel"]),
        "chunk_dim": int(cfg["vae_chunk_dim"]),
    }
    source_dir = model_code_dir / cfg["source_path"]
    source_config = AutoencoderKLLegacy.load_config(str(source_dir))
    core, _ = AutoencoderKLLegacy.from_config(
        source_config, return_unused_kwargs=True, **load_kwargs)
    core.eval()
    return core, cfg


def _load_weights(core, weights_path: str):
    if not weights_path:
        raise ValueError("Set H3_VAE_WEIGHTS_PATH or select a VAE file")
    if not Path(weights_path).is_file():
        raise FileNotFoundError(weights_path)
    weights_path = str(weights_path)
    if weights_path.endswith(".safetensors"):
        import safetensors.torch
        sd = safetensors.torch.load_file(weights_path)
    else:
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    missing, unexpected = core.load_state_dict(sd, strict=False)
    allowed = {"latents_mean", "latents_std"}
    bad_unexpected = set(unexpected) - allowed
    if missing or bad_unexpected:
        raise RuntimeError(
            f"weight mismatch for {weights_path}: missing={list(missing)[:8]} "
            f"unexpected={sorted(bad_unexpected)[:8]}")
    if unexpected:
        logger.info("[H3VAE-PyOpt] ignored non-model buffer keys: %s", sorted(unexpected))


class H3VAEPyOptRuntime(torch.nn.Module):
    """comfy first_stage_model interface over the optimized klvae runtime."""

    comfy_has_chunked_io = True

    def __init__(self, model_code_dir: str = DEFAULT_MODEL_CODE_DIR,
                 weights_path: str = DEFAULT_WEIGHTS_FALLBACK,
                 device: str = "cuda", dtype: torch.dtype = torch.float16,
                 decoder_tile_size: int = 256, tile_batch: int = 2,
                 compile_decoder: bool = True, compile_encoder: bool = True,
                 qk_rows: int = 8, encoder_staged_batch: int = 4,
                 log_calls: bool = True, encoder_tile_size: int = 0,
                 fast_linear: bool = False, int8_decode: bool = False,
                 int8_encode: bool = False):
        super().__init__()
        int8_decode = bool(int8_decode)
        int8_encode = bool(int8_encode)
        _validate_int8_decode_options(
            enabled=int8_decode,
            fast_linear=bool(fast_linear),
            device=device,
            dtype=dtype,
            check_dependencies=False,
        )
        if int8_decode:
            ck_version, has_quantize_api, has_linear_api = _int8_dependency_info()
            _validate_int8_decode_options(
                enabled=True,
                fast_linear=bool(fast_linear),
                device=device,
                dtype=dtype,
                ck_version=ck_version,
                has_quantize_api=has_quantize_api,
                has_linear_api=has_linear_api,
            )
        _validate_int8_encode_options(
            enabled=int8_encode,
            device=device,
            dtype=dtype,
        )
        if encoder_tile_size < 0 or encoder_tile_size % 16:
            raise ValueError("encoder_tile_size must be 0 (auto) or a positive multiple of 16")
        if int(tile_batch) < 0:
            raise ValueError("tile_batch must be non-negative")
        (clone_with_qk_fusion, fused_suffix, next_fused_prefix, replace_selected,
         encode_staged, enable_decoder_attention_in_graph) = _import_opt_modules()
        self._encode_staged = encode_staged

        core, cfg = _load_klvae_core(model_code_dir)
        if encoder_tile_size and encoder_tile_size <= core.tile_overlap_min:
            raise ValueError("encoder_tile_size must exceed encoder tile overlap")
        _load_weights(core, weights_path)
        core.to(device=device, dtype=dtype)
        self.core = core
        self.config = cfg
        self.encoder_tile_size = int(encoder_tile_size)
        core.tile_size = self.encoder_tile_size or 672

        # ---------------- decoder wiring ----------------
        raw_dec = core.decoder
        enable_decoder_attention_in_graph(raw_dec)
        new_dec = clone_with_qk_fusion(raw_dec, qk_rows)
        if fast_linear:
            from opt.kitchen_linear import install_kitchen_linears
            count = install_kitchen_linears(new_dec)
            logger.info("[H3VAE-PyOpt] experimental comfy-kitchen linears: %d", count)
        elif int8_decode:
            from opt.kitchen_int8 import install_kitchen_int8_ffn
            try:
                count = install_kitchen_int8_ffn(
                    new_dec, mode="INT8both", require_cuda=True
                )
            except Exception as exc:  # noqa: BLE001 - explicit option has no fallback
                raise RuntimeError(
                    "int8_decode installation failed; no fallback was applied"
                ) from exc
            logger.info(
                "[H3VAE-PyOpt] experimental comfy-kitchen INT8 FFNs: %d",
                count,
            )
        if compile_decoder:
            new_dec = torch.compile(new_dec, mode=DECODER_COMPILE_MODE, dynamic=False)
        core.decoder = new_dec
        core.decoder_tile_size = int(decoder_tile_size)
        core.stack_tiling = False
        if int(tile_batch) == 0 or int(tile_batch) > 1:
            self._install_batched_tiles(core, int(tile_batch))

        # ---------------- encoder wiring ----------------
        enc = core.encoder.to(memory_format=torch.channels_last_3d)
        quant = core.quant_conv.to(memory_format=torch.channels_last_3d)
        replace_selected(enc, ["down.0.downsample.conv"])
        prefix = next_fused_prefix(enc, "extend_bias_pack")
        suffix = fused_suffix(enc, quant, all_stages=True)
        int8_encoder_metadata = None
        if int8_encode:
            from opt.encoder_int8_integration import install_int8_encoder_prefix
            try:
                int8_encoder_metadata = install_int8_encoder_prefix(
                    prefix, require_cuda=True, tile_variant="auto"
                )
            except Exception as exc:  # noqa: BLE001 - explicit option has no fallback
                raise RuntimeError(
                    "int8_encode installation failed; no fallback was applied"
                ) from exc
            logger.info(
                "[H3VAE-PyOpt] experimental mixed INT8 encoder convolutions: %d",
                int8_encoder_metadata["count"],
            )
        if compile_encoder:
            prefix = torch.compile(prefix, options=ENCODER_COMPILE_OPTIONS,
                                   dynamic=False, fullgraph=True)
            suffix = torch.compile(suffix, options=ENCODER_COMPILE_OPTIONS,
                                   dynamic=False, fullgraph=True)
        self.prefix = prefix
        self.suffix = suffix
        self.encoder_staged_batch = int(encoder_staged_batch)
        self.fast_linear = bool(fast_linear)
        self.int8_decode = int8_decode
        self.int8_encode = int8_encode
        self.int8_encoder_metadata = int8_encoder_metadata

        # ---------------- normalization constants ----------------
        lat_mean = torch.tensor(cfg["latents_mean"], dtype=dtype, device=device)
        lat_std = torch.tensor(cfg["latents_std"], dtype=dtype, device=device)
        self.latents_mean = lat_mean.view(1, -1, 1, 1, 1)
        self.latents_std = lat_std.view(1, -1, 1, 1, 1)
        img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32, device=device)
        img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32, device=device)
        self._img_mean = img_mean.view(1, 3, 1, 1, 1)
        self._img_std = img_std.view(1, 3, 1, 1, 1)

        self.log_calls = bool(log_calls)
        self._dec_calls = 0
        self._dec_total_ms = 0.0
        self._enc_calls = 0
        self._enc_total_ms = 0.0

        logger.info(
            "[H3VAE-PyOpt] runtime ready: decoder_tile=%d encoder_tile=%s "
            "tile_batch=%d compile_dec=%s compile_enc=%s staged_batch=%d "
            "fast_linear=%s int8_decode=%s int8_encode=%s sdpa=%s",
            core.decoder_tile_size, self.encoder_tile_size or "auto", tile_batch, compile_decoder,
            compile_encoder, self.encoder_staged_batch,
            self.fast_linear, self.int8_decode, self.int8_encode,
            os.environ.get("MINIMAX_H3_TORCH_SDPA_BACKEND", "auto"))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _install_batched_tiles(core, batch: int):
        """Batch decoder tiles; zero selects a conservative free-memory cap."""
        original = core._run_tile_tasks

        def batched(tiles, indices, forward_fn, stack_tiling, cls_agg=None):
            if (cls_agg is not None or not tiles or tiles[0].shape[1] != 24
                    or tiles[0].shape[0] != 1 or torch.is_grad_enabled()):
                return original(tiles, indices, forward_fn, stack_tiling, cls_agg)
            current_batch = batch
            if batch == 0:
                free, _ = torch.cuda.mem_get_info(tiles[0].device)
                # One full video output can already occupy several GiB. Keep
                # headroom for it and for other GPU users before pairing tiles.
                reserve = 12 * 1024**3
                current_batch = max(1, min(2, (free - reserve) // (4 * 1024**3)))
            if current_batch == 1:
                core._last_tile_batch = 1
                return original(tiles, indices, forward_fn, stack_tiling, cls_agg)
            core._last_tile_batch = current_batch
            outputs = []
            for start in range(0, len(indices), current_batch):
                selected = indices[start:start + current_batch]
                pixels = forward_fn(torch.cat([tiles[i] for i in selected], dim=0))
                outputs.extend(pixels.split(1, dim=0))
            return outputs

        core._run_tile_tasks = batched

    def _device(self):
        return self.latents_mean.device

    def _frames_from_tokens(self, t: int) -> int:
        if t <= 1:
            return 1
        return max(1, (t - 2) // 5 * 17 + 5)

    def _tokens_from_frames(self, frames: int) -> int:
        if frames <= 1:
            return 1
        return max(1, (frames - 5) // 17 * 5 + 2)

    def _log_call(self, kind: str, t0: float):
        if not self.log_calls:
            return
        if self._device().type == "cuda":
            torch.cuda.synchronize()
        dt = (time.monotonic() - t0) * 1000.0
        if kind == "decode":
            self._dec_calls += 1
            self._dec_total_ms += dt
            mean = self._dec_total_ms / self._dec_calls
            logger.info("[H3VAE-PyOpt] decode #%d: %.1f ms (running mean %.1f ms)",
                        self._dec_calls, dt, mean)
        else:
            self._enc_calls += 1
            self._enc_total_ms += dt
            mean = self._enc_total_ms / self._enc_calls
            logger.info("[H3VAE-PyOpt] encode #%d: %.1f ms (running mean %.1f ms)",
                        self._enc_calls, dt, mean)

    # ------------------------------------------------------------------
    # comfy first_stage_model interface
    # ------------------------------------------------------------------

    def decode_output_shape(self, input_shape):
        b, c, t, h, w = input_shape
        frames = self._frames_from_tokens(t)
        return (b, self.core.decoder.out_channels
                if hasattr(self.core.decoder, "out_channels") else 3,
                frames, h * 16, w * 16)

    def decode(self, z, output_buffer=None, **kwargs):
        t0 = time.monotonic()
        dev = self._device()
        if self.int8_decode and z.dtype != torch.float16:
            raise RuntimeError(
                "int8_decode requires FP16 latent input; use the FP16 VAE dtype"
            )
        z = z.to(device=dev)
        zn = z * self.latents_std + self.latents_mean

        t = zn.shape[2]
        if t == 1:
            dec = self.core.decode(zn)[:, :, -1:, :, :]
        else:
            if t < 7:
                # too few tokens for one temporal chunk: pad an extra chunk
                pad = 7 - t
                zn = torch.cat([zn, zn[:, :, -1:, :, :].repeat(1, 1, pad, 1, 1)], dim=2)
            dec = self.core.decode_base(zn, frame_num=self._frames_from_tokens(t))

        # raw decoder output -> float32 pixels in [0,1] (matches core _finalize_pixels)
        pixels = dec.float()
        pixels = pixels * self._img_std + self._img_mean
        pixels = pixels.clamp_(0.0, 1.0)

        if output_buffer is not None:
            output_buffer.copy_(pixels)
            out = output_buffer
        else:
            out = pixels
        self._log_call("decode", t0)
        return out

    def _normalize_pixels(self, x):
        # comfy process_input gives [-1,1]; the model wants ImageNet-normalized [0,1]
        x01 = (x.float() + 1.0) * 0.5
        normed = (x01 - self._img_mean.to(x01.dtype)) / self._img_std.to(x01.dtype)
        return normed.to(self.latents_mean.dtype)

    def _encode_moments(self, normed):
        t = normed.shape[2]
        if t == 1:
            moments = self.suffix(self.prefix(
                normed.contiguous(memory_format=torch.channels_last_3d)))
            return moments[:, :, -1:, :, :]
        return self._encode_staged(
            normed, self.prefix, self.suffix,
            clip_length=self.core.clip_length,
            token_drop=self.core.token_drop,
            batch_size=self.encoder_staged_batch, cut=2)

    @staticmethod
    def _split_encoder_tiles(length, tile_size, overlap_min, ratio):
        """Match the reference VAE split_tiles without mutating shared core state."""
        if tile_size <= overlap_min:
            raise ValueError("encoder tile must exceed its minimum overlap")
        if tile_size >= length:
            return [0], [length], []
        count = math.ceil(length / tile_size)
        while True:
            overlaps = [overlap_min] * (count - 1)
            remaining = tile_size * count - sum(overlaps) - length
            if remaining >= 0:
                break
            count += 1
        for i in range(remaining // ratio):
            overlaps[i % (count - 1)] += ratio
        starts = [0]
        for overlap in overlaps:
            starts.append(starts[-1] + tile_size - overlap)
        return starts, [tile_size] * count, overlaps

    @staticmethod
    def _select_encoder_tile(configured, height, width):
        return configured or (672 if max(height, width) <= 672 else 256)

    def _encode_spatial_tiled(self, x, tile_size):
        core = self.core
        split = lambda length: self._split_encoder_tiles(
            length, tile_size, core.tile_overlap_min, core.vae_ratio)
        y_idx, y_len, y_overlap = split(x.shape[-2])
        x_idx, x_len, x_overlap = split(x.shape[-1])
        rows = []
        for y_pos, height in zip(y_idx, y_len):
            row = []
            for x_pos, width in zip(x_idx, x_len):
                tile = x[..., y_pos:y_pos + height, x_pos:x_pos + width]
                row.append(self._encode_moments(self._normalize_pixels(tile)))
            rows.append(row)

        y_overlap = [overlap // core.vae_ratio for overlap in y_overlap]
        x_overlap = [overlap // core.vae_ratio for overlap in x_overlap]
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i:
                    tile = core.blend(rows[i - 1][j], tile, y_overlap[i - 1], dim=-2)
                if j:
                    tile = core.blend(row[j - 1], tile, x_overlap[j - 1], dim=-1)
                if i < len(rows) - 1:
                    tile = tile[..., :-y_overlap[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, :-x_overlap[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def encode(self, x, device=None):
        t0 = time.monotonic()
        dev = self._device()
        if self.int8_encode and x.dtype != torch.float16:
            raise RuntimeError(
                "int8_encode requires FP16 pixel input; use the FP16 VAE dtype"
            )
        x = x.to(device=dev)
        if x.ndim == 4:
            x = x.unsqueeze(2)
        tile_size = self._select_encoder_tile(
            self.encoder_tile_size, x.shape[-2], x.shape[-1])
        if x.shape[-2] > tile_size or x.shape[-1] > tile_size:
            moments = self._encode_spatial_tiled(x, tile_size)
        else:
            moments = self._encode_moments(self._normalize_pixels(x))

        mean = moments.float()[:, :moments.shape[1] // 2]
        latents = (mean - self.latents_mean.float()) / self.latents_std.float()
        self._log_call("encode", t0)
        return latents

    def decode_tiled(self, z, **kwargs):
        return self.decode(z)

    def encode_tiled(self, x, **kwargs):
        return self.encode(x)

    # ------------------------------------------------------------------
    # warmup
    # ------------------------------------------------------------------

    @torch.no_grad()
    def warmup(self, frames: int, width: int, height: int, mode: str = "both"):
        """Trigger compile + cuDNN search for the given video shape.

        The first call on a fresh process pays inductor autotune (cached to
        TORCHINDUCTOR_CACHE_DIR afterwards) and the cuDNN benchmark search
        (per-process); running it here moves that cost out of the first user
        request.
        """
        dev = self._device()
        is_cuda = dev.type == "cuda"
        tokens = self._tokens_from_frames(frames)
        results = {}
        if mode in ("decode", "both"):
            z = torch.randn((1, 24, tokens, height // 16, width // 16),
                            device=dev, dtype=self.latents_mean.dtype)
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.monotonic()
            y = self.decode(z)
            if is_cuda:
                torch.cuda.synchronize()
            results["decode_first_s"] = round(time.monotonic() - t0, 3)
            del y, z
            if is_cuda:
                torch.cuda.empty_cache()
        if mode in ("encode", "both"):
            x = torch.rand((1, 3, frames, height, width), device=dev,
                           dtype=self.latents_mean.dtype) * 2.0 - 1.0
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.monotonic()
            y = self.encode(x)
            if is_cuda:
                torch.cuda.synchronize()
            results["encode_first_s"] = round(time.monotonic() - t0, 3)
            del y, x
            if is_cuda:
                torch.cuda.empty_cache()
        # reset rolling stats so A/B numbers start clean
        self._dec_calls = self._dec_total_ms = 0
        self._enc_calls = self._enc_total_ms = 0
        logger.info("[H3VAE-PyOpt] warmup %s (%d frames, %dx%d): %s",
                    mode, frames, width, height, results)
        return results


def build_comfy_vae(runtime: H3VAEPyOptRuntime, dtype=None):
    """Wrap the runtime in a comfy.sd.VAE configured like the stock H3 branch.

    Replicates the MiniMax-H3 branch of comfy/sd.py VAE.__init__ (latent
    ratios, memory estimators, tiling flags) so the stock VAEDecode /
    VAEEncode / *Tiled nodes drive the runtime unchanged.
    """
    import comfy.model_management as model_management
    import comfy.model_patcher
    import comfy.sd

    class H3VAEPyOptVAE(comfy.sd.VAE):
        def __init__(self, first_stage_model, device=None, dtype=None):
            # NOTE: parent __init__ sniffs a state dict; configure directly instead.
            self.first_stage_model = first_stage_model
            self.latent_channels = 24
            self.latent_dim = 3
            self.output_channels = 3
            # frames 17k+5 <-> latents 5k+2, 16x spatial
            self.upscale_ratio = (lambda a: max(1, (a - 2) // 5 * 17 + 5), 16, 16)
            self.upscale_index_formula = (4, 16, 16)
            self.downscale_ratio = (lambda a: max(1, (a - 5) // 17 * 5 + 2) if a > 1 else 1, 16, 16)
            self.downscale_index_formula = (4, 16, 16)
            self.working_dtypes = (
                [torch.float16]
                if getattr(runtime, "int8_decode", False)
                or getattr(runtime, "int8_encode", False)
                else [torch.float16, torch.float32]
            )
            # the runtime tiles internally; stock tiling fallbacks are no-ops
            self.handles_tiling = True
            # decode already finalizes to [0,1] float32
            self.process_output = lambda image: image
            # encode receives [-1,1] and converts internally
            self.process_input = lambda image: image * 2.0 - 1.0

            # base-class defaults replicated (parent __init__ skipped)
            self.pad_channel_value = None
            self.disable_offload = False
            self.not_video = False
            self.size = None
            self.extra_1d_channel = None
            self.crop_input = True
            self.format_encoded = None

            if device is None:
                device = model_management.vae_device()
            self.device = device
            offload_device = model_management.vae_offload_device()
            if dtype is None:
                dtype = model_management.vae_dtype(self.device, self.working_dtypes)
            if (
                getattr(runtime, "int8_decode", False)
                or getattr(runtime, "int8_encode", False)
            ) and dtype != torch.float16:
                raise RuntimeError(
                    "INT8 VAE mode requires ComfyUI to select an FP16 VAE dtype"
                )
            self.vae_dtype = dtype
            self.output_device = model_management.intermediate_device()

            # memory estimators: unlike the streaming stock implementation the
            # optimized runtime materializes the full decoded video plus an
            # untiled staged encoder, so reserve accordingly.
            def estimate_encode_memory(shape, dtype_):
                b, c, t, h, w = shape
                dsize = model_management.dtype_size(dtype_)
                # staged prefix activations measured ~950 B per pixel of one clip
                act = 1000 * min(t, 17) * h * w
                return int(self.size + (act + 1_100_000_000) * dsize / 2 * 1.15)

            def estimate_decode_memory(shape, dtype_):
                b, c, t, h, w = shape
                dsize = model_management.dtype_size(dtype_)
                frames = self.upscale_ratio[0](t)
                # full fp16 canvas + tile outputs + blending temporaries
                out = 3 * frames * (h * 16) * (w * 16) * 2 * 2.5
                return int(self.size + (out + 1_100_000_000) * dsize / 2 * 1.1)

            self.memory_used_encode = estimate_encode_memory
            self.memory_used_decode = estimate_decode_memory

            self.patcher = comfy.model_patcher.CoreModelPatcher(
                first_stage_model, load_device=self.device, offload_device=offload_device)
            self.model_size()

    return H3VAEPyOptVAE(runtime, dtype=dtype)
