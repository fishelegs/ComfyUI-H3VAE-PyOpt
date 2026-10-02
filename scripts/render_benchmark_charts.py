"""Regenerate README SVG charts from committed numeric evidence.

Requires matplotlib only for documentation generation, not VAE inference.
Use --preview-dir to also render PNG previews for local visual inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "benchmarks"
IMAGES = ROOT / "docs" / "images"
NAVY = "#19344a"
MUTED = "#607789"
TEAL = "#008b80"
BLUE = "#3979b5"
GRAY = "#9eabb8"
ORANGE = "#bf7236"


def chart(
    name,
    title,
    subtitle,
    headline,
    labels,
    values,
    colors,
    notes,
    *,
    xmax,
    preview_dir,
    separator=None,
):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "svg.hashsalt": "h3vae-benchmarks-20261002",
        }
    )
    fig = plt.figure(figsize=(14, 8), facecolor="#f4f7fb")
    fig.text(0.045, 0.93, title, fontsize=24, weight="bold", color=NAVY)
    fig.text(0.045, 0.883, subtitle, fontsize=11.5, color=MUTED)
    fig.text(0.045, 0.815, headline, fontsize=15, weight="bold", color=TEAL)
    ax = fig.add_axes((0.315, 0.29, 0.605, 0.46), facecolor="#f4f7fb")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#dce5ec", linewidth=0.8)
    y = list(range(len(labels)))
    ax.barh(y, values, height=0.57, color=colors, zorder=3)
    ax.set_yticks(y, labels, fontsize=11.5, color=NAVY)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xticks(list(range(0, int(xmax) + 1, 2)))
    ax.tick_params(axis="both", length=0, pad=10, labelcolor=MUTED)
    ax.set_xlabel(
        "Steady-state latency (seconds)  ·  Lower is better",
        fontsize=11,
        color=MUTED,
        labelpad=13,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, value in enumerate(values):
        ax.text(
            value + 0.15,
            i,
            f"{value:.3f} s",
            va="center",
            fontsize=13,
            weight="bold",
            color=NAVY,
        )
    if separator is not None:
        ax.axhline(separator, color="#a7bac9", linestyle=(0, (4, 4)), linewidth=1)
    for i, line in enumerate(notes):
        fig.text(0.045, 0.16 - i * 0.031, line, fontsize=10.5, color=MUTED)
    fig.text(
        0.045,
        0.025,
        "ComfyUI-H3VAE-PyOpt  ·  Measured on RTX PRO 5000 72GB / SM120",
        fontsize=10,
        color=MUTED,
    )
    path = IMAGES / f"{name}.svg"
    fig.savefig(
        path,
        format="svg",
        metadata={"Date": None, "Creator": "ComfyUI-H3VAE-PyOpt benchmark charts"},
    )
    # Keep SVG text selectable, with an accessible title/description on GitHub.
    svg = path.read_text()
    svg = svg.replace(
        "<svg ", '<svg role="img" aria-labelledby="chart-title chart-description" ', 1
    )
    description = (
        f"{subtitle}. {headline}. "
        + "; ".join(
            f"{label}: {value:.6f} seconds" for label, value in zip(labels, values)
        )
        + ". "
        + " ".join(notes)
    )
    svg = svg.replace(
        " <metadata>",
        f' <title id="chart-title">{escape(title)}</title>\n <desc id="chart-description">{escape(description)}</desc>\n <metadata>',
        1,
    )
    path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
    if preview_dir:
        fig.savefig(preview_dir / f"{name}.png", dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    enc = json.loads((EVIDENCE / "encoder_int8_optimized_2026-10-02.json").read_text())
    timing = next(s["timing"] for s in enc["samples"] if "timing" in s)
    secs = {k: v["mean_cuda_event_ms"] / 1000 for k, v in timing.items()}
    lower_old = 100 * (1 - secs["optimized"] / secs["previous"])
    lower_fp16 = 100 * (1 - secs["optimized"] / secs["fp16"])
    chart(
        "h3vae_int8_encode_comparison",
        "INT8 encode: optimized full-video runtime",
        "2026-10-02 paired experiment  |  768 × 1344 × 124  |  Tile 256 / staged batch 4  |  2 warmups / 5 runs",
        f"{lower_old:.2f}% lower latency vs previous INT8  ·  {lower_fp16:.2f}% lower vs FP16",
        [
            "Previous INT8",
            "FP16 paired control",
            "INT8: tile change only",
            "INT8: tile + quantization",
        ],
        [secs[k] for k in ["previous", "fp16", "tile_only", "optimized"]],
        [ORANGE, BLUE, GRAY, TEAL],
        [
            "Same inputs, weights and compiled INT8 graph; dynamic quantization included.",
            "8 videos / 992 frames: optimized and previous INT8 latents + RGB are bitwise equal.",
            "Source PSNR: FP16 35.110 dB; old/new INT8 34.486 dB. Existing quantization loss remains.",
            "Encoder only. decode_fusions=false; do not add this to the fused decoder time.",
        ],
        xmax=14,
        preview_dir=args.preview_dir,
    )

    followup = json.loads((EVIDENCE / "int8_nsys_pipeline_2026-10-02.json").read_text())
    paired = followup["encode_timing"]["means_seconds"]
    chart(
        "h3vae_int8_encode_pipeline",
        "INT8 encode: Nsight-guided pipeline optimization",
        "2026-10-02 follow-up  |  768 × 1344 × 124  |  Tile 256 / staged batch 4  |  6 runs per variant",
        f"{followup['encode_timing']['latency_reduction_percent']:.2f}% lower full-encode latency vs optimized INT8 baseline",
        ["INT8: tile + quantization", "INT8: four-stage pipeline"],
        [paired["baseline"], paired["candidate"]],
        [BLUE, TEAL],
        [
            "Same-process ABBA / BAAB / ABBA blocks; dynamic quantization included. Background GPU work retained.",
            "8 videos / 992 frames: bitwise-equal latents. One full video also rechecked for bitwise-equal RGB.",
            "New schedule auto-selected only on SM120. Existing INT8 quantization loss remains.",
            "Encoder only; no new decoder change. Separate experiment from the earlier FP16 comparison.",
        ],
        xmax=12,
        preview_dir=args.preview_dir,
    )

    producer = json.loads((EVIDENCE / "int8_norm_producer_2026-10-02.json").read_text())
    paired = producer["encode_timing"]["means_seconds"]
    chart(
        "h3vae_int8_encode_norm_producer",
        "INT8 encode: reuse norm output absmax",
        "2026-10-02 paired experiment  |  768 × 1344 × 124  |  Tile 256 / staged batch 4  |  6 runs per variant",
        f"{producer['encode_timing']['latency_reduction_percent']:.2f}% lower full-encode latency vs four-stage pipeline baseline",
        ["INT8: four-stage pipeline", "INT8: pipeline + norm absmax"],
        [paired["baseline"], paired["candidate"]],
        [BLUE, TEAL],
        [
            "Same-process ABBA / BAAB / ABBA; same weights, layouts and compile options; dynamic quantization included.",
            "8 videos / 992 frames: bitwise-equal latents; one 124-frame video's RGB also rechecked exactly.",
            "Measured peak allocated memory increases by 0.60 MiB. Auto-enabled on SM120 only.",
            "Existing INT8 quality loss remains. Decoder GEMM candidates did not provide a stable improvement.",
        ],
        xmax=12,
        preview_dir=args.preview_dir,
    )

    recompute = json.loads((EVIDENCE / "int8_norm_recompute_2026-10-02.json").read_text())
    paired = recompute["encode_timing"]["means_seconds"]
    chart(
        "h3vae_int8_encode_norm_recompute",
        "INT8 encode: skip the FP16 activation scratch",
        "2026-10-02 paired experiment  |  768 × 1344 × 124  |  Tile 256 / staged batch 4  |  6 runs per variant",
        f"{recompute['encode_timing']['latency_reduction_percent']:.2f}% lower full-encode latency vs norm/absmax fusion baseline",
        ["INT8: norm + absmax", "INT8: recompute + direct INT8"],
        [paired["baseline"], paired["candidate"]],
        [BLUE, TEAL],
        [
            "Same compiled graph, weights and input layout; ABBA / BAAB / ABBA; dynamic quantization included.",
            "8 videos / 992 frames: bitwise-equal latents; one complete 124-frame video's RGB also rechecked exactly.",
            f"Peak allocated memory is {recompute['encode_timing']['peak_saved_bytes'] / 2**20:.2f} MiB lower in this comparison.",
            "Auto-enabled on SM120 only. Existing INT8 quality loss remains. Decoder implementation unchanged.",
        ],
        xmax=12,
        preview_dir=args.preview_dir,
    )

    rows = list(csv.DictReader((EVIDENCE / "decode_comparison_2026-09-24.csv").open()))
    csv_values = {r["Implementation"]: float(r["Decode_seconds"]) for r in rows}
    fusion = json.loads((EVIDENCE / "decode_fusions_2026-09-24.json").read_text())
    latest = {
        r["profile"]: r["mean_ms"] / 1000
        for r in fusion["timing"]
        if r["shape_hwt"] == [768, 1344, 124]
    }
    chart(
        "h3vae_fp16_decode_comparison",
        "FP16 decode: historical references + fusion profile",
        "768 × 1344 × 124  |  Decoder tile 256  |  Independent experiments; dates and configurations differ",
        f"Latest opt-in FP16 fusion: {latest['fp16_fusions']:.3f} s  ·  decode_fusions=true / batch 8",
        [
            "ComfyUI default · Sep 18",
            "ComfyUI --fast · Sep 18",
            "TensorRT rebuilt · Sep 18",
            "PyOpt v0.2.0 · Sep 22",
            "PyOpt FP16 fusion · Sep 24",
        ],
        [
            csv_values[k]
            for k in [
                "ComfyUI 387f98a default",
                "ComfyUI 387f98a fast",
                "TRT locally rebuilt same-weight engine",
                "PyOpt FP16 v0.2.0",
            ]
        ]
        + [latest["fp16_fusions"]],
        [GRAY, GRAY, GRAY, BLUE, TEAL],
        [
            "ComfyUI --fast uses FP16 accumulation; TensorRT uses a different software stack.",
            "Latest fusion: 2 warmups / 5 runs. Historical controls were not remeasured in that experiment.",
            "FP16 fusion source PSNR: 35.110 dB across 992 frames. FP16 encoder retained.",
            "No cross-experiment speedup ratio or combined encode/decode claim. BF16 is not validated.",
        ],
        xmax=17,
        preview_dir=args.preview_dir,
    )

    history = json.loads((EVIDENCE / "int8_2026-09-22_timing.json").read_text())
    dec = history["timing"]["decode"]["timing"]
    fp16 = dec["default"]["mean_cuda_event_ms"] / 1000
    int8 = dec["d_only"]["mean_cuda_event_ms"] / 1000
    chart(
        "h3vae_int8_decode_comparison",
        "INT8 decode: release baseline + fusion profiles",
        "768 × 1344 × 124  |  Decoder tile 256  |  FP16 encoder retained in every row",
        f"Latest INT8 fusion: {latest['int8_fusions']:.3f} s  ·  144 quantized linears / batch 4",
        [
            "Sep 22 · FP16 paired control",
            "Sep 22 · INT8 / 72 linears / B2",
            "Sep 24 · FP16 fusion / B8",
            "Sep 24 · INT8 fusion / 144 / B4",
        ],
        [fp16, int8, latest["fp16_fusions"], latest["int8_fusions"]],
        [BLUE, GRAY, BLUE, TEAL],
        [
            f"Sep 22: paired FP16 → INT8 decode latency −{100 * (1 - int8 / fp16):.2f}%; 2 warmups / 3 runs.",
            "Sep 24: separate profile experiments, 2 warmups / 5 runs; not a four-way paired comparison.",
            "INT8 fusion source PSNR: 34.940 dB vs FP16 fusion 35.110 dB on 992 frames.",
            "Fusion requires int8_encode=false. Do not combine this with the 10.121 s INT8 encoder.",
        ],
        xmax=14,
        preview_dir=args.preview_dir,
        separator=1.5,
    )


if __name__ == "__main__":
    main()
