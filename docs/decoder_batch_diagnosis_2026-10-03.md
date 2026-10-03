# INT8 decoder batch divergence and norm/quant budget — 2026-10-03

**The first propagated batch-4/batch-8 discrepancy is now localized to the
second transformer's residual + norm1 → QKV boundary.** Reconstructing the
recorded normalization schedules reproduces the captured tensors exactly.
Feeding the batch-4 normalization outputs to a batch-8 QKV call restores exact
quantized values, scales and QKV output. This identifies a numerical obstacle
to larger batches; it is not a new speedup or a repair of the full decoder.

No runtime change was adopted. The measured INT8 configuration stays at batch 4,
and the records remain **8.458 s INT8 encode, 6.509 s INT8 decode and 10.923 s
fused FP16 decode**. This follows the
[batch-8 screen](decoder_gemm_followup_2026-10-03.md#batch-4-versus-batch-8-configuration-screen).

## Scope and controls

The diagnostic used repository `0c51963785e930432cabba49d813521744b5b447`
with the production kernels from `7bd9809`: FP16 decoder inputs,
`int8_decode=true`, `decode_fusions=true`, tile256 and the same cached FP16
latent/checkpoint as the earlier screen. It ran on the same RTX PRO 5000 72GB
SM120 host, Linux, Python 3.12.14, PyTorch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0
and comfy-kitchen 0.2.34. FP32 normalization was enabled and FP16 accumulation
was disabled; diagnostic seed was 20261003. Source, weight and tensor hashes
are archived below. The runtime's `core.decode` path includes `post_quant_conv`
before the compiled transformer decoder.

The earlier complete screen changed grouping from 49 batch-4 calls to 21
batch-8 calls plus 7 batch-4 calls, retaining tile inventory and order.
Both 124-frame RGB outputs were finite, but their RMSE was **0.00215067815**
and maximum absolute difference **0.158203125**. Batch 8 failed the exactness
gate before timing. This was cached-latent decoding, not a fresh video encode
or a new comparison against a quality reference.

The following captures examine the first temporal collection's first eight
logical spatial tiles: two batch-4 groups `[0..3]`, `[4..7]`, and one batch-8
group `[0..7]`. They do not cover every layer input in the complete video.
There are no timed samples in this diagnosis. Model loading, compilation,
capture, hashing and offline replay are diagnostic work, not latency results.

## Phase 1: first numerical difference

Input tiles, post-quant-convolution outputs and layer-0 `block_x` matched
bitwise on all eight tiles. Layer-0 norm1 outputs, captured at the QKV input,
differed in **734 / 29,442,048 FP16 values**; RMSE was about **1.54236e-6** and
maximum difference **0.001953125**. QKV outputs, attention outputs, FFN
residuals and FFN outputs all remained bitwise equal.

Three observer-free group replays matched the corresponding captured raw tile
outputs bitwise and were finite. Complete captures also retained the expected
49/28 decoder-call and 1,764/1,008 FFN-call counts. These controls validate the
observed groups; they are not another observer-free full-video decode.

The final raw tile tensors for those eight tiles differed in
42,425,289 / 44,040,192 values, with RMSE **0.0111150391** and maximum difference
**0.6744384766**. These are pre-finalization tensors, not normalized `[0,1]`
RGB quality metrics. The layer-0 norm discrepancy does not propagate through
its QKV operation in these captures.

## Phase 2: first propagated difference

A second capture checked six boundaries across all 36 transformer blocks and
eight tiles in both groupings: **3,456 finite tensor/hash records**. All three
observer-free group anchors matched both their hooked outputs and Phase 1.
Layers are identified by the actual QKV/FFN weight pointers, rather than by
kernel names alone.

At **layer 1, the second block**, the preceding FFN outputs and physical
`block_x` still matched on all eight tiles. The norm1 output differed on all
eight. The first differing operator output was **QKV, on tiles 3 and 7**;
its downstream attention and FFN boundaries first differed on the same tiles.
Later blocks continue to diverge.

The archived layer-1 generated norm bodies are identical after replacing only
the row-count parameter. Their selected schedules differ:

| Grouping | Rows | XBLOCK | R0_BLOCK | Reduction chunks | Warps | Stages |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Batch 4 | 7,188 | 2 | 2,048 | 1 | 8 | 1 |
| Batch 8 | 14,376 | 8 | 128 | 16 | 2 | 1 |

This is a fused **residual + norm1** operation. It accumulates RMS statistics
from the **unrounded residual**, stores the residual to FP16, then reloads the
FP16 value for normalization. Therefore ordinary RMSNorm of the physical
`block_x` tensor does not reproduce this compiled operation's full arithmetic
contract. A fusion must retain both the statistics and the rounding sequence.

## Phase 3: exact reconstruction and QKV counterfactual

The final diagnostic replays each archived norm body at its own original
shape and launch configuration, using the actual captured layer-0 FFN
residual/output and shared model weights. FP fusion is enabled, as in the
recorded computation. All three groups reproduce the Phase-2 layer-1
`block_x`, norm1 output and QKV output **exactly by per-tile SHA-256, shape,
dtype and stride**, and all tensors are finite. No production decoder
forward or production graph replacement is used in this offline step.

The QKV replay calls the comfy-kitchen public rowwise quantizer and public
`int8_linear`. Both are observed calling the same CUDA C quantization entry
exactly once, on the same stream, with stochastic rounding disabled and seed
zero. Feeding the public quantizer's INT8 values and FP32 scales to the existing
prequantized GEMM reconstructs the actual QKV output bitwise. This ties the
reported quantization values to the tested QKV path.

| Boundary | Layer 0 differing values | Layer 1 differing values | Layer 1 maximum difference |
| --- | ---: | ---: | ---: |
| FP16 norm1 output / QKV input | 734 / 29,442,048 | 1,511 / 29,442,048 | 0.015625 |
| INT8 quantized activation | 0 / 29,442,048 | 5 / 29,442,048 | 1 |
| FP32 row scale | 0 / 14,376 | 2 / 14,376 | 0.000123038888 |
| FP16 QKV output | 0 / 88,326,144 | 11,655 / 88,326,144 | 0.015625 |

At layer 1, the quantized activation, scale and QKV differences occur only on
tiles 3 and 7. QKV output RMSE is **3.764615e-5**. Layer 0's normalization
differences leave both quantized values and scales unchanged.

For the counterfactual, concatenate the two batch-4 norm outputs and run the
same QKV operation with batch-8 shape. **Quantized values, scales and QKV
output all match the concatenated batch-4 results bitwise.** Thus batch-8 QKV
shape alone does not force the observed discrepancy. Together with exact
norm reconstruction, this localizes the first propagated discrepancy to the
schedule-dependent residual/norm1 output feeding quantization. It does not
prove that every difference in the complete RGB output has this single cause,
nor that replacing this boundary would make the entire decoder exact.

### Rejected attempts retained in the archive

The first live targeted replay matched its same-process observer-free output
but failed the earlier cross-process raw-tile anchor. It stopped before the
quantization counterfactual; that record does not identify a responsible kernel.
A second live attempt passed its batch-4 anchors, but batch 8 selected
`R0_BLOCK=64` instead of the archived `128` for the same layer-1 source.
Its local boundary/configuration gate failed before the causal check. This
shows why source hashes alone are insufficient to establish a repeated
compiled computation.

The first offline launch then failed during runtime construction because
`decode_fusions=true` requires `compile_decoder=true`. The corrected launch
uses the lazy compile wrapper only to obtain the installed modules and weights;
it never invokes decoder forward. No norm configuration was tuned to make
reconstruction pass. All rejected records are preserved separately from the
successful diagnostic, including their original status fields.

## Remaining profile budget and next candidate

A CPU-only reclassification of an existing Nsight Systems SQLite capture
separates norm1/QKV work from attention-output and FFN-side quantization.
Immediate predecessor/successor events on the same stream identify the QKV
quantizers; they are not grouped by a shared short kernel name alone.

| Historical fused batch-4 segment | Calls | Summed kernel activity |
| --- | ---: | ---: |
| Transformer norm1 | 1,764 | 127.511 ms |
| QKV activation rowwise quantization | 1,764 | 64.717 ms |
| QKV INT8 GEMM | 1,764 | 832.641 ms |
| All three segments | | 1,024.869 ms |

Norm1 plus quantization totals **192.229 ms**. This is an extremely optimistic
whole-segment budget, not expected removable latency. A fusion still needs
normalization, reduction and quantization work, and the QKV GEMM remains.
The 1,024.869 ms three-segment sum is about 15.8% of that capture's 6.499606 s
NVTX decode interval; it is not a new end-to-end timing record. NCU hardware
counters remain unavailable, so this report makes no measured occupancy,
bandwidth or stall-cause claim.

The profile JSON records Git HEAD `dbd49e0`, but its dirty experimental working
tree already contained the FFN-up fusion later adopted at `7bd9809`:
the SQLite trace contains 1,764 `up_swiglu` and 1,764 `quantize_act` launches.
The shared source fingerprints support alignment of the upstream path, not
identity of generated kernels or launch configurations. See the
[FFN fusion report](decoder_int8_ffn_2026-10-03.md). The companion profile JSON
also preserves a separate baseline capture; those profiles are not paired,
so their difference is not a speedup claim.

The next bounded candidate is **residual + norm1 + QKV activation quantization
fusion at the retained batch-4 shape**. It must first preserve the unrounded
residual statistics, FP16 normalization boundary and the actual INT8 values /
FP32 scales. Only a local complete-chain gain followed by paired full-video
latency and quality validation can justify adoption. Larger batches need the
same numerical control before their reduced call count can become a useful
performance result.

## Evidence

Public artifacts retain numeric results, validity flags, hashes and launch
metadata. Local paths are replaced by aliases; private tensors, checkpoint
contents, media and generated third-party source bodies are not included.
The two larger JSON files use gzip compression. Python can read them with
`json.load(gzip.open(path, "rt"))`.

- [Phase 1 captures and full-output screen](benchmarks/decoder_batch_phase1_2026-10-03.json)
- [Phase 2 layer hashes and launch metadata](benchmarks/decoder_batch_phase2_2026-10-03.json.gz)
- [Phase 3 attempts and successful counterfactual](benchmarks/decoder_batch_phase3_2026-10-03.json.gz)
- [Norm source comparison](benchmarks/decoder_batch_norm_source_review_2026-10-03.json)
- [Historical profile queries and classification](benchmarks/decoder_batch_profile_budget_2026-10-03.json)
- [Artifact checksums and archive metadata](benchmarks/decoder_batch_manifest_2026-10-03.json)

## 中文摘要

本轮定位到了 batch 4 → 8 首次传播的数值差异：**第 2 个 transformer
block 的 residual + norm1 → QKV 边界**。前驱 FFN 输出与物理 `block_x`
一致，但归约配置从 B4 的一次 R=2048 变为 B8 的十六次 R=128。该融合算子先
使用未舍入 residual 计算统计量，再写入 FP16 并回读归一化，不能直接当作普通
FP16 输入的 RMSNorm 替换。

使用捕获的真实前驱张量、归档源码和原始配置离线重建后，三个 tile 组的
`block_x`、norm 输出和 QKV 输出都逐位匹配原始捕获。第 1 个 block 虽有
734 个 norm 值不同，但量化值、scale 和 QKV 输出一致。第 2 个 block 有
1,511 个 norm 值、5 个 INT8 值和 2 个 scale 不同，最终造成 11,655 个 QKV
输出值不同；量化及 QKV 差异集中在 tile 3、7。

**将 B4 的 norm 输出拼成 B8 形状后，量化值、scale 和 QKV 输出全部恢复逐位
一致。** 这表明该处偏差并非由 B8 的 QKV 形状单独造成，而是从受调度影响的
residual/norm1 输出进入量化后传播。这个局部对照尚不能证明完整 RGB 的所有
差异只有这一个原因，也没有完成 batch 8 的数值修复。

此前完整 124 帧 RGB 筛选的 RMSE 为 0.00215067815、最大误差 0.158203125，
所以 batch 8 未进入测速。本轮另外保留了两次因捕获门槛失败而停止的 live
尝试，以及一次离线脚本构造参数错误；这些失败均未被当作有效性能或因果结果。

历史 Nsight 记录中，norm1 与 QKV 量化合计 **192.229 ms**，QKV GEMM 为
832.641 ms。下一候选是保留 batch 4 并融合 residual + norm1 + 量化，但
192.229 ms 是整段耗时预算，实际可节省时间会更少，仍须保持舍入和量化结果，
再做完整链路及视频验证。本轮没有采用生产改动：**INT8 encode 8.458 s、
INT8 decode 6.509 s、FP16 decode 10.923 s** 的记录和图表均保持不变。
