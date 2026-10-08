# Block0 RMS contract and final LayerNorm follow-up — 2026-10-08

The independent block0 RMS kernel reproduces the original FP16 norm and its
public comfy-kitchen QKV path exactly on the captured first tile group. This
removes one numerical uncertainty in the W16 norm1 fusion experiment. The
naturally compiled candidate still fails first-group RGB exactness, so this
round does not establish a new complete-decoder speedup or adopt a runtime
change. Current records remain INT8 encode **8.458108 s**, INT8 decode
**6.508537 s**, and fused FP16 decode **10.923379 s**.

The [preceding W16 controls](decoder_norm1_causal_2026-10-08.md#corrected-acn-result)
produced exact full RGB when other reductions were bound to the original graph.
Natural compilation changed both block0 norm1 and final LayerNorm scheduling.
Those controls did not establish which change caused the RGB difference.

## Independent block0 kernel

Block0 uses ordinary RMSNorm before QKV; the 35 fused residual/norm1 targets
start at block index 1. The new research helper handles contiguous FP16
`[4,1797,2048]` input and `[2048]` gamma. It accumulates squares in FP32,
divides by 2048, adds the original FP32 epsilon, uses CUDA libdevice rsqrt,
multiplies input by inverse RMS and then gamma, and stores a separate FP16
output. The [authored research helper](benchmarks/block0_rms_x2_w16_research_helper.py)
has a fixed X2/R2048/W16/stage1 launch. Public CK activation
quantization and QKV remain in place.

The private integration changes only this first-block norm branch. It reuses
all 144 INT8 weight/scale storages and the existing W16 inter-block candidate;
it does not bind the original compiled producer into the actual candidate path.
Original-kernel shadow buffers are independent and used only for validation.
The helper is an archived research artifact, not an enabled production path;
this round validates the observed first group, not all 49 block0 calls in a
complete candidate video.

| Correctness check | Result |
| --- | --- |
| Original full 124-frame RGB versus retained baseline | Bitwise exact |
| Raw versus observed original first-group input/output | Bitwise exact |
| NaN-filled helper output versus original block0 norm | All 14,721,024 elements finite and bitwise exact |
| Candidate block0 norm versus same-input original shadow | Bitwise exact; inputs unchanged; output owns separate storage |
| Candidate block0 CK input / INT8 Q / FP32 scale / QKV output versus original | All bitwise exact |
| Candidate first-group RGB versus original | 25,053 of 9,633,792 values differ; max absolute delta 0.001953125 |

The last check stops this attempt before a complete candidate decode. There is
no new full-video quality measurement or timing. The first-group difference
must not be reported as a full-video rejection metric. The post-full model
integrity check was not reached. Hooks were restored, and both retained cache
manifests remained exactly unchanged.

## Actual compiled resources

The helper compiles with **16 warps, 28 registers, zero spills and 4 KiB shared
memory**. Its matched saved TTGIR uses the same input/reduction layout as the
original: elements per thread `[1,8]`, threads per warp `[1,32]`, warps per CTA
`[2,8]`, order `[1,0]`, with FP32 addition along axis 1.

The original kernel allocates 8 KiB shared memory. TTGIR/PTX show that the
original widens gamma to FP32 before its layout conversion, whereas the helper
converts gamma while still FP16 and then widens it. That conversion stages
16 versus 8 bytes per thread, explaining the different shared-memory counts.
This resource difference is not a measured performance improvement.

## Final LayerNorm causal control

The live original final LayerNorm selects X1/R2048/W8/stage1; the candidate
selects X1/R1024/W8/stage1. Their function bodies and compiler metadata match
after the two explicitly checked kernel-name substitutions. The candidate
choice is recorded naturally, with a distinct tuner and launcher; its
configuration is not forced to match the original.

The initial block0 attempt alone did not establish the cause. A separately
registered A/P/D control then reproduces both original and candidate tile
hashes using new, verified private cache copies. It captures all five inputs
before the final kernel writes its output in place: H, FFN output O, residual
gate, LayerNorm gamma and beta.

All five A/P inputs are **bitwise identical**. P and D also have identical
inputs and use the **same candidate compiled object**. D replaces only the
final LayerNorm dispatch: it runs the original R2048/W8 kernel on independent
clones and copies only output slot 0 back into the candidate. The natural P
path remains independently compiled; D is explicitly a diagnostic control.

| First-group comparison | Final LayerNorm output | RGB tile output |
| --- | --- | --- |
| A versus natural P | 4,718 of 14,721,024 values differ; max delta 0.0078125 | 25,053 of 9,633,792 values differ; max delta 0.001953125 |
| A versus D | Bitwise exact; zero delta | Bitwise exact; zero delta |

For this reproduced first group, changing only the final LayerNorm dispatch
is sufficient to remove the remaining difference. Its source bodies and
metadata agree under the strict identifier checks; the selected reduction
chunk changes from R2048 to R1024. This establishes a numerical integration
barrier, not a performance bottleneck or a full-video equivalence result.

All 585 named model tensors and 144 INT8 weight/scale storages remain
unchanged after the control. The source caches, earlier report and archive
remain unchanged, generated-source owners are inside their respective private
caches, and hooks are restored. This separate window completes **three tiles,
zero full decodes and zero timed runs**. Its
[redacted causal archive](benchmarks/decoder_norm1_final_causal_2026-10-08.json)
references the block0 archive by SHA.

The next implementation should own the final gated-residual/LayerNorm math
with the original X1/R2048/W8 schedule. It must preserve FP32 residual FMA and
Welford statistics before FP16 output, rather than normalize a residual sum
already rounded to FP16. It should reuse standard Welford helpers as lazy
dependencies, return its own output, and preserve the decoder's padding and
spatial-parallel behavior. Only an independently implemented candidate that
passes full-video quality can proceed to paired complete-decode timing and
runtime/offload validation. The earlier **4.43% local norm1-chain gain remains
unadopted**.

## Evidence and limits

The [redacted block0 archive](benchmarks/decoder_norm1_block0_2026-10-08.json)
preserves the experiment card, candidate preparation, frozen execution plan,
actual CPU fixtures, terminal GPU report and root reviews. Each record keeps
its numeric, boolean and full-SHA leaves; private media, paths, tensors and
generated third-party source are excluded. It references the preceding W16
archive by SHA instead of duplicating earlier attempts.

The experiment uses RTX PRO 5000 / SM120, Python 3.12.14, PyTorch
2.11.0+cu130, CUDA 13, Triton 3.6.0 and comfy-kitchen 0.2.34. Input is 124
frames at 768×1344, tile256/batch4, FP16, FP32 norms and FP16 accumulation
disabled, with the identical saved channels-last latent. Loading and
compilation are diagnostic setup; no timed runs or warmup-based latency
claim exist. This window completes one full original decode, three
first-group tile probes and one standalone helper launch. Execution commit
is `d171d87`. Nsight Compute counters remain unavailable; these resource
observations come from compiled artifacts, not a new Nsight profile.

A clean snapshot of tracked working files plus the four new research/doc
artifacts passes compileall, the required critical Ruff scope and 105
repository tests (91 passed, 14 CUDA tests skipped). The actual private
dispatch fixture also passes 17 CPU checks and five additional rejection
cases. Those CPU checks are separate from the GPU evidence above. No
production runtime implementation, accepted performance number or comparison
chart changes in this round.
