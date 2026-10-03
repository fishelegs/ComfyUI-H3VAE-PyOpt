# INT8 decode：融合 FFN-up GEMM 与 SwiGLU（2026-10-03）

完整视频 decode 的正式配对结果为 **6.640226 → 6.508537 s（耗时降低 1.98%）**。
独立宽度复测为 **6.635767 → 6.505977 s（降低 1.96%）**，两轮峰值 allocated 均不变。
新实现仅在 SM120 的 `int8_decode=true, decode_fusions=true` 模式自动选择。
FP16 默认值、INT8 量化范围及 encoder 保持不变。

## 优化机制

原 FFN-up 将 INT8 GEMM 的 16,384 列结果写成 FP16，再由 SwiGLU/量化 kernel
读回。新 Triton GEMM 在 epilogue 内成对处理 gate/up，保留原来的两次 FP16 舍入，
直接写出 8,192 列 SwiGLU 激活，然后执行相同的逐行量化。两条链路都是两个 kernel，
减少的是中间张量的写入和读回量，不是 kernel 启动次数。

- 保留 INT8×INT8→INT32；K=2048 时全范围累加上界为 33,554,432，小于 INT32 上限。
- 保留 FP32 activation/weight scales、comfy-kitchen 0.2.34 的 FMA bias、FP16 linear
  舍入、FP16 SwiGLU 舍入及量化分母的 FP16 舍入，包括零/下溢处理。
- 使用 128×64×64 tile、4 warps、4 stages；不缓存另一份权重，不改 checkpoint。
- FFN-down 继续使用已验证的 comfy-kitchen 0.2.34 预量化入口；缺失或版本错误明确失败。
- 其他受支持 GPU 保留原来的 INT8 融合实现；CPU offload 时关闭 SM120 分支，CUDA reload
  后按实际设备重新选择。未扩大可用平台范围。
- 对 M>262144 明确拒绝，避免 INT32 输出地址溢出；实际已测 M=7188 不接近此边界。

真实输入的局部 FFN-up＋SwiGLU＋量化回放为 **1.276208 → 1.188275 ms（−6.89%）**，
20 次配对均胜出，q 和 FP32 scale 精确一致。局部收益仅用于筛选，首页采用下面的完整阶段结果。

## 完整阶段对照

| 输入 H×W×帧 | 原 FFN 基线 | 新 FFN | 耗时降低 | 峰值 allocated |
| --- | ---: | ---: | ---: | --- |
| 768×1344×124，各 6 次 | 6.640226 s | **6.508537 s** | **1.98%** | 均为 5,119,705,088 bytes |
| 768×1376×124，各 2 次，独立进程 | 6.635767 s | **6.505977 s** | **1.96%** | 均为 5,176,293,376 bytes |

基线来自 `dbd49e0` 的两条原 registered custom ops；候选使用新的生产模块。
为隔离 FFN 改动，两者在同一新 opaque op 内切换，共用 compiled graph、权重、输入、
layout、tile 和归一化调度。旧生产 decoder 另做完整 RGB 数值锚点，不用重编译后的时间差
代替这个配对实验。历史 **6.667 s** 是 2026-09-24 的独立成绩，不用于计算本轮 1.98%。

测量合同：

- 同机历史硬件身份 RTX PRO 5000 72GB；driver580.82.07 报 NVIDIA Graphics Device、SM120。
- Linux；Python 3.12.14；PyTorch 2.11.0+cu130 / CUDA13.0；Triton3.6.0；comfy-kitchen0.2.34。
- decoder tile256 / batch4，encoder tile256 / staged4；相同 FP16 encode latent。
  `int8_encode=false, int8_decode=true, decode_fusions=true`；SDPA auto、FP32 norm 开启，
  FP16 accumulation 关闭。cuDNN benchmark=true / limit5。
- `max-autotune-no-cudagraphs`，各预热 2 次；主对照 ABBA / BAAB / ABBA，各 6 次；
  独立宽度为一块 ABBA，各 2 次。seed20261002。
- CUDA Event 计时包括完整 `runtime.decode` 的动态量化、分配、拼接和 RGB 输出处理；
  排除 FP16 encode、加载、首次编译、媒体 I/O 和数值检查。
- 正式计时之前一次性绑定相同源码、元数据及实参合同的归一化 autotuner，**计时期间没有
  run 观测/路由钩子**。两变体仍保留相同的目标调用计数检查。
- 每次 decode 覆盖 36 组独立 FFN 权重、1764 次新 opaque 调用；旧生产锚点的新调用数为零。
- 其他 GPU 任务保持运行，本轮实验串行；三块主对照加一块独立宽度并不等于 ACA 建议的
  六块/三 seed 完整工作流确认。峰值包含常驻模型及 compiled graph 缓冲区，不是部署显存。

## 归一化差异的定位与数值范围

早期原型改写已编译类时曾复用旧图，候选调用数为零，该轮计时作废。改成独立图后，
即使 FFN 仍调用原实现，RGB 也偶发不同。逐层检查将首个差异定位到 norm1 输出，发生在
新 FFN 之前；norm 权重、epsilon 和生成的归一化源码相同。

本轮同一进程记录到了不同的实际 launch 配置：norm1 `_4` 的旧图为
`XBLOCK=2, R0_BLOCK=512, warps=16`，新图为 `1, 2048, 8`；`_8` 也选择了不同
XBLOCK。这个尚未启用新 FFN 的基线 RGB 已出现 RMSE 0.00118045、max_abs 0.120398，
因此没有继续计时，也没有放宽数值门槛。

固定相同归一化 autotuner 后，**8 段真实视频、992 帧的旧生产 RGB、原 FFN 基线 RGB
和新 FFN RGB 全部逐位一致且有限**。原图中的 shadow 检查也验证了 36 组实际权重的
q/scale 精确一致。最终不带计时钩子的主视频与独立宽度再次通过完整 RGB 精确检查。

这一控制只存在于 benchmark。生产代码没有替换 norm 算法或锁定 Inductor 的调优结果。
因此结论是“在相同归一化调度下，新 FFN 没有增加已测输入的误差”，**不承诺独立重新
编译后的输出逐位一致，也不表示 INT8 相对 FP16 无损**。旧有 INT8 画质取舍继续适用。

## Nsight 与集成验证

两次独立的预热后 profile，完整 decode NVTX 为 **6.639774 / 6.499606 s**，
均有 **25,426 个 kernel**。新路径明确出现 1764 次 `up_swiglu` 和 1764 次
`quantize_act`；旧 FFN-up CUTLASS＋SwiGLU/量化链合计约 **2.703 s**，新链约
**2.542 s**。新 GEMM epilogue 本身稍慢，但后续量化的读取工作明显减少，整条链净收益为正。
这与减少中间张量读写的实现机制一致；完整阶段 **1.98%** 仍以同进程配对为准。

新 profile 的主要 kernel 时间为：FFN-up/SwiGLU **2.269 s**、其余 CUTLASS GEMM
**2.108 s**、FlashAttention **1.171 s**、FFN 激活量化 **0.273 s**。
本进程未记录 GPU 活动的间隙约 **77 ms**；剩余预算主要集中在 GEMM 与 attention。

不能把 Nsight 的 `Kernel2` 短名都视为 INT8 FFN：按完整符号与 launch 几何拆分后，
少掉的 1813 次中，1764 次才是 FFN-up；另外 49 次是 tile 前的 FP16 卷积在两次
独立 profile 中选择了不同 kernel 名称，合计仅约 0.13 ms，不是额外的 FFN 收益。

没有可用的 NCU 硬件计数器；此前探测返回 `ERR_NVGPUCTRPERM`，未修改驱动权限。
以上时间线用于核对调用和时间归因，不据此声称已测到实际带宽、occupancy 或 warp stall。

- 3 项 CUDA 测试通过：覆盖 M1/17/129 尾块、可选 bias、零/下溢，以及完整新 custom op
  对原 registered ops 的等价性。
- 干净公开源码副本的 CPU 检查为 **91 项通过、14 项 CUDA 测试跳过**；compileall、关键
  Ruff 检查通过。包含大 M 地址边界及 benchmark 调度绑定/恢复保护。
- ComfyUI SDK `c1716a45`，17 帧 256×256 FP32 IMAGE 输入、实际 encode→decode 接线通过；
  144 个 INT8 leaf 的 dtype/device、FP32 scales 及 compiled 路径均保持正确。
- 36 个 FFN 融合选择在 CUDA→CPU→CUDA 中为 **36→0→36**；wrapper/direct 与 offload
  前后输出的 RMSE/max_abs 均为 0。该随机输入 smoke 只证明接口与迁移，不替代视频质量测试。

大 M 拒绝保护是在计时后补充的 host guard，Triton kernel 函数体未变化；原测量源码指纹、
最终模块指纹与 guard patch 指纹均在 JSON 中区分记录。边界测试在 CPU meta tensor 上执行，
无需分配数 GiB 张量。

## 后续空间

先保留这项已通过完整阶段验证的改动。encoder 地址计算/BK32 和 FP16 CTA 分组的其他
候选未达到采用标准，详见[本轮未采用的实验](encoder_fp16_followup_2026-10-03.md)。
新的候选应继续围绕 Nsight 中实际剩余的 GEMM、attention 与 norm 时间筛选，并保留完整
阶段配对和数值检查；局部 kernel 加速不能直接换算成完整 decode 加速。

## 复现

```bash
python bench_decoder_int8_ffn.py \
  --video /path/to/primary.mp4 --videos-dir /path/to/eight-videos \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" --weights "$H3_VAE_WEIGHTS_PATH" \
  --reuse-reference-reductions --warmup 2 --blocks 3 \
  --output results/decoder_int8_ffn.json

H3VAE_TEST_CUDA=1 python -m unittest discover -s tests -p 'test_int8_ffn_up_fused_gpu.py'
```

不传 `--reuse-reference-reductions` 时，基准记录独立调优的实际配置，并严格检查旧生产
RGB；若 norm 调度导致基线不同会在计时前失败。该开关控制实验条件，不是 Loader 参数。

[原始计时、配置、失败定位与验证证据](benchmarks/decoder_int8_ffn_2026-10-03.json) ·
[最初的融合配置与画质](decode_fusions.md) · [INT8 合同](experimental_int8.md)
