# INT8 encode：跳过因果 padding 的零卷积（2026-10-02）

正式实现的完整视频 encode 从 **8.682963 s 降至 8.458108 s（耗时降低 2.59%）**。
基线为 `b5f68c1` 的重算 norm 路径；本轮只减少已知为零的卷积乘积，不改量化、
INT32 累加、FP32 scale 或 FP16 舍入。默认 FP16 模式不变。

## 优化机制

现有 norm/SiLU/padding producer 保证量化输入的前两个时间平面为零。
3×3×3 因果卷积计算第 0、1 个输出平面时，原实现仍加载这些零并执行 Tensor Core 乘积。

现在仅在 producer 明确提供这一保证，且每个 M block 都位于同一输出时间平面时，
从第一个非零 temporal tap 开始归约。D=17 时可省去约 5.88% 的卷积归约迭代；
这是算术工作量，完整 encode 收益以实际 A/B 为准。

- 仅现有 SM120 自动 norm 融合路径开启。
- 要求 kernel3³、stride1、输出空间平面可整除 M block，且每个 temporal tap 的 K 范围对齐。
- 普通卷积输入默认关闭；不满足几何条件时继续执行完整 INT8 归约。
- 权重、scale、padding 和输出 epilogue 保持原合同，没有额外 tensor 缓存。

## 完整阶段对照

| 输入 H×W×帧 | 基线 | 当前实现 | 耗时降低 | 峰值 allocated |
| --- | ---: | ---: | ---: | --- |
| 768×1344×124，各 6 次 | 8.682963 s | **8.458108 s** | **2.59%** | 均为 7,241,622,016 bytes |
| 768×1376×124，各 2 次，独立进程 | 8.680698 s | **8.457235 s** | **2.57%** | 均为 7,261,126,144 bytes |

峰值包含保留的验证输出，不是独立模型显存。两种输入分别做同进程配对，不把跨进程
绝对时间相减来推导提升。

测量合同：

- RTX PRO 5000 72GB 的本机历史硬件身份；驱动显示 NVIDIA Graphics Device / SM120，约 71.1 GiB。
- Linux；Python 3.12.14；PyTorch 2.11.0+cu130 / CUDA 13.0；Triton 3.6.0；driver580.82.07。
- 相同 FP16 视频输入、layout、权重；encoder tile256 / staged batch4；decoder tile256 / batch2。
  `int8_encode=true`、`int8_decode=false`、`decode_fusions=false`。
- **同一 compiled graph**，只在 opaque 卷积内部切换 `causal_prefix_zero=false/true`。
  正式代码重新测量，不将隔离原型的数字当作正式模块成绩。
- 初次正确性校验后各预热 2 次；主对照 ABBA / BAAB / ABBA，各 6 次 CUDA Event 计时；
  独立宽度各 2 次。seed20261002；cuDNN benchmark=true / limit5；不使用 CUDA graphs。
- 包含动态量化、分配、拼接及输出处理；排除模型加载、首次编译、媒体 I/O 和质量计算。
- 其他 GPU 任务保持运行，所有本轮 GPU 实验串行。三块主对照加一块独立宽度复测
  尚不等于 ACA 建议的六块/三 seed 完整工作流确认，不宣称整条视频生成流程的提升。
- 权重 SHA256：`7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522`。
  正式源码指纹、原始采样、环境和每种 shape 的调用数均保留在 JSON。

## Nsight 复核

预热后的完整 encode NVTX 为 **8.457937 s**，21610 个 kernel 合计 **8.381035 s**，
本进程未记录 GPU 活动的空隙约 67 ms。1792 次 INT8 卷积合计约 **3.812 s**；
上一轮独立 profile 中同类卷积约 4.055 s。kernel 数量相同，节省来自减少卷积内部的零计算。
独立 profile 用于归因，正式 2.59% 来自同进程配对。

资源记录仍为 254 registers/thread、49,152 bytes shared memory；局部回放没有报告 spill。
没有可用的 NCU 硬件计数器，之前探测返回 `ERR_NVGPUCTRPERM`，本轮未修改驱动权限。
不从上述时间线推断实际带宽、occupancy 或 warp stall。

## 正确性与集成

- 正式路径 8 段视频、共 992 帧的 latent 全部逐位一致；另对一段完整 124 帧视频复查 RGB，也逐位一致。
- 8 组真实权重的卷积输出逐位一致，实际 producer 的前两帧确认为零；每次 encode 命中 1792 次目标卷积。
- 11 个 GPU 测试通过：覆盖 D1/D2/D17、bias、通道尾块、非对齐空间、stride/kernel 变化、
  普通非零前缀，以及 compiled producer。CPU-only 模式在干净源码副本中为 82 项通过、11 项 CUDA 测试跳过。
- compileall、关键 Ruff 检查通过。
- ComfyUI SDK `c1716a45` 的 17 帧 FP32 IMAGE wrapper、实际 encode→decode 接线、CPU offload/CUDA reload
  通过；wrapper/direct 和 offload 前后的 RMSE、max_abs 均为 0。随机输入 smoke 只证明接口兼容。

这些是相对已有 INT8 实现的等价性检查，原有 INT8 相对 FP16 的画质损失仍然存在。

## 其他候选与剩余空间

沿用本机 ACA 0.3.1 的 Nsight Systems / Triton 方法，先查旧实验，再做有限配置回放，
数值通过后进入完整阶段对照。

二维广播 norm producer 的 q 和 FP32 scale 精确一致，但局部仅约 2%–4% 改善，
对应完整阶段预算较小，本轮不采用。

INT8 decoder 的真实输入回放中，融合 GEMM＋SwiGLU、去掉冗余 partial absmax 后，
完整局部链路从 **1.276208 ms 降至 1.188275 ms（−6.89%）**，q 和 FP32 scale 精确相等。
但本轮未接入生产 decoder：

- 第一轮完整 A/B 的候选调用数为零，修改已编译类后重新 compile 仍复用了旧图；该轮计时已明确作废。
- 改为独立 block 类型并清理本进程 Dynamo 缓存后，确实覆盖 36 组权重、每次 1764 次调用，
  但新编译图的基线 RGB 未能与原生产路径一致（max_abs 0.123840）；在候选计时前停止。
- 目前只能报告算子回放信号，不能宣称完整 decode 已加速。需先定位重组调用后出现的数值差异，
  再做逐层 q/scale 及完整视频验证；没有放宽门槛来采用候选。

默认 FP16 decoder 的像素 finalizer 也做了独立检查：49 组布局、dtype、输出缓冲区及
全部有限 FP16 数值检查通过。768×1344×124 的局部输出处理，无输出缓冲区时
**12.442 → 2.552 ms**，有连续 FP32 输出缓冲区时 **15.707 → 2.555 ms**。
本轮不修改默认路径：仅节省约 10–13 ms，尚未做完整 decode 对照；当前最快的融合
FP16/INT8 decoder 已经使用该 finalizer。局部少分配中间张量也不等于完整 runtime
峰值会下降同样大小。

上述 decoder 实验与 INT8 encoder 使用不同配置，不能把时间相加成一个已测总耗时。

当前 encode 的 INT8 卷积仍约占 kernel 时间 45.5%，是主要剩余预算；FP16 卷积及
norm producer 也仍有开销。后续可继续检查卷积地址计算、数据复用及 residual/producer 边界，
但现有时间线不能证明某个具体硬件瓶颈或保证新候选会更快。

## 复现

```bash
python bench_encoder_int8_causal.py \
  --video /path/to/00015.mp4 --videos-dir /path/to/eight-videos \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" --weights "$H3_VAE_WEIGHTS_PATH" \
  --warmup 2 --blocks 3 --check-rgb \
  --output results/int8_causal_zero.json

H3VAE_TEST_CUDA=1 python -m unittest discover -s tests -p 'test_encoder_int8_gpu.py'
H3VAE_TEST_CUDA=1 python -m unittest discover -s tests -p 'test_encoder_int8_norm.py'
```

[完整数值证据](benchmarks/int8_causal_zero_2026-10-02.json) ·
[上一轮重算 producer](int8_norm_recompute_2026-10-02.md) ·
[INT8 精度取舍](experimental_int8.md)
