# INT8 VAE：norm/absmax 融合与 decoder GEMM 回放（2026-10-02）

本轮完整 INT8 encode 从 **9.506353 s 降至 9.138146 s（−3.87%）**。
基线是上一轮未发布的四级卷积流水线，而非原始 `4d79fd5`。
新增优化复用 norm/SiLU/padding 写出时的 absmax，避免量化前再扫描整个 FP16 张量。
8 段视频共 992 帧的 latent 逐位一致，另复查一段完整 124 帧视频的 RGB，也逐位一致。
这是相对原 INT8 实现的等价性，已有 INT8 相对 FP16 的画质损失仍然存在。

Decoder 的四类真实 INT8 GEMM 已做调度、权重布局、分块和 cuBLAS 回放，
未找到稳定收益，没有替换生产 decoder。不能将新的 encode 时间与融合 decoder
时间相加：两者仍是不同的配置。

## 测量合同

沿用 ACA 0.3.1 的 Nsight Systems / Triton 优化方法：历史筛选、有限配置回放、
精确数值门槛、完整阶段交错计时，以及另一素材/宽度复核。未启动自动 runner 或代理。

- GPU：本机历史识别为 RTX PRO 5000 72GB；驱动报告 NVIDIA Graphics Device、SM120、约 71.1 GiB。
- ComfyUI wrapper 验证使用 SDK revision `c1716a45`；Nsight Systems 2025.3.2、driver 580.82.07。
- Linux；Python 3.12.14；PyTorch 2.11.0+cu130、CUDA 13.0、Triton 3.6.0、comfy-kitchen 0.2.34。
- 主视频 00015：768×1344×124；另一视频 00007：768×1376×124。
- 相同 FP16 输入、layout、权重；encoder tile256 / staged batch4；decoder tile256 / batch2。
  `int8_encode=true`，`int8_decode=false`，`decode_fusions=false`。
- 两个 prefix 共享不可变参数，使用相同 fullgraph 编译选项；后缀相同。
  基线与候选分别编译，因为 producer 融合改变了 custom-op 边界。
- 每个变体初始校验后预热 2 次，主对照 **ABBA / BAAB / ABBA**，各 6 次 CUDA Event 测量。
  cuDNN benchmark=true / limit5，seed20261002；不启用 CUDA graphs。
- 动态量化、分配和 runtime 输出处理计入耗时；加载、编译、媒体 I/O、质量计算不计入。
- 其他 GPU 任务保持运行，所有本轮 GPU 实验串行持锁；不是独占 GPU 的跨硬件保证。
- 基础提交 `4d79fd5`；具体未发布实现以原始 JSON 中的源码 SHA256 为准。
  权重 SHA256：`7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522`。

## 采纳的优化

先前 Nsight 测得 encode 中 partial absmax 约占 **0.440 s**。
旧路径在 norm/SiLU/padding 写出 FP16 张量后，再完整读取它计算 absmax。
现在同一个 pack kernel 将每 1024 个最终 FP16 元素的绝对值最大值写到一个小型 FP32 数组，
沿用既有两级归约和连续 INT8 量化内核获得完全相同的全局 scale、INT8 值。

保持原来的 norm 统计归约次序、bias 加法舍入、norm/SiLU 两次 FP16 舍入、
时间 causal 零 padding、空间 reflect padding。卷积仍是 INT8×INT8→INT32，
仍用上一轮的 `128x128x64_pipeline4`。没有降低 scale 精度，也没有 FP16 卷积回退。

新增 producer 只在 SM120 的自动 INT8 安装路径启用；其他架构和显式 tile 选择保留
原边界。默认 FP16 不变。静态 INT8 权重和 FP32 scale 的 offload 合同不变。

| 完整 encode 对照 | 原流水线 | 加 norm/absmax 融合 | 耗时降低 |
| --- | ---: | ---: | ---: |
| 768×1344×124，各 6 次 | 9.506353 s | 9.138146 s | 3.87% |
| 768×1376×124，各 2 次 | 9.506903 s | 9.138076 s | 3.88% |

主对照峰值 allocated **7,280,176,640 → 7,280,809,472 bytes**，仅增加
**632,832 bytes（0.60 MiB）**。数字包括保留的校验输出，不是独立模型显存。
最初原型将 FP16 中间结果一直保留到卷积结束，额外增加约 272 MiB；正式实现
在卷积输出分配前释放该 scratch，原型没有被直接采用。

局部 norm＋量化微测约改善 13%–15%，完整 encode 实测为 3.87%；不混用这两个比例。
每次完整编码确实命中 **1792 次 producer / 8 组权重**，没有因绕过计算而提速。

## Nsight 复测归因

预热后以 cudaProfilerStart/Stop 采集一个完整 encode，NVTX 范围末尾同步。
以下基线 profile 来自上一轮流水线版本；不同进程的 profile 只用于归因，
正式 **3.87%** 来自上述同进程交错计时。

| Kernel 类别 | 流水线基线 profile | 加 producer 融合 profile | 调用数 |
| --- | ---: | ---: | ---: |
| Partial absmax | 0.453996 s | 0.006073 s | 均为 1792 |
| Norm pack（含带 bias 版） | 0.826019 s | 0.833413 s | 均为 1792 |
| 连续 INT8 量化 | 0.640566 s | 0.709277 s | 均为 1792 |
| INT8 卷积 | 约 4.037 s | 约 4.034 s | 均为 1792 |

新 profile 的完整 encode NVTX 为 **9.148426 s**，GPU kernel 合计 **9.066802 s**，
本进程未记录 GPU 活动的空隙约 **72 ms**。主要收益确实来自消除大张量 absmax 扫描，
同时保留了量化耗时增加的实测结果；不能把扫描节省的 448 ms 全部当成端到端收益。
没有用时间线比例推断硬件 utilization 或具体 stall。

## Decoder：真实 GEMM 回放未胜过现有实现

上一轮 Nsight 测得，融合 INT8 decoder 的 GEMM 占 kernel 时间约 **65.1% / 4.258 s**，
FlashAttention 占 17.6%，SwiGLU＋量化占 8.3%。因此本轮优先回放 GEMM。
输入捕获自真实 00015 视频 latent 的融合 decoder：batch4、144 个 INT8 Linear；
保存 q、FP32 activation scale、INT8 weight、FP32 weight scale、FP16 bias 和参考输出。
媒体、权重、捕获 tensor 均未加入仓库。

下表是同轮 3 次预热、12 次轮换 CUDA Event 微测；每个 shape 筛选 6 个代表配置，
保留全部样本和异常高值。所有下表候选均已通过 FP16 输出逐位一致检查。

| GEMM | M×N×K | CUTLASS 基线 | 最佳 Triton 候选 |
| --- | --- | ---: | ---: |
| FFN up | 7188×16384×2048 | 1.017843 ms | 1.025757 ms |
| Attention output | 7188×2048×2048 | 0.142523 ms | 0.145800 ms |
| FFN down | 7188×2048×8192 | 0.472571 ms | 0.525595 ms |
| QKV | 7188×6144×2048 | 0.397787 ms | 0.403200 ms |

其他未采纳实验：

- 静态权重转为连续 `[K,N]`：4 个 shape 的最佳候选仍慢约 29%–51%。
- 调低 stages、增加 K tile/warps：未找到稳定胜出配置；部分候选出现编译器报告的 spill。
  Attention output 一轮基线升到 0.177 ms、候选 0.156 ms，但其他轮稳定基线约 0.143 ms，
  不据此宣称收益。
- 原 CUTLASS 按 M 拆为 1024/2048/4096 行：数值一致，全部更慢。
- `torch._int_mm`＋单独 FP32 dequant/bias epilogue：数值一致，但多出 INT32 写回。
  FFN up 从 1.017 ms 增至 1.786 ms，其他三类也更慢。
- 初版 Triton epilogue 用普通乘加，部分 tile 有 FP16 尾数差异；显式
  `fma(acc.float()*activation_scale, weight_scale, bias)` 后才全部精确。
  数值不通过的初版保留在原始证据中，没有用于性能推荐。

资源元数据不能证明当前 kernel 属于某种硬件瓶颈。
**Nsight Compute 已实测返回 `ERR_NVGPUCTRPERM`**，没有修改驱动权限；
本轮没有可用的 DRAM 带宽、occupancy、roofline 或 warp-stall 计数器。
这些回放只说明测试过的替代方案未胜出，不表示 decoder 已达到硬件上限。

## 另一个未采纳的 encoder 候选

卷积 block 分组复用在 C256 代表微测中约从 2.269 ms 降到 2.061 ms，
但三块完整编码 A/B 仅 **9.504354 → 9.474023 s**，约省 30 ms（0.32%），
低于本轮预先设定的 0.15 s 有用收益门槛。因此保留原流水线调度，仅采纳 producer 融合。

## 验证与边界

- 8 段视频/992 帧完整 latent 全部 `torch.equal=true`，max_abs=0；一段完整 124 帧重建 RGB 逐位一致。
- 干净发布源码副本：**82 个 CPU 测试通过，8 个 CUDA 测试跳过**；实机另外执行 **8 个 CUDA 测试通过**。
  新测试覆盖 norm pack、q、FP32 scale、bias、padding 尾块、非连续布局、零/极小值，以及完整编译后的卷积输出。
- ComfyUI 17 帧 FP32 IMAGE wrapper、实际编码 latent 的解码，以及 CPU offload/CUDA reload
  均通过；前后/直接 runtime 与 wrapper 的所有 max_abs、RMSE 均为 0。
  INT8 权重、FP32 scale 和设备状态检查通过。它是随机输入的接口 smoke，不是视频画质测试。
- compileall、关键 Ruff 规则和 diff whitespace 检查通过。
- 三块主计时加一块独立宽度复测尚未达到 ACA 建议的六块/三 seed 稳定性门槛，
  不宣称整条视频生成流程的 S4 加速。

## 复现

```bash
python bench_encoder_int8_norm.py \
  --video /path/to/00015.mp4 --videos-dir /path/to/eight-videos \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" --weights "$H3_VAE_WEIGHTS_PATH" \
  --warmup 2 --blocks 3 --check-rgb \
  --output results/int8_norm_producer.json

H3VAE_TEST_CUDA=1 python -m unittest discover -s tests -p 'test_encoder_int8_gpu.py'
H3VAE_TEST_CUDA=1 python -m unittest discover -s tests -p 'test_encoder_int8_norm.py'
```

`bench_encoder_int8_optimization.py` 保留卷积调度的历史比较用途，显式关闭 producer 融合，
避免其旧 dispatch 被绕过。新的融合必须使用上面的两个 prefix 对照工具。

[原始计时、所有回放、数值结果及源码指纹](benchmarks/int8_norm_producer_2026-10-02.json)
· [上一轮 Nsight 报告](int8_nsys_optimization_2026-10-02.md)
· [INT8 原有画质取舍](experimental_int8.md)
