# INT8 VAE：Nsight 定位与第二轮优化（2026-10-02）

本轮将完整 INT8 encode 从 **10.120132 s 降至 9.505431 s（−6.07%）**。
基线为 `4d79fd5`，已经包含上一轮 tile/量化优化。采用的是卷积归约循环流水化；
精度、量化层数、tile 的空间边界及默认 FP16 模式均保持原合同。
新调度只在 **SM120** 自动选择，其他架构及显式旧 tile 选项保留原实现。

融合 INT8 decode 热态参考 **6.592 / 6.598 s**。本轮定位了其热点并筛选了
SwiGLU 调度，没有采纳新的 decoder 改动，不能把上述参考相对历史 6.667 s
的差异当成新加速。encode 和融合 decode 仍是不同配置，不能相加。

## 方法与测量合同

采用本机 ACA 0.3.1 的 Nsight Systems / Triton 优化技能，以及历史证据筛选流程，
由当前会话编排实验；没有启动 ACA 自动优化 runner，也没有委派代理。
候选先核对历史，再做有限回放、真实层数值检查及完整 VAE 阶段 A/B。

- GPU：驱动报告 NVIDIA Graphics Device，SM120，约 71.1 GiB；本机历史识别为 RTX PRO 5000 72GB。
- Linux；Python 3.12.14；PyTorch 2.11.0+cu130 / CUDA 13.0；Triton 3.6.0；comfy-kitchen 0.2.34。
- Nsight Systems 2025.3.2；Nsight Compute 2025.3.0；driver 580.82.07。
- 源视频 00015，768×1344×124；FP16 输入，encoder tile256、staged batch4。
  编码对照的 decoder batch2；融合 decoder profile 为 batch4、144 个 INT8 Linear。
- encoder/decoder 编译开启，禁用 CUDA graphs；cuDNN benchmark=true / limit5；seed20261002。
- 动态量化、分配、拼接及 runtime 输出处理包含在计时中；模型加载、编译、媒体读取和质量检查排除。
- 其他 GPU 任务保留；本轮 GPU 实验串行持锁。结果不是独占 GPU 或跨硬件保证。

## Nsight 热点

通过 `cudaProfilerStart/Stop` 只采集预热后的完整阶段，NVTX 范围末尾同步。
没有叠加 torch.profiler。三次 Nsight 采集均正常退出。

| 原实现阶段 | GPU kernel 时间 | 次数 | 占该阶段 kernel 时间 |
| --- | ---: | ---: | ---: |
| Encode：INT8 implicit Conv3D | 4.631 s | 1792 | 46.4% |
| Encode：连续量化 | 0.636 s | 1792 | 6.4% |
| Encode：partial absmax | 0.440 s | 1792 | 4.4% |
| Decode：两种主力 CUTLASS INT8 GEMM | 4.258 s | 7056 | 65.1% |
| Decode：FlashAttention | 1.154 s | 1764 | 17.6% |
| Decode：SwiGLU＋量化 | 0.541 s | 1764 | 8.3% |

编码 NVTX 10.052 s 内，本进程已记录 GPU 活动并集 9.985 s；解码 NVTX
6.627 s 内为 6.544 s。未记录活动的空隙分别约 **67 / 83 ms**，不是主要预算。
这些比例不是硬件 SM utilization，也不说明 kernel 是 compute-bound 或 memory-bound。

**NCU 实际探测返回 `ERR_NVGPUCTRPERM`**；驱动 `RmProfilingAdminOnly=1`。
没有改驱动权限，因而没有实测 roofline、DRAM 带宽、occupancy 或 warp-stall 数据。
寄存器和 shared-memory 数值来自 launch 元数据/Triton 编译结果，不冒充硬件计数器。

## 采纳的编码优化

原卷积按 kd/kh/kw/channel 嵌套循环，每个 channel 循环只有 2 或 4 次迭代。
新 kernel 特化静态尺寸与 stride，并展平为一个长 K 归约循环，使用 4 级软件流水线。
仍直接 gather 卷积窗口，不物化 im2col；使用相同 INT8 输入、INT32 累加、
FP32 scale 及 FP16 epilogue。量化、bias、causal/reflect padding 不变。

配置仍为 128×128×64、4 warps；新调度 shared memory 49,152 B，
旧配置 16,384 B；代表形状均无编译器报告的 spill。并不是通过提高 occupancy 实现的优化。

先验证 8 个真实卷积调用及完整 latent，再对同一编译图中的 opaque op 切换实现。
初始校验后每个实现预热 2 次，执行 **ABBA / BAAB / ABBA** 三块，共各 6 次。

| 完整 encode | 4d79fd5 基线 | 流水线候选 | 耗时降低 |
| --- | ---: | ---: | ---: |
| 768×1344×124，6 次/实现 | 10.120132 s | 9.505431 s | 6.07% |
| 768×1376×124，独立素材 00007，BAAB 各 2 次 | 10.050176 s | 9.441204 s | 6.06% |

主对照峰值 allocated 均为 **7,251,533,312 bytes**（含校验输出保留），不宣称省显存。
独立宽度 holdout 使用最终生产入口，安装元数据确认 8 层都选择 `128x128x64_pipeline4`。

优化后重新采集：完整 encode NVTX **9.520 s**，1792 次 INT8 卷积累计
**4.037 s**，调用数不变。前后 profile 不是同时期配对计时，只用于验证热点归因；
正式 6.07% 来自上表的同进程 A/B。不同进程的 cuDNN 算法选择也可能不同。

8 段视频共 **992 帧**的完整 latent 全部逐位一致（max_abs=0）；另外在 00015
完整 124 帧上复查 FP16 decoder 重建 RGB，也逐位一致。不是声称本轮重解码了全部
8 段。已有 INT8 相对 FP16 的量化误差仍存在。

检查：发布源码副本 CPU **80 passed / 4 GPU skipped**；GPU **4 passed**，覆盖
量化边界、C/O 尾块、stride、bias、非连续输入与新旧卷积结果；compileall、关键 Ruff 及
`git diff --check` 通过。Comfy 17 帧 FP32 IMAGE 的 wrapper、编码后解码及 CPU offload/CUDA reload smoke 也通过（随机输入，仅证明接口兼容）。三块主对照加单块 holdout 尚未达到 ACA 建议的六块/三 seed
稳定性验证门槛，不宣称整条视频生成工作流 S4 收益。

## 未采纳实验与下一步

后续已完成 decoder 四类 GEMM 回放，并采纳 encoder norm/absmax 融合，
完整 encode 再降至 **9.138 s**；详见[第三轮报告](int8_norm_producer_2026-10-02.md)。
下文保留本轮结束时的实验结果与优先级。

- 仅地址常量特化：没有稳定收益；展平循环但仍用 2 级流水线，收益也很小。
- 卷积调度只筛选 6 个代表配置；64×128 小 tile 更慢，更大的 tile/8 warps 不如选中配置。
- 编码量化 block1024→4096/8192：q 完全相同，但稳态微测仅小幅改善，未晋升到完整运行。
- 解码真实 `[4,1797,16384]` SwiGLU 输入，4/8/16 warps 的 q 和 FP32 scale
  逐位相同，稳态时间接近；含全部样本的均值约 0.289/0.290/0.297 ms，
  原 16-warp 首样本有明显高值。没有把这点差异包装成 decode 加速。

**Decoder 下一优先级是按真实 shape 调整 INT8 GEMM 调度。** 主力 CUTLASS 使用
128×256×64 tile、3 stages、约 73,728 B shared memory、224–254 registers/thread。
需要分别回放 QKV、attention output、FFN up/down，比较 tile/流水线/Stream-K 的实际收益，
并检查 INT32 与 FP32 dequant epilogue 是否完全一致。资源数字提示值得测试，但不足以断言
哪种 stall 主导；若有 NCU 权限，再补计数器定位。FlashAttention 是次优先级，不能因
换后端更快就放宽当前精度合同。通用去同步/CUDA graphs 的时间线空隙预算较小。

## 复现与证据

[脱敏原始数据](benchmarks/int8_nsys_pipeline_2026-10-02.json) 包含全部 A/B 样本、
质量结果、前后 kernel 汇总、所有筛选候选、源码 SHA256 和 NCU 权限结果。
Nsight 二进制报告保留在本地，不加入仓库。

```bash
# 用实际路径替换 CODE / WEIGHTS / VIDEO；确保 comfy-kitchen==0.2.34 可导入。
nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
  --capture-range-end=stop --sample=none --cpuctxsw=none --kill=none \
  -o encode_profile python bench_int8_profile.py --mode encode \
  --model-code-dir CODE --weights WEIGHTS --video VIDEO --output encode.json
# decode 使用相同入口 --mode decode（默认融合 INT8 / batch4）。

# 复测第一轮优化后的基线，显式指定其 tile；默认仍保持旧工具的 v0.2.0 对照方式。
git show 4d79fd5:opt/encoder_int8.py > /tmp/encoder_4d79fd5.py
python bench_encoder_int8_optimization.py --video VIDEO \
  --model-code-dir CODE --weights WEIGHTS \
  --baseline-kernel-file /tmp/encoder_4d79fd5.py \
  --baseline-tile-variant 128x128x64 --output-dir new_pipeline_ab
```

公开 A/B 工具还包含 FP16 对照，使用轮换顺序；本报告主数据的三块 ABBA 原始顺序保留在 JSON。
