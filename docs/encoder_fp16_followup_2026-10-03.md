# INT8 encode 与 FP16 decode 后续筛选（2026-10-03）

本轮没有采纳新的 encoder 或 FP16 decoder 改动。已验证的 **INT8 encode
8.458 s**、**融合 FP16 decode 10.923 s** 保持不变；它们是不同配置的独立成绩，
不能相加。这份记录归档未晋升的候选，避免重复尝试；不包含另一条 INT8 decoder
研究的结论。

实验基线为 `dbd49e0`，SM120，同机历史标识 RTX PRO 5000 72GB；驱动报告
NVIDIA Graphics Device。Linux、Python 3.12.14、PyTorch 2.11.0+cu130、CUDA 13.0、
Triton 3.6.0，seed20261002。后台 GPU 工作保留，本轮实验通过协作锁串行运行。
以下都是**局部算子回放**，没有新的完整阶段加速或画质结论。

## INT8 encode：地址计算与较小 K tile 未胜出

采用已验证的 causal-zero 流水线作为基线。最新 Nsight Systems 中 1792 次 INT8
卷积合计 **3.812 s，占 kernel 时间 45.49%**，因此先检查地址计算与流水线。
这只是时间归因；本轮没有 NCU 计数器，不能据此判断带宽、利用率或具体 stall。

使用三个实际热点几何的合成 INT8 输入，前两帧精确置零，另测 D2 小 shape。
BM128/BN128、4 warps/4 stages，3 次预热、各 12 次轮换采样。
计时复用预量化输入与预分配输出，**不包含 norm、量化、分配或完整视频 encode**。

| 候选 | C128→128，D17/H256 | C256→256，D17/H128 | C128→256，D17/H128 |
| --- | ---: | ---: | ---: |
| 标量地址分解 | 1.981 → 3.573 ms | 1.987 → 3.530 ms | 1.006 → 1.616 ms |
| 指针递增/进位 | 1.981 → 13.909 ms | 1.987 → 13.421 ms | 1.006 → 6.631 ms |
| 标量地址＋对齐提示 | 1.965 → 1.979 ms | 1.949 → 1.943 ms | 1.000 → 1.021 ms |
| BK64 → BK32 | 1.959 → 3.900 ms | 1.950 → 3.857 ms | 0.997 → 1.790 ms |

每格都是**同轮基线 → 候选均值**；不同轮基线不交叉计算加速比。所有局部输出逐位一致，
但没有候选进入完整视频验证。

TTGIR/PTX 显示，初版标量地址及指针递增使 activation 从三缓冲异步拷贝退化为
循环内普通 load；指针递增还出现 68–72 个 spill。为已证明对齐的基址添加
`tl.multiple_of(..., 16)` 后，activation 异步拷贝恢复，资源回到 254 registers、
0 spill、48 KiB shared，但三个真实 shape 仍无稳定改善。D2 小 shape 的 7.73%
改善不能代表完整视频。BK32 将 shared 降到 24 KiB，registers 仍为 254，实际大
shape 慢 79.5%–99.1%，因此也否决。

[完整计时、源码指纹与编译器诊断](benchmarks/encoder_candidates_2026-10-03.json)。

## FP16 decode：GROUP_M=4 的局部小幅改善尚未晋升

本轮仅调整已有 FFN-up GEMM＋SwiGLU 的 CTA 分组顺序：生产 `GROUP_M=8`，
有限回放 1/4/8/16/32。保持 BM128/BN64/BK64、4 warps/2 stages、FP32 累加、
原有 FP16 舍入和权重布局；没有新增缓存、量化或算子边界。

输入捕获自真实融合 FP16 decoder 的第 0/17/35 层，含 `[8,1797,2048]` 主 batch
和 `[4,1797,2048]` 尾 batch，共六组。捕获时关闭 INT8 encode/decode，decoder
batch8，FP32 norm 开启，FP16 accumulation 关闭。每实现预热 2 次、轮换/反向
采样 12 次；记录 CUDA Event 与 wall time。与上表不同，这里的算子调用会分配输出，
仍不含媒体读取、编译或完整 decoder 计时。

| batch / 层 | 生产 GROUP_M=8 | GROUP_M=4 | 局部均值下降 |
| --- | ---: | ---: | ---: |
| 8 / 0 | 4.875 ms | 4.798 ms | 1.59% |
| 8 / 17 | 4.866 ms | 4.776 ms | 1.85% |
| 8 / 35 | 4.836 ms | 4.830 ms | 0.11% |
| 4 / 0 | 2.329 ms | 2.328 ms | 0.06% |
| 4 / 17 | 2.362 ms | 2.340 ms | 0.93% |
| 4 / 35 | 2.362 ms | 2.306 ms | 2.38% |

GROUP_M=4 的均值优势较小，batch4/第 0 层中位数反而慢约 1.07%；GROUP_M=1/16/32
均更慢。所有候选在六组捕获上逐位一致，资源均为 255 registers、0 spill、32 KiB
shared。现有证据不足以替换生产调度：没有候选完整 decoder A/B 或视频画质复查，
也不能把这些局部百分比或调用次数换算成完整 decode 加速。

[捕获合同、全部 432 个计时样本、局部数值检查与源码指纹](benchmarks/fp16_group_schedule_2026-10-03.json)。
模型权重、原始媒体和捕获 tensor 不公开，证据已去除私有文件名及本地绝对路径。

## 后续方向

保留当前生产 encode 与 FP16 decode 路径，不重复上述已否决的地址/小 K tile 搜索。
后续 encoder 候选应提出新的 producer/consumer 融合机制，或针对剩余 FP16 卷积提出
可验证方案，并保留 FP16 舍入与 GroupNorm 归约顺序。FP16 的 GROUP_M=4 只有在
新的稳定性或完整阶段证据支持时才值得晋升；当前首页性能无需更新。
