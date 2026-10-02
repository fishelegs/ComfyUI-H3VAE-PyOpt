# INT8 encode：重算 norm，直接写 INT8（2026-10-02）

完整视频 encode 从 **9.139973 s 降至 8.683874 s（−4.99%）**，峰值 allocated
减少 **39,187,456 bytes（37.37 MiB）**。基线是 `239d830`，已包含卷积流水线和
norm/absmax 融合。本轮没有修改 decoder，也没有额外量化或降低 scale 精度。

## 为什么继续重算

上一轮 Nsight 中连续量化约占 **0.709 s**，norm pack 合计约 **0.833 s**。
虽然 absmax 扫描已被消除，仍需写出完整 FP16 norm/SiLU/padding 输出，然后重新读取它量化。

新的两遍 producer 使用相同 norm 统计值：

1. 重现 norm/SiLU 的两次 FP16 舍入及原 padding，只写每 block 的 absmax。
2. 将这些小型 FP32 partials 归约为原来的单个 FP32 scale。
3. 重算相同的 norm/SiLU/padding，直接写 INT8，完全不分配归一化后的 FP16 大张量。

这是用重复算术换取更少的显存读写。不是把 GroupNorm 统计重算两遍，也不是改为
per-block scale。bias 舍入、half-away rounding、causal/reflect padding 和
INT8×INT8→INT32 卷积均保持原合同。两遍 pack 使用 block4096、4 warps，并禁止
norm 中的 FMA contraction。只在现有 SM120 INT8 自动路径启用；默认 FP16 不变。

## 筛选与完整对照

沿用 ACA 的有限候选回放、精确数值门槛和完整阶段配对验证方法。
先在两种真实 H3 shape、各有/无 pre-bias 上筛选基线及五个候选，3 次预热、16 次轮换微测：

- 连续量化 block4096、block8192 和 cache 调度：局部仅小幅改善，没有单独采用。
- 重算 block1024 与 block4096：q 和 FP32 scale 全部逐位一致，局部约改善 24%–28%。
  block4096 晋升完整视频测试，原型各 6 次均值为 9.138571 → 8.684769 s。
- 正式模块再次运行相同配对实验；下表使用正式代码，而非原型数字。

| 主视频 768×1344×124，6 次/实现 | 均值 | 峰值 allocated |
| --- | ---: | ---: |
| `239d830`，保留 FP16 norm 输出 | 9.139973 s | 7,280,809,472 bytes |
| 重算并直接写 INT8 | **8.683874 s** | **7,241,622,016 bytes** |

峰值包括保留的校验输出，不是独立模型显存。消除一个 FP16 张量不意味着完整阶段
峰值减少该张量的全部大小，报告以上实测差值。

测量合同：

- 同一 SM120 GPU：本机历史 RTX PRO 5000 72GB；驱动报告 NVIDIA Graphics Device，约 71.1 GiB。
- Linux；Python 3.12.14；PyTorch 2.11.0+cu130 / CUDA 13.0；Triton 3.6.0；comfy-kitchen 0.2.34。
- 视频 00015，FP16 输入且 layout 不变；encoder tile256 / staged batch4；decoder tile256 / batch2。
  `int8_encode=true`、`int8_decode=false`、`decode_fusions=false`。
- **同一个 compiled graph、权重和输入**，仅在 opaque custom op 内切换 producer；没有分别编译两套 prefix。
- 初始精确校验后各预热 2 次，**ABBA / BAAB / ABBA**，各 6 次 CUDA Event 计时。
  seed20261002，cuDNN benchmark=true / limit5，不启用 CUDA graphs。
- 包含动态量化、分配、runtime 拼接/输出处理；不含加载、编译、媒体 I/O 和质量计算。
- 其他 GPU 任务保持运行；本轮 GPU 实验串行持锁。结果不代表独占 GPU 或跨硬件保证。
- 权重 SHA256：`7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522`。

## 独立宽度与 Nsight 复核

另一段 00007 视频（768×1376×124）在独立进程中各预热 2 次、ABBA 各测 2 次：
**9.137206 → 8.680030 s（−5.00%）**，latent 逐位一致，峰值 allocated 同样减少 37.37 MiB。

预热后的完整 Nsight encode 范围为 **8.687092 s**，21610 个 GPU kernel 合计
**8.609392 s**，未记录本进程 GPU 活动的空隙约 68 ms。旧的 1792 次 FP16 norm pack
和 1792 次连续量化消失，被 **3584 次 recompute_pack、合计 1.057904 s** 替代；
原组合约为 0.833413＋0.709277＝1.542690 s。kernel 总数不变，收益来自数据路径变化。

1792 次 INT8 卷积仍在，合计约 **4.055 s，占 kernel 时间 47.1%**。
Nsight profile 属于独立进程，只用于归因；正式 4.99% 来自前面的同进程 A/B。
未将这些时间线比例解释为硬件 utilization、带宽或 stall。

## 正确性与适用范围

8 段视频、992 帧的完整 latent 均逐位一致；其中一段完整 124 帧视频的重建 RGB
也逐位一致。另检查 8 个真实 producer 的 q 和 scale 精确相等，完整编码每个变体
均命中 1792 次调用。已有 INT8 相对 FP16 的画质损失仍然存在。

CUDA 回归覆盖 q/scale、零/极小值、非连续布局、padding 尾块、bias、完整编译输出，
以及原卷积的 tile/stride/尾块合同；**8 个 GPU 测试通过**。干净发布源码副本
**82 个 CPU 测试通过 / 8 个 CUDA 测试跳过**，compileall、关键 Ruff、diff 检查通过。
ComfyUI SDK `c1716a45` 的 17 帧 FP32 IMAGE、实际 encode→decode 连接及 CPU offload/CUDA
reload 检查通过，前后和 wrapper/direct 的 max_abs、RMSE 都为 0。
该随机输入 smoke 证明接口兼容，不代替视频画质测试。

没有 NCU 硬件计数器：之前实际探测为 `ERR_NVGPUCTRPERM`，未修改驱动权限。
此处的读写量分析是实现机制说明，不能替代硬件带宽或 stall 实测。
三块主对照加独立宽度复核也不等于 ACA 建议的六块/三 seed 稳定性门槛。

## 剩余优化空间

当前最大的已测预算仍是 INT8 卷积约 4.055 s，其次包括原 FP16 卷积和 producer。
后续应针对卷积访存/流水线或更深的 producer-consumer 融合提出新候选，并保留完整 A/B 门槛。
已测试过的 block 分组和 decoder GEMM 替代没有稳定收益，见上一轮报告，不重复当成已证实的提速方案。
Decoder 本轮未改，GEMM＋SwiGLU 跨算子融合仍需独立实现和真实数值/性能验证。

## 复现

```bash
git show 239d830:opt/encoder_int8_norm.py > /tmp/h3vae_norm_239d830.py
python bench_encoder_int8_norm.py \
  --video /path/to/00015.mp4 --videos-dir /path/to/eight-videos \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" --weights "$H3_VAE_WEIGHTS_PATH" \
  --baseline-producer-file /tmp/h3vae_norm_239d830.py \
  --warmup 2 --blocks 3 --check-rgb \
  --output results/int8_norm_recompute.json

H3VAE_TEST_CUDA=1 python -m unittest discover -s tests -p 'test_encoder_int8_gpu.py'
H3VAE_TEST_CUDA=1 python -m unittest discover -s tests -p 'test_encoder_int8_norm.py'
```

`--baseline-producer-file` 会加载明确指定的本地基线源码，适用于这一轮同图对照。
不传该参数时，工具比较不融合 norm 的 prefix 与当前 producer，需要分别编译；
不能把两种模式的结果混称同一基线。

[原始计时、微测、源码指纹与验证](benchmarks/int8_norm_recompute_2026-10-02.json)
· [上一轮 norm/absmax 结果](int8_norm_producer_2026-10-02.md)
· [原有 INT8 画质取舍](experimental_int8.md)
