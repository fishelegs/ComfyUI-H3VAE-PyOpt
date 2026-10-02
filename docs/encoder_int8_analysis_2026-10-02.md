# INT8 encode 后续优化分析（2026-10-02）

> 本文保留优化前的调查记录。后续实现与完整视频验证见
> [INT8 encoder 优化结果](encoder_int8_optimization_2026-10-02.md)。

已将远端 `origin/main` 的 `c1893e2` 合并到本地 `b5ab429`，保留本地
`a8b47d9` decode fusion 提交及原有未跟踪实验文件。README 冲突已解决；
没有 push。生产 encoder 内核和默认精度设置未修改。

## 结论

优先验证 **更大的输出通道块 BN128**，然后优化 **动态量化的数据读写**。
本轮完整单算子测试已观察到约 17%–18% 的延迟下降，但尚未做候选配置的
完整视频 encode、ComfyUI 或画质回归，不能报告端到端加速。

现有完整视频证据仍为：768×1344×124、encoder tile256/staged4，FP16
encode 11.946492 s，INT8 encode 12.537563 s，INT8 慢 4.95%。8 段视频
仅开启 INT8 encode 时，源重建平均 PSNR 从 35.109612 降到 34.486258 dB。
这些是 [2026-09-22 历史实测](experimental_int8.md)，不是本轮复测。

## 本轮 GPU 实测

本机驱动名称 NVIDIA Graphics Device，SM120，约 71.1 GiB 可见显存；
历史记录将其标为 RTX PRO 5000 72GB。Linux、Python 3.12.14、
PyTorch 2.11.0+cu130、CUDA 13.0、Triton 3.6.0。

两组几何来自既有真实 encoder 热点捕获，数据与权重为随机 FP16：

- C128：输入 padding 后 `[1,128,19,258,258]`，输出 `[1,128,17,256,256]`。
- C256：输入 padding 后 `[1,256,19,130,130]`，输出 `[1,256,17,128,128]`。

3×3×3、stride1、channels-last-3d；seed20261002；2 次预热、7 次轮换顺序
CUDA Event 采样；cuDNN benchmark=True/limit5。表格使用中位数，JSON 同时
保留均值与全部样本。静态权重量化、编译与 padding 不计时；完整 INT8
算子包含动态量化、卷积、反量化/bias 和输出分配。存在其他 GPU 工作，未停止
任何其他进程。本实验不加载 VAE/ComfyUI/comfy-kitchen，不测 decode 或峰值显存。

| 热点 | FP16 Conv3d | 当前完整 INT8 算子 | BN128 完整 INT8 算子 | 相对当前 INT8 延迟降低 |
| --- | ---: | ---: | ---: | ---: |
| C128/T17/H256 | 5.111 ms | 5.146 ms | 4.222 ms | 17.97% |
| C256/T17/H128 | 4.386 ms | 4.102 ms | 3.403 ms | 17.03% |

候选为 `BM128/BN128/BK64, num_warps=4, num_stages=2`。
当前 C128 为 `128/64/64`，C256 为 `128/64/128`，所以 C256 同时改变 BN/BK。
候选两组输出相对当前 INT8 的 maxabs 均为 0；所有被筛选卷积输出均有限。
这仅证明这些随机输入上的等价，未覆盖实际 C128→C256 通道转换层、单帧、
边界 tile、全部真实权重和编译集成。

当前 INT8 相对 FP16 的随机算子 RMSE 分别约 0.015416/0.015722，不能当作
真实视频画质指标。候选沿用现有量化方式，不能消除历史 INT8 encode 画质损失。

原始数据：[初筛与确认 JSON](benchmarks/encoder_int8_probe_2026-10-02.json)。
复现入口（需兼容 CUDA/Triton 环境）：

```bash
python bench_encoder_int8_tile_analysis.py --output results/encoder_int8_tile_analysis.json
```

脚本只在自身进程内给 `_TILE_VARIANTS` 增加试验配置，不修改 runtime 策略。
`conv_*` 计时复用已量化输入和预分配输出，不能与 `int8_full*` 混淆。
Profiler 原始数据包含 ATen 与 CUDA kernel 的重叠条目，不可相加。

## 按优先级推进

### 1. 扩展 kernel tile 搜索，然后验证完整 encoder

`opt/encoder_int8.py` 仅提供三个 tile，输出通道块均为 64，warps/stages
固定 4/2。本轮六组有限搜索中，128/128/64 在两组热点上最好；更大的 BM、
BK 或更多 stages 并未自然带来收益。

下一步先在所有八个实际 INT8 层的 shape/权重上验证候选，尤其是输入/输出
通道不同的层，再将候选接入独立实验配置。优先保持量化数值、padding、bias
位置和 FP32 scale 不变。进行固定视频、tile256/staged4、相同输入 layout、
相同编译设置的三路交错 A/B：FP16、现有 INT8、候选 INT8。

### 2. 消除动态量化的完整 abs 临时张量

`quantize_activation_tensor()` 当前依次执行 `x.abs().amax()`、scale 运算、
独立 Triton quantize。C128/C256 动态量化中位耗时约 1.488/0.744 ms，
相当于原完整算子的 28.9%/18.1%（独立测量的近似占比，非严格可加分解）。

C128 的 abs 临时 FP16 张量约 309 MiB。Profiler 显示 abs、amax、quantize
均有可见成本。先试两级 Triton absmax reduction：第一阶段直接读取 x 并
输出分块最大值，第二阶段归约和生成 FP32 scale，避免完整 abs 写回与重读。
再试 quantize 的更大 block 和连续 NDHWC 线性索引特化；当前 block 仅 256。

保持有限输入上的 scale、全零保护和 half-away-from-zero 舍入语义。
由于全局 per-tensor scale 依赖完整输入，不应把“单个普通 kernel 一遍读取
直接完成全局动态量化”当成已经成立的方案。新 reduction 需同时测误差与速度。

### 3. 融合 norm/SiLU/padding 与量化准备

`BiasFusedResidualBlock` 已融合 norm/SiLU/padding，但产物仍为大块 FP16
padded tensor，INT8 随后重新扫描求 scale 并量化。

可以研究 producer 输出时顺带生成 absmax partials，之后归约 scale，再执行
quantize；进一步考虑把 reflect/causal padding 放进量化 pack，减少 padded
FP16 中间结果。必须保留现有 FP16 中间舍入、反射边界和前两帧 causal-zero
语义。这比单纯 tile 修改侵入性大，放在第二阶段。

### 4. 更深层的 implicit-GEMM 数据搬运优化

现有 kernel 以 kd/kh/kw/channel 循环 gather 输入，再执行 INT8 dot。
若前两步后卷积本体仍占主导，再测索引常量特化、循环组织、数据复用和流水线。
实际是否受访存、指令发射或占用率限制，需要 Nsight Compute 证据，当前
torch.profiler 不能确定。可参考 [NVIDIA CUTLASS implicit-GEMM 文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html)
关于直接生成矩阵 tile、避免 im2col 物化和优化地址迭代的设计；该文的 2D
示例不代表 H3 3D Conv 或 SM120 已可直接替换。

## 不优先做的方向与验收标准

- 不再优先尝试通用 im2col + GEMM：本地既有 T3 探针慢 6.6–22 倍，物化
  3D 窗口增加大量数据搬运。这只否定该原型，不能否定全部实现。
- 不把增大 `encoder_staged_batch` 当作八个 INT8 prefix 层的直接 batch
  优化：`encode_staged()` 当前逐 clip 执行 prefix，拼 batch 后仅交给 suffix。
- 不立即扩大 INT8 层数或改为静态 activation scale：已有量化精度损失，
  前者未必加速，后者需要单独校准与视频质量验证。
- 完整路径复测至少报告 encode 耗时、候选对原 INT8 latent 误差、对 FP16
  与源视频的重建误差、峰值显存；质量沿用 8 段/992 帧协议。算子提速不能
  直接按比例换算为整段 encode 提速。
- 本地 decode fusion 配置当前要求 `int8_encode=false`。测试 encoder 时
  保持 `decode_fusions=false`，组合功能需要另行集成验证。

合并检查：compileall、关键 Ruff 检查、99 项 unittest 通过；新增诊断脚本
也通过 py_compile 和关键 Ruff。生产 INT8 内核与默认路径未修改，当时尚未接入生产优化；后续实现与完整视频证据见上方链接。
