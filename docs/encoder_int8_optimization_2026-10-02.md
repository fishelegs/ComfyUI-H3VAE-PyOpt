# INT8 encoder 优化结果（2026-10-02，未发布）

本轮将两项经过测量的优化接入现有 `int8_encode=true` 路径：完整视频
encode 平均从 **12.545 s 降到 10.121 s，耗时降低 19.32%**；同轮 FP16
为 **11.931 s**，优化 INT8 比它低 **15.17%**。默认仍为 FP16。

## 改动

- 八个 prefix INT8 卷积采用 `BM128/BN128/BK64`，warps4/stages2。旧 C128
  为 128/64/64，旧 C256 为 128/64/128，旧 tile 仍可显式使用。
- 动态 activation scale 改为两级 Triton absmax：8192 元素分块，最多
  2048 个 partials（8 KiB），grid-stride 扫描输入，归约成原有 FP32 scale。
  消除完整 FP16 abs 临时张量及其写回/重读。
- 量化使用连续 NDHWC 线性访问，block 从 256 调到 1024。
- 仍然是同样八层的 INT8×INT8→INT32；静态权重量化、动态 per-tensor scale、
  half-away-from-zero activation rounding、FP16 输出及原 padding/bias 位置
  均保留。不新增依赖，不改权重文件，不把不支持的配置静默回退为 FP16。

## 完整 encode A/B

输入：真实视频 sample00015，768×1344×124，encoder/decoder tile256，
encoder staged batch4，decode batch2；相同输入值及 layout；FP16 和 INT8
使用两个 runtime，encoder/decoder 均编译。`decode_fusions=false`，
`int8_decode=false`，cuDNN benchmark=True/limit5，seed20261002。

GPU 驱动名称 NVIDIA Graphics Device，SM120，约 71.1 GiB 可见显存；历史
记录标为 RTX PRO 5000 72GB。Linux、Python3.12.14、PyTorch2.11.0+cu130、
CUDA13.0、Triton3.6.0；环境中 comfy-kitchen0.2.34，此 encoder/FP16 decoder
计时不调用该库。后台 GPU 工作未停止。模型权重 SHA256 与既有公开验证一致。

比较基线为 `b5ab429` 中的 `opt/encoder_int8.py`，与公开远端
`c1893e2` 的该文件相同；其 SHA256：
`c3bce9ed316b784b6f2d15c17def5b8dfa124118275c1354d718bf4f115600dd`。

两个 INT8 配置共享相同编译图、权重和 custom-op 边界，仅在独立 benchmark
进程里切换 op 的 Python 实现。encoder CUDA graphs 关闭，dispatch 计数确认
各 INT8 路径均实际执行 14,336 次卷积，不存在缓存另一配置结果的情况。
真实层检查在计时外完成，八个实际层输出均与旧内核一致，包括 128→256 通道层。

每条路径先完成编译/诊断，再显式预热2次，5次轮换顺序 CUDA Event 测量。
下表使用均值；动态量化、输出后处理包含在内，加载、静态权重准备、编译、
媒体 IO、质量统计不计时。

| 配置 | Encode 均值 | 相对旧 INT8 耗时降低 |
| --- | ---: | ---: |
| FP16 对照 | 11.931490 s | — |
| 旧 INT8：旧 tile + 旧量化 | 12.545198 s | — |
| 仅新 tile + 旧量化 | 11.278200 s | 10.10% |
| 新 tile + 新量化 | **10.121342 s** | **19.32%** |

这证明 tile 与量化优化都有端到端收益。没有测量新的完整生成 pipeline，
也没有新测 ComfyUI 原生 VAE/TensorRT 竞争路径，不据此拼接历史成绩。

两 runtime 同时驻留时，旧/新 INT8 timed call 的 PyTorch peak allocated
均为 12,454,661,632 bytes；当前完整路径的峰值没有下降，不宣称省显存。
量化临时张量变小不等于整个模型峰值必然降低。decode 只做质量回归，报告中的
单次 decode 墙钟时间可能包含编译，不作为稳态 decode 性能。

## 数值与质量

首段完整视频的新旧 INT8 latent、FP16 decoder 重建 RGB 的 MAE/RMSE/maxabs
均为 0；仅新 tile 的 latent 也完全一致。源重建平均逐帧 PSNR：FP16
34.852193 dB，旧/新 INT8 均 34.198108 dB。

本轮共重新检查 **8 段真实视频、992 帧**，768×1344 和 768×1376 两种宽度，
不缩放、不裁剪、不丢帧。每段都通过相同 compiled runtime 分别执行旧/新 INT8
encode，再使用相同 FP16 decoder 分别重建。所有视频的 latent 和重建 RGB
MAE/RMSE/maxabs **均为 0**，所有输出有限。

| 路径 | 源重建平均逐帧 PSNR | 最差帧 PSNR | 低于30dB帧数 |
| --- | ---: | ---: | ---: |
| FP16 | 35.109668 dB | 32.709591 dB | 0/992 |
| 旧 INT8 | 34.486258 dB | 32.338038 dB | 0/992 |
| 优化 INT8 | 34.486258 dB | 32.338038 dB | 0/992 |

ComfyUI SDK `c1716a45`、comfy-kitchen0.2.34 的独立进程兼容性检查通过：
256×256 的单张 FP32 IMAGE（包含实际 encode→decode）及17帧 FP16 输入，
wrapper/direct 输出一致，CPU offload→CUDA reload 前后 encode/decode 误差为0。
8个 encoder 与72个 decoder 量化模块在全部搬运状态下保持 INT8 权重、FP32
scale 和正确设备。17帧案例是组件连接检查，不宣称17帧完整重建；这些检查
没有向运行中的 ComfyUI 服务提交 workflow。

[完整计时样本、逐帧质量、实现哈希与兼容性证据](benchmarks/encoder_int8_optimized_2026-10-02.json)
已去除本地模型/素材路径，仅保留数值与配置。原始私有输入与权重未加入仓库。


INT8 相对 FP16 的原有量化误差仍存在；“优化前后相等”不表示 INT8 相对
FP16 无损。其他 GPU、输入规格、Windows 的速度不作保证。

## 使用与复现

更新代码、重启 ComfyUI 并重新加载 VAE 后，现有 Loader 的
`int8_encode=true` 即使用新实现。保持
`dtype=fp16`、`decode_fusions=false`，encoder tile256/staged4 用于复现实测。
`int8_decode` 是独立选项；若开启，仍需 comfy-kitchen0.2.34。当前 decode
fusion profile 仍不能与 INT8 encode 同时开启。

```bash
# 提取本轮优化前的算子供受控 A/B 使用；不切换当前工作区。
mkdir -p results
git show c1893e2:opt/encoder_int8.py > results/h3_encoder_before.py
python bench_encoder_int8_optimization.py \
  --video "$H3_BENCH_VIDEO" \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" --weights "$H3_VAE_WEIGHTS_PATH" \
  --baseline-kernel-file results/h3_encoder_before.py \
  --encoder-tile 256 --encoder-staged-batch 4 \
  --decoder-tile 256 --tile-batch 2 \
  --warmup 2 --runs 5 --include-tile-only \
  --output-dir results/encoder_int8_ab
```

多视频检查：将 `--video` 替换为 `--videos-dir`，加 `--quality-only`，使用新的
输出目录。脚本会实际编码并解码新旧 INT8 与 FP16 三路，不以旧报告替代本次结果。

```bash
H3VAE_TEST_CUDA=1 python -m unittest discover -s tests -p 'test_encoder_int8_gpu.py'
```

新增4项 GPU 回归覆盖大张量/归约尾部、非连续布局、零/次正规/最大 FP16 值、
舍入边界、有限性保护、单帧对应 padded 输入、不同通道/bias/stride。CPU
发布范围的隔离副本共84项，其中80项通过、4项 GPU 测试默认跳过；GPU 测试
已另行执行通过。未提交的其他本地实验测试不计入此数。

附带修复本地既有 `opt/fp16_finalize.py` 的一处分号格式，使新上游 Registry
E702 检查通过；只拆分语句，不改变计算。此次更改不包含其他本地实验，也不创建 release。
