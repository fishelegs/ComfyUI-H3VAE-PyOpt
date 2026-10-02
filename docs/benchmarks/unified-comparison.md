# 统一 Decode 竞品复测

状态：**待 GPU 执行，尚无本轮结果**。本页定义复测与验收口径，不是新的性能报告。现有 README 的百分比仅来自链接的历史实验。

## 要回答的问题

在同一 GPU、相同模型权重与 latent 上，当前 ComfyUI 默认、ComfyUI `--fast fp16_accumulation`、TensorRT、PyOpt FP16、PyOpt INT8 decoder 的稳态 decode 延迟、输出差异、峰值显存和首次运行成本分别是多少？

主图只画 **decode 秒数，慢到快**；FP16 和 INT8 分成两张图。BF16 尚未验证，不能替换 FP16 标签。INT8 图可加入本轮实测的竞品参照，但不能借用历史数字计算本轮领先比例。

## 固定条件

| 项目 | 本轮要求 |
| --- | --- |
| 硬件 | 同一张 GPU；优先复用 RTX PRO 5000 72GB。记录驱动、功率限制、温度、时钟和后台进程；测试时停止其他 GPU 作业 |
| 版本 | 固定 PyOpt、ComfyUI、模型代码的完整 commit；记录 Python、OS、PyTorch、CUDA、Triton、comfy-kitchen、TensorRT 版本 |
| 权重与 engine | 同一权重 SHA256；TRT 使用匹配权重的 FP16 256-tile engine，记录 engine SHA256、构建命令、精度和 profile |
| 主规格 | H×W×F = 768×1344×124；batch 1；FP16 latent；decoder tile 256、tile batch 2 |
| 输入 | 一次生成并保存公共 latent，记录 SHA256、shape、dtype、layout；所有后端读取相同数据。seed 固定为 20261002，不能只依赖跨版本相同 seed |
| 编译与后端 | 显式记录 compile 模式、SDPA、cuDNN 与 TF32/FP16 accumulation 设置；默认模式和 fast 模式分开 |
| 预热与采样 | 每个变体至少 2 次预热，编译/调优稳定后再测；每轮 7 次 CUDA Event，保留每次样本；三轮轮换后端执行顺序 |
| 隔离 | 各变体独立进程，仅一个模型驻留；GPU 空闲后顺序执行，避免多个模型相互占用显存 |
| 汇总 | 每轮报告 mean、median、min、max；主表使用全部 21 个稳态样本的 median，并展示各轮 median 范围 |

若某后端无法满足 tile/batch 或软件栈条件，必须单独分组并解释差异，不能称为完全同条件。TRT 若需要独立 Python/PyTorch 栈，可保留为工程对比，但应在图注中标明。

## 变体矩阵

| 变体 | 必须确认 |
| --- | --- |
| ComfyUI 默认 | 固定当前 commit，实际加载官方 H3 runtime，记录有效 tile 与编译设置 |
| ComfyUI fast | 验证 `--fast fp16_accumulation` 确实进入目标算子；不能只修改一个 PyTorch 全局开关就视为官方 fast |
| TensorRT FP16 | 匹配权重的 256-tile engine；记录内部及外部 GPU 内存 |
| PyOpt FP16 | `dtype=fp16`，`fast_linear=false`，两个 INT8 开关为 false |
| PyOpt INT8 decoder | `int8_encode=false`、`int8_decode=true`、`fast_linear=false`；记录实际量化模块与有效后端 |

可追加 `fast_linear`，但不与 INT8 同时开启。INT8 encoder 不纳入加速推荐。

## 分别采集四类指标

1. **稳态 decode**：公共 latent 已在 GPU 上；计时只包含 decoder 调用，包含每次动态量化，排除加载、静态量化、编译、CPU 拷贝、媒体 I/O。CUDA Event 前后同步。另列端到端 wall time，不混作同一指标。
2. **质量**：首先检查 shape、finite、输出范围与布局；对同 latent 的输出计算相对 FP16 参考的逐帧 PSNR、MAE、最大绝对误差。另用同一批真实视频、固定 FP16 encoder 得到公共 latent，比较源重建逐帧平均/最差 PSNR，并检查纹理、接缝和运动连续性。随机输入测试不能代替视频质量证据。
3. **显存**：单变体进程内记录加载后常驻、预热后基线以及 decode 峰值 allocated/reserved；同时监测设备/进程显存。TRT 自有分配不完全包含在 PyTorch 统计里。不要把双 runtime 的总占用称为某一个模式的显存。
4. **首次运行成本**：单独记录模型加载、静态量化或 engine 初始化、首次 decode 墙钟时间；明确编译缓存冷/热状态。TRT engine 构建时间单列，不藏入稳态，也不能忽略。不要删除用户现有缓存，应使用独立测试缓存目录。

## 现在可以运行的局部检查

以下命令的参数已对照仓库脚本检查；它们**尚未在本轮 GPU 上执行，也不能合并成统一竞品排行榜**。先设置 `H3_VAE_MODEL_CODE_DIR`、`H3_VAE_WEIGHTS_PATH`，INT8 使用兼容 CUDA GPU 和 `comfy-kitchen==0.2.34`。

### PyOpt FP16 与 TensorRT 局部对照

在可同时导入依赖的环境中，设置 `H3_TRT_DECODER_ENGINE`、`H3_TRT_ENCODER_ENGINE` 为匹配的 256-tile engine：

```bash
python bench_pyopt_vs_trt.py \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" \
  --weights "$H3_VAE_WEIGHTS_PATH" \
  --decoder-engine "$H3_TRT_DECODER_ENGINE" \
  --encoder-engine "$H3_TRT_ENCODER_ENGINE" \
  --height 768 --width 1344 --frames 124 \
  --decoder-tile 256 --encoder-tile 256 --tile-batch 2 --staged-batch 4 \
  --warmup 2 --runs 7 --seed 20261002 \
  --output results/pyopt_trt_256.json
```

只验证 PyOpt 时加 `--pyopt-only` 并去掉两个 engine 参数。该脚本用随机输入、顺序运行两个后端，也会测 encode；它尚不支持公共 latent 文件、完整 ComfyUI/INT8 变体和独立进程显存对照。

### 真实视频 FP16 / INT8 质量与局部耗时

将 `H3_BENCH_VIDEO` 指向主规格的视频；脚本按输入实际尺寸运行，须检查输出报告的 shape，不会通过上述环境变量自动裁剪：

```bash
python bench_int8_roundtrip.py \
  --video "$H3_BENCH_VIDEO" \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" \
  --weights "$H3_VAE_WEIGHTS_PATH" \
  --decoder-tile 256 --encoder-tile 256 \
  --tile-batch 2 --encoder-staged-batch 4 \
  --warmup 2 --runs 7 --seed 20261002 \
  --per-frame-csv --output-dir results/int8_roundtrip
```

使用新输出目录以保留原始结果。该工具运行四条 encode/decode 路径；关注 E0/D0 与 E0/D1 才是固定 FP16 encode 的 decoder 比较。它同时驻留两个 runtime，不能用于独立单模型显存结论。目录批量质量检查可把 `--video` 换为 `--videos-dir` 并添加 `--quality-only`。

## 完成统一复测前还需补齐

- 固定并检查当前 ComfyUI H3 API，补充官方默认 / fast 的性能采集入口。仓库的 `bench_comfy_fast_quality.py` 是质量探针，不是完整官方 fast 性能基准；历史性能采集脚本未收录到仓库。
- 为各后端补公共 latent 读写与一致的输入/输出契约，记录哈希；增加独立子进程编排、有效配置和原始采样输出。
- 补充单变体显存、TRT 内存与首次运行计时；在真实 GPU 上核对计时边界及输出质量。
- 完成三轮采样后归档脱敏 JSON/CSV、命令、版本与质量结果。脚本输出可能包含本地路径/素材名；公开前去除私人路径和未授权媒体信息，不上传权重或 engine。

只有上述条件齐备且实际运行成功，才将结果标记为“已完成”，更新 README 与两张 decode 图。耗时降低公式为 `(参照秒数 - PyOpt秒数) / 参照秒数 × 100%`；仅在同一轮可比实验内计算。
