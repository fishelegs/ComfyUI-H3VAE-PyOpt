# ComfyUI-H3VAE-PyOpt

MiniMax H3 视频 VAE 的 PyTorch 优化运行时和 ComfyUI custom node。它不需要 TensorRT、ONNX、TensorRT engine 或 TensorRT Python binding；在当前已测 shape 上，完整 VAE encode+decode 达到参考 [ComfyUI-H3VAE_TRT](https://github.com/Windowsislamicgroup6102/ComfyUI-H3VAE_TRT) 的水平，并在严格同 tile 测试中略快。

> 结论范围：下面的“更快”是本机、指定 shape、FP16、warmup 后的工程实测，不是对所有 GPU、CUDA/PyTorch 版本和 tile 配置的保证。发布前请在目标机器上复测并做画质回归。

## 主要优势

- **不依赖 TRT**：不需要构建或分发静态 TensorRT engine，也不受 TensorRT binding、engine profile 和版本兼容问题限制；部署只需要目标 ComfyUI 已有的 CUDA PyTorch，再安装 Triton 和 safetensors。
- **性能达到或略超 TRT**：当前 `768×1344×124`（即 `1344×768×124`）严格同 tile A/B 中，PyOpt 完整 VAE 为 `34.233 s`，参考 TRT 路径为 `35.235 s`，PyOpt 快约 `2.84%`；decoder 约快 `0.88%`、encoder 约快 `4.11%`。历史 `672×672×124/243` 同轮结果分别快 `1.69%/1.58%`。
- **直接接入 ComfyUI**：只替换 VAE Loader，继续使用 ComfyUI 原生 `VAE Encode`、`VAE Decode`、`VAE Decode (Tiled)` 等节点；H3 的采样、音频和视频节点无需改写。
- **保留标准 PyTorch 语义**：优化集中在 QK RMSNorm/RoPE、GroupNorm/SiLU/padding、layout、tile batching、staged encoder 和 `torch.compile`，权重和模型源码仍由外部 MiniMax H3 目录提供。
- **更容易扩展机型**：不锁定某一组静态 engine shape；后续可以在不同 NVIDIA GPU、PyTorch/CUDA 版本上独立调优并把结果追加到测试矩阵。

本项目不包含 MiniMax 模型源码、权重或 TRT engine。请遵守 [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)。

## 性能对比

### 测试机和软件

当前已完成的结果都来自同一台 Linux 机器；云环境没有返回具体 marketing model 名称，所以 README 保留运行时报告的名称，不猜测为某个 RTX 型号。

| 机器 ID | GPU 报告名 / Compute Capability | 显存 | Driver | 独立 A/B 环境 | ComfyUI 生产环境 |
| --- | --- | ---: | --- | --- | --- |
| M1 | `NVIDIA Graphics Device` / `12.0` | `73,415 MiB` | `580.82.07` | Python `3.11.13`、PyTorch `2.8.0+cu128`、CUDA `12.8`、TensorRT `11.2.1.2` | ComfyUI `0.33.0`、Python `3.12.14`、PyTorch `2.11.0+cu130` |

独立 A/B 使用 FP16、同一随机输入、2 次 warmup + 7 次 CUDA Event；模型加载、首次 compile 和 engine 反序列化不计入稳态时间。测试期间 GPU 仍有 ComfyUI/trpc 负载（约 82–100%），因此 1–2% 级差异应在空闲 GPU 上换序复测。

### 严格同 tile：优化 PyTorch vs 参考 TRT

TRT 对照采用与参考仓库同类的静态 decoder/encoder engine（decoder tile `368`、encoder tile `672`）。PyOpt 使用相同 tile 计划、相同输入和相同 warmup/计时口径。

| Shape（H×W×帧） | PyOpt decode / encode / 总计 | TRT decode / encode / 总计 | PyOpt 相对 TRT | 记录 |
| --- | ---: | ---: | ---: | --- |
| `672×672×124` | `3.208 / 3.013 / 6.220 s` | `3.228 / 3.100 / 6.327 s` | **快 1.69%** | decoder 368、encoder 672；28/8 次 tile 调用 |
| `672×672×243` | `6.430 / 5.664 / 12.094 s` | `6.485 / 5.803 / 12.288 s` | **快 1.58%** | decoder 368、encoder 672；56/15 次 tile 调用 |
| `768×1344×124` | `13.647 / 20.586 / 34.233 s` | `13.768 / 21.468 / 35.235 s` | **快 2.84%** | 两次交换顺序 A/B 的 median 平均；105/48 次调用 |

前两行来自旧 672×672 正式同轮报告，使用仓库保留的 legacy 优化 benchmark 路径；它们是同一优化系列的历史参考，不等同于现行 ComfyUI runtime 的新 A/B。后一行是当前 ComfyUI 分辨率的两次交换顺序严格 A/B，使用项目 benchmark 的空间分块优化 PyTorch 路径。生产 ComfyUI runtime 的 tile 256 结果单列在后面的生产表中。完整细节见 [`docs/h3_same_tile_ab_benchmark_2026-09-16.md`](docs/h3_same_tile_ab_benchmark_2026-09-16.md) 和 [`docs/h3_trt_pipeline_benchmark_2026-09-16.md`](docs/h3_trt_pipeline_benchmark_2026-09-16.md)。

### 默认 PyTorch 对比

默认行分两种用途：

1. **严格同 tile 的上游默认 PyTorch**：未启用本项目的融合、compile、channels-last 或 staged batch，用来回答优化收益。
2. **当前 ComfyUI 原生 `nodes.VAEDecode`**：来自实际 ComfyUI 8080 热启动日志，用来回答生产管线收益。它使用不同的 ComfyUI/PyTorch/CUDA 运行栈，不能和独立上游数字混为同一个实现。

| Shape | 默认/原生 PyTorch decode | 优化 PyTorch decode | TRT decode | tile / 运行栈 |
| --- | ---: | ---: | ---: | --- |
| `768×1344×124`，严格同 tile | **19.643 s** | **13.647 s** | **13.768 s** | tile 368；Python 3.11 / PyTorch 2.8 / FP16 |
| `768×1344×124`，默认常用 tile | `18.740 s` | — | — | 上游 PyTorch tile 256；独立基准 |
| `768×1344×124`，生产 ComfyUI | **约 15.896 s** | **12.028 s** | `13.750 s`* | tile 256（PyOpt/原生）或 368（TRT）；ComfyUI 0.33 |

`*` TRT 的 `13.750 s` 是静态 tile 368 的独立 engine 测量，不是生产 tile 256 的严格 A/B。当前 ComfyUI 原生 `nodes.VAEDecode` 既有日志范围约 `15.894–15.912 s`，代表值 `15.896 s`；它可作为生产 PyTorch baseline，和 PyOpt 的 `12.028 s` 同属 ComfyUI 生产口径。默认上游 PyTorch 的完整测量、输出 shape 和口径说明见 [`docs/h3_default_pytorch_vae_decode_2026-09-16.md`](docs/h3_default_pytorch_vae_decode_2026-09-16.md)。

严格同 tile 的当前结果可以直接概括为：默认上游 PyTorch decoder `19.643 s`，优化 PyTorch `13.647 s`，TRT `13.768 s`；优化 PyTorch 和 TRT 相对默认分别少约 `30.5%` 和 `29.9%`。这支持“无需 TRT 也能达到甚至略快于参考 TRT”的工程结论，但不应解读为所有显卡上固定的收益。

参考 TRT 仓库公开 README 使用“up to 1.7x faster”的宣传口径，但没有提供与本项目相同 shape、tile、输入和软件版本的完整公开 A/B 表格；本 README 的数字只引用本机可复核记录，不把该上限当作普适值。

### 后续机型矩阵

目前只有 M1 一台机器。后续增加 GPU 时，建议每个 shape 都记录 GPU marketing model、compute capability、显存、driver、Python/PyTorch/CUDA、ComfyUI 版本、tile、warmup/runs、GPU 是否空闲和画质误差；不要用不同 tile 或不同精度拼接成“加速”。

## 安装

### 1. 准备外部模型代码和权重

本项目只提供 runtime 和 custom node。准备以下两个外部路径：

```text
H3_VAE_MODEL_CODE_DIR=/path/to/MiniMax-H3/FL2VA/video_vae
H3_VAE_WEIGHTS_PATH=/path/to/minimax_h3_video_vae_fp16.safetensors
```

权重需要与 `FL2VA/video_vae` 配置和代码匹配；不要把它们复制进 GitHub 仓库。

### 2. 安装到 ComfyUI custom_nodes

Linux（有 NVIDIA CUDA，使用启动 ComfyUI 的同一个 Python）：

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/fishelegs/ComfyUI-H3VAE-PyOpt.git
cd /path/to/ComfyUI
python -m pip install -r custom_nodes/ComfyUI-H3VAE-PyOpt/requirements.txt
```

Windows portable ComfyUI 使用其内置解释器，不要使用系统 Python：

```bat
cd /d C:\ComfyUI
python_embeded\python.exe -m pip install -r custom_nodes\ComfyUI-H3VAE-PyOpt\requirements.txt
```

`requirements.txt` 只列出插件直接依赖：PyTorch `>=2.8`、Triton、safetensors。优先复用 ComfyUI 已安装且匹配 CUDA 的 PyTorch；不要为了这个节点随意替换 ComfyUI 的 CUDA torch。**PyOpt custom node 不需要安装 TensorRT。**

当前性能验证是在 Linux 上完成的；Windows portable 只有在你的 Python/PyTorch/CUDA 组合存在可用 Triton wheel 时才能启用这些融合 kernel。如果 Windows 上找不到兼容的 `triton` 包，建议使用 WSL2/Linux，或先在不启用本项目 custom node 的情况下保留原生 VAE；不要把 Linux 的 benchmark 数字直接套到 Windows。

### 3. 设置路径并重启 ComfyUI

在启动 ComfyUI 的同一个 shell 中设置：

```bash
export H3_VAE_MODEL_CODE_DIR=/path/to/MiniMax-H3/FL2VA/video_vae
export H3_VAE_WEIGHTS_PATH=/path/to/minimax_h3_video_vae_fp16.safetensors
export TORCHINDUCTOR_CACHE_DIR=/path/to/writable/torchinductor-cache
```

然后完全重启 ComfyUI。也可以不设置 `H3_VAE_WEIGHTS_PATH`，把权重放进 `ComfyUI/models/vae`，在 loader 的 `vae_name` 中选择文件；`model_code_dir` 仍需填写或通过环境变量提供。

### 4. 在 workflow 中替换 VAE

搜索节点 **MiniMax H3 VAE Load (PyTorch Optimized)**（内部 class type：`H3VAEPyOptLoader`），推荐初始参数：

| 参数 | 推荐值 | 说明 |
| --- | --- | --- |
| dtype | `fp16` | 与当前 benchmark 一致 |
| decoder_tile_size | `256` | 生产默认画质口径；不要为了速度直接改成 368 |
| tile_batch | `2` | 当前验证过的 decoder tile batch |
| compile_decoder / compile_encoder | `true` | 首次请求会 compile；可用 warmup 移出首请求 |
| encoder_staged_batch | `4` | 当前生产配置 |
| cudnn_benchmark | `true` | 形状稳定时开启 |
| warmup | `decode` 或 `both` | 建议填写与实际视频相同的帧数/宽/高 |

将 loader 的 `VAE` 输出连接到原有的 `MiniMax H3 Image to Video`、`VAE Encode` 或 `VAE Decode` 节点即可。不要同时保留两个 VAE loader；音频 VAE 仍使用原来的音频 VAE。

## 最小 ComfyUI workflow

仓库提供一个可直接提交到 ComfyUI `/prompt` API 的最小请求体：[`examples/minimal_h3vae_pyopt_prompt.json`](examples/minimal_h3vae_pyopt_prompt.json)。它只做：

```text
H3VAEPyOptLoader ── VAE ──> VAEDecode ──> PreviewImage
EmptyMiniMaxH3LatentAV ─ samples ────────┘
```

使用前把 JSON 中的 `model_code_dir` 和 `weights_path` 改成真实路径，或删除这两个 optional 字段并在启动 ComfyUI 前设置上面的环境变量。提交示例：

```bash
curl -sS -X POST http://127.0.0.1:8188/prompt \
  -H 'Content-Type: application/json' \
  --data-binary @examples/minimal_h3vae_pyopt_prompt.json
```

这个 workflow 使用 `1344×768×124` 的全零 H3 视频 latent，输出 124 帧预览；它用于验证节点加载、连接和 decode，不代表完整 DiT 生成质量。若你的 ComfyUI 版本没有 `EmptyMiniMaxH3LatentAV`，保留 loader/`VAEDecode` 连接，并把 latent 输入换成该版本现有的 MiniMax H3 latent 节点。

## 最精简的直接性能测试

在项目目录运行 `bench_pyopt_vs_trt.py`，`--pyopt-only` 不需要 TRT engine。下面以当前 PyOpt runtime 已验证的 `672×672×124` 为最小无 TRT 示例：

```bash
cd /path/to/ComfyUI-H3VAE-PyOpt
python bench_pyopt_vs_trt.py \
  --pyopt-only \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" \
  --weights "$H3_VAE_WEIGHTS_PATH" \
  --height 672 --width 672 --frames 124 \
  --decoder-tile 256 --encoder-tile 672 \
  --tile-batch 2 --staged-batch 4 \
  --warmup 2 --runs 7 \
  --output results/pyopt_672x672x124.json
```

常见 shape 只需替换三项：

```text
--height 672 --width 672  --frames 124   # 672×672×124
--height 672 --width 672  --frames 243   # 672×672×243
--height 768 --width 1344 --frames 124   # 1344×768×124；严格同 tile 见下方命令
```

若只想测无 TRT 的 1344×768，当前 `H3VAEPyOptRuntime` 的整帧 encoder 不是严格同 tile 路径；请先用上面的 672×672 shape，或在 ComfyUI 生产管线中按实际 tile 测量。要复现 README 中 1344×768 的严格 TRT A/B，使用 `bench_h3vae_trt.py` 的空间分块模式（需要匹配 engine 和 TensorRT 环境）：

```bash
python bench_h3vae_trt.py \
  --full-video \
  --height 768 --width 1344 --frames 124 \
  --runs 7 --warmup 2 \
  --decoder-engine /path/to/minimax_h3_vae_decoder_368_opt5.engine \
  --encoder-engine /path/to/minimax_h3_vae_encoder_672.engine \
  --original-vae "$H3_VAE_MODEL_CODE_DIR" \
  --decoder-tile-size 368 --encoder-tile-size 672 \
  --pytorch-decoder-tile-size 368 --pytorch-encoder-tile-size 672 \
  --pytorch-whole-decoder-compile --pytorch-attention-in-graph \
  --pytorch-compile-mode max-autotune-no-cudagraphs \
  --pytorch-sdpa-backend flash --pytorch-whole-encoder-compile \
  --pytorch-encoder-channels-last-3d --pytorch-encoder-quant-conv-in-graph \
  --pytorch-encoder-use-mean --pytorch-cudnn-benchmark \
  --pytorch-cudnn-benchmark-limit 5 \
  --pytorch-encoder-compile-mode max-autotune-no-cudagraphs \
  --json-out results/pyopt_vs_trt_768x1344x124.json \
  --md-out results/pyopt_vs_trt_768x1344x124.md
```

脚本使用随机 FP16 输入，报告 decode/encode 的 mean、median、min、max、输出 shape、显存和文件 SHA-256；compile 首次开销不计入稳态测量。完整的历史结果和限制见 `docs/`。

无需 GPU 的基础检查：

```bash
python -m unittest discover -s tests -v
python -m py_compile h3vae_nodes.py h3vae_runtime.py bench_pyopt_vs_trt.py bench_h3vae_trt.py
```

## 目录和报告

- `h3vae_nodes.py`：ComfyUI loader，输出兼容 stock VAE 节点的 `VAE`。
- `h3vae_runtime.py`：PyOpt runtime 和 ComfyUI wrapper。
- `opt/`：decoder/encoder 融合与 staged batching 实现。
- `bench_pyopt_vs_trt.py`：PyOpt-only 或 PyOpt/TRT 全视频 benchmark。
- [`docs/h3_same_tile_ab_benchmark_2026-09-16.md`](docs/h3_same_tile_ab_benchmark_2026-09-16.md)：当前尺寸严格同 tile A/B。
- [`docs/h3_default_pytorch_vae_decode_2026-09-16.md`](docs/h3_default_pytorch_vae_decode_2026-09-16.md)：默认上游 PyTorch decode baseline。
- [`docs/h3_comfyui_pipeline_benchmark_2026-09-16.md`](docs/h3_comfyui_pipeline_benchmark_2026-09-16.md)：ComfyUI 生产管线记录。
- [`environment/trt_requirements.txt`](environment/trt_requirements.txt)：仅用于复现 TRT 对照的兼容环境；PyOpt custom node 不需要它。

## 已知限制

- 当前优化和验证重点是 MiniMax H3 的 FP16 视频 VAE；不同权重、不同 latent layout 或非 H3 VAE 不保证可用。
- decoder tile 会影响边界 padding、RoPE 坐标和输出；`256` 与 `368` 的速度不能直接横比，改 tile 后必须做画质回归。
- 首次加载/compile/cuDNN 搜索可能很慢并占用显存；启用 loader warmup 或预热一个与生产相同的 shape。
- PyOpt 为整段视频 materialize/staged 运行，显存估算与 TRT engine 的静态 activation 统计不是同一口径；不能用 Torch allocator 峰值宣称显存优劣。
- 当前只有 M1 机器数据；新 GPU、PyTorch/CUDA 或 ComfyUI 版本请先跑 benchmark，再决定默认参数。

## GitHub

本地项目默认分支为 `main`，提交身份为 `fishelegs <socialnetwork@163.com>`。目标仓库：[fishelegs/ComfyUI-H3VAE-PyOpt](https://github.com/fishelegs/ComfyUI-H3VAE-PyOpt)。
