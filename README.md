# ComfyUI-H3VAE-PyOpt

面向 MiniMax H3 的 PyTorch 优化 VAE runtime 和 ComfyUI custom node。

本项目不依赖 TensorRT、ONNX、TensorRT engine 或 TensorRT Python binding。它直接复用 PyTorch/CUDA，通过 Triton kernel、`torch.compile`、空间 tile batching 和 staged encoder 优化 VAE encode/decode。在已完成的同口径测试中，PyOpt 达到参考 [ComfyUI-H3VAE_TRT](https://github.com/Windowsislamicgroup6102/ComfyUI-H3VAE_TRT) 的性能，并在当前测试 shape 上略快。

> 性能数字是指定 GPU、FP16、shape、tile 和软件版本下的稳态实测，不代表所有硬件和配置都能得到相同加速。部署到新机器后建议重新 benchmark，并做输出质量回归。

## 主要优势

- **无需 TensorRT**：不需要构建、分发或维护静态 engine，减少 TensorRT binding、engine profile 和版本兼容问题。
- **性能接近或超过 TRT**：在当前已测 shape 上，完整 encode+decode 与参考 TRT 持平或更快。
- **原生 ComfyUI 接入**：只替换 VAE Loader，继续使用原生 `VAE Encode`、`VAE Decode` 和 `VAE Decode (Tiled)` 节点。
- **保持 PyTorch 语义**：权重和模型源码不变，优化集中在 attention、normalization、layout、tile batching、staged encoder 和 compile。
- **更容易扩展**：不绑定单一静态 engine shape，便于在不同 NVIDIA GPU、PyTorch 和 CUDA 版本上继续调优。

本仓库不包含 MiniMax H3 模型源码、权重或 TensorRT engine。请遵守 [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)。

## 性能

### 测试平台

| 机器 | GPU | 显存 | Compute Capability | Driver | 独立 A/B 环境 | ComfyUI 生产环境 |
| --- | --- | ---: | ---: | --- | --- | --- |
| M1 | NVIDIA RTX PRO 6000 72GB | 72 GB（73,415 MiB） | 12.0 | 580.82.07 | Python 3.11.13、PyTorch 2.8.0+cu128、TensorRT 11.2.1.2 | ComfyUI 0.33.0、Python 3.12.14、PyTorch 2.11.0+cu130 |

独立 A/B 使用 FP16、同一随机输入、2 次 warmup 和 7 次 CUDA Event；模型加载、首次 compile 和 engine 初始化不计入稳态时间。

### 优化 PyTorch vs TensorRT：严格同 tile

两边使用相同输入、相同帧数和相同 tile 计划（decoder tile `368`、encoder tile `672`）。

| Shape（H×W×帧） | PyOpt decode / encode / 总计 | TRT decode / encode / 总计 | PyOpt 相对 TRT |
| --- | ---: | ---: | ---: |
| `672×672×124` | `3.208 / 3.013 / 6.220 s` | `3.228 / 3.100 / 6.327 s` | **快 1.69%** |
| `672×672×243` | `6.430 / 5.664 / 12.094 s` | `6.485 / 5.803 / 12.288 s` | **快 1.58%** |
| `768×1344×124` | `13.647 / 20.586 / 34.233 s` | `13.768 / 21.468 / 35.235 s` | **快 2.84%** |

`672×672` 是历史同轮 shape，`768×1344×124` 是当前 ComfyUI 分辨率的严格交换顺序 A/B。完整测试口径见 [`docs/h3_same_tile_ab_benchmark_2026-09-16.md`](docs/h3_same_tile_ab_benchmark_2026-09-16.md)。

### 默认 PyTorch 与 ComfyUI 原生 VAE

| Shape / 场景 | 默认 PyTorch decode | 优化 PyTorch decode | TRT decode | 说明 |
| --- | ---: | ---: | ---: | --- |
| `768×1344×124`，严格同 tile | **19.643 s** | **13.647 s** | **13.768 s** | tile 368；独立 PyTorch 2.8 环境 |
| `768×1344×124`，默认 tile | `18.740 s` | — | — | 上游默认 PyTorch，tile 256 |
| `768×1344×124`，ComfyUI 生产管线 | **约 15.896 s** | **12.028 s** | `13.750 s`* | PyOpt/原生 tile 256；TRT tile 368 |

严格同 tile 下，优化 PyTorch 相比默认上游 PyTorch 少约 `30.5%`，TRT 少约 `29.9%`。生产管线中的 `12.028 s` 与独立 benchmark 的 `13.647 s` 使用不同 tile/运行路径，不应直接横向比较。默认 PyTorch 的详细记录见 [`docs/h3_default_pytorch_vae_decode_2026-09-16.md`](docs/h3_default_pytorch_vae_decode_2026-09-16.md)。

`*` TRT 的 `13.750 s` 是独立 tile 368 engine 测量，不是生产 tile 256 的严格 A/B。

目前只有 M1 一台机器。后续增加 GPU 时，建议同步记录 GPU 型号、显存、driver、Python/PyTorch/CUDA、ComfyUI 版本、tile、warmup/runs 和质量误差，再把结果追加到表格中。

## 安装

### 准备模型代码和权重

本项目只提供 runtime 和 custom node。准备与权重匹配的 MiniMax H3 `FL2VA/video_vae` 目录，以及 FP16 VAE 权重：

```text
H3_VAE_MODEL_CODE_DIR=/path/to/MiniMax-H3/FL2VA/video_vae
H3_VAE_WEIGHTS_PATH=/path/to/minimax_h3_video_vae_fp16.safetensors
```

模型源码和权重不要复制进本仓库。

### 安装到 ComfyUI custom_nodes

Linux 使用启动 ComfyUI 的同一个 Python：

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/fishelegs/ComfyUI-H3VAE-PyOpt.git
cd /path/to/ComfyUI
python -m pip install -r custom_nodes/ComfyUI-H3VAE-PyOpt/requirements.txt
```

Windows portable ComfyUI 使用内置解释器：

```bat
cd /d C:\ComfyUI
python_embeded\python.exe -m pip install -r custom_nodes\ComfyUI-H3VAE-PyOpt\requirements.txt
```

`requirements.txt` 仅包含插件直接依赖：PyTorch `>=2.8`、Triton 和 safetensors。优先复用 ComfyUI 已安装且匹配 CUDA 的 PyTorch；PyOpt custom node 不需要 TensorRT。

当前性能验证在 Linux 上完成。Windows 只有在对应 Python/PyTorch/CUDA 组合存在可用 Triton wheel 时才能启用融合 kernel；如果安装不到兼容 Triton，建议使用 WSL2/Linux。

### 设置路径并重启

在启动 ComfyUI 的同一个 shell 中设置：

```bash
export H3_VAE_MODEL_CODE_DIR=/path/to/MiniMax-H3/FL2VA/video_vae
export H3_VAE_WEIGHTS_PATH=/path/to/minimax_h3_video_vae_fp16.safetensors
export TORCHINDUCTOR_CACHE_DIR=/path/to/writable/torchinductor-cache
```

然后完全重启 ComfyUI。也可以把权重放到 `ComfyUI/models/vae`，在 loader 的 `vae_name` 中选择文件；`model_code_dir` 仍需填写或通过环境变量提供。

## 接入 ComfyUI

搜索节点 **MiniMax H3 VAE Load (PyTorch Optimized)**，内部 class type 为 `H3VAEPyOptLoader`。推荐初始参数：

| 参数 | 推荐值 |
| --- | --- |
| dtype | `fp16` |
| decoder_tile_size | `256` |
| tile_batch | `2` |
| compile_decoder / compile_encoder | `true` |
| encoder_staged_batch | `4` |
| cudnn_benchmark | `true` |
| warmup | `decode` 或 `both` |

将 loader 的 `VAE` 输出连接到现有的 `MiniMax H3 Image to Video`、`VAE Encode` 或 `VAE Decode`。保持 `decoder_tile_size=256` 可与当前 ComfyUI 生产画质口径对齐；修改 tile 后请重新做质量回归。

## 最小 workflow

仓库提供一个最小 API workflow：[`examples/minimal_h3vae_pyopt_prompt.json`](examples/minimal_h3vae_pyopt_prompt.json)。连接关系如下：

```text
H3VAEPyOptLoader ── VAE ──> VAEDecode ──> PreviewImage
EmptyMiniMaxH3LatentAV ─ samples ────────┘
```

把 JSON 中的 `model_code_dir` 和 `weights_path` 改成真实路径，或删除这两个字段并使用环境变量。ComfyUI 启动后可以通过 API 提交：

```bash
curl -sS -X POST http://127.0.0.1:8188/prompt \
  -H 'Content-Type: application/json' \
  --data-binary @examples/minimal_h3vae_pyopt_prompt.json
```

该示例使用 `1344×768×124` 的全零视频 latent，只用于验证 loader、连接和 decode，不代表完整 DiT 生成质量。

## 直接性能测试

### 无 TensorRT 的 PyOpt 单测

下面是当前 PyOpt runtime 已验证的 `672×672×124` 示例：

```bash
cd /path/to/ComfyUI-H3VAE-PyOpt
python bench_pyopt_vs_trt.py \
  --pyopt-only \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" \
  --weights "$H3_VAE_WEIGHTS_PATH" \
  --height 672 --width 672 --frames 124 \
  --decoder-tile 256 --tile-batch 2 --staged-batch 4 \
  --warmup 2 --runs 7 \
  --output results/pyopt_672x672x124.json
```

常见 shape：

```text
--height 672 --width 672  --frames 124
--height 672 --width 672  --frames 243
```

### 严格 TRT A/B

当前 `768×1344×124` 的严格同 tile 对照使用 `bench_h3vae_trt.py` 空间分块模式，需要匹配的 TensorRT engine 和兼容环境。完整命令与环境清单见 [`docs/h3_trt_pipeline_benchmark_2026-09-16.md`](docs/h3_trt_pipeline_benchmark_2026-09-16.md)。

## 项目结构

- `h3vae_nodes.py`：ComfyUI loader。
- `h3vae_runtime.py`：PyOpt runtime 和 ComfyUI VAE wrapper。
- `opt/`：decoder/encoder 优化实现。
- `bench_pyopt_vs_trt.py`：PyOpt-only 或 PyOpt/TRT 全视频 benchmark。
- `docs/`：性能记录、环境说明和复现口径。

## 限制

- 当前重点是 MiniMax H3 FP16 视频 VAE；其他 VAE、权重或 layout 不保证兼容。
- 首次加载和 `torch.compile` 可能需要较长时间，建议使用 warmup。
- decoder tile 会影响边界处理、RoPE 坐标和输出质量；速度比较必须同时记录 tile 并做画质回归。
- PyOpt 和 TRT 的显存统计口径不同，不据此宣称显存优势。
