# ComfyUI-H3VAE-PyOpt

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/fishelegs/ComfyUI-H3VAE-PyOpt/actions/workflows/ci.yml/badge.svg)](https://github.com/fishelegs/ComfyUI-H3VAE-PyOpt/actions/workflows/ci.yml)

MiniMax H3 视频 VAE 的 PyTorch 加速实现与 ComfyUI 插件。默认使用经过验证的 **FP16 浮点路径**运行完整 VAE encode/decode；如果愿意用很小的画质差异换取更低延迟，可单独开启实验性的 **INT8 decoder**。两种模式都不需要 TensorRT 或预生成 engine。

- **默认 FP16，保持浮点基线**：`int8_encode=false`、`int8_decode=false`，不引入 INT8 量化误差，适合作为日常推荐配置。
- **可选 INT8 decoder**：仅量化 decoder 的 72 个热点 FFN Linear；本机实测 decode 耗时降低 **27.9%**，encode+decode 合计降低 **13.7%**。
- **即插即用**：只替换 VAE Loader，继续使用 ComfyUI 原生 `VAE Encode` / `VAE Decode` 节点。
- **比 ComfyUI 原生 VAE 更快**：相对 [ComfyUI 优化提交 `b2e31e8`](https://github.com/Comfy-Org/ComfyUI/commit/b2e31e89412a01a67be599571cc57ff74b242a82) 默认配置，672×672×124 的 encode+decode 耗时低 **40.75%**；768×1344×124 在相同 encoder tile 下低 **17.40%**。
- **无需 TensorRT，也可超过同规格 TRT**：768×1344×124、encoder/decoder tile 均为 `256` 时，默认 FP16 runtime 为 **23.420 s**，相同权重在本机重新构建的 TRT 静态 engine 为 **26.237 s**，合计快 **10.74%**。

> [!IMPORTANT]
> 当前经过验证并用于下列公开 benchmark 的浮点基线是 **FP16，不是 BF16**。这里的“浮点基线”表示没有额外的 INT8 量化；VAE encode→decode 本身是有损过程，因此本项目不使用“数学无损”表述。BF16 尚未接入和验证，不能将现有 FP16 数据标记为 BF16。

## 性能

以下结果均在 **NVIDIA RTX PRO 5000 72GB** 上测得。性能数字均为预热后的稳态时间，不含加载、首次编译和 engine 初始化。

### 默认 FP16 与可选 INT8 decoder

这是本项目建议用户优先比较的两种配置。测试输入为 768×1344×124，encoder/decoder tile 均为 `256`，staged batch `4`、decode batch `2`；2 次预热、3 次交错 CUDA Event 测量。INT8 路径只量化 decoder 的 72 个 FFN Linear；其他 decoder 算子保留原有浮点精度，完整 encoder 沿用 FP16 基线。

| 配置 | Loader 关键参数 | Encode | Decode | Encode + Decode¹ | 相对 FP16 |
| --- | --- | ---: | ---: | ---: | ---: |
| **FP16 浮点基线（推荐）** | `dtype=fp16`、两个 INT8 开关均关闭 | 11.946 s | 11.477 s | 23.423 s | — |
| **FP16 + INT8 decoder（实验性）** | `dtype=fp16`、`int8_decode=true` | 11.946 s | **8.278 s** | **20.224 s** | Decode **−27.9%**；合计 **−13.7%** |

¹ 合计为分别测得的 encode 与 decode 稳态均值之和，不是完整视频生成耗时。动态量化已计入，模型加载、首次编译和视频 I/O 未计入。环境为 Python 3.12.14、PyTorch 2.11.0+cu130、CUDA 13.0、Triton 3.6.0、comfy-kitchen 0.2.34；测试期间有后台 GPU 负载。

画质数据来自内部 8 段视频、共 **992 帧**（768×1376×124 与 768×1344×124）。PSNR 在视频编码前，对未压缩 RGB `[0,1]` 逐帧计算；“平均”是 992 个逐帧 PSNR 的算术平均值。

| 配置 | 对原视频：逐帧平均 PSNR | 对原视频：最差帧 PSNR | 对 FP16 重建：逐帧平均 PSNR | 低于 30 dB |
| --- | ---: | ---: | ---: | ---: |
| **FP16 浮点基线** | 35.110 dB | 32.711 dB | — | 0 / 992 |
| **FP16 + INT8 decoder** | **34.963 dB** | **32.618 dB** | **49.585 dB** | **0 / 992** |

在这组内部视频上，INT8 decoder 相对 FP16 基线的源视频重建平均 PSNR 下降 **0.146 dB**。这些数字是已测素材上的观测结果，不是对任意生成视频的质量保证；30 dB 筛查通过也不代表视觉无损或不存在时序差异。建议在自己的典型素材上复测后再用于正式工作流。[完整逐帧方法、量化范围与原始汇总](docs/experimental_int8.md)。

### 与 ComfyUI 官方 runtime 对比

对比 [ComfyUI `b2e31e8`](https://github.com/Comfy-Org/ComfyUI/commit/b2e31e89412a01a67be599571cc57ff74b242a82) 及其父提交的 runtime 直测：Python 3.12、PyTorch 2.11+cu130、decoder tile `256`，每个进程 1 次预热、3 次 CUDA Event 均值。ComfyUI encoder tile 为 `256`；本项目在 672×672 使用 `672`，在 768×1344 使用 `256`。`--fast` 列额外启用 `fp16_accumulation`。

| 视频 H×W×帧 | ComfyUI 提交前 | ComfyUI `b2e31e8` | `b2e31e8` + `--fast` | 本项目 runtime |
| --- | ---: | ---: | ---: | ---: |
| 672×672×124 | 9.781 / 13.325 / 23.107 | 8.340 / 7.627 / 15.967 | 6.965 / 6.890 / 13.854 | **6.511 / 2.951 / 9.461** |
| 768×1344×124 | 17.329 / 23.325 / 40.653 | 14.995 / 13.369 / 28.364 | 12.492 / 12.090 / 24.582 | **11.443 / 11.985 / 23.428** |

672×672×124 的本插件总时长比提交后默认配置低 **40.75%**，比 `--fast` 配置低 **31.71%**。768×1344×124 使用同样的 encoder tile `256` 时，总时长分别低 **17.40%** 和 **4.70%**；与同 tile 原始 VAE 编码器的 latent 相对 RMSE 为 **0.166%**。默认 `encoder_tile_size=0` 会按尺寸选择上述 tile；改变 tile 会改变输出。[完整 A/B 与精度说明](docs/h3_spatial_encoder_followup_2026-09-17.md)。

对 2026-09-17 最新 ComfyUI [`387f98a`](https://github.com/Comfy-Org/ComfyUI/commit/387f98aa2822f684b8597959a52a467d88cc4806) 再测 768×1344×124：默认 **28.370 s**，`--fast fp16_accumulation` **24.499 s**。本项目默认 runtime 为 **23.428 s**；可选 `fast_linear` 模式为 **23.029 s**（decode 11.062 / encode 11.967 s），比官方 `--fast` 合计低约 **6.0%**。`fast_linear` 只作用于 decoder，需 comfy-kitchen 0.2.34，且会改变数值；同输入相对本项目默认 decode 的像素 PSNR 约 **61.2 dB**。它默认关闭，不需要开启 ComfyUI 全局 `--fast`。[测试口径与精度细节](docs/h3_latest_fast_ab_2026-09-18.md)。

### INT8 encoder 研究结果

项目同时提供独立的 `int8_encode` 研究开关：encoder 的 8 个热点卷积使用真实 **INT8×INT8→INT32**，其余算子保持 FP16。这是混合精度 INT8，不是全网络 INT8，也不是无损模式。

RTX PRO 5000 72GB，768×1344×124，encoder/decoder tile 均为 `256`，staged batch `4`、decode batch `2`；2 次预热、3 次交错 CUDA Event 测量：

| 路径 | Encode | Decode | 合计¹ | 相对 FP16 合计耗时 |
| --- | ---: | ---: | ---: | ---: |
| 默认 PyOpt | 11.946 s | 11.477 s | 23.423 s | — |
| 仅 INT8 decode | 11.946 s | **8.278 s** | **20.224 s** | 降低 13.7% |
| 仅 INT8 encode | 12.538 s | 11.476 s | 24.014 s | 增加 2.5% |
| INT8 encode + decode | 12.538 s | 8.278 s | 20.816 s | 降低 11.1% |

¹ 与上方主表口径相同。**INT8 encoder 实测慢 4.9%，因此不建议为了加速而开启；当前最佳速度/质量折中是保持 FP16 encode，只开启 `int8_decode`。** 不将本轮数据与其他表中的历史 TRT/ComfyUI 绝对时延混算。

8 段视频、共 **992 帧**（768×1376×124 和 768×1344×124）的完整四路重建检查：

| 路径 | 与原视频：平均 / 最差帧 PSNR | 与默认重建：平均 PSNR |
| --- | ---: | ---: |
| 默认 | 35.110 / 32.711 dB | — |
| 仅 INT8 decode | 34.963 / 32.618 dB | 49.585 dB |
| 仅 INT8 encode | 34.486 / 32.338 dB | 42.589 dB |
| INT8 encode + decode | 34.355 / 32.235 dB | 41.767 dB |

七组比较全部为 **0/992 帧低于 30 dB**。相对原视频，INT8 decode、INT8 encode、两者同时开启的平均 PSNR 分别下降 0.146、0.623、0.755 dB；INT8 encoder 的 latent 相对 RMSE 为 8.05–12.07%。

### 当前规格与 TensorRT：严格同 tile A/B

下面这组结果使用相同的 768×1344×124 视频规格、FP16、随机种子、decoder/encoder tile `256`、2 次预热和 7 次 CUDA Event 测量；数字为 **decode / encode / 合计**，单位为秒。TRT 使用本机由同一权重导出的 256-tile 静态 engine，不是上游仓库预置的 368-tile engine。

| 路径 | Decoder tile | Encoder tile | Decode / Encode / 合计 | 相对 TRT 合计 |
| --- | ---: | ---: | ---: | ---: |
| PyOpt 默认 runtime | 256 | 256 | **11.437 / 11.983 / 23.420** | **快 10.74%** |
| PyOpt `fast_linear`（实验性） | 256 | 256 | **11.107 / 11.989 / 23.096** | **快 11.97%** |
| TensorRT 11.2.1.2 静态 engine | 256 | 256 | 11.966 / 14.270 / **26.237** | — |
| 原 TRT 仓库 TensorRT engine | 368 | 672 | 13.768 / 21.468 / **35.235** | 不适用（不同 tile） |

这证明在当前 RTX PRO 5000 72GB 和当前 engine 构建条件下，PyOpt 不仅达到 TRT 级别，而且在完整 encode+decode 上有约 **10–12%** 余量；其中主要差异来自 encoder。PyOpt 环境为 PyTorch 2.11+cu130，TRT 环境为 PyTorch 2.8+cu128 / TensorRT 11.2.1.2，因此这是同 GPU、同输入计划的工程对比，不应解释为所有 GPU 或所有 engine 的固定收益。[完整 TRT 256/256 A/B 记录](docs/h3_trt_256_same_tile_benchmark_2026-09-18.md)。

表中最后一行是原 [ComfyUI-H3VAE_TRT](https://github.com/Windowsislamicgroup6102/ComfyUI-H3VAE_TRT) engine 的历史结果，用于展示原始 TRT 基线；它的 368/672 tile 与前三行不同，不能用来计算 256/256 的 10.74%/11.97% 优势。原 engine 的两轮交换顺序 A/B 记录见[历史报告](docs/h3_same_tile_ab_benchmark_2026-09-16.md)。

### 历史 TensorRT engine：368/672 同 tile 基准

本仓库 `bench_h3vae_trt.py` 中的**空间分块 PyTorch 实现**与 TRT 使用同一输入、decoder tile `368`、encoder tile `672`。它是独立基准路径，**不是上表的 ComfyUI 插件 runtime**。环境为 Python 3.11、PyTorch 2.8+cu128、TensorRT 11.2；2 次预热、7 次 CUDA Event。768×1344×124 的结果取两轮交换执行顺序的 median 平均。

| 视频 H×W×帧 | 优化 PyTorch | TensorRT | PyTorch 合计优势 |
| --- | ---: | ---: | ---: |
| 672×672×124 | 3.208 / 3.013 / 6.220 | 3.228 / 3.100 / 6.327 | 1.69% |
| 672×672×243 | 6.430 / 5.664 / 12.094 | 6.485 / 5.803 / 12.288 | 1.58% |
| 768×1344×124 | **13.647 / 20.586 / 34.233** | 13.768 / 21.468 / 35.235 | **2.84%** |

同 tile 的 768×1344×124 默认上游 PyTorch decode 为 **19.643 s**，空间分块优化实现为 **13.647 s**，耗时低 **30.5%**。[同 tile A/B 详情](docs/h3_same_tile_ab_benchmark_2026-09-16.md) · [默认 PyTorch 基线](docs/h3_default_pytorch_vae_decode_2026-09-16.md)。两组表采用不同的 PyTorch 环境和调用路径，不应把绝对时延混合比较。测试时 GPU 有其他负载；换机器或修改 tile 后请重新测试并检查输出质量。

## 安装

需要 NVIDIA CUDA GPU、兼容的 PyTorch 与 Triton，以及与权重匹配的 MiniMax H3 `FL2VA/video_vae` 模型代码。本仓库不包含模型代码、权重或 TRT engine；模型使用须遵守 [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)。

在启动 ComfyUI 所用的 Python 环境中安装：

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/fishelegs/ComfyUI-H3VAE-PyOpt.git
cd /path/to/ComfyUI
python -m pip install -r custom_nodes/ComfyUI-H3VAE-PyOpt/requirements.txt
```

优先沿用 ComfyUI 已安装、与本机 CUDA 匹配的 PyTorch。设置模型路径后重启 ComfyUI：

```bash
export H3_VAE_MODEL_CODE_DIR=/path/to/MiniMax-H3/FL2VA/video_vae
export H3_VAE_WEIGHTS_PATH=/path/to/minimax_h3_video_vae_fp16.safetensors
```

权重也可放入 `ComfyUI/models/vae`，在节点中选择；模型代码目录仍需设置。当前性能测试在 Linux 完成；Windows 需要为其 Python/PyTorch/CUDA 组合安装兼容的 Triton。Decoder 的 SDPA 后端默认设为 `auto`：PyTorch 会按当前 GPU、dtype 和输入 shape 选择可用实现，在 Flash Attention 不可用时回退，避免 `No available kernel`。如需固定后端做可复现基准，可在启动 ComfyUI 前显式设置 `MINIMAX_H3_TORCH_SDPA_BACKEND=flash`；强制 `flash` 的环境或输入不受支持时仍会报错。

## ComfyUI 用法

添加 **MiniMax H3 VAE Load (PyTorch Optimized)** 节点（`H3VAEPyOptLoader`），把 `VAE` 输出连接到现有的 `VAE Encode`、`VAE Decode` 或 MiniMax H3 workflow。推荐先使用 `dtype=fp16`、`int8_encode=false`、`int8_decode=false`、decoder tile `256`、tile batch `2`、encoder staged batch `4`。Encoder tile 默认自动选择（672×672 用 `672`，768×1344 用 `256`），也可显式设置。`tile_batch=0` 可按空闲显存选择 1 或 2。需要排除首次编译开销时将 `warmup` 设为 `decode` 或 `both`。

需要尝试实验性 decoder 加速时，先在 ComfyUI 的 Python 环境中安装 `python -m pip install 'comfy-kitchen==0.2.34'`，再把 Loader 的 `fast_linear` 设为 `true`；无需全局 `--fast`。该模式是精度/速度折中，建议对真实视频检查细节、接缝和运动连续性。

希望进一步降低 decode 延迟时，保持 `dtype=fp16` 和 `int8_encode=false`，仅将 `int8_decode=true`。需要 NVIDIA CUDA SM80+ 和 `comfy-kitchen==0.2.34`，且不能与 `fast_linear` 同时开启。`int8_encode` 仅供研究，目前实测更慢。未支持的配置会明确报错，不会静默退回 FP16 并标为 INT8。

[INT8 encode→decode 最小 API workflow](examples/minimal_h3vae_pyopt_int8_roundtrip_prompt.json) 使用 `LoadImage → VAE Encode → VAE Decode → PreviewImage`：先上传一张 256×256 图片并替换示例文件名，再配置模型路径。[仅 INT8 decode 示例](examples/minimal_h3vae_pyopt_int8_prompt.json)也可单独使用。两者只演示节点连接；视频质量需用真实素材验证。

最小 decode workflow：[examples/minimal_h3vae_pyopt_prompt.json](examples/minimal_h3vae_pyopt_prompt.json)。把其中的模型路径改成实际位置后，在仓库目录执行：

```bash
curl -sS -X POST http://127.0.0.1:8188/prompt \
  -H 'Content-Type: application/json' \
  --data-binary @examples/minimal_h3vae_pyopt_prompt.json
```

示例使用全零 latent，只验证节点连接与 decode，不代表生成画质。

## 直接测试

用当前插件 runtime 测试 672×672×124，无需 TensorRT：

```bash
python bench_pyopt_vs_trt.py --pyopt-only \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" \
  --weights "$H3_VAE_WEIGHTS_PATH" \
  --height 672 --width 672 --frames 124 \
  --decoder-tile 256 --encoder-tile 672 --tile-batch 2 --staged-batch 4 \
  --warmup 1 --runs 3 --seed 20260917 \
  --output results/pyopt_672x672x124.json
```

测试 768×1344×124 时改用 `--height 768 --width 1344 --encoder-tile 256 --output results/pyopt_768x1344x124.json`；其他参数不变。可用 `bench_encoder_quality.py` 对照原始 VAE 做同 tile 数值回归。

同 tile TRT 基准的复现口径见 [A/B 记录](docs/h3_same_tile_ab_benchmark_2026-09-16.md)。首次编译可能耗时较长；调整 tile 会改变边界处理与输出，需要重新做画质回归。

## Contributing and compatibility

- Contribution expectations: [CONTRIBUTING.md](CONTRIBUTING.md)
- CPU CI vs GPU/manual validation: [docs/testing.md](docs/testing.md)
- Maintainer-tested compatibility and benchmark evidence: [docs/compatibility.md](docs/compatibility.md)
- Community results: open an issue with the **Benchmark report** template so environment, latency, and correctness evidence stay comparable.

## License

The source code authored for this repository is available under the
[MIT License](LICENSE).

ComfyUI, PyTorch, Triton, safetensors, optional comfy-kitchen support, and the
MiniMax H3 / FL2VA model code and model weights are separate third-party
components and retain their respective licenses. This repository does not
include or relicense MiniMax H3 model code or model weights.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the licensing
boundary and third-party component notes.
