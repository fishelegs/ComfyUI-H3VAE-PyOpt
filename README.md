# ComfyUI-H3VAE-PyOpt

MiniMax H3 视频 VAE 的 PyTorch 加速实现与 ComfyUI 插件。通过 Triton 融合算子、`torch.compile`、tile batching 和分阶段编码优化 encode/decode；日常推理无需 TensorRT 或预生成 engine。

- **即插即用**：替换 VAE Loader，继续使用 ComfyUI 原生 `VAE Encode` / `VAE Decode` 节点。
- **当前 runtime 比 ComfyUI 原生 VAE 更快**：672×672×124 的 encode+decode 比 [ComfyUI 优化提交 `b2e31e8`](https://github.com/Comfy-Org/ComfyUI/commit/b2e31e89412a01a67be599571cc57ff74b242a82) 默认配置低 **40.75%**。
- **优化 PyTorch 达到 TRT 级性能**：本仓库的空间分块基准在相同 tile 下与 [ComfyUI-H3VAE_TRT](https://github.com/Windowsislamicgroup6102/ComfyUI-H3VAE_TRT) 基本持平；768×1344×124 的 encode+decode 低 **2.84%**。这条基准路径与当前 ComfyUI 插件 runtime 不同。

## 性能

以下结果均在 **NVIDIA RTX PRO 5000 72GB**、FP16 下测得。表中数字依次为 **decode / encode / 合计**，单位为秒；均为预热后的稳态时间，不含加载、首次编译和 engine 初始化。

### 当前插件 runtime

对比 [ComfyUI `b2e31e8`](https://github.com/Comfy-Org/ComfyUI/commit/b2e31e89412a01a67be599571cc57ff74b242a82) 及其父提交的 runtime 直测：Python 3.12、PyTorch 2.11+cu130、decoder tile `256`，每个进程 1 次预热、3 次 CUDA Event 均值。`--fast` 列额外启用 `fp16_accumulation`。

| 视频 H×W×帧 | ComfyUI 提交前 | ComfyUI `b2e31e8` | `b2e31e8` + `--fast` | 本项目 runtime |
| --- | ---: | ---: | ---: | ---: |
| 672×672×124 | 9.781 / 13.325 / 23.107 | 8.340 / 7.627 / 15.967 | 6.965 / 6.890 / 13.854 | **6.511 / 2.951 / 9.461** |
| 768×1344×124 | 17.329 / 23.325 / 40.653 | 14.995 / 13.369 / 28.364 | 12.492 / 12.090 / 24.582 | **11.443 / — / —** |

672×672×124 的本插件总时长比提交后默认配置低 **40.75%**，比 `--fast` 配置低 **31.71%**。768×1344×124 的本插件 decode 比提交后默认配置低 **23.7%**；当前 encoder 在该整帧尺寸会触发 int32 索引保护，因此没有可报告的 encode 或合计时间。

### 优化 PyTorch 与 TensorRT：同 tile 基准

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

权重也可放入 `ComfyUI/models/vae`，在节点中选择；模型代码目录仍需设置。当前性能测试在 Linux 完成；Windows 需要为其 Python/PyTorch/CUDA 组合安装兼容的 Triton。

## ComfyUI 用法

添加 **MiniMax H3 VAE Load (PyTorch Optimized)** 节点（`H3VAEPyOptLoader`），把 `VAE` 输出连接到现有的 `VAE Encode`、`VAE Decode` 或 MiniMax H3 workflow。建议从 FP16、decoder tile `256`、tile batch `2`、encoder staged batch `4` 开始；需要排除首次编译开销时将 `warmup` 设为 `decode` 或 `both`。

最小 decode workflow：[examples/minimal_h3vae_pyopt_prompt.json](examples/minimal_h3vae_pyopt_prompt.json)。把其中的模型路径改成实际位置后，在仓库目录执行：

```bash
curl -sS -X POST http://127.0.0.1:8188/prompt \
  -H 'Content-Type: application/json' \
  --data-binary @examples/minimal_h3vae_pyopt_prompt.json
```

示例使用全零 latent，只验证节点连接与 decode，不代表生成画质。

## 直接测试

用当前插件 runtime 测试已支持的 672×672×124，无需 TensorRT：

```bash
python bench_pyopt_vs_trt.py --pyopt-only \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" \
  --weights "$H3_VAE_WEIGHTS_PATH" \
  --height 672 --width 672 --frames 124 \
  --decoder-tile 256 --tile-batch 2 --staged-batch 4 \
  --warmup 1 --runs 3 \
  --output results/pyopt_672x672x124.json
```

同 tile TRT 基准的复现口径见 [A/B 记录](docs/h3_same_tile_ab_benchmark_2026-09-16.md)。首次编译可能耗时较长；调整 tile 会改变边界处理与输出，需要重新做画质回归。
