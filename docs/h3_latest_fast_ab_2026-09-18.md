# MiniMax H3 VAE：最新 ComfyUI `--fast` 与 PyOpt 可选线性内核

测试日期：2026-09-18。机器此前确认是 NVIDIA RTX PRO 5000 72GB（当前驱动 `nvidia-smi` 名称显示为 `NVIDIA Graphics Device`，总显存 73415 MiB）；Python 3.12.14、PyTorch 2.11.0+cu130、FP16。ComfyUI 为 [`387f98a`](https://github.com/Comfy-Org/ComfyUI/commit/387f98aa2822f684b8597959a52a467d88cc4806)，comfy-kitchen 0.2.34；权重 SHA-256 为 `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522`。GPU 可能有其他负载；数字是本机、该环境下的稳态实测，不保证跨机器复现。

## 性能

768×1344×124、encoder/decoder tile 256、PyOpt decoder tile batch 2；每项 1 次预热、3 次 CUDA Event 测量均值，排除加载/首次编译。ComfyUI 行来自最新官方 runtime；PyOpt 默认行沿用上一轮同口径实测，可选模式行是本次完整 runtime 实测。各行在独立进程中运行，不是单进程交替 A/B。

| 路径 | Decode | Encode | 合计 |
| --- | ---: | ---: | ---: |
| 最新 ComfyUI 默认 | 15.021 s | 13.349 s | 28.370 s |
| 最新 ComfyUI `--fast fp16_accumulation` | 12.440 s | 12.059 s | 24.499 s |
| PyOpt 默认 | 11.443 s | 11.985 s | 23.428 s |
| PyOpt `fast_linear` 可选模式 | **11.062 s** | **11.967 s** | **23.029 s** |

本次 ComfyUI `--fast` 相对其默认模式合计低 **13.64%**。PyOpt 可选模式相对本项目默认旧测合计低 **1.70%**，相对最新 ComfyUI `--fast` 低 **6.00%**。这些是延迟差，不代表各实现输出逐值相同。

## 加速机制与精度

[ComfyUI PR #16187](https://github.com/Comfy-Org/ComfyUI/pull/16187) 将 encoder 的 GroupNorm/SiLU/causal padding 合并到 comfy-kitchen 算子，并在 `--fast fp16_accumulation` 下选用 FP16 累加的 3D 卷积；decoder 使用支持偏置/残差融合的 FP16 线性算子，并改进 tile 批处理。其 [comfy-kitchen PR #167](https://github.com/Comfy-Org/comfy-kitchen/pull/167) 说明 `fp16_conv3d`/`fp16_linear` 采用更快的 FP16 累加，因此加速不是单纯打开 PyTorch 后端开关。本项目已有独立的 Triton 归一化/填充、QK/RoPE 融合、整体编译和 tile batching；本次只借用可验证的 decoder 线性内核，没有把官方 encoder 卷积内核直接替换进现有全图编译路径。

同一随机输入、相同权重与 tile 的数值比较：

| 对比 | 指标 | 结果 |
| --- | --- | ---: |
| 最新 ComfyUI `--fast` vs 其默认 decode | 像素 PSNR / MAE | 61.04 dB / 0.000624 |
| 最新 ComfyUI `--fast` vs 其默认 encode | latent 相对 RMSE / MAE | 0.547% / 0.001227 |
| PyOpt `fast_linear` vs 其默认 decode，768×1344×124 | 像素 PSNR / MAE | 61.17 dB / 0.000616 |
| PyOpt `fast_linear` vs 其默认 decode，672×672×124 | 像素 PSNR / MAE | 61.65 dB / 0.000581 |

这里的像素 PSNR 按 [0,1] 输出、峰值 1 计算；latent 相对 RMSE = RMSE / 参考 latent RMS。随机输入的这些数值只能说明不是逐值无损，**不能证明真实视频没有可见画质变化**；真实样例还需检查纹理、接缝和时序。`fast_linear` 仅改 decoder，encoder 路径未变。

同一 PyOpt 进程内 768×1344×124 decoder A/B（1 次预热、3 次测量）：默认全局 FP16 开关关闭 **11.426 s**；仅打开该开关 **11.783 s**（变慢）；加上 comfy-kitchen 线性自定义算子 **11.099 s**。672×672×124 下，可选算子在全局开关关闭/开启分别为 **6.356 / 6.351 s**，说明该可选模式不需要改变 ComfyUI 的全局配置。真实集成后的完整 runtime 数字见上表。默认值继续关闭该模式。

## 复现

在 ComfyUI 所用环境安装 `comfy-kitchen==0.2.34`，设置模型代码和权重路径，然后：

```bash
python bench_pyopt_vs_trt.py --pyopt-only --fast-linear \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" --weights "$H3_VAE_WEIGHTS_PATH" \
  --height 768 --width 1344 --frames 124 \
  --decoder-tile 256 --encoder-tile 256 --tile-batch 2 --staged-batch 4 \
  --warmup 1 --runs 3 --seed 20260917
```

去掉 `--fast-linear` 可复测默认路径。`bench_kitchen_linear_trial.py` 在同进程中比较 PyOpt decoder 默认、仅全局 FP16 开关、comfy-kitchen 线性算子；`bench_comfy_fast_quality.py` 对最新 ComfyUI 默认/`--fast` 做输出误差 A/B。后者需要传入 `--comfy-root` 与权重路径；官方性能脚本在独立 `/tmp/comfyui_h3_commit_bench.py` 运行，未作为本仓库依赖。
