# MiniMax H3 VAE：空间编码与算子复测

测试日期：2026-09-17。设备为 NVIDIA RTX PRO 5000 72GB；Python 3.12.14、PyTorch 2.11.0+cu130、FP16。GPU 上有其他负载，因此结果是本机实测，不是跨机型保证。所用权重 SHA-256：`7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522`。

## 实现与结论

当前插件把原始 VAE 的空间 tile 切分、overlap blend 规则接入已有的 staged/fused encoder，修复了 768×1344×124 整帧路径超过 int32 索引范围的问题。默认 `encoder_tile_size=0` 按尺寸选 tile：最大边不超过 672 用 672，否则用 256；decoder tile 为 256、batch 为 2。也可以显式指定 encoder tile。tile 改变会改变输出，不能视作纯粹的实现开关。

对 768×1344×124 的 256-tile 插件配置，1 次预热、3 次 CUDA Event 均值：decode **11.443 s**、encode **11.985 s**、合计 **23.428 s**。相同机器上的 ComfyUI `b2e31e8` 默认配置为 14.995 / 13.369 / **28.364 s**：本插件合计低 17.40%；对其 `--fast fp16_accumulation` 配置的 24.582 s 低 4.70%。两者 encoder tile 都为 256。先前的 672×672×124 PyOpt 结果仍为 6.511 / 2.951 / **9.461 s**，比 ComfyUI `b2e31e8` 默认的 15.967 s 低 40.75%；该 shape 下两者的 encoder tile 不同。

768×1344×124 若强制使用 encoder tile 672，decode / encode / 合计为 11.419 / 17.798 / **29.217 s**，比 ComfyUI 默认配置慢 3.01%，且和其 256-tile 输出差异明显。故该尺寸默认选择 256。

最终自动选择路径 `encoder_tile_size=0` 另做 1 次预热、2 次测量的烟测，得到 11.412 / 11.974 / **23.386 s**；对原始 256-tile 编码器复测的 latent 相对 RMSE 仍为 0.166%。正式对比表仍采用上面的 3 次测量结果。

## 输出一致性

随机 FP16 `[-1,1]` 视频，按 latent 张量计算误差；参考实现为相同权重的原始 `AutoencoderKLLegacy`。`相对 RMSE = RMSE / 参考 latent RMS`。这不是主观画质指标。

| Shape | PyOpt tile / 参考 tile | latent 相对 RMSE | MAE | 解释 |
| --- | ---: | ---: | ---: | --- |
| 672×672×124 | 672 / 672 | 0.202% | 0.000510 | 已有无空间分块路径 |
| 768×1344×124 | 256 / 256 | **0.166%** | 0.000502 | 推荐配置；与原始实现同 tile，形状一致 |
| 768×1344×124 | 672 / 672 | 0.173% | 0.000447 | 新空间分块路径，形状一致 |
| 768×1344×124 | 672 / 256 | **14.924%** | 0.111652 | 不同分块布局造成显著输出变化；不能宣传为逐值等价 |

因此，大尺寸默认使用 256 tile 同时兼顾速度与贴近 ComfyUI 默认输出；正式视频仍需目视画质回归，尤其是 tile 接缝和运动连续性。

## 其他优化尝试

- 同进程 decoder tile batch A/B（768×1344×124，tile 256，每项 1 次预热、2 次 CUDA Event）：batch 1/2/4 的 median 分别为 12.316 / **11.456** / 11.637 s。batch 2 相对 batch 1 的像素 PSNR 为 72.44 dB、MAE 0.000158。自适应 batch `0` 已加入，最多选 2，并为大画布保留显存；最终代码复测实际选 2，decode **11.464 s**、相对串行像素 PSNR 72.45 dB。默认仍为实测较快的固定 batch 2。
- PyTorch `allow_fp16_accumulation` 在 672×672×124 的 off/on/off 测试中，decode median 为 6.507 / 6.732 / 6.544 s，encode 为 3.096 / 3.103 / 3.105 s。本地没有收益，故不默认开启。这不是 ComfyUI comfy-kitchen `--fast` 路径的等价替代。
- 672×672×124 encode 的算子 profile 中，`aten::cudnn_convolution` 累计约 2.490 s（140 次），是主要热点；已融合的归一化/填充算子只占较小部分。后续优化应优先研究卷积布局和算法，而非盲目叠加已有融合。
- 隔离安装的 comfy-kitchen 0.2.34 `fp16_conv3d` 在代表性单算子输入上比 cuDNN 快约 11.85%，但数值也有变化；当前运行环境中的 comfy-kitchen 0.2.31 没有该算子。单算子结果不能外推成整段 encoder 加速，本轮没有将其接入默认路径。

## 复现

```bash
python bench_pyopt_vs_trt.py --pyopt-only \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" --weights "$H3_VAE_WEIGHTS_PATH" \
  --height 768 --width 1344 --frames 124 \
  --decoder-tile 256 --encoder-tile 256 --tile-batch 2 --staged-batch 4 \
  --warmup 1 --runs 3 --seed 20260917

python bench_encoder_quality.py \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" --weights "$H3_VAE_WEIGHTS_PATH" \
  --height 768 --width 1344 --frames 124 \
  --encoder-tile 256 --reference-tile 256
```

其余 A/B 脚本：`bench_decoder_tile_batch.py`、`bench_fp16_accum.py`、`bench_operator_profile.py`。旧 PyTorch/TRT 同 tile 对比使用不同 Python/PyTorch 与 benchmark 路径，详见 [独立 A/B 记录](h3_same_tile_ab_benchmark_2026-09-16.md)，不能与本页绝对时延混算。
