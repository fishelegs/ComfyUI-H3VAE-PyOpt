# MiniMax H3 VAE：256/256 TRT 与 PyOpt 严格同规格 A/B

测试日期：2026-09-18。设备为 NVIDIA RTX PRO 5000 72GB；当前 `nvidia-smi` 名称显示为 `NVIDIA Graphics Device`，总显存 73415 MiB。权重 SHA-256：`7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522`。

## 口径

- 视频规格：`768×1344×124`，FP16，随机种子 `20260917`
- decoder tile：`256`；encoder tile：`256`
- 每个实现独立进程，2 次 warmup、7 次 CUDA Event 稳态测量；不含权重加载、ONNX 导出、TRT engine 构建和首次编译
- TRT 全视频计划：每轮 196 次 decoder engine 调用、224 次 encoder engine 调用
- PyOpt：Python 3.12.14、PyTorch 2.11.0+cu130，decoder tile batch 2、encoder staged batch 4、两端 compile
- TRT：Python 3.11.13、PyTorch 2.8.0+cu128、TensorRT 11.2.1.2、driver 580.82.07

TRT decoder engine 从同一 MiniMax H3 权重导出 256×256 tile ONNX 后，以 FP16、builder optimization level 3、8 GiB workspace 构建；encoder 从已有 672-tile ONNX retarget 到 256×256 后以 FP16 构建。engine 没有提交到 Git（单个文件约数 GiB），可按仓库中的 `export_static_decoder_onnx.py`、`build_static_decoder_engine.py` 和 `build_static_encoder_engine.py` 重建。

## 结果

| 实现 | Decode | Encode | 合计 |
| --- | ---: | ---: | ---: |
| PyOpt 默认 runtime | **11.437 s** | **11.983 s** | **23.420 s** |
| PyOpt `fast_linear` | **11.107 s** | **11.989 s** | **23.096 s** |
| TensorRT 11.2.1.2 | 11.966 s | 14.270 s | **26.237 s** |

相对 TRT 合计：PyOpt 默认快 **10.74%**（少 2.817 s）；PyOpt `fast_linear` 快 **11.97%**（少 3.141 s）。分项上，默认 PyOpt decode 快 **4.42%**、encode 快 **16.03%**；`fast_linear` decode 快 **7.18%**、encode 快 **15.99%**。`fast_linear` 只改 decoder，encoder 时间基本不变。

## 可复现命令

TRT engine 构建（在 `/data/miniconda3/envs/h3_trt_11_2` 环境）：

```bash
python /mnt/gyfs_cq2/models/keyishen/trt/export_static_decoder_onnx.py \
  --model-root /mnt/gyfs_cq2/models/keyishen/MiniMaxAI/MiniMax-H3/FL2VA/video_vae \
  --output /tmp/h3_trt_tile256/minimax_h3_vae_decoder_256.onnx --tile-size 256
python /mnt/gyfs_cq2/models/keyishen/trt/build_static_decoder_engine.py \
  --onnx /tmp/h3_trt_tile256/minimax_h3_vae_decoder_256.onnx \
  --engine /tmp/h3_trt_tile256/minimax_h3_vae_decoder_256.engine \
  --workspace-gib 8 --builder-optimization-level 3
python /mnt/gyfs_cq2/models/keyishen/trt/build_static_encoder_engine.py \
  --source /mnt/gyfs_cq2/models/keyishen/trt/minimax_h3_vae_encoder_672.onnx \
  --onnx-out /tmp/h3_trt_tile256/minimax_h3_vae_encoder_256.onnx \
  --engine-out /tmp/h3_trt_tile256/minimax_h3_vae_encoder_256.engine \
  --tile-size 256 --builder-optimization-level 3
python /mnt/gyfs_cq2/models/keyishen/trt/bench_h3vae_trt.py \
  --full-video --height 768 --width 1344 --frames 124 \
  --runs 7 --warmup 2 --seed 20260917 \
  --decoder-engine /tmp/h3_trt_tile256/minimax_h3_vae_decoder_256.engine \
  --encoder-engine /tmp/h3_trt_tile256/minimax_h3_vae_encoder_256.engine \
  --original-vae /mnt/gyfs_cq2/models/keyishen/MiniMaxAI/MiniMax-H3/FL2VA/video_vae \
  --decoder-tile-size 256 --encoder-tile-size 256 --skip-original
```

PyOpt 默认和实验模式：

```bash
python bench_pyopt_vs_trt.py --pyopt-only \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" --weights "$H3_VAE_WEIGHTS_PATH" \
  --height 768 --width 1344 --frames 124 \
  --decoder-tile 256 --encoder-tile 256 --tile-batch 2 --staged-batch 4 \
  --warmup 2 --runs 7 --seed 20260917
python bench_pyopt_vs_trt.py --pyopt-only --fast-linear \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" --weights "$H3_VAE_WEIGHTS_PATH" \
  --height 768 --width 1344 --frames 124 \
  --decoder-tile 256 --encoder-tile 256 --tile-batch 2 --staged-batch 4 \
  --warmup 2 --runs 7 --seed 20260917
```

本次 TRT JSON 的 SHA-256 为 `34ff43520ad5a578fb978e2a921fc91c775527fa81c545ad4e5e29c1cbc19ee2`；PyOpt 默认为 `8ec1d273e3363923979dc8c3db5b0b7319707e89df64c525bd054ba328614470`；PyOpt `fast_linear` 为 `7d7ecb2d8c371439bed6fd82b0a8641054001d39754d8ab86751c8181b08936e`。对应 JSON 暂存于 `/tmp`，不作为仓库依赖。

## 边界

这是同 GPU、同视频计划、同 tile 的工程 A/B，但 PyOpt 与 TRT 使用不同 Python/PyTorch 环境，且 TRT engine 是本机重新构建的 256-tile engine，不是 `ComfyUI-H3VAE_TRT` 仓库原有的 368-tile engine。结果适合说明当前机器和当前构建条件下的方向与量级；换 GPU、TensorRT 版本、builder 配置或 tile 后应重新测量。该性能对比没有宣称输出逐值等价，发布前仍应做真实视频画质和时序回归。
