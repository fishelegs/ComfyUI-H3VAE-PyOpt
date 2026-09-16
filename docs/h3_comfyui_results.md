# MiniMax H3 ComfyUI 推理优化阶段结果

日期：2026-09-16。原始交接见 `/mnt/gyfs_cq2/models/keyishen/trt/handoff.md`；本文件补充其后的注意力 A/B，并保留 VAE 对照结论。

## 生产配置与速度

质量优先：20 steps、完整 INT8 ConvRot DiT、`res_multistep`，不用少步数 LoRA。H3 attention 保留 PyTorch 原生 Flash SDPA。VAE 使用 H3VAE PyOpt FP16，decoder tile 256、tile batch 2，encoder staged batch 4，两端 `torch.compile`，cuDNN benchmark 开启。8080 正式服务未改动。

1344×768、124 帧代表性热启动：DiT sampler 约 317.30 s，视频 VAE decode 约 12.07 s，参考图 VAE encode 约 0.75 s，完整 workflow 约 332.86 s。主要瓶颈仍是 DiT。

## cuDNN attention A/B（独立 8081 服务）

20 steps、1344×768、124 帧、同 prompt、同 seed；完整 INT8 和 PyOpt VAE 未变。选择性切换 H3 `[B,56,S,128]` attention 到 cuDNN，包含短 TokenRefiner 与长 DiT 序列。

| Seed | 默认 sampler | cuDNN sampler | 提速 | 默认 workflow | cuDNN workflow | 视频 PSNR / SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 27001 | 317.790 s | 309.780 s | 2.52% | 334.144 s | 326.262 s | 23.265 dB / 0.8165 |
| 27002 | 317.843 s | 309.905 s | 2.50% | 333.820 s | 325.831 s | 23.136 dB / 0.8280 |

默认 SDPA 重启后同 seed 27001 复测为 sampler 317.882 s、workflow 334.235 s；与第一次默认输出相较 PSNR 45.512 dB、SSIM 0.9893。由此 cuDNN 全覆盖的画质漂移显著大于默认自身波动；**不作为质量保持的生产方案**。长序列限定试验因动态 wrapper 触发 TorchDynamo guard、PyOpt VAE compile fallback，未得有效完整热启动数据；须把 wrapper 定义在稳定模块后再测。

原始数据：`/tmp/h3_cudnn_ab/results/comparison.json`、`default_repeat_comparison.json`；视频与 montage 在 `/tmp/h3_cudnn_ab/`。8081 已停止。

## VAE encode/decode 性能结论与新项目口径

旧 672×672 FP16 同一进程、同一输入、2 次 warmup + 7 次 CUDA Event 的完整视频对照（decoder tile 368，encoder tile 672）：

| 帧数 | 优化 PyTorch decode | 优化 PyTorch encode | PyTorch 总计 | 原 TRT decode | 原 TRT encode | TRT 总计 | PyTorch 总计优势 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 124 | 3207.635 ms | 3012.665 ms | 6220.300 ms | 3227.537 ms | 3099.612 ms | 6327.148 ms | 1.69% |
| 243 | 6430.243 ms | 5663.703 ms | 12093.946 ms | 6484.875 ms | 5803.279 ms | 12288.154 ms | 1.58% |

这是**旧基准脚本里的优化 PyTorch 路径**，不是现行 ComfyUI PyOpt 运行时的新 A/B；不能据此声称新项目已经超过 TRT。当前 TRT engine 固定 672 配置，亦不能直接与 1344×768 ComfyUI 时间比较。历史细节见 `bench_h3vae_672_summary_report.md`。

新项目应保留优化实现和公平 benchmark：同一 FP16 权重、相同随机输入、相同 tile/帧数、充分预热、CUDA Event、encode/decode 分项与画质误差；报告 GPU/软件版本、编译与热启动分别计时。当前兼容 TensorRT 环境已经安装并完成 1344×768×124 的 engine 实测：TRT decoder median 13.750 s、encoder median 21.467 s；随后按完全同 tile（decoder 368、encoder 672）交换顺序复测两轮，优化 PyTorch 总计比 TRT 快约 2.84%。详细数字、顺序和 GPU 负载说明见 [`h3_same_tile_ab_benchmark_2026-09-16.md`](h3_same_tile_ab_benchmark_2026-09-16.md)。权重、engine、上游 MiniMax 模型代码不得打包提交。

## 本地项目状态

新 Git 项目已在当前目录的 `ComfyUI-H3VAE-PyOpt/` 建立，包含 PyOpt ComfyUI loader、历史优化版 PyTorch/原 TRT benchmark，以及新的 `bench_pyopt_vs_trt.py` 全视频同输入对照。CPU 测试与全部 Python 文件语法检查通过。TensorRT 11.2.1.2 对照环境已安装在 `/data/miniconda3/envs/h3_trt_11_2` 并完成当前尺寸 engine 测试；严格同 tile A/B 仍应在目标机器上复测。GitHub 目标仓库为 `fishelegs/ComfyUI-H3VAE-PyOpt`。
