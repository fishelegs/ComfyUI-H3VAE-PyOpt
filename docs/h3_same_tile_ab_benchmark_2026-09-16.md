# MiniMax H3 PyTorch / TensorRT 同 tile A/B

测试日期：2026-09-16

## 口径

这是对当前 `1344×768×124`（24 fps，约 5.167 s）视频的严格性能对照。两边使用同一随机 latent/pixel 输入、同一 Python 3.11.13 + PyTorch 2.8.0+cu128 进程、同一 FP16 权重、同一空间/时间切分、同一 `2` 次 warmup + `7` 次 CUDA Event 测量：

- decoder tile：`368`
- encoder tile：`672`
- 每次 decoder `105` 个 tile engine/模块调用，encoder `48` 个调用
- PyTorch：`bench_h3vae_trt.py` 中支持空间分块的优化路径（decoder whole compile + QK/RoPE attention-in-graph；encoder channels-last-3d + quant-conv-in-graph + whole compile），`stack_tiling=false`，encoder 返回 posterior mean
- TensorRT：`minimax_h3_vae_decoder_368_opt5.engine` + `minimax_h3_vae_encoder_672.engine`

这里没有把 `H3VAEPyOptRuntime` 的整帧 encoder 结果混进来：它当前的 fused encoder 只验证过单个 672×672 tile，对 1344×768 整帧会主动触发 int32 索引保护。上面的 PyTorch 路径逐个 672 tile 运行，和 TRT 的空间切分计划一致。

## 两次交换顺序复测

每一行都是同一进程内 7 次测量的 median；第二轮把 PyTorch 放到前面，以检查顺序偏差。

| 顺序 | 实现 | decoder | encoder | decode+encode |
| --- | --- | ---: | ---: | ---: |
| TRT → PyTorch | TensorRT | 13.756 s | 21.474 s | 35.230 s |
| TRT → PyTorch | 优化 PyTorch | 13.650 s | 20.701 s | 34.351 s |
| PyTorch → TRT | 优化 PyTorch | 13.644 s | 20.472 s | 34.116 s |
| PyTorch → TRT | TensorRT | 13.780 s | 21.461 s | 35.241 s |
| 两轮 median 平均 | TensorRT | **13.768 s** | **21.468 s** | **35.235 s** |
| 两轮 median 平均 | 优化 PyTorch | **13.647 s** | **20.586 s** | **34.233 s** |

按两轮 median 平均计算，优化 PyTorch 相对 TRT：

- decoder 快约 **0.88%**（约 0.121 s）
- encoder 快约 **4.11%**（约 0.881 s）
- decode+encode 总计快约 **2.84%**（约 1.002 s，PyTorch/TRT = `0.972×`）

两种顺序都得到相同方向，说明不是单纯由先后顺序造成的偶然反转。输出 shape 两边一致：decoder `(1,3,124,768,1344)`，encoder `(1,24,37,48,84)`。

## 结论边界

在当前 engine、当前 GPU 和这套同 tile 口径下，**没有测出 TRT 比优化 PyTorch 更快；优化 PyTorch 小幅胜出**。decoder 的优势只有约 1%，可视作基本持平；encoder 约 4%，使完整 VAE encode+decode 总体约 2–3% 优于 TRT。

本机测试期间仍有 ComfyUI/trpc 业务负载（GPU 利用率约 82–100%），所以这里的可靠结论是“方向和量级”，不是宣称固定的 2.84% 产品收益。发布前应在空闲 GPU 上交替运行更多轮，并补充同 tile 的像素/latent 误差和视频画质回归。

生产 ComfyUI PyOpt decoder 的 `12.028 s` 使用 tile `256`，不能直接与这里的 PyTorch `13.647 s` 或 TRT `13.768 s` 比较；原生 ComfyUI VAE 既有代表值约 `15.896 s`，同样只作方向性参考。

## 默认 PyTorch decode 基线

为补齐对比表，另外测量了未启用本项目优化的上游默认 PyTorch VAE。仍是 `1344×768×124`、同一 FP16 latent、`2` 次 warmup + `7` 次 CUDA Event；不启用 decoder 融合、`torch.compile`、channels-last 或 staged batch。decoder tile `368` 时 median 为 **19.643 s**，tile `256` 时为 **18.740 s**。因此严格同 tile 表应增加：默认 PyTorch `19.643 s`、优化 PyTorch `13.647 s`、TRT `13.768 s`；优化 PyTorch 和 TRT 相对默认分别少约 `30.5%` 和 `29.9%`。完整测量和与 ComfyUI 原生 `≈15.896 s` 日志的口径区别见 [`h3_default_pytorch_vae_decode_2026-09-16.md`](h3_default_pytorch_vae_decode_2026-09-16.md)。
