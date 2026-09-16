# MiniMax H3 默认 PyTorch VAE decode 基线

测试日期：2026-09-16

## 测试口径

这是当前 ComfyUI 管线尺寸 `1344×768×124`（24 fps，约 5.167 s）的**上游默认 PyTorch VAE** 独立测量，不启用本项目的融合、`torch.compile`、channels-last 或 staged batch 优化。两种 decoder tile 在同一个持久化环境和进程中测量：

- Python `3.11.13`、PyTorch `2.8.0+cu128`、FP16
- 上游 `MiniMaxH3VideoVAE`，同一随机 latent，输入 shape `(1,24,37,48,84)`
- 输出 shape `(1,3,124,768,1344)`
- `2` 次 warmup + `7` 次 CUDA Event；模型加载和首次初始化不计入稳态时间
- `stack_tiling=false`；没有 PyOpt 的 decoder 融合或 compile

原始 JSON：`/tmp/h3_default_vae_decode_1344x768x124.json`。测试时 GPU 仍有 ComfyUI/trpc 业务负载，因此数字用于工程基线，发布前应在空闲 GPU 上交替复测。

## 默认 PyTorch 结果

| decoder tile | mean | median | min–max | 输出 shape |
| ---: | ---: | ---: | ---: | --- |
| 368（与 TRT 同 tile） | 19.641 s | **19.643 s** | 19.629–19.649 s | `(1,3,124,768,1344)` |
| 256（生产常用 tile） | 18.744 s | **18.740 s** | 18.736–18.769 s | `(1,3,124,768,1344)` |

## 放入同 tile 对比表

为了和之前的 TRT A/B 使用完全相同的 decoder tile `368`，应使用上面的 `19.643 s`，而不是把不同实现的 ComfyUI 日志值混入同一行：

| 实现 | decoder tile | median decode | 相对默认 PyTorch |
| --- | ---: | ---: | ---: |
| 默认上游 PyTorch（无优化） | 368 | **19.643 s** | — |
| 优化 PyTorch（两次交换顺序 A/B 的 median 平均） | 368 | **13.647 s** | 低约 **30.5%** |
| TensorRT（两次交换顺序 A/B 的 median 平均） | 368 | **13.768 s** | 低约 **29.9%** |

在这个严格同 tile 表里，优化 PyTorch 比默认 PyTorch 少约 `5.996 s`，TRT 少约 `5.875 s`；优化 PyTorch 与 TRT 之间只有约 `0.121 s`（约 `0.88%`）差异，属于基本持平的量级。此前的严格结论“当前没有测出 TRT 胜过优化 PyTorch”仍然成立。

## 与当前 ComfyUI 原生日志的关系

正式 ComfyUI `8080` 的既有同尺寸原生 `nodes.VAEDecode` 记录约 **15.896 s**（日志范围约 `15.894–15.912 s`），这是 ComfyUI `0.33.0` / PyTorch `2.11.0+cu130` 的生产实现，使用 tile `256`。当前 ComfyUI PyOpt 热启动为 `12.028 s`，同样使用 tile `256`。

这个 `15.896 s` 可以作为“实际 ComfyUI 原生基线”放在生产管线表中，但不能和上面的 `19.643 s` 当成同一实现：独立基准使用上游模型和 PyTorch `2.8+cu128`，ComfyUI 原生代码、PyTorch/CUDA 版本和运行时调度不同。建议报告同时保留两张表：

1. **严格同 tile 表**：默认上游 PyTorch、优化 PyTorch、TRT，专门回答实现间的公平速度问题。
2. **生产 ComfyUI 表**：原生 `nodes.VAEDecode` `≈15.896 s`、PyOpt `12.028 s`，专门回答当前服务的实际收益。

