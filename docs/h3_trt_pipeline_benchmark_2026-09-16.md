# MiniMax H3 TensorRT 全视频管线实测

测试日期：2026-09-16

这是在当前 ComfyUI PyOpt 管线使用的分辨率和时长下，对已有 TensorRT engine 的实际复测。测试只替换 VAE 对照路径；正式 ComfyUI `8080` 服务未修改。

## 环境

- 持久化环境：`/data/miniconda3/envs/h3_trt_11_2`
- Python `3.11.13`
- PyTorch `2.8.0+cu128`，Torch CUDA `12.8`
- TensorRT Python binding `11.2.1.2`
- NumPy `2.4.6`，safetensors `0.8.0`
- GPU：`NVIDIA Graphics Device`，driver `580.82.07`
- CUDA toolkit（`nvcc`）：`13.0.48`

环境曾先在 `/tmp/h3_trt_env` 完成安装和 engine smoke test，随后克隆到上述持久路径并再次验证 `torch.cuda.is_available()`、TensorRT import 和两个 engine 反序列化均成功。完整依赖清单见 [`environment/trt_requirements.txt`](../environment/trt_requirements.txt)。

## 测试口径

- 视频：`1344×768×124`（24 fps，约 5.167 s）
- FP16；decoder tile `368`，encoder tile `672`
- `2` 次 warmup、`7` 次 CUDA Event 测量；模型/engine 加载及首次初始化不计入稳态数字
- 每次完整视频：decoder 调用 `105` 次，encoder 调用 `48` 次
- engine：`minimax_h3_vae_decoder_368_opt5.engine`、`minimax_h3_vae_encoder_672.engine`
- 重现命令（路径按本机实际权重和 engine 修改）：

```bash
/data/miniconda3/envs/h3_trt_11_2/bin/python bench_h3vae_trt.py \
  --full-video --height 768 --width 1344 --frames 124 \
  --runs 7 --warmup 2 \
  --decoder-engine /mnt/gyfs_cq2/models/keyishen/trt/minimax_h3_vae_decoder_368_opt5.engine \
  --encoder-engine /mnt/gyfs_cq2/models/keyishen/trt/minimax_h3_vae_encoder_672.engine \
  --original-vae /mnt/gyfs_cq2/models/keyishen/MiniMaxAI/MiniMax-H3/FL2VA/video_vae \
  --decoder-tile-size 368 --encoder-tile-size 672 --skip-original
```

## 结果

| 阶段 | mean | median | min | max | 输出 shape |
| --- | ---: | ---: | ---: | ---: | --- |
| TRT decoder | 13.743 s | **13.750 s** | 13.707 s | 13.761 s | `(1,3,124,768,1344)` |
| TRT encoder | 21.467 s | **21.467 s** | 21.456 s | 21.473 s | `(1,24,37,48,84)` |

显存记录中的 Torch allocator 峰值为 `3.07 GiB`；它不包含 TensorRT engine/context/activation 的完整显存，不能当作 TRT 进程总显存。

## 与当前 ComfyUI 数字的关系

严格同 tile 的两轮交换顺序 A/B 已补测：优化 PyTorch（decoder 368、encoder 672）decode median 平均 `13.647 s`、encode `20.586 s`、合计 `34.233 s`；TRT 分别为 `13.768 s`、`21.468 s`、`35.235 s`。因此 PyTorch 总体快约 `2.84%`，decoder 约 `0.88%`、encoder 约 `4.11%`。详见 [`h3_same_tile_ab_benchmark_2026-09-16.md`](h3_same_tile_ab_benchmark_2026-09-16.md)。

- 当前 ComfyUI PyOpt 视频 VAE decode：`12.028 s`（decoder tile `256`、tile batch `2`、compile 开启）。在本次实测条件下，PyOpt 比 TRT decoder median `13.750 s` 快约 **12.5%**（TRT 为 PyOpt 的约 `1.143×`）。
- 同版本原生 ComfyUI VAE decode 的既有代表值：约 `15.896 s`；该数字与本次 TRT 的 tile/实现口径不同，只能作方向性参考。
- 这不是完全同 tile 的严格 A/B：当前 engine 是静态 decoder tile `368`，PyOpt 管线使用 tile `256`；tile 会改变调用次数、边界 padding、RoPE 坐标和输出。要发布最终结论，应在空闲 GPU、同一 tile、同一输入及交替顺序下重新测量，并做画质回归。

本次机器上同时存在 ComfyUI/trpc 业务负载（运行期间 GPU 非空闲），因此结果适合作为当前可用 engine 的工程基线；小于约 1–2% 的差异不应据此下结论。

原始机器生成的 JSON/Markdown 暂存于 `/tmp/h3_trt_1344x768x124_results.json` 和 `/tmp/h3_trt_1344x768x124_report.md`；本文件修正了生成报告中继承的 `conda_default_env` 字段，并保留可复现参数。
