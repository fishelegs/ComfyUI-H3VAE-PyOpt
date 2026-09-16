# ComfyUI-H3VAE-PyTorch

MiniMax H3 视频 VAE 的 PyTorch encode/decode 优化项目，包含 ComfyUI loader、独立全视频基准和旧 TensorRT VAE 对照。**仓库不包含 MiniMax 模型源码、权重或 TensorRT engine。**

## 当前结论

历史同轮 672×672 FP16、124/243 帧测试里，旧优化 PyTorch 路径的 decode+encode 总耗时分别比原 TensorRT 路径低 **1.69% / 1.58%**。在当前 ComfyUI 使用的 1344×768×124 视频上，按 decoder 368、encoder 672 完全同 tile、交换先后顺序复测两轮后，优化 PyTorch 的 decode+encode median 平均为 **34.233 s**，TRT 为 **35.235 s**，PyTorch 小幅快约 **2.84%**；decoder 基本持平（约 0.88%），encoder 约快 4.11%。完整结果见 `docs/h3_same_tile_ab_benchmark_2026-09-16.md`。生产 ComfyUI PyOpt 的 12.028 s 使用 tile 256，不能直接套用到该同 tile 数字。

同一尺寸下，未启用优化的上游默认 PyTorch decoder 在 tile 368 时为 **19.643 s**，可作为严格同 tile 表的 baseline；优化 PyTorch 为 **13.647 s**，TRT 为 **13.768 s**。默认基线与当前 ComfyUI 原生日志的 `≈15.896 s` 不属于同一运行栈，分别见 [`docs/h3_default_pytorch_vae_decode_2026-09-16.md`](docs/h3_default_pytorch_vae_decode_2026-09-16.md)。

本项目保留两条 PyTorch 路径：

- `h3vae_runtime.py` + `opt/`：ComfyUI 现用的 PyOpt 运行时，decoder QK RMSNorm/RoPE 融合、whole-decoder compile、空间 tile batch；encoder GN/SiLU/padding 融合、channels-last-3d、分阶段 clip batch 和 compile。
- `pytorch_decoder_optim.py`、`pytorch_encoder_optim.py` + `bench_h3vae_trt.py`：旧 672 FP16 历史对照的优化实现与原 benchmark。它们作为可复现的 legacy 基线保留。

## 外部依赖

需要 CUDA、PyTorch（历史结果为 2.8；当前 ComfyUI 为 2.11）、Triton、safetensors、MiniMax H3 `FL2VA/video_vae` 目录和匹配的 VAE FP16 权重。TRT 对照另需与 engine 版本兼容的 TensorRT Python binding；不能用新版不兼容 runtime 直接加载旧 engine。请遵守 MiniMax H3 原许可，不要把模型代码/权重/engine 提交到本仓库。

已验证的 TensorRT 对照环境持久化在 `/data/miniconda3/envs/h3_trt_11_2`，依赖清单见 `environment/trt_requirements.txt`。若需重建，可使用 Python 3.11.13、PyTorch 2.8.0+cu128 和 TensorRT 11.2.1.2；TensorRT binding 必须与 engine 兼容。

设置：

```bash
export H3_VAE_MODEL_CODE_DIR=/path/to/MiniMax-H3/FL2VA/video_vae
export H3_VAE_WEIGHTS_PATH=/path/to/minimax_h3_video_vae_fp16.safetensors
export TORCHINDUCTOR_CACHE_DIR=/path/to/writable/inductor-cache
```

将此目录放进 ComfyUI `custom_nodes/` 后，用 `MiniMax H3 VAE Load (PyTorch Optimized)` 替换原 `VAE Loader`，后接原有的 `VAE Encode` / `VAE Decode`。保持 `decoder_tile_size=256` 才能与现有 ComfyUI 默认画质口径对比；672×672 的旧 TRT engine 则使用 decoder tile 368、encoder tile 672。不同 tile 的 RoPE 坐标会改变输出，不能只比较速度。

## 新 PyOpt vs TRT 全视频基准

下面是 672×672×124 的同输入热启动基准。脚本使用 FP16、2 次预热、7 次 CUDA Event，分别报告 encode/decode 的样本、median、质量误差、版本和文件 SHA-256；模型加载、compile 首次开销和 engine 反序列化不计入稳态耗时。请在空闲 GPU 上运行，换序重跑以确认 1–2% 级别的差异，并用真实视频做画质回归。

```bash
python bench_pyopt_vs_trt.py \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" \
  --weights "$H3_VAE_WEIGHTS_PATH" \
  --decoder-engine /path/to/minimax_h3_vae_decoder_368_opt5.engine \
  --encoder-engine /path/to/minimax_h3_vae_encoder_672.engine \
  --height 672 --width 672 --frames 124 \
  --decoder-tile 368 --encoder-tile 672 --warmup 2 --runs 7
```

输出默认在 `results/pyopt_vs_trt.json`。`--frames 243` 用于第二个历史口径；`--pyopt-only` 可在没有 TensorRT 的环境中单测 PyOpt。PyOpt encoder 接收 `[-1,1]`，TRT encoder 接收 `[0,1]`，脚本从同一份像素张量转换。TRT 的 Torch allocator 数字**不**覆盖 engine/context/activation 的全部显存；不能据此断言它显存更低。

无需 GPU 的基础测试：`python -m unittest discover -s tests -v`。

历史 benchmark 的原命令与详细结果见 `docs/h3_comfyui_results.md` 指向的旧报告；当前尺寸 TRT 实测见 `docs/h3_trt_pipeline_benchmark_2026-09-16.md`。新旧脚本都要求使用同一权重生成的 TRT engines。若新的 PyOpt 672 路径不比 TRT 快，应先定位 encoder staged batch 在该 shape 的收益/退化，再选择经过画质验证的优化组合；不能通过改 tile 或降低精度偷换比较口径。

## 发布注意

本地项目尚未发布到 GitHub。公开或跨地域分发前需要核对 [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) 的地域与衍生作品条款；请勿默认将其视为普通 MIT/Apache 开源许可。项目未代用户选择 GitHub 账号、仓库可见性或新代码授权。
