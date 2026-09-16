# ComfyUI H3VAE PyOpt 管线实测

测试日期：2026-09-16

## 测试口径

- ComfyUI `0.33.0`
- PyTorch `2.11.0+cu130`，Python `3.12.14`
- NVIDIA Graphics Device，FP16 VAE、BF16 UNet/text encoder
- 1344×768，124 帧，24 fps，`res_multistep`，20 steps
- 独立 ComfyUI `8081` 测试服务；正式 `8080` 未改动
- workflow 仅将节点 119 从原生 `VAELoader` 换成 `H3VAEPyOptLoader`
- PyOpt：decoder tile 256、tile batch 2、encoder staged batch 4、decoder/encoder compile、cuDNN benchmark
- 先跑 1 step 冷启动预热，再跑 1 次 20-step 热启动；热启动数字不包含首次加载/编译

## 结果

| 阶段 | 时间 |
| --- | ---: |
| 1 step 冷启动完整 workflow | 125.956 s |
| 冷启动 PyOpt loader | 17.172 s |
| 冷启动 sampler | 48.967 s |
| 冷启动视频 VAE decode | 19.023 s |
| 20 step 热启动完整 workflow | **333.860 s** |
| 20 step 热启动 sampler | **317.504 s** |
| 20 step 热启动视频 VAE decode | **12.028 s** |
| 20 step 热启动音频 VAE decode | 0.672 s |
| 20 step 热启动 VideoCombine | 2.402 s |

相对既有同版本默认管线代表值（约 332.86 s workflow、317.30 s sampler、12.07 s 视频 VAE decode），本次 PyOpt 结果分别约慢 1.00 s、0.20 s、快 0.04 s；差异在当前 GPU 有其他业务负载时属于噪声级，未观察到明显回归。主要瓶颈仍是 DiT sampler，约占热启动 workflow 的 95.1%。

## 输出校验

输出文件：`/tmp/h3_comfyui_test/output/h3_cudnn_ab/pyopt_current_27003_s20_00001-audio.mp4`

- H.264，1344×768，124 帧，24 fps
- 容器时长 5.167 s，包含 AAC 音频
- 文件大小 2,609,388 bytes
- 8081 任务完成后队列为空；8081 已停止，8080 仍在线且队列为空

完整请求、history、节点计时和结果 JSON 保存在 `/tmp/h3_comfyui_test/artifacts/`，可用 `bench_h3_cudnn_ab.py` 重跑。由于本次没有在同一 8081 进程再跑原生 VAELoader，性能判断使用之前同版本默认基线；若要发表严格 A/B 数字，应在空闲 GPU、同一进程交替重复 stock/PyOpt，并报告中位数和画质误差。
