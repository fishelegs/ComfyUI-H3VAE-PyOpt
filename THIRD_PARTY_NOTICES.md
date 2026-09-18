# Third-Party Notices

The source code authored for this repository is licensed under the
[MIT License](LICENSE).

This project interoperates with or can optionally use third-party software,
model code, and model assets that are distributed separately under their own
licenses. The MIT License in this repository does not relicense those
third-party components.

## Third-party components

- **ComfyUI** — this project is a ComfyUI custom-node integration and imports
  ComfyUI APIs at runtime. ComfyUI is distributed separately under its own
  license.
- **PyTorch** — runtime tensor and compilation framework, distributed
  separately under its own license.
- **Triton** — used for custom GPU kernels, distributed separately under its
  own license.
- **safetensors** — used to load model weights, distributed separately under
  its own license.
- **comfy-kitchen** — optional dependency used only when the experimental
  `fast_linear` path is enabled; distributed separately under its own
  license.
- **MiniMax H3 / FL2VA model code** — loaded from a user-provided external
  model-code directory and not included in this repository.
- **MiniMax H3 model weights and other model assets** — downloaded or supplied
  separately and not included in this repository. Their use is governed by
  the applicable MiniMax model license and any terms accompanying the assets.

If source code from a third-party project is added to this repository in the
future, its original copyright and license notices must be preserved and this
document must be updated when required.
