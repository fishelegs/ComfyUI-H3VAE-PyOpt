# ComfyUI Registry Publication

This repository now contains the standard project metadata needed for Registry
packaging in `pyproject.toml`, plus `.comfyignore` rules for excluding
development-only files from the published archive.

Publication is intentionally not enabled yet because the Registry publisher
identity must be a real publisher ID owned by the maintainer. During this
readiness pass, no public evidence of an existing `fishelegs` publisher was
found, so the repository does not guess or reserve that identity.

## Final publisher step

After creating or confirming the publisher in the ComfyUI Registry, add the
following section to `pyproject.toml` using the exact publisher ID shown on
the Registry profile:

```toml
[tool.comfy]
PublisherId = "<the exact publisher id owned by the maintainer>"
DisplayName = "ComfyUI-H3VAE-PyOpt"
Icon = ""
```

The placeholder above is documentation only. Do not commit it verbatim to
`pyproject.toml`.

The publisher ID is an account/Registry identity, not simply a repository
owner string. It should only be added after ownership is confirmed.

## Current package metadata

The current `pyproject.toml` records:

- package name: `comfyui-h3vae-pyopt`
- version: `0.1.0`
- license file: `LICENSE`
- repository URL
- Python requirement: `>=3.10`
- runtime dependencies matching `requirements.txt`:
  - `torch>=2.8`
  - `triton`
  - `safetensors`
- current validated platform metadata: Linux and NVIDIA CUDA

The MiniMax H3 / FL2VA model code and weights are not included in the package
and are not relicensed by this project.

## Publishing workflow

Once a publisher ID exists:

1. Add the verified `[tool.comfy]` section to `pyproject.toml`.
2. Validate that the Registry package metadata matches the release version.
3. Create or obtain a Comfy Registry API token using the official publishing
   flow.
4. Publish only from a reviewed release commit.
5. Keep the Registry version synchronized with `__version__`, the changelog,
   and the Git tag.

Do not store Registry access tokens in the repository.

## Official documentation

- Publishing custom nodes:
  https://docs.comfy.org/registry/publishing
- `pyproject.toml` specification:
  https://docs.comfy.org/registry/specifications
- Registry overview:
  https://docs.comfy.org/registry/overview
