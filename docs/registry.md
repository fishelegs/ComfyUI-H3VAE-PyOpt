# ComfyUI Registry Publication

The repository is configured for ComfyUI Registry publication.

## Registry identity

- Publisher ID: `fishelegs`
- Node ID: `h3vae-pyopt`
- Display name: `ComfyUI-H3VAE-PyOpt`
- Version: `0.1.0`

The node ID is intentionally shorter than the GitHub repository name because
ComfyUI's Registry guidance recommends not including `ComfyUI` in
`[project].name`. The GitHub repository name does not need to change.

The MiniMax H3 / FL2VA model code and weights are not included in the package
and are not relicensed by this project.

## API key handling

Registry publishing keys are secrets. Never commit a key to this repository,
paste it into an issue or pull request, or place it directly in a workflow
file.

For GitHub Actions, create a repository secret named:

```text
REGISTRY_ACCESS_TOKEN
```

and store the Registry publishing key as its value.

GitHub path:

```text
Settings
→ Secrets and variables
→ Actions
→ Repository secrets
→ New repository secret
```

The workflow at `.github/workflows/publish_action.yml` references the secret
as `${{ secrets.REGISTRY_ACCESS_TOKEN }}`. It is intentionally
`workflow_dispatch`-only so publishing requires an explicit manual action.

## Publish with GitHub Actions

After the secret exists:

1. Open the repository's **Actions** tab.
2. Select **Publish to Comfy Registry**.
3. Click **Run workflow**.
4. Run it from `main`.
5. Confirm the workflow completes successfully.
6. Verify the published node in the ComfyUI Registry under publisher
   `@fishelegs`.

Do not bump the version merely to retry a failed workflow unless the Registry
has already accepted that version.

## Publish from a local CLI

Alternatively, install the Comfy CLI and run from the repository root:

```bash
comfy node publish
```

The CLI will prompt for the Registry API key interactively. The key should not
be stored in the repository.

## Versioning

Registry versions must remain synchronized with:

- `__version__`
- `pyproject.toml`
- `CHANGELOG.md`
- the corresponding Git tag / GitHub Release

Before publishing a new version, update these version references together and
run CI.

## Package contents

`.comfyignore` excludes development-only files from the Registry archive.
Runtime source code, README, license files, requirements, examples, and other
required package files remain available.

## Official documentation

- Publishing custom nodes:
  https://docs.comfy.org/registry/publishing
- `pyproject.toml` specification:
  https://docs.comfy.org/registry/specifications
- Registry overview:
  https://docs.comfy.org/registry/overview
