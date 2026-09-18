import tomllib
from pathlib import Path


def test_project_metadata():
    data = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]

    assert project["name"] == "comfyui-h3vae-pyopt"
    assert project["version"] == "0.1.0"
    assert project["license"] == {"file": "LICENSE"}
    assert project["urls"]["Repository"] == (
        "https://github.com/fishelegs/ComfyUI-H3VAE-PyOpt"
    )
    assert set(project["dependencies"]) == {
        "torch>=2.8",
        "triton",
        "safetensors",
    }
    assert "comfy" not in data.get("tool", {})
