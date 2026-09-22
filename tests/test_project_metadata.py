import tomllib
import unittest
from pathlib import Path


class ProjectMetadataTest(unittest.TestCase):
    def test_project_metadata(self):
        path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        project = data["project"]

        self.assertEqual(project["name"], "h3vae-pyopt")
        self.assertEqual(project["version"], "0.2.0")
        self.assertEqual(project["license"], {"file": "LICENSE"})
        self.assertEqual(
            project["urls"]["Repository"],
            "https://github.com/fishelegs/ComfyUI-H3VAE-PyOpt",
        )
        self.assertEqual(
            set(project["dependencies"]),
            {"torch>=2.8", "triton", "safetensors"},
        )
        comfy = data["tool"]["comfy"]
        self.assertEqual(comfy["PublisherId"], "fishelegs")
        self.assertEqual(comfy["DisplayName"], "ComfyUI-H3VAE-PyOpt")


if __name__ == "__main__":
    unittest.main()
