"""CPU structure checks for the executable experimental INT8 workflow."""
from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "examples" / "minimal_h3vae_pyopt_int8_prompt.json"
ROUNDTRIP_PROMPT_PATH = ROOT / "examples" / "minimal_h3vae_pyopt_int8_roundtrip_prompt.json"


def _links(value):
    if isinstance(value, list):
        if (
            len(value) == 2
            and isinstance(value[0], str)
            and isinstance(value[1], int)
        ):
            yield (value[0], value[1])
        else:
            for item in value:
                yield from _links(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _links(item)


class Int8PromptCpuTest(unittest.TestCase):
    def test_executable_four_node_prompt_and_links(self):
        document = json.loads(PROMPT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(set(document), {"prompt", "client_id"})
        self.assertEqual(document["client_id"], "h3vae-pyopt-int8-example")
        prompt = document["prompt"]
        self.assertEqual(set(prompt), {"1", "2", "3", "4"})
        self.assertEqual(prompt["1"]["class_type"], "H3VAEPyOptLoader")
        self.assertEqual(
            prompt["1"]["_meta"]["title"],
            "MiniMax H3 VAE Load (Experimental INT8 Decode)",
        )
        loader = prompt["1"]["inputs"]
        self.assertTrue(loader["int8_decode"])
        self.assertFalse(loader["fast_linear"])
        self.assertEqual(loader["dtype"], "fp16")
        self.assertEqual(prompt["2"]["class_type"], "EmptyMiniMaxH3LatentAV")
        self.assertEqual(prompt["3"]["class_type"], "VAEDecode")
        self.assertEqual(prompt["4"]["class_type"], "PreviewImage")

        for node_id, node in prompt.items():
            for source_id, output_index in _links(node["inputs"]):
                self.assertIn(
                    source_id,
                    prompt,
                    msg=f"node {node_id} references missing node {source_id}",
                )
                self.assertGreaterEqual(output_index, 0)

        self.assertEqual(prompt["3"]["inputs"]["samples"], ["2", 0])
        self.assertEqual(prompt["3"]["inputs"]["vae"], ["1", 0])
        self.assertEqual(prompt["4"]["inputs"]["images"], ["3", 0])

    def test_executable_roundtrip_prompt_has_both_int8_paths(self):
        document = json.loads(ROUNDTRIP_PROMPT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(set(document), {"prompt", "client_id"})
        self.assertEqual(
            document["client_id"], "h3vae-pyopt-int8-roundtrip-example"
        )
        prompt = document["prompt"]
        self.assertEqual(set(prompt), {"1", "2", "3", "4", "5"})
        self.assertEqual(prompt["1"]["class_type"], "H3VAEPyOptLoader")
        self.assertEqual(
            prompt["1"]["_meta"]["title"],
            "MiniMax H3 VAE Load (Experimental INT8 Encode + Decode)",
        )
        loader = prompt["1"]["inputs"]
        self.assertTrue(loader["int8_encode"])
        self.assertTrue(loader["int8_decode"])
        self.assertFalse(loader["fast_linear"])
        self.assertEqual(loader["dtype"], "fp16")
        self.assertEqual(prompt["2"]["class_type"], "LoadImage")
        self.assertEqual(
            prompt["2"]["inputs"]["image"], "replace_with_256x256.png"
        )
        self.assertEqual(prompt["3"]["class_type"], "VAEEncode")
        self.assertEqual(prompt["4"]["class_type"], "VAEDecode")
        self.assertEqual(prompt["5"]["class_type"], "PreviewImage")

        for node_id, node in prompt.items():
            for source_id, output_index in _links(node["inputs"]):
                self.assertIn(
                    source_id,
                    prompt,
                    msg=f"node {node_id} references missing node {source_id}",
                )
                self.assertGreaterEqual(output_index, 0)

        self.assertEqual(prompt["3"]["inputs"]["pixels"], ["2", 0])
        self.assertEqual(prompt["3"]["inputs"]["vae"], ["1", 0])
        self.assertEqual(prompt["4"]["inputs"]["samples"], ["3", 0])
        self.assertEqual(prompt["4"]["inputs"]["vae"], ["1", 0])
        self.assertEqual(prompt["5"]["inputs"]["images"], ["4", 0])


if __name__ == "__main__":
    unittest.main()
