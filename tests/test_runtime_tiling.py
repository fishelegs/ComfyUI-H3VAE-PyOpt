"""CPU-only regression checks for the reference-compatible tile planner."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h3vae_runtime import H3VAEPyOptRuntime  # noqa: E402


class EncoderTilingTest(unittest.TestCase):
    def test_auto_selects_measured_tile_sizes(self):
        choose = H3VAEPyOptRuntime._select_encoder_tile
        self.assertEqual(choose(0, 672, 672), 672)
        self.assertEqual(choose(0, 768, 1344), 256)
        self.assertEqual(choose(672, 768, 1344), 672)

    def test_split_matches_reference_layout(self):
        split = H3VAEPyOptRuntime._split_encoder_tiles
        self.assertEqual(split(672, 672, 64, 16), ([0], [672], []))
        self.assertEqual(split(768, 256, 64, 16),
                         ([0, 160, 336, 512], [256] * 4, [96, 80, 80]))
        self.assertEqual(split(1344, 256, 64, 16),
                         ([0, 176, 352, 528, 704, 896, 1088],
                          [256] * 7, [80, 80, 80, 80, 64, 64]))

    def test_rejects_tile_no_larger_than_overlap(self):
        with self.assertRaises(ValueError):
            H3VAEPyOptRuntime._split_encoder_tiles(768, 64, 64, 16)


if __name__ == "__main__":
    unittest.main()
