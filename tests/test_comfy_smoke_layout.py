"""CPU contract for the Comfy IMAGE layout used by the INT8 smoke bench."""

from __future__ import annotations

import unittest

import torch

from bench_int8_comfy_smoke import _raw_normalized_input


class ComfySmokeLayoutTests(unittest.TestCase):
    def test_helper_matches_standard_comfy_image_transform(self) -> None:
        for frames in (1, 17):
            for rgb_dtype in (torch.float16, torch.float32):
                with self.subTest(frames=frames, rgb_dtype=str(rgb_dtype)):
                    generator = torch.Generator(device="cpu").manual_seed(700 + frames)
                    rgb = torch.rand(
                        (frames, 16, 32, 3), generator=generator, dtype=rgb_dtype
                    )
                    before = rgb.clone()

                    # Mirror Comfy's 4-D IMAGE transform and process_input
                    # without forcing contiguous().
                    expected = rgb.movedim(-1, 1)
                    expected = expected.movedim(1, 0).unsqueeze(0)
                    expected = (expected * 2.0 - 1.0).to(torch.float16)
                    actual = _raw_normalized_input(rgb)

                    self.assertTrue(torch.equal(rgb, before))
                    self.assertEqual(actual.dtype, torch.float16)
                    self.assertEqual(tuple(actual.shape), (1, 3, frames, 16, 32))
                    self.assertEqual(tuple(actual.stride()), tuple(expected.stride()))
                    self.assertTrue(actual.is_contiguous(memory_format=torch.channels_last_3d))
                    self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
