"""Regression check for the package-level SDPA compatibility policy."""
import ast
import unittest
from pathlib import Path


class SDPABackendDefaultTest(unittest.TestCase):
    def test_default_is_auto_and_preserves_explicit_environment_value(self):
        source = (Path(__file__).resolve().parents[1] / "__init__.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        defaults = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "setdefault":
                continue
            if len(node.args) != 2:
                continue
            key, value = node.args
            if isinstance(key, ast.Constant) and key.value == (
                "MINIMAX_H3_TORCH_SDPA_BACKEND"
            ):
                defaults.append(value)

        self.assertEqual(len(defaults), 1)
        self.assertIsInstance(defaults[0], ast.Constant)
        self.assertEqual(defaults[0].value, "auto")


if __name__ == "__main__":
    unittest.main()
