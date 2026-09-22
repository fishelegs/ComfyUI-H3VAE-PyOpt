import ast
import unittest
from pathlib import Path


class VersionMetadataTest(unittest.TestCase):
    def test_literal_version_metadata(self):
        source = (Path(__file__).resolve().parents[1] / "__init__.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        versions = []

        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    versions.append(node.value)

        self.assertEqual(len(versions), 1)
        value = versions[0]
        self.assertIsInstance(value, ast.Constant)
        self.assertEqual(value.value, "0.2.0")


if __name__ == "__main__":
    unittest.main()
