import ast
from pathlib import Path


def test_literal_version_metadata():
    source = Path("__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    versions = []

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__version__":
                versions.append(node.value)

    assert len(versions) == 1
    value = versions[0]
    assert isinstance(value, ast.Constant)
    assert value.value == "0.1.0"
