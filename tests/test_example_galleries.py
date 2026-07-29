"""Static import checks for data-backed example galleries.

The examples are scripts rather than library modules: importing them executes
meshing, opens remote catalogues, or loads local training data.  These checks
therefore validate their Python import graph without executing the workloads.
"""

from __future__ import annotations

import ast
from pathlib import Path
import tomllib

import pytest

ROOT = Path(__file__).parents[1]
GALLERIES = ("fenics", "mast")
EXAMPLES = [
    path
    for gallery in GALLERIES
    for path in sorted((ROOT / "examples" / gallery).glob("*.py"))
]


@pytest.mark.parametrize("gallery", GALLERIES)
def test_gallery_has_optional_dependency_group(gallery):
    """Each data-backed gallery has an explicitly requested dependency group."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert gallery in project["project"]["optional-dependencies"]


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.stem)
def test_gallery_source_compiles(path):
    """Every gallery script has a valid Python import graph."""
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    compile(tree, str(path), "exec")

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if not node.module.startswith("nova."):
            continue
        module_path = ROOT.joinpath(*node.module.split(".")).with_suffix(".py")
        package_path = ROOT.joinpath(*node.module.split("."), "__init__.py")
        assert module_path.exists() or package_path.exists(), node.module
