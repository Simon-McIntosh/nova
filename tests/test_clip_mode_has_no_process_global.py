"""The support clip mode is owned by an operator or solve request."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_NAMES = {"set_support_clip_mode", "support_clip_mode", "_SUPPORT_CLIP_MODE"}
SCANNED_DIRECTORIES = ("nova", "benchmarks", "scripts", "tests")


def _references_process_global(path: Path) -> bool:
    """Return whether Python syntax names one retired process-global API."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, ast.Name)
        and node.id in FORBIDDEN_NAMES
        or isinstance(node, ast.Attribute)
        and node.attr in FORBIDDEN_NAMES
        or isinstance(node, ast.alias)
        and node.name in FORBIDDEN_NAMES
        for node in ast.walk(tree)
    )


def test_no_python_source_references_the_retired_process_global() -> None:
    """A known-present forbidden name makes this repository-wide scan fail."""

    self_path = Path(__file__).resolve()
    findings = sorted(
        path.relative_to(ROOT).as_posix()
        for directory in SCANNED_DIRECTORIES
        for path in (ROOT / directory).rglob("*.py")
        if path.resolve() != self_path and _references_process_global(path)
    )

    assert not findings, "retired process-global references:\n" + "\n".join(findings)
