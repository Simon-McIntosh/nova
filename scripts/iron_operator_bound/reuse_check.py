"""Refuse a second implementation of a block this repair gave one owner.

The review of the first delivery scored reuse below its floor because the
measurements beside the rule rebuilt what the owners already carry.  Four
owners were named, and this check fails when any of them is copied again:

* the section-quadrature volume block belongs to
  :func:`nova.biot.tiledassembly.volume_mutual`;
* the closed octagonal gate mesh belongs to
  :func:`nova.biot.magnetisation.gate_shell`;
* the dense generalised eigenproblem — its block metric, its ``1/mu0``
  normalisation and its solve — belongs to
  :func:`nova.biot.magnetisation.galerkin_block_spectrum`, which
  :func:`~nova.biot.magnetisation.galerkin_spectrum` and every study call;
* inside ``_polygon_current_response`` the caller's ``columns`` mask and the
  tile loop's index vector are different names.

The scan covers the whole study tree (``scripts/iron_operator_bound``,
recursively) and ``tests/``, because a copy in a test is the same defect as a
copy in a script: the test would then measure its own block.

    python scripts/iron_operator_bound/reuse_check.py

Exit status 0 when every owner stands alone, 1 with one line per copy found.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "iron_operator_bound"
TESTS = ROOT / "tests"
TILED = ROOT / "nova" / "biot" / "tiledassembly.py"
MAGNETISATION = ROOT / "nova" / "biot" / "magnetisation.py"
OWNERS = {
    "volume_mutual": TILED,
    "gate_shell": MAGNETISATION,
    "galerkin_spectrum": MAGNETISATION,
    "galerkin_block_spectrum": MAGNETISATION,
}
SOLVE = re.compile(r"\b(eigh|eigvalsh|block_diag)\s*\(")
GATE = re.compile(r"buffer\(\s*0\.(?:28|4)\s*,\s*2\s*\)")


def definitions(path: Path) -> set[str]:
    """The module-level and nested function names a file defines."""
    tree = ast.parse(path.read_text())
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def callers(path: Path) -> set[str]:
    """The attribute names a file calls as methods on a module alias (``x.y(...)``)."""
    tree = ast.parse(path.read_text())
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
    }


def scanned() -> list[Path]:
    """The files the check reads: the study tree and the test tree."""
    return sorted(SCRIPTS.rglob("*.py")) + sorted(TESTS.rglob("*.py"))


def main() -> int:
    findings = []
    for name, owner in OWNERS.items():
        if name not in definitions(owner):
            findings.append(f"{name} has no owner: {owner} does not define it")
        for path in scanned():
            if path.name == "reuse_check.py" or path == owner:
                continue
            if name in definitions(path):
                findings.append(f"{name} is defined again in {path}")
    for path in sorted(SCRIPTS.rglob("*.py")):
        if path.name == "reuse_check.py":
            continue
        text = path.read_text()
        if SOLVE.search(text) and "galerkin_block_spectrum" not in callers(path):
            findings.append(
                f"{path} builds a block metric or solves the generalised "
                "eigenproblem itself instead of calling the owner's "
                "galerkin_block_spectrum"
            )
        if GATE.search(text):
            findings.append(
                f"{path} rebuilds the octagonal gate region inline instead of "
                "calling magnetisation.gate_shell()"
            )
    for path in sorted(TESTS.rglob("*.py")):
        if GATE.search(path.read_text()):
            findings.append(
                f"{path} rebuilds the octagonal gate region inline instead of "
                "calling magnetisation.gate_shell()"
            )
    body = TILED.read_text().split("def _polygon_current_response(", 1)[-1]
    body = body.split("\ndef ", 1)[0]
    if re.search(r"(?<!tile_)columns = np\.arange", body):
        findings.append(
            "nova/biot/tiledassembly.py:_polygon_current_response rebinds the "
            "caller's columns mask as the tile loop's index vector"
        )
    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
