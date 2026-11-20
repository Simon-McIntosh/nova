"""Refuse a second implementation of a block this repair gave one owner.

The review of the first delivery scored reuse below its floor because the
measurements beside the rule rebuilt what the owners already carry.  Four
owners were named, and this check fails when any of them is copied again:

* the section-quadrature volume block belongs to
  :func:`nova.biot.tiledassembly.volume_mutual`;
* the closed octagonal gate mesh belongs to
  :func:`nova.biot.magnetisation.gate_shell`;
* the dense generalised eigenproblem belongs to
  :func:`nova.biot.magnetisation.galerkin_spectrum`, which the measurement may
  call but must not solve again;
* inside ``_polygon_current_response`` the caller's ``columns`` mask and the
  tile loop's index vector are different names.

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
OWNERS = {
    "volume_mutual": ROOT / "nova" / "biot" / "tiledassembly.py",
    "gate_shell": ROOT / "nova" / "biot" / "magnetisation.py",
    "galerkin_spectrum": ROOT / "nova" / "biot" / "magnetisation.py",
}


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


def main() -> int:
    findings = []
    for name, owner in OWNERS.items():
        if name not in definitions(owner):
            findings.append(f"{name} has no owner: {owner} does not define it")
        for path in sorted(SCRIPTS.glob("*.py")):
            if path.name == "reuse_check.py":
                continue
            if name in definitions(path) and path != owner:
                findings.append(f"{name} is defined again in {path}")
    measure = SCRIPTS / "measure.py"
    if "eigh" in (measure.read_text()) and "galerkin_spectrum" not in callers(measure):
        findings.append(
            f"{measure} solves the generalised eigenproblem itself instead of "
            "calling the owner's galerkin_spectrum"
        )
    tiled = OWNERS["volume_mutual"].read_text()
    body = tiled.split("def _polygon_current_response(", 1)[-1].split("\ndef ", 1)[0]
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
