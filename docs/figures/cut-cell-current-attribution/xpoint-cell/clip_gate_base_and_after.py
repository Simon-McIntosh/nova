"""Run the clip gate at the base revision and at the working head, from one allocation.

The gate is the two test modules that reach the changed oracle: the committed
cell-wedge clip cases and the forward census that imports the oracle module.  The
base revision's copy of each is materialised into the allocation's own scratch
directory, so the two runs differ only in the revision under test.

Neither run is path-filtered by marker, so the modules' own selections apply.
The driver prints one summary line per run and exits non-zero if the working head
adds a failure against the base.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[4]
assert (WORKTREE / "pyproject.toml").exists(), f"not the repository root: {WORKTREE}"
BASE_REVISION = "7a869bc1a"
GATE_MODULES = ("tests/test_xpoint_cell_wedge_clip.py", "tests/test_forward_census.py")
PYTHON = sys.executable


def _materialise(module: str, destination: Path) -> Path:
    source = subprocess.run(
        ["git", "-C", str(WORKTREE), "show", f"{BASE_REVISION}:{module}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    target = destination / Path(module).name
    target.write_text(source, encoding="utf-8")
    return target


def _run(targets: list[Path], log: Path) -> tuple[int, list[str]]:
    environment = dict(os.environ, TMPDIR="/tmp", JAX_PLATFORMS="cpu")
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "-q",
                *[str(target) for target in targets],
            ],
            cwd=WORKTREE,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    text = log.read_text(encoding="utf-8")
    failures = sorted(re.findall(r"^(?:FAILED|ERROR) (\S+)", text, flags=re.MULTILINE))
    summary = next(
        (line for line in reversed(text.splitlines()) if re.search(r"\d+ (passed|failed)", line)),
        "no summary line",
    )
    print(f"{log.name}: exit {completed.returncode} | {summary}")
    for failure in failures:
        print(f"  {failure}")
    return completed.returncode, failures


def _key(failures: list[str]) -> set[str]:
    """Identify a failure by its module basename and node id, not its filesystem path.

    The base run collects its modules from the allocation's scratch directory, so
    the two runs spell the same node id with different prefixes.  Comparing the
    raw strings reports every base failure as added, which is a defect in the
    comparison rather than a fact about the revision.
    """
    return {
        f"{Path(failure.split('::', 1)[0]).name}::{failure.split('::', 1)[1]}"
        if "::" in failure
        else Path(failure).name
        for failure in failures
    }


def main() -> int:
    print(f"base revision: {BASE_REVISION}")
    scratch = Path("/tmp/cca-clip-gate")
    scratch.mkdir(parents=True, exist_ok=True)
    records = Path(__file__).resolve().parent
    base_targets = [_materialise(module, scratch) for module in GATE_MODULES]
    head_targets = [Path(module) for module in GATE_MODULES
                    if (WORKTREE / module).exists()]
    _, base_failures = _run(base_targets, records / "clip-gate-base.log")
    _, head_failures = _run(head_targets, records / "clip-gate-after.log")
    added = sorted(_key(head_failures) - _key(base_failures))
    print(f"added failures against the base revision: {len(added)}")
    for failure in added:
        print(f"  ADDED {failure}")
    return 1 if added else 0


if __name__ == "__main__":
    raise SystemExit(main())