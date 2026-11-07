"""Run the six JVP identity rows at the base and at the head revision.

Those rows exceed the suite's configured per-test bound on a compile-bound
host, so the bound is raised on this command line only and every row's own
outcome and wall is recorded per arm.  The base arm resolves its source through
the same overlay the gate uses, so the two arms differ only in the code under
test.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

IDS = tuple(
    f"tests/biot/test_edge_term_tangents.py::{name}[{case}]"
    for name in (
        "test_tangent_matches_base_jvp",
        "test_truncated_tangent_fails_identity",
    )
    for case in ("edge_integrands", "flux_and_moment_terms", "edge_terms")
)
BASE_FILES = ("gradedresidual.py", "polygonanalytic.py")
PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"


def _base_overlay(root: Path, revision: str, directory: Path) -> Path:
    overlay = directory / "overlay"
    package = overlay / "nova" / "biot"
    package.mkdir(parents=True)
    for name in BASE_FILES:
        source = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{revision}:nova/biot/{name}"]
        )
        (package / name).write_bytes(source)
    (directory / "sitecustomize.py").write_text(
        "import nova\n"
        f"nova.__path__.insert(0, {str(overlay / 'nova')!r})\n"
        "import nova.biot\n"
        f"nova.biot.__path__.insert(0, {str(package)!r})\n",
        encoding="utf-8",
    )
    return directory


def _rows(log_path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped.startswith(("PASSED ", "FAILED ", "ERROR ")):
            continue
        status, rest = stripped.split(" ", 1)
        node = rest.split(" ")[0]
        rows[node] = {"status": status.lower()}
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--arm", choices=("base", "head", "both"), default="both")
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    args.log_root.mkdir(parents=True, exist_ok=True)
    head_revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    receipt: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(
        prefix="polygon-identity-", dir=os.environ["TMPDIR"]
    ) as directory:
        scratch = Path(directory)
        overlay = _base_overlay(root, args.base_revision, scratch)
        arms = (
            (("base", args.base_revision), ("head", head_revision))
            if args.arm == "both"
            else (
                (args.arm, args.base_revision if args.arm == "base" else head_revision),
            )
        )
        for arm, revision in arms:
            env = os.environ.copy()
            env["JAX_PLATFORMS"] = "cpu"
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONPATH"] = os.pathsep.join(
                [str(overlay), str(root)] if arm == "base" else [str(root)]
            )
            log_path = args.log_root / f"identity-{arm}.log"
            command = [
                PYTHON,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "-v",
                "--durations=0",
                f"--timeout={args.timeout}",
                "--basetemp",
                str(scratch / f"pytest-{arm}"),
                *((args.only,) if args.only else IDS),
            ]
            with log_path.open("w", encoding="utf-8") as log:
                log.write(
                    f"# revision: {revision} arm: {arm} tree: {root} "
                    f"command: {shlex.join(command)}\n"
                )
                log.flush()
                result = subprocess.run(
                    command,
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                log.write(f"EXIT={result.returncode}\n")
            rows = _rows(log_path)
            receipt[arm] = {
                "revision": revision,
                "command": shlex.join(command),
                "exit_status": result.returncode,
                "log_path": str(log_path),
                "rows": rows,
            }
            print(
                f"{arm.upper()} EXIT={result.returncode} ROWS={json.dumps(rows)}",
                flush=True,
            )
            (args.log_root / "identity-receipt.json").write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
