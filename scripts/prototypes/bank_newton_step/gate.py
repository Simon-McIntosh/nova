"""Run a focused test gate with a bounded node-local pytest scratch tree."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("target")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    python = Path("/home/ITER/mcintos/Code/nova/.venv/bin/python")
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="nova-gate-", dir=os.environ["TMPDIR"]
    ) as scratch:
        command = [
            str(python),
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(Path(scratch) / "pytest"),
            args.target,
            "-q",
        ]
        environment = os.environ.copy()
        environment.update(
            TMPDIR=scratch,
            PYTHONPATH=str(root),
            JAX_PLATFORMS="cpu",
        )
        module_command = [
            str(python),
            "-c",
            "import nova, nova.equilibrium.clip_quadrature as m; "
            "print('NOVA='+nova.__file__+' MODULE='+m.__file__)",
        ]
        with args.log.open("w") as handle:
            header = f"REVISION={revision} TREE={root} CWD={root}"
            print(
                f"{header} COMMAND={shlex.join(command)}",
                file=handle,
                flush=True,
            )
            subprocess.run(
                module_command,
                cwd=root,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
            result = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            print(f"EXIT={result.returncode}", file=handle, flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
