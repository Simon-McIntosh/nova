"""Prove the Jacobian check rejects a missing booking derivative."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile

PYTHON = Path("/home/ITER/mcintos/Code/nova/.venv/bin/python")
ROOT = Path(__file__).resolve().parents[3]


def archive(revision: str, destination: Path) -> None:
    command = [
        "git",
        "-C",
        str(ROOT),
        "archive",
        revision,
        "nova",
        "tests",
        "scripts",
        "benchmarks",
        "conftest.py",
        "pyproject.toml",
        "tox.ini",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    assert process.stdout is not None
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive_file:
        archive_file.extractall(destination, filter="data")
    if process.wait() != 0:
        raise RuntimeError("the scratch archive did not complete")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(
        prefix="nova-negative-", dir=os.environ["TMPDIR"]
    ) as scratch:
        tree = Path(scratch) / "source"
        tree.mkdir()
        archive("HEAD", tree)
        path = tree / "nova/equilibrium/support_read.py"
        source = path.read_text()
        original = (
            "value - change for value, change in zip(direction, tangent, strict=True)"
        )
        mutated = "value for value, change in zip(direction, tangent, strict=True)"
        if source.count(original) != 1:
            raise RuntimeError("the booking-derivative expression changed shape")
        path.write_text(source.replace(original, mutated))
        target = (
            "tests/test_bank_newton_step.py::"
            "test_bank_jvp_composes_flux_and_level_derivatives"
        )
        command = [
            str(PYTHON),
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(Path(scratch) / "pytest"),
            target,
            "-q",
        ]
        environment = os.environ.copy()
        environment.update(TMPDIR=scratch, PYTHONPATH=str(tree), JAX_PLATFORMS="cpu")
        with args.log.open("w") as log:
            header = f"REVISION={revision} TREE={tree} CWD={tree}"
            print(
                f"{header} COMMAND={shlex.join(command)}",
                file=log,
                flush=True,
            )
            print(
                "MUTATION=drop the booking-derivative term in a scratch copy; "
                "the JVP check fails",
                file=log,
                flush=True,
            )
            subprocess.run(
                [
                    str(PYTHON),
                    "-c",
                    "import nova, nova.equilibrium.support_read as m; "
                    "print('NOVA='+nova.__file__+' MODULE='+m.__file__)",
                ],
                cwd=tree,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
            result = subprocess.run(
                command,
                cwd=tree,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            print(f"EXIT={result.returncode}", file=log, flush=True)
    return 0 if result.returncode != 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
