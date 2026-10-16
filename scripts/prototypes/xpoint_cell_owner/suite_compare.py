"""Run the same focused conic tests against a baseline and the working tree."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


TEST_PATH = Path("scripts/prototypes/xpoint_cell_owner/test_conic_integral.py")
BASE_PATHS = (
    Path("conftest.py"),
    Path("pyproject.toml"),
    Path("nova/__init__.py"),
    Path("nova/jax/__init__.py"),
    Path("nova/jax/config.py"),
    Path("scripts/__init__.py"),
)


def _copy_base(repository, revision, destination):
    for relative in BASE_PATHS:
        content = subprocess.check_output(
            ["git", "-C", str(repository), "show", f"{revision}:{relative}"]
        )
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    target = destination / TEST_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repository / TEST_PATH, target)


def _run(repository, revision, arm, log, scratch):
    if arm == "base":
        working = scratch / "base"
        _copy_base(repository, revision, working)
    else:
        working = repository
    environment = os.environ.copy()
    environment.update(
        JAX_PLATFORMS="cpu", PYTHONPATH=str(working), TMPDIR=str(scratch)
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        f"--basetemp={scratch / 'pytest'}",
        str(TEST_PATH),
    ]
    with log.open("w") as stream:
        stream.write(
            f"revision={revision} tree={working} command=" + " ".join(command) + "\n"
        )
        stream.write(f"cwd={working.resolve()}\n")
        stream.write(f"measurement_module={(working / TEST_PATH).resolve()}\n")
        stream.flush()
        outcome = subprocess.run(
            command,
            cwd=working,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
        stream.write(f"EXIT={outcome.returncode}\n")
    return outcome.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--arm", choices=("base", "head"), required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="mcintos-conic-suite-") as temp:
        result = _run(
            args.repository.resolve(),
            args.revision,
            args.arm,
            args.log,
            Path(temp),
        )
    print(json.dumps({"arm": args.arm, "revision": args.revision, "exit": result}))
    if args.arm == "head" and result:
        raise SystemExit(result)


if __name__ == "__main__":
    main()
