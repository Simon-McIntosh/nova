"""Run focused base and head regression arms in fresh CPU processes."""

import argparse
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile


PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
TARGETS = (
    "tests/test_bank_newton_step.py::test_dense_newton_step_solves_scaled_residual",
    "tests/test_bank_newton_step.py::test_terminal_verdict_requires_fresh_read",
    "tests/test_bank_newton_step.py::test_bank_jvp_composes_flux_and_level_derivatives",
)
ARCHIVE_PATHS = (
    "nova",
    "scripts/prototypes/bank_newton_step/measure.py",
    "tests/test_bank_newton_step.py",
    "pyproject.toml",
)


def arm(label, revision, tree, log_path, scratch, head):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tree) + ("" if head else f":{Path.cwd().resolve()}")
    command = [
        PYTHON,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        *TARGETS,
        "-q",
        f"--basetemp={scratch / label}",
    ]
    with log_path.open("w") as log:
        log.write(f"REVISION={revision} TREE={tree} COMMAND={' '.join(command)}\n")
        log.flush()
        probe = subprocess.run(
            [
                PYTHON,
                "-c",
                "import os, nova; from nova.equilibrium import support_read; "
                "print('CWD=' + os.getcwd() + ' NOVA=' + nova.__file__ + "
                "' MODULE=' + support_read.__file__)",
            ],
            cwd=tree,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if probe.returncode:
            result = probe.returncode
        else:
            completed = subprocess.run(
                command,
                cwd=tree,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            result = completed.returncode
        log.write(f"EXIT={result}\n")
    print(f"ARM {label} REVISION={revision} EXIT={result} LOG={log_path}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--logs", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    args.logs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bank-suite-", dir="/tmp") as temporary:
        scratch = Path(temporary)
        base_tree = scratch / "base"
        base_tree.mkdir()
        archive = scratch / "base.tar"
        with archive.open("wb") as output:
            subprocess.run(
                ["git", "-C", str(root), "archive", args.base, *ARCHIVE_PATHS],
                stdout=output,
                check=True,
            )
        with tarfile.open(archive) as payload:
            payload.extractall(base_tree, filter="data")
        base_status = arm(
            "base",
            args.base,
            base_tree,
            args.logs / "baseline-suite.log",
            scratch,
            False,
        )
        head_status = arm(
            "head", head, root, args.logs / "after-suite.log", scratch, True
        )
    return max(base_status, head_status)


if __name__ == "__main__":
    raise SystemExit(main())
