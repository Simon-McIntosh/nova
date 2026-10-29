"""Run repair checks against the parent contents of their source commits."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile


PYTHON = Path("/home/ITER/mcintos/Code/nova/.venv/bin/python")
ROOT = Path(__file__).resolve().parents[3]
CONTROLS = (
    (
        "bc89776bd786bb52fdb1b7507ec77006482c191e",
        "tests/test_bank_newton_step.py",
        "revert-step-test",
        ("--test",),
    ),
    (
        "416fd89c436804d1739de9ba9e35538380a8cf57",
        "scripts/prototypes/bank_newton_step/measure.py",
        "revert-factor-comparison",
        ("--compare-present", "--converge"),
    ),
    (
        "a120b3d4864d961ce42fbe4bf9144eebf13c9a07",
        "scripts/prototypes/bank_newton_step/measure.py",
        "revert-limited-comparison",
        ("--kind", "limited", "--converge"),
    ),
    (
        "9edff1e29e99a8a055652d4845a7bd3f059aab8d",
        "scripts/prototypes/bank_newton_step/measure.py",
        "revert-tracked-line-search",
        ("--tracked-line-search", "--converge"),
    ),
)


def record(path, revision, tree, command, environment):
    with path.open("w") as log:
        print(
            f"REVISION={revision} TREE={tree} CWD={tree} "
            f"COMMAND={shlex.join(str(part) for part in command)}",
            file=log,
            flush=True,
        )
        module = subprocess.run(
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
        )
        if module.returncode:
            result = module.returncode
        else:
            result = subprocess.run(
                [str(part) for part in command],
                cwd=tree,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        print(f"EXIT={result}", file=log, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="bank-newton-revert-", dir=os.environ["TMPDIR"]
    ) as name:
        scratch = Path(name)
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(ROOT), JAX_PLATFORMS="cpu", TMPDIR=name)
        for commit, source, label, arguments in CONTROLS:
            parent = subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", commit + "^"], text=True
            ).strip()
            content = subprocess.check_output(
                ["git", "-C", str(ROOT), "show", f"{commit}^:{source}"]
            )
            old = scratch / f"{label}-{Path(source).name}"
            old.write_bytes(content)
            if arguments == ("--test",):
                command = (
                    str(PYTHON),
                    "-m",
                    "pytest",
                    "-p",
                    "no:cacheprovider",
                    str(old) + "::test_dense_newton_step_solves_scaled_residual",
                    "-q",
                    "--basetemp",
                    str(scratch / "pytest-revert"),
                )
            else:
                command = (str(PYTHON), str(old), *arguments)
            exit_status = record(
                args.log_dir / f"{label}.log", parent, scratch, command, environment
            )
            print(f"CONTROL={label} EXIT={exit_status}", flush=True)

        mutated = scratch / "mutated"
        shutil.copytree(ROOT / "nova", mutated / "nova")
        test = mutated / "test_bank_newton_step.py"
        shutil.copy2(ROOT / "tests/test_bank_newton_step.py", test)
        module_path = mutated / "nova/equilibrium/support_read.py"
        source = module_path.read_text()
        original = "solution = column * linalg.lu_solve(factor, right / row)"
        if source.count(original) != 1:
            raise RuntimeError("dense update marker was not unique")
        module_path.write_text(
            source.replace(
                original,
                "solution = -column * linalg.lu_solve(factor, right / row)",
                1,
            )
        )
        mutated_environment = environment.copy()
        mutated_environment["PYTHONPATH"] = f"{mutated}:{ROOT}"
        command = (
            str(PYTHON),
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            str(test) + "::test_dense_newton_step_solves_scaled_residual",
            "-q",
            "--basetemp",
            str(scratch / "pytest-mutation"),
        )
        exit_status = record(
            args.log_dir / "mutated-dense-update.log",
            subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            mutated,
            command,
            mutated_environment,
        )
        print(f"CONTROL=mutated-dense-update EXIT={exit_status}", flush=True)


if __name__ == "__main__":
    main()
