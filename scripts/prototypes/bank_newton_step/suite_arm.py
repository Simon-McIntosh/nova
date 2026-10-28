"""Run the support-read regression files in separate fresh pytest processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import tempfile


PYTHON = Path("/home/ITER/mcintos/Code/nova/.venv/bin/python")
ROOT = Path(__file__).resolve().parents[3]
FILES = ("tests/test_support_read.py", "tests/test_operator_bank_compile.py")


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
        raise RuntimeError("the base archive did not complete")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--include-step-test", action="store_true")
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    tests = (
        (*FILES, "tests/test_bank_newton_step.py") if args.include_step_test else FILES
    )
    failures: list[str] = []
    complete = True
    exit_status = 0
    with tempfile.TemporaryDirectory(
        prefix="nova-suite-", dir=os.environ["TMPDIR"]
    ) as scratch:
        tree = ROOT if args.revision == "HEAD" else Path(scratch) / "base"
        if tree != ROOT:
            tree.mkdir()
            archive(args.revision, tree)
        revision = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", args.revision], text=True
        ).strip()
        files = []
        for ordinal, test_file in enumerate(tests):
            command = [
                str(PYTHON),
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(Path(scratch) / f"pytest-{ordinal}"),
                test_file,
                "-q",
            ]
            environment = os.environ.copy()
            environment.update(
                TMPDIR=scratch, PYTHONPATH=str(tree), JAX_PLATFORMS="cpu"
            )
            path = args.log_dir / f"{Path(test_file).stem}.log"
            with path.open("w") as log:
                header = f"REVISION={revision} TREE={tree} CWD={tree}"
                print(
                    f"{header} COMMAND={shlex.join(command)}",
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
            text = path.read_text(errors="replace")
            file_failures = re.findall(r"^FAILED (\S+)", text, re.MULTILINE)
            failures.extend(file_failures)
            finished = bool(
                re.search(r"\b\d+ (?:passed|failed|skipped|error|errors)\b", text)
            )
            complete &= finished
            exit_status = max(exit_status, result.returncode)
            files.append(
                {
                    "test": test_file,
                    "log_path": str(path),
                    "exit_status": result.returncode,
                    "completed": finished,
                    "failure_ids": file_failures,
                }
            )
            print(
                f"FILE={test_file} EXIT={result.returncode} COMPLETED={finished} "
                f"FAILURES={len(file_failures)}",
                flush=True,
            )
        receipt = {
            "revision": revision,
            "command": shlex.join(
                [str(PYTHON), str(Path(__file__).resolve()), *os.sys.argv[1:]]
            ),
            "exit_status": exit_status,
            "log_path": str(args.log_dir / "summary.log"),
            "completed": complete,
            "failure_count": len(failures),
            "failure_ids": failures,
            "files": files,
        }
        summary = args.log_dir / "summary.log"
        with summary.open("w") as log:
            header = f"REVISION={revision} TREE={tree} CWD={tree}"
            print(
                f"{header} COMMAND={receipt['command']}",
                file=log,
            )
            print(json.dumps(receipt, sort_keys=True), file=log)
            print(f"EXIT={exit_status}", file=log)
        (args.log_dir / "receipt.json").write_text(json.dumps(receipt, sort_keys=True))
        print("SHA256=" + hashlib.sha256(summary.read_bytes()).hexdigest(), flush=True)
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
