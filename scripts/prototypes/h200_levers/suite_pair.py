"""Run a focused baseline and final-head suite in archived CPU trees."""

import argparse
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import tempfile

PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
TEST = "tests/test_booking_scalar_tile.py"
COMMAND = (
    f"PYTHONPATH=. JAX_PLATFORMS=cpu {PYTHON} -m pytest -p no:cacheprovider -q {TEST}"
)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("base")
    parser.add_argument("head")
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    receipts = {}
    with tempfile.TemporaryDirectory(prefix="booking-suite-", dir="/tmp") as temporary:
        for label, revision in (
            ("negative", args.head),
            ("baseline", args.base),
            ("after", args.head),
        ):
            tree = Path(temporary) / label
            tree.mkdir()
            archive = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(root),
                    "archive",
                    revision,
                    "nova",
                    "tests",
                    "scripts",
                    "benchmarks",
                    "conftest.py",
                    "pyproject.toml",
                ]
            )
            with tarfile.open(fileobj=io.BytesIO(archive)) as source:
                source.extractall(tree, filter="data")
            assert (tree / TEST).is_file()
            command = COMMAND
            if label == "negative":
                guard = tree / "scripts/prototypes/h200_levers/run.py"
                source = guard.read_text()
                assert source.count("if a == b:") == 1
                guard.write_text(source.replace("if a == b:", "if False and a == b:"))
                command += "::test_identical_timing_programs_are_refused"
            scratch = Path(temporary) / (label + "-scratch")
            scratch.mkdir()
            env = dict(
                os.environ,
                PYTHONPATH=".",
                JAX_PLATFORMS="cpu",
                TMPDIR=str(scratch),
                PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="7",
                OPENBLAS_NUM_THREADS="7",
            )
            env.pop("PYTEST_ADDOPTS", None)
            log = run / (label + "-suite.log")
            with log.open("w") as output:
                output.write(
                    f"# revision: {revision} tree: {tree} command: {command}\n"
                )
                output.flush()
                inspect = (
                    "from nova.jax.config import configure_dtypes; "
                    "configure_dtypes(); import jax; "
                    "assert jax.config.jax_enable_x64 is True; "
                    "from nova.equilibrium import current; "
                    "from pathlib import Path; "
                    "print('MODULE='+current.__file__); "
                    "print('CWD='+str(Path.cwd().resolve())); "
                    "print('SCALAR_DEFAULT='+str(current._SCALAR_CUT_BATCH_WIDTH))"
                )
                subprocess.run(
                    [PYTHON, "-c", inspect],
                    cwd=tree,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                result = subprocess.run(
                    shlex.split(command.split(" ", 2)[2]),
                    cwd=tree,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
                output.write(f"EXIT={result.returncode}\n")
            text = log.read_text()
            failed = re.findall(r"^FAILED (\S+)", text, re.MULTILINE)
            complete = bool(re.search(r"\d+ (?:passed|failed).* in [\d.]+s", text))
            receipts[label] = dict(
                revision=revision,
                command=command,
                exit_status=result.returncode,
                log_path=str(log),
                completed=complete,
                failure_count=len(failed),
                failure_ids=failed,
                tree=str(tree),
            )
            (run / "suite-pair.json").write_text(json.dumps(receipts, indent=2))
            print(
                f"SUITE {label} revision={revision} exit={result.returncode}",
                flush=True,
            )
    assert receipts["negative"]["completed"]
    assert receipts["negative"]["exit_status"] == 1
    assert receipts["negative"]["failure_ids"] == [
        TEST + "::test_identical_timing_programs_are_refused"
    ]
    assert all(
        receipts[label]["completed"] and receipts[label]["exit_status"] == 0
        for label in ("baseline", "after")
    )
    print("SUMMARY baseline and final-head scalar booking suites passed", flush=True)


if __name__ == "__main__":
    main()
