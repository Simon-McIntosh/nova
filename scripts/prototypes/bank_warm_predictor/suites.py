"""Run base/head files and an unchanged-seed mutant in fresh interpreters."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
BASE = "65315dc89facef18876b46b637a5466fc4a401fa"
FILES = ("tests/test_bank_drive_continuation.py", "tests/test_bank_solve_driver.py")


def snapshot(root, revision, destination):
    destination.mkdir()
    archive = destination.parent / (destination.name + ".tar")
    with archive.open("wb") as output:
        subprocess.run(
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
            ],
            stdout=output,
            check=True,
        )
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")
    archive.unlink()


def run(tree, revision, target, log, basetemp):
    command = (
        f"PYTHONPATH=. JAX_PLATFORMS=cpu {PYTHON} "
        f"-m pytest -p no:cacheprovider -q {target}"
    )
    environment = dict(
        os.environ,
        PYTHONPATH=str(tree),
        JAX_PLATFORMS="cpu",
        PYTEST_DEBUG_TEMPROOT=str(basetemp),
    )
    basetemp.mkdir()
    with log.open("w") as output:
        output.write(f"# revision: {revision}; tree: {tree}; command: {command}\n")
        output.flush()
        subprocess.run(
            [
                PYTHON,
                "-c",
                "from pathlib import Path; "
                "import nova.equilibrium.drive_continuation as m; "
                'print("# module:", m.__file__); print("# cwd:", Path.cwd().resolve())',
            ],
            cwd=tree,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )
        result = subprocess.run(
            command,
            shell=True,
            cwd=tree,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        output.write(f"\nEXIT={result.returncode}\n")
    text = log.read_text()
    failures = [
        line.split(" - ")[0].removeprefix("FAILED ")
        for line in text.splitlines()
        if line.startswith("FAILED ")
    ]
    return dict(
        revision=revision,
        command=command,
        exit_status=result.returncode,
        log_path=str(log),
        completed=(" passed" in text or " failed" in text),
        failure_count=len(failures),
        failure_ids=failures,
        measurement_module=str(tree / "nova/equilibrium/drive_continuation.py"),
        measurement_cwd=str(tree),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(
        ["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True
    ).strip()
    scratch = Path(
        tempfile.mkdtemp(prefix="bank-predictor-gates-", dir=os.environ["TMPDIR"])
    )
    receipts = {}
    try:
        for arm, revision in [("base", BASE), ("head", head)]:
            tree = scratch / arm
            snapshot(args.root, revision, tree)
            for index, target in enumerate(FILES):
                key = arm + "-" + Path(target).stem
                receipts[key] = run(
                    tree,
                    revision,
                    target,
                    args.run / (key + ".log"),
                    scratch / (key + "-temp"),
                )
                (args.run / "suite-receipts.json").write_text(
                    json.dumps(receipts, indent=2)
                )
        mutant = scratch / "mutant"
        snapshot(args.root, head, mutant)
        path = mutant / "nova/equilibrium/drive_continuation.py"
        source = path.read_text()
        onset = source.index(
            "    if previous_moments is None", source.index("def predict_seed(")
        )
        end = source.index("\n\ndef advance_drive(", onset)
        path.write_text(source[:onset] + "    return moments\n" + source[end:])
        key = "unchanged-seed-mutant"
        receipts[key] = run(
            mutant,
            head + " with unchanged-seed mutant",
            FILES[0] + "::test_linear_stream_predicts_next_state_to_rounding",
            args.run / (key + ".log"),
            scratch / (key + "-temp"),
        )
        (args.run / "suite-receipts.json").write_text(json.dumps(receipts, indent=2))
        expected = (
            "tests/test_bank_drive_continuation.py::"
            "test_linear_stream_predicts_next_state_to_rounding"
        )
        assert (
            receipts[key]["exit_status"] != 0
            and expected in receipts[key]["failure_ids"]
        )
        for target in FILES:
            key = Path(target).stem
            assert set(receipts["head-" + key]["failure_ids"]) <= set(
                receipts["base-" + key]["failure_ids"]
            )
            assert (
                receipts["head-" + key]["completed"]
                and receipts["base-" + key]["completed"]
            )
        print(
            "SUMMARY paired files: zero added failures; unchanged-seed mutant refused",
            flush=True,
        )
    finally:
        print("Removing job-owned scratch:", scratch, flush=True)
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
