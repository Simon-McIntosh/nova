"""Run independent CPU file gates and a shared-sub-step negative control."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from scripts.prototypes.bank_warm_predictor.suites import snapshot

PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
FILES = (
    "tests/test_bank_drive_continuation.py",
    "tests/test_bank_solve_driver.py",
    "tests/test_bank_lane_drive.py",
)


def run(tree, revision, target, log, basetemp):
    command = (
        f"PYTHONPATH={tree} JAX_PLATFORMS=cpu {PYTHON} "
        f"-m pytest -p no:cacheprovider -q {tree / target}"
    )
    environment = dict(
        os.environ,
        PYTHONPATH=str(tree),
        JAX_PLATFORMS="cpu",
        PYTEST_DEBUG_TEMPROOT=str(basetemp),
        OMP_NUM_THREADS="7",
        OPENBLAS_NUM_THREADS="1",
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
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(
        ["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True
    ).strip()
    scratch = Path(tempfile.mkdtemp(prefix="lane-gates-", dir=os.environ["TMPDIR"]))
    receipts = {}
    try:
        for arm, revision in [("base", args.base), ("head", head)]:
            tree = scratch / arm
            snapshot(args.root, revision, tree)
            for target in FILES:
                if not (tree / target).exists():
                    continue
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
        marker = "        active = ~done & ~stopped\n"
        assert source.count(marker) == 1
        path.write_text(
            source.replace(
                marker,
                marker
                + "        sub_step = jnp.full_like(sub_step, jnp.min(sub_step))\n",
            )
        )
        key = "shared-step-mutant"
        receipts[key] = run(
            mutant,
            head + " shared-step mutant",
            FILES[2] + "::test_stiff_lane_does_not_shrink_easy_lane",
            args.run / (key + ".log"),
            scratch / "mutant-temp",
        )
        (args.run / "suite-receipts.json").write_text(json.dumps(receipts, indent=2))
        assert receipts[key]["exit_status"] != 0 and receipts[key]["failure_ids"]
        for target in FILES[:2]:
            name = Path(target).stem
            assert receipts["base-" + name]["completed"]
            assert receipts["head-" + name]["completed"]
            assert set(receipts["head-" + name]["failure_ids"]) <= set(
                receipts["base-" + name]["failure_ids"]
            )
        assert receipts["head-test_bank_lane_drive"]["exit_status"] == 0
        print(
            "SUMMARY paired files: zero added failures; shared-step mutant refused",
            flush=True,
        )
    finally:
        print("Removing job-owned scratch:", scratch, flush=True)
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
