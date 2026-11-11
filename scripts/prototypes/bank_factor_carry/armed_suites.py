"""Run the armed two-file command at each revision in one allocation."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from scripts.prototypes.bank_warm_predictor.suites import snapshot, FILES, PYTHON


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(
        ["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True
    ).strip()
    scratch = Path(tempfile.mkdtemp(prefix="factor-armed-", dir=os.environ["TMPDIR"]))
    receipts = {}
    try:
        for arm, revision in [("base", args.base), ("head", head)]:
            tree = scratch / arm
            snapshot(args.root, revision, tree)
            command = (
                f"PYTHONPATH=. JAX_PLATFORMS=cpu {PYTHON} -m pytest -p no:cacheprovider -q "
                + " ".join(FILES)
            )
            log = args.run / f"armed-{arm}.log"
            env = dict(os.environ, PYTHONPATH=str(tree), JAX_PLATFORMS="cpu")
            with log.open("w") as output:
                output.write(
                    f"# revision: {revision}; tree: {tree}; command: {command}\n"
                )
                output.flush()
                subprocess.run(
                    [
                        PYTHON,
                        "-c",
                        "from pathlib import Path; "
                        "from nova.equilibrium import drive_continuation, support_read; "
                        'print("# module:", drive_continuation.__file__); '
                        'print("# support_module:", support_read.__file__); '
                        'print("# cwd:", Path.cwd().resolve())',
                    ],
                    env=env,
                    cwd=tree,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                completed = subprocess.run(
                    command,
                    shell=True,
                    cwd=tree,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
                output.write(f"\nEXIT={completed.returncode}\n")
            text = log.read_text()
            failures = [
                line.split(" - ")[0].removeprefix("FAILED ")
                for line in text.splitlines()
                if line.startswith("FAILED ")
            ]
            receipts[arm] = dict(
                revision=revision,
                command=command,
                exit_status=completed.returncode,
                log_path=str(log),
                completed=(" passed" in text or " failed" in text),
                failure_count=len(failures),
                failure_ids=failures,
                measurement_module=str(tree / "nova/equilibrium/drive_continuation.py"),
                measurement_cwd=str(tree),
            )
            (args.run / "armed-suite-receipts.json").write_text(
                json.dumps(receipts, indent=2)
            )
        assert receipts["base"]["completed"] and receipts["head"]["completed"]
        assert set(receipts["head"]["failure_ids"]) <= set(
            receipts["base"]["failure_ids"]
        )
        print("SUMMARY armed two-file command: zero added failures", flush=True)
    finally:
        print("Removing job-owned scratch:", scratch, flush=True)
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
