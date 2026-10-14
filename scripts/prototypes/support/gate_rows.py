"""Run matching forward test modules in fresh processes for both revisions."""

import argparse
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    baseline = Path(tempfile.mkdtemp(prefix="moment-base-", dir=os.environ["TMPDIR"]))
    archive = subprocess.Popen(
        [
            "git",
            "archive",
            args.base,
            "nova",
            "tests",
            "benchmarks",
            "scripts",
            "conftest.py",
            "pyproject.toml",
        ],
        stdout=subprocess.PIPE,
    )
    with tarfile.open(fileobj=archive.stdout, mode="r|") as stream:
        stream.extractall(baseline, filter="data")
    if archive.wait() != 0:
        raise RuntimeError("baseline extraction failed")
    (baseline / "docs").symlink_to(root / "docs", target_is_directory=True)
    for name in args.files:
        shutil.copy2(root / name, baseline / name)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    command = shlex.join(
        [sys.executable, "-m", "scripts.prototypes.support.gate_rows", *sys.argv[1:]]
    )
    receipts = {}
    for arm, tree, revision in (
        ("baseline", baseline, args.base),
        ("after", root, head),
    ):
        records = []
        master = args.out / f"{arm}-suite.log"
        with master.open("w") as combined:
            combined.write(f"revision={revision} TREE={tree} COMMAND={command}\n")
            for index, name in enumerate(args.files):
                log = args.out / f"{arm}-{Path(name).stem}.log"
                test_args = [
                    "-p",
                    "no:cacheprovider",
                    "--timeout=180",
                    "-v",
                    "--basetemp="
                    + str(baseline.parent / f"moment-{arm}-{baseline.name}-{index}"),
                    str(tree / name),
                ]
                program = (
                    "from nova.jax.config import configure_dtypes; "
                    "configure_dtypes(); import jax; "
                    "assert jax.config.jax_enable_x64 is True; "
                    "import os,nova,pytest,importlib; "
                    "forward=importlib.import_module('nova.equilibrium.forward'); "
                    "print('NOVA_MODULE='+nova.__file__+"
                    "' MEASUREMENT_MODULE='+forward.__file__+"
                    "' CWD='+os.getcwd(),flush=True); "
                    f"raise SystemExit(pytest.main({test_args!r}))"
                )
                invocation = [sys.executable, "-c", program]
                env = dict(os.environ, PYTHONPATH=str(tree))
                with log.open("w") as output:
                    output.write(
                        f"revision={revision} TREE={tree} "
                        f"COMMAND={shlex.join(invocation)}\n"
                    )
                    output.flush()
                    try:
                        result = subprocess.run(
                            invocation,
                            cwd=tree,
                            env=env,
                            stdout=output,
                            stderr=subprocess.STDOUT,
                            timeout=1800,
                        ).returncode
                    except subprocess.TimeoutExpired:
                        result = 124
                    output.write(f"\nEXIT={result}\n")
                text = log.read_text()
                failed = [
                    line.removeprefix("FAILED ").removeprefix("ERROR ").split(" - ")[0]
                    for line in text.splitlines()
                    if line.startswith(("FAILED ", "ERROR "))
                ]
                completed = (
                    bool(
                        re.search(
                            r"^=+.*(?: passed| failed| error| skipped).* in .* =+$",
                            text,
                            re.MULTILINE,
                        )
                    )
                    and result != 124
                )
                records.append(
                    dict(
                        file=name,
                        exit_status=result,
                        completed=completed,
                        failure_ids=failed,
                        log_path=str(log),
                    )
                )
                combined.write(text)
                combined.flush()
                print(f"ARM={arm} FILE={name} EXIT={result}", flush=True)
            failures = [item for row in records for item in row["failure_ids"]]
            status = max(abs(row["exit_status"]) for row in records)
            completed = all(row["completed"] for row in records)
            combined.write(
                f"SUITE_COMPLETED={completed} FILES={len(records)} "
                f"FAILURE_COUNT={len(failures)}\nEXIT={status}\n"
            )
        receipt = dict(
            revision=revision,
            command=command,
            exit_status=status,
            log_path=str(master),
            completed=completed,
            failure_count=len(failures),
            failure_ids=failures,
            modules=records,
        )
        (args.out / f"{arm}-suite.json").write_text(json.dumps(receipt) + "\n")
        receipts[arm] = receipt
    base_failures = set(receipts["baseline"]["failure_ids"])
    head_failures = set(receipts["after"]["failure_ids"])
    complete = all(receipt["completed"] for receipt in receipts.values())
    delta = dict(
        completed=complete,
        added=sorted(head_failures - base_failures),
        removed=sorted(base_failures - head_failures),
        passed=complete and not (head_failures - base_failures),
    )
    (args.out / "delta.json").write_text(json.dumps(delta) + "\n")
    print("DELTA=" + json.dumps(delta), flush=True)
    raise SystemExit(0 if delta["passed"] else 1)


if __name__ == "__main__":
    main()
