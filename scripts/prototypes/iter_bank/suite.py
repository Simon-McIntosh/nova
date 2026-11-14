"""Run paired bank test files in fresh interpreters with durable receipts."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--negative", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = [
        "tests/test_iter_bank.py",
        "tests/test_bank_traced_inputs.py",
        "tests/test_support_read.py",
        "tests/test_bank_drive_continuation.py",
        "tests/test_bank_lane_drive.py",
    ]
    if args.negative:
        files = files[:1]
    env = dict(os.environ, PYTHONPATH=".", JAX_PLATFORMS="cpu")
    if args.negative:
        env["NOVA_ITER_BANK_DROP_COIL"] = "1"
    else:
        env.pop("NOVA_ITER_BANK_DROP_COIL", None)
    module = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import nova.equilibrium.support_read as m; print(m.__file__)",
        ],
        cwd=args.tree,
        env=env,
        text=True,
    ).strip()
    records = []
    for name in files:
        record = dict(revision=args.revision, file=name, completed=False)
        records.append(record)
        destination = args.output / "receipt.json"
        if not (args.tree / name).exists():
            record["note"] = "Test file absent at this revision; pytest arm not run"
            destination.write_text(json.dumps(records, indent=2) + "\n")
            continue
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            name,
            "-rfP",
        ]
        command_text = "PYTHONPATH=. JAX_PLATFORMS=cpu " + shlex.join(command)
        log = args.output / (Path(name).stem + ".log")
        record.update(
            command=command_text, log_path=str(log), module=module, cwd=str(args.tree)
        )
        destination.write_text(json.dumps(records, indent=2) + "\n")
        env["PYTEST_ADDOPTS"] = "--basetemp=" + str(
            Path(os.environ["TMPDIR"]) / (args.output.name + "-" + Path(name).stem)
        )
        with log.open("w") as stream:
            stream.write(
                f"# revision: {args.revision}; tree: {args.tree}; "
                f"command: {command_text}\n"
            )
            stream.write(
                f"# measurement_module: {module}\n# measurement_cwd: {args.tree}\n"
            )
            for key in (
                "NOVA_ITER_BANK_RECEIPT",
                "NOVA_ITER_BANK_COIL_RECEIPT",
                "NOVA_ITER_BANK_DROP_COIL",
            ):
                stream.write(f"# {key}: {env.get(key, '')}\n")
            stream.flush()
            result = subprocess.run(
                command, cwd=args.tree, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            stream.write(f"EXIT={result.returncode}\n")
        text = log.read_text()
        record.update(
            exit_status=result.returncode,
            completed=bool(re.search(r"\d+ (passed|failed|skipped|error)", text)),
            failure_ids=re.findall(r"^FAILED (\S+)", text, re.MULTILINE),
            summaries=re.findall(
                r"^.*\d+ (?:passed|failed|skipped|error).*(?:in .*s|seconds).*$",
                text,
                re.MULTILINE,
            )[-1:],
        )
        record["failure_count"] = len(record["failure_ids"])
        destination.write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
