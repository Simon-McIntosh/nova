"""Run the covering test files in independent Python processes."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
FILES = (
    "tests/equilibrium/test_bounded_point_program.py",
    "tests/biot/test_polygon_edge_line_tangent.py",
    "tests/biot/test_polygon_stable_evaluator.py",
    "tests/test_moment_field_hessian.py",
)


def main():
    revision, output, source = sys.argv[1:]
    output, source = Path(output), Path(source)
    environment = os.environ.copy()
    environment.pop("NOVA_POLYGON_INTEGRATED_RECEIPT", None)
    records = []
    started = time.monotonic()
    for file in FILES:
        if not (source / file).is_file():
            continue
        command = (
            f"PYTHONPATH=. JAX_PLATFORMS=cpu {PYTHON} -m pytest "
            f"-p no:cacheprovider -q {file}"
        )
        log_path = output / (Path(file).stem + ".log")
        with log_path.open("w") as log:
            log.write(f"# revision: {revision} tree: {source}\n")
            log.write(f"# command: {command}\n")
            log.flush()
            subprocess.run(
                [
                    PYTHON,
                    "-c",
                    "import os; from nova.biot import polygonanalytic; "
                    "print('MEASUREMENT_MODULE', polygonanalytic.__file__); "
                    "print('MEASUREMENT_CWD', os.getcwd())",
                ],
                cwd=source,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=source,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            log.write(f"EXIT={result.returncode}\n")
        text = log_path.read_text()
        summaries = [
            line
            for line in text.splitlines()
            if re.search(r"\b\d+ (?:passed|failed|skipped|errors?)\b.*\bin \d", line)
        ]
        summary = summaries[-1] if summaries else ""
        counts = {
            name: int(count)
            for count, name in re.findall(
                r"\b(\d+) (passed|failed|skipped|errors?|xfailed|xpassed|deselected)\b",
                summary,
            )
        }
        failures = re.findall(r"^(?:FAILED|ERROR) (\S+)", text, re.MULTILINE)
        record = dict(
            revision=revision,
            command=command,
            exit_status=result.returncode,
            log_path=str(log_path),
            completed=bool(summary),
            failure_count=counts.get("failed", 0)
            + counts.get("error", 0)
            + counts.get("errors", 0),
            failure_ids=failures,
            counts=counts,
            summary=summary,
        )
        records.append(record)
        (output / "per-file.json").write_text(json.dumps(records, indent=2) + "\n")
        print(file, summary or "NO COMPLETED SUMMARY", flush=True)

    completed = bool(records) and all(row["completed"] for row in records)
    code = int(not completed or any(row["exit_status"] for row in records))
    receipt = dict(
        revision=revision,
        command=records[0]["command"],
        exit_status=code,
        log_path=str(output / "suite.log"),
        completed=completed,
        failure_count=sum(row["failure_count"] for row in records),
        failure_ids=[name for row in records for name in row["failure_ids"]],
        counts={
            name: sum(row["counts"].get(name, 0) for row in records)
            for name in ("passed", "failed", "skipped", "errors")
        },
        per_file=records,
        wall_seconds=time.monotonic() - started,
    )
    (output / "suite.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with (output / "suite.log").open("w") as log:
        log.write(f"# revision: {revision} tree: {source}\n")
        log.write(f"# command: {records[0]['command']}\n")
        log.write(f"MEASUREMENT_MODULE {source}/nova/biot/polygonanalytic.py\n")
        log.write(f"MEASUREMENT_CWD {source}\n")
        for row in records:
            log.write(f"# per-file command: {row['command']}\n")
            log.write(f"# per-file log: {row['log_path']}\n")
            log.write(row["summary"] + "\n")
            for name in row["failure_ids"]:
                log.write(f"FAILED {name}\n")
        log.write("SUITE_SUMMARY " + json.dumps(receipt["counts"]) + "\n")
        log.write(f"COMPLETED={str(completed).lower()}\nEXIT={code}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
