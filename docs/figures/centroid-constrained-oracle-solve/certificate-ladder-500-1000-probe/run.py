"""Run four independent bounded measurements inside one allocation."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = Path(os.environ["CERTIFICATE_PROBE_OUTPUT"])
PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
ROWS = (
    ("weak-rotation-reactor-static", -1000),
    ("moderate-rotation-conventional-static", -1000),
    ("strong-rotation-compact-static", -1000),
    ("diverted-single-null", -500),
)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    any_failed = False
    for case, cells in ROWS:
        stem = f"{case}-{abs(cells)}"
        receipt_path = OUTPUT / f"{stem}.json"
        log_path = OUTPUT / f"{stem}.log"
        rss_path = OUTPUT / f"{stem}.rss-kib"
        command = [PYTHON, str(HERE / "measure.py"), case, str(cells)]
        with log_path.open("w") as log:
            log.write(
                f"REVISION {revision} TREE {HERE.parents[3]} "
                f"COMMAND {' '.join(command)}\n"
            )
            log.flush()
            completed = subprocess.run(
                [
                    "/usr/bin/time",
                    "-f",
                    "RSS_KIB=%M",
                    "-o",
                    str(rss_path),
                    "timeout",
                    "-v",
                    "--kill-after=15s",
                    "720s",
                    *command,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log.write(f"EXIT={completed.returncode}\n")
        row = (
            json.loads(receipt_path.read_text())
            if receipt_path.exists()
            else {
                "case": case,
                "requested_cells": cells,
                "clip_mode_requested": "exact",
            }
        )
        timed_out = completed.returncode in (124, 137) and (
            "timeout: sending signal" in log_path.read_text()
        )
        if timed_out:
            row.update(status="timed-out", failure_text="row exceeded 720 seconds")
        elif completed.returncode != 0 and row.get("status") == "in-progress":
            row.update(
                status="compile-failed", failure_text="process exited before receipt"
            )
        row["process_exit_code"] = completed.returncode
        rss_lines = rss_path.read_text().splitlines() if rss_path.exists() else []
        rss_values = [
            line.split("=", 1)[1] for line in rss_lines if line.startswith("RSS_KIB=")
        ]
        row["peak_host_rss_kib"] = int(rss_values[-1]) if rss_values else None
        row["log_path"] = str(log_path)
        receipt_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        print(f"ROW {stem} {row['status']} EXIT={completed.returncode}", flush=True)
        any_failed |= completed.returncode != 0 or row["status"] != "solved"
    print("MEASUREMENT_COMPLETE", flush=True)
    if any_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
