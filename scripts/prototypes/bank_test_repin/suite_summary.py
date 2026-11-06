"""Compose a suite verdict from the terminal status of every named pytest log."""

import argparse
from pathlib import Path


def summary(logs):
    """Retain each pytest summary and fail closed on missing terminal statuses."""
    statuses = []
    for log in logs:
        lines = log.read_text().splitlines() if log.exists() else []
        for line in lines:
            if line.startswith(
                (
                    "# revision",
                    "# tree",
                    "# command",
                    "nova.__file__",
                    "measurement_",
                    "FAILED",
                )
            ):
                print(line)
        print(f"# source {log}")
        print(
            next(
                (
                    line
                    for line in reversed(lines)
                    if " in " in line
                    and any(
                        word in line
                        for word in ("passed", "failed", "error", "skipped")
                    )
                ),
                "# pytest summary missing",
            )
        )
        terminal = lines[-1] if lines else "EXIT=missing"
        statuses.append(terminal)
    status = int(not statuses or any(value != "EXIT=0" for value in statuses))
    print("# per-file status " + " ".join(statuses))
    print(f"EXIT={status}", flush=True)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path, nargs="+")
    raise SystemExit(summary(parser.parse_args().logs))
