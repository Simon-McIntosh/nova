"""Run the production certificate rows sequentially with durable progress."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from time import time


OUTPUT_ROOT = Path(__file__).parent
CASE_RUNNER = OUTPUT_ROOT / "measure_certificate_case.py"
PROGRESS = OUTPUT_ROOT / "certificate-progress.json"
ROWS = (
    ("weak-rotation-reactor-static", -110, 0.007447526861632619),
    ("moderate-rotation-conventional-static", -110, 0.012457640060526081),
    ("weak-rotation-reactor-static", -300, 0.0055589300334190185),
    ("weak-rotation-reactor-static", -500, 0.0012464509730622084),
    ("weak-rotation-reactor-static", -1000, 0.0054014429054834836),
    ("moderate-rotation-conventional-static", -300, 0.008455791545012673),
)


def receipt_path(case_name: str, requested_cells: int) -> Path:
    resolution = (
        "reduced" if requested_cells == -110 else f"cells-{abs(requested_cells)}"
    )
    return (
        OUTPUT_ROOT
        / "certificate-parts"
        / f"{case_name}-production-route-{resolution}.json"
    )


def persist(rows: list[dict[str, object]]) -> None:
    temporary = PROGRESS.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"rows": rows}, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(PROGRESS)


def main() -> int:
    landed: list[dict[str, object]] = []
    persist(landed)
    for case_name, requested_cells, baseline in ROWS:
        print(
            f"CERTIFICATE_ROW_BEGIN case={case_name} cells={requested_cells}",
            flush=True,
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(CASE_RUNNER),
                case_name,
                str(requested_cells),
                str(baseline),
                "--criterion",
                "record",
            ],
            check=False,
        )
        path = receipt_path(case_name, requested_cells)
        record: dict[str, object] = {
            "baseline_residual": baseline,
            "case": case_name,
            "completed_at_unix_seconds": time(),
            "exit_status": completed.returncode,
            "requested_cells": requested_cells,
            "receipt": str(path),
        }
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            residual = row["solver"]["terminal_fixed_point_residual"]
            record.update(
                {
                    "qualification": row["solver"]["qualification"],
                    "residual_ratio_to_baseline": residual / baseline,
                    "terminal_residual": residual,
                }
            )
        landed.append(record)
        persist(landed)
        print("CERTIFICATE_ROW_END " + json.dumps(record, sort_keys=True), flush=True)
    return 0 if all(row["exit_status"] == 0 for row in landed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
