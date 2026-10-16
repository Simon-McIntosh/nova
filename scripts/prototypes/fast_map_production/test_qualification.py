"""Keep measured map, transition, and cold-solve failures visible."""

import json
import subprocess
import sys


def _check(tmp_path, kind, receipts):
    paths = []
    for index, receipt in enumerate(receipts):
        path = tmp_path / f"receipt-{index}.json"
        path.write_text(json.dumps(receipt))
        paths.append(str(path))
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.prototypes.fast_map_production.qualification",
            kind,
            *paths,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_diverted_map_limit_and_resolution_trend(tmp_path):
    rows = [
        {"cells": 282, "centre_map_flux_error": 0.001086579452202},
        {"cells": 550, "centre_map_flux_error": 0.000487272102151},
        {"cells": 2098, "centre_map_flux_error": 0.000086665898991},
    ]
    assert _check(tmp_path, "map", rows).returncode == 0
    rows[0]["centre_map_flux_error"] = 0.014939596484155807
    failed = _check(tmp_path, "map", rows)
    assert failed.returncode != 0
    assert "282-cell map error" in failed.stderr


def test_fine_scan_refuses_support_exchange(tmp_path):
    row = {
        "step": 0.00007871,
        "support_ids": [9, 71],
        "boundary_class": 1,
        "residual": 0.002684368,
        "membership_71": 0.233988925,
        "cell_current_residual": -0.002684368,
        "wall_candidate": -0.0000048718,
        "saddle_candidate": -0.0000049000,
    }
    next_row = dict(row, step=row["step"] + 3.6e-8)
    assert _check(tmp_path, "scan", [{"scan": [row, row, next_row]}]).returncode == 0
    next_row["support_ids"] = [4, 6]
    next_row["membership_71"] = 0.0
    next_row["residual"] = 1.683654263
    failed = _check(tmp_path, "scan", [{"scan": [row, row, next_row]}])
    assert failed.returncode != 0
    assert "support jumps" in failed.stderr


def test_cold_solve_refuses_fine_residual(tmp_path):
    good = {"history": [{"residual": 1.24833744271e-13}], "converged": True}
    assert _check(tmp_path, "solve", [good]).returncode == 0
    bad = {"history": [{"residual": 0.001550315837434896}], "converged": False}
    failed = _check(tmp_path, "solve", [bad])
    assert failed.returncode != 0
    assert "cold solve residual" in failed.stderr
