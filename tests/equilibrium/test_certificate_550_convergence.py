"""The published limited carrier is solved by the certificate route."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.slow
def test_certified_limited_carrier_converges(tmp_path: Path) -> None:
    receipt_path = tmp_path / "limited-solve.json"
    command = [
        sys.executable,
        "-m",
        "scripts.prototypes.fast_map_production.solver",
        "--kind",
        "limited",
        "--cells",
        "550",
        "--method",
        "gmres",
        "--line-search",
        "--line-search-depth",
        "20",
        "--vertical-pin",
        "--output",
        str(receipt_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )
    assert receipt_path.exists(), completed.stdout + completed.stderr
    receipt = json.loads(receipt_path.read_text())
    assert receipt["cells"] == 553
    assert receipt["carrier_cache"]["hit"]
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "vertical_pin" in receipt
    assert receipt["converged"]
    assert receipt["history"][-1]["residual"] <= 1e-12
