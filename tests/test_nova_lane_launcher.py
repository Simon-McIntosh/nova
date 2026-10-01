"""The general SLURM lane launcher: rung resolution and the --mem=0 refusal.

These pin the launcher's placement rules without submitting anything.  The
dry-run path prints the sbatch line for every rung from one preference list,
so a reader can check the partition, reservation and platform a rung would use.
An explicit --mem=0 is refused before it can reach sbatch, where SLURM reads it
as the whole node memory, leaving the job pending on Resources while blocking
the queue behind it.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = REPOSITORY_ROOT / "scripts" / "nova_lane" / "run.sh"


def _launch(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the launcher with a scrubbed, non-interactive environment."""
    return subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "TMPDIR": "/tmp"},
    )


def test_dry_run_prints_a_submission_line_for_every_rung(tmp_path: Path) -> None:
    result = _launch(
        "--dry-run",
        "--log",
        str(tmp_path / "lane.log"),
        "--",
        "tests/example_target.py",
        "-vv",
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    h200 = next(line for line in lines if line.startswith("RUNG=h200 "))
    titan = next(line for line in lines if line.startswith("RUNG=titan "))
    cpu = next(line for line in lines if line.startswith("RUNG=cpu "))

    assert "--partition=betelgeuse" in h200
    assert "--reservation=gpu_0003_grpA" in h200
    assert "--gres=gpu:1" in h200
    assert "JAX_PLATFORMS=cuda,cpu" in h200

    assert "--partition=titan" in titan
    assert "--gres=gpu:1" in titan

    assert "--partition=all_debug" in cpu
    assert "JAX_PLATFORMS=cpu" in cpu
    assert "--gres" not in cpu


def test_mem_zero_is_refused(tmp_path: Path) -> None:
    result = _launch(
        "--dry-run",
        "--mem=0",
        "--log",
        str(tmp_path / "lane.log"),
        "--",
        "tests/example_target.py",
    )
    assert result.returncode != 0
    assert "--mem=0" in (result.stderr + result.stdout)
