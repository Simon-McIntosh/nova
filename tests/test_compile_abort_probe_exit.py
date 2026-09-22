"""The sampler wrapper's process status must carry the sampled child's status.

The probe exists to leave a trace of a driver that dies inside the compiler, and
the driver's own status is the signal that says it died at all.  A wrapper that
records that status in its summary and then exits zero lets a caller gating on
the wrapper's status read an abort as a successful measurement, so the recorded
status and the process status have to agree in both directions.

The child is a trivial shell command rather than a compile, so each case costs a
scheduling slice: the wrapper's status is the subject, not the driver's work.
The arbitrary-status case is kept apart from the two named ones because a wrapper
that special-cased a single well-known code would satisfy the named pair while
still discarding every other failure.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = [sys.executable, "-m", "benchmarks.compile_abort_probe"]


def _sample_a_child(
    tmp_path: Path, child_command: str
) -> tuple[int, dict[str, object]]:
    """Run the probe wrapper around a trivial shell child and read its summary."""
    samples = tmp_path / "samples.jsonl"
    summary_path = tmp_path / "summary.json"
    completed = subprocess.run(
        [
            *PROBE,
            "--samples",
            str(samples),
            "--summary",
            str(summary_path),
            "--interval",
            "0.05",
            "--",
            "/bin/sh",
            "-c",
            child_command,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert summary_path.exists(), completed.stderr
    assert samples.exists(), completed.stderr
    for line in samples.read_text().splitlines():
        json.loads(line)
    return completed.returncode, json.loads(summary_path.read_text())


def test_failed_child_fails_the_wrapper(tmp_path: Path) -> None:
    wrapper_status, summary = _sample_a_child(tmp_path, "exit 7")

    assert summary["exit_code"] == 7
    assert wrapper_status == 7


def test_successful_child_leaves_the_wrapper_at_zero(tmp_path: Path) -> None:
    wrapper_status, summary = _sample_a_child(tmp_path, "exit 0")

    assert summary["exit_code"] == 0
    assert wrapper_status == 0


def test_the_wrapper_carries_an_arbitrary_child_status(tmp_path: Path) -> None:
    wrapper_status, summary = _sample_a_child(tmp_path, "exit 3")

    assert summary["exit_code"] == 3
    assert wrapper_status == 3
