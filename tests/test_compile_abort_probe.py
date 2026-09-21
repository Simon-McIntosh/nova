"""The sampler wrapper must exit with the status of the child it sampled.

The probe exists to keep a trace of a driver that dies inside the compiler, and
the driver's own status is the only signal that says it died at all.  A wrapper
that records that status in its summary and then exits zero lets a caller that
gates on the wrapper's status read an abort as a successful measurement, so the
recorded status and the process status have to agree.

Both directions are asserted: a child that fails must fail the wrapper, and a
child that succeeds must leave it at zero, so a wrapper that failed whenever it
ran would not pass this file either.
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
    """Run the probe wrapper around a trivial shell child and read its summary.

    The child is trivial so the case costs a scheduling slice rather than a
    compile: the wrapper's exit status is the subject, not the driver's work.
    """
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
