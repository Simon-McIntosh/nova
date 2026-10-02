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

import pytest

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


def _h200_submission_line(result: subprocess.CompletedProcess[str]) -> str:
    return next(
        line for line in result.stdout.splitlines() if line.startswith("RUNG=h200 ")
    )


def test_dry_run_pytest_uses_a_long_default_per_test_timeout(tmp_path: Path) -> None:
    result = _launch(
        "--dry-run",
        "--log",
        str(tmp_path / "lane.log"),
        "--",
        "tests/example_target.py",
    )

    assert result.returncode == 0, result.stderr
    assert "--timeout 3600" in _h200_submission_line(result)


def test_dry_run_pytest_keeps_the_callers_timeout(tmp_path: Path) -> None:
    result = _launch(
        "--dry-run",
        "--log",
        str(tmp_path / "lane.log"),
        "--",
        "tests/example_target.py",
        "--timeout",
        "45",
    )

    assert result.returncode == 0, result.stderr
    line = _h200_submission_line(result)
    assert "--timeout 45" in line
    assert "--timeout 3600" not in line


def test_dry_run_pytest_keeps_the_callers_equals_form_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NOVA_LANE_TEST_TIMEOUT", "7200")
    result = _launch(
        "--dry-run",
        "--log",
        str(tmp_path / "lane.log"),
        "--",
        "tests/example_target.py",
        "--timeout=45",
    )

    assert result.returncode == 0, result.stderr
    line = _h200_submission_line(result)
    assert "--timeout=45" in line
    assert "--timeout 3600" not in line
    assert "--timeout 7200" not in line
    assert line.count("--timeout") == 1


def test_dry_run_pytest_reads_the_timeout_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NOVA_LANE_TEST_TIMEOUT", "7200")
    result = _launch(
        "--dry-run",
        "--log",
        str(tmp_path / "lane.log"),
        "--",
        "tests/example_target.py",
    )

    assert result.returncode == 0, result.stderr
    assert "--timeout 7200" in _h200_submission_line(result)


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


H200_CALLER = REPOSITORY_ROOT / "scripts" / "h200_test_lane" / "run.sh"


def test_h200_payload_keeps_sampler_and_pin_diagnostic(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(H200_CALLER),
            "--dry-run",
            "--log",
            str(tmp_path / "lane.log"),
            "--",
            "tests/example_target.py",
        ],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "TMPDIR": "/tmp"},
    )
    assert result.returncode == 0, result.stderr
    assert "sample_gpu_utilisation" in result.stdout
    assert "PINNED_REVISION" in result.stdout


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


SBATCH_FAKE = "\n".join(
    [
        "#!/usr/bin/env bash",
        'count=$(cat "${FAKE_LANE_STATE}" 2>/dev/null || echo 12344)',
        "next=$((count + 1))",
        'printf "%s" "${next}" > "${FAKE_LANE_STATE}"',
        'printf "%s;fakecluster\\n" "${next}"',
    ]
)

SQUEUE_FAKE = "\n".join(
    [
        "#!/usr/bin/env bash",
        'job=""',
        'while [ $# -gt 0 ]; do case "$1" in -j) job=$2; shift 2 ;;',
        "*) shift ;; esac; done",
        'if [ "$job" = "12345" ]; then printf "PENDING ReqNodeNotAvail\\n"; fi',
    ]
)

SCANCEL_FAKE = "#!/usr/bin/env bash\nexit 0\n"


def test_pending_resource_reason_falls_to_next_rung(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "sbatch", SBATCH_FAKE)
    _write_executable(fake_bin / "squeue", SQUEUE_FAKE)
    _write_executable(fake_bin / "scancel", SCANCEL_FAKE)
    environment = {
        **os.environ,
        "TMPDIR": "/tmp",
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "NOVA_LANE_PENDING_WAIT_SECONDS": "5",
        "FAKE_LANE_STATE": str(tmp_path / "job-counter"),
    }
    environment.pop("SLURM_JOB_ID", None)
    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--rungs",
            "h200,titan",
            "--log",
            str(tmp_path / "lane.log"),
            "--",
            "tests/example_target.py",
        ],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
    )
    assert "RUNG_REFUSED rung=h200 job=12345" in result.stdout
    assert "SLURM_JOB_ID=12346 RUNG=titan" in result.stdout
    assert result.returncode == 0, result.stderr
