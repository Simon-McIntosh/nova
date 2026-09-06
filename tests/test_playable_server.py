"""The playable serving launcher's dry-run path resolution.

The launcher's ``--dry-run`` must resolve every launch path from the checkout
root and fail when any required path is missing — without ever touching the
scheduler — because the dry run is the only part of the serving contract that
runs off the cluster.  These tests pin that resolution and the fixed serving
contract values (port, job name, partition, reservation, websocket origins).
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "scripts" / "playable_server" / "run.sh"
STATUS = ROOT / "scripts" / "playable_server" / "status.sh"


def run_launcher(
    *, log: Path, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Invoke the launcher dry run from a neutral working directory."""
    environment = os.environ.copy()
    environment["PLAYABLE_ALLOCATION_LOG"] = str(
        Path.home() / ".local" / "share" / "nova" / "playable" / "allocation.log"
    )
    if extra_env:
        environment.update(extra_env)
    return subprocess.run(
        [str(RUN), "--log", str(log), "--dry-run"],
        cwd="/",
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )


def parse_dry_run(stdout: str) -> dict[str, str]:
    """Return the ``KEY=value`` lines the dry run prints."""
    entry = {}
    for line in stdout.splitlines():
        if "=" in line and not line.startswith("  "):
            key, _, value = line.partition("=")
            entry[key.strip()] = value.strip()
    return entry


def test_dry_run_resolves_every_path_from_the_checkout_root(tmp_path):
    result = run_launcher(log=tmp_path / "nova-playable.log")

    assert result.returncode == 0, f"dry run failed:\n{result.stdout}\n{result.stderr}"
    assert "MISSING_REQUIRED_PATHS" not in result.stderr
    entry = parse_dry_run(result.stdout)

    assert entry["PLAYABLE_ROOT"] == str(ROOT)
    assert entry["PLAYABLE_APP_DIR"] == str(ROOT / "apps" / "playable")
    assert Path(entry["PLAYABLE_APP_DIR"]).is_dir()
    assert Path(entry["PLAYABLE_PAYLOAD"]).is_file()
    assert Path(entry["PLAYABLE_PYTHON"]).is_file()
    assert Path(entry["PLAYABLE_CACHE_ROOT"]).parent.is_dir()
    assert entry["PLAYABLE_ALLOCATION_LOG"].endswith(
        os.path.join(".local", "share", "nova", "playable", "allocation.log")
    )
    assert entry["DRY_RUN_EXIT_STATUS"] == "0"


def test_dry_run_pins_the_serving_contract_values(tmp_path):
    result = run_launcher(log=tmp_path / "nova-playable.log")
    entry = parse_dry_run(result.stdout)

    assert entry["PLAYABLE_PORT"] == "18506"
    assert entry["PLAYABLE_JOB_NAME"] == "nova-playable"
    assert entry["PLAYABLE_PARTITION"] == "betelgeuse"
    assert entry["PLAYABLE_RESERVATION"] == "gpu_0003_grpA"
    assert entry["PLAYABLE_ORIGIN_PREFIX"] == "localhost:18506"


def test_dry_run_fails_when_a_required_path_is_missing(tmp_path):
    missing = str(tmp_path / "no-such-interpreter")
    result = run_launcher(
        log=tmp_path / "nova-playable.log",
        extra_env={"PLAYABLE_SERVER_PYTHON": missing},
    )

    assert result.returncode != 0
    assert "MISSING_REQUIRED_PATHS" in result.stderr
    assert f"PLAYABLE_PYTHON={missing}" in result.stderr


def test_status_command_reads_the_allocation_record(tmp_path):
    record = tmp_path / "allocation.log"
    record.write_text(
        "123456\t98dci4-gpu-0003\t18506\t2026-09-06T00:00:00Z\t"
        "0123456789abcdef0123456789abcdef01234567\n"
    )
    environment = os.environ.copy()
    environment["PLAYABLE_ALLOCATION_LOG"] = str(record)
    result = subprocess.run(
        [str(STATUS)],
        cwd="/",
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0
    assert "RECORD job=123456 node=98dci4-gpu-0003 port=18506" in result.stdout
    assert "revision=0123456789abcdef0123456789abcdef01234567" in result.stdout


@pytest.mark.parametrize(
    "parameter,delta",
    [("bulk_r", 0.02), ("bulk_z", 0.01), ("inner_gap", 0.005), ("elongation", 0.05)],
)
def test_warm_keyframe_commands_are_bound_app_keys(parameter, delta):
    """The warm-up's command set is a subset of the app's bound key steps."""
    from apps.playable.shape import STEPS

    assert STEPS[parameter] == delta
