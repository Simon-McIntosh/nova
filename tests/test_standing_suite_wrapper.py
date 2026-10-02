"""Contract tests for the standing-suite wrapper.

The wrapper drives every test file in its own pytest process and prints one
aggregate line at the end. This module runs the real wrapper against three
tiny fixture files — one passing, one failing, one killed by a short per-file
bound — with the H200 routing disabled, and asserts the log it prints is the
one reckon's standing-suite runner parses: every result category present on
the final summary line, the killed file announced as an ``ERROR`` line and
folded into the ``errors during collection`` total.

The parse is asserted with reckon's own reader when it is importable, so the
test checks the contract at its source; the categories are asserted from the
line's grammar either way. The wrapper path is read from ``NOVA_SUITE_WRAPPER``
(default: the repository wrapper) so a mutation can be run against a modified
copy and shown to fail this test.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

standing_suite = pytest.importorskip(
    "reckon.crew.standing_suite",
    reason="reckon carries the parse contract the wrapper's log must satisfy",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = Path(
    os.environ.get(
        "NOVA_SUITE_WRAPPER",
        str(REPO_ROOT / "scripts/nova_lane/standing_suite.sh"),
    )
)

PASSING = """
def test_passes():
    assert True
"""

FAILING = """
def test_fails():
    assert False
"""

KILLED = """
import time


def test_sleeps_past_the_per_file_bound():
    time.sleep(60)
"""

_CATEGORIES = ("passed", "failed", "errors", "skipped", "xfailed", "xpassed")
_SUMMARY_RE = re.compile(
    r"^\d+ passed, \d+ failed, \d+ errors, \d+ skipped, "
    r"\d+ xfailed, \d+ xpassed in \d+s$",
    re.MULTILINE,
)


@pytest.fixture()
def fixtures(tmp_path: Path) -> dict[str, Path]:
    """Three tiny files: one green, one red, one that outlives its bound."""

    directory = tmp_path / "fixtures"
    directory.mkdir()
    paths = {
        "pass": directory / "test_fixtures_pass.py",
        "fail": directory / "test_fixtures_fail.py",
        "killed": directory / "test_fixtures_killed.py",
    }
    paths["pass"].write_text(PASSING)
    paths["fail"].write_text(FAILING)
    paths["killed"].write_text(KILLED)
    return paths


def _run_wrapper(targets, log_dir, file_bound=4):
    """Run the wrapper over ``targets`` with the H200 routing disabled."""

    env = dict(os.environ)
    env["NOVA_SUITE_PYTHON"] = sys.executable
    env["NOVA_SUITE_TARGETS"] = " ".join(str(p) for p in targets)
    env["NOVA_SUITE_GPU_FILES"] = ""  # set and empty: no H200 routing
    env["NOVA_SUITE_FILE_TIMEOUT"] = str(file_bound)
    env["NOVA_SUITE_LOG_DIR"] = str(log_dir)
    return subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_summary_line_carries_every_category(fixtures, tmp_path):
    """The final line is the aggregate the runner parses, every category named."""

    completed = _run_wrapper(list(fixtures.values()), tmp_path / "logs")
    stdout = completed.stdout

    matches = list(_SUMMARY_RE.finditer(stdout))
    assert matches, f"no aggregate summary line in wrapper output:\n{stdout}"
    final = matches[-1].group(0)
    for word in _CATEGORIES:
        assert word in final, f"{word!r} missing from summary: {final!r}"

    counts = standing_suite._build_counts(stdout, completed.returncode)
    assert counts["collection_failed"] is True
    assert counts["passed"] == 1
    assert counts["failed"] == 1
    # The killed file is announced and folded into the collection-error total.
    assert "errors during collection" in stdout
    killed = [fid for fid in counts["failure_ids"] if "test_fixtures_killed" in fid]
    assert killed, f"killed file not among failure ids: {counts['failure_ids']}"


def test_killed_file_is_named_and_exits_red(fixtures, tmp_path):
    """A file over its bound exits the wrapper red and names the file."""

    completed = _run_wrapper(list(fixtures.values()), tmp_path / "logs", file_bound=3)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "ERROR " in completed.stdout
    assert "test_fixtures_killed.py" in completed.stdout
    assert "TIMEOUT" in completed.stdout


def test_list_mode_prints_the_errors_line(fixtures, tmp_path):
    """--list prints the per-file plan and exits 0 without running pytest."""

    completed = subprocess.run(
        ["bash", str(WRAPPER), "--list"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "cpu_pass_files=" in completed.stdout
    assert "h200_routed_files=" in completed.stdout
    assert "markexpr=" in completed.stdout
    # No pytest process was started, so no aggregate summary line was produced.
    assert not _SUMMARY_RE.search(completed.stdout)
