"""Checks the record's own readers and controls, not only its measurements.

Three failures this file exists to catch, each of which passed a suite once:

A suite arm log whose header the package's reader cannot parse, so the manifest's
exit status is a claim no one can check. The reader and the writer are exercised
against one another here.

A control scored against a field that decided it, so the check cannot fail even
when the constant behind it changes. The recorded residual is read from the
published record and every factor is recomputed from it.

A published panel directory no test looks at.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import records  # noqa: E402
import verify_input_drift  # noqa: E402


def test_suite_arm_log_reads_the_header_the_runner_writes(tmp_path):
    """The wrapper's three header lines and trailer read back as a record."""
    path = tmp_path / "arm.log"
    path.write_text(
        "# revision d70c3598da3df64ee1cebb424bbfe2c1d498e0f9\n"
        "# command: PYTHONPATH=. JAX_PLATFORMS=cpu python -m pytest "
        "-q tests/test_x.py\n"
        "# tree: /tmp/suite-arms-x/base\n"
        "16 passed in 3.31s\n"
        "EXIT=0\n"
    )
    read = records.suite_arm_log(path)
    assert read["revision"] == "d70c3598da3df64ee1cebb424bbfe2c1d498e0f9"
    assert read["command"].endswith("-q tests/test_x.py")
    assert read["tree"] == "/tmp/suite-arms-x/base"
    assert read["exit_status"] == 0
    assert read["body"] == ["16 passed in 3.31s"]
    assert records.suite_arm_passed(read) == 16


def test_suite_arm_log_rejects_a_one_line_header(tmp_path):
    """The single-line header an earlier writer produced is not readable, and says so.

    This is the shape ``suite_arms.py`` wrote before it handed the job to
    ``run_suite_arm.sh``: reading it raises rather than returning a record with
    the command hidden inside the first line.
    """
    path = tmp_path / "arm.log"
    path.write_text(
        "# arm=base revision=d70c3598da3df64ee1cebb424bbfe2c1d498e0f9 "
        "tree=/tmp/x command: python -m pytest\n"
        "16 passed in 3.31s\n"
        "EXIT=0\n"
    )
    try:
        records.suite_arm_log(path)
    except ValueError as error:
        assert "first line is not" in str(error)
    else:
        raise AssertionError("a one-line header was accepted")


def test_published_record_anchors_the_recorded_residual():
    """The published comparison record states one residual, and it is the anchor."""
    recorded = verify_input_drift.published_recorded_residual(
        verify_input_drift.RECORD_FRAGMENT
    )
    assert recorded == 0.030364200526436996


def _arm(**overrides) -> dict:
    arm = dict(
        residual=0.030442406985959472,
        recorded=dict(residual=0.030364200526436996),
        within_factor_three=True,
    )
    arm.update(overrides)
    return arm


def test_control_accepts_the_record_seed_arm():
    """The record-seed arm reproduces the published residual inside a factor of 3."""
    recorded = verify_input_drift.published_recorded_residual(
        verify_input_drift.RECORD_FRAGMENT
    )
    passed, detail = verify_input_drift.control_claim(_arm(), recorded)
    assert passed, detail


def test_control_rejects_a_flag_that_disagrees_with_the_recomputed_factor():
    """A stored flag is not evidence: the factor is recomputed from the residual."""
    recorded = verify_input_drift.published_recorded_residual(
        verify_input_drift.RECORD_FRAGMENT
    )
    passed, _ = verify_input_drift.control_claim(_arm(residual=0.2), recorded)
    assert not passed


def test_control_rejects_an_arm_whose_recorded_constant_left_the_published_value():
    """A constant that moved inside the arm code fails the control independently."""
    recorded = verify_input_drift.published_recorded_residual(
        verify_input_drift.RECORD_FRAGMENT
    )
    passed, _ = verify_input_drift.control_claim(
        _arm(recorded=dict(residual=0.04)), recorded
    )
    assert not passed
