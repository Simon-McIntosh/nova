"""Durability checks for the stored-state refusal record.

Each check answers one way the record could rot between runs: a receipt ordered
by its filesystem time, a history read that exists twice and diverges, a
submission that inherits a node-local temporary directory, a suite log whose
revision or exit status cannot be read back.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import records  # noqa: E402


def test_receipts_order_by_written_sequence(tmp_path):
    """A copied or restored receipt keeps the order it was written with."""
    early = tmp_path / "early/measurement.json"
    late = tmp_path / "late/measurement.json"
    for path, sequence, value in ((early, 1, 0.5), (late, 2, 0.25)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"sequence": sequence, "arm": {"value": value}}))
    os.utime(early, (2_000_000_000, 2_000_000_000))
    os.utime(late, (1_000_000_000, 1_000_000_000))
    assert records.measurements(tmp_path)["arm"]["value"] == 0.25
    assert records.owner(tmp_path, "arm") == late
    assert records.next_sequence(tmp_path) == 3


def test_a_job_sequences_after_the_run_it_joins(tmp_path):
    """A job counts the run's receipts, never only its own directory.

    The receipt sits at the top of the job's output directory, one level below
    the run directory that holds every job. Counting the job directory alone
    finds nothing there, so every fresh job issues sequence 1 and a re-measured
    arm ties the run it was meant to supersede.
    """
    run = tmp_path / "run"
    for name, sequence in (("measurement", 1), ("tcv-iter-long", 7)):
        receipt = run / name / "measurement.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text(json.dumps({"sequence": sequence, "arm": {"value": name}}))
    assert records.next_sequence(run) == 8
    job = run / "mastu-titan3"
    job.mkdir()
    (job / "measurement.json").write_text(json.dumps({"sequence": 1}))
    assert records.next_sequence(job) == 2


def test_the_receipt_is_written_with_the_runs_next_sequence(tmp_path, monkeypatch):
    """The sequence the job writes is read back from the receipt it wrote."""
    run = tmp_path / "run"
    first = run / "first" / "measurement.json"
    first.parent.mkdir(parents=True)
    first.write_text(json.dumps({"sequence": 4}))
    output = run / "second"
    output.mkdir()
    figures = tmp_path / "figures"
    figures.mkdir()
    monkeypatch.setenv("STORED_STATE_REFUSALS_OUTPUT", str(output))
    monkeypatch.setenv("STORED_STATE_REFUSALS_FIGURES", str(figures))
    run_cases = importlib.reload(importlib.import_module("run_cases"))
    run_cases._report({"arm": {"value": 5}})
    written = json.loads((output / "measurement.json").read_text())
    assert written["sequence"] == 5
    run_cases._report({"arm": {"value": 6}})
    written = json.loads((output / "measurement.json").read_text())
    assert written["sequence"] == 6
    assert records.measurements(run)["arm"]["value"] == 6


def _local_history_readers(module):
    return [
        name
        for name, value in vars(module).items()
        if inspect.isfunction(value)
        and value.__module__ == module.__name__
        and ("history" in name or "residual" in name)
    ]


def test_the_history_reader_is_defined_once(tmp_path, monkeypatch):
    """Both callers read a rejected history through one function."""
    report = importlib.import_module("report")
    for name, directory in (
        ("STORED_STATE_REFUSALS_OUTPUT", tmp_path / "output"),
        ("STORED_STATE_REFUSALS_FIGURES", tmp_path / "figures"),
    ):
        directory.mkdir()
        monkeypatch.setenv(name, str(directory))
    run_cases = importlib.import_module("run_cases")
    assert report.residual_fields is records.residual_fields
    assert run_cases.residual_fields is records.residual_fields
    assert _local_history_readers(report) == []
    assert _local_history_readers(run_cases) == []


def test_submit_sets_the_temporary_directory_before_sbatch():
    """slurmstepd makes the inherited TMPDIR before the payload's export runs.

    The payload's own ``export TMPDIR`` sits inside the heredoc that precedes
    the ``sbatch`` line, so a check on the file's first occurrence proves
    nothing: the export has to be in the submitting shell, ahead of the payload
    text itself.
    """
    lines = (HERE / "submit.sh").read_text().splitlines()
    payload = next(
        index for index, line in enumerate(lines) if 'cat > "${payload_file}"' in line
    )
    exported = [
        index for index, line in enumerate(lines) if line.startswith("export TMPDIR=")
    ]
    assert exported and min(exported) < payload


def test_a_suite_arm_log_carries_its_revision_command_and_exit(tmp_path):
    log = tmp_path / "arm.log"
    log.write_text(
        "# revision " + "a" * 40 + "\n# command: /shared/python -m pytest -q x.py\n"
        "4 passed in 87.00s\nEXIT=0\n"
    )
    read = records.suite_arm_log(log)
    assert read["revision"] == "a" * 40
    assert read["command"].endswith("-q x.py")
    assert read["exit_status"] == 0
    assert read["body"] == ["4 passed in 87.00s"]

    log.write_text("# revision " + "b" * 40 + "\npytest output only\n")
    with pytest.raises(ValueError, match="second line"):
        records.suite_arm_log(log)
