"""Regression checks for width admission, repeat agreement and launcher gates."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts.prototypes.bank_width_probe import report, run


@pytest.fixture
def receipts(tmp_path, monkeypatch):
    directory = tmp_path / "receipts"
    source = directory / "measured-source/nova/equilibrium/support_read.py"
    source.parent.mkdir(parents=True)
    source.write_text("source identity\n")
    inputs = tmp_path / "inputs.npz"
    inputs.write_bytes(b"common population")
    summary = dict(
        revision="fixture-revision",
        complete=True,
        inert_control="inert lever: identical lowered programs and execution counts",
        input_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
        measurement_module="/scratch/nova/equilibrium/support_read.py",
        measurement_cwd="/scratch",
        largest_admitted_width=78,
        first_refused_width=79,
        arms=[],
    )
    monkeypatch.setattr(
        report.subprocess, "check_output", lambda command: b"source identity\n"
    )
    for allocator, width, process in (
        [("default", w, p) for w in (64, 78) for p in (0, 1, 2)]
        + [("default", w, 0) for w in (79, 80, 96, 256)]
        + [("async", 96, 0)]
    ):
        admitted = width in (64, 78) or allocator == "async"
        outcome = (
            "executed"
            if admitted
            else "compiled-memory-refusal"
            if width == 256
            else "runtime-oom"
        )
        exit_status = 0 if admitted else 77 if width == 256 else 1
        path = directory / f"{allocator}-{width}-{process}"
        path.mkdir()
        memory = 2000 if width == 256 else width
        record = dict(
            revision=summary["revision"],
            complete=outcome != "runtime-oom",
            module_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            module=summary["measurement_module"],
            cwd=summary["measurement_cwd"],
            inputs_sha256=summary["input_sha256"],
            inputs_path=str(inputs),
            device="NVIDIA H200 NVL",
            x64=True,
            programs={"solve": {"working_bytes": memory}},
            fresh_residual=[1e-12] * width,
            precision={k: [1e-6] * width for k in ("axis", "saddle", "boundary")},
        )
        if width == 256:
            record.update(memory_limited=True, limiting_program="solve")
        (path / "receipt.json").write_text(json.dumps(record))
        message = (
            "jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: "
            "Out of memory while trying to allocate 8GiB.\n"
            if outcome == "runtime-oom"
            else ""
        )
        (path / "gate.log").write_text(message + f"EXIT={exit_status}\n")
        summary["arms"].append(
            dict(
                allocator=allocator,
                width=width,
                process=process,
                allocator_setting="unset"
                if allocator == "default"
                else "cuda_malloc_async",
                admitted=admitted,
                outcome=outcome,
                exit_status=exit_status,
                working_bytes=memory,
                device_limit_bytes=1000,
                receipt=str(path / "receipt.json"),
                log=str(path / "gate.log"),
                batch_ms=float(width) if admitted else None,
                compile_seconds=1.0,
            )
        )
    (directory / "summary.json").write_text(json.dumps(summary))
    return directory, summary


def audit(receipts):
    directory, summary = receipts
    (directory / "summary.json").write_text(json.dumps(summary))
    return report.audit(directory, directory.parent)


def test_complete_family_accepts_runtime_refusals_and_separate_allocator(receipts):
    result = audit(receipts)
    assert result["largest_admitted_width"] == 78
    assert result["first_refused_width"] == 79
    rows = {(r["allocator"], r["width"]): r for r in result["widths"]}
    assert rows["default", 78]["spread"] == 0.0
    assert rows["async", 96]["admitted"]
    assert rows["async", 96]["spread"] is None
    assert rows["default", 96]["batch_ms"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_repeat",
        "disagreement",
        "nonadjacent",
        "missing_async",
        "false_execution",
        "incomplete",
    ],
)
def test_report_refuses_incomplete_or_inconsistent_families(receipts, mutation):
    _, summary = receipts
    if mutation == "missing_repeat":
        summary["arms"] = [
            a for a in summary["arms"] if not (a["width"] == 78 and a["process"] == 2)
        ]
    elif mutation == "disagreement":
        next(a for a in summary["arms"] if a["width"] == 78 and a["process"] == 2)[
            "batch_ms"
        ] *= 1.1
    elif mutation == "nonadjacent":
        summary["first_refused_width"] = 80
    elif mutation == "missing_async":
        summary["arms"] = [a for a in summary["arms"] if a["allocator"] != "async"]
    elif mutation == "false_execution":
        next(a for a in summary["arms"] if a["width"] == 79)["outcome"] = "executed"
    else:
        summary["complete"] = False
    with pytest.raises(AssertionError):
        audit(receipts)


def test_report_refuses_changed_input_bytes(receipts):
    directory, _ = receipts
    (directory.parent / "inputs.npz").write_bytes(b"different population")
    with pytest.raises(AssertionError):
        audit(receipts)


def simulated_processes(fault=None):
    calls = []

    def execute(width, process, allocator="default"):
        calls.append((width, process, allocator))
        admitted = width <= 78 or allocator == "async"
        if fault == "repeat_refused" and width == 78 and process == 2:
            admitted = False
        outcome = (
            "executed"
            if admitted
            else "compiled-memory-refusal"
            if width == 256
            else "runtime-oom"
        )
        if fault == "oversized_executed" and width == 256:
            admitted, outcome = True, "executed"
        wall = float(width)
        if fault == "disagreement" and width == 78 and process == 2:
            wall *= 1.1
        return dict(
            admitted=admitted,
            outcome=outcome,
            batch_ms=wall if admitted else None,
            hlo_sha256="same" if fault == "inert" else f"width-{width}",
        )

    return execute, calls


def test_bisection_reaches_adjacent_boundary_and_repeats_only_endpoints():
    execute, calls = simulated_processes()
    summary = {"complete": False}
    run.measure_widths(execute, summary, lambda: None)
    assert summary["complete"]
    assert (summary["largest_admitted_width"], summary["first_refused_width"]) == (
        78,
        79,
    )
    assert [w for w, p, a in calls if p == 0 and a == "default"] == [
        64,
        96,
        256,
        80,
        72,
        76,
        78,
        79,
    ]
    assert {(w, p) for w, p, _ in calls if p} == {(64, 1), (64, 2), (78, 1), (78, 2)}
    assert calls[-1] == (96, 0, "async")


@pytest.mark.parametrize(
    "fault", ["repeat_refused", "disagreement", "oversized_executed", "inert"]
)
def test_measurement_refuses_invalid_repeats_or_controls(fault):
    execute, _ = simulated_processes(fault)
    summary = {"complete": False}
    with pytest.raises((AssertionError, ValueError)):
        run.measure_widths(execute, summary, lambda: None)
    assert not summary["complete"]


@pytest.mark.parametrize(
    "message",
    ["out of memory", "jax.errors.JaxRuntimeError: INVALID_ARGUMENT: bad operand"],
)
def test_unwitnessed_runtime_failures_are_not_memory_refusals(message):
    with pytest.raises(RuntimeError, match="unclassified"):
        run.classify_outcome(1, {"complete": False, "programs": {"solve": {}}}, message)


@pytest.mark.parametrize("launcher", ["job.sh", "report_job.sh"])
def test_launchers_stop_when_contract_tests_fail(tmp_path, launcher):
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/sh\nif [ \"$1 $2\" = '-m pytest' ]; then "
        "echo CONTRACT_CONTROL_REFUSED; exit 23; fi\n"
        "echo MEASUREMENT_MUST_NOT_START; exit 99\n"
    )
    fake_python.chmod(0o755)
    root = Path(__file__).resolve().parents[3]
    command = [
        "bash",
        str(root / "scripts/prototypes/bank_width_probe" / launcher),
        str(tmp_path),
        str(tmp_path),
        "fixture-revision",
        str(tmp_path / "input"),
    ]
    if launcher == "job.sh":
        command.append(str(tmp_path / "figure.svg"))
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=dict(
            os.environ,
            NOVA_MEASUREMENT_PYTHON=str(fake_python),
            SLURM_JOB_ID="contract-check",
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 23, result.stdout + result.stderr
    assert "CONTRACT_CONTROL_REFUSED" in result.stdout
    assert "MEASUREMENT_MUST_NOT_START" not in result.stdout
    assert result.stdout.splitlines()[-1] == "EXIT=23"
