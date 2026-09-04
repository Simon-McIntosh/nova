"""Command-line routing contracts for the MAST forward parity benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import efit_forward_parity_slice as parity


def _current_receipt() -> dict:
    return {
        "aggregate": {
            "shot_count": 1,
            "constrained_converged_plasma_roots": 0,
            "registered_tolerance_pass_count": 0,
            "verdict": "measured-current-route",
        }
    }


def _replay_receipt() -> dict:
    return {
        "aggregate": {
            "shot_count": 1,
            "outcome_counts": {"bounded_non_convergence": 1},
            "all_carried_tolerances_pass_count": 0,
            "verdict": "measured-replay-route",
        }
    }


def test_no_route_flag_selects_current_constrained_public_solve(monkeypatch) -> None:
    calls = []

    def current(store, bank, output, *, shots=None):
        calls.append(("current", store, bank, output, shots))
        return _current_receipt()

    monkeypatch.setattr(parity, "run_current_constrained", current)
    monkeypatch.setattr(
        parity,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("absolute-source replay must require its diagnostic flag")
        ),
    )

    parity.main(["--shot", "21986"])

    assert calls == [
        (
            "current",
            parity.SHOT_STORE,
            parity.DECOMPOSITION_BANK,
            parity.CURRENT_CONSTRAINED_OUTPUT,
            (21986,),
        )
    ]


def test_named_diagnostic_flag_selects_absolute_source_replay(monkeypatch) -> None:
    calls = []

    def replay(store, bank, output, shots=None):
        calls.append(("replay", store, bank, output, shots))
        return _replay_receipt()

    monkeypatch.setattr(parity, "run", replay)
    monkeypatch.setattr(
        parity,
        "run_current_constrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(
                "diagnostic replay must not enter the current-constrained route"
            )
        ),
    )

    output = Path("diagnostic-output")
    parity.main(
        ["--absolute-source-replay", "--shot", "21986", "--output", str(output)]
    )

    assert calls == [
        (
            "replay",
            parity.SHOT_STORE,
            parity.DECOMPOSITION_BANK,
            output,
            (21986,),
        )
    ]


def test_raising_diagnostic_replay_writes_failure_receipt(
    monkeypatch, tmp_path: Path
) -> None:
    class CompilationCache:
        def receipt(self) -> dict:
            return {
                "directory": "/cache/runtime-test",
                "version_key": "runtime-test",
            }

    monkeypatch.setattr(
        parity,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("the closest-state replay changed the bounded solve")
        ),
    )
    monkeypatch.setattr(
        parity,
        "configure_persistent_compilation_cache",
        lambda _root: CompilationCache(),
    )
    monkeypatch.setattr(
        parity,
        "_diagnostic_frames",
        lambda _bank, _shots: [{"shot": 21978, "slice_index": 35}],
    )
    monkeypatch.setattr(
        parity,
        "_execution_environment",
        lambda: {
            "jax_backend": "cpu",
            "device_kinds": ["test-device"],
            "device_count": 1,
            "slurm_job_id": "test-job",
        },
    )
    monkeypatch.setattr(parity, "_source_revision", lambda: "test-revision")

    with pytest.raises(
        RuntimeError, match="the closest-state replay changed the bounded solve"
    ):
        parity.main(
            [
                "--absolute-source-replay",
                "--shot",
                "21978",
                "--output",
                str(tmp_path),
            ]
        )

    receipt = json.loads((tmp_path / parity.FROZEN_SCORECARD_RECEIPT_NAME).read_text())
    assert receipt["execution_contract"]["invocation_route"] == (
        "absolute_source_replay_diagnostic"
    )
    assert receipt["status"] == "failed"
    assert receipt["failure"] == {
        "exception_class": "RuntimeError",
        "message": "the closest-state replay changed the bounded solve",
    }
    assert receipt["frame"] == {"shot": 21978, "slice_index": 35}
    assert receipt["backend"]["slurm_job_id"] == "test-job"
    assert receipt["source_revision"] == "test-revision"
    assert receipt["compilation_cache"] == {
        "directory": "/cache/runtime-test",
        "version_key": "runtime-test",
    }
