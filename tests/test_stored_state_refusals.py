"""Assert the stored-state refusals from the arms' own receipts.

The arms are produced by ``scripts/prototypes/stored_state_refusals/
run_recorded_cases.py`` on a GPU lane, which drives the earlier comparison
record's own case reader, fixture machine, receipt mesh, stored seed, fitted
conductor currents and policy.  This module holds those receipts to the values
the record reports, each set beside the recorded value, and needs no GPU.

``STORED_STATE_REFUSALS_RECEIPTS`` names the directory holding the arm
receipts; unset, it is the durable cache path the runner's output is copied to.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

RECORD_SOURCE = (
    "docs/evidence/fragments/inverse-equilibrium-baseline/efit-comparison.html"
)
RECORDED_RESIDUAL = 0.030364200526436996
RECORDED_REASON = 12
RESIDUAL_BAND = 3.0
ARMS = (
    "tcv-stored-default",
    "tcv-stored-tightened",
    "iter-stored-default",
    "iter-stored-tightened",
)


def _receipts():
    directory = os.environ.get("STORED_STATE_REFUSALS_RECEIPTS")
    if directory is None:
        directory = "~/.cache/nova-stored-state-refusals/recorded-cases"
    path = Path(directory).expanduser()
    if not path.is_dir():
        raise AssertionError(f"no arm receipts at {path}")
    return path


def _receipt(label):
    path = _receipts() / f"{label}.json"
    if not path.is_file():
        raise AssertionError(f"arm {label} left no receipt at {path}")
    return json.loads(path.read_text())


def _measurement():
    return json.loads((_receipts() / "measurement.json").read_text())


def test_every_arm_is_unqualified_and_states_the_record():
    """Every arm refuses, and carries the recorded value beside its own."""
    for label in ARMS:
        receipt = _receipt(label)
        assert receipt["qualified"] is False, label
        assert receipt["converged"] is False, label
        band = receipt["recorded_band"]
        assert band["recorded_termination_reason"] in (3, 12), label
        assert band["measured_termination_reason"] == (receipt["termination_reason"]), (
            label
        )


def test_tcv_stops_at_the_recorded_residual():
    """The TCV arm built on the record's construction stops where it stopped."""
    receipt = _receipt("tcv-stored-tightened")
    assert receipt["case"] == "tcv"
    assert receipt["termination_reason"] == RECORDED_REASON
    residual = receipt["residual"]
    assert residual is not None and residual > 0.0, (
        f"TCV stopped with no finite residual ({receipt['residual_state']})"
    )
    factor = residual / RECORDED_RESIDUAL
    assert 1.0 / RESIDUAL_BAND <= factor <= RESIDUAL_BAND, (
        f"TCV residual {residual} is {factor:.3f}x the recorded {RECORDED_RESIDUAL}"
    )
    band = receipt["recorded_band"]
    assert band["recorded_residual"] == RECORDED_RESIDUAL
    assert band["within_factor_three"] is True


def test_iter_admits_no_magnetic_axis():
    """The ITER arm built on the record's construction admits no axis."""
    receipt = _receipt("iter-stored-tightened")
    assert receipt["case"] == "iter"
    assert receipt["termination_reason"] == 3
    assert receipt["residual"] is None
    axis = receipt["axis"]
    assert axis["admitted"] is False
    assert axis["error"] == "NoQualifiedAxisError"
    band = receipt["recorded_band"]
    assert band["recorded_axis_admitted"] is False
    assert band["measured_axis_admitted"] is False


def test_each_arm_is_the_comparison_records_own_fixture():
    """Every arm names the comparison module, its mesh, route and coefficients."""
    measurement = _measurement()
    assert measurement["measurement_module"].endswith(
        "tests/test_magnetics_fit_efit_comparison.py"
    )
    assert measurement["record_source"] == RECORD_SOURCE
    coefficients = {}
    for case in ("tcv", "iter"):
        receipt = _receipt(f"{case}-stored-default")
        assert receipt["seed_policy"] == "ExplicitSolveSeed"
        assert receipt["route"] == "newton_krylov"
        assert receipt["current_present"] is True
        mesh = receipt["mesh"]
        assert mesh["kind"] == "StencilMesh"
        assert mesh["nodes"] > 0
        assert mesh["stencil_rows"] >= mesh["nodes"] - 1
        assert len(receipt["coefficients"]) >= 3
        coefficients[case] = receipt["coefficients"]
    assert coefficients["tcv"] != coefficients["iter"]


def test_each_panel_draws_both_series():
    """A panel that carries a solved state draws both series on shared levels."""
    for label in ARMS:
        receipt = _receipt(label)
        if receipt.get("panel") is None:
            assert receipt.get("panel_refused"), receipt
            continue
        assert Path(receipt["panel"]).is_file(), receipt["panel"]
        drawn = receipt["panel_record"]
        assert drawn["imas_efit_levels_drawn"] > 0, label
        assert drawn["nova_levels_drawn"] > 0, label
        assert drawn["imas_efit_range"] is not None, label
        assert drawn["nova_range"] is not None, label
