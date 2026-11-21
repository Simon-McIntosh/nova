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


def test_tcv_refuses_at_the_recorded_residual():
    """TCV's stored-field solve stops where the record says it stops."""
    receipt = _receipt("tcv-stored-default")
    assert receipt["case"] == "tcv"
    assert receipt["qualified"] is False
    assert receipt["converged"] is False
    assert receipt["termination_reason"] == RECORDED_REASON
    residual = receipt["residual"]
    assert residual is not None and residual > 0.0, (
        f"TCV refused with no finite residual ({receipt.get('residual_state')})"
    )
    factor = residual / RECORDED_RESIDUAL
    assert 1.0 / RESIDUAL_BAND <= factor <= RESIDUAL_BAND, (
        f"TCV residual {residual} is {factor:.3f}x the recorded {RECORDED_RESIDUAL}"
    )
    band = receipt["recorded_band"]
    assert band["recorded_residual"] == RECORDED_RESIDUAL
    assert band["measured_termination_reason"] == RECORDED_REASON
    assert band["within_factor_three"] is True


def test_iter_admits_no_magnetic_axis():
    """ITER's stored-field solve finds no qualified axis, as the record says."""
    receipt = _receipt("iter-stored-default")
    assert receipt["case"] == "iter"
    assert receipt["qualified"] is False
    assert receipt["converged"] is False
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
        label = f"{case}-stored-default"
        receipt = _receipt(label)
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
    tcv = _receipt("tcv-stored-default")
    assert "panel_refused" not in tcv, tcv.get("panel_refused")
    assert Path(tcv["panel"]).is_file(), tcv["panel"]
    drawn = tcv["panel_record"]
    assert drawn["imas_efit_levels_drawn"] > 0
    assert drawn["nova_levels_drawn"] > 0
    assert drawn["imas_efit_range"] is not None
    assert drawn["nova_range"] is not None


def test_an_arm_with_no_state_states_its_refusal():
    """An arm that leaves no state says so instead of showing an empty panel."""
    for case in ("tcv", "iter"):
        receipt = _receipt(f"{case}-stored-default")
        if receipt.get("panel"):
            assert Path(receipt["panel"]).is_file()
        else:
            assert receipt.get("panel_refused"), receipt
