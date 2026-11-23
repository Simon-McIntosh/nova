"""Check the measurement driver forwards and records what a receipt claims.

The driver is a script rather than a library, and its stored-case path is the
one the record reads, so the two decisions that a receipt depends on are
pinned here: the fitter is asked for the landed coupled step rule with the
run's own flux-travel bound, and a case whose fit did not converge does not
spend a second full fit on a gain-error control no band reads.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def _driver():
    spec = importlib.util.spec_from_file_location("fit_iteration_run", HERE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class _Measurement:
    measured: np.ndarray | None = None
    vacuum: np.ndarray | None = None
    scale: np.ndarray | None = None
    plasma_current: float = 0.0
    vacuum_flux: np.ndarray | None = None
    conductor_current: np.ndarray | None = None
    conductor_scale: np.ndarray | None = None


class _Terminal:
    def __init__(self, current: float) -> None:
        self.flux = np.zeros((2, 2))
        self.cell_current = np.array([current / 2.0, current / 2.0])
        self.topology = SimpleNamespace(axis=np.array([0.5, 0.0]), flux_span=1.0)


def _fitted(converged: bool, current: float = 1.0e6):
    passes = [
        SimpleNamespace(flux_update=0.5, misfit=3.0, fraction=1.0, trials=1),
        SimpleNamespace(flux_update=0.25, misfit=2.0, fraction=1.0, trials=1),
    ]
    return SimpleNamespace(
        converged=converged,
        terminal_misfit=2.0,
        terminal_state=_Terminal(current),
        last_flux_update=0.25,
        flux_evaluations=2,
        sensor_evaluations=2,
        wall_seconds=1.0,
        stalled_pass=None,
        stalled_message=None,
        axis_lost_pass=None,
        axis_loss_message=None,
        rejected_pass=None,
        rejected_message=None,
        topology_error=None,
        coefficients=np.array([1.0]),
        coefficient_names=["basis_0"],
        singular_values=np.array([1.0]),
        undetermined_coefficients=[],
        passes=passes,
        stopped_improving_pass=1,
        relaxation=1.0,
        max_coefficient_step=0.1,
        whitened_residuals=np.zeros(2),
        first_pass_values=np.zeros(2),
        first_pass_residual=np.zeros(2),
        source_profile=None,
    )


def _case():
    measurement = _Measurement(
        measured=np.array([2.0, 2.0]),
        vacuum=np.array([0.5, 0.5]),
        scale=np.ones(2),
        plasma_current=1.0e6,
        vacuum_flux=np.zeros(2),
        conductor_current=np.ones(1),
        conductor_scale=np.ones(1),
    )
    return {
        "name": "stored",
        "profile": object(),
        "magnetics": object(),
        "rows": lambda current: np.asarray(current),
        "basis": object(),
        "conductor_response": object(),
        "conductor_names": ["conductor"],
        "prescribed_current": 0.0,
        "measurement": measurement,
        "initial_flux": np.zeros((2, 2)),
        "plasma_current": 1.0e6,
        "reference_plasma_current": 1.0e6,
        "data": {
            "plasma_current_scale": 1.0,
            "constraint_context": {"comparison_label": "test label"},
            "reference_eq": object(),
            "reference_slice": 0,
        },
    }


class _Fitter:
    """Record each fit request and answer it with the requested terminal state."""

    requests: list[dict] = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args

    def fit(self, measurement, initial_flux, **kwargs):
        type(self).requests.append(dict(kwargs))
        return _fitted(converged=False, current=float(measurement.plasma_current))


@pytest.fixture
def patched(monkeypatch):
    run = _driver()
    _Fitter.requests = []
    flux_module = types.ModuleType("nova.equilibrium.flux_iteration")
    flux_module.FluxIterationFit = _Fitter
    study = types.ModuleType("tests.test_magnetics_fit_efit_comparison")
    study._panel = lambda *args, **kwargs: None
    study._as_equilibrium = lambda *args, **kwargs: (object(), 0.0)
    study._reference_centroid = lambda data: 0.0
    referee = types.ModuleType("nova.imas.mast_efit_referee")
    referee.score_with_efit_referee = lambda *args, **kwargs: {"bands": {}}
    for name in ("nova", "nova.equilibrium", "tests"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "nova.equilibrium.flux_iteration", flux_module)
    monkeypatch.setitem(sys.modules, "nova.imas.mast_efit_referee", referee)
    monkeypatch.setitem(sys.modules, "tests.test_magnetics_fit_efit_comparison", study)
    args = SimpleNamespace(
        max_passes=96,
        flux_tolerance=1.0e-10,
        relaxation=1.0,
        max_update=0.75,
    )
    return run, args


def test_stored_fit_requests_the_landed_rule_and_records_it(patched):
    run, args = patched
    row = run._run_stored(_case(), args)

    assert row["step_rule"] == run.STEP_RULE == "coupled"
    assert row["max_passes"] == 96
    for request in _Fitter.requests:
        assert request["step_rule"] == "coupled"
        assert request["max_update"] == 0.75
        assert request["flux_tolerance"] == 1.0e-10
        assert request["max_passes"] == 96


def test_unconverged_case_spends_no_control_fit(patched):
    run, args = patched
    row = run._run_stored(_case(), args)

    assert len(_Fitter.requests) == 1
    assert row["scaled_control_judged"] is False
    assert row["scaled_control_relative_error"] is None
    assert row["scaled_control_fired"] is False
    assert row["scaled_control_error"].startswith("not judged")


def test_converged_case_runs_the_gain_error_control(patched):
    run, args = patched

    class Converged(_Fitter):
        def fit(self, measurement, initial_flux, **kwargs):
            type(self).requests.append(dict(kwargs))
            current = float(measurement.plasma_current)
            return _fitted(converged=True, current=current)

    sys.modules["nova.equilibrium.flux_iteration"].FluxIterationFit = Converged
    row = run._run_stored(_case(), args)

    assert len(_Fitter.requests) == 2
    assert row["scaled_control_judged"] is True
    assert row["scaled_control_relative_error"] == pytest.approx(1.10 - 1.0)
    assert row["scaled_control_fired"] is True
    assert row["scaled_control_error"] is None