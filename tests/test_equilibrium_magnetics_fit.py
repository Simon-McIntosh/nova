"""Profile-coefficient fitting and its public-forward authority boundary."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import MagneticsFit, MagneticsFitReceipt
from nova.equilibrium import magnetics_fit
from nova.equilibrium.measurement import Magnetics, SliceMeasurement, whitened_solve
from nova.equilibrium.solve_request import ForwardSolveReceipt, ForwardSolveRequest
from nova.equilibrium.source import DomainProfile, ForwardSource, PolynomialFluxFunction
from nova.jax.config import configure_dtypes
from tests.magnetics_fit_harness import _persist_forward_failure


def test_refused_residuals_preserve_strict_json_receipts(tmp_path):
    from tests.test_magnetics_fit_efit_comparison import _residual_fields, _write

    path = tmp_path / "receipt.json"
    for value, state in [
        (float("nan"), "nan"),
        (float("inf"), "positive_infinity"),
        (-float("inf"), "negative_infinity"),
    ]:
        _write(path, dict(marker="refusal-recorded", **_residual_fields(value)))
        assert json.loads(path.read_text()) == dict(
            marker="refusal-recorded", residual=None, residual_state=state
        )
    assert _residual_fields(0.0) == dict(residual=0.0, residual_state="finite")
    assert _residual_fields(None) == dict(residual=None, residual_state="unavailable")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="not JSON compliant") as refusal:
        _write(path, dict(unexpected=float("nan")))
    print(f"STRICT_JSON_GUARD_REFUSED {refusal.value}")
    assert path.read_bytes() == before


def test_comparison_receipts_delegate_to_durable_json_owner(tmp_path, monkeypatch):
    from nova.io.json import write_atomic
    from tests import test_magnetics_fit_efit_comparison as comparison

    destination = tmp_path / "receipt.json"
    pending = destination.with_suffix(".pending")
    pending.write_text("another writer's pending receipt")
    assert pending.read_text() == "another writer's pending receipt"
    calls = []

    def record_owner(path, value, **options):
        calls.append((path, value, options))
        return write_atomic(path, value, **options)

    monkeypatch.setattr(comparison, "write_atomic", record_owner, raising=False)
    payload = dict(marker="durable-owner-reached", nested=dict(value=2))
    comparison._write(destination, payload)
    print(f"ATOMIC_OWNER_CALLS={len(calls)} FIXED_PENDING_PRESERVED={pending.exists()}")
    assert len(calls) == 1
    assert calls[0] == (
        destination,
        payload,
        dict(indent=2, sort_keys=False, newline=True),
    )
    assert (
        destination.read_text() == json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    assert pending.read_text() == "another writer's pending receipt"
    assert list(tmp_path.glob("*.tmp")) == []
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="not JSON compliant") as refusal:
        comparison._write(destination, dict(raw=float("nan")))
    assert len(calls) == 1
    assert destination.read_bytes() == before
    print(f"STRICT_JSON_BEFORE_ATOMIC_OWNER_REFUSED {refusal.value}")


def test_passive_zero_turns_remain_zero_response_weights():
    from tests.test_magnetics_fit_efit_comparison import _passive_signed_turns

    assert _passive_signed_turns(0.0) == 0.0
    assert _passive_signed_turns(-2.0) == -2.0
    with pytest.raises(ValueError, match="invalid passive"):
        _passive_signed_turns(float("nan"))


def test_unqualified_forward_receipt_preserves_convergence(tmp_path, monkeypatch):
    from tests import test_magnetics_fit_efit_comparison as comparison

    configure_dtypes()
    history = SimpleNamespace(residual=4.86e-10, converged=True)
    receipt = SimpleNamespace(
        qualified=False,
        termination_reason=1,
        terminal_state=SimpleNamespace(flux=np.array([1.0]), fixed_point=history),
    )
    data = dict(
        name="case",
        profile=SimpleNamespace(solve=lambda request: receipt),
        constraint_context={},
    )
    monkeypatch.setenv("NOVA_EFIT_COMPARISON_OUTPUT", str(tmp_path))
    persisted = []
    # The comparison module imports this helper from the harness at call time,
    # so the patch lands on the harness namespace and the stub is exercised.
    monkeypatch.setattr(
        sys.modules["tests.magnetics_fit_harness"],
        "_persist_forward",
        lambda *args: persisted.append(args),
    )
    monkeypatch.setattr(comparison, "_panel", lambda *args, **kwargs: None)
    assert comparison._attempt(data, None, "seed") is None
    assert persisted, "the comparison module stopped reaching the patched helper"
    comparison._record_forward_refusal(data)
    result = json.loads((tmp_path / "case-result.json").read_text())
    assert result["qualified"] is False
    assert result["converged"] is True
    assert result["termination_reason"] == 1
    assert result["residual"] == 4.86e-10


def test_failed_forward_history_counts_nonfinite_outer_trip(tmp_path):
    from nova.equilibrium.fixed_point import FixedPointResult

    history = FixedPointResult(
        state=np.array([1.0]),
        residual=float("nan"),
        trace=np.array([float("nan")]),
        active_set_iterations=4,
        active_set_residuals=np.array([1.0, 2.0, 3.0, float("nan")]),
    )
    try:
        raise RuntimeError("terminal topology refuses the fourth reconciliation")
    except RuntimeError as error:
        _persist_forward_failure(error, tmp_path, "refused")
    result = json.loads((tmp_path / "refused-generating-failure.json").read_text())
    assert result["trips"] == history.active_set_iterations
    assert result["finite_residual_entries"] == 3


def test_fitted_replay_family_guard_refuses_before_public_seeds():
    from tests.test_magnetics_fit_efit_comparison import _require_replay_agreement

    report = dict(
        conductor_state="stored fitted",
        equilibrium_solves=0,
        wall_seconds=1.0,
        source_cells=2,
        current_relative_error=0.0,
        families={
            "probes": dict(nova_vs_efit=dict(reference_rms=1.0, relative_l2=0.01))
        },
    )
    _require_replay_agreement(report)
    report["families"]["probes"]["nova_vs_efit"]["relative_l2"] = 0.0736
    with pytest.raises(ValueError, match="physical fit refused") as refusal:
        _require_replay_agreement(report)
    print(f"REPLAY_ADMISSION_GUARD_REFUSED {refusal.value}")


def test_fitted_circuit_binding_preserves_output_split_and_shared_members(
    tmp_path, monkeypatch
):
    from tests.test_magnetics_fit_efit_comparison import _fitted_current_binding
    from tests.test_equilibrium_forward_reference import Conductor

    @dataclass
    class Case:
        time: float
        active: tuple
        passive: tuple

    coils = [
        SimpleNamespace(name=str(i), current=SimpleNamespace(data=np.array([10.0])))
        for i in range(3)
    ]
    active = SimpleNamespace(
        supply=[None],
        coil=coils,
        circuit=[SimpleNamespace(connections=np.array([[0, -1, 1, 0], [0, 0, -1, 1]]))],
        time=np.array([0.0]),
        ids_properties=SimpleNamespace(homogeneous_time=1),
    )
    passive = SimpleNamespace(
        loop=[
            SimpleNamespace(
                name="wall",
                current=np.array([4.0]),
                element=[SimpleNamespace(geometry=SimpleNamespace(geometry_type=2))],
            )
        ],
        time=np.array([0.0]),
        ids_properties=SimpleNamespace(homogeneous_time=1),
    )

    def row(measured, fitted):
        return SimpleNamespace(measured=measured, reconstructed=fitted, weight=1.0)

    constraints = SimpleNamespace(
        pf_current=[row(10.0, 2.0), row(10.0, 3.0), row(4.0, 5.0)],
        pf_passive_current=[],
    )
    case = Case(
        0.0,
        tuple(
            Conductor(str(i), 10.0, 1.0, rectangle=(1.0, 0.0, 0.1, 0.1))
            for i in range(3)
        ),
        (Conductor("wall", 4.0, 1.0, rectangle=(1.0, 0.0, 0.1, 0.1)),),
    )
    data = dict(
        name="synthetic",
        case=case,
        input_ids=dict(pf_active=active, pf_passive=passive),
        reference_eq=SimpleNamespace(
            time_slice=[SimpleNamespace(constraints=constraints)]
        ),
        reference_slice=0,
        conductor_names=("0", "1", "2", "passive:wall"),
        projection=np.eye(4),
    )
    monkeypatch.setenv("NOVA_EFIT_COMPARISON_OUTPUT", str(tmp_path))
    result = _fitted_current_binding(data)
    np.testing.assert_array_equal(
        result["fitted_conductor_current"], [2.0, 5.0, 3.0, 5.0]
    )
    assert result["fitted_current_map"][2]["source_kind"] == "passive"
    assert result["fitted_current_map"][2]["output"] == "pf_current[2]"
    constraints.pf_current[2].measured = 99.0
    with pytest.raises(ValueError, match="unique measured-row witness") as refusal:
        _fitted_current_binding(data)
    print(f"CIRCUIT_WITNESS_GUARD_REFUSED {refusal.value}")


def test_rank_deficient_data_exposes_its_null_direction():
    response = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    target = np.array([2.0, 4.0, 6.0])
    coefficient, covariance, spectrum, null = whitened_solve(
        response, target, np.ones(3), 0.0, diagnostics=True
    )
    np.testing.assert_allclose(response @ coefficient, target, atol=2.0e-15)
    assert np.all(np.isfinite(covariance))
    assert spectrum[0] > 1.0
    assert spectrum[1] < 1.0e-14
    assert null.shape == (1, 2)
    np.testing.assert_allclose(response @ null.T, 0.0, atol=2.0e-15)


def test_regularisation_does_not_claim_data_identifiability():
    response = np.array([[1.0, 1.0]])
    _, _, spectrum, null = whitened_solve(
        response,
        np.array([1.0]),
        np.ones(1),
        1.0e-3,
        diagnostics=True,
        penalty=np.array([1.0, 4.0]),
    )
    assert spectrum.size == 1
    assert null.shape == (1, 2)
    np.testing.assert_allclose(response @ null.T, 0.0, atol=1.0e-15)


def test_current_anchor_participates_in_the_data_spectrum():
    response = np.array([[1.0, 1.0]])
    coefficient, _, spectrum, null = whitened_solve(
        response,
        np.array([3.0]),
        np.array([0.5]),
        0.0,
        anchor=(np.array([1.0, -1.0]), 1.0, 2.0),
        diagnostics=True,
    )
    np.testing.assert_allclose(coefficient, [2.0, 1.0], atol=1.0e-15)
    assert spectrum.shape == (2,)
    assert spectrum[-1] > 0.0
    assert null.shape == (0, 2)


def test_fit_receipt_and_forward_request_keep_the_authority_boundary():
    assert MagneticsFitReceipt.__module__ == "nova.equilibrium.magnetics_fit"
    assert MagneticsFit.__module__ == "nova.equilibrium.magnetics_fit"
    names = {item.name for item in fields(ForwardSolveRequest)}
    assert "measurement" not in names
    with pytest.raises(TypeError, match="measurement"):
        ForwardSolveRequest(measurement=np.zeros(4))


@pytest.fixture
def linear_fit(monkeypatch):
    """An algebraic instrument checks orchestration without a physics claim."""
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    source = ForwardSource(
        core=DomainProfile(
            p_prime=PolynomialFluxFunction(jnp.array([0.8, 0.1])),
            ff_prime=PolynomialFluxFunction(jnp.array([1.2])),
        )
    )
    request = ForwardSolveRequest.from_defaults(
        carrier_identity="linear-response",
        source_profile=source,
        seed_policy=magnetics_fit.ExplicitSolveSeed(jnp.zeros(3)),
    )
    response = np.array(
        [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [2.0, 1.0, 1.0], [1.0, -2.0, 1.0]]
    )
    calls = []

    def solve(active):
        assert isinstance(active, ForwardSolveRequest)
        calls.append(active)
        coefficients = np.r_[
            active.source_profile.core.p_prime.coefficients,
            active.source_profile.core.ff_prime.coefficients,
        ]
        return ForwardSolveReceipt(
            terminal_state=SimpleNamespace(
                flux=coefficients,
                cell_current=coefficients,
            ),
            qualified=True,
            termination_reason=0,
            residual_history=np.zeros(1),
            changed_mask_entries=np.zeros(1),
            support_cell_count=np.ones(1),
            globalisation_decisions=([], []),
            amplitude_history=np.ones(1),
            topology_read=None,
            polish_receipt=None,
            compilation_cache_hit=True,
            wall_seconds=0.0,
            resolved_defaults=None,
        )

    def signal(biot, coordinates, *args):
        indices = coordinates[:, 0].astype(int) - 1
        return SimpleNamespace(values=response[indices] @ biot)

    monkeypatch.setattr(magnetics_fit, "virtual_flux_loops", signal)
    monkeypatch.setattr(magnetics_fit, "virtual_poloidal_probes", signal)
    geometry = Magnetics(
        r=np.arange(1.0, 5.0),
        z=np.zeros(4),
        angle=np.zeros(4),
        flux_loop=np.array([True, False, True, False]),
    )
    fitter = MagneticsFit(
        SimpleNamespace(solve=solve),
        request,
        geometry,
        lambda receipt: receipt.terminal_state.flux,
    )
    truth = np.array([1.0, 0.3, 0.9])
    measurement = SliceMeasurement(
        measured=response @ truth,
        vacuum=np.zeros(4),
        mask=np.ones(4, dtype=bool),
        scale=np.array([0.5, 2.0, 1.0, 1.0]),
        plasma_current=truth.sum(),
    )
    return fitter, measurement, truth, calls


def test_coefficient_updates_issue_only_typed_source_requests(linear_fit):
    fitter, measurement, truth, calls = linear_fit
    receipt = fitter.fit(measurement, plasma_current_scale=1.0)
    assert receipt.converged
    assert receipt.iterations > 0
    assert receipt.forward_evaluations == len(calls)
    np.testing.assert_allclose(receipt.coefficients, truth, atol=1.0e-11)
    assert np.max(np.abs(receipt.whitened_residuals)) <= 1.0e-8
    assert receipt.undetermined_coefficients == ()
    assert receipt.singular_values.shape == (3,)
    assert all(active.target_current is None for active in calls)
    np.testing.assert_array_equal(
        fitter.request.source_profile.core.p_prime.coefficients, [0.8, 0.1]
    )


def _refuse_large_fit_steps(fitter, limit):
    solve = fitter.profile.solve

    def bounded(request):
        receipt = solve(request)
        delta = float(
            np.max(np.abs(receipt.terminal_state.flux - request.seed_policy.state))
        )
        return replace(
            receipt,
            qualified=delta <= limit,
            termination_reason=12 if delta > limit else 1,
            terminal_state=SimpleNamespace(
                **vars(receipt.terminal_state),
                fixed_point=SimpleNamespace(residual=delta, converged=delta <= limit),
            ),
        )

    fitter.profile.solve = bounded


def test_unqualified_fit_trial_backtracks_to_a_qualified_step(linear_fit):
    fitter, measurement, truth, calls = linear_fit
    # The initial public request uses a different seed from subsequent trials.
    fitter.request = replace(
        fitter.request, seed_policy=magnetics_fit.ExplicitSolveSeed(fitter._initial)
    )
    _refuse_large_fit_steps(fitter, 0.08)
    receipt = fitter.fit(measurement, plasma_current_scale=1.0)
    assert receipt.converged
    np.testing.assert_allclose(receipt.coefficients, truth, atol=1e-8)
    assert receipt.forward_evaluations == len(calls)
    assert receipt.rejected_trials
    rejected = receipt.rejected_trials[0]
    assert rejected["fraction"] == 1.0
    assert rejected["termination_reason"] == 12
    assert rejected["residual"] > 0.08
    assert rejected["forward"].qualified is False


def test_unqualified_fit_trials_exhaust_the_backtracking_floor(linear_fit):
    fitter, measurement, _, calls = linear_fit
    fitter.request = replace(
        fitter.request, seed_policy=magnetics_fit.ExplicitSolveSeed(fitter._initial)
    )
    _refuse_large_fit_steps(fitter, 0.001)
    with pytest.raises(RuntimeError, match="line search.*12") as refusal:
        fitter.fit(measurement, plasma_current_scale=1.0, max_backtracks=4)
    receipt = refusal.value.fit_receipt
    assert not receipt.converged
    assert receipt.forward.qualified
    assert receipt.forward_evaluations == len(calls)
    assert [row["fraction"] for row in receipt.rejected_trials] == [
        1.0,
        0.5,
        0.25,
        0.125,
        0.0625,
    ]
    assert all(row["termination_reason"] == 12 for row in receipt.rejected_trials)
    assert all(row["residual"] > 0.001 for row in receipt.rejected_trials)
    np.testing.assert_array_equal(receipt.coefficients, fitter._initial)
    print(f"UNQUALIFIED_TRIAL_FLOOR_REFUSED {refusal.value}")


def test_unqualified_fit_start_is_not_backtracked(linear_fit):
    fitter, measurement, _, calls = linear_fit
    solve = fitter.profile.solve
    fitter.profile.solve = lambda request: replace(solve(request), qualified=False)
    with pytest.raises(
        RuntimeError, match="forward evaluation did not qualify"
    ) as refusal:
        fitter.fit(measurement, plasma_current_scale=1.0)
    assert len(calls) == 1
    assert not refusal.value.receipt.qualified


def test_fit_does_not_swallow_unrelated_trial_errors(linear_fit, monkeypatch):
    fitter, measurement, _, _ = linear_fit
    evaluate = fitter._evaluate
    calls = 0

    def broken(coefficients, seed):
        nonlocal calls
        calls += 1
        if calls == 8:
            raise RuntimeError("observation source is unavailable")
        return evaluate(coefficients, seed)

    monkeypatch.setattr(fitter, "_evaluate", broken)
    with pytest.raises(RuntimeError, match="observation source is unavailable"):
        fitter.fit(measurement, plasma_current_scale=1.0)
    assert calls == 8


@pytest.fixture
def completed_comparison(linear_fit, monkeypatch, tmp_path):
    """Drive the physical comparison's reporting and admission with fitted receipts."""
    from tests import test_magnetics_fit_efit_comparison as comparison
    from nova.frame import coilset
    from nova.imas import mast_efit_referee

    request = linear_fit[0].request
    measurement = SliceMeasurement(
        measured=np.ones(1),
        vacuum=np.zeros(1),
        mask=np.ones(1, dtype=bool),
        scale=np.ones(1),
        plasma_current=1.0,
        conductor_current=np.ones(1),
        conductor_scale=np.full(1, 0.1),
    )

    class Source:
        def __init__(self, polygons):
            pass

        @staticmethod
        def _cache_factory(value):
            pass

    conductor = SimpleNamespace(
        coil=SimpleNamespace(insert=lambda *args, **kwargs: None),
        sloc={},
        point=None,
        probe=None,
        poloidal_flux_loop=None,
    )
    harness = sys.modules["tests.magnetics_fit_harness"]
    monkeypatch.setattr(harness, "_PlasmaDiagnosticSource", Source)
    monkeypatch.setattr(harness, "_physical_sensor_rows", lambda *args: np.ones(1))
    monkeypatch.setattr(harness, "_persist_forward", lambda *args: None)
    monkeypatch.setattr(coilset, "CoilSet", lambda **kwargs: conductor)
    monkeypatch.setattr(comparison, "_as_equilibrium", lambda *args: (None, 0.0))
    monkeypatch.setattr(comparison, "_reference_centroid", lambda data: 0.0)
    monkeypatch.setattr(comparison, "_panel", lambda *args, **kwargs: None)
    monkeypatch.setenv("NOVA_EFIT_COMPARISON_OUTPUT", str(tmp_path))

    def run(name, *, bands_pass=False, scaled_converged=True, scaled_qualified=True):
        def receipt(current, converged, qualified):
            return SimpleNamespace(
                converged=converged,
                coefficients=np.ones(1),
                rejected_trials=(),
                source_profile=request.source_profile,
                device="stub",
                forward_evaluations=1,
                whitened_residuals=np.zeros(2),
                forward=SimpleNamespace(
                    qualified=qualified,
                    terminal_state=SimpleNamespace(
                        flux=np.ones(1),
                        cell_current=np.array([current]),
                        fixed_point=SimpleNamespace(residual=1e-12, converged=True),
                    ),
                ),
            )

        fitted = receipt(1.0, True, True)
        fits = iter([fitted, receipt(1.10, scaled_converged, scaled_qualified)])
        monkeypatch.setattr(
            comparison,
            "MagneticsFit",
            lambda *args, **kwargs: SimpleNamespace(
                fit=lambda *args, **kwargs: next(fits)
            ),
        )
        score = dict(
            bands=dict(plasma_current_relative_error=bands_pass),
            plasma_current_relative_error=0.25,
        )
        monkeypatch.setattr(
            mast_efit_referee, "score_with_efit_referee", lambda *args, **kwargs: score
        )
        data = dict(
            name=name,
            machine=SimpleNamespace(cell_polygons=[]),
            case=SimpleNamespace(active=(), passive=(), plasma_current=1.0),
            projection=np.eye(1),
            geometry=None,
            request=request,
            profile=None,
            basis=None,
            source=request.source_profile,
            conductor_names=("coil",),
            measurement=measurement,
            plasma_current_scale=0.01,
            reference_eq=None,
            reference_slice=0,
            sensor_names=("loop",),
            reference_predictions=np.ones(1),
            constraint_context=comparison._constraint_context(name),
        )
        comparison._compare_fit(data, fitted.forward)
        return json.loads((tmp_path / f"{name}-result.json").read_text())

    run.output = tmp_path
    return run


def test_different_constraint_comparison_reports_band_gaps(completed_comparison):
    result = completed_comparison("iter")
    assert result["scores"]["bands"]["plasma_current_relative_error"] is False
    assert result["scores"]["plasma_current_relative_error"] == 0.25
    assert result["constraint_context"]["reference_extra_priors"]
    assert result["bands_applied"] is False


@pytest.mark.parametrize("name", ["mastu", "tcv"])
def test_same_constraint_comparisons_require_the_bands(completed_comparison, name):
    with pytest.raises(AssertionError):
        completed_comparison(name)


@pytest.mark.parametrize("name", ["mastu", "tcv", "iter"])
def test_scaled_control_requires_a_stationary_fit(completed_comparison, name):
    with pytest.raises(AssertionError, match="scaled.*stationary") as refusal:
        completed_comparison(name, bands_pass=True, scaled_converged=False)
    print(f"NONSTATIONARY_SCALED_FIT_REFUSED {refusal.value}")
    result = json.loads(
        (completed_comparison.output / f"{name}-result.json").read_text()
    )
    assert not result["scaled_control_fired"]


def test_scaled_control_requires_a_qualified_forward(completed_comparison):
    with pytest.raises(AssertionError, match="scaled.*qualified"):
        completed_comparison("iter", bands_pass=True, scaled_qualified=False)


@pytest.mark.parametrize("name", ["mastu", "tcv", "iter"])
def test_stationary_qualified_scaled_fit_certifies_displacement(
    completed_comparison, name
):
    result = completed_comparison(name, bands_pass=True)
    assert result["scaled_control_fired"]
    assert result["scaled_fit_converged"] and result["scaled_forward_qualified"]


def test_zero_extent_flux_row_retains_residual_and_fractional_response(linear_fit):
    fitter, measurement, _, _ = linear_fit
    fitter.magnetics = replace(
        fitter.magnetics, flux_loop_fraction=np.array([0.0, 1.0, 0.5, 1.0])
    )
    design = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, -1.0],
            [1.0, 0.5, 0.5],
            [1.0, -2.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    scale = np.r_[measurement.scale, 1.0]
    target = np.r_[measurement.measured, measurement.plasma_current]
    expected = np.linalg.lstsq(design / scale[:, None], target / scale, rcond=None)[0]
    result = fitter.fit(measurement, plasma_current_scale=1.0)
    np.testing.assert_allclose(result.coefficients, expected, atol=1e-7)
    assert result.whitened_residuals[0] == pytest.approx(
        -measurement.measured[0] / measurement.scale[0]
    )
    np.testing.assert_allclose(
        fitter.magnetics.project(np.ones(4), np.ones(4), np.zeros(4)),
        [0.0, 1.0, 0.5, 1.0],
    )


def test_loop_reader_preserves_polarity_and_integrates_toroidal_legs():
    from tests.test_magnetics_fit_efit_comparison import _flux_loop_segments
    from nova.equilibrium.convention import SIGMA_BP

    point = SimpleNamespace(r=1.0, z=0.0, phi=0.0)
    np.testing.assert_array_equal(
        _flux_loop_segments(SimpleNamespace(position=[point])), [[1.0, 0.0, SIGMA_BP]]
    )
    assert _flux_loop_segments(SimpleNamespace(position=[point, point])).shape == (0, 3)
    path = [
        point,
        SimpleNamespace(r=1.0, z=0.4, phi=0.0),
        SimpleNamespace(r=1.0, z=0.4, phi=np.pi),
        SimpleNamespace(r=1.0, z=0.0, phi=np.pi),
        point,
    ]
    terms = _flux_loop_segments(SimpleNamespace(position=path))
    np.testing.assert_allclose(
        terms, [[1.0, 0.4, SIGMA_BP * 0.5], [1.0, 0.0, -SIGMA_BP * 0.5]]
    )
    other = SimpleNamespace(r=1.0, z=0.1, phi=0.2)
    with pytest.raises(ValueError, match="distributed segment observations"):
        _flux_loop_segments(SimpleNamespace(position=[point, other]))


def test_segmented_loop_fitter_uses_every_signed_observation_term(linear_fit):
    fitter, measurement, _, _ = linear_fit
    fitter.magnetics = replace(
        fitter.magnetics,
        flux_loop_segments=(
            np.array([[1.0, 0.0, 0.25], [3.0, 0.0, -0.5]]),
            np.empty((0, 3)),
            np.array([[3.0, 0.0, 1.0]]),
            np.empty((0, 3)),
        ),
    )
    design = np.array(
        [
            [0.25, 0.0, 0.5],
            [0.0, 1.0, -1.0],
            [2.0, 1.0, 1.0],
            [1.0, -2.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    design[0] -= 0.5 * design[2]
    scale = np.r_[measurement.scale, 1.0]
    target = np.r_[measurement.measured, measurement.plasma_current]
    expected = np.linalg.lstsq(design / scale[:, None], target / scale, rcond=None)[0]
    result = fitter.fit(measurement, plasma_current_scale=1.0)
    np.testing.assert_allclose(result.coefficients, expected, atol=1e-7)
    with pytest.raises(ValueError, match="loop_quadrature"):
        fitter.magnetics.project(np.ones(4), np.ones(4), np.zeros(4))


def test_passive_current_uses_primitive_sigma_and_the_ids_clock():
    from tests.test_magnetics_fit_efit_comparison import _sample_passive

    loop = SimpleNamespace(
        current=np.array([10.0, 20.0]),
        current_error_upper=np.array([0.2, 0.4]),
        time=np.array([]),
    )
    ids = SimpleNamespace(
        ids_properties=SimpleNamespace(homogeneous_time=1), time=np.array([0.0, 1.0])
    )
    assert _sample_passive(loop, ids, 0.5) == 15.0
    assert _sample_passive(loop, ids, 0.5, "current_error_upper") == pytest.approx(0.3)
    loop.current_error_upper = np.array([])
    assert _sample_passive(loop, ids, 0.5, "current_error_upper") is None


def test_empty_coil_is_omitted_only_with_explicit_deselection_and_no_connection():
    from tests.test_magnetics_fit_efit_comparison import _omit_empty_unconnected_coil

    active = SimpleNamespace(
        supply=[object()],
        coil=[
            SimpleNamespace(current=SimpleNamespace(data=np.array([10.0]))),
            SimpleNamespace(current=SimpleNamespace(data=np.array([]))),
        ],
        circuit=[SimpleNamespace(connections=np.array([[1, -1, 0], [-1, 1, 0]]))],
    )
    assert not _omit_empty_unconnected_coil(active, 0, True)
    assert not _omit_empty_unconnected_coil(active, 1, False)
    assert _omit_empty_unconnected_coil(active, 1, True)
    active.circuit[0].connections[:, 2] = [1, -1]
    assert not _omit_empty_unconnected_coil(active, 1, True)


def test_circuit_deselection_reads_multiple_xml_fragments(tmp_path):
    from tests.test_magnetics_fit_efit_comparison import _deselect_missing_current

    path = tmp_path / "constraints.xml"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<efitCircuits deselectWhenNoData="true"/>\n'
        '<magneticProbes deselectWhenNoData="true"/>\n'
    )
    assert _deselect_missing_current(path)
    path.write_text('<efitCircuits deselectWhenNoData="false"/><fluxLoops/>')
    assert not _deselect_missing_current(path)


def test_fixed_pressure_and_quadratic_field_spaces_follow_resolved_counts():
    from tests.test_magnetics_fit_efit_comparison import (
        CASES,
        _profile_audit,
        _profile_basis,
    )

    assert CASES["mastu"][-2:] == (2, 2)
    assert CASES["tcv"][-2:] == (0, 2)
    assert CASES["iter"][-2:] == (2, 3)
    assert _profile_basis(0).shape == (1, 0)
    x = np.linspace(0.0, 1.0, 65)
    fixed = _profile_audit(x, np.zeros_like(x), 0)
    nonzero = _profile_audit(x, 1.0 - x, 0)
    curved = _profile_audit(x, 3.0 * (1.0 - x) ** 2 - 0.4 * x * (1.0 - x), 3)
    assert fixed["representable"] and fixed["live_coefficients"] == 0
    assert not nonzero["representable"]
    assert curved["representable"] and curved["live_coefficients"] == 2
    assert curved["lowest_polynomial_degree"] == 2


def test_reference_prior_context_does_not_add_nova_priors():
    from tests.test_magnetics_fit_efit_comparison import _constraint_context

    context = _constraint_context("iter")
    assert (
        context["comparison_label"] == "same magnetics rows; different constraint sets"
    )
    assert context["reference_extra_priors"] == [
        "synthesised pressure prior",
        "q0 prior",
    ]
    assert context["nova_extra_priors"] == []
    assert "predate" in context["counts_caveat"]
    assert "neither" in context["caption"]
    assert _constraint_context("tcv")["reference_extra_priors"] == []


def test_stored_density_integral_keeps_sign_and_generator_cell_area():
    from tests.test_magnetics_fit_efit_comparison import _density_quadrature

    radius, height = np.array([1.0, 1.2, 1.4]), np.array([-0.3, 0.0, 0.3])
    density = np.array([[0.0, 2.0, 0.0], [-3.0, 4.0, 0.0], [0.0, 1.0, 0.0]])
    r, z, current, dr, dz = _density_quadrature(radius, height, density)
    np.testing.assert_allclose(current.sum(), density.sum() * 0.2 * 0.3)
    np.testing.assert_array_equal(current < 0, density[density != 0] < 0)
    assert len(r) == len(z) == len(current) == 4
    assert dr == pytest.approx(0.2) and dz == pytest.approx(0.3)
    with pytest.raises(ValueError, match="nonzero"):
        _density_quadrature(radius, height, np.zeros_like(density))


def test_row_agreement_exposes_a_reversed_current_sign_and_family_gain():
    from tests.test_magnetics_fit_efit_comparison import _row_agreement

    reference = np.array([1.0, -2.0, 4.0])
    scale = np.full(3, 0.1)
    exact = _row_agreement(reference, reference, scale)
    reversed_rows = _row_agreement(-reference, reference, scale)
    doubled = _row_agreement(2 * reference, reference, scale)
    assert exact["whitened_rms"] == exact["relative_l2"] == 0
    assert reversed_rows["relative_l2"] == 2
    assert reversed_rows["signed_gain"] == -1
    assert reversed_rows["opposite_sign_rows"] == 3
    assert reversed_rows["whitened_rms"] > 50
    assert doubled["relative_l2"] == 1 and doubled["signed_gain"] == 2


def test_row_replay_control_keeps_zero_extent_rows_and_refuses_missing_scaling():
    from tests.test_magnetics_fit_efit_comparison import (
        _assert_row_replay_control,
        _row_replay_control,
    )

    reference = np.array([1.0, 0.0, -2.0, 3.0])
    measurement = SimpleNamespace(mask=np.ones(4, dtype=bool), scale=np.ones(4))
    sensors = SimpleNamespace(flux_loop=np.array([True, True, False, False]))
    control = _row_replay_control(reference, reference, measurement, sensors)
    _assert_row_replay_control(control)
    assert control["flux loops"]["unscaled"]["rows"] == 2
    for scores in control.values():
        assert scores["unscaled"]["relative_l2"] == 0
        assert scores["scaled_reference"]["relative_l2"] == pytest.approx(1 / 11)
    control["flux loops"]["scaled_reference"] = control["flux loops"]["unscaled"]
    with pytest.raises(AssertionError, match="scaled-reference replay did not leave"):
        _assert_row_replay_control(control)
    control = _row_replay_control(1.10 * reference, reference, measurement, sensors)
    with pytest.raises(AssertionError, match="unscaled replay outside") as refusal:
        _assert_row_replay_control(control)
    print(f"ROW_REPLAY_BAND_REFUSED {refusal.value}")


def test_known_conductor_subtraction_changes_the_target(linear_fit):
    fitter, measurement, truth, _ = linear_fit
    vacuum = np.array([0.7, -0.4, 1.1, 0.2])
    total = replace(
        measurement,
        measured=measurement.measured + vacuum,
        vacuum=vacuum,
    )
    np.testing.assert_allclose(total.signature, measurement.measured, atol=2e-15)
    fitted = fitter.fit(total, plasma_current_scale=1.0)
    assert fitted.converged
    np.testing.assert_allclose(fitted.coefficients, truth, atol=1e-11)
    assert np.max(np.abs(fitted.whitened_residuals)) <= 1e-8


def test_masked_missing_rows_cannot_poison_the_fit(linear_fit):
    fitter, measurement, truth, _ = linear_fit
    missing = measurement.measured.copy()
    missing[1] = np.nan
    measurement = replace(
        measurement, measured=missing, mask=np.array([1, 0, 1, 1], bool)
    )
    receipt = fitter.fit(measurement, plasma_current_scale=1.0)
    assert receipt.converged
    assert receipt.whitened_residuals[1] == 0.0
    np.testing.assert_allclose(receipt.coefficients, truth, atol=1.0e-11)


def test_unqualified_forward_state_is_refused_before_sensor_read(linear_fit):
    fitter, measurement, _, _ = linear_fit
    solve = fitter.profile.solve
    fitter.profile.solve = lambda request: replace(solve(request), qualified=False)
    with pytest.raises(RuntimeError, match="did not qualify") as caught:
        fitter.fit(measurement, plasma_current_scale=1.0)
    assert isinstance(caught.value.receipt, ForwardSolveReceipt)
    assert not caught.value.receipt.qualified


def test_declared_basis_recovers_coefficients_and_a_resolvable_source(linear_fit):
    fitter, measurement, truth, _ = linear_fit
    basis = np.column_stack(([0.8, 0.1, 1.2], truth))
    fitted = MagneticsFit(
        fitter.profile,
        fitter.request,
        fitter.magnetics,
        fitter.magnetic_source,
        coefficient_basis=basis,
    ).fit(measurement, plasma_current_scale=1.0)
    assert fitted.converged
    np.testing.assert_allclose(fitted.coefficients, [0.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(
        np.r_[
            fitted.source_profile.core.p_prime.coefficients,
            fitted.source_profile.core.ff_prime.coefficients,
        ],
        truth,
        atol=1e-10,
    )
    assert fitted.singular_values.shape == (2,)


def test_dependent_basis_is_refused(linear_fit):
    fitter, _, _, _ = linear_fit
    with pytest.raises(ValueError, match="independent columns"):
        MagneticsFit(
            fitter.profile,
            fitter.request,
            fitter.magnetics,
            fitter.magnetic_source,
            coefficient_basis=np.ones((3, 2)),
        )


def test_conductor_currents_use_measurement_uncertainties_and_typed_requests(
    linear_fit,
):
    fitter, measurement, truth, calls = linear_fit
    conductor_response = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0], [-1.0, 1.0]])
    current = np.array([0.4, -0.2])
    total = replace(
        measurement,
        measured=measurement.measured + conductor_response @ current + [0.1, 0, 0, 0],
        conductor_current=current + [0.05, -0.1],
        conductor_scale=np.array([0.01, 0.3]),
    )
    active = replace(fitter.request, current=jnp.zeros(2))
    receipt = MagneticsFit(
        fitter.profile,
        active,
        fitter.magnetics,
        fitter.magnetic_source,
        conductor_response=conductor_response,
        conductor_names=("upper_coil", "lower_coil"),
    ).fit(total, plasma_current_scale=1.0)
    response = np.array(
        [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [2.0, 1.0, 1.0], [1.0, -2.0, 1.0]]
    )
    design = np.block(
        [
            [response, conductor_response],
            [np.ones((1, 3)), np.zeros((1, 2))],
            [np.zeros((2, 3)), np.eye(2)],
        ]
    )
    weights = np.r_[total.weight, 1.0, 1.0 / total.conductor_scale]
    target = np.r_[total.measured, total.plasma_current, total.conductor_current]
    expected = np.linalg.lstsq(design * weights[:, None], target * weights, rcond=None)[
        0
    ]
    assert receipt.converged
    assert np.max(np.abs(receipt.whitened_residuals)) > 0.01
    np.testing.assert_allclose(receipt.coefficients, expected, atol=1e-10)
    np.testing.assert_allclose(receipt.conductor_currents, expected[-2:], atol=1e-10)
    np.testing.assert_allclose(
        receipt.whitened_residuals, (design @ expected - target) * weights, atol=1e-10
    )
    assert receipt.coefficient_names[-2:] == ("upper_coil", "lower_coil")
    assert receipt.covariance.shape == (5, 5)
    assert receipt.forward_evaluations == len(calls)
    assert receipt.wall_seconds > 0
    assert receipt.device
    assert any(np.linalg.norm(call.current) > 0 for call in calls)
    np.testing.assert_array_equal(active.current, [0.0, 0.0])


def test_fitted_conductor_currents_refuse_missing_uncertainties(linear_fit):
    fitter, measurement, _, _ = linear_fit
    fitter = MagneticsFit(
        fitter.profile,
        replace(fitter.request, current=jnp.zeros(1)),
        fitter.magnetics,
        fitter.magnetic_source,
        conductor_response=np.ones((4, 1)),
    )
    with pytest.raises(ValueError, match="positive uncertainties"):
        fitter.fit(
            replace(measurement, conductor_current=np.zeros(1)),
            plasma_current_scale=1.0,
        )
