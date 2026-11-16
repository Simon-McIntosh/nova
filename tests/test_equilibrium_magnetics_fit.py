"""Profile-coefficient fitting and its public-forward authority boundary."""

from __future__ import annotations

import json
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


def test_passive_zero_turns_remain_zero_response_weights():
    from tests.test_magnetics_fit_efit_comparison import _passive_signed_turns

    assert _passive_signed_turns(0.0) == 0.0
    assert _passive_signed_turns(-2.0) == -2.0
    with pytest.raises(ValueError, match="invalid passive"):
        _passive_signed_turns(float("nan"))


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


def _physical_problem(name):
    """Load fixture-owned machine operands and declare a polynomial source."""
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.forward_operator import ForwardFluxOperator
    from nova.equilibrium.solve_request import ColdSeedPortfolio, ExplicitSolveSeed

    if name == "solovev":
        from benchmarks import solovev_certificate as lane
        from benchmarks.profile_coefficients_recompile_audit import _scaled_source

        kind = "moderate-rotation-conventional-static"
        profile, _, request, _ = lane._certificate_compile_problem(kind, -300)
        carrier, _, exact = lane._case(kind)
        machine = lane._case_machine(kind, carrier, exact, -300)
        source = _scaled_source(profile.source, 1.0, 1.0)
        profile = ForwardProfile(
            replace(profile.operator, source=source), profile.lattice
        )
        request = replace(
            request, source_profile=source, target_current=None, vertical_pin="omit"
        )
        coordinates = np.vstack(
            (machine.node, machine.wall_node, machine.sample_coordinates)
        )
        physical, _, _ = lane.oracle_fixture.cached_fixture_exterior(
            carrier,
            exact,
            machine,
            profile.operator,
            lane._exact_state(kind, exact, coordinates),
        )
        known = dict(
            analytic=exact, reference_current=np.asarray(physical.cell_current)
        )
        return (
            profile,
            request,
            machine.node,
            machine.wall_node,
            machine.cell_polygons,
            known,
        )

    if name == "mast":
        from benchmarks import coil_edit_latency as cached
        from benchmarks import efit_forward_parity_slice as lane
        from benchmarks.mast_response_carrier_warm import DEFAULT_CARRIER
        from nova.imas.mast_solve_inputs import SHOT_STORE

        response, _ = cached._response_cache(DEFAULT_CARRIER)
        case, context = lane._mast_case_from_selection(
            SHOT_STORE,
            dict(shot=22086, slice_index=43),
            None,
        )
        case, declared, _ = lane._passive_inclusive_case(case, context, response)
        old = declared.operator
        operator = ForwardFluxOperator(
            grid=old.grid,
            wall=old.wall,
            source=old.source,
            external_current=old.external_current,
            area=old.area,
            polarity=old.polarity,
            inside_material=old.inside_material,
            use_linear_moments=False,
            prescribed_current_field=old.prescribed_current_field,
        )
        profile = ForwardProfile(operator, declared.lattice)
        seed = jnp.asarray(case["state"])
        coordinate = np.asarray(profile.lattice.coordinate)
        wall = np.asarray(profile.operator.wall.coordinate)
        dr = float(profile.lattice.radial_step)
        dz = float(profile.lattice.vertical_step)
        corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) * [
            dr,
            dz,
        ]
        sections = tuple(point + corners for point in coordinate)
        prescribed = old.prescribed_current_field.current
        group = context["group"]
        circuit = np.asarray(group["fcoil_circ"], dtype=int) - 1
        element = {
            key: np.asarray(group["fcoil_" + key])
            for key in ("r", "z", "width", "height", "ang1", "ang2", "turns", "xmult")
        }
        known = dict(
            conductors=tuple(
                (
                    (
                        lane.shaped_section_vertices(
                            *(
                                element[key][index]
                                for key in ("r", "z", "width", "height", "ang1", "ang2")
                            )
                        ),
                    ),
                    float(element["turns"][index] * element["xmult"][index]),
                    float(prescribed[circuit[index]]),
                )
                for index in range(len(circuit))
            )
        )
        plasma_current = float(case["reference"]["plasma_current_a"])
    else:
        from tests import test_equilibrium_forward_reference as lane

        case = lane.reference_case()
        if isinstance(case, str):
            raise RuntimeError(case)
        machine = lane.cached_machine(case, -500, passive=True)
        sample_nodes = machine.moment_geometry.sample_node_coordinates
        if len(machine.sample_coordinates) != len(sample_nodes):
            from scipy.spatial import cKDTree

            with pytest.raises(ValueError, match="direct sample target rows"):
                lane.forward_profile(case, machine)
            distance, rows = cKDTree(machine.sample_coordinates).query(sample_nodes)
            roundoff = (
                128.0
                * np.finfo(float).eps
                * max(float(np.max(np.abs(sample_nodes))), 1.0)
            )
            assert np.max(distance) <= roundoff
            assert len(np.unique(rows)) == len(sample_nodes)
            print(
                f"CACHED_SAMPLE_ALIGNMENT guard_refused=True "
                f"cached_rows={len(machine.sample_coordinates)} "
                f"required_rows={len(sample_nodes)} "
                f"max_coordinate_distance={np.max(distance):.17g}",
                flush=True,
            )
            machine = replace(
                machine,
                sample_coordinates=np.asarray(sample_nodes),
                **{
                    field: np.asarray(getattr(machine, field))[rows]
                    for field in (
                        "source_to_sample",
                        "plasma_to_sample",
                        "plasma_to_sample_r",
                        "plasma_to_sample_z",
                    )
                },
            )
        profile = lane.forward_profile(case, machine)
        seed = lane.seed_flux(case, machine)
        coordinate, wall, sections = (
            machine.node,
            machine.wall_node,
            machine.cell_polygons,
        )
        prescribed = None
        known = dict(
            conductors=tuple(
                (item.placement, item.turns, item.current)
                for item in case.drive(passive=True)
            )
        )
        plasma_current = float(case.plasma_current)
    _, stored_topology = profile.operator.read(seed)
    centroid = tuple(np.asarray(stored_topology.axis, dtype=float))
    sign = -np.sign(plasma_current)
    source = ForwardSource(
        core=DomainProfile(
            p_prime=PolynomialFluxFunction(
                jnp.asarray([1.0, -1.0]), sign * 3.0e5 / centroid[0] ** 2
            ),
            ff_prime=PolynomialFluxFunction(jnp.asarray([1.0, -1.0]), sign * 0.25),
        ),
        boundary_pressure=profile.source.boundary_pressure,
        boundary_field_function=profile.source.boundary_field_function,
    )
    profile = ForwardProfile(replace(profile.operator, source=source), profile.lattice)
    estimated_current = float(np.sum(profile.operator.cell_current(seed)))
    amplitude = plasma_current / estimated_current
    assert np.isfinite(amplitude) and amplitude > 0.0
    source = replace(
        source,
        core=replace(
            source.core,
            p_prime=replace(
                source.core.p_prime,
                normalisation=source.core.p_prime.normalisation * amplitude,
            ),
            ff_prime=replace(
                source.core.ff_prime,
                normalisation=source.core.ff_prime.normalisation * amplitude,
            ),
        ),
    )
    profile = ForwardProfile(replace(profile.operator, source=source), profile.lattice)
    request = ForwardSolveRequest.from_defaults(
        carrier_identity=f"{name}-cached-polynomial-source",
        source_profile=source,
        seed_policy=(
            ColdSeedPortfolio(plasma_current, centroid, requested_class="limited")
            if name == "mast"
            else ExplicitSolveSeed(seed)
        ),
        prescribed_current=prescribed,
        vertical_pin="omit",
        policy_overrides=dict(kernel_tolerance=1e-12, qualification_tolerance=1e-12),
    )
    return profile, request, coordinate, wall, sections, known


def _persist_forward(receipt, path):
    """Persist the complete forward result before deriving any score."""
    import copyreg
    import pickle
    from types import MappingProxyType
    from nova.equilibrium.forward_operator import ForwardTopologyState

    copyreg.pickle(type(MappingProxyType({})), lambda value: (dict, (dict(value),)))
    copyreg.pickle(
        ForwardTopologyState,
        lambda value: (
            ForwardTopologyState.tree_unflatten,
            (None, value.tree_flatten()[0]),
        ),
    )
    pending = path.with_suffix(".pending")
    with pending.open("wb") as stream:
        pickle.dump(receipt, stream)
    pending.replace(path)


def _persist_forward_failure(error, output, name):
    """Retain the public call's terminal history when receipt qualification raises."""
    import json
    import pickle
    import traceback
    from nova.equilibrium.fixed_point import FixedPointResult

    frames = list(traceback.walk_tb(error.__traceback__))
    histories = [
        frame.f_locals["history"]
        for frame, _ in frames
        if isinstance(frame.f_locals.get("history"), FixedPointResult)
    ]
    summary = dict(case=name, exception=type(error).__name__, message=str(error))
    if histories:
        history = histories[-1]
        destination = output / f"{name}-rejected-history.pkl"
        with destination.open("wb") as stream:
            pickle.dump(history, stream)
        summary.update(
            residual=float(history.residual),
            converged=bool(history.converged),
            termination_reason=int(history.termination_reason),
            trips=int(
                np.count_nonzero(np.isfinite(np.asarray(history.active_set_residuals)))
            ),
            history_path=str(destination),
        )
    admitted = [
        frame.f_locals["admitted"]
        for frame, _ in frames
        if frame.f_code.co_name == "require_qualified_axis"
    ]
    if admitted:
        summary["axis_admitted"] = bool(admitted[-1])
    (output / f"{name}-generating-failure.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("GENERATING_FAILURE " + json.dumps(summary), flush=True)


class _PlasmaDiagnosticSource:
    """Bind terminal cell currents to fixed authored diagnostic sections."""

    def __init__(self, sections):
        from nova.frame.coilset import CoilSet

        self.biot = CoilSet(dcoil=-1, field_attrs=["Psi", "Br", "Bz"])
        for index, section in enumerate(sections):
            self.biot.coil.insert(
                np.asarray(section),
                nturn=1.0,
                Ic=0.0,
                name=f"cell_{index}",
            )
        for name in ("point", "probe", "poloidal_flux_loop"):
            factory = getattr(self.biot, name)
            self._cache_factory(factory)

    @staticmethod
    def _cache_factory(factory):
        """Retain factory-owned Green tables while source currents vary."""
        original = factory.solve
        tables = {}

        def solve(*args):
            if args:
                coordinates = np.asarray(args[0], dtype=np.float64)
            else:
                coordinates = np.column_stack(
                    (factory.target["x"], factory.target["z"])
                )
            key = coordinates.shape, coordinates.tobytes()
            if key not in tables:
                original(*args)
                tables[key] = factory.data.copy(deep=True)
            else:
                factory.data = tables[key]
                factory.load_operators()

        factory.solve = solve

    def __call__(self, receipt):
        self.biot.sloc["Ic"] = np.asarray(receipt.terminal_state.cell_current)
        return self.biot


def _physical_sensor_rows(biot, sensors):
    """Synthesize row-ordered observations through the diagnostic kernel owners."""
    from nova.equilibrium.observation_kernels import (
        virtual_flux_loops,
        virtual_poloidal_probes,
    )

    points = np.column_stack((sensors.r, sensors.z))
    loops = np.asarray(sensors.flux_loop, dtype=bool)
    rows = np.empty(sensors.number)
    loop_points, loop_rows, fractions = sensors.loop_quadrature
    rows[loops] = 0.0
    if len(loop_points):
        np.add.at(
            rows,
            loop_rows,
            virtual_flux_loops(biot, loop_points).values * fractions,
        )
    angle = np.deg2rad(sensors.angle[~loops])
    direction = np.column_stack((np.cos(angle), np.sin(angle)))
    rows[~loops] = virtual_poloidal_probes(biot, points[~loops], direction).values
    return rows


def _known_field_reader(known, source):
    """Bind prescribed currents or the analytic fixture exterior independently."""
    if "conductors" in known:
        from nova.frame.coilset import CoilSet

        biot = CoilSet(dcoil=-1, field_attrs=["Psi", "Br", "Bz"])
        for index, (placement, turns, current) in enumerate(known["conductors"]):
            biot.coil.insert(*placement, nturn=turns, Ic=current, name=f"known_{index}")
        for name in ("point", "probe", "poloidal_flux_loop"):
            _PlasmaDiagnosticSource._cache_factory(getattr(biot, name))
        assert np.any(np.asarray(biot.sloc["Ic"]) != 0.0)

        def read(sensors):
            rows = _physical_sensor_rows(biot, sensors)
            current = np.asarray(biot.sloc["Ic"]).copy()
            biot.sloc["Ic"] = np.zeros_like(current)
            try:
                np.testing.assert_array_equal(_physical_sensor_rows(biot, sensors), 0.0)
            finally:
                biot.sloc["Ic"] = current
            return rows
    else:
        from nova.equilibrium.convention import TOTAL_FLUX_FACTOR

        def read(sensors):
            exact = known["analytic"]
            radial, vertical = exact.flux_gradient(sensors.r, sensors.z)
            angle = np.deg2rad(sensors.angle)
            analytic = np.where(
                sensors.flux_loop,
                TOTAL_FLUX_FACTOR * exact.flux(sensors.r, sensors.z),
                (-vertical * np.cos(angle) + radial * np.sin(angle)) / sensors.r,
            )
            source.biot.sloc["Ic"] = known["reference_current"]
            return analytic - _physical_sensor_rows(source.biot, sensors)

    return read


def _total_observations(source, generated, sensors, known_field, scales=None):
    """Measure total and known fields without calling the inverse evaluator."""
    plasma = _physical_sensor_rows(source(generated), sensors)
    vacuum = known_field(sensors)
    total = plasma + vacuum
    if scales is None:
        loops = sensors.flux_loop
        scales = np.where(
            loops,
            max(np.max(np.abs(plasma[loops])), 1e-12),
            max(np.max(np.abs(plasma[~loops])), 1e-12),
        )
    assert np.max(np.abs(vacuum[sensors.flux_loop] / scales[sensors.flux_loop])) > 1e-8
    assert (
        np.max(np.abs(vacuum[~sensors.flux_loop] / scales[~sensors.flux_loop])) > 1e-8
    )
    np.testing.assert_allclose(
        (total - vacuum) / scales, plasma / scales, rtol=0, atol=1e-12
    )
    return total, vacuum, plasma, scales


@pytest.fixture(scope="module")
def physical_cases(tmp_path_factory):
    """Keep each carrier, compiled program and diagnostic table in one process."""
    import os
    from pathlib import Path

    output = Path(
        os.environ.get(
            "NOVA_MAGNETICS_FIT_EVIDENCE", tmp_path_factory.mktemp("magnetics")
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    cases = {}

    def get(name):
        import json

        if name in cases:
            return cases[name]
        profile, request, coordinate, wall, sections, known = _physical_problem(name)
        recorded = os.environ.get("NOVA_MAGNETICS_FIT_GENERATING_RECEIPTS")
        recorded_path = (
            Path(recorded) / f"{name}-generating-receipt.pkl" if recorded else None
        )
        if recorded and (Path(recorded) / f"{name}-unqualified.json").exists():
            raise RuntimeError(
                f"{name}: all three recorded prescribed profiles failed qualification"
            )
        if recorded_path is not None and recorded_path.exists():
            import pickle

            with recorded_path.open("rb") as stream:
                generated = pickle.load(stream)
            request_path = Path(recorded) / f"{name}-generating-request.pkl"
            if request_path.exists():
                with request_path.open("rb") as stream:
                    request = pickle.load(stream)
            print(f"GENERATING_RECEIPT {recorded_path}", flush=True)
        else:
            import json

            attempts = (1.0,) if name == "solovev" else (1.0, 0.7, 1.3)
            original = request
            qualified = False
            for attempt, factor in enumerate(attempts, 1):
                label = f"{name}-profile-{attempt}"
                source = original.source_profile
                source = replace(
                    source,
                    core=replace(
                        source.core,
                        p_prime=replace(
                            source.core.p_prime,
                            coefficients=source.core.p_prime.coefficients * factor,
                        ),
                        ff_prime=replace(
                            source.core.ff_prime,
                            coefficients=source.core.ff_prime.coefficients * factor,
                        ),
                    ),
                )
                request = replace(original, source_profile=source)
                inputs = dict(
                    pressure_coefficients=np.asarray(
                        source.core.p_prime.coefficients
                    ).tolist(),
                    pressure_normalisation=float(source.core.p_prime.normalisation),
                    diamagnetic_coefficients=np.asarray(
                        source.core.ff_prime.coefficients
                    ).tolist(),
                    diamagnetic_normalisation=float(source.core.ff_prime.normalisation),
                    seed=request.seed_policy.provenance().to_dict(),
                )
                (output / f"{label}-inputs.json").write_text(
                    json.dumps(inputs, indent=2)
                )
                try:
                    generated = profile.solve(request)
                except Exception as error:
                    _persist_forward_failure(error, output, label)
                    continue
                _persist_forward(generated, output / f"{label}-receipt.pkl")
                summary = dict(
                    case=name,
                    attempt=attempt,
                    qualified=bool(generated.qualified),
                    converged=bool(generated.terminal_state.fixed_point.converged),
                    residual=float(generated.terminal_state.fixed_point.residual),
                    termination_reason=int(generated.termination_reason),
                    trips=int(
                        np.count_nonzero(
                            np.isfinite(np.asarray(generated.residual_history))
                        )
                    ),
                )
                (output / f"{label}-receipt.json").write_text(
                    json.dumps(summary, indent=2)
                )
                print("PRESCRIBED_GENERATING " + json.dumps(summary), flush=True)
                if summary["qualified"] and summary["converged"]:
                    qualified = True
                    break
            if not qualified:
                (output / f"{name}-unqualified.json").write_text(
                    json.dumps(dict(case=name, attempts=len(attempts), qualified=False))
                )
                raise RuntimeError(
                    f"{name}: no qualified prescribed profile "
                    f"in {len(attempts)} attempts"
                )
        _persist_forward(generated, output / f"{name}-generating.pkl")
        _persist_forward(generated, output / f"{name}-generating-receipt.pkl")
        if name != "solovev":
            _persist_forward(request, output / f"{name}-generating-request.pkl")
        residual = float(generated.terminal_state.fixed_point.residual)
        print(
            f"GENERATING {name} qualified={bool(generated.qualified)} "
            f"residual={residual:.17g}",
            flush=True,
        )
        assert bool(generated.qualified), (
            name,
            residual,
            request.policy.qualification_tolerance,
        )
        assert bool(generated.terminal_state.fixed_point.converged), (name, residual)
        if os.environ.get("NOVA_MAGNETICS_FIT_GENERATE_ONLY") == "1":
            return dict(profile=profile, request=request, generated=generated)
        index = np.linspace(0, len(wall) - 1, 24, dtype=int)
        point = np.asarray(wall)[index]
        sensors = Magnetics(
            r=np.tile(point[:, 0], 2),
            z=np.tile(point[:, 1], 2),
            angle=np.r_[np.zeros(24), np.linspace(0.0, 360.0, 24, endpoint=False)],
            flux_loop=np.r_[np.ones(24, bool), np.zeros(24, bool)],
        )
        source = _PlasmaDiagnosticSource(sections)
        known_field = _known_field_reader(known, source)
        total, vacuum, plasma, scales = _total_observations(
            source, generated, sensors, known_field
        )
        shifted_radius = sensors.r.copy()
        shifted_radius[0] += 0.05
        shifted = replace(sensors, r=shifted_radius)
        shifted_total, shifted_vacuum, shifted_plasma, _ = _total_observations(
            source, generated, shifted, known_field, scales
        )
        geometry_delta = float(np.max(np.abs((shifted_vacuum - vacuum) / scales)))
        assert geometry_delta > 1e-12, geometry_delta
        subtraction = dict(
            case=name,
            conductor_count=len(known.get("conductors", ())),
            exterior_kind="stored_conductors"
            if "conductors" in known
            else "analytic_fixture",
            nominal_vacuum=vacuum.tolist(),
            displaced_vacuum=shifted_vacuum.tolist(),
            nominal_total=total.tolist(),
            displaced_total=shifted_total.tolist(),
            nominal_subtraction_error=float(
                np.max(np.abs((total - vacuum - plasma) / scales))
            ),
            displaced_subtraction_error=float(
                np.max(
                    np.abs((shifted_total - shifted_vacuum - shifted_plasma) / scales)
                )
            ),
            conductor_geometry_delta=geometry_delta,
        )
        (output / f"{name}-conductor-subtraction.json").write_text(
            json.dumps(subtraction, indent=2)
        )
        print(
            "CONDUCTOR_SUBTRACTION "
            + json.dumps(
                {
                    key: value
                    for key, value in subtraction.items()
                    if not key.endswith(("vacuum", "total"))
                }
            ),
            flush=True,
        )
        raw = np.r_[
            request.source_profile.core.p_prime.coefficients,
            request.source_profile.core.ff_prime.coefficients,
        ]
        measurement = SliceMeasurement(
            measured=total,
            vacuum=vacuum,
            mask=np.ones(48, bool),
            scale=scales,
            plasma_current=float(np.sum(generated.terminal_state.cell_current)),
        )
        pressure = request.source_profile.core.p_prime.coefficients.size
        basis = np.zeros((raw.size, 2))
        basis[:pressure, 0] = raw[:pressure]
        basis[pressure:, 1] = raw[pressure:]
        perturbation = 0.02 if name == "solovev" else 2e-4
        initial_source = replace(
            request.source_profile,
            core=replace(
                request.source_profile.core,
                p_prime=replace(
                    request.source_profile.core.p_prime,
                    coefficients=jnp.asarray(raw[:pressure] * (1.0 - perturbation)),
                ),
                ff_prime=replace(
                    request.source_profile.core.ff_prime,
                    coefficients=jnp.asarray(raw[pressure:] * (1.0 + perturbation)),
                ),
            ),
        )
        initial = replace(request, source_profile=initial_source)
        if name != "solovev":
            from nova.equilibrium.solve_request import ExplicitSolveSeed

            initial = replace(
                initial, seed_policy=ExplicitSolveSeed(generated.terminal_state.flux)
            )
        cases[name] = dict(
            name=name,
            profile=profile,
            request=request,
            initial=initial,
            generated=generated,
            coordinate=np.asarray(coordinate),
            wall=np.asarray(wall),
            source=source,
            known_field=known_field,
            sensors=sensors,
            measurement=measurement,
            basis=basis,
            output=output,
        )
        return cases[name]

    return get


def _recover_physical(case, displacement=0.0):
    import json

    sensors = case["sensors"]
    label = f"{case['name']}-" + ("displaced-fit" if displacement else "fit")
    initial_record = {}

    def magnetic_source(receipt):
        if not initial_record:
            _persist_forward(receipt, case["output"] / f"{label}-initial.pkl")
            expected = case["generated"].terminal_state
            count = len(case["coordinate"])
            core = np.asarray(expected.domains.core)
            error = float(
                np.max(
                    np.abs(
                        np.asarray(
                            receipt.terminal_state.flux[:count] - expected.flux[:count]
                        )
                    )[core]
                )
                / abs(float(expected.topology.flux_span))
            )
            initial_record["flux_sup_fraction"] = error
            (case["output"] / f"{label}-initial.json").write_text(
                json.dumps(initial_record, indent=2)
            )
            print("INITIAL_RECOVERY " + json.dumps(initial_record), flush=True)
            assert error > 1e-6, initial_record
        return case["source"](receipt)

    if displacement:
        radius = sensors.r.copy()
        radius[0] += displacement
        sensors = replace(sensors, r=radius)
    total, vacuum, plasma, scales = _total_observations(
        case["source"],
        case["generated"],
        sensors,
        case["known_field"],
        case["measurement"].scale,
    )
    measurement = replace(case["measurement"], vacuum=vacuum)
    observation = dict(
        displacement_m=displacement,
        conductor_flux_sup=float(np.max(np.abs(vacuum[sensors.flux_loop]))),
        conductor_probe_sup=float(np.max(np.abs(vacuum[~sensors.flux_loop]))),
        subtraction_error=float(np.max(np.abs((total - vacuum - plasma) / scales))),
        geometry=sensors.r.tolist(),
        total=total.tolist(),
        vacuum=vacuum.tolist(),
        plasma=plasma.tolist(),
    )
    if displacement:
        delta = float(np.max(np.abs((vacuum - case["measurement"].vacuum) / scales)))
        assert delta > 1e-12, delta
        observation["conductor_geometry_delta"] = delta
    (case["output"] / f"{label}-observations.json").write_text(
        json.dumps(observation, indent=2)
    )
    print(
        "TOTAL_OBSERVATIONS "
        + json.dumps(
            {
                k: v
                for k, v in observation.items()
                if k not in ("geometry", "total", "vacuum", "plasma")
            }
        ),
        flush=True,
    )
    fitted = MagneticsFit(
        case["profile"],
        case["initial"],
        sensors,
        magnetic_source,
        coefficient_basis=case["basis"],
    ).fit(
        measurement,
        plasma_current_scale=abs(case["measurement"].plasma_current),
    )
    forward_path = case["output"] / f"{label}-forward.pkl"
    _persist_forward(fitted.forward, forward_path)
    core = fitted.source_profile.core
    record = dict(
        coefficients=fitted.coefficients.tolist(),
        coefficient_names=list(fitted.coefficient_names),
        forward_receipt=str(forward_path),
        iterations=fitted.iterations,
        singular_values=fitted.singular_values.tolist(),
        undetermined_coefficients=list(fitted.undetermined_coefficients),
        null_directions=fitted.null_directions.tolist(),
        whitened_residuals=fitted.whitened_residuals.tolist(),
        covariance=fitted.covariance.tolist(),
        converged=fitted.converged,
        forward_evaluations=fitted.forward_evaluations,
        source_profile=dict(
            carrier=case["request"].carrier_identity,
            pressure_coefficients=np.asarray(core.p_prime.coefficients).tolist(),
            pressure_normalisation=float(core.p_prime.normalisation),
            diamagnetic_coefficients=np.asarray(core.ff_prime.coefficients).tolist(),
            diamagnetic_normalisation=float(core.ff_prime.normalisation),
            boundary_pressure=float(fitted.source_profile.boundary_pressure),
            boundary_field_function=float(
                fitted.source_profile.boundary_field_function
            ),
        ),
    )
    (case["output"] / f"{label}.json").write_text(json.dumps(record, indent=2))
    expected = case["generated"].terminal_state
    terminal = fitted.forward.terminal_state
    count = len(case["coordinate"])
    core = np.asarray(expected.domains.core)
    span = abs(float(expected.topology.flux_span))
    error = float(
        np.max(np.abs(np.asarray(terminal.flux[:count] - expected.flux[:count]))[core])
        / span
    )
    return fitted, error


def _draw_recovery_panels(name, case, fitted):
    """Save shared-level contour panels through the existing poloidal painters."""
    import os
    from pathlib import Path
    import xml.etree.ElementTree as element_tree
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from nova.media import poloidal
    from nova.media.ink import DEFAULT_INK, poloidal_axes

    directory = os.environ.get("NOVA_MAGNETICS_FIT_FIGURES")
    if directory is None:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    generated = case["generated"].terminal_state
    recovered = fitted.forward.terminal_state
    levels = np.sort(
        np.linspace(
            float(generated.topology.axis_flux),
            float(generated.topology.boundary_flux),
            14,
        )
    )[1:-1]
    grey, blue = "#929292", DEFAULT_INK.flux_color
    original_style = DEFAULT_INK.variant(
        axis_color=grey, xpoint_color=grey, axis_markersize=9.0, xpoint_markersize=10.0
    )
    recovered_style = DEFAULT_INK.variant(axis_color=blue, xpoint_color=blue)
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 7.2), dpi=100)
    figure.subplots_adjust(left=0.03, right=0.97, bottom=0.06, top=0.84, wspace=0.12)
    coordinate = case["coordinate"]
    for axis, state, color, label in zip(
        axes,
        (generated, recovered),
        (grey, blue),
        ("generating", "recovered"),
        strict=True,
    ):
        poloidal_axes(axis)
        poloidal.draw_scattered_contours(
            axis,
            coordinate[:, 0],
            coordinate[:, 1],
            np.asarray(state.flux)[: len(coordinate)],
            levels,
            case["wall"],
            color=color,
            linewidth=1.2,
        )
        poloidal.draw_wall(axis, units=case["wall"])
        for landmarks, style in (
            (generated.topology, original_style),
            (recovered.topology, recovered_style),
        ):
            diverted = bool(landmarks.diverted)
            poloidal.draw_nulls(
                axis,
                magnetic_axis=np.asarray(landmarks.axis),
                x_points=np.asarray(landmarks.x_point) if diverted else None,
                other_x_points=np.asarray(landmarks.x_point) if not diverted else None,
                contain=case["wall"],
                style=style,
            )
        axis.text(0.5, 1.02, label, ha="center", transform=axis.transAxes, fontsize=16)
    legend = figure.legend(
        [Line2D([], [], color=grey, lw=2.6), Line2D([], [], color=blue, lw=3.0)],
        ["generating", "recovered"],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=14,
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    texts = [
        *figure.texts,
        *legend.get_texts(),
        *(text for axis in axes for text in axis.texts),
    ]
    boxes = [(text.get_text(), text.get_window_extent(renderer)) for text in texts]
    for label, box in boxes:
        if not figure.bbox.contains(box.x0, box.y0) or not figure.bbox.contains(
            box.x1, box.y1
        ):
            raise ValueError(f"figure text outside canvas: {label}")
    for index, (label, box) in enumerate(boxes):
        for other, candidate in boxes[index + 1 :]:
            if box.overlaps(candidate):
                raise ValueError(f"figure text overlap: {label}, {other}")
    destination = directory / f"{name}.svg"
    figure.savefig(destination, format="svg")
    tree = element_tree.parse(destination)
    tree.getroot().set("width", "31.1cm")
    tree.getroot().set("height", "16cm")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    figure.savefig(case["output"] / f"{name}-preview.png", dpi=100)
    plt.close(figure)
    print(f"FIGURE {destination} text_bounds=passed text_overlap=passed", flush=True)


@pytest.mark.slow
@pytest.mark.timeout(3500)
@pytest.mark.parametrize("name", ("solovev", "mast", "iter"))
def test_known_state_recovery(name, physical_cases):
    import json
    import os

    case = physical_cases(name)
    displacement = (
        float(os.environ.get("NOVA_MAGNETICS_FIT_DISPLACEMENT", "0"))
        if name == "solovev"
        else 0.0
    )
    fitted, error = _recover_physical(case, displacement)
    _persist_forward(fitted.forward, case["output"] / f"{name}-recovered.pkl")
    resolved = case["profile"].solve(
        replace(case["request"], source_profile=fitted.source_profile)
    )
    _persist_forward(resolved, case["output"] / f"{name}-resolved.pkl")
    agreement = float(
        np.max(
            np.abs(
                np.asarray(
                    resolved.terminal_state.flux - fitted.forward.terminal_state.flux
                )
            )
        )
        / np.max(np.abs(np.asarray(fitted.forward.terminal_state.flux)))
    )
    summary = dict(
        case=name,
        flux_sup_fraction=error,
        residual_sup=float(np.max(np.abs(fitted.whitened_residuals))),
        largest_residual_row=int(np.argmax(np.abs(fitted.whitened_residuals))),
        resolve_relative_sup=agreement,
        iterations=fitted.iterations,
        singular_values=fitted.singular_values.tolist(),
        undetermined_coefficients=list(fitted.undetermined_coefficients),
        forward_evaluations=fitted.forward_evaluations,
        coefficients=fitted.coefficients.tolist(),
        coefficient_names=list(fitted.coefficient_names),
        terminal_residual=float(fitted.forward.terminal_state.fixed_point.residual),
        terminal_converged=bool(fitted.forward.terminal_state.fixed_point.converged),
        forward_tolerance=case["request"].policy.kernel_tolerance,
    )
    (case["output"] / f"{name}-recovery.json").write_text(json.dumps(summary, indent=2))
    print("RECOVERY " + json.dumps(summary), flush=True)
    _draw_recovery_panels(name, case, fitted)
    assert error <= 1e-6, summary
    assert summary["residual_sup"] <= 1e-8, summary
    assert bool(resolved.qualified), summary
    assert agreement <= case["request"].policy.kernel_tolerance, summary
    assert fitted.singular_values.size == 2
    assert fitted.iterations > 0


@pytest.mark.slow
@pytest.mark.timeout(3500)
def test_displaced_flux_loop_is_identified(physical_cases):
    import json

    case = physical_cases("solovev")
    fitted, error = _recover_physical(case, 0.05)
    summary = dict(
        flux_sup_fraction=error,
        largest_residual_row=int(np.argmax(np.abs(fitted.whitened_residuals))),
        residual_sup=float(np.max(np.abs(fitted.whitened_residuals))),
    )
    (case["output"] / "displaced-loop.json").write_text(json.dumps(summary, indent=2))
    print("DISPLACED_LOOP " + json.dumps(summary), flush=True)
    assert error > 1e-6, summary
    assert summary["largest_residual_row"] == 0, summary
