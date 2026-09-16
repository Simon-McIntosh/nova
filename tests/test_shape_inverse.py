"""Contracts for bounding-box targets solved through prescribed currents."""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from apps.playable.shape import PlasmaShape, move_bounding_box
import nova.equilibrium.shape_inverse as shape_inverse_module
from nova.equilibrium.shape_inverse import (
    GAMMA,
    NoAdmissibleShapeStepError,
    _admissible_delta,
    _cap_current_delta,
    _refine_turning_point,
    _secant_refreshed_tangent,
    _turning_point_current_update,
    achieved_target,
    bounding_box_pairs,
    observed_values,
    response_matrix,
    shape_response_matrix,
    shape_steering_target,
    shape_values,
    solve_shape_inverse,
    turning_point_response_matrix,
)
from nova.equilibrium.topology import NoQualifiedAxisError


@pytest.fixture(scope="module")
def machine():
    """Build the limited prescribed-current fixture once."""
    from apps.playable.solovev import build_machine

    return build_machine()


def _span(profile, flux) -> float:
    """Return the flux span magnitude at one state."""
    _masks, topology = profile.operator.read(jnp.asarray(flux))
    return abs(float(np.asarray(topology.flux_span)))


@pytest.fixture(scope="module")
def seed_target(machine):
    """Return the exact turning-point target read from the analytic seed."""
    return achieved_target(machine.profile, machine.seed)


def test_limited_target_drops_the_null_row(machine, seed_target):
    """A limited plasma carries flux and field rows but no fictive null row."""
    pairs = bounding_box_pairs(
        machine.profile, seed_target, span=_span(machine.profile, machine.seed)
    )
    assert seed_target.x_point is None
    assert tuple(pair.row_count for pair in pairs) == (4, 4)


def test_x_point_adds_both_field_components(machine, seed_target):
    """One X-point coordinate contributes both Br and Bz rows."""
    target = replace(seed_target, x_point=np.asarray([1.0, -0.2]))
    values = shape_values(machine.profile, target, machine.seed)
    response = shape_response_matrix(machine.profile, target, machine.seed)
    assert values.shape == (10,)
    assert response.shape == (10, machine.circuit_count)


def test_shape_steering_target_carries_the_full_boundary_polygon(machine, seed_target):
    """The inverse constrains every measured boundary point, not four extrema."""
    rows, previous = shape_steering_target(machine.profile, seed_target, machine.seed)
    assert rows.flux_points.shape[0] > 100
    np.testing.assert_allclose(rows.flux_points[:4], seed_target.flux_points)
    np.testing.assert_allclose(previous[:4], seed_target.flux_points)


def test_shape_steering_target_keeps_a_moved_x_point_commanded(machine, seed_target):
    """Null rows are evaluated where the operator commands the X-point."""
    commanded_x_point = np.asarray([1.05, -0.15])
    rows, _previous = shape_steering_target(
        machine.profile,
        replace(seed_target, x_point=commanded_x_point),
        machine.seed,
    )
    np.testing.assert_allclose(rows.x_point, commanded_x_point)


def test_shape_steering_target_holds_uncommanded_turning_points(machine, seed_target):
    """An upper-point command adds residual rows at the other prior extrema."""
    commanded = np.asarray(seed_target.flux_points).copy()
    commanded[1, 1] += 0.02
    target = replace(
        seed_target,
        flux_points=commanded,
        radial_field_points=commanded[[0, 2]],
        vertical_field_points=commanded[[1, 3]],
    )

    rows, previous = shape_steering_target(machine.profile, target, machine.seed)

    np.testing.assert_allclose(rows.flux_points[-3:], previous[[0, 2, 3]])
    np.testing.assert_allclose(rows.radial_field_points[-2:], previous[[0, 2]])
    np.testing.assert_allclose(rows.vertical_field_points[-1:], previous[[3]])


def test_axis_admissibility_contracts_the_current_delta():
    """A refused forward state halves the proposed current update until admitted."""

    class Operator:
        @staticmethod
        def read(flux, requested_class=None):
            del requested_class
            if float(np.asarray(flux)[0]) > 5.0:
                raise NoQualifiedAxisError("refused trial")
            return None, None

    profile = SimpleNamespace(operator=Operator())
    delta, fraction, trials = _admissible_delta(
        profile,
        jnp.zeros(1),
        np.asarray([0.0]),
        np.asarray([0]),
        np.asarray([12.0]),
        forward_solve=lambda current: jnp.asarray(current),
    )

    np.testing.assert_allclose(delta, [3.0])
    assert fraction == 0.25
    assert trials == 3


def test_axis_admissibility_records_every_nonzero_refusal():
    """An exhausted nonlinear referee reports each trial and never accepts zero."""

    class Operator:
        @staticmethod
        def read(_flux, requested_class=None):
            del requested_class
            raise NoQualifiedAxisError("refused trial")

    profile = SimpleNamespace(operator=Operator())
    with pytest.raises(NoAdmissibleShapeStepError) as caught:
        _admissible_delta(
            profile,
            jnp.zeros(1),
            np.asarray([0.0]),
            np.asarray([0]),
            np.asarray([12.0]),
            forward_solve=lambda current: jnp.asarray(current),
        )

    np.testing.assert_allclose(caught.value.proposed_delta, [12.0])
    assert caught.value.refusal_sequence[0] == 1.0
    assert caught.value.refusal_sequence[-1] == 2.0**-20
    assert len(caught.value.refusal_sequence) == 21
    assert 0.0 not in caught.value.refusal_sequence


def test_turning_point_refinement_rejects_off_plasma_root(monkeypatch):
    """A Newton root at the observed -12.88 m failure falls back to the ray seed."""
    start = np.asarray([1.0, 1.0])
    boundary = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    profile = SimpleNamespace(
        lattice=SimpleNamespace(
            radial_step=0.1,
            height=jnp.asarray([0.0, 0.1]),
        )
    )

    def off_plasma_residual(_lattice, _grid, _level, point, *, radial):
        del radial
        return point - jnp.asarray([1.0, -12.88])

    monkeypatch.setattr(
        shape_inverse_module, "_turning_point_residual", off_plasma_residual
    )
    refined = _refine_turning_point(
        profile,
        jnp.zeros((2, 2)),
        0.0,
        start,
        radial=False,
        boundary=boundary,
    )

    np.testing.assert_array_equal(refined, start)


def test_achieved_shape_secant_corrects_a_fourfold_response_error():
    """The admitted nonlinear motion replaces an inaccurate local gain."""
    local_tangent = np.eye(2)
    admitted_current_delta = np.asarray([1.0, 0.0])
    achieved_motion = np.asarray([4.0, 0.0])

    refreshed = _secant_refreshed_tangent(
        local_tangent, admitted_current_delta, achieved_motion
    )
    correction = _turning_point_current_update(
        refreshed,
        np.asarray([-2.0, 0.0]),
        regularisation=0.0,
        delta_regularisation=0.0,
        delta_scale=np.ones(2),
    )

    np.testing.assert_allclose(refreshed @ admitted_current_delta, achieved_motion)
    np.testing.assert_allclose(correction, [-0.5, 0.0])


def test_turning_point_tangent_reads_physical_extrema(machine):
    """The current tangent is expressed in the eight turning-point coordinates."""
    tangent = turning_point_response_matrix(
        machine.profile,
        machine.seed,
        (0, machine.circuit_count // 2),
    )

    assert tangent.shape == (8, 2)
    assert np.all(np.isfinite(tangent))
    assert np.linalg.norm(tangent) > 0.0


def test_response_matrix_matches_central_differences(machine, seed_target):
    """Carrier contractions reproduce direct current perturbations of each row."""
    profile = machine.profile
    state = jnp.asarray(machine.seed)
    pairs = bounding_box_pairs(profile, seed_target, span=_span(profile, state))
    analytic = response_matrix(profile, pairs, state)
    response = jnp.asarray(profile.operator.prescribed_current_field.response)
    current_step = 100.0
    for circuit in (0, profile.operator.prescribed_current_field.circuit_count // 2):
        tangent = response[:, circuit] * current_step
        plus = observed_values(profile, pairs, state + tangent)
        minus = observed_values(profile, pairs, state - tangent)
        central = (plus - minus) / (2.0 * current_step)
        np.testing.assert_allclose(
            analytic[:, circuit], central, rtol=2.0e-8, atol=1.0e-12
        )

    direct = shape_response_matrix(profile, seed_target, state)
    for circuit in (0, profile.operator.prescribed_current_field.circuit_count // 2):
        tangent = response[:, circuit] * current_step
        plus = shape_values(profile, seed_target, state + tangent)
        minus = shape_values(profile, seed_target, state - tangent)
        central = (plus - minus) / (2.0 * current_step)
        np.testing.assert_allclose(
            direct[:, circuit], central, rtol=2.0e-8, atol=1.0e-12
        )


def test_unmoved_inverse_solves_seed_anchored_delta(machine, seed_target):
    """The inverse regularises current changes about the immutable seed."""
    current = np.asarray(machine.profile.operator.prescribed_current_field.current)
    solved = solve_shape_inverse(
        machine.profile,
        seed_target,
        machine.seed,
        prescribed_current=current,
    )
    np.testing.assert_allclose(
        solved.delta,
        solved.currents[solved.free_circuits] - current[solved.free_circuits],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert solved.row_kinds == ("flux",) * solved.flux_points.shape[0] + ("field",) * 4
    assert solved.flux_points.shape[0] > 100
    assert np.all(solved.consistency_floor > 0.0)
    assert solved.gamma == pytest.approx(GAMMA * solved.plasma_current)
    assert solved.picard_currents.shape[0] == 1
    row_target, _previous = shape_steering_target(
        machine.profile, seed_target, machine.seed
    )
    expected_target = shape_values(machine.profile, row_target, machine.seed)
    np.testing.assert_allclose(solved.target, expected_target, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(solved.right_hand_side, 0.0, rtol=0.0, atol=0.0)
    assert solved.picard_boundary_flux[0] == pytest.approx(expected_target[0])
    matrix = solved.response[:, solved.free_circuits] * solved.row_weight[:, None]
    vector = solved.right_hand_side * solved.row_weight
    normal_residual = (
        matrix.T @ (matrix @ solved.delta - vector) + solved.gamma**2 * solved.delta
    )
    assert np.linalg.norm(normal_residual) < 3.0e-10 * np.sqrt(matrix.shape[0])
    assert solved.right_null_space.shape == (
        solved.free_circuits.size - solved.numerical_rank,
        solved.free_circuits.size,
    )


def test_null_command_preserves_seed_through_one_forward_solve(machine):
    """A seed-derived target leaves its boundary and circuit currents unchanged."""
    from apps.playable.production import ProductionSolver

    profile = machine.profile
    seed_current = np.asarray(profile.operator.prescribed_current_field.current)
    solver = ProductionSolver(machine)
    seed_result = solver._reduced(profile, machine.seed, seed_current)
    seed = ProductionSolver._reduced_receipt(profile, seed_result)
    seed_target = achieved_target(profile, seed.flux)
    inverse = solve_shape_inverse(
        profile,
        seed_target,
        seed.flux,
        prescribed_current=seed_current,
        free_circuits=machine.drivable_circuits,
        forward_solve=solver._forward_axis_referee(profile, seed.flux),
    )
    equilibrium, _trips, _program = solver._forward_after_admission(
        profile, seed.flux, inverse.currents
    )
    achieved_target(profile, equilibrium.flux)

    assert float(np.max(np.abs(inverse.right_hand_side))) < 1.0e-12
    np.testing.assert_array_equal(inverse.delta, np.zeros_like(inverse.delta))
    assert inverse.iterations == ()
    assert inverse.admissibility_trials == 0
    np.testing.assert_array_equal(
        inverse.turning_point_residual,
        np.zeros_like(inverse.turning_point_residual),
    )
    assert inverse.turning_point_residual_norm == 0.0
    np.testing.assert_array_equal(inverse.currents, seed_current)


def test_current_step_cap_is_relative_to_each_seed_circuit():
    """Every update stays within its circuit's fixed seed-current box."""
    applied, limited = _cap_current_delta(
        np.asarray([20.0, -40.0, 1.0]),
        np.asarray([100.0, 200.0, 0.0]),
        0.1,
    )
    np.testing.assert_array_equal(applied, np.asarray([10.0, -20.0, 0.0]))
    assert limited


def test_dimensionless_delta_regularisation_uses_the_stated_current_scale(
    machine, seed_target
):
    """A delta penalty applies to fractions of each caller-stated ceiling."""
    current = np.asarray(machine.profile.operator.prescribed_current_field.current)
    points = np.asarray(seed_target.flux_points).copy()
    points[1, 1] += 0.02
    target = replace(
        seed_target,
        flux_points=points,
        radial_field_points=points[[0, 2]],
        vertical_field_points=points[[1, 3]],
    )
    ceiling = 20_000.0
    weight = 0.25
    free_circuits = np.arange(0, current.size, 2)
    solved = solve_shape_inverse(
        machine.profile,
        target,
        machine.seed,
        prescribed_current=current,
        gamma=0.0,
        picard_rounds=0,
        free_circuits=free_circuits,
        delta_regularisation=weight,
        delta_current_scale=ceiling,
    )

    matrix = solved.response[:, solved.free_circuits] * solved.row_weight[:, None]
    scale = np.full(solved.free_circuits.size, ceiling)
    rhs = solved.right_hand_side * solved.row_weight
    scaled_matrix = matrix * scale[np.newaxis, :]
    augmented_matrix = np.vstack((scaled_matrix, np.sqrt(weight) * np.eye(scale.size)))
    augmented_rhs = np.concatenate((rhs, np.zeros(scale.size)))
    expected_delta = (
        scale * np.linalg.lstsq(augmented_matrix, augmented_rhs, rcond=None)[0]
    )
    stronger = solve_shape_inverse(
        machine.profile,
        target,
        machine.seed,
        prescribed_current=current,
        gamma=0.0,
        picard_rounds=0,
        free_circuits=free_circuits,
        delta_regularisation=2.0 * weight,
        delta_current_scale=ceiling,
    )

    assert solved.delta_regularisation == weight
    np.testing.assert_allclose(solved.delta_current_scale, scale)
    np.testing.assert_allclose(solved.delta, expected_delta, rtol=1.0e-6)
    assert np.linalg.norm(stronger.delta) < np.linalg.norm(solved.delta)


def test_dimensionless_delta_regularisation_requires_a_scale(machine, seed_target):
    """A nonzero dimensionless penalty cannot silently use seed currents."""
    with pytest.raises(ValueError, match="delta_current_scale"):
        solve_shape_inverse(
            machine.profile,
            seed_target,
            machine.seed,
            delta_regularisation=1.0,
        )


@pytest.mark.parametrize("parameter", ("bulk_r", "bulk_z"))
def test_bulk_motion_translates_all_four_turning_points_rigidly(seed_target, parameter):
    """Bulk controls move the bounding box without deforming it."""
    shape = PlasmaShape()
    delta = 0.017
    moved = move_bounding_box(seed_target, shape, parameter, delta)
    before = np.asarray(seed_target.flux_points)
    after = np.asarray(moved.flux_points)
    component = 0 if parameter == "bulk_r" else 1
    expected = np.zeros_like(before)
    expected[:, component] = delta
    np.testing.assert_allclose(after - before, expected, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(moved.radial_field_points), after[[0, 2]])
    np.testing.assert_allclose(np.asarray(moved.vertical_field_points), after[[1, 3]])


def test_production_solver_runs_one_forward_after_the_inverse(monkeypatch, seed_target):
    """The supplied program serves the one admitting forward solve and result."""
    from apps.playable import production

    current_field = SimpleNamespace(current=np.asarray([2.0, -3.0]))
    profile = SimpleNamespace(
        operator=SimpleNamespace(prescribed_current_field=current_field)
    )
    machine = production.ForwardMachine(
        profile=profile,
        seed=np.zeros(3),
        wall=np.zeros((0, 2)),
        identity="stub",
    )
    solver = production.ProductionSolver(machine)
    previous = SimpleNamespace(flux=np.zeros(3))
    forward_calls = []
    carried_program = object()
    admitted_program = object()

    monkeypatch.setattr(
        production, "achieved_target", lambda _profile, _flux: seed_target
    )

    def inverse(
        _profile,
        _target,
        _flux,
        *,
        prescribed_current,
        free_circuits,
        gamma,
        current_step_fraction,
        current_step_reference,
        forward_solve,
    ):
        assert free_circuits is None
        assert gamma == production.GAMMA
        assert current_step_fraction is None
        np.testing.assert_allclose(current_step_reference, [2.0, -3.0])
        currents = np.asarray(prescribed_current) + 1.0
        forward_solve(currents)
        return SimpleNamespace(currents=currents)

    monkeypatch.setattr(production, "solve_shape_inverse", inverse)
    monkeypatch.setattr(
        production, "turning_point_error", lambda _profile, _target, _flux: 0.01
    )

    def forward(_profile, flux, prescribed_current):
        assert solver._program_handle is carried_program
        forward_calls.append(np.asarray(prescribed_current).copy())
        return (
            SimpleNamespace(
                flux=np.asarray(flux) + 1.0,
                fixed_point=SimpleNamespace(active_set_iterations=2),
            ),
            2,
            admitted_program,
        )

    solver._forward = forward
    result = solver(
        previous,
        PlasmaShape().apply("bulk_z", 0.01),
        action=("bulk_z", 0.01),
        program=carried_program,
    )
    assert len(forward_calls) == 1
    np.testing.assert_allclose(forward_calls, [[3.0, -2.0]])
    assert len(solver.last_rounds) == 1
    assert solver.last_rounds[-1].turning_point_error == 0.01
    assert result.trips == 2
    assert result.reused is True
    assert result.program is admitted_program


def test_limited_fixture_linear_upper_authority_has_commanded_sign_and_gain(machine):
    """The limited fixture retains increasing linear upper-point authority."""
    from apps.playable.production import ProductionSolver

    profile = machine.profile
    current = np.asarray(profile.operator.prescribed_current_field.current)
    prime_solver = ProductionSolver(machine)
    prime_result = prime_solver._reduced(profile, machine.seed, current)
    prime = ProductionSolver._reduced_receipt(profile, prime_result)
    prime_target = achieved_target(profile, prime.flux)

    upper_prediction = []
    current_change = []
    evidence = []
    for command in (0.005, 0.010, 0.020):
        points = np.asarray(prime_target.flux_points).copy()
        points[1, 1] += command
        target = replace(
            prime_target,
            flux_points=points,
            radial_field_points=points[[0, 2]],
            vertical_field_points=points[[1, 3]],
        )
        inverse = solve_shape_inverse(
            profile, target, prime.flux, prescribed_current=current
        )
        upper_prediction.append(float(inverse.linear_prediction[1]))
        current_change.append(float(np.linalg.norm(inverse.delta)))
        evidence.append(
            {
                "command_m": command,
                "current_change_l2_a": current_change[-1],
                "linear_upper_flux_prediction_wb": upper_prediction[-1],
            }
        )
    print("limited-directional-evidence " + json.dumps(evidence, sort_keys=True))
    assert np.all(np.asarray(upper_prediction) > 0.0)
    assert np.all(np.diff(upper_prediction) > 0.0)
    assert np.all(np.diff(current_change) > 0.0)
