"""Contracts for bounding-box targets solved through prescribed currents."""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from apps.playable.shape import PlasmaShape, move_bounding_box
from nova.equilibrium.shape_inverse import (
    GAMMA,
    PICARD_ROUNDS,
    _cap_current_delta,
    achieved_target,
    bounding_box_pairs,
    observed_values,
    response_matrix,
    shape_response_matrix,
    shape_row_target,
    shape_values,
    solve_shape_inverse,
)


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
    assert solved.row_kinds == ("flux",) * 4 + ("field",) * 4
    assert solved.gamma == pytest.approx(GAMMA * solved.plasma_current)
    assert solved.picard_currents.shape[0] == PICARD_ROUNDS + 1
    expected_target = shape_row_target(machine.profile, seed_target, machine.seed)
    np.testing.assert_allclose(solved.target, expected_target, rtol=0.0, atol=0.0)
    assert solved.picard_boundary_flux[0] == pytest.approx(expected_target[0])
    weight = np.asarray(
        [
            1.0 if kind == "flux" else np.sqrt(solved.field_weight)
            for kind in solved.row_kinds
        ]
    )
    matrix = solved.response[:, solved.free_circuits] * weight[:, None]
    vector = solved.right_hand_side * weight
    normal_residual = (
        matrix.T @ (matrix @ solved.delta - vector) + solved.gamma**2 * solved.delta
    )
    assert np.linalg.norm(normal_residual) < 1.0e-10
    assert solved.right_null_space.shape == (
        solved.free_circuits.size - solved.numerical_rank,
        solved.free_circuits.size,
    )


def test_current_step_cap_is_relative_to_each_seed_circuit():
    """Every update stays within its circuit's fixed seed-current box."""
    applied, limited = _cap_current_delta(
        np.asarray([20.0, -40.0, 1.0]),
        np.asarray([100.0, 200.0, 0.0]),
        0.1,
    )
    np.testing.assert_array_equal(applied, np.asarray([10.0, -20.0, 0.0]))
    assert limited


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
    """Placement Picard stays inside the inverse before one forward solve."""
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
    ):
        assert free_circuits is None
        assert gamma == production.GAMMA
        assert current_step_fraction is None
        np.testing.assert_allclose(current_step_reference, [2.0, -3.0])
        return SimpleNamespace(currents=np.asarray(prescribed_current) + 1.0)

    monkeypatch.setattr(production, "solve_shape_inverse", inverse)
    monkeypatch.setattr(
        production, "turning_point_error", lambda _profile, _target, _flux: 0.01
    )

    def forward(_profile, flux, prescribed_current):
        forward_calls.append(np.asarray(prescribed_current).copy())
        return (
            SimpleNamespace(
                flux=np.asarray(flux) + 1.0,
                fixed_point=SimpleNamespace(active_set_iterations=2),
            ),
            2,
            object(),
        )

    solver._forward = forward
    result = solver(
        previous,
        PlasmaShape().apply("bulk_z", 0.01),
        action=("bulk_z", 0.01),
        program=object(),
    )
    assert len(forward_calls) == 1
    np.testing.assert_allclose(forward_calls, [[3.0, -2.0]])
    assert len(solver.last_rounds) == 1
    assert solver.last_rounds[-1].turning_point_error == 0.01
    assert result.trips == 2
    assert result.reused is False


def test_limited_fixture_motion_has_commanded_sign_and_monotonic_gain(
    machine, seed_target
):
    """Weak shaping authority remains directional over increasing commands."""
    from apps.playable.production import ProductionSolver

    profile = machine.profile
    current = np.asarray(profile.operator.prescribed_current_field.current)
    prime_solver = ProductionSolver(machine)
    prime_result = prime_solver._reduced(profile, machine.seed, current)
    prime = ProductionSolver._reduced_receipt(profile, prime_result)
    prime_target = achieved_target(profile, prime.flux)

    # Every comparison gets the same warm re-solve drift. The unchanged-current
    # arm is therefore the physical zero for the commanded arms, rather than
    # the already-converged prime flux.
    null_result = ProductionSolver(machine)._reduced(profile, prime.flux, current)
    null_equilibrium = ProductionSolver._reduced_receipt(profile, null_result)
    null_target = achieved_target(profile, null_equilibrium.flux)
    null_upper = float(np.asarray(null_target.flux_points)[1, 1])
    achieved_motion = []
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
        linear_prediction = inverse.linear_prediction
        linear_command = inverse.right_hand_side
        reduced = ProductionSolver(machine)._reduced(
            profile, prime.flux, inverse.currents
        )
        equilibrium = ProductionSolver._reduced_receipt(profile, reduced)
        achieved = achieved_target(profile, equilibrium.flux)
        motion = float(np.asarray(achieved.flux_points)[1, 1]) - null_upper
        achieved_motion.append(motion)
        evidence.append(
            {
                "command_m": command,
                "relative_motion_m": motion,
                "current_change_l2_a": float(np.linalg.norm(inverse.delta)),
                "linear_row_prediction": linear_prediction.tolist(),
                "linear_row_command": linear_command.tolist(),
            }
        )
    print("limited-directional-evidence " + json.dumps(evidence, sort_keys=True))
    for item in evidence:
        assert item["current_change_l2_a"] > 1.0e-3
        np.testing.assert_allclose(
            item["linear_row_prediction"],
            item["linear_row_command"],
            rtol=2.0e-6,
            atol=2.0e-10,
        )
    assert np.all(np.asarray(achieved_motion) > 0.0)
    assert np.all(np.diff(achieved_motion) > 0.0)
