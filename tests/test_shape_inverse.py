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
    achieved_target,
    bounding_box_pairs,
    observed_values,
    response_matrix,
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


def test_unmoved_inverse_preserves_the_seed_currents(machine, seed_target):
    """The seed's own turning points command no edit to its own currents."""
    current = np.asarray(machine.profile.operator.prescribed_current_field.current)
    solved = solve_shape_inverse(
        machine.profile,
        seed_target,
        machine.seed,
        prescribed_current=current,
    )
    np.testing.assert_allclose(solved.currents, current, rtol=0.0, atol=5.0e-4)
    assert np.linalg.norm(solved.delta) < 2.0e-3
    assert solved.row_kinds == ("flux",) * 4 + ("field",) * 4


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


def test_production_solver_uses_at_most_two_unconstrained_forward_rounds(
    monkeypatch, seed_target
):
    """A large point error earns exactly one corrective inverse-forward round."""
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

    def inverse(_profile, _target, _flux, *, prescribed_current, free_circuits):
        assert free_circuits is None
        return SimpleNamespace(currents=np.asarray(prescribed_current) + 1.0)

    monkeypatch.setattr(production, "solve_shape_inverse", inverse)
    errors = iter((0.01, 0.001))
    monkeypatch.setattr(
        production, "turning_point_error", lambda _profile, _target, _flux: next(errors)
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
    assert len(forward_calls) == 2
    np.testing.assert_allclose(forward_calls, [[3.0, -2.0], [4.0, -1.0]])
    assert len(solver.last_rounds) == 2
    assert solver.last_rounds[-1].turning_point_error == 0.001
    assert result.trips == 4
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
        linear_prediction = inverse.response[:, inverse.free_circuits] @ inverse.delta
        linear_command = inverse.target - inverse.observed
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
