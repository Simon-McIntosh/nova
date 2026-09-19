"""Bounded exterior-field rows on the analytic fixture."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from benchmarks import centroid_constrained_fixture_receipt as control_receipt_module
from scripts.analytic_oracle_fixtures.centroid_row import (
    DEFAULT_FIELD_BOUND_T,
    DEFAULT_FIELD_SCALE_T,
    DEFAULT_LEVEL_SCALE_WB,
    DEFAULT_STEP_LIMIT,
    centroid_constraint_pair,
    exterior_field_identity,
    fixture_constraint_pairs,
    level_constraint_pair,
)
from scripts.analytic_oracle_fixtures.measure import (
    analytic_case,
    exact_state,
    gauge_free_flux_read,
    uniform_exterior_compensation_response,
    uniform_exterior_field_flux,
    uniform_exterior_field_response,
)
from tests.rotating_equilibrium_references import reference_cases


def _reactor_case():
    """Return the reactor-scale closed form the banked control rows use."""
    return reference_cases()["weak-rotation-reactor"].static_limit()


def test_uniform_field_contribution_is_zero_on_the_compensator_anchor() -> None:
    case = _reactor_case()
    axis = np.asarray(case.magnetic_axis, dtype=np.float64)
    field = np.asarray((2.0e-3, -3.0e-4), dtype=np.float64)

    on_anchor = uniform_exterior_field_flux(case, axis.reshape(1, 2), field)
    np.testing.assert_allclose(on_anchor[0], 0.0, rtol=0.0, atol=0.0)

    # the reference radius is the axis, so the column grows with R**2 outward
    radius = 7.718433652945465
    outward = uniform_exterior_field_flux(case, np.asarray(((radius, 0.0),)), field)
    np.testing.assert_allclose(
        outward[0], field[0] * np.pi * (radius**2 - float(axis[0]) ** 2), rtol=0.0
    )
    assert outward[0] > 0.0


def test_gauge_free_span_offset_ignores_an_added_flux_constant() -> None:
    case = _reactor_case()
    axis = np.asarray(case.magnetic_axis, dtype=np.float64)
    boundary = np.asarray(((7.718433652945465, -1.2320178076396662),))
    field = np.asarray((0.002387200675934618, 0.0), dtype=np.float64)
    analytic = exact_state(case, np.vstack((axis, boundary)))

    reading = gauge_free_flux_read(
        case, axis, boundary[0], float(analytic[0]), float(analytic[1]), field
    )
    np.testing.assert_allclose(reading["gauge_free_flux_offset_wb"], 0.0, atol=1e-9)

    # a constant added to both levels is a gauge choice, not a fixed-point error
    shifted = gauge_free_flux_read(
        case,
        axis,
        boundary[0],
        float(analytic[0]) + 12.5,
        float(analytic[1]) + 12.5,
        field,
    )
    np.testing.assert_allclose(
        shifted["gauge_free_flux_offset_wb"],
        reading["gauge_free_flux_offset_wb"],
        rtol=0.0,
        atol=0.0,
    )

    # the authored zero level sits on the analytic separatrix, not at the axis
    np.testing.assert_allclose(
        float(exact_state(case, np.asarray(((7.801395788446069, 0.0),)))[0]),
        0.0,
        atol=1.0e-6,
    )


def test_uniform_exterior_response_has_declared_field_components() -> None:
    case = analytic_case()
    points = np.asarray(
        ((1.1, -0.2), (1.1, 0.3), (1.6, -0.2), (1.6, 0.3)),
        dtype=np.float64,
    )
    machine = SimpleNamespace(
        node=points[:2],
        wall_node=points[2:3],
        sample_coordinates=points[3:],
    )

    response = uniform_exterior_field_response(case, machine)
    radius = points[:, 0]
    vertical_radial_derivative = (response[2:, 0] - response[:2, 0]) / 0.5
    radial_midpoint = 0.5 * (radius[2:] + radius[:2])
    radial_vertical_derivative = response[[1, 3], 1] - response[[0, 2], 1]

    np.testing.assert_allclose(
        vertical_radial_derivative / (2.0 * np.pi * radial_midpoint),
        1.0,
        rtol=0.0,
        atol=3.0e-16,
    )
    np.testing.assert_allclose(
        -radial_vertical_derivative / (2.0 * np.pi * radius[[0, 2]] * 0.5),
        1.0,
        rtol=0.0,
        atol=3.0e-16,
    )


def test_centroid_pair_maps_rows_to_bounded_field_directions() -> None:
    pair = centroid_constraint_pair(
        jnp.asarray([1.4, 0.0]),
        pitch=0.05,
    )

    assert pair.functional.components == ("centroid_r", "centroid_z")
    np.testing.assert_array_equal(
        np.asarray(pair.unknown.direction),
        np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))),
    )
    np.testing.assert_allclose(pair.unknown.field_bound, DEFAULT_FIELD_BOUND_T)
    assert exterior_field_identity()["centroid_r_compensator"] == (
        "uniform vertical field"
    )
    with np.testing.assert_raises_regex(
        ValueError, "exterior-field amplitude exceeds its declared finite bound"
    ):
        pair.unknown.require_within_bound(jnp.asarray([1.0e6, 0.0]))


def test_centroid_pair_step_control_caps_and_records_refusal() -> None:
    pair = centroid_constraint_pair(
        jnp.asarray([1.4, 0.0]),
        pitch=0.05,
    )
    np.testing.assert_allclose(pair.unknown.step_limit, DEFAULT_STEP_LIMIT)

    # a row residual asking for more than the cap is damped, not clipped:
    # the per-trip change lands exactly on the cap and the tangent survives
    step, refused = pair.unknown.damped_step(jnp.zeros(2), jnp.asarray([10.0, 0.0]))
    np.testing.assert_allclose(step[0], -DEFAULT_STEP_LIMIT, rtol=0.0, atol=0.0)
    assert not bool(np.asarray(refused).any())

    # a physical field past the declared bound is refused and the step holds
    over_bound = jnp.full(2, 2.0 * DEFAULT_FIELD_BOUND_T / DEFAULT_FIELD_SCALE_T)
    step, refused = pair.unknown.damped_step(over_bound, jnp.zeros(2))
    np.testing.assert_array_equal(step, np.zeros(2))
    assert bool(np.asarray(refused).all())


def _sample_machine(points: np.ndarray) -> SimpleNamespace:
    """Return a three-part target layout over the given coordinates in order."""
    return SimpleNamespace(
        node=points[:2],
        wall_node=points[2:3],
        sample_coordinates=points[3:],
    )


def test_compensation_family_level_column_is_a_uniform_offset() -> None:
    case = analytic_case()
    points = np.asarray(
        ((1.1, -0.2), (1.1, 0.3), (1.6, -0.2), (1.6, 0.3)),
        dtype=np.float64,
    )
    machine = _sample_machine(points)

    family = uniform_exterior_compensation_response(case, machine)
    fields = uniform_exterior_field_response(case, machine)
    assert family.shape[1] == 3
    np.testing.assert_allclose(family[:, :2], fields, rtol=0.0, atol=0.0)
    # the level column is constant at every target, so its gradient vanishes
    # and it contributes no poloidal field
    np.testing.assert_array_equal(family[:, 2], np.ones(len(points)))

    # the fixture closed form reads the same level the third column applies
    level_only = uniform_exterior_field_flux(
        case, points, np.asarray((0.0, 0.0, DEFAULT_LEVEL_SCALE_WB))
    )
    np.testing.assert_allclose(level_only, family[:, 2] * DEFAULT_LEVEL_SCALE_WB)


def test_level_pair_carries_its_amplitude_outside_the_field_bound() -> None:
    pairs = fixture_constraint_pairs(
        jnp.asarray([1.42, 0.0]),
        level_point=np.asarray((1.7, 0.0)),
        level_target=jnp.asarray([1.38]),
        pitch=0.05,
    )
    centroid_pair, level_pair = pairs

    np.testing.assert_array_equal(
        np.asarray(centroid_pair.unknown.direction[2]), np.zeros(2)
    )
    assert level_pair.functional.row_count == 1
    np.testing.assert_array_equal(
        np.asarray(level_pair.unknown.direction), np.asarray(((0.0,), (0.0,), (1.0,)))
    )
    assert not bool(np.asarray(level_pair.unknown.field_bound_applies)[0])
    assert np.isinf(float(np.asarray(level_pair.unknown.field_bound)[0]))

    # the level amplitude keeps its own scale and its own 1e-12 clause
    payload = np.asarray(level_pair.binding.payload)
    np.testing.assert_allclose(payload, np.asarray(((1.7, 0.0),)))
    np.testing.assert_allclose(
        np.asarray(level_pair.binding.scale), DEFAULT_LEVEL_SCALE_WB
    )
    np.testing.assert_allclose(
        np.asarray(level_pair.binding.tolerance),
        DEFAULT_LEVEL_SCALE_WB * 1.0e-12,
    )

    # a level amplitude far past the tesla bound is never refused
    step, refused = level_pair.unknown.damped_step(jnp.zeros(1), jnp.asarray([1.0e6]))
    assert not bool(np.asarray(refused).any())
    assert float(np.asarray(step)[0]) < 1.0e6

    identity = exterior_field_identity()
    assert identity["response_columns"] == ["vertical", "radial", "level"]
    assert identity["level_bound_wb"] is None
    assert identity["level_bound_is_field_bound"] is False
    assert identity["level_compensator"] == "uniform flux offset in weber"


def test_level_pair_rejects_a_non_level_centroid_component() -> None:
    with np.testing.assert_raises_regex(ValueError, "level column"):
        centroid_constraint_pair(
            jnp.asarray([1.42, 0.0]), pitch=0.05, components=("centroid_r", "level")
        )
    with np.testing.assert_raises_regex(ValueError, "one target value"):
        level_constraint_pair(np.asarray((1.7, 0.0)), jnp.asarray([1.38, 0.0, 0.0]))


def test_control_render_refuses_a_state_that_misses_its_receipt_digest(
    tmp_path: Path,
) -> None:
    """A panel is drawn only from the state its receipt names and digests.

    The digest is checked before the fixture context is built, so a receipt
    naming a state it did not hash is refused rather than drawn into a figure
    that would then be cited as that arm's terminal panel.
    """
    state_path = tmp_path / "control-positive-state.npy"
    np.save(state_path, np.asarray([1.0, 2.0, 3.0], dtype=np.float64))
    (tmp_path / "control-positive.json").write_text(
        json.dumps(
            {
                "terminal_state_path": str(state_path),
                "terminal_state_sha256_binary64": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with np.testing.assert_raises_regex(ValueError, "does not hash to the digest"):
        control_receipt_module.render_control_state(
            tmp_path, tmp_path / "control-positive.png", "positive"
        )
