"""Fixed-capacity separatrix branch assembly over the global tensor spline."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.flux_surface_connectivity import (
    fit_tensor_spline,
    polish_stationary_points,
)
from nova.equilibrium.separatrix_branches import assemble_separatrix_branches


jax.config.update("jax_enable_x64", True)


def _grid(lower=-2.0, upper=2.0, count=34):
    coordinate = jnp.linspace(lower, upper, count, dtype=jnp.float64)
    radial, vertical = jnp.meshgrid(coordinate, coordinate)
    return coordinate, radial, vertical


def _diverted_field(coordinate):
    radial, vertical = jnp.meshgrid(coordinate, coordinate)
    along_lobe = (radial + vertical) / jnp.sqrt(2.0)
    across_lobe = (vertical - radial) / jnp.sqrt(2.0)
    return across_lobe**2 - along_lobe**2 + along_lobe**3


def _double_null_field(coordinate, saddle_distance=0.82):
    radial, vertical = jnp.meshgrid(coordinate, coordinate)
    along_lobe = (radial + vertical) / jnp.sqrt(2.0)
    across_lobe = (vertical - radial) / jnp.sqrt(2.0)
    return across_lobe**2 - (along_lobe**2 - saddle_distance**2) ** 2


def _double_null_level(values, coordinate, saddle_distance=0.82):
    spline = fit_tensor_spline(coordinate, coordinate, values)
    offset = saddle_distance / jnp.sqrt(2.0)
    seeds = jnp.asarray(((offset, offset), (-offset, -offset)))
    polished = polish_stationary_points(spline, seeds, jnp.asarray((True, True)))
    np.testing.assert_allclose(
        np.asarray(polished["value"]),
        np.full(2, float(jnp.mean(polished["value"]))),
        atol=2e-17,
        rtol=0.0,
    )
    return jnp.mean(polished["value"]), polished["position_rz"]


def _active_controls(result, key, valid_key, branch=None):
    controls = np.asarray(result[key])
    valid = np.asarray(result[valid_key])
    if branch is not None:
        controls = controls[branch]
        valid = valid[branch]
    return controls[valid]


def test_diverted_field_has_closed_lobe_and_two_distinct_saddle_ended_legs():
    coordinate, _radial, _vertical = _grid()
    result = assemble_separatrix_branches(
        _diverted_field(coordinate),
        coordinate,
        coordinate,
        jnp.asarray(0.0),
        jnp.asarray((0.47, 0.47)),
    )

    assert bool(result["well_formed"])
    assert int(result["closed_candidate_count"]) == 1
    assert int(result["open_branch_count"]) == 2
    closed = _active_controls(result, "closed_controls_rz", "closed_valid")
    legs = [
        _active_controls(result, "open_controls_rz", "open_valid", branch=index)
        for index in range(2)
    ]
    np.testing.assert_array_equal(closed[0, 0], closed[-1, -1])
    np.testing.assert_array_equal(legs[0][-1, -1], closed[0, 0])
    np.testing.assert_array_equal(legs[1][-1, -1], closed[0, 0])
    assert not np.array_equal(legs[0][0, 0], legs[1][0, 0])


def test_limited_field_has_one_closed_branch_and_zero_open_slots():
    coordinate, radial, vertical = _grid()
    result = assemble_separatrix_branches(
        radial**2 + vertical**2,
        coordinate,
        coordinate,
        jnp.asarray(1.0),
        jnp.asarray((0.0, 0.0)),
    )

    assert bool(result["well_formed"])
    assert int(result["closed_candidate_count"]) == 1
    assert int(result["open_branch_count"]) == 0
    assert not np.any(np.asarray(result["open_branch_valid"]))
    np.testing.assert_array_equal(np.asarray(result["open_controls_rz"]), 0.0)


def test_double_null_uses_deterministic_distinct_leg_slots():
    coordinate, _radial, _vertical = _grid()
    values = _double_null_field(coordinate)
    level, saddles = _double_null_level(values, coordinate)
    result = assemble_separatrix_branches(
        values,
        coordinate,
        coordinate,
        level,
        jnp.asarray((0.0, 0.0)),
        open_branch_capacity=6,
    )

    assert bool(result["well_formed"])
    assert int(result["open_branch_count"]) == 4
    terminals = []
    starts = []
    for branch in range(4):
        controls = _active_controls(
            result, "open_controls_rz", "open_valid", branch=branch
        )
        starts.append(controls[0, 0])
        terminals.append(controls[-1, -1])
    terminals = np.asarray(terminals)
    distances = np.linalg.norm(terminals[:, None, :] - np.asarray(saddles), axis=-1)
    np.testing.assert_array_equal(np.sum(distances < 1e-12, axis=0), (2, 2))
    assert len({tuple(point) for point in starts}) == 4


def test_transforms_zero_padding_and_overflow_fail_closed():
    coordinate, radial, vertical = _grid()
    fields = jnp.stack((_diverted_field(coordinate), radial**2 + vertical**2))
    levels = jnp.asarray((0.0, 1.0))
    axes = jnp.asarray(((0.47, 0.47), (0.0, 0.0)))

    batched = jax.vmap(
        lambda field, level, axis: assemble_separatrix_branches(
            field, coordinate, coordinate, level, axis
        )
    )(fields, levels, axes)
    per_slice = jax.tree.map(
        lambda *items: jnp.stack(items),
        *[
            assemble_separatrix_branches(
                fields[index], coordinate, coordinate, levels[index], axes[index]
            )
            for index in range(2)
        ],
    )
    compiled = jax.jit(
        lambda field, level, axis: assemble_separatrix_branches(
            field, coordinate, coordinate, level, axis
        )
    )(fields[0], levels[0], axes[0])
    for key in batched:
        if jnp.issubdtype(batched[key].dtype, jnp.inexact):
            np.testing.assert_allclose(
                np.asarray(batched[key]),
                np.asarray(per_slice[key]),
                atol=2e-15,
                rtol=0.0,
            )
            np.testing.assert_allclose(
                np.asarray(compiled[key]),
                np.asarray(per_slice[key][0]),
                atol=2e-15,
                rtol=0.0,
            )
        else:
            np.testing.assert_array_equal(
                np.asarray(batched[key]), np.asarray(per_slice[key])
            )
            np.testing.assert_array_equal(
                np.asarray(compiled[key]), np.asarray(per_slice[key][0])
            )

    for geometry, valid in (
        (batched["closed_controls_rz"], batched["closed_valid"]),
        (batched["open_controls_rz"], batched["open_valid"]),
    ):
        np.testing.assert_array_equal(np.asarray(geometry)[~np.asarray(valid)], 0.0)

    overflow = assemble_separatrix_branches(
        fields[0],
        coordinate,
        coordinate,
        levels[0],
        axes[0],
        branch_capacity=4,
    )
    assert bool(overflow["overflow"])
    assert not bool(overflow["well_formed"])
    np.testing.assert_array_equal(np.asarray(overflow["closed_controls_rz"]), 0.0)
    np.testing.assert_array_equal(np.asarray(overflow["open_controls_rz"]), 0.0)


def test_closed_branch_geometry_is_invariant_to_open_leg_domain_extension():
    short_coordinate = jnp.arange(-2.0625, 2.0626, 0.125)
    long_coordinate = jnp.arange(-3.0625, 2.0626, 0.125)
    short = assemble_separatrix_branches(
        _diverted_field(short_coordinate),
        short_coordinate,
        short_coordinate,
        jnp.asarray(0.0),
        jnp.asarray((0.47, 0.47)),
    )
    long = assemble_separatrix_branches(
        _diverted_field(long_coordinate),
        long_coordinate,
        long_coordinate,
        jnp.asarray(0.0),
        jnp.asarray((0.47, 0.47)),
    )
    short_closed = _active_controls(short, "closed_controls_rz", "closed_valid")
    long_closed = _active_controls(long, "closed_controls_rz", "closed_valid")

    assert int(long["open_segment_count"][0]) > int(short["open_segment_count"][0])
    assert int(long["open_segment_count"][1]) > int(short["open_segment_count"][1])
    np.testing.assert_array_equal(long_closed, short_closed)
