"""Fixed-shape finite-arc driver and tile registry contract."""

import warnings

import numpy as np
import pytest

from nova.biot.polygon import pad_batch
from nova.biot.polygonarc import packed_arc_greens, polygon_arc_greens
from nova.biot.tiledassembly import TilePlan, tile_evaluator


def rectangle():
    return np.array([[5.9, -0.1], [6.1, -0.1], [6.1, 0.1], [5.9, 0.1]])


def diamond():
    return np.array([[6.0, -0.12], [6.12, 0.0], [6.0, 0.12], [5.88, 0.0]])


def regular(sides, radius):
    angle = 0.17 + np.arange(sides) * 2.0 * np.pi / sides
    return np.column_stack([6.0 + radius * np.cos(angle), radius * np.sin(angle)])


def test_the_packed_arc_matches_the_shortcut_host_driver():
    sections = [rectangle(), diamond()]
    edge, weight, norm = pad_batch(sections)
    target_r = np.array([5.7, 6.3, 5.8, 6.2])
    target_z = np.array([0.2, -0.2, 0.0, 0.1])
    target_phi = np.array([0.2, 1.3, 2.1, -0.4])
    source = np.array([0, 0, 1, 1])
    start = np.array([0.4, 0.4, -0.2, -0.2])
    end = np.array([2.1, 2.1, 1.4, 1.4])
    got = np.stack(
        packed_arc_greens(
            np,
            target_r,
            target_z,
            target_phi,
            edge[:, :, source],
            weight[:, source],
            norm[source],
            start,
            end,
        )
    )
    expected = np.column_stack(
        [
            polygon_arc_greens(
                target_r[index],
                target_z[index],
                target_phi[index],
                sections[source[index]],
                start[index],
                end[index],
            )
            for index in range(len(target_r))
        ]
    )
    np.testing.assert_allclose(got, expected, rtol=3e-11, atol=5e-19)


def test_the_registry_rejects_incompatible_arc_routes_before_compiling():
    plan = TilePlan(2, 2, 1, 1, 1)
    with pytest.raises(ValueError, match="only the closed"):
        tile_evaluator(plan, geometry="arc", kernel="quadrature")
    with pytest.raises(ValueError, match="require batched"):
        tile_evaluator(plan, geometry="arc", kernel="closed", batched=False, devices=2)
    with pytest.raises(ValueError, match="unknown geometry"):
        tile_evaluator(plan, geometry="helix")


def test_heterogeneous_padding_is_finite_at_source_corners():
    """Four-, five- and six-edge arcs hold every dead row before reduction."""
    sections = [regular(4, 0.08), regular(5, 0.07), regular(6, 0.06)]
    edge, weight, norm = pad_batch(sections)
    target_r = np.array([section[index, 0] for index, section in enumerate(sections)])
    target_z = np.array([section[index, 1] for index, section in enumerate(sections)])
    target_phi = np.array([0.2, 1.1, -0.4])
    start = np.array([0.4, -0.2, 0.1])
    end = np.array([2.1, 1.4, 1.9])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        got = np.stack(
            packed_arc_greens(
                np,
                target_r,
                target_z,
                target_phi,
                edge,
                weight,
                norm,
                start,
                end,
                nodes=32,
            )
        )
    assert np.all(np.isfinite(got))
    expected = np.column_stack(
        [
            polygon_arc_greens(
                target_r[index],
                target_z[index],
                target_phi[index],
                section,
                start[index],
                end[index],
                nodes=32,
            )
            for index, section in enumerate(sections)
        ]
    )
    np.testing.assert_allclose(got, expected, rtol=3e-11, atol=5e-19)


def test_arc_padding_coordinates_carry_zero_jax_tangents():
    """The finite-arc trace holds every padded lane before differentiation."""
    import jax
    import jax.numpy as jnp

    from nova.jax.config import configure_dtypes

    configure_dtypes()
    sections = [regular(4, 0.08), regular(5, 0.07), regular(6, 0.06)]
    edge, weight, norm = pad_batch(sections)
    target_r = np.array([0.0, sections[1][1, 0], sections[2][2, 0]])
    target_z = np.array([0.0, sections[1][1, 1], sections[2][2, 1]])
    target_phi = np.array([0.2, 1.1, -0.4])
    start = np.array([0.4, -0.2, 0.1])
    end = np.array([2.1, 1.4, 1.9])
    pad_tangent = np.zeros_like(edge)
    pad_tangent[4:, :, 0] = 1.0
    pad_tangent[5:, :, 1] = 1.0

    def evaluate(one_edge):
        return jnp.stack(
            packed_arc_greens(
                jnp,
                jnp.asarray(target_r),
                jnp.asarray(target_z),
                jnp.asarray(target_phi),
                one_edge,
                jnp.asarray(weight),
                jnp.asarray(norm),
                jnp.asarray(start),
                jnp.asarray(end),
                nodes=4,
            )
        )

    primal, tangent = jax.jvp(
        evaluate,
        (jnp.asarray(edge),),
        (jnp.asarray(pad_tangent),),
    )
    assert np.all(np.isfinite(primal))
    assert np.all(np.isfinite(tangent))
    np.testing.assert_array_equal(tangent, 0.0)
