"""Three-dimensional loop-flux coupling and rigid-transform contracts."""

from __future__ import annotations

import numpy as np
import pytest

from nova.biot.biotframe import Source
from nova.biot.fluxcoupling import (
    axis_angle_rotation,
    circular_loop_quadrature,
    circular_mutual_inductance,
    frame_loop_coupling,
    polygon_loop_quadrature,
    transform_points,
)
from nova.geometry.polyline import PolyLine


def _circle_source(radius: float, height: float = 0.0) -> Source:
    return Source(
        {
            "x": [radius],
            "y": [0.0],
            "z": [height],
            "dl": [1e-7],
            "dt": [1e-7],
        },
        segment="circle",
        section="circle",
        nturn=1,
    )


def _circle_vertices(radius: float, height: float = 0.0, count: int = 96) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.stack(
        [
            radius * np.cos(angle),
            radius * np.sin(angle),
            np.full(count, height),
        ],
        axis=-1,
    )


def _line_source(vertices: np.ndarray) -> Source:
    closed = np.concatenate([vertices, vertices[:1]])
    geometry = PolyLine(closed, minimum_arc_nodes=len(closed) + 1).path_geometry
    geometry["x"] = geometry["x0"]
    geometry["y"] = geometry["y0"]
    geometry["z"] = geometry["z0"]
    geometry["dx"] = 0.01
    geometry["dz"] = 0.01
    return Source(geometry, section="circle", nturn=1)


def test_circular_pair_matches_maxwell_mutual_inductance():
    source_radius = 1.2
    target_radius = 0.8
    separation = 0.5
    coupling = frame_loop_coupling(
        _circle_source(source_radius, height=0.3),
        circular_loop_quadrature(target_radius, center=np.array([0.0, 0.0, -0.2])),
        turns=False,
        reduce=False,
    )
    expected = circular_mutual_inductance(source_radius, target_radius, separation)
    np.testing.assert_allclose(coupling, [expected], rtol=2e-13)


def test_circular_pair_is_reciprocal():
    first = frame_loop_coupling(
        _circle_source(1.4, -0.3),
        circular_loop_quadrature(0.7, center=np.array([0.0, 0.0, 0.4])),
        turns=False,
        reduce=False,
    )
    second = frame_loop_coupling(
        _circle_source(0.7, 0.4),
        circular_loop_quadrature(1.4, center=np.array([0.0, 0.0, -0.3])),
        turns=False,
        reduce=False,
    )
    np.testing.assert_allclose(first, second, rtol=3e-13)


def test_polygon_loop_coupling_is_rigid_transform_invariant():
    source_vertices = _circle_vertices(1.0, count=72)
    target_vertices = _circle_vertices(0.55, height=0.8, count=64)
    baseline = frame_loop_coupling(
        _line_source(source_vertices),
        polygon_loop_quadrature(target_vertices),
        turns=False,
        reduce=False,
    ).sum()

    rotation = axis_angle_rotation(np.array([0.4, -0.2, 0.7]), 0.63)
    translation = np.array([1.7, -0.8, 0.4])
    transformed_source = transform_points(source_vertices, translation, rotation)
    transformed_target = transform_points(target_vertices, translation, rotation)
    moved = frame_loop_coupling(
        _line_source(transformed_source),
        polygon_loop_quadrature(transformed_target),
        turns=False,
        reduce=False,
    ).sum()
    np.testing.assert_allclose(moved, baseline, rtol=2e-12)


def test_zero_rigid_transform_preserves_coupling_exactly():
    vertices = _circle_vertices(0.9, count=48)
    transformed = transform_points(vertices, np.zeros(3), np.eye(3))
    np.testing.assert_array_equal(transformed, vertices)


def test_small_vertical_displacement_matches_the_maxwell_expansion():
    source_radius = 1.1
    target_radius = 0.65
    separation = 0.4
    delta = 2e-5

    def computed(gap: float) -> float:
        value = frame_loop_coupling(
            _circle_source(source_radius),
            circular_loop_quadrature(target_radius, center=np.array([0.0, 0.0, gap])),
            turns=False,
            reduce=False,
        )
        return float(value[0])

    measured = (computed(separation + delta) - computed(separation - delta)) / (
        2.0 * delta
    )
    expected = (
        circular_mutual_inductance(source_radius, target_radius, separation + delta)
        - circular_mutual_inductance(source_radius, target_radius, separation - delta)
    ) / (2.0 * delta)
    np.testing.assert_allclose(measured, expected, rtol=2e-9)


def test_rigid_transform_has_fixed_shapes_under_vmap():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    points = jnp.asarray(_circle_vertices(0.5, count=12))
    translations = jnp.asarray([[0.0, 0.0, 0.0], [0.2, -0.1, 0.3]])
    rotations = jnp.stack([jnp.eye(3), jnp.eye(3)])
    transformed = jax.vmap(transform_points)(
        points[None].repeat(2, axis=0), translations, rotations
    )
    assert transformed.shape == (2, 12, 3)
    np.testing.assert_allclose(
        np.asarray(transformed[1] - transformed[0]),
        np.broadcast_to(np.asarray(translations[1]), points.shape),
        atol=3e-8,
    )
