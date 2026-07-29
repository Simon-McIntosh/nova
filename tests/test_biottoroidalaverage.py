"""Toroidal averaging of axisymmetric and tilted three-dimensional sources."""

from __future__ import annotations

import numpy as np

from nova.biot.biotframe import Source
from nova.biot.fluxcoupling import axis_angle_rotation, transform_points
from nova.biot.toroidalaverage import (
    average_toroidal_field,
    frame_magnetic_field,
    frame_toroidal_average,
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


def _line_source(vertices: np.ndarray) -> Source:
    closed = np.concatenate([vertices, vertices[:1]])
    geometry = PolyLine(closed, minimum_arc_nodes=len(closed) + 1).path_geometry
    geometry["x"] = geometry["x0"]
    geometry["y"] = geometry["y0"]
    geometry["z"] = geometry["z0"]
    geometry["dx"] = 0.01
    geometry["dz"] = 0.01
    return Source(geometry, section="circle", nturn=1)


def _circle_vertices(radius: float, count: int = 96) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.stack(
        [radius * np.cos(angle), radius * np.sin(angle), np.zeros(count)], axis=-1
    )


def test_axisymmetric_average_matches_the_native_operator():
    source = _circle_source(1.3, 0.2)
    radius = np.array([0.5, 0.9, 1.8])
    height = np.array([-0.4, 0.7, -0.1])
    averaged = frame_toroidal_average(
        source, radius, height, count=24, turns=False, reduce=False
    )
    native = frame_magnetic_field(
        source,
        np.stack([radius, np.zeros_like(radius), height], axis=-1),
        turns=False,
        reduce=False,
    )
    np.testing.assert_allclose(averaged.radial, native[..., 0], rtol=2e-14, atol=1e-20)
    np.testing.assert_allclose(averaged.vertical, native[..., 2], rtol=2e-14)
    np.testing.assert_allclose(averaged.toroidal, 0.0, atol=2e-22)
    np.testing.assert_allclose(averaged.residual_rms, 0.0, atol=3e-21)


def test_tilted_source_average_matches_dense_toroidal_sampling():
    vertices = _circle_vertices(1.1)
    rotation = axis_angle_rotation(np.array([1.0, 0.0, 0.0]), 0.07)
    tilted = transform_points(vertices, np.array([0.04, -0.03, 0.1]), rotation)
    source = _line_source(tilted)
    radius = np.array([0.45, 0.75])
    height = np.array([-0.2, 0.4])

    def evaluator(points):
        return frame_magnetic_field(source, points, turns=False, reduce=False).sum(
            axis=1
        )

    sampled = average_toroidal_field(evaluator, radius, height, count=64)
    dense = average_toroidal_field(evaluator, radius, height, count=512)
    np.testing.assert_allclose(sampled.radial, dense.radial, rtol=2e-11, atol=1e-18)
    np.testing.assert_allclose(sampled.toroidal, dense.toroidal, rtol=2e-11, atol=1e-18)
    np.testing.assert_allclose(sampled.vertical, dense.vertical, rtol=2e-11, atol=1e-18)
    np.testing.assert_allclose(
        sampled.residual_rms, dense.residual_rms, rtol=2e-11, atol=1e-18
    )
    assert np.all(dense.residual_rms > 0.0)
