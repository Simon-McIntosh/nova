"""Analytic controls for the grid-seeded stationary-point finder."""

import numpy as np
import pytest

from scripts.prototypes.tie_pair_topology.stationary import stationary_points


def analytic_field(point, saddle=True):
    x, y = point
    if saddle:
        return (
            x**4 / 4 - x**2 / 2 - y**2 / 2,
            np.array([x**3 - x, -y]),
            np.diag([3 * x * x - 1, -1]),
        )
    return -x * x / 2 - y * y / 2, np.array([-x, -y]), np.diag([-1, -1])


def find(saddle=True):
    def values(points):
        return np.array([analytic_field(point, saddle)[0] for point in points])

    return stationary_points(
        values,
        lambda point: analytic_field(point, saddle),
        (-1.5, 1.5, -1, 1),
        lambda points: np.ones(len(points), bool),
        resolution=101,
    )[0]


def test_finder_recovers_analytic_saddle_and_extremum():
    points = find()
    assert sorted(point.kind for point in points) == ["maximum", "saddle", "saddle"]
    assert any(
        np.linalg.norm(point.position) < 1e-8 and point.kind == "maximum"
        for point in points
    )
    assert any(
        abs(abs(point.position[0]) - 1) < 1e-8 and point.kind == "saddle"
        for point in points
    )


def test_saddle_removed_is_detected():
    points = find(saddle=False)
    assert len(points) == 1
    assert points[0].kind == "maximum"
    with pytest.raises(AssertionError):
        assert any(point.kind == "saddle" for point in points)
