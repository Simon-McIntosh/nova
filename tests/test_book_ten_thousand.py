"""Exact-support reference clips retain polygon parts of mixed intersections."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from types import SimpleNamespace

import numpy as np
from shapely.geometry import Polygon

from tests.equilibrium.test_topology_read import _kernel_backed_field


class PolygonalOracle:
    """A concave support with a line contact beside positive cell area."""

    major_radius = 1.5
    pressure_coefficient = 1e-7
    field_coefficient = 1e-7

    def separatrix(self, _points):
        return np.asarray(((0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)), dtype=float)


def test_reference_integrates_polygon_in_geometry_collection():
    """A lower-arm overlap survives an upper-arm edge contact."""
    problem = np.asarray(((1, 0.5), (2, 0.5), (2, 2), (1, 2)), dtype=float)
    control = np.asarray(((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)))
    oracle = PolygonalOracle()
    clipped = Polygon(problem).intersection(Polygon(oracle.separatrix(0)))
    assert clipped.geom_type == "GeometryCollection"
    assert clipped.area == 0.5
    assert any(part.geom_type == "Polygon" for part in clipped.geoms)
    geometry = SimpleNamespace(
        vertices=np.stack((problem, control)), vertex_count=np.asarray((4, 4))
    )
    reference = _kernel_backed_field("diverted", oracle, None, geometry)
    assert float(reference.moments.cell_current[1]) > 0
    assert float(reference.moments.cell_current[0]) > 0
