"""The boundary event lies inside independently certified escape brackets."""

import jax
import pytest

from nova.jax.config import configure_dtypes
from scripts.prototypes.efit_field_swap.crosscheck import read_trace_record
from scripts.prototypes.efit_maximin_certificate.adapter_probe import decision
from scripts.prototypes.efit_maximin_certificate.run import ROWS


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    ("spec", "lower", "upper", "boundary_class"),
    [
        (ROWS[0], 0.0196524250845, 0.019652442526100, 1),
        (ROWS[1], 0.014214693, 0.014214735554751, 0),
        (ROWS[2], 0.012577131, 0.012577168750081, 0),
    ],
    ids=[row["case"] for row in ROWS],
)
def test_boundary_matches_certified_escape(spec, lower, upper, boundary_class):
    """Attaining paths and continuous event bounds enclose the escape level.

    The synthetic upper bounds enclose the whole left wall in Bernstein
    pieces of length 1e-5 with a 1e-12 arithmetic pad. The specified endpoint
    at Z=0.2 is below the nearby wall tangency and cannot bound that event.
    """
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    reading = decision(read_trace_record(spec["source"]))
    level = reading["boundary_flux"]
    assert lower <= level <= upper, (
        f"{spec['case']}: boundary {level:.17g} is outside the "
        f"certified escape interval [{lower:.17g}, {upper:.17g}]"
    )
    assert reading["boundary_class"] == boundary_class


@pytest.mark.parametrize("resolution", [16, 24, 32])
@pytest.mark.parametrize("spec", ROWS[1:], ids=[row["case"] for row in ROWS[1:]])
def test_contour_domain_reaches_the_vessel(spec, resolution):
    """The authority carries the supplied wall, including at coarse resolution."""
    import numpy as np
    from shapely.geometry import Polygon
    from nova.equilibrium.topology import TopologyGeometry
    from scripts.prototypes.efit_field_swap.crosscheck import domain_rings, lattice

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    boundary, obstacles = domain_rings(read_trace_record(spec["source"]))
    cells, samples = lattice(boundary, resolution, obstacles)
    geometry = TopologyGeometry.from_cells(cells, samples, (boundary, *obstacles))
    mesh = geometry.contour_mesh
    live = np.asarray(mesh.vertex_valid)
    wall = live & np.asarray(mesh.vertex_is_wall)
    assert wall.any(), "nonempty vessel has zero contour-wall vertices"
    all_points = np.asarray(mesh.vertex_rz)[live]
    assert len(np.unique(np.round(all_points, 10), axis=0)) == len(all_points)
    points = np.asarray(mesh.vertex_rz)[wall]
    vessel = Polygon(boundary)
    from shapely.geometry import Point

    assert min(Point(point).distance(vessel.boundary) for point in points) < 1e-10
    assert abs(points[:, 0].min() - boundary[:, 0].min()) < 1e-10


def test_contour_domain_preserves_a_material_hole_and_capacity_refusal():
    """An internal material ring removes triangles and declared ceilings refuse."""
    import numpy as np
    from shapely.geometry import Polygon
    from nova.equilibrium.topology import TopologyGeometry
    from nova.equilibrium.contour_tree_mesh import (
        build_contour_mesh,
        ContourMeshOverflow,
    )
    from nova.equilibrium.wall_mask import WallUnit
    from scripts.prototypes.efit_field_swap.crosscheck import lattice

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    boundary = np.array(((1.0, -1.0), (3.0, -1.0), (3.0, 1.0), (1.0, 1.0)))
    obstacle = np.array(((1.8, -0.2), (2.2, -0.2), (2.2, 0.2), (1.8, 0.2)))
    cells, samples = lattice(boundary, 16, (obstacle,))
    geometry = TopologyGeometry.from_cells(cells, samples, (boundary, obstacle))
    mesh = geometry.contour_mesh
    points = np.asarray(mesh.vertex_rz)
    triangles = np.asarray(mesh.triangles)[np.asarray(mesh.triangle_valid)]
    hole = Polygon(obstacle)
    assert hole.area > 0.15
    assert sum(Polygon(points[t]).intersection(hole).area for t in triangles) < 1e-12
    with pytest.raises(ContourMeshOverflow, match="vertex capacity 1 is below"):
        build_contour_mesh(
            np.asarray(geometry.centre),
            np.zeros(len(cells)),
            (WallUnit(boundary[:, 0], boundary[:, 1], kind="vessel"),),
            vertex_capacity=1,
        )
