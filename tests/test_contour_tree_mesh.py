"""Contour-mesh carrier checks.

The carrier is the three-direction triangulation of hex cell centres, clipped to
the vessel polygon, with wall vertices valued by linear interpolation on the
containing triangle of the same piecewise-linear field.
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from nova.equilibrium.contour_tree_mesh import (  # noqa: E402
    ContourMesh,
    build_contour_mesh,
)
from nova.equilibrium.wall_mask import material_unit, vessel_unit  # noqa: E402

pytestmark = pytest.mark.slow

# synthetic linear field psi = A*R + B*Z + C
_A, _B, _C = 0.37, -0.21, 1.13


def linear_flux(rz: np.ndarray) -> np.ndarray:
    return _A * rz[:, 0] + _B * rz[:, 1] + _C


def hex_lattice(nx: int, nz: int, pitch: float = 1.0):
    """Return hex cell centres on a regular hex lattice (odd rows offset)."""
    row, col = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    r = col + 0.5 * (row % 2)
    z = row * np.sqrt(3.0) / 2.0
    return np.c_[r.ravel(), z.ravel()] * pitch


def circle_unit(radius: float, n: int = 48, name: str = "vessel"):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return vessel_unit(radius * np.cos(theta), radius * np.sin(theta), name=name)


def bbox_unit(rz: np.ndarray, pad: float = 0.6):
    rmin, zmin = rz.min(axis=0) - pad
    rmax, zmax = rz.max(axis=0) + pad
    return vessel_unit([rmin, rmax, rmax, rmin], [zmin, zmin, zmax, zmax], name="bbox")


def build(rz, wall, **kw):
    return build_contour_mesh(rz, linear_flux(rz), wall, **kw)


def flat(mesh: ContourMesh):
    valid_v = np.asarray(mesh.vertex_valid)
    valid_e = np.asarray(mesh.edge_valid)
    valid_t = np.asarray(mesh.triangles[mesh.triangle_valid])
    return valid_v, valid_e, valid_t


def test_wall_vertex_flux_is_linear_interpolation():
    rz = hex_lattice(9, 9)
    wall = [circle_unit(3.5)]
    mesh = build(rz, wall)
    wall_mask = np.asarray(mesh.vertex_is_wall) & np.asarray(mesh.vertex_valid)
    assert wall_mask.sum() > 0
    got = np.asarray(mesh.vertex_psi)[wall_mask]
    want = linear_flux(np.asarray(mesh.vertex_rz)[wall_mask])
    error = np.max(np.abs(got - want))
    assert error < 1e-12


def test_nearest_centre_valuation_fails_linear_exactness():
    rz = hex_lattice(9, 9)
    mesh = build(rz, [circle_unit(3.5)])
    wall_mask = np.asarray(mesh.vertex_is_wall) & np.asarray(mesh.vertex_valid)
    points = np.asarray(mesh.vertex_rz)[wall_mask]
    centre_flux = linear_flux(rz)
    nearest = centre_flux[
        np.argmin(np.sum((points[:, None, :] - rz[None, :, :]) ** 2, axis=2), axis=1)
    ]
    want = linear_flux(points)
    nearest_error = np.max(np.abs(nearest - want))
    assert nearest_error > 1e-6


def test_valid_vertices_lie_in_vessel_polygon():
    rz = hex_lattice(11, 11)
    circle = circle_unit(4.2)
    mesh = build(rz, [circle])
    valid = np.asarray(mesh.vertex_valid)
    points = np.asarray(mesh.vertex_rz)[valid]
    polygon = shapely.Polygon(np.c_[circle.r, circle.z])
    covered = shapely.covers(polygon, shapely.points(points[:, 0], points[:, 1]))
    # vertices read as outside must sit on the polygon to rounding error
    outside = points[~covered]
    if len(outside):
        distance = shapely.distance(
            polygon, shapely.points(outside[:, 0], outside[:, 1])
        )
        assert float(np.max(distance)) <= 1e-12


def test_bounding_box_control_admits_excludes_vertices():
    rz = hex_lattice(11, 11)
    circle = circle_unit(3.2)
    inside_circle = build(rz, [circle])
    inside_box = build(rz, [bbox_unit(rz)])
    circle_valid = np.asarray(inside_circle.vertex_valid)[: len(rz)]
    box_valid = np.asarray(inside_box.vertex_valid)[: len(rz)]
    # the bounding box admits centres the polygon excludes
    assert box_valid.sum() > circle_valid.sum()
    assert np.any(box_valid & ~circle_valid)


def test_euler_characteristic_of_clipped_domain():
    rz = hex_lattice(9, 9)
    mesh = build(rz, [circle_unit(3.5)])
    valid_v, valid_e, valid_t = flat(mesh)
    v = int(round(valid_v.sum()))
    e = int(round(valid_e.sum()))
    t = int(valid_t.shape[0])
    assert v - e + t == 1


def test_over_capacity_sets_overflow_flag():
    rz = hex_lattice(9, 9)
    mesh = build_contour_mesh(
        rz, linear_flux(rz), [circle_unit(3.5)], vertex_capacity=4
    )
    assert mesh.overflow is True
    # a generous capacity on the same input is not an overflow
    ok = build(rz, [circle_unit(3.5)])
    assert ok.overflow is False


def test_mesh_is_jit_traceable_at_fixed_capacity():
    mesh = build(hex_lattice(9, 9), [circle_unit(3.5)])

    def reduce(m: ContourMesh):
        wall = m.vertex_is_wall & m.vertex_valid
        return jnp.sum(jnp.where(wall, m.vertex_psi, 0.0)) + m.vertex_rz.shape[0]

    eager = reduce(mesh)
    traced = jax.jit(reduce)(mesh)
    np.testing.assert_allclose(float(traced), float(eager))


def test_build_rejects_missing_vessel():
    with pytest.raises(ValueError):
        build_contour_mesh(
            hex_lattice(5, 5),
            linear_flux(hex_lattice(5, 5)),
            [material_unit([0, 1, 1, 0], [0, 0, 1, 1])],
        )
