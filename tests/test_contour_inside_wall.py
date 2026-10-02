"""A scattered-node contour never draws a vertex outside the first wall.

A Delaunay triangulation spans the convex hull of its nodes, so across a
concave wall it bridges the gap and a contour through the bridge is a field
the machine cannot hold. The fixture is a vessel with a slot cut into its top
and nodes that fill only the vessel interior; contours of a field centred on
the slot cross it. Every vertex the painter draws must lie inside the wall
units, and the unmasked triangulation of the same nodes must not -- the second
assertion is what shows the first one can fail.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pytest

from nova.equilibrium.wall_mask import material_unit, vessel_unit
from nova.media.poloidal import (
    draw_flux_contours,
    draw_scattered_contours,
    draw_wall,
    triangles_outside_wall,
)
from nova.media.sources.frame import inside_wall_units

SLOT = np.array(
    [
        (1.0, -2.0),
        (3.0, -2.0),
        (3.0, 2.0),
        (2.2, 2.0),
        (2.2, 0.0),
        (1.8, 0.0),
        (1.8, 2.0),
        (1.0, 2.0),
    ]
)
LEVELS = np.array([0.05, 0.1, 0.2, 0.3, 0.5])


def _nodes(wall) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    radius, height = np.meshgrid(np.linspace(1.0, 3.0, 41), np.linspace(-2.0, 2.0, 81))
    points = np.column_stack((radius.ravel(), height.ravel()))
    points = points + rng.normal(scale=1.0e-3, size=points.shape)
    points = points[inside_wall_units(points, wall)]
    flux = (points[:, 0] - 2.0) ** 2 + (points[:, 1] - 1.0) ** 2
    return points[:, 0], points[:, 1], flux


def _vertices(contours) -> np.ndarray:
    segments = [seg for group in contours.allsegs for seg in group if len(seg)]
    assert segments, "the contour set drew nothing"
    return np.vstack(segments)


@pytest.fixture
def wall():
    return (vessel_unit(SLOT[:, 0], SLOT[:, 1]),)


def test_contour_vertices_lie_inside_the_wall(wall):
    radius, height, flux = _nodes(wall)
    fig, axes = plt.subplots()
    try:
        contours = draw_scattered_contours(axes, radius, height, flux, LEVELS, wall)
        vertices = _vertices(contours)
        outside = ~inside_wall_units(vertices, wall)
        assert not outside.any(), (
            f"{int(outside.sum())} of {len(vertices)} drawn contour vertices "
            f"lie outside the wall, first at {vertices[outside][0]}"
        )
    finally:
        plt.close(fig)


def test_unmasked_triangulation_of_the_same_nodes_draws_outside_the_wall(wall):
    radius, height, flux = _nodes(wall)
    fig, axes = plt.subplots()
    try:
        contours = axes.tricontour(
            mtri.Triangulation(radius, height), flux, levels=LEVELS
        )
        vertices = _vertices(contours)
        assert (~inside_wall_units(vertices, wall)).any()
    finally:
        plt.close(fig)


def test_mask_flags_bridging_triangles_and_keeps_interior_ones(wall):
    radius, height, _ = _nodes(wall)
    triangles = mtri.Triangulation(radius, height).triangles
    outside = triangles_outside_wall(radius, height, triangles, wall)
    assert outside.any() and not outside.all()
    kept = np.stack((radius[triangles[~outside]], height[triangles[~outside]]), -1)
    assert inside_wall_units(kept.mean(axis=1), wall).all()


def test_contour_vertices_avoid_a_material_hole():
    vessel = vessel_unit([1.0, 3.0, 3.0, 1.0], [-2.0, -2.0, 2.0, 2.0], name="vessel")
    hole = material_unit([1.8, 2.2, 2.2, 1.8], [0.0, 0.0, 1.0, 1.0], name="tile")
    wall = (vessel, hole)
    radius, height, flux = _nodes(wall)
    fig, axes = plt.subplots()
    try:
        contours = draw_scattered_contours(axes, radius, height, flux, LEVELS, wall)
        vertices = _vertices(contours)
        assert inside_wall_units(vertices, wall).all()
    finally:
        plt.close(fig)


def test_wall_is_required_and_an_empty_mesh_is_refused(wall):
    radius, height, flux = _nodes(wall)
    fig, axes = plt.subplots()
    try:
        with pytest.raises(TypeError):
            draw_scattered_contours(axes, radius, height, flux, LEVELS)
        far = (vessel_unit([10.0, 11.0, 11.0, 10.0], [0.0, 0.0, 1.0, 1.0]),)
        with pytest.raises(ValueError, match="outside the wall"):
            draw_scattered_contours(axes, radius, height, flux, LEVELS, far)
    finally:
        plt.close(fig)


def test_painter_draws_alongside_the_wall(wall):
    radius, height, flux = _nodes(wall)
    fig, axes = plt.subplots()
    try:
        draw_wall(axes, units=wall)
        contours = draw_scattered_contours(axes, radius, height, flux, LEVELS, wall)
        assert len(contours.allsegs) == len(LEVELS)
    finally:
        plt.close(fig)


def test_raster_contours_blank_the_field_outside_the_wall(wall):
    radius, height, flux = _nodes(wall)
    grid_r = np.linspace(1.0, 3.0, 161)
    grid_z = np.linspace(-2.0, 2.0, 321)
    raster = mtri.LinearTriInterpolator(mtri.Triangulation(radius, height), flux)(
        *np.meshgrid(grid_r, grid_z)
    )
    raster = np.asarray(np.ma.filled(raster, np.nan))
    fig, axes = plt.subplots()
    try:
        bare = draw_flux_contours(axes, grid_r, grid_z, raster, LEVELS)
        assert (~inside_wall_units(_vertices(bare), wall)).any()
        masked = draw_flux_contours(axes, grid_r, grid_z, raster, LEVELS, wall=wall)
        vertices = _vertices(masked)
        slot = (vertices[:, 0] > 1.8) & (vertices[:, 0] < 2.2) & (vertices[:, 1] > 0.0)
        assert not slot.any(), f"{int(slot.sum())} vertices drawn in the slot"
    finally:
        plt.close(fig)
