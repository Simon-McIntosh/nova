"""The edge-sheet tile against the host closed form, entry by entry."""

from __future__ import annotations

import numpy as np
import pytest

import jax

from nova.biot.sheet import sheet_field
from nova.biot.tiledassembly import (
    TilePlan,
    cell_edges,
    cell_interaction,
    packed_sheet_field,
)
from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64, "the sheet tile's contract is float64"


def diamond_mesh(rows: int = 4, columns: int = 4, size: float = 0.5):
    """Return square cells rotated so that no edge is horizontal."""
    angle = 0.3
    turn = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    polygons = []
    for row in range(rows):
        for column in range(columns):
            r = 2.0 + column * size
            z = -1.0 + row * size
            corners = np.array(
                [[r, z], [r + size, z], [r + size, z + size], [r, z + size]]
            )
            polygons.append(corners @ turn.T)
    return polygons


def host_interaction(polygons, target_r, target_z):
    """Return ``G`` from the host sheet route, one edge at a time."""
    edges = cell_edges(polygons)
    count = len(polygons)
    radial = np.zeros((target_r.size, edges.start_r.size))
    height = np.zeros_like(radial)
    for index, (lower, upper) in enumerate(
        zip(
            np.stack([edges.start_r, edges.start_z], axis=1),
            np.stack([edges.end_r, edges.end_z], axis=1),
        )
    ):
        for row, (r, z) in enumerate(zip(target_r, target_z)):
            radial[row, index], height[row, index] = sheet_field(r, z, lower, upper)
    interaction = np.empty((2 * target_r.size, 2 * count))
    interaction[: target_r.size, :count] = radial @ edges.coefficient_r.T
    interaction[target_r.size :, :count] = height @ edges.coefficient_r.T
    interaction[: target_r.size, count:] = radial @ edges.coefficient_z.T
    interaction[target_r.size :, count:] = height @ edges.coefficient_z.T
    return interaction


def contribution_scale(polygons, target_r, target_z):
    """Return the absolute sum of every entry's edge-sheet contributions."""
    edges = cell_edges(polygons)
    count = len(polygons)
    radial = np.zeros((target_r.size, edges.start_r.size))
    height = np.zeros_like(radial)
    for index, (lower, upper) in enumerate(
        zip(
            np.stack([edges.start_r, edges.start_z], axis=1),
            np.stack([edges.end_r, edges.end_z], axis=1),
        )
    ):
        for row, (r, z) in enumerate(zip(target_r, target_z)):
            radial[row, index], height[row, index] = sheet_field(r, z, lower, upper)
    scale = np.empty((2 * target_r.size, 2 * count))
    scale[: target_r.size, :count] = np.abs(radial) @ np.abs(edges.coefficient_r).T
    scale[target_r.size :, :count] = np.abs(height) @ np.abs(edges.coefficient_r).T
    scale[: target_r.size, count:] = np.abs(radial) @ np.abs(edges.coefficient_z).T
    scale[target_r.size :, count:] = np.abs(height) @ np.abs(edges.coefficient_z).T
    return scale


@pytest.fixture(scope="module")
def mesh():
    """Return the cell polygons, their centroids and a tile plan for them."""
    polygons = diamond_mesh()
    centroids = np.array([polygon.mean(axis=0) for polygon in polygons])
    return polygons, centroids[:, 0], centroids[:, 1]


def test_tile_agrees_with_the_host_sheet_route(mesh):
    """Every entry of the device build matches the host closed form to 1e-13."""
    polygons, target_r, target_z = mesh
    plan = TilePlan(
        target_tile=len(polygons),
        source_tile=4 * len(polygons),
        block=64,
        n_panels=16,
        n_nodes=48,
    )
    device = cell_interaction(polygons, target_r, target_z, plan=plan, batched=True)
    host = host_interaction(polygons, target_r, target_z)
    scale = contribution_scale(polygons, target_r, target_z)
    worst = np.max(np.abs(device - host) / scale)
    assert worst <= 1e-13, f"worst entry error {worst:.3e} relative to 1e-13"


def test_reversed_edge_normals_break_the_agreement(mesh):
    """Reversing one cell's edge normals makes the agreement check fail."""
    polygons, target_r, target_z = mesh
    plan = TilePlan(
        target_tile=len(polygons),
        source_tile=4 * len(polygons),
        block=64,
        n_panels=16,
        n_nodes=48,
    )
    flipped = cell_interaction(
        polygons, target_r, target_z, plan=plan, batched=True, reverse_normals=0
    )
    host = host_interaction(polygons, target_r, target_z)
    scale = contribution_scale(polygons, target_r, target_z)
    worst = np.max(np.abs(flipped - host) / scale)
    assert worst > 1e-13, f"the reversed cell still agrees to {worst:.3e}"


def test_packed_kernel_matches_one_host_evaluation():
    """One broadcast call reproduces the scalar closed form for sloped edges."""
    starts = np.array([[1.2, -0.4], [3.1, 0.7], [4.4, -1.9]])
    ends = np.array([[1.35, 0.2], [2.9, 1.1], [4.6, -1.6]])
    radius = np.array([2.0, 2.5, 3.6])
    height = np.array([0.1, -0.5, 0.9])
    radial, vertical = packed_sheet_field(
        jax.numpy, radius, height, starts[:, 0], starts[:, 1], ends[:, 0], ends[:, 1]
    )
    for index in range(starts.shape[0]):
        expected = sheet_field(
            float(radius[index]), float(height[index]), starts[index], ends[index]
        )
        assert np.allclose(
            [float(radial[index]), float(vertical[index])], expected, rtol=1e-13
        )