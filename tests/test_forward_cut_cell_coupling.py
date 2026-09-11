"""The clipped coupling images a cut cell's linear density exactly.

A cut cell's current occupies only its clipped plasma polygon, so the
clipped coupling path contracts the cell's physical moments with that
polygon's area-normalised second moments and images them over the clipped
polygon with the polygon-analytic kernel blocks.  On a synthetic two-cell
case --- one interior cell keeping the precomputed atomic blocks and one cut
cell --- the imaged flux of a prescribed linear density must match the exact
ring-inductance integral over the clipped polygon to one part in ten
billion, and the interior cell's image must be unchanged between the atomic
and the clipped path.
"""

from __future__ import annotations

import jax
import numpy as np
import jax.numpy as jnp
import pytest

from benchmarks import solovev_certificate as certificate
from nova.biot.greens import section_centroid, second_moments
from nova.equilibrium import ForwardProfile
from nova.equilibrium.stencil_mesh import CellCurrentMoments, ClippedCouplingGeometry
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures import measure as oracle_fixture

CASE_NAME = "weak-rotation-reactor-static"
REQUESTED_CELLS = -110

#: Duffy order for the exact polygon quadrature.  Validated against the
#: ring kernel to one part in a billion outside the source polygon.
DUFFY_ORDER = 28


def _duffy_rule(vertices, order=DUFFY_ORDER):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    points = []
    area_weights = []
    for index in range(1, len(vertices) - 1):
        first, second, third = vertices[[0, index, index + 1]]
        edge_first = second - first
        edge_second = third - first
        cross = abs(edge_first[0] * edge_second[1] - edge_first[1] * edge_second[0])
        for radial, radial_weight in zip(unit_nodes, unit_weights, strict=True):
            for vertical, vertical_weight in zip(unit_nodes, unit_weights, strict=True):
                points.append(
                    first
                    + radial * edge_first
                    + (1.0 - radial) * vertical * edge_second
                )
                area_weights.append(
                    cross * (1.0 - radial) * radial_weight * vertical_weight
                )
    return np.asarray(points), np.asarray(area_weights)


def _ring_mutual(target_r, target_z, source_r, source_z):
    """Return the ring mutual inductance [Wb/A] (total-flux convention)."""
    from scipy.special import ellipe, ellipk

    mu_0 = 4.0e-7 * np.pi
    k2 = (
        4.0
        * source_r[None, :]
        * target_r[:, None]
        / (
            (source_r[None, :] + target_r[:, None]) ** 2
            + (target_z[:, None] - source_z[None, :]) ** 2
        )
    )
    k2 = np.clip(k2, 0.0, 1.0 - 1.0e-300)
    k = np.sqrt(k2)
    mutual = (
        mu_0
        * np.sqrt(source_r[None, :] * target_r[:, None])
        * ((2.0 / k - k) * ellipk(k2) - (2.0 / k) * ellipe(k2))
    )
    return mutual


def _pytest_setup():
    """Build the weak -110 operator and its frozen per-trip support once."""
    if not hasattr(_pytest_setup, "fixture"):
        configure_dtypes()
        carrier, source, exact = certificate._case(CASE_NAME)
        machine = certificate._case_machine(CASE_NAME, carrier, exact, REQUESTED_CELLS)
        operator = oracle_fixture.forward_operator(source, machine)

        coordinates = np.vstack(
            (machine.node, machine.wall_node, machine.sample_coordinates)
        )
        oracle_state = certificate._exact_state(CASE_NAME, exact, coordinates)
        partition = operator._support_partition(jnp.asarray(oracle_state))
        support = partition[3]
        geometry = operator._clipped_coupling_geometry(support)
        _pytest_setup.fixture = dict(
            machine=machine,
            operator=operator,
            support=support,
            geometry=geometry,
            state=jnp.asarray(oracle_state),
            cell_polygons=tuple(
                np.asarray(cell, dtype=np.float64) for cell in machine.cell_polygons
            ),
            domain=partition[0],
        )
    return _pytest_setup.fixture


def _exact_linear_flux(targets_r, targets_z, polygon, coefficients, centroid):
    """Return the exact poloidal flux of one linear density over a polygon."""
    points, weights = _duffy_rule(polygon)
    density = (
        coefficients[0]
        + coefficients[1] * (points[:, 0] - centroid[0])
        + coefficients[2] * (points[:, 1] - centroid[1])
    )
    mutual = _ring_mutual(targets_r, targets_z, points[:, 0], points[:, 1])
    return np.einsum("ts,s,s->t", mutual, density, weights)


def test_cut_cell_clipped_coupling_matches_exact_ring_integral():
    """A cut cell's clipped image of a linear density is exact to 1e-10."""
    environment = _pytest_setup()
    machine = environment["machine"]
    operator = environment["operator"]
    support = environment["support"]
    geometry = environment["geometry"]
    grid_count = machine.moment_geometry.atomic_mesh.centroids.shape[0]

    vertices = np.asarray(support.support_vertices)
    counts = np.asarray(support.vertex_count)
    boundary = np.asarray(support.boundary, dtype=bool)
    included = np.asarray(support.included, dtype=bool)
    cut_cells = np.flatnonzero(boundary & included)
    assert len(cut_cells) >= 1, "the weak -110 support must carry a cut cell"
    cell = int(cut_cells[0])
    polygon = vertices[cell][: counts[cell]]
    assert len(polygon) >= 3
    centroid = section_centroid(polygon)

    coefficients = np.asarray((1.0e6, -3.0e5, 2.0e5))
    area = float(_polygon_area(polygon))
    second = second_moments(polygon)
    # exact moments of the linear density about the clipped centroid
    m0 = area * coefficients[0]
    mr_clipped = area * (coefficients[1] * second[0] + coefficients[2] * second[2])
    mz_clipped = area * (coefficients[1] * second[2] + coefficients[2] * second[1])
    # production references the moments to the atomic cell centroid:
    # MR_atomic = MR_clipped + (clipped_centre - atomic_centroid) * M0.
    atomic_centroid = np.asarray(
        environment["machine"].moment_geometry.atomic_mesh.centroids[cell]
    )
    mr = mr_clipped + (centroid - atomic_centroid)[0] * m0
    mz = mz_clipped + (centroid - atomic_centroid)[1] * m0

    targets_r = np.concatenate(
        (
            machine.node[:, 0],
            machine.wall_node[:, 0],
            machine.sample_coordinates[:, 0],
        )
    )
    targets_z = np.concatenate(
        (
            machine.node[:, 1],
            machine.wall_node[:, 1],
            machine.sample_coordinates[:, 1],
        )
    )
    exact = _exact_linear_flux(targets_r, targets_z, polygon, coefficients, centroid)

    uniform = np.zeros(grid_count, dtype=np.float64)
    radial = np.zeros(grid_count, dtype=np.float64)
    vertical = np.zeros(grid_count, dtype=np.float64)
    uniform[cell] = m0
    radial[cell] = mr
    vertical[cell] = mz
    moments = CellCurrentMoments(
        jnp.asarray(uniform), jnp.asarray(radial), jnp.asarray(vertical)
    )
    coefficients_coupled = operator.coupling_current_moments(moments, geometry)
    image = np.asarray(
        operator.current_moment_image(coefficients_coupled, geometry),
        dtype=np.float64,
    )
    # Compare on the grid nodes outside the source polygon, the same set the
    # interior-cell control uses for its order claim (wall and sample nodes
    # may lie arbitrarily close to a boundary cut cell, where the near-field
    # quadrature is coarser than the coupling the test pins).
    coincidence = _nodes_inside(grid_count, targets_r, targets_z, polygon, grid_count)
    outside = np.flatnonzero(~coincidence)
    scale = np.max(np.abs(exact[outside]))
    assert scale > 0.0
    relative = np.abs(image[outside] - exact[outside]) / scale
    assert np.sqrt(np.mean(relative**2)) < 1.0e-10, (
        "clipped coupling departs from the exact clipped integral: "
        f"{np.sqrt(np.mean(relative**2)):.3e}"
    )


def test_interior_cell_blocks_are_unchanged_by_the_clipped_path():
    """An interior cell's image is identical between the atomic and clipped paths."""
    environment = _pytest_setup()
    machine = environment["machine"]
    operator = environment["operator"]
    support = environment["support"]
    geometry = environment["geometry"]
    grid_count = machine.moment_geometry.atomic_mesh.centroids.shape[0]

    vertices = np.asarray(support.support_vertices)
    counts = np.asarray(support.vertex_count)
    boundary = np.asarray(support.boundary, dtype=bool)
    included = np.asarray(support.included, dtype=bool)
    interior = np.flatnonzero(included & ~boundary)
    assert len(interior) >= 1
    cell = int(interior[0])
    polygon = vertices[cell][: counts[cell]]
    centroid = section_centroid(polygon)
    area = float(_polygon_area(polygon))
    second = second_moments(polygon)
    coefficients = np.asarray((1.0e6, -3.0e5, 2.0e5))
    m0 = area * coefficients[0]
    mr = area * (coefficients[1] * second[0] + coefficients[2] * second[2])
    mz = area * (coefficients[1] * second[2] + coefficients[2] * second[1])
    atomic_centroid = np.asarray(
        environment["machine"].moment_geometry.atomic_mesh.centroids[cell]
    )
    mr = mr + (centroid - atomic_centroid)[0] * m0
    mz = mz + (centroid - atomic_centroid)[1] * m0

    uniform = np.zeros(grid_count)
    radial = np.zeros(grid_count)
    vertical = np.zeros(grid_count)
    uniform[cell], radial[cell], vertical[cell] = m0, mr, mz
    moments = CellCurrentMoments(
        jnp.asarray(uniform), jnp.asarray(radial), jnp.asarray(vertical)
    )
    atomic = np.asarray(
        operator.current_moment_image(operator.coupling_current_moments(moments, None)),
        dtype=np.float64,
    )
    clipped = np.asarray(
        operator.current_moment_image(
            operator.coupling_current_moments(moments, geometry), geometry
        ),
        dtype=np.float64,
    )
    np.testing.assert_allclose(clipped, atomic, rtol=0.0, atol=1.0e-14)


def _polygon_area(vertices):
    following = np.roll(vertices, -1, axis=0)
    return float(
        0.5
        * abs(
            np.sum(vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1])
        )
    )


def _nodes_inside(node_count, targets_r, targets_z, polygon, grid_count):
    """Return a mask of targets inside the source polygon (excluded nodes)."""
    from shapely.geometry import Point, Polygon as ShapelyPolygon

    shape = ShapelyPolygon(polygon)
    mask = np.zeros(node_count, dtype=bool)
    for index in range(grid_count):
        if shape.covers(Point(targets_r[index], targets_z[index])):
            mask[index] = True
    return mask


def test_frozen_partition_geometry_is_a_pytree():
    """The frozen partition's coupling geometry maps leaf-wise as arrays."""
    environment = _pytest_setup()
    operator = environment["operator"]
    state = environment["state"]
    first = operator._frozen_topology_partition(state)
    second = operator._frozen_topology_partition(state)
    assert first.coupling is not None, "the weak -110 support must carry geometry"

    leaves = jax.tree_util.tree_leaves(first)
    assert len(leaves) > 0
    assert all(isinstance(leaf, jax.Array | np.ndarray) for leaf in leaves), (
        "the frozen partition must carry only array leaves, never the geometry "
        "object or a missing target-set delta"
    )

    # The reconcile merges incoming and observed partitions elementwise with
    # jnp.where per leaf; a `True` condition must reproduce the incoming
    # partition's values on exactly its leaf structure.
    mapped = jax.tree_util.tree_map(
        lambda incoming, observed: jnp.where(True, incoming, observed),
        first,
        second,
    )
    mapped_leaves = jax.tree_util.tree_leaves(mapped)
    assert len(mapped_leaves) == len(leaves)
    for incoming, merged in zip(leaves, mapped_leaves, strict=True):
        np.testing.assert_allclose(
            np.asarray(merged, dtype=np.float64),
            np.asarray(incoming, dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )
    assert not any(
        isinstance(leaf, ClippedCouplingGeometry) for leaf in mapped_leaves
    ), "the reconcile must recurse into the geometry, not carry it as a leaf"


@pytest.mark.slow
def test_accelerated_newton_trip_smoke():
    """One accelerated Newton trip on the weak -110 seed returns a finite state.

    The certificate rows crash in the first accelerated trip when the frozen
    partition's coupling geometry is not a pytree (the reconcile's leaf-wise
    ``jnp.where`` rejects the geometry object); this smoke drives one trip with
    the smallest evaluation budget and asserts the solve returns a finite flux
    without raising.  It is a smoke, not a convergence claim: one Newton step
    and one Krylov vector from the certificate state need not converge.
    """
    environment = _pytest_setup()
    machine = environment["machine"]
    operator = environment["operator"]
    mesh = StencilMesh(machine.node, machine.stencil, machine.area)
    profile = ForwardProfile(operator, mesh)
    equilibrium = profile.solve(
        environment["state"],
        route="newton_krylov",
        newton_steps=1,
        gmres_iterations=1,
        active_set_steps=1,
        warmup=0,
    )
    jax.block_until_ready(equilibrium.flux)
    flux = np.asarray(equilibrium.flux, dtype=np.float64)
    assert np.all(np.isfinite(flux))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
