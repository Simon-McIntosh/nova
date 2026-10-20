"""Check a traced machine carrier and an explicitly undersized neighbour table."""

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from nova.equilibrium import contour_tree, topology  # noqa: E402
from scripts.prototypes.efit_field_swap.crosscheck import (  # noqa: E402
    domain_rings,
    lattice,
    read_trace_record,
)
from scripts.prototypes.efit_field_swap.orientation import read_convention  # noqa: E402
from scripts.prototypes.efit_field_swap.spline_field import TraceSplineField  # noqa: E402
from scripts.prototypes.efit_maximin_certificate.run import ROWS  # noqa: E402

assert jax.config.jax_enable_x64 is True


def mastu_inputs(resolution):
    record = read_trace_record(ROWS[0]["source"])
    field = TraceSplineField.from_record(record)
    boundary, obstacles = domain_rings(record)
    cells, samples = lattice(boundary, resolution, obstacles)
    geometry = topology.TopologyGeometry.from_cells(
        cells, samples, (boundary, *obstacles)
    )
    _, centres = jax.jit(topology._sampled_flux)(field, geometry)
    values = jnp.sum(
        centres[geometry.contour_sources] * geometry.contour_weights, axis=1
    )
    mesh = geometry.contour_mesh
    sigma = read_convention(polarity=record.polarity).sigma
    return (
        (
            values,
            mesh.vertex_valid,
            mesh.vertex_is_wall,
            mesh.edges,
            mesh.edge_valid,
            sigma,
        ),
        len(cells),
        mesh,
    )


def test_small_field_stays_within_capacity():
    tree = contour_tree.build_contour_tree(
        jnp.array((3.0, 2.0, 1.0, 0.0, 0.0)),
        jnp.array((True, True, True, True, False)),
        jnp.array((False, False, True, False, False)),
        jnp.array(((0, 1), (1, 2), (2, 3)), dtype=jnp.int32),
        jnp.ones(3, dtype=bool),
        jnp.array(1.0),
    )
    assert not bool(tree.overflow)


def test_machine_carrier_fits_tree_capacity():
    arguments, _, mesh = mastu_inputs(16)
    edges = np.asarray(mesh.edges)[np.asarray(mesh.edge_valid)]
    degree = np.bincount(edges.ravel(), minlength=mesh.vertex_capacity)
    assert mesh.neighbour_capacity == int(degree.max()) > 0
    tree = contour_tree.build_contour_tree(
        *arguments, neighbour_capacity=mesh.neighbour_capacity
    )
    assert not bool(tree.overflow), (
        f"tree refused the carrier: nodes={int(tree.node_valid.sum())}/"
        f"{contour_tree.ContourTreeResult.node_capacity}, "
        f"edges={int(tree.edge_valid.sum())}/"
        f"{contour_tree.ContourTreeResult.edge_capacity}"
    )


def test_machine_overflow_reproduces():
    arguments, _, _ = mastu_inputs(16)
    tree = contour_tree.build_contour_tree(*arguments)
    assert not bool(tree.overflow)


def test_machine_declared_width_refuses_overflow():
    arguments, _, mesh = mastu_inputs(16)
    tree = contour_tree.build_contour_tree(
        *arguments, neighbour_capacity=mesh.neighbour_capacity - 1
    )
    assert bool(tree.overflow)
