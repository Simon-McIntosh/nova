"""Refuse a carrier wider than the neighbour table a caller declares.

The wall-complete MAST-U carrier no longer overflows the tree's node and edge
capacities, so the recorded refusal is reproduced where it still belongs: a
declared neighbour width below the carrier's measured degree refuses, and the
same carrier fits a table sized from that measured degree.  Selecting the fixed
width of 16 for this carrier is the negative control; the small-field test is
the positive control on the ordinary graph.
"""

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from nova.equilibrium import contour_tree  # noqa: E402
from scripts.prototypes.contour_tree_overflow.probe import mastu_inputs  # noqa: E402

assert jax.config.jax_enable_x64 is True

# For each lattice resolution: the measured maximum degree of the live carrier.
# The degree rises with the mesh and is the width the production read supplies.
MEASURED_DEGREES = ((16, 19), (24, 22), (32, 18))


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


def test_machine_carrier_fits_the_measured_degree_table():
    arguments, _, mesh = mastu_inputs(16)
    tree = contour_tree.build_contour_tree(
        *arguments, neighbour_capacity=mesh.neighbour_capacity
    )
    assert not bool(tree.overflow), (
        f"tree refused the carrier: nodes={int(tree.node_valid.sum())}/"
        f"{tree.node_vertex.size}, "
        f"edges={int(tree.edge_valid.sum())}/{tree.edges.shape[0]}"
    )


def test_declared_fixed_width_still_refuses_the_machine_carrier():
    """The historical refusal is now expected only at the fixed width of 16.

    Every MAST-U carrier here has a live degree above 16, so a caller that
    declares 16 neighbours loses real adjacency and the receipt must refuse
    rather than report a topology built from a partial table.
    """

    for resolution, degree in MEASURED_DEGREES:
        arguments, _, mesh = mastu_inputs(resolution)
        assert int(mesh.neighbour_capacity) == degree, (
            f"resolution {resolution} measured degree {int(mesh.neighbour_capacity)}"
            f" rather than {degree}"
        )

        fixed = contour_tree.build_contour_tree(*arguments, neighbour_capacity=16)
        assert bool(fixed.overflow), f"resolution {resolution} did not refuse width 16"

        measured = contour_tree.build_contour_tree(
            *arguments, neighbour_capacity=degree
        )
        assert not bool(measured.overflow), (
            f"resolution {resolution} did not fit its measured degree {degree}"
        )
