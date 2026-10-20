"""Refuse a carrier wider than the neighbour table a caller declares.

The wall-complete MAST-U carrier no longer overflows the tree's node and edge
capacities, and its wall repair also lowered its measured vertex degree, so the
recorded refusal is reproduced where it still belongs: a declared neighbour
width below the carrier's measured degree refuses, and the same carrier fits a
table sized from that measured degree.  The small-field test is the positive
control on an ordinary graph.

The refusal record on the pre-repair carrier reported measured degrees 19, 22
and 18 at these resolutions, all above the then fixed width of 16.  The wall
coverage repair re-cut the carrier, and this probe now measures the degrees
pinned in ``RESOLUTION_DEGREES``; width 16 therefore no longer refuses here and
the reproduction uses a width below the carrier's own degree.
"""

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from nova.equilibrium import contour_tree  # noqa: E402
from scripts.prototypes.contour_tree_overflow.probe import mastu_inputs  # noqa: E402

assert jax.config.jax_enable_x64 is True

# The live maximum degree of each lattice resolution, as the carrier reports it.
# The historical pre-repair degrees were 19, 22 and 18 at these resolutions.
RESOLUTION_DEGREES = ((16, 9), (24, 11), (32, 12))


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


def test_declared_width_below_the_measured_degree_refuses_the_machine_carrier():
    """A width under the carrier's own degree refuses; its own degree fits.

    The refusal is what the test is named for, and it is asserted against the
    carrier's measured degree rather than the historical fixed width, because
    that width no longer sits below this carrier's degree.
    """

    for resolution, degree in RESOLUTION_DEGREES:
        arguments, _, mesh = mastu_inputs(resolution)
        assert int(mesh.neighbour_capacity) == degree, (
            f"resolution {resolution} measured degree {int(mesh.neighbour_capacity)}"
            f" rather than {degree}"
        )

        undersized = contour_tree.build_contour_tree(
            *arguments, neighbour_capacity=degree - 1
        )
        assert bool(undersized.overflow), (
            f"resolution {resolution} did not refuse width {degree - 1} against a "
            f"degree of {degree}"
        )

        measured = contour_tree.build_contour_tree(
            *arguments, neighbour_capacity=degree
        )
        assert not bool(measured.overflow), (
            f"resolution {resolution} did not fit its measured degree {degree}"
        )
