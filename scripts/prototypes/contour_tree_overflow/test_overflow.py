"""Reproduce refusal of a traced machine carrier while keeping a small control."""

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from nova.equilibrium import contour_tree  # noqa: E402
from scripts.prototypes.contour_tree_overflow.probe import mastu_inputs  # noqa: E402

assert jax.config.jax_enable_x64 is True


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
    arguments, _, _ = mastu_inputs(16)
    tree = contour_tree.build_contour_tree(*arguments)
    assert not bool(tree.overflow), (
        f"tree refused the carrier: nodes={int(tree.node_valid.sum())}/"
        f"{contour_tree.ContourTreeResult.node_capacity}, "
        f"edges={int(tree.edge_valid.sum())}/"
        f"{contour_tree.ContourTreeResult.edge_capacity}"
    )


def test_machine_overflow_reproduces():
    arguments, _, _ = mastu_inputs(16)
    tree = contour_tree.build_contour_tree(*arguments)
    assert bool(tree.overflow)
