"""Equivalence of the sparse contour-tree build with the recorded base build.

The sparse build resolves carrier adjacency, split-node insertion and the
node-to-vertex inverse map by sorting and scatter instead of the repeated
full-table scans and the dense node-by-vertex equality matrix. The receipts it
produces are pinned bit-for-bit against digests recorded from the byte-identical
base build on the two analytic cases at 550 and 2,000 realized cells, so the
carrier slot order, wall-contact semantics, tree arcs, critical-type assignment
and every node field are shown unchanged rather than assumed. The digests come
from the same fixture path the H200 ladder measures.
"""

from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import contour_tree, topology
from nova.equilibrium.contour_tree import build_contour_tree
from nova.jax.config import configure_dtypes
from tests.equilibrium import test_topology_read as oracle


configure_dtypes()
assert jax.config.jax_enable_x64 is True

_FIELDS = (
    "node_vertex",
    "node_psi",
    "node_valid",
    "critical_type",
    "edges",
    "edge_valid",
    "overflow",
)

#: Receipt digests recorded from the base build, keyed ``<kind>-<cells>``.
_BASE_DIGESTS = {
    "limited-550": "370f09336502a318e7c22723ede6a7258a0f9b5b1965befddb230fd024921ed0",
    "limited-2000": "b1633e4ca1d6a004d7cf15f3a03456c3ad6144eb926d210715d9391d56f3a4e4",
    "diverted-550": "435b4357987cac5831b605cd5147990531f3b7f50a19c1276624c07114c8da3a",
    "diverted-2000": "5067b635f72a8b931dc0c898e6db081eea31598dc9dd42aeb61f56cffd8170d5",
}


def _receipt(kind: str, cells: int) -> contour_tree.ContourTreeArrays:
    """Build one receipt from the measured fixture path at ``kind``, ``cells``."""

    _oracle, field, wall, _axis, _saddle = oracle._analytic_inputs(kind)
    geometry = oracle._realised_hex_geometry(wall, cells)
    _coefficient, centre_flux = topology._sampled_flux(field, geometry)
    vertex_flux = jnp.sum(
        centre_flux[geometry.contour_sources] * geometry.contour_weights, axis=1
    )
    mesh = geometry.contour_mesh
    return build_contour_tree(
        jnp.asarray(vertex_flux),
        mesh.vertex_valid,
        mesh.vertex_is_wall,
        mesh.edges,
        mesh.edge_valid,
        jnp.asarray(1, dtype=jnp.int32),
    )


def _digest(receipt: contour_tree.ContourTreeArrays) -> str:
    """A field-ordered digest over every receipt array's exact bytes."""

    digest = hashlib.sha256()
    for name in _FIELDS:
        digest.update(name.encode())
        digest.update(np.asarray(getattr(receipt, name)).tobytes())
    return digest.hexdigest()


def test_sparse_indexed_build_marker_is_present():
    """The sparse build path marks itself, so its absence is a failing test.

    A restored dense equality matrix and the repeated scans drop this marker;
    the equivalence digests below then describe a tree the module no longer
    builds, which is the declared mutation this check must fail against.
    """

    assert contour_tree.SPARSE_INDEXED_BUILD is True


@pytest.mark.parametrize("kind", ["limited", "diverted"])
@pytest.mark.parametrize("cells", [550, 2000])
def test_sparse_build_matches_base_receipt(kind: str, cells: int):
    """The sparse receipt is bit-identical to the recorded base receipt."""

    receipt = _receipt(kind, cells)
    assert np.asarray(receipt.overflow) == np.asarray(False)
    assert int(np.sum(receipt.node_valid)) >= 2
    assert _digest(receipt) == _BASE_DIGESTS[f"{kind}-{cells}"]
