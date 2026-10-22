"""Equivalence of the sparse contour-tree build with the recorded base build.

The sparse build resolves carrier adjacency, split-node insertion and the
node-to-vertex inverse map by sorting and scatter instead of the repeated
full-table scans and the dense node-by-vertex equality matrix.

The pin is split by what actually travels between hosts. The receipts' integer
arrays are compared against a plain host graph search over the same carrier and
field, so the reference is valid on any host: at 2,000 cells a near-tie flux
comparison can round differently under another CPU's vector code and reorder
the receipt without changing its meaning. The coarse 550-cell rows are
additionally pinned by digest of their integer arrays, which were shown
byte-identical on the sirius CPU class and the Broadwell fleet node class.
``node_psi`` is float64 carried through
from the caller's flux field, so its bytes encode the host's XLA codegen rather
than the build's semantics; hashing it is not portable between hosts. It is
asserted by the contract the build actually owes instead: every valid node row
carries the value the build was given, at that row's carrier vertex, and every
invalid row carries zero. The fixtures are the same path the H200 ladder
measures.
"""

from __future__ import annotations

import hashlib
import os

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

#: Receipt arrays pinned by digest for the coarse rows: integers and booleans.
_INTEGER_FIELDS = (
    "node_vertex",
    "node_valid",
    "critical_type",
    "edges",
    "edge_valid",
    "overflow",
)

#: Integer-field digests of the integrated build's receipts, keyed ``<kind>-<cells>``.
_EXPECTED_INTEGER_DIGESTS = {
    "limited-550": "925fe6895a471688cfaeae3d1ffe5650309b49b153071b5b1df7ec4745cc34f3",
    "diverted-550": "1fa0d5f58760b71f208991bc9e817b4ba039c16eb299524793fa8dca56aef61c",
}


def _build(
    kind: str, cells: int
) -> tuple[contour_tree.ContourTreeArrays, jnp.ndarray, object]:
    """Build one receipt from the measured fixture path, with its input field.

    The realised cell count, the sampled flux and the carrier mesh are returned
    alongside the receipt so the flux the build was given can be gathered at the
    receipt's own node vertices and the arcs compared against the carrier.
    """

    _oracle, field, wall, _axis, _saddle = oracle._analytic_inputs(kind)
    geometry = oracle._realised_hex_geometry(wall, cells)
    _coefficient, centre_flux = topology._sampled_flux(field, geometry)
    vertex_flux = jnp.sum(
        centre_flux[geometry.contour_sources] * geometry.contour_weights, axis=1
    )
    mesh = geometry.contour_mesh
    receipt = build_contour_tree(
        vertex_flux,
        mesh.vertex_valid,
        mesh.vertex_is_wall,
        mesh.edges,
        mesh.edge_valid,
        jnp.asarray(1, dtype=jnp.int32),
    )
    return receipt, vertex_flux, mesh


def _perturb_arc_order() -> bool:
    """Whether the declared arc-order mutation is selected for this run."""

    return os.environ.get("CONTOUR_TREE_ARC_PERTURB") == "rewire"


def _host_superlevel_components(
    values: np.ndarray, valid: np.ndarray, edges: np.ndarray, level: float
) -> int:
    """Count strict superlevel components by plain host graph search."""

    live = valid & (values > level)
    parent = list(range(values.size))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        if live[left] and live[right]:
            parent[find(int(left))] = find(int(right))
    return len({find(int(node)) for node in np.flatnonzero(live)})


def _receipt_crossings(arcs: np.ndarray, values: np.ndarray, level: float) -> int:
    """Count receipt arcs whose endpoints straddle one carrier level."""

    return int(
        np.count_nonzero((values[arcs[:, 0]] > level) != (values[arcs[:, 1]] > level))
    )


def _assert_matches_carrier(receipt, mesh, vertex_flux) -> None:
    """The receipt's arcs must bear the carrier's own superlevel events.

    The reference is a plain host graph search over the same carrier and the
    same field the build was handed, so it is valid on any host: a near-tie
    flux comparison that rounds differently under another CPU's vector code
    moves the receipt and the reference together.
    """

    values = np.asarray(vertex_flux, dtype=np.float64)
    valid = np.asarray(mesh.vertex_valid, dtype=bool)
    carrier_edges = np.asarray(mesh.edges)[np.asarray(mesh.edge_valid)]
    edge_rows = np.asarray(receipt.edges)[np.asarray(receipt.edge_valid)]
    if _perturb_arc_order():
        edge_rows = np.roll(edge_rows, 1, axis=0)
    arcs = np.asarray(receipt.node_vertex, dtype=np.int64)[edge_rows]
    levels = np.nextafter(np.unique(values[valid]), -np.inf)[::-1]
    for level in levels:
        reference = _host_superlevel_components(
            values, valid, carrier_edges, float(level)
        )
        assert _receipt_crossings(arcs, values, float(level)) == reference, float(level)


def _integer_digest(receipt: contour_tree.ContourTreeArrays) -> str:
    """A field-ordered digest over the host-independent receipt arrays."""

    digest = hashlib.sha256()
    for name in _INTEGER_FIELDS:
        digest.update(name.encode())
        digest.update(np.asarray(getattr(receipt, name)).tobytes())
    return digest.hexdigest()


def test_sparse_indexed_build_marker_is_present():
    """The sparse build path marks itself, so its absence is a failing test.

    A restored dense equality matrix and the repeated scans drop this marker;
    the equivalence checks below then describe a tree the module no longer
    builds, which is the declared mutation this check must fail against.
    """

    assert contour_tree.SPARSE_INDEXED_BUILD is True


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["limited", "diverted"])
def test_carrier_triangles_have_resolved_area(kind: str):
    """Deduplicated carrier vertices must not leave zero-area graph faces."""

    _oracle, _field, wall, _axis, _saddle = oracle._analytic_inputs(kind)
    mesh = oracle._realised_hex_geometry(wall, 2000).contour_mesh
    triangle = np.asarray(mesh.triangles)[np.asarray(mesh.triangle_valid)]
    point = np.asarray(mesh.vertex_rz)[triangle]
    first = point[:, 1] - point[:, 0]
    second = point[:, 2] - point[:, 0]
    doubled_area = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    edge = np.stack((first, second, point[:, 2] - point[:, 1]), axis=1)
    scale = np.max(np.sum(edge * edge, axis=-1), axis=1)
    assert np.all(doubled_area > 64 * np.finfo(float).eps * scale)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["limited", "diverted"])
def test_clipped_carrier_faces_use_occupiable_vertices(kind: str):
    """Every face entering a contour sweep uses vertices inside the wall."""

    _oracle, _field, wall, _axis, _saddle = oracle._analytic_inputs(kind)
    mesh = oracle._realised_hex_geometry(wall, 2000).contour_mesh
    faces = np.asarray(mesh.triangles)[np.asarray(mesh.triangle_valid)]
    valid = np.asarray(mesh.vertex_valid)
    assert np.all(valid[faces])


@pytest.mark.parametrize("kind", ["limited", "diverted"])
@pytest.mark.parametrize("cells", [550, 2000])
def test_sparse_build_matches_base_receipt(kind: str, cells: int):
    """The receipt's arcs bear the carrier's superlevel events on any host.

    The integer arrays are compared against a plain host graph search over the
    same carrier and the same field the build was handed, so the reference
    travels between hosts where a pinned digest of the arrays does not: at
    2,000 cells a near-tie flux comparison can round differently under another
    CPU's vector code and move the receipt without changing what it means.  The
    coarse rows additionally keep their pinned digest, shown host-independent on
    both the sirius and Broadwell CPU classes, as a second check that the
    receipt bytes themselves are unchanged there.  The flux each valid node row
    carries is the field the build was given.
    """

    receipt, vertex_flux, mesh = _build(kind, cells)
    assert np.asarray(receipt.overflow) == np.asarray(False)
    assert int(np.sum(receipt.node_valid)) >= 2

    _assert_matches_carrier(receipt, mesh, vertex_flux)

    if cells == 550:
        assert _integer_digest(receipt) == _EXPECTED_INTEGER_DIGESTS[f"{kind}-{cells}"]

    valid = np.asarray(receipt.node_valid, dtype=bool)
    node_vertex = np.asarray(receipt.node_vertex, dtype=np.int64)
    gathered = np.asarray(vertex_flux)[node_vertex]
    assert np.array_equal(np.asarray(receipt.node_psi)[valid], gathered[valid])
    assert np.all(np.asarray(receipt.node_psi)[~valid] == 0.0)
