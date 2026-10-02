"""Guards pinning the explanatory figure to the receipt it is drawn from.

Four things are asserted, each against the artifact or the drawn panel rather
than against ``build_state`` alone, so a stale, missing or divergent artifact
fails the guard:

* the drawn node set equals the receipt's valid nodes in count and carrier
  vertices, rebuilt independently from the fixture;
* the committed receipt JSON matches that drawn set and keeps the contour-tree
  node-minus-edge invariant, and the committed PNG and SVG exist;
* the graph panel's signed-flux axis excludes the virtual outside node's
  ``-1.007`` mesh-padding sentinel, drawing it at its wall-contact join level
  instead;
* the poloidal panel draws no lines between nodes, since the tree's adjacency
  is topological rather than geometric.

The two figure guards are falsified by hooks that restore the pre-repair
behaviour: ``CONTOUR_TREE_FIGURE_SENTINEL`` re-plots the sentinel and
``CONTOUR_TREE_FIGURE_POLOIDAL_EDGES`` restores the spatial edges.  Pointing
``CONTOUR_TREE_FIGURE_DIR`` at a stale receipt falsifies the receipt guard.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from benchmarks import contour_tree_explanatory_figure as figure
from nova.equilibrium.contour_tree import build_contour_tree
from nova.jax.config import configure_dtypes


configure_dtypes()


def _receipt():
    """Rebuild the certificate tree from the fixture, independently of the figure."""

    fixture = figure._fixture()
    mesh = fixture.mesh
    tree = build_contour_tree(
        mesh.vertex_psi,
        mesh.vertex_valid,
        mesh.vertex_is_wall,
        mesh.edges,
        mesh.edge_valid,
        jnp.asarray(1, dtype=jnp.int32),
    )
    assert not bool(tree.overflow)
    return mesh, tree


def _drawn_carrier_vertices(state):
    return sorted(int(node.carrier_vertex) for node in state.nodes)


def test_figure_draws_exactly_the_receipts_valid_nodes():
    """Count and carrier vertices of the drawn set equal the receipt's valid nodes."""

    _, tree = _receipt()
    state = figure.build_state()
    valid = np.asarray(tree.node_valid)
    node_vertex = np.asarray(tree.node_vertex)
    expected = sorted(int(node_vertex[row]) for row in np.flatnonzero(valid))
    assert _drawn_carrier_vertices(state) == expected


def test_committed_receipt_matches_the_computed_tree():
    """The committed receipt is present and agrees with the tree the figure draws."""

    receipt_path = figure.FIGURE_STEM.with_suffix(".json")
    assert receipt_path.is_file(), "the committed receipt is missing"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    state = figure.build_state()
    assert payload["fixture"] == state.fixture
    assert payload["node_count"] == len(state.nodes)
    assert payload["edge_count"] == len(state.edges)
    assert payload["node_count_minus_edge_count"] == 1
    assert payload["drawn_carrier_vertices"] == _drawn_carrier_vertices(state)


def test_committed_figure_files_exist():
    """The rendered PNG and SVG are present and non-empty."""

    for suffix in (".png", ".svg"):
        path = figure.FIGURE_STEM.with_suffix(suffix)
        assert path.is_file(), "the committed figure %s is missing" % path.name
        assert path.stat().st_size > 1024, "the committed figure %s is empty" % path.name


def _render():
    state = figure.build_state()
    figure_ = plt.figure(figsize=(14.0, 8.0))
    poloidal_panel = figure._draw_poloidal(figure_, state)
    graph_panel = figure._draw_graph(figure_, state)
    return state, poloidal_panel, graph_panel


def test_graph_omits_the_virtual_outside_sentinel():
    """The graph's flux axis spans physical levels, not the outside sentinel."""

    state, _, graph_panel = _render()
    outside = figure._by_row(state, state.outside_row)
    assert outside.psi_wb < -0.5  # the padded carrier emits a sentinel
    assert abs(outside.drawn_psi_wb) < 0.02  # the plotted level is a real flux
    low, high = graph_panel.get_ylim()
    assert low > -0.05, "the sentinel entered the signed-flux axis"


def test_poloidal_panel_draws_no_lines_between_nodes():
    """Adjacency is carried by the graph, never by a spatial line on the panel."""

    _, poloidal_panel, _ = _render()
    dashed = (0.0, (1.0, 1.0))
    offenders = [
        line
        for line in poloidal_panel.lines
        if line.get_linestyle() in (dashed, "--", "dashed")
    ]
    assert offenders == [], "the panel drew topological adjacency as a spatial line"