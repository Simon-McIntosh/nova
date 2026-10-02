"""Render the contour-tree explanatory figure from the computed tree.

One persisted diverted single-null Solov'ev certificate state is drawn so a
reader can see what the topology authority is built from, rather than read it
described.  Unlike an earlier hand-placed version, every node and edge in the
figure is the one ``nova.equilibrium.contour_tree.build_contour_tree`` actually
emits for the state's own piecewise-linear hex carrier:

* the state's own poloidal flux as unfilled line contours on stated raw-flux
  levels in webers, sampled from the same piecewise-linear field the tree is
  built on and masked to the vessel interior;
* the wall, drawn as its own unit;
* the two analytic nulls via the media ``draw_nulls`` vocabulary -- the magnetic
  axis as an O-point and the axis's own saddle as the admitted X-point;
* every critical node the tree emits, overlaid at its carrier vertex position
  and marked by critical type, with the private-region wall maximum singled out
  as its own node.  No line is drawn between nodes on this panel: the tree's
  adjacency is a topological relation, not a geometric path, so a straight line
  between two critical points would read as a field line or a sightline;
* beside the panel, the tree itself as a graph against signed flux level, which
  is where the adjacency is carried: the critical type by marker and the edges
  drawn as arcs.  The virtual outside node is drawn at the level where the
  outside region joins the tree, so its mesh-padding sentinel never enters the
  signed-flux axis.

The field is piecewise-linear, so beyond the analytic axis and X-point the tree
also emits low-persistence maxima, minima and saddles: the discretisation's own
critical points.  They are drawn rather than hidden, and the caption names them
as piecewise-linear extrema that the geometry and selection sections of the plan
must absorb by persistence and primary selection.

``psi`` throughout is the state's raw poloidal flux in webers, the same
convention the certificate receipt stores.  Sigma is the flux sign under which
``sigma * psi`` makes the current-carrying axis the in-vessel maximum; for this
state it is +1, because the axis flux exceeds the boundary flux.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Bbox
from matplotlib.tri import LinearTriInterpolator, Triangulation
from scipy.interpolate import griddata

from benchmarks.contour_tree_brute_force import PART_ROOT, certificate_rung_fixtures
from nova.equilibrium.contour_tree import build_contour_tree
from nova.equilibrium.wall_mask import vessel_unit
from nova.jax.config import configure_dtypes
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.sources.frame import inside_wall_units

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR_ENV = "CONTOUR_TREE_FIGURE_DIR"
PLOT_SENTINEL_ENV = "CONTOUR_TREE_FIGURE_SENTINEL"
POLOIDAL_EDGES_ENV = "CONTOUR_TREE_FIGURE_POLOIDAL_EDGES"
FIXED_LABELS_ENV = "CONTOUR_TREE_FIGURE_FIXED_LABELS"
FIGURE_ROOT = Path(
    os.environ.get(
        FIGURE_DIR_ENV,
        ROOT / "docs/figures/contour-tree-topology-authority/explanatory",
    )
)
FIGURE_STEM = FIGURE_ROOT / "contour-tree-computed"
FIXTURE = "diverted-single-null-340-cells"

SIGMA = 1
MINIMUM = 0
SADDLE = 1
MAXIMUM = 2
CRITICAL_NAME = {MINIMUM: "minimum", SADDLE: "saddle", MAXIMUM: "maximum"}
CRITICAL_MARKER = {MINIMUM: "v", SADDLE: "s", MAXIMUM: "^"}

CONTOUR_RADIUS_SAMPLES = 480
CONTOUR_HEIGHT_SAMPLES = 620
CONTOUR_LEVEL_COUNT = 14

LABEL_FONTSIZE = DEFAULT_INK.label_fontsize * 1.4
NODE_COLOR = "#111111"
ANALYTIC_COLOR = DEFAULT_INK.separatrix_color
EDGE_COLOR = "#111111"
OUTSIDE_COLOR = "#7a3b00"
PAD = 0.05

# Candidate anchor offsets, in points, for a graph-panel direct label.  The
# first that clears every placed label box and every node marker is used, so
# two nodes drawn at the same flux level cannot stack their labels.  Offsets
# start far enough out to clear the label bbox padding and the marker glyph.
GRAPH_LABEL_OFFSETS = (
    (11.0, 7.0, "left"),
    (11.0, -17.0, "left"),
    (-11.0, 7.0, "right"),
    (-11.0, -17.0, "right"),
    (11.0, 22.0, "left"),
    (11.0, -32.0, "left"),
    (-11.0, 22.0, "right"),
    (-11.0, -32.0, "right"),
)

INK = DEFAULT_INK.variant(label_fontsize=LABEL_FONTSIZE)


@dataclass(frozen=True)
class Node:
    row: int
    critical_type: int
    kind: str
    radius: float
    height: float
    psi_wb: float
    drawn_psi_wb: float
    carrier_vertex: int
    role: str
    join_level_wb: float | None = None


@dataclass(frozen=True)
class Edge:
    first: int
    second: int


@dataclass(frozen=True)
class State:
    fixture: str
    sigma: int
    wall: np.ndarray
    wall_unit: Any
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    levels: np.ndarray
    axis_row: int
    xpoint_row: int
    wall_row: int
    outside_row: int


def _fixture():
    fixtures = certificate_rung_fixtures()
    selected = [item for item in fixtures if item.name.endswith(FIXTURE)]
    if len(selected) != 1:
        raise RuntimeError(
            "expected one %s fixture, found %d" % (FIXTURE, len(selected))
        )
    return selected[0]


def _rung_label(filename: str) -> int:
    for rung, marker in ((340, "cells-300"), (550, "cells-500"), (1074, "cells-1000")):
        if marker in filename:
            return rung
    raise RuntimeError("unrecognised certificate part name: " + filename)


def _wall(fixture) -> tuple[Any, np.ndarray]:
    for path in sorted(PART_ROOT.glob("*production-route-cells-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        case = payload.get("case")
        if not case:
            continue
        name = "%s-%d-cells" % (case, _rung_label(path.name))
        if name != fixture.name:
            continue
        polyline = np.asarray(
            payload["render_data"]["wall_units_rz_m"][0], dtype=np.float64
        )
        return vessel_unit(polyline[:, 0], polyline[:, 1], name=str(case)), polyline
    raise RuntimeError("no persisted wall for " + fixture.name)


def _nodes(mesh, tree) -> tuple[Node, ...]:
    """Return the tree's valid node rows, labelled by analytic role.

    The roles are read off the tree and the carrier, never placed by hand: the
    magnetic axis is the greatest valid node of sigma * psi; the admitted
    X-point is the saddle the tree attaches directly to that axis; the
    private-region wall maximum is the tree's own node at the highest-flux wall
    vertex; the outside node is the virtual slot, whose carrier vertex is mesh
    padding and so carries no position of its own.
    """

    carrier = np.asarray(mesh.vertex_rz)
    psi = np.asarray(mesh.vertex_psi)
    valid = np.asarray(tree.node_valid)
    node_vertex = np.asarray(tree.node_vertex)
    carrier_valid = np.asarray(mesh.vertex_valid)
    is_wall = np.asarray(mesh.vertex_is_wall)
    critical_type = np.asarray(tree.critical_type)

    rows = [int(row) for row in np.flatnonzero(valid)]
    if not rows:
        raise RuntimeError("the computed tree emitted no valid nodes")
    edges = np.asarray(tree.edges)[np.asarray(tree.edge_valid)]
    neighbours: dict[int, set[int]] = {}
    for row in rows:
        neighbours[row] = set()
    for left, right in edges:
        neighbours[int(left)].add(int(right))
        neighbours[int(right)].add(int(left))

    def psi_of(row: int) -> float:
        return float(psi[int(node_vertex[row])])

    axis_row = max(rows, key=psi_of)
    attached = []
    for row in rows:
        if int(critical_type[row]) == SADDLE and axis_row in neighbours[row]:
            attached.append(row)
    if not attached:
        raise RuntimeError("no saddle attaches directly to the computed axis")
    xpoint_row = max(attached, key=psi_of)

    wall_rows = [row for row in rows if bool(is_wall[int(node_vertex[row])])]
    wall_row = max(wall_rows, key=psi_of) if wall_rows else -1
    join_level = psi_of(wall_row) if wall_row >= 0 else None
    outside_rows = [
        row for row in rows if not bool(carrier_valid[int(node_vertex[row])])
    ]
    if len(outside_rows) != 1:
        raise RuntimeError(
            "expected one positionless outside node, found %d" % len(outside_rows)
        )
    outside_row = outside_rows[0]

    nodes = []
    for row in rows:
        vertex = int(node_vertex[row])
        radius = float(carrier[vertex, 0])
        height = float(carrier[vertex, 1])
        if row == outside_row and wall_row >= 0:
            contact = carrier[int(node_vertex[wall_row])]
            radius = float(contact[0])
            height = float(contact[1])
        role = "piecewise-linear critical point"
        if row == axis_row:
            role = "magnetic axis"
        elif row == xpoint_row:
            role = "admitted X-point"
        elif row == wall_row:
            role = "private-region wall maximum"
        elif row == outside_row:
            role = "outside node"
        if row == outside_row:
            # The virtual slot is typed outside the DD critical-point
            # vocabulary, so it carries no name from CRITICAL_NAME.
            kind = "outside (virtual)"
        else:
            kind = CRITICAL_NAME[int(critical_type[row])]
        raw_psi = psi_of(row)
        drawn_psi = raw_psi
        join_wb = None
        if row == outside_row:
            # The virtual slot's carrier vertex is mesh padding, so its stored
            # raw psi is a sentinel rather than a flux.  Draw it at the level
            # where the outside region actually joins the tree -- its wall
            # contact -- so the sentinel never enters the signed-flux axis.
            drawn_psi = join_level if join_level is not None else raw_psi
            join_wb = join_level
        nodes.append(
            Node(
                row=row,
                critical_type=int(critical_type[row]),
                kind=kind,
                radius=radius,
                height=height,
                psi_wb=raw_psi,
                drawn_psi_wb=drawn_psi,
                carrier_vertex=vertex,
                role=role,
                join_level_wb=join_wb,
            )
        )
    return tuple(nodes)


def _levels(mesh) -> np.ndarray:
    psi = np.asarray(mesh.vertex_psi)[np.asarray(mesh.vertex_valid)]
    levels = np.linspace(float(np.min(psi)), float(np.max(psi)), CONTOUR_LEVEL_COUNT)
    levels[int(np.argmin(np.abs(levels)))] = 0.0
    return np.unique(levels)


def _masked_field(mesh, wall_units, radius, height) -> np.ndarray:
    carrier = np.asarray(mesh.vertex_rz)
    psi = np.asarray(mesh.vertex_psi)
    live = np.asarray(mesh.vertex_valid)
    points = carrier[live]
    values = psi[live]
    radius_grid, height_grid = np.meshgrid(radius, height)
    field = griddata(points, values, (radius_grid, height_grid), method="linear")
    inside = inside_wall_units(
        np.column_stack((radius_grid.ravel(), height_grid.ravel())),
        wall_units,
    ).reshape(radius_grid.shape)
    return np.where(inside, field, np.nan)


def _format_flux(value: float) -> str:
    if abs(value) < 1.0e-12:
        return "+0.000e+00"
    return f"{value:+.3e}"


def _by_row(state, row: int) -> Node:
    for node in state.nodes:
        if node.row == row:
            return node
    raise KeyError(row)


def _role_row(nodes, role: str) -> int:
    for node in nodes:
        if node.row < 0:
            continue
        if node.role == role:
            return node.row
    return -1


def _poloidal_extent(state: State):
    r_min = float(state.wall[:, 0].min()) - PAD
    r_max = float(state.wall[:, 0].max()) + PAD
    z_min = float(state.wall[:, 1].min()) - PAD
    z_max = float(state.wall[:, 1].max()) + PAD
    return r_min, r_max, z_min, z_max


def _node_mark(node):
    if node.role == "outside node":
        return "D", OUTSIDE_COLOR, DEFAULT_INK.xpoint_markersize
    return CRITICAL_MARKER[node.critical_type], NODE_COLOR, 5.5


def _draw_poloidal(figure, state: State):
    wall_units = (state.wall_unit,)
    r_min, r_max, z_min, z_max = _poloidal_extent(state)
    radius = np.linspace(r_min, r_max, CONTOUR_RADIUS_SAMPLES)
    height = np.linspace(z_min, z_max, CONTOUR_HEIGHT_SAMPLES)
    field = _masked_field(_fixture().mesh, wall_units, radius, height)
    panel = figure.add_axes((0.02, 0.02, 0.46, 0.96))
    poloidal_axes(panel)
    poloidal.draw_flux_contours(panel, radius, height, field, state.levels, style=INK)
    poloidal.draw_wall(panel, units=wall_units, style=INK)
    # The tree's adjacency is a topological relation, not a geometric path: a
    # straight line between two critical points would read as a field line or a
    # sightline.  It is carried only by the graph panel.  The hook restores the
    # spatial edges for the negative control that shows the guard fires.
    if os.environ.get(POLOIDAL_EDGES_ENV) == "1":
        for edge in state.edges:
            first = _by_row(state, edge.first)
            second = _by_row(state, edge.second)
            panel.plot(
                [first.radius, second.radius],
                [first.height, second.height],
                color=EDGE_COLOR,
                linewidth=1.0,
                linestyle=(0.0, (1.0, 1.0)),
                zorder=DEFAULT_INK.zorder_separatrix,
            )
    for node in state.nodes:
        if node.role == "magnetic axis" or node.role == "admitted X-point":
            continue
        marker, color, size = _node_mark(node)
        panel.plot(
            node.radius,
            node.height,
            marker=marker,
            markersize=size,
            color=color,
            markerfacecolor="none",
            markeredgewidth=1.2,
            linestyle="none",
            zorder=DEFAULT_INK.zorder_markers,
        )
    axis = _by_row(state, state.axis_row)
    xpoint = _by_row(state, state.xpoint_row)
    wallmax = _by_row(state, state.wall_row)
    poloidal.draw_nulls(
        panel,
        magnetic_axis=[axis.radius, axis.height],
        x_points=np.array([[xpoint.radius, xpoint.height]]),
        contain=wall_units,
        style=INK,
    )
    label_axis = "magnetic axis\npsi = " + _format_flux(axis.psi_wb) + " Wb"
    label_x = "admitted X-point\npsi = " + _format_flux(xpoint.psi_wb) + " Wb"
    entries = [
        (axis, -0.05, 0.06, "right", label_axis, ANALYTIC_COLOR),
        (xpoint, 0.05, 0.06, "left", label_x, ANALYTIC_COLOR),
        (wallmax, 0.06, -0.05, "left", "private-region\nwall maximum", NODE_COLOR),
    ]
    for node, dr, dz, align, label, color in entries:
        panel.annotate(
            label,
            xy=(node.radius + dr, node.height + dz),
            fontsize=LABEL_FONTSIZE,
            ha=align,
            va="center",
            color=color,
            bbox=DEFAULT_INK.label_bbox,
            zorder=DEFAULT_INK.zorder_label,
        )
    return panel


def _plotted_level(node):
    """Return the signed-flux level at which a node is drawn.

    The virtual outside node's stored raw psi is mesh-padding sentinel, not a
    flux, so the node is drawn at the level where the outside region joins the
    tree, its wall contact.  The hook restores the sentinel for the negative
    control.
    """

    if os.environ.get(PLOT_SENTINEL_ENV) == "1":
        return node.psi_wb
    return node.drawn_psi_wb


def _annotation_display_box(annotation, renderer) -> Bbox:
    """Display-space bounding box of a label, text and its padding together."""

    annotation.update_positions(renderer)
    box = annotation.get_window_extent(renderer=renderer)
    pad = float(DEFAULT_INK.label_bbox.get("pad", 0.0)) * annotation.figure.dpi / 72.0
    return Bbox.from_extents(box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad)


def _marker_display_box(figure, panel, x_value, y_value, size) -> Bbox:
    """Display-space bounding box of a marker of ``size`` points."""

    x_display, y_display = panel.transData.transform((x_value, y_value))
    half = 0.5 * size * figure.dpi / 72.0
    return Bbox.from_extents(
        x_display - half, y_display - half, x_display + half, y_display + half
    )


def _place_graph_labels(figure, panel, state, rank, names) -> None:
    """Place each named graph label at the first offset that collides with nothing.

    Candidate offsets are tried in order; a candidate is kept only when its
    display box clears every already-placed label box and every node marker.
    Two nodes drawn at the same flux level therefore cannot stack their labels,
    and no label can be hidden behind a marker.
    """

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    if os.environ.get(FIXED_LABELS_ENV) == "1":
        # The pre-repair hook: every label at one fixed offset, so nodes drawn
        # at the same flux level stack their labels.  It exists so the guard
        # that requires a collision-free placement can be shown to fail.
        for row, label in names:
            node = _by_row(state, row)
            panel.annotate(
                label,
                xy=(rank[node.row], _plotted_level(node)),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=LABEL_FONTSIZE,
                color=NODE_COLOR,
                bbox=DEFAULT_INK.label_bbox,
                zorder=DEFAULT_INK.zorder_label,
            )
        return
    occupied = [
        _marker_display_box(
            figure, panel, rank[node.row], _plotted_level(node), _node_mark(node)[2]
        )
        for node in state.nodes
    ]
    for row, label in names:
        node = _by_row(state, row)
        point = (rank[node.row], _plotted_level(node))
        for offset_x, offset_y, align in GRAPH_LABEL_OFFSETS:
            annotation = panel.annotate(
                label,
                xy=point,
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                fontsize=LABEL_FONTSIZE,
                color=NODE_COLOR,
                bbox=DEFAULT_INK.label_bbox,
                zorder=DEFAULT_INK.zorder_label,
                ha=align,
                va="center",
            )
            box = _annotation_display_box(annotation, renderer)
            if any(box.overlaps(other) for other in occupied):
                annotation.remove()
                continue
            occupied.append(box)
            break
        else:
            raise RuntimeError(
                "no collision-free placement found for the %r graph label" % label
            )


def _draw_graph(figure, state: State):
    ordered = sorted(state.nodes, key=lambda node: (-_plotted_level(node), node.row))
    rank = {}
    for position, node in enumerate(ordered):
        rank[node.row] = float(position)
    panel = figure.add_axes((0.56, 0.11, 0.40, 0.82))
    for side in ("top", "right"):
        panel.spines[side].set_visible(False)
    panel.grid(False)
    panel.tick_params(labelsize=LABEL_FONTSIZE)
    panel.set_xlabel("node rank (descending sigma*psi)", fontsize=LABEL_FONTSIZE)
    panel.set_ylabel("signed flux sigma*psi [Wb]", fontsize=LABEL_FONTSIZE)
    levels = [_plotted_level(node) for node in state.nodes]
    span = max(levels) - min(levels)
    pad = 0.10 * span if span > 0.0 else 1.0e-3
    x_limits = (-0.5, max(0.5, len(state.nodes) - 0.5))
    y_limits = (min(levels) - pad, max(levels) + pad)
    panel.set_xlim(*x_limits)
    panel.set_ylim(*y_limits)

    def to_fraction(x_value, y_value):
        x_fraction = (x_value - x_limits[0]) / (x_limits[1] - x_limits[0])
        y_fraction = (y_value - y_limits[0]) / (y_limits[1] - y_limits[0])
        return x_fraction, y_fraction

    for edge in state.edges:
        first = _by_row(state, edge.first)
        second = _by_row(state, edge.second)
        patch = FancyArrowPatch(
            to_fraction(rank[first.row], _plotted_level(first)),
            to_fraction(rank[second.row], _plotted_level(second)),
            transform=panel.transAxes,
            arrowstyle="-",
            connectionstyle="arc3,rad=0.18",
            color=EDGE_COLOR,
            linewidth=1.0,
            zorder=DEFAULT_INK.zorder_flux,
        )
        panel.add_patch(patch)
    for node in state.nodes:
        marker, color, size = _node_mark(node)
        panel.plot(
            rank[node.row],
            _plotted_level(node),
            marker=marker,
            markersize=size,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.3,
            linestyle="none",
            zorder=DEFAULT_INK.zorder_markers,
        )
    names = [
        (state.axis_row, "axis"),
        (state.xpoint_row, "X-point"),
        (state.wall_row, "wall maximum"),
        (state.outside_row, "outside (wall contact)"),
    ]
    _place_graph_labels(figure, panel, state, rank, names)
    return panel


def _caption(state: State) -> str:
    mins = 0
    saddles = 0
    maxima = 0
    outside = 0
    for node in state.nodes:
        if node.critical_type == MINIMUM:
            mins += 1
        elif node.critical_type == SADDLE:
            saddles += 1
        elif node.critical_type == MAXIMUM:
            maxima += 1
        else:
            outside += 1
    extra_extrema = (maxima - 1) + mins
    extra_saddles = saddles - 1
    parts = [
        "Diverted single-null Solovev certificate at its persisted 340-cell rung",
        "(fixture " + state.fixture + ").",
        "The computed contour tree carries %d nodes and %d edges"
        % (len(state.nodes), len(state.edges)),
        "so the node count exceeds the edge count by %d."
        % (len(state.nodes) - len(state.edges)),
        "Two nodes are the analytic nulls: the magnetic-axis maximum and the",
        "admitted X-point saddle. The other %d are piecewise-linear critical"
        % (len(state.nodes) - 2 - outside),
        "points beyond them: %d piecewise-linear extrema" % extra_extrema,
        "(%d maximum, %d minima) and %d further saddles, plus the positionless"
        % (maxima - 1, mins, extra_saddles),
        "virtual outside node the padded carrier emits, typed outside the critical-point",
        "vocabulary. The private-region wall maximum is the tree's own node at the",
        "highest-flux wall vertex. These low-persistence piecewise-linear",
        "extrema are drawn rather than hidden, and the critical-point geometry and",
        "selection sections (§3 and §4) must absorb them by persistence and primary selection.",
        "The poloidal panel draws no lines between nodes: the tree's adjacency is a",
        "topological relation, not a geometric path, so it is carried only by the",
        "graph panel, where every node is placed at its signed flux level and the",
        "edges are drawn as arcs. The virtual outside node is drawn at the level",
        "where the outside region joins the tree, its wall contact, not at its",
        "mesh-padding sentinel, which is omitted from the signed-flux axis.",
    ]
    return " ".join(parts)


def build_state() -> State:
    fixture = _fixture()
    mesh = fixture.mesh
    tree = build_contour_tree(
        mesh.vertex_psi,
        mesh.vertex_valid,
        mesh.vertex_is_wall,
        mesh.edges,
        mesh.edge_valid,
        jnp.asarray(SIGMA, dtype=jnp.int32),
    )
    if bool(tree.overflow):
        raise RuntimeError("the tree refused this certificate carrier")
    wall_unit, wall = _wall(fixture)
    nodes = list(_nodes(mesh, tree))
    drop = os.environ.get("CONTOUR_TREE_FIGURE_DROP_NODE")
    if drop is not None:
        dropped = int(drop)
        kept = [node for node in nodes if node.row != dropped]
        if len(kept) == len(nodes):
            raise RuntimeError("the mutation did not drop node row %d" % dropped)
        nodes = kept
    rows = set(node.row for node in nodes)
    edges = []
    for left, right in np.asarray(tree.edges)[np.asarray(tree.edge_valid)]:
        if int(left) in rows and int(right) in rows:
            edges.append(Edge(int(left), int(right)))
    return State(
        fixture=fixture.name,
        sigma=SIGMA,
        wall=wall,
        wall_unit=wall_unit,
        nodes=tuple(nodes),
        edges=tuple(edges),
        levels=_levels(mesh),
        axis_row=_role_row(nodes, "magnetic axis"),
        xpoint_row=_role_row(nodes, "admitted X-point"),
        wall_row=_role_row(nodes, "private-region wall maximum"),
        outside_row=_role_row(nodes, "outside node"),
    )


def _svg_sha256(path: Path) -> str:
    """Digest of an SVG with its only volatile record, the render timestamp, removed.

    The receipt carries this digest so a committed figure can be compared to
    the receipt that describes it; a figure left from another revision, or a
    receipt regenerated without its figure, fails the comparison.
    """

    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<dc:date>.*?</dc:date>", "", text, flags=re.DOTALL)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURE_STEM)
    arguments = parser.parse_args()
    configure_dtypes()
    mpl.rcParams["font.size"] = LABEL_FONTSIZE
    state = build_state()
    caption = _caption(state)
    figure = plt.figure(figsize=(14.0, 8.0), dpi=DEFAULT_INK.figure_dpi)
    _draw_poloidal(figure, state)
    _draw_graph(figure, state)
    figure_path = arguments.output.with_suffix(".png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=DEFAULT_INK.figure_dpi)
    figure.savefig(figure_path.with_suffix(".svg"))
    receipt = {
        "schema": "nova.contour-tree-explanatory-figure",
        "version": 2,
        "fixture": state.fixture,
        "sigma": state.sigma,
        "flux_units": "Wb",
        "caption": caption,
        "node_count": len(state.nodes),
        "edge_count": len(state.edges),
        "node_count_minus_edge_count": len(state.nodes) - len(state.edges),
        "drawn_carrier_vertices": sorted(node.carrier_vertex for node in state.nodes),
        "nodes": [asdict(node) for node in state.nodes],
        "edges": [asdict(edge) for edge in state.edges],
        "contour_levels_wb": [float(level) for level in state.levels],
        "wall_points": int(state.wall.shape[0]),
        "figure": str(figure_path.relative_to(ROOT)),
        "svg_sha256": _svg_sha256(figure_path.with_suffix(".svg")),
    }
    receipt_path = arguments.output.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(caption)
    print("nodes=%d edges=%d" % (receipt["node_count"], receipt["edge_count"]))


if __name__ == "__main__":
    main()
