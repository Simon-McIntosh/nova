"""Render the contour-tree explanatory figure for the topology authority.

One diverted single-null Solovev reference is drawn so a reader can see what the
topology authority is built from, rather than read it described:

* the analytic poloidal flux as unfilled line contours on stated raw-flux
  levels in webers, masked to the vessel interior;
* the wall, drawn as its own unit;
* both kinds of null via the media ``draw_nulls`` vocabulary -- the magnetic
  axis as an O-point and the admitted X-point as a saddle;
* the contour tree overlaid: a node at the magnetic axis (the maximum of
  sigma*psi), a node at the X-point (the saddle) and a virtual outside node
  attached at the first wall contact.  The outside node carries no flux of its
  own; the level at which it joins the tree is labelled on its edge, together
  with whether that contact lies in the private region below the X-point, since
  a private-region contact can never be a limiter.  Each interior node is
  labelled directly with its critical type and flux.

The tree is not asserted.  Its nodes are located by solving the reference's
stationarity conditions and classified by the Hessian signature, and its shape
is checked against an independent brute-force component count of the superlevel
sets at levels bracketing every node value.  The outside node is valued below
every interior level by construction: descending from the axis the outside
region first joins the tree where the superlevel set first meets the body.

``psi`` throughout is the raw poloidal flux in webers, evaluated at the
fixture's own value (the same convention the certificate fixture reads as its
state flux).  ``sigma`` is the flux sign under which the magnetic axis is the
in-vessel maximum; every level comparison is a comparison of ``sigma * psi``.

The receipt beside the figure lists every node (critical type, position, flux)
and every edge, and records that the node count minus the edge count is one.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
from scipy import ndimage
from scipy.optimize import root

from nova.equilibrium.analytic_single_null import CerfonFreidbergSingleNull
from nova.jax.config import configure_dtypes
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK
from nova.media.layout import poloidal_view
from nova.media.sources.frame import inside_wall_units
from scripts.analytic_oracle_fixtures import measure as oracle_fixture

ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = ROOT / "docs/figures/contour-tree-topology-authority/explanatory"
FIGURE_STEM = FIGURE_ROOT / "contour-tree-single-null"

# DD 4 contour_tree critical_type: 0 minimum, 1 saddle, 2 maximum.
MINIMUM = 0
SADDLE = 1
MAXIMUM = 2
CRITICAL_NAME = {MINIMUM: "minimum", SADDLE: "saddle", MAXIMUM: "maximum"}

WALL_CLEARANCE_FRACTION = 0.35
WALL_POINT_COUNT = 121
SEPARATRIX_POINTS = 1441

CONTOUR_RADIUS_SAMPLES = 520
CONTOUR_HEIGHT_SAMPLES = 700
CHECK_RADIUS_SAMPLES = 620
CHECK_HEIGHT_SAMPLES = 820
CONTOUR_LEVEL_COUNT = 12

# Levels bracketing every node value, as a fraction of the flux span measured
# from the X-point (the separatrix level) toward the axis.  The wall contact
# sits near 0.17 of that span, so the 0.10 level falls between the X-point and
# the contact and the 0.25 level falls above it.
CHECK_FRACTIONS = (-3.0, -1.0, 0.10, 0.25, 0.50, 0.90, 1.50, 2.50)

LABEL_FONTSIZE = DEFAULT_INK.label_fontsize * 1.4
TREE_EDGE_COLOR = "#111111"
OUTSIDE_MARKER = "D"
PAD = 0.05

TREE_INK = DEFAULT_INK.variant(label_fontsize=LABEL_FONTSIZE)


@dataclass(frozen=True)
class Node:
    """One contour-tree node.

    ``psi_wb`` is the node's own raw flux, or null for the virtual outside
    node, which is valued below every interior level and therefore has no flux
    of its own.  The level at which it joins the tree belongs to its edge.
    """

    index: str
    critical_type: int
    kind: str
    radius: float
    height: float
    psi_wb: float | None


@dataclass(frozen=True)
class Edge:
    """One contour-tree edge, the region between two nodes.

    ``join_level_wb`` is the sigma*psi level at which the region between the
    two nodes first meets the wall; it is null for an interior edge.
    """

    first: str
    second: str
    join_level_wb: float | None = None
    contact_region: str | None = None
    limiter: bool | None = None


def _reference() -> CerfonFreidbergSingleNull:
    return CerfonFreidbergSingleNull()


def _wall(exact: CerfonFreidbergSingleNull) -> np.ndarray:
    """Return the wall as the smooth outward offset of the separatrix."""
    return oracle_fixture.offset_wall(
        exact.separatrix(SEPARATRIX_POINTS),
        clearance=WALL_CLEARANCE_FRACTION * exact.minor_radius,
        points=WALL_POINT_COUNT,
    )


def _flux(exact: CerfonFreidbergSingleNull, points: np.ndarray) -> np.ndarray:
    """Return the raw poloidal flux [Wb] at ``(R, Z)`` points."""
    return np.asarray(exact.flux(np.asarray(points, dtype=np.float64)))


def _polish(seed: np.ndarray, exact: CerfonFreidbergSingleNull) -> np.ndarray:
    """Return the stationary point of the flux nearest ``seed``.

    A stationarity solve rather than a Hessian Newton
    step, so a saddle -- whose Hessian is indefinite -- is located too; the
    Hessian is read separately, only for classification.
    """

    def residual(point: np.ndarray) -> np.ndarray:
        return np.asarray(exact.gradient(point[None, :]), dtype=np.float64)[0]

    solved = root(residual, np.asarray(seed, dtype=np.float64), method="hybr")
    if not solved.success:
        raise RuntimeError(f"stationary-point solve failed: {solved.message}")
    return np.asarray(solved.x, dtype=np.float64)


def _classify(point: np.ndarray, exact: CerfonFreidbergSingleNull) -> int:
    """Return the critical type of a stationary point from its Hessian."""
    eigenvalues = np.linalg.eigvalsh(
        np.asarray(exact.hessian(point[None, :]), dtype=np.float64)[0]
    )
    if np.all(eigenvalues < 0.0):
        return MAXIMUM
    if np.all(eigenvalues > 0.0):
        return MINIMUM
    return SADDLE


def _sigma(exact: CerfonFreidbergSingleNull, axis: np.ndarray) -> int:
    """Return the flux sign under which the magnetic axis is a maximum.

    A state's authority read derives sigma from its declared COCOS sign tuple
    and the sign of its plasma current; the analytic reference declares
    neither, so sigma is fixed here by the one property every definition rests
    on -- the axis is the in-vessel maximum of sigma*psi.
    """
    return 1 if _classify(axis, exact) == MAXIMUM else -1


def _wall_contact(
    exact: CerfonFreidbergSingleNull, wall: np.ndarray, sigma: int
) -> tuple[np.ndarray, float]:
    """Return the first wall contact and its sigma*psi level.

    Descending from the axis the superlevel set first meets the wall where the
    wall's own flux is greatest, so the first contact is the wall vertex that
    maximises sigma*psi.
    """
    values = sigma * _flux(exact, wall)
    return np.asarray(wall[int(np.argmax(values))], dtype=np.float64), float(
        np.max(values)
    )


def _inside_mask(
    radius: np.ndarray, height: np.ndarray, wall: np.ndarray
) -> np.ndarray:
    """Return the vessel-interior mask over a (height, radius) mesh."""
    radius_grid, height_grid = np.meshgrid(radius, height)
    points = np.column_stack((radius_grid.ravel(), height_grid.ravel()))
    inside = inside_wall_units(points, wall)
    return np.asarray(inside, dtype=bool).reshape(radius_grid.shape)


def _components(flux: np.ndarray, inside: np.ndarray, level: float) -> tuple[int, int]:
    """Count superlevel-set components with the outside node glued in.

    Returns the component count and how many of them touch the wall.  Every
    wall-touching component is joined to the single virtual outside node, so it
    counts once however many separate wall segments it meets.
    """
    mask = inside & (flux >= level)
    labelled, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if count == 0:
        return 0, 0
    boundary = ndimage.binary_dilation(~inside, structure=np.ones((3, 3)))
    touching = np.unique(labelled[mask & boundary])
    touching = touching[touching > 0]
    glued = count - len(touching) + min(1, len(touching))
    return glued, len(touching)


def _tree(
    exact: CerfonFreidbergSingleNull, wall: np.ndarray
) -> tuple[list[Node], list[Edge], int, float]:
    """Return the contour tree of the reference on the vessel domain.

    The interior nodes are the in-vessel stationary points of the flux,
    located by solving for stationarity from the reference's own declared
    seeds and classified by their Hessian.  The outside node is virtual: it
    carries no flux of its own, is placed at the first wall contact, and the
    level at which it joins the tree is recorded on its edge.
    """
    axis = _polish(exact.magnetic_axis, exact)
    xpoint = _polish(exact.x_point, exact)
    if _classify(axis, exact) != MAXIMUM:
        raise RuntimeError("the polished magnetic axis is not a flux maximum")
    if _classify(xpoint, exact) != SADDLE:
        raise RuntimeError("the polished X-point is not a flux saddle")

    sigma = _sigma(exact, axis)
    contact, contact_level = _wall_contact(exact, wall, sigma)
    axis_psi = float(sigma * _flux(exact, axis[None, :])[0])
    xpoint_psi = float(sigma * _flux(exact, xpoint[None, :])[0])
    if not axis_psi > xpoint_psi:
        raise RuntimeError("the axis does not sit above the X-point in sigma*psi")

    nodes = [
        Node(
            "axis",
            MAXIMUM,
            CRITICAL_NAME[MAXIMUM],
            float(axis[0]),
            float(axis[1]),
            axis_psi,
        ),
        Node(
            "x_point",
            SADDLE,
            CRITICAL_NAME[SADDLE],
            float(xpoint[0]),
            float(xpoint[1]),
            xpoint_psi,
        ),
        Node(
            "outside",
            MINIMUM,
            "outside (virtual)",
            float(contact[0]),
            float(contact[1]),
            None,
        ),
    ]
    edges = [
        Edge("axis", "x_point"),
        Edge("x_point", "outside", join_level_wb=contact_level),
    ]
    return nodes, edges, sigma, contact_level


def _verify(
    exact: CerfonFreidbergSingleNull,
    wall: np.ndarray,
    nodes: list[Node],
) -> list[dict[str, Any]]:
    """Check the tree shape against a brute-force superlevel-set census.

    The census is independent of how the nodes were located: it counts
    connected components of sigma*psi above a level, with every wall-touching
    component glued to the single outside node.  A level above the axis (the
    positive control) must carry no component at all, so an empty count there
    is a real absence and not a broken census.
    """
    axis_psi = nodes[0].psi_wb
    xpoint_psi = nodes[1].psi_wb
    scale = axis_psi - xpoint_psi

    radius = np.linspace(
        min(wall[:, 0].min(), nodes[0].radius - 0.6),
        max(wall[:, 0].max(), nodes[0].radius + 0.6),
        CHECK_RADIUS_SAMPLES,
    )
    height = np.linspace(
        wall[:, 1].min() - PAD,
        wall[:, 1].max() + PAD,
        CHECK_HEIGHT_SAMPLES,
    )
    radius_grid, height_grid = np.meshgrid(radius, height)
    points = np.column_stack((radius_grid.ravel(), height_grid.ravel()))
    flux = _flux(exact, points).reshape(radius_grid.shape)
    inside = _inside_mask(radius, height, wall)

    records: list[dict[str, Any]] = []
    for fraction in CHECK_FRACTIONS:
        level = xpoint_psi + fraction * scale
        components, wall_touching = _components(flux, inside, level)
        records.append(
            {
                "fraction": float(fraction),
                "level_wb": float(level),
                "components": int(components),
                "wall_touching": int(wall_touching),
                "control": bool(fraction > 1.0),
            }
        )
    return records


def _assert_verification(
    records: list[dict[str, Any]], contact_fraction: float
) -> None:
    """Fail loudly unless the census matches a three-node path.

    Above the axis: nothing.  Between the axis and the wall contact: one
    interior core component, touching no wall.  Between the wall contact and
    the X-point: the core plus the outside, so two components with one of them
    on the wall.  Below the X-point: the two have merged into one wall-touching
    component.
    """
    for record in records:
        fraction = record["fraction"]
        components = record["components"]
        wall_touching = record["wall_touching"]
        if fraction > 1.0:
            if components != 0:
                raise RuntimeError(
                    f"a level above the axis carries {components} region(s)"
                )
        elif fraction > contact_fraction:
            if components != 1 or wall_touching != 0:
                raise RuntimeError(
                    f"the interior above the wall contact is not one core "
                    f"region at fraction {fraction}"
                )
        elif fraction > 0.0:
            if components != 2 or wall_touching != 1:
                raise RuntimeError(
                    f"the core and the outside are not separate at fraction "
                    f"{fraction}: components={components} "
                    f"wall_touching={wall_touching}"
                )
        elif components != 1 or wall_touching != 1:
            raise RuntimeError(
                f"the core and the outside are not merged at fraction "
                f"{fraction}: components={components} "
                f"wall_touching={wall_touching}"
            )


def _class_read(
    records: list[dict[str, Any]], xpoint_psi: "float | None", contact_level_wb: float
) -> dict[str, Any]:
    """Read the class from where the wall contact sits relative to the saddle.

    A probe level strictly between the X-point and the wall contact shows
    whether the contact's region is the plasma's own: if the core and the
    outside are still two components there, the contact was reached through the
    X-point, so it lies in a private region and can never be a limiter.  If
    they have already merged, the contact was reached without passing through
    any saddle and the state is limited.
    """
    probe = None
    for record in records:
        if xpoint_psi < record["level_wb"] < contact_level_wb:
            probe = record
    if probe is None:
        raise RuntimeError("no census level falls between the saddle and the contact")
    limiter = probe["components"] == 1
    return {
        "probe_level_wb": probe["level_wb"],
        "probe_components": probe["components"],
        "probe_wall_touching": probe["wall_touching"],
        "contact_region": "plasma (no saddle crossed)" if limiter else "private",
        "limiter": limiter,
        "class": "limited" if limiter else "diverted",
    }


def _levels(
    exact: CerfonFreidbergSingleNull, wall: np.ndarray, axis_psi: float
) -> np.ndarray:
    """Return the stated raw-flux contour levels [Wb], the separatrix included."""
    low = float(np.min(_flux(exact, wall)))
    levels = np.linspace(low, axis_psi, CONTOUR_LEVEL_COUNT)
    levels[int(np.argmin(np.abs(levels)))] = 0.0
    return np.unique(levels)


def _format_flux(value: float) -> str:
    """Format a flux [Wb], showing a stationary-point residual as zero."""
    if abs(value) < 1.0e-12:
        return "+0.000e+00"
    return f"{value:+.3e}"


def _node_label(node: Node) -> str:
    """Return a node's direct label; a virtual node carries no flux."""
    if node.psi_wb is None:
        return node.kind
    return node.kind + "\nψ = " + _format_flux(node.psi_wb) + " Wb"


def _join_label(edge: Edge) -> str:
    """Return the label of the edge joining the outside node to the tree."""
    if edge.contact_region == "private":
        region = "private region below the X-point"
        verdict = "never a limiter → diverted"
    else:
        region = "plasma region"
        verdict = "limiter -> limited"
    level = _format_flux(edge.join_level_wb)
    return "wall contact " + level + " Wb\n" + region + "\n" + verdict


def _panel(
    exact: CerfonFreidbergSingleNull,
    wall: np.ndarray,
    nodes: list[Node],
    edges: list[Edge],
    levels: np.ndarray,
) -> Any:
    """Draw the poloidal panel and return the view holding its figure."""
    r_min = float(wall[:, 0].min()) - PAD
    r_max = float(wall[:, 0].max()) + PAD
    z_min = float(wall[:, 1].min()) - PAD
    z_max = float(wall[:, 1].max()) + PAD

    radius = np.linspace(r_min, r_max, CONTOUR_RADIUS_SAMPLES)
    height = np.linspace(z_min, z_max, CONTOUR_HEIGHT_SAMPLES)
    radius_grid, height_grid = np.meshgrid(radius, height)
    points = np.column_stack((radius_grid.ravel(), height_grid.ravel()))
    flux = _flux(exact, points).reshape(radius_grid.shape)
    inside = _inside_mask(radius, height, wall)
    masked = np.where(inside, flux, np.nan)
    wall_units = (wall,)

    view = poloidal_view((r_min, r_max, z_min, z_max), height=9.5, style=TREE_INK)
    axes = view.poloidal

    poloidal.draw_flux_contours(axes, radius, height, masked, levels, style=TREE_INK)
    separatrix = exact.separatrix(SEPARATRIX_POINTS)
    poloidal.draw_boundary(axes, separatrix[:, 0], separatrix[:, 1], style=TREE_INK)
    poloidal.draw_wall(axes, units=wall_units, style=TREE_INK)

    by_index = {node.index: node for node in nodes}
    for edge in edges:
        first = by_index[edge.first]
        second = by_index[edge.second]
        axes.plot(
            [first.radius, second.radius],
            [first.height, second.height],
            color=TREE_EDGE_COLOR,
            linewidth=1.1,
            linestyle=(0.0, (1.6, 1.6)),
            zorder=DEFAULT_INK.zorder_separatrix,
        )

    poloidal.draw_nulls(
        axes,
        magnetic_axis=[nodes[0].radius, nodes[0].height],
        x_points=np.array([[nodes[1].radius, nodes[1].height]]),
        contain=wall_units,
        style=TREE_INK,
    )
    axes.plot(
        nodes[2].radius,
        nodes[2].height,
        marker=OUTSIDE_MARKER,
        markersize=DEFAULT_INK.xpoint_markersize,
        markerfacecolor=TREE_EDGE_COLOR,
        markeredgecolor=TREE_EDGE_COLOR,
        markeredgewidth=DEFAULT_INK.xpoint_markeredgewidth,
        linestyle="none",
        zorder=DEFAULT_INK.zorder_markers,
    )

    placements = (
        (nodes[0], -0.06, 0.07, "right"),
        (nodes[1], 0.06, 0.09, "left"),
        (nodes[2], 0.06, -0.01, "left"),
    )
    for node, dr, dz, horizontal in placements:
        axes.annotate(
            _node_label(node),
            xy=(node.radius + dr, node.height + dz),
            fontsize=LABEL_FONTSIZE,
            ha=horizontal,
            va="center",
            color=TREE_EDGE_COLOR,
            bbox=DEFAULT_INK.label_bbox,
            zorder=DEFAULT_INK.zorder_label,
        )
    join = next(edge for edge in edges if edge.join_level_wb is not None)
    axes.annotate(
        _join_label(join),
        xy=(nodes[1].radius + 0.20, 0.5 * (nodes[1].height + nodes[2].height)),
        fontsize=LABEL_FONTSIZE,
        ha="left",
        va="center",
        color=TREE_EDGE_COLOR,
        bbox=DEFAULT_INK.label_bbox,
        zorder=DEFAULT_INK.zorder_label,
    )
    return view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURE_STEM)
    arguments = parser.parse_args()

    configure_dtypes()
    exact = _reference()
    wall = _wall(exact)
    nodes, edges, sigma, contact_level = _tree(exact, wall)
    records = _verify(exact, wall, nodes)
    scale = nodes[0].psi_wb - nodes[1].psi_wb
    contact_fraction = (contact_level - nodes[1].psi_wb) / scale
    _assert_verification(records, contact_fraction)
    class_read = _class_read(records, nodes[1].psi_wb, contact_level)
    join = edges[1]
    edges = [
        edges[0],
        Edge(
            join.first,
            join.second,
            join_level_wb=join.join_level_wb,
            contact_region=class_read["contact_region"],
            limiter=class_read["limiter"],
        ),
    ]
    if len(nodes) - len(edges) != 1:
        raise RuntimeError(
            f"a contour tree does not have one more node than edge: "
            f"{len(nodes)} nodes, {len(edges)} edges"
        )
    levels = _levels(exact, wall, nodes[0].psi_wb)

    figure_path = arguments.output.with_suffix(".png")
    receipt_path = arguments.output.with_suffix(".json")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    view = _panel(exact, wall, nodes, edges, levels)
    view.figure.savefig(figure_path, dpi=DEFAULT_INK.figure_dpi)
    view.figure.savefig(figure_path.with_suffix(".svg"))

    receipt = {
        "schema": "nova.contour-tree-explanatory-figure",
        "version": 1,
        "flux_units": "Wb",
        "flux_note": (
            "raw poloidal flux at the analytic reference's own value. The "
            "outside node is virtual: it has no flux of its own and is valued "
            "below every interior level; the level at which it joins the tree "
            "is recorded on its edge, not on the node."
        ),
        "sigma": sigma,
        "class_read": class_read,
        "nodes": [asdict(node) for node in nodes],
        "edges": [asdict(edge) for edge in edges],
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_count_minus_edge_count": len(nodes) - len(edges),
        "contour_levels_wb": [float(level) for level in levels],
        "separatrix_level_wb": 0.0,
        "wall_source": (
            "the certificate fixture carries no wall units; the wall drawn is "
            "the fixture's own diverted wall, a smooth outward offset of the "
            "analytic separatrix (scripts.analytic_oracle_fixtures.measure."
            "offset_wall), and is not a machine wall"
        ),
        "wall_points": int(wall.shape[0]),
        "wall_clearance_fraction_of_minor_radius": WALL_CLEARANCE_FRACTION,
        "superlevel_census": records,
        "figure": str(figure_path.relative_to(ROOT)),
        "figure_svg": str(figure_path.with_suffix(".svg").relative_to(ROOT)),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        f"nodes={receipt['node_count']} edges={receipt['edge_count']} "
        f"difference={receipt['node_count_minus_edge_count']} "
        f"sigma={receipt['sigma']}"
    )
    for node in nodes:
        flux = (
            "virtual (no psi)"
            if node.psi_wb is None
            else _format_flux(node.psi_wb) + " Wb"
        )
        print(f"  {node.kind:18s} R={node.radius:+.5f} Z={node.height:+.5f} psi={flux}")
    print(
        f"  join level={_format_flux(contact_level)} Wb  "
        f"class={class_read['class']}  region={class_read['contact_region']}"
    )


if __name__ == "__main__":
    main()
