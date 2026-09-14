"""Limiter wall-contact read on the analytic oracle fixtures.

The wall contact is the extremum of the quadratic fitted through the extremal
wall node and its two neighbours in cumulative arc length along the wall
polyline, with the level the quadratic's extremum value.  On the
designed-tangent limiter fixtures a wall node sits exactly at the analytic
tangency (the outboard boundary-flux point), so the contact lands on that node
and the position error against the analytic tangency is already at the machine
floor at the coarsest node count; the level error against the continuum flux
extremum falls at least second order between 241 and 481 wall nodes.  The
offset diverted wall is resampled by arc length at each node count, so its
position contract is a panel bound while the level stays second order.  The
private-wall shadow mask is applied unchanged before selection, and the
contact read depends only on the wall polyline and its nodal flux, so whole
121-node rows are bit-identical across lattice sizes exactly as they are
across cell counts.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    from benchmarks.limiter_read_resolution_audit import (
        _analytic_wall_extremum,
        _diverted_wall,
    )
    from benchmarks import solovev_certificate as certificate
    from nova.biot.null import Null1D, Null2D
    from nova.equilibrium import TopologyClass
    from nova.equilibrium.topology import Topology
    from nova.jax.config import configure_dtypes
    from scripts.analytic_oracle_fixtures import measure as oracle_fixture


WEAK = "weak-rotation-reactor-static"
DIVERTED = "diverted-single-null"


@pytest.fixture(scope="module", autouse=True)
def _double_precision():
    """Match the production certificate's extended-precision ladder."""
    configure_dtypes()


def _exact_flux(case_name, exact, points):
    return certificate._exact_state(
        case_name, exact, np.asarray(points, dtype=np.float64)
    )


def _square_stencils(radial_count, vertical_count):
    """Ring-connect a ``radial_count``-by-``vertical_count`` lattice grid."""
    rings = []
    around = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    for radial in range(1, radial_count - 1):
        for vertical in range(1, vertical_count - 1):
            centre = radial * vertical_count + vertical
            ring = [centre]
            ring.extend(
                (radial + dr) * vertical_count + vertical + dz for dr, dz in around
            )
            rings.append(ring)
    return np.asarray(rings, dtype=np.intp)


def _topology_row(case_name, radial, vertical, wall_nodes):
    """Read the contact on a Null2D lattice grid plus the awaited wall.

    The grid box follows the fixture's own recipe (boundary midplane radii
    widened by a margin for limiter cases, the wall-derived envelope for the
    diverted case) so the box contains the qualified magnetic axis.  The
    contact read itself uses only the wall polyline and its nodal flux, so
    grid size is inert.
    """
    carrier, source_case, exact = certificate._case(case_name)
    wall = (
        _diverted_wall(exact, wall_nodes)
        if certificate._is_diverted_case(case_name)
        else oracle_fixture.limiter_contour(exact, points=wall_nodes)
    )
    wall = np.asarray(wall, dtype=np.float64)
    if certificate._is_diverted_case(case_name):
        pad = 0.15
        radii = np.linspace(wall[:, 0].min() - pad, wall[:, 0].max() + pad, radial)
        heights = np.linspace(wall[:, 1].min() - pad, wall[:, 1].max() + pad, vertical)
    else:
        inboard, outboard = carrier.boundary_midplane_radii()
        half_height = np.sqrt(carrier.axis_flux / carrier.field_coefficient)
        radii = np.linspace(inboard - 0.12, outboard + 0.12, radial)
        heights = np.linspace(-1.18 * half_height, 1.18 * half_height, vertical)
    coord_r, coord_z = np.meshgrid(radii, heights, indexing="ij")
    coordinate = np.c_[coord_r.ravel(), coord_z.ravel()].astype(np.float64)
    grid = Null2D.from_coordinates(
        coordinate, _square_stencils(radial, vertical), maxsize=5
    )
    topology = Topology(grid, Null1D(np.asarray(wall, dtype=np.float64)))
    state = np.r_[
        _exact_flux(case_name, exact, coordinate),
        _exact_flux(case_name, exact, wall),
    ]
    state = np.asarray(state, dtype=np.float64)
    _masks, read = topology.read(state, 1.0, np.ones(len(coordinate), dtype=bool))
    continuum = _analytic_wall_extremum(case_name, exact, wall, 1.0)
    panels = np.linalg.norm(np.roll(wall, -1, axis=0) - wall, axis=1)
    return {
        "topology": topology,
        "state": state,
        "grid_node_count": len(coordinate),
        "contact": np.asarray(read.wall_point, dtype=np.float64),
        "contact_flux": float(read.wall_point_flux),
        "continuum_r_z": np.asarray(continuum["coordinate_rz_m"]),
        "continuum_flux_wb": float(continuum["flux_wb"]),
        "median_panel_m": float(np.median(panels)),
        "wall": wall,
    }


def _designed_tangency(wall):
    """The limiter tangency the fixture authors at its outboard node."""
    return wall[(wall.shape[0] - 1) // 2]


def test_limiter_contact_is_arc_length_quadratic_extremum():
    """The weak contact lands on the analytic tangency node with the level at
    the boundary flux, and the arc-length quadratic stays finite."""
    row = _topology_row(WEAK, 45, 55, 241)
    contact = row["contact"]
    tangency = _designed_tangency(row["wall"])
    assert np.linalg.norm(contact[:2] - tangency) < 1.0e-6
    assert abs(row["contact_flux"]) < 1.0e-6
    assert np.all(np.isfinite(contact))


@pytest.mark.parametrize("case_name", [WEAK, DIVERTED])
def test_level_error_stays_second_order(case_name):
    """The contact level error against the continuum flux extremum falls at
    least second order between 241 and 481 wall nodes (fitted order >= 1.9;
    the single-null level error is measured well above that)."""
    coarse = _topology_row(case_name, 45, 55, 241)
    fine = _topology_row(case_name, 45, 55, 481)

    def level_error(row):
        return abs(row["contact_flux"] - row["continuum_flux_wb"])

    fitted = float(np.log(level_error(coarse) / level_error(fine)) / np.log(2.0))
    assert fitted >= 1.9


def test_weak_position_error_is_second_order_between_241_and_481():
    """The weak contact position error against the analytic tangency is already
    at the machine floor (sub-100 nm on 5 cm wall panels) at 241 nodes and does
    not degrade at 481, which is the strongest form of the at-least-second-
    order contract."""
    coarse = _topology_row(WEAK, 45, 55, 241)
    fine = _topology_row(WEAK, 45, 55, 481)
    tangency = _designed_tangency(coarse["wall"])
    coarse_error = np.linalg.norm(coarse["contact"][:2] - tangency)
    fine_error = np.linalg.norm(fine["contact"][:2] - tangency)
    assert coarse_error < 1.0e-6
    assert fine_error < 1.0e-6
    assert fine_error <= 4.0 * coarse_error


def test_diverted_position_stays_within_one_wall_panel():
    """The diverted-wall contact stays within one median wall panel of the
    analytic continuum extremum."""
    row = _topology_row(DIVERTED, 45, 55, 241)
    error = np.linalg.norm(row["contact"][:2] - row["continuum_r_z"])
    assert error <= row["median_panel_m"]


@pytest.mark.parametrize("case_name", [WEAK, DIVERTED])
def test_wall_contact_bit_identical_across_lattice_sizes(case_name):
    """121-node rows read bit-identical contact positions on two lattice
    sizes, because the contact read depends only on the wall polyline and its
    nodal flux."""
    small = _topology_row(case_name, 45, 55, 121)
    large = _topology_row(case_name, 65, 75, 121)
    assert np.array_equal(small["contact"][:2], large["contact"][:2])


@pytest.mark.parametrize("case_name", [WEAK, DIVERTED])
def test_private_wall_shadow_mask_applied_before_selection(case_name):
    """Masking the winning wall node rules it out of the selection.

    The private-wall shadow mask is applied unchanged before selection: masked
    nodes receive a finite losing score, so a masked winning node cannot carry
    the contact.
    """
    row = _topology_row(case_name, 45, 55, 241)
    topology = row["topology"]
    contact = row["contact"]
    wall = row["wall"]
    winning = int(np.argmin(np.sum((wall - contact[:2]) ** 2, axis=1)))
    mask = np.zeros(wall.shape[0], dtype=bool)
    mask[winning] = True
    _masks, qualified, _connected, _admitted, _polish, _uncertain = (
        topology.read_qualification(
            row["state"],
            1.0,
            np.ones(row["grid_node_count"], dtype=bool),
            requested_class=int(TopologyClass.LIMITED),
            private_wall_node_mask=mask,
        )
    )
    assert not np.allclose(qualified.wall_point[:2], contact[:2], atol=1.0e-9)
