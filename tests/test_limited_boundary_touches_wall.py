"""Physical contact contracts for limited and diverted terminal boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from nova.biot.null import Null1D, Null2D
from nova.equilibrium.connectivity_boundary import _points_inside_polygon
from nova.equilibrium.flux_surface_geometry import _trace_surfaces
from nova.equilibrium.topology import Topology, TopologyClass
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes


ROOT = Path(__file__).parents[1]
LIMITED_TOUCH = ROOT / "docs/figures/null-identification-authority/limited-touch"
MAST_BANK = (
    ROOT / "docs/figures/topology-visual-corroboration/mast-bank-solve-operands.npz"
)
CERTIFICATE_PARTS = (
    ROOT / "docs/figures/gs-absolute-accuracy/solovev/production-route-parts"
)
LIMITED_CERTIFICATE_ROWS = (
    "weak-rotation-reactor-static-production-route-reduced.json",
    "weak-rotation-reactor-static-production-route-cells-300.json",
    "weak-rotation-reactor-static-production-route-cells-500.json",
    "weak-rotation-reactor-static-production-route-cells-1000.json",
)


@pytest.fixture(scope="module", autouse=True)
def _double_precision():
    """Construct every fixture after extended precision is enabled."""

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True


def _topology(radius: np.ndarray, height: np.ndarray, wall: np.ndarray) -> Topology:
    radial, vertical = np.meshgrid(radius, height, indexing="ij")
    coordinate = np.column_stack((radial.ravel(), vertical.ravel()))
    grid = Null2D.from_coordinates(
        coordinate,
        hex_stencil((radius.size, height.size)),
        maxsize=30,
    )
    return Topology(grid, Null1D(jnp.asarray(wall, dtype=jnp.float64)))


def _polyline_distance(points: np.ndarray, polyline: np.ndarray) -> float:
    return _closest_polyline_pair(points, polyline)[0]


def _closest_polyline_pair(
    points: np.ndarray, polyline: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    start = polyline
    end = np.roll(polyline, -1, axis=0)
    segment = end - start
    length_squared = np.sum(segment**2, axis=1)
    length_squared = np.where(length_squared > 0.0, length_squared, 1.0)
    best = np.inf
    best_point = np.full(2, np.nan)
    best_polyline_point = np.full(2, np.nan)
    for point in points:
        offset = point[None, :] - start
        fraction = np.clip(
            np.sum(offset * segment, axis=1) / length_squared,
            0.0,
            1.0,
        )
        closest = start + fraction[:, None] * segment
        distance = np.linalg.norm(closest - point, axis=1)
        index = int(np.argmin(distance))
        if distance[index] < best:
            best = float(distance[index])
            best_point = point.copy()
            best_polyline_point = closest[index].copy()
    return float(best), best_point, best_polyline_point


def _topology_read(
    radius: np.ndarray,
    height: np.ndarray,
    flux: np.ndarray,
    wall: np.ndarray,
    wall_zone: np.ndarray,
    requested_class: int | None,
):
    topology = _topology(radius, height, wall)
    radial, vertical = np.meshgrid(radius, height, indexing="ij")
    coordinate = np.column_stack((radial.ravel(), vertical.ravel()))
    inside_material = np.asarray(
        _points_inside_polygon(
            coordinate[:, 0],
            coordinate[:, 1],
            wall[:, 0],
            wall[:, 1],
        ),
        dtype=bool,
    )
    state = np.concatenate((flux.ravel(), np.asarray(wall_zone, dtype=np.float64)))
    _masks, topology_state = topology.read(
        jnp.asarray(state),
        1.0,
        jnp.asarray(inside_material),
        requested_class=requested_class,
    )
    return topology, topology_state


def _limited_read(
    radius: np.ndarray,
    height: np.ndarray,
    flux: np.ndarray,
    wall: np.ndarray,
    wall_zone: np.ndarray,
):
    return _topology_read(
        radius, height, flux, wall, wall_zone, int(TopologyClass.LIMITED)
    )


def _traced_boundary(
    radius: np.ndarray,
    height: np.ndarray,
    flux: np.ndarray,
    axis: np.ndarray,
    boundary_flux: float,
) -> np.ndarray:
    interpolant = RectBivariateSpline(radius, height, flux, kx=3, ky=3, s=0)
    axis_flux = float(interpolant.ev(float(axis[0]), float(axis[1])))
    traced = _trace_surfaces(
        interpolant,
        tuple(axis),
        axis_flux,
        boundary_flux - axis_flux,
        np.asarray([1.0]),
        radius,
        height,
        256,
    )
    return np.column_stack((traced.radius[:, 0], traced.height[:, 0]))


def test_persisted_limited_terminal_boundary_contact_is_axis_connected():
    """The converged MAST contact belongs to the axis-enclosing contour."""

    with np.load(LIMITED_TOUCH / "row-16-terminal-state.npz") as state:
        radius = np.asarray(state["radius_axis"], dtype=np.float64)
        height = np.asarray(state["height_axis"], dtype=np.float64)
        flux = np.asarray(state["values"], dtype=np.float64)
        wall = np.asarray(state["wall"], dtype=np.float64)
        wall_zone = np.asarray(state["wall_zone"], dtype=np.float64)
    receipt = json.loads((LIMITED_TOUCH / "receipt.json").read_text())
    detached_contact = np.asarray(
        receipt["before"]["rows"][0]["contact_position_m"], dtype=np.float64
    )
    measured = receipt["after"]["rows"][0]

    _topology_instance, topology_state = _limited_read(
        radius, height, flux, wall, wall_zone
    )
    contact = np.r_[
        np.asarray(topology_state.wall_point, dtype=np.float64),
        float(topology_state.wall_point_flux),
    ]
    interpolant = RectBivariateSpline(radius, height, flux, kx=3, ky=3, s=0)
    sampled_flux = float(interpolant.ev(float(contact[0]), float(contact[1])))
    np.testing.assert_allclose(contact[2], sampled_flux, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        contact[:2],
        np.asarray(measured["contact_position_m"], dtype=np.float64),
        rtol=0.0,
        atol=2.0e-12,
    )

    contour = _traced_boundary(
        radius,
        height,
        flux,
        np.asarray(topology_state.axis, dtype=np.float64),
        float(contact[2]),
    )
    pitch = max(float(np.mean(np.diff(radius))), float(np.mean(np.diff(height))))
    contact_distance = _polyline_distance(contour, contact[None, :2])
    wall_distance, _contour_point, closest_wall_point = _closest_polyline_pair(
        contour, wall
    )
    assert contact_distance < pitch
    assert np.linalg.norm(closest_wall_point - contact[:2]) < pitch
    assert wall_distance <= contact_distance + np.finfo(float).eps * 32
    assert np.linalg.norm(contact[:2] - detached_contact) > pitch


def test_persisted_class_free_read_publishes_the_axis_connected_contact():
    """A class-free read evaluates containment and reads the wall contact.

    The emergent read has no requested class to pin containment to, so the
    screen has to be evaluated from each pass' own state. If it is decided once
    from the provisional pass the detached divertor lobe survives and the read
    still classifies diverted, which is the defect this test guards.
    """

    with np.load(LIMITED_TOUCH / "row-16-terminal-state.npz") as state:
        radius = np.asarray(state["radius_axis"], dtype=np.float64)
        height = np.asarray(state["height_axis"], dtype=np.float64)
        flux = np.asarray(state["values"], dtype=np.float64)
        wall = np.asarray(state["wall"], dtype=np.float64)
        wall_zone = np.asarray(state["wall_zone"], dtype=np.float64)
    receipt = json.loads((LIMITED_TOUCH / "receipt.json").read_text())
    measured = receipt["after"]["rows"][0]
    detached_contact = np.asarray(
        receipt["before"]["rows"][0]["contact_position_m"], dtype=np.float64
    )

    _topology_instance, topology_state = _topology_read(
        radius, height, flux, wall, wall_zone, None
    )
    contact = np.r_[
        np.asarray(topology_state.wall_point, dtype=np.float64),
        float(topology_state.wall_point_flux),
    ]
    assert not bool(np.asarray(topology_state.diverted))
    np.testing.assert_allclose(
        contact[:2],
        np.asarray(measured["contact_position_m"], dtype=np.float64),
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        contact[2],
        measured["boundary_flux_wb"],
        rtol=0.0,
        atol=2.0e-15,
    )

    contour = _traced_boundary(
        radius,
        height,
        flux,
        np.asarray(topology_state.axis, dtype=np.float64),
        float(contact[2]),
    )
    pitch = max(float(np.mean(np.diff(radius))), float(np.mean(np.diff(height))))
    assert _polyline_distance(contour, contact[None, :2]) < pitch
    assert np.linalg.norm(contact[:2] - detached_contact) > pitch


def test_wall_zone_disagreement_cannot_move_structured_boundary_level():
    """A conflicting wall-zone vector cannot author a structured contour."""

    with np.load(LIMITED_TOUCH / "row-16-terminal-state.npz") as state:
        radius = np.asarray(state["radius_axis"], dtype=np.float64)
        height = np.asarray(state["height_axis"], dtype=np.float64)
        flux = np.asarray(state["values"], dtype=np.float64)
        wall = np.asarray(state["wall"], dtype=np.float64)
        wall_zone = np.asarray(state["wall_zone"], dtype=np.float64)
    _baseline_topology, baseline = _limited_read(radius, height, flux, wall, wall_zone)
    contradictory = wall_zone.copy()
    contradictory[0] = float(np.max(wall_zone) + 1.0)
    _guarded_topology, guarded = _limited_read(
        radius, height, flux, wall, contradictory
    )
    np.testing.assert_array_equal(
        np.asarray(guarded.wall_point), np.asarray(baseline.wall_point)
    )
    np.testing.assert_array_equal(
        np.asarray(guarded.wall_point_flux), np.asarray(baseline.wall_point_flux)
    )


def test_every_diverted_bank_boundary_passes_through_admitted_saddle():
    """All twelve bank arms carry the admitted saddle on the boundary level."""

    with np.load(MAST_BANK, allow_pickle=False) as bank:
        metadata = json.loads(str(bank["metadata"]))
        assert metadata["arm_count"] == 12
        for index, row in enumerate(metadata["rows"]):
            assert row["nova_achieved_class"] == "diverted"
            prefix = f"arm_{index:02d}_"
            radius = np.asarray(bank[prefix + "radius"], dtype=np.float64)
            height = np.asarray(bank[prefix + "height"], dtype=np.float64)
            flux = np.asarray(bank[prefix + "flux"], dtype=np.float64)
            saddle = np.asarray(bank[prefix + "selected_saddle"], dtype=np.float64)
            interpolant = RectBivariateSpline(
                radius,
                height,
                flux.T,
                kx=3,
                ky=3,
                s=0,
            )
            boundary_flux = float(bank[prefix + "binding_flux"])
            saddle_flux = float(interpolant.ev(float(saddle[0]), float(saddle[1])))
            np.testing.assert_allclose(
                saddle_flux,
                boundary_flux,
                rtol=0.0,
                atol=8.0 * np.finfo(float).eps * max(abs(boundary_flux), 1.0),
            )


@pytest.mark.parametrize("filename", LIMITED_CERTIFICATE_ROWS)
def test_limited_certificate_boundary_is_its_wall_contact(filename: str):
    """Each limited certificate row publishes a level carried by its wall."""

    row = json.loads((CERTIFICATE_PARTS / filename).read_text())
    render = row["render_data"]
    terminal = render["terminal_topology"]
    assert terminal["class"] == "limited"
    wall_units = [
        np.asarray(unit, dtype=np.float64) for unit in render["wall_units_rz_m"]
    ]
    wall = np.vstack(wall_units)
    offsets = np.cumsum([0, *(unit.shape[0] for unit in wall_units)])
    cell_count = int(row["realised_cells"])
    flux = np.asarray(render["terminal_flux_wb"], dtype=np.float64)
    wall_flux = flux[cell_count : cell_count + wall.shape[0]]

    dummy_radius = np.asarray([0.0, 0.5, 1.0])
    dummy_height = np.asarray([-0.5, 0.0, 0.5])
    topology = _topology(dummy_radius, dummy_height, wall)
    topology.wall_unit_offsets = jnp.asarray(offsets, dtype=jnp.int32)
    topology.wall_unit_closed = jnp.ones(len(wall_units), dtype=bool)
    contact = np.asarray(topology.wall_anchor_data(wall_flux, 1.0), dtype=np.float64)

    assert np.isfinite(contact[2])
    assert _polyline_distance(contact[None, :2], wall) < np.finfo(float).eps * 16
