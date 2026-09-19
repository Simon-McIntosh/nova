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
from nova.equilibrium.flux_surface_geometry import _trace_surfaces
from nova.equilibrium.topology import Topology, TopologyClass
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes
from nova.linalg.tensor_spline import fit_tensor_spline


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
    start = polyline
    end = np.roll(polyline, -1, axis=0)
    segment = end - start
    length_squared = np.sum(segment**2, axis=1)
    length_squared = np.where(length_squared > 0.0, length_squared, 1.0)
    best = np.inf
    for point in points:
        offset = point[None, :] - start
        fraction = np.clip(
            np.sum(offset * segment, axis=1) / length_squared,
            0.0,
            1.0,
        )
        closest = start + fraction[:, None] * segment
        best = min(best, float(np.min(np.linalg.norm(closest - point, axis=1))))
    return float(best)


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


def test_persisted_limited_terminal_boundary_uses_contact_flux():
    """The converged MAST boundary touches its wall on its contour authority."""

    with np.load(LIMITED_TOUCH / "row-16-terminal-state.npz") as state:
        radius = np.asarray(state["radius_axis"], dtype=np.float64)
        height = np.asarray(state["height_axis"], dtype=np.float64)
        flux = np.asarray(state["values"], dtype=np.float64)
        wall = np.asarray(state["wall"], dtype=np.float64)
        wall_zone = jnp.asarray(state["wall_zone"], dtype=jnp.float64)
    receipt = json.loads((LIMITED_TOUCH / "receipt.json").read_text())
    measured = receipt["after"]["rows"][0]

    surface = fit_tensor_spline(
        jnp.asarray(radius),
        jnp.asarray(height),
        jnp.asarray(flux.T),
    )
    topology = _topology(radius, height, wall)
    contact = topology.wall_anchor_data(
        jnp.asarray(wall_zone),
        1.0,
        int(TopologyClass.LIMITED),
        surface=surface,
    )
    contact = np.asarray(contact, dtype=np.float64)
    sampled_flux = float(surface(contact[0], contact[1]))
    np.testing.assert_allclose(contact[2], sampled_flux, rtol=1.0e-9, atol=0.0)

    contour = _traced_boundary(
        radius,
        height,
        flux,
        np.asarray(measured["axis_position_m"], dtype=np.float64),
        float(contact[2]),
    )
    pitch = max(float(np.mean(np.diff(radius))), float(np.mean(np.diff(height))))
    assert _polyline_distance(contour, wall) < pitch


def test_wall_zone_disagreement_cannot_move_structured_boundary_level():
    """A conflicting wall-zone vector cannot author a structured contour."""

    with np.load(LIMITED_TOUCH / "row-16-terminal-state.npz") as state:
        radius = np.asarray(state["radius_axis"], dtype=np.float64)
        height = np.asarray(state["height_axis"], dtype=np.float64)
        flux = np.asarray(state["values"], dtype=np.float64)
        wall = np.asarray(state["wall"], dtype=np.float64)
        wall_zone = jnp.asarray(state["wall_zone"], dtype=jnp.float64)
    surface = fit_tensor_spline(radius, height, flux.T)
    topology = _topology(radius, height, wall)
    baseline = topology.wall_anchor_data(
        wall_zone,
        1.0,
        int(TopologyClass.LIMITED),
        surface=surface,
    )
    contradictory = wall_zone.at[0].set(jnp.max(wall_zone) + 1.0)
    guarded = topology.wall_anchor_data(
        contradictory,
        1.0,
        int(TopologyClass.LIMITED),
        surface=surface,
    )
    np.testing.assert_array_equal(np.asarray(guarded), np.asarray(baseline))


def test_every_diverted_bank_boundary_passes_through_admitted_saddle():
    """All twelve bank arms carry the admitted saddle on the traced contour."""

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
            contour = _traced_boundary(
                radius,
                height,
                flux.T,
                np.asarray(bank[prefix + "axis"], dtype=np.float64),
                float(bank[prefix + "binding_flux"]),
            )
            pitch = max(
                float(np.mean(np.diff(radius))),
                float(np.mean(np.diff(height))),
            )
            assert np.min(np.linalg.norm(contour - saddle, axis=1)) < pitch


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
