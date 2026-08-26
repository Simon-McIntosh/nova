"""External coil nulls cannot capture the plasma topology read."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.biot.greens import hybrid_greens
from nova.biot.null import Null1D, Null2D
from nova.equilibrium.connectivity_boundary import _raster_hex_partition_geometry
from nova.equilibrium.domain import PlasmaDomain
from nova.equilibrium.topology import Topology, TopologyClass
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes


@dataclass(frozen=True)
class ManufacturedCoilStack:
    """Plasma rings and a separated column of coil-like current centres."""

    plasma_centres: np.ndarray
    plasma_currents: np.ndarray
    coil_centres: np.ndarray
    coil_current: float

    def flux(self, coordinate: np.ndarray, *, include_coils: bool) -> np.ndarray:
        """Evaluate the smoothed circular-current flux on physical points."""
        values = np.zeros(coordinate.shape[0], dtype=np.float64)
        for (radius, height), current in zip(
            self.plasma_centres, self.plasma_currents, strict=True
        ):
            values += (
                current
                * hybrid_greens(
                    coordinate[:, 0], coordinate[:, 1], radius, height, 0.06, 0.06
                )[0]
            )
        if include_coils:
            for radius, height in self.coil_centres:
                values += (
                    self.coil_current
                    * hybrid_greens(
                        coordinate[:, 0], coordinate[:, 1], radius, height, 0.03, 0.03
                    )[0]
                )
        return values


STACK = ManufacturedCoilStack(
    plasma_centres=np.asarray([[1.0, 0.18], [1.0, -0.48], [1.0, 0.62]]),
    plasma_currents=np.asarray([1.0e6, 4.0e5, 3.0e5]),
    coil_centres=np.asarray(
        [[2.175, -0.75], [2.175, -0.25], [2.175, 0.25], [2.175, 0.75]]
    ),
    coil_current=5.0e4,
)


@pytest.fixture(scope="module", autouse=True)
def _double_precision():
    """Match the precision used by the production topology read."""
    configure_dtypes()


def _fixture(shape: tuple[int, int]):
    radius = np.linspace(0.55, 2.50, shape[0])
    height = np.linspace(-1.0, 1.0, shape[1])
    radial, vertical = np.meshgrid(radius, height, indexing="ij")
    coordinate = np.c_[radial.ravel(), vertical.ravel()]
    wall_angle = np.linspace(0.0, 2.0 * np.pi, 192, endpoint=False)
    wall = np.c_[1.0 + 0.46 * np.cos(wall_angle), 0.82 * np.sin(wall_angle)]
    topology = Topology(
        Null2D.from_coordinates(coordinate, hex_stencil(shape), maxsize=12),
        Null1D(jnp.asarray(wall)),
    )
    inside_material = ((coordinate[:, 0] - 1.0) / 0.44) ** 2 + (
        coordinate[:, 1] / 0.78
    ) ** 2 <= 1.0
    return topology, coordinate, wall, inside_material, radius, height


def _partition_boundary_band(labels: np.ndarray, rings: np.ndarray) -> np.ndarray:
    """Return every cell on, or one hex link from, a label boundary."""
    flat = labels.reshape(-1)
    boundary = np.zeros(flat.shape, dtype=bool)
    for ring in rings:
        centre = int(ring[0])
        boundary[centre] = np.any(flat[ring[1:]] != flat[centre])
    band = boundary.copy()
    for ring in rings:
        if boundary[int(ring[0])]:
            band[ring] = True
    return band


@pytest.mark.parametrize("shape", [(31, 41), (37, 49)], ids=["coarse", "fine"])
def test_external_coil_nulls_leave_plasma_selection_and_partition_invariant(shape):
    """A separated multi-null coil column cannot own or shadow the plasma."""
    topology, coordinate, wall, inside, radius, height = _fixture(shape)
    plasma_flux = STACK.flux(coordinate, include_coils=False)
    plasma_wall_flux = STACK.flux(wall, include_coils=False)
    combined_flux = STACK.flux(coordinate, include_coils=True)
    combined_wall_flux = STACK.flux(wall, include_coils=True)
    inside_device = jnp.asarray(inside)

    vmap_o, vmap_x = topology.grid(jnp.asarray(combined_flux))
    data_w = topology.wall(jnp.asarray(combined_wall_flux), 1)
    qualified = topology.qualified_o_candidates(
        vmap_o,
        vmap_x,
        data_w,
        1,
        jnp.asarray(combined_flux),
        inside_device,
    )
    selected_o = topology.o_point_data(vmap_o, 1, qualified)
    selected_x = topology.x_point_data(vmap_x, 1, selected_o[2])
    finite_o = np.asarray(vmap_o)[np.isfinite(np.asarray(vmap_o)[:, 0])]
    finite_x = np.asarray(vmap_x)[np.isfinite(np.asarray(vmap_x)[:, 0])]

    coil_o = finite_o[finite_o[:, 0] > 1.90]
    intercoil_x = finite_x[finite_x[:, 0] > 2.0]
    assert coil_o.shape[0] >= 3
    assert intercoil_x.shape[0] >= 3
    assert np.all(
        np.min(np.linalg.norm(coil_o[:, None, :2] - STACK.coil_centres, axis=2), axis=1)
        < 0.08
    )
    for lower, upper in zip(
        STACK.coil_centres[:-1, 1], STACK.coil_centres[1:, 1], strict=True
    ):
        assert np.any((intercoil_x[:, 1] > lower) & (intercoil_x[:, 1] < upper))

    coil_candidate = np.asarray(vmap_o)[:, 0] > 1.90
    assert not np.any(np.asarray(qualified) & coil_candidate)
    assert float(selected_o[0]) < 1.25
    assert float(selected_x[0]) < 1.30

    combined_psi = jnp.asarray(np.r_[combined_flux, combined_wall_flux])
    plasma_psi = jnp.asarray(np.r_[plasma_flux, plasma_wall_flux])

    def solve(psi):
        return topology.read_with_connectivity(
            psi, 1, inside_device, TopologyClass.DIVERTED
        )

    eager_masks, eager_state, eager_connected = solve(combined_psi)
    compiled_masks, compiled_state, compiled_connected = jax.jit(solve)(combined_psi)
    np.testing.assert_array_equal(eager_masks.label, compiled_masks.label)
    np.testing.assert_array_equal(eager_connected, compiled_connected)
    np.testing.assert_array_equal(eager_state.axis, compiled_state.axis)
    np.testing.assert_array_equal(eager_state.x_point, compiled_state.x_point)
    np.testing.assert_allclose(eager_state.axis, selected_o[:2], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        eager_state.x_point, selected_x[:2], rtol=0.0, atol=1.0e-12
    )

    plasma_masks, plasma_state, _plasma_connected = solve(plasma_psi)
    combined_labels = np.asarray(eager_masks.label)
    plasma_labels = np.asarray(plasma_masks.label)
    rings = np.asarray(
        _raster_hex_partition_geometry(jnp.asarray(radius), jnp.asarray(height))[0]
    )
    comparison_band = _partition_boundary_band(
        combined_labels, rings
    ) | _partition_boundary_band(plasma_labels, rings)
    comparison = inside & ~comparison_band
    np.testing.assert_array_equal(
        combined_labels[comparison], plasma_labels[comparison]
    )

    stable_plasma_core = (plasma_labels == int(PlasmaDomain.CORE)) & ~comparison_band
    coil_shadow = (
        combined_labels == int(PlasmaDomain.PRIVATE_FLUX)
    ) & stable_plasma_core
    assert not np.any(coil_shadow)
    assert float(plasma_state.axis[0]) < 1.25
    assert float(plasma_state.x_point[0]) < 1.30
