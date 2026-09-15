"""Identity contracts for the standalone parallel component classifier."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import ndimage

from nova.biot.greens import hybrid_greens
from nova.biot.null import Null1D, Null2D
from nova.equilibrium import flux_surface_connectivity as canonical
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.parallel_components import (
    label_parallel_connected_components_with_steps,
)
from nova.equilibrium.topology import Topology
from nova.equilibrium.wall_mask import inside_polygon
from nova.geometry.hexstencil import hex_stencil
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes


configure_dtypes()
assert jax.config.jax_enable_x64 is True

_MAST_SHOT = 22086
_MAST_ROW = 43
_GRID_STRIDE = 2
_DOUBLE_NULL_RING = np.array([[1.0, 0.0], [1.0, -0.62], [1.0, 0.62]])
_DOUBLE_NULL_CURRENT = np.array([1.0e6, 5.0e5, 4.0e5])


def _minimum_index_reference(mask: np.ndarray) -> np.ndarray:
    """Return one-based minimum flat indices from a host flood label."""
    components, count = ndimage.label(mask)
    labels = np.zeros(mask.shape, dtype=np.int32)
    for component in range(1, count + 1):
        selected = components == component
        labels[selected] = np.flatnonzero(selected.reshape(-1))[0] + 1
    return labels


def _assert_parallel_identity(mask: np.ndarray) -> int:
    """Require exact agreement with both host and canonical device labels."""
    parallel, steps, settled = label_parallel_connected_components_with_steps(
        jnp.asarray(mask)
    )
    fixed_point = canonical.label_connected_components(jnp.asarray(mask), mask.size)
    expected = _minimum_index_reference(mask)
    np.testing.assert_array_equal(np.asarray(parallel), np.asarray(fixed_point))
    np.testing.assert_array_equal(np.asarray(parallel), expected)
    assert bool(settled)
    assert int(steps) <= int(np.ceil(np.log2(mask.size))) + 2
    return int(steps)


def _ring_flux(coordinate: np.ndarray) -> np.ndarray:
    """Return the double-null ring set's poloidal flux on coordinates."""
    columns = np.stack(
        [
            hybrid_greens(
                coordinate[:, 0], coordinate[:, 1], radius, height, 0.06, 0.06
            )[0]
            for radius, height in _DOUBLE_NULL_RING
        ],
        axis=1,
    )
    return columns @ _DOUBLE_NULL_CURRENT


def _double_null_fixture() -> tuple[Topology, jax.Array, jax.Array]:
    """Build the analytic double-null topology used by the topology contracts."""
    lattice = FluxLattice(np.linspace(0.55, 1.45, 45), np.linspace(-0.75, 0.75, 71))
    coordinate = lattice.coordinate
    angle = 2 * np.pi * np.arange(48) / 48
    wall = np.c_[1.0 + 0.42 * np.cos(angle), 0.62 * np.sin(angle)]
    topology = Topology(
        Null2D.from_coordinates(coordinate, hex_stencil(lattice.shape), maxsize=5),
        Null1D(jnp.asarray(wall)),
    )
    inside = jnp.asarray(
        ((coordinate[:, 0] - 1.0) / 0.42) ** 2 + (coordinate[:, 1] / 0.62) ** 2 <= 1.0
    )
    psi = jnp.asarray(np.r_[_ring_flux(coordinate), _ring_flux(wall)])
    return topology, psi, inside


def _mast_confined_mask(row: int) -> np.ndarray:
    """Return the stride-two confined mask for one persisted MAST row."""
    store = Path(SHOT_STORE) / f"{_MAST_SHOT}.zarr"
    if not store.exists():
        pytest.skip("the MAST shot store is not present on this host")
    import zarr

    group = zarr.open_group(str(store), mode="r")["efm"]
    full_radius = np.asarray(group["gridr"], dtype=np.float64)
    full_height = np.asarray(group["gridz"], dtype=np.float64)
    radius = full_radius[::_GRID_STRIDE]
    height = full_height[::_GRID_STRIDE]
    limiter = np.column_stack(
        (
            np.asarray(group["limiterr"], dtype=np.float64),
            np.asarray(group["limiterz"], dtype=np.float64),
        )
    )
    radial_grid, vertical_grid = np.meshgrid(radius, height, indexing="ij")
    inside = (
        np.asarray(
            inside_polygon(
                radial_grid.reshape(-1),
                vertical_grid.reshape(-1),
                limiter[:, 0],
                limiter[:, 1],
            ),
            dtype=bool,
        )
        .reshape((len(radius), len(height)))
        .T
    )
    raw = np.asarray(group["psirz"][row], dtype=np.float64)
    live_columns = np.flatnonzero(np.all(np.isfinite(raw), axis=0))
    assert live_columns.size == full_radius.size
    psi = raw[:, live_columns].T[::_GRID_STRIDE, ::_GRID_STRIDE].T
    axis_flux = float(np.asarray(group["psi_axis"][row]))
    boundary_flux = float(np.asarray(group["psi_boundary"][row]))
    span = boundary_flux - axis_flux
    if abs(span) < 1e-12:
        span = 1e-12
    return ((psi - axis_flux) / span < 1.0) & inside


def test_random_masks_with_holes_and_enclaves_are_bit_identical():
    """Random perforated masks retain canonical minimum-index components."""
    rng = np.random.default_rng(9143)
    masks = []
    for shape in ((9, 13), (19, 27), (33, 33)):
        for _ in range(12):
            mask = ndimage.binary_closing(rng.random(shape) > 0.43)
            mask &= rng.random(shape) > 0.09
            masks.append(mask)

    steps = [_assert_parallel_identity(mask) for mask in masks]
    assert min(steps) > 0


def test_narrow_enclave_path_resolves_to_its_minimum_index():
    """A winding one-cell path defeats local minima but remains one component."""
    mask = np.zeros((17, 19), dtype=bool)
    for row in range(1, 16, 2):
        mask[row, 1:18] = True
        if row + 1 < 16:
            mask[row + 1, 17 if (row // 2) % 2 == 0 else 1] = True
    mask[5:12, 7:12] = False
    mask[7, 7:12] = True

    _assert_parallel_identity(mask)


def test_double_null_confined_mask_and_flux_decisions_are_unchanged():
    """The demanding analytic domain keeps exact labels and deciding fluxes."""
    topology, psi, inside = _double_null_fixture()
    masks, state = topology.read(psi, 1, inside)
    psi_grid = topology.split_flux_map(psi)[0]
    confined = np.asarray(
        ((psi_grid - state.axis_flux) / (state.boundary_flux - state.axis_flux) < 1.0)
        & inside
    )
    shape = (
        int(topology.connectivity_height.size),
        int(topology.connectivity_radius.size),
    )
    _assert_parallel_identity(confined.reshape((shape[1], shape[0])).T)

    repeated_masks, repeated_state = topology.read(psi, 1, inside)
    np.testing.assert_array_equal(repeated_masks.label, masks.label)
    for name in ("wall_point_flux", "x_point_flux"):
        first = np.asarray(getattr(state, name))
        repeated = np.asarray(getattr(repeated_state, name))
        np.testing.assert_array_equal(first.view(np.uint64), repeated.view(np.uint64))


@pytest.mark.slow
def test_mast_keyframe_is_bit_identical_at_the_canonical_iteration_count():
    """The stride-two MAST keyframe matches the full fixed-point classifier."""
    mask = _mast_confined_mask(_MAST_ROW)
    _assert_parallel_identity(mask)
