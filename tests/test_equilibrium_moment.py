"""Analytic contract for the current-moment boundary reconstruction.

No machine data is needed: a tiny synthetic cell set with a well-conditioned
random sensor coupling carries everything the fit reads. The physics claim under
test is that a current confined to the low-order moment span is recovered
near-exactly from its own noise-free external-field signature, that the
Rogowski anchor pins the total current, and that the over-fit gate rejects a
stage which swings the boundary.
"""

import numpy as np
import pytest
from types import SimpleNamespace

from nova.equilibrium.measurement import Magnetics, SliceMeasurement
from nova.equilibrium.moment import (
    CurrentCells,
    MomentConfig,
    MomentOrder,
    ReconstructMoment,
    UnsupportedSlice,
    build_moment_basis,
    limiter_radial_extent,
    moment_terms,
    ring_shift_rms,
    sensor_coupling,
)


def synthetic_read(n_side=7, n_sensor=24, r0=0.9, seed=0):
    """Return a reconstruction over a square cell block and a random coupling."""
    rng = np.random.default_rng(seed)
    r, z = np.meshgrid(
        np.linspace(r0 - 0.4, r0 + 0.4, n_side),
        np.linspace(-0.5, 0.5, n_side),
    )
    cells = CurrentCells(r.ravel(), z.ravel(), 0.05, 0.05)
    coupling = rng.standard_normal((n_sensor, cells.number)) * 1e-6
    read = ReconstructMoment(cells, sensor_coupling=coupling, major_radius=r0)
    return read, rng


def measurement(coupling, cell_current, plasma_current=None, vacuum=None, mask=None):
    """Return the noise-free measurement a given cell current would produce."""
    n_sensor = coupling.shape[0]
    vacuum = np.zeros(n_sensor) if vacuum is None else vacuum
    return SliceMeasurement(
        measured=vacuum + coupling @ cell_current,
        vacuum=vacuum,
        mask=np.ones(n_sensor, bool) if mask is None else mask,
        scale=np.full(n_sensor, 1e-4),
        plasma_current=(
            float(np.sum(cell_current)) if plasma_current is None else plasma_current
        ),
    )


# --- the order ladder -------------------------------------------------------


def test_moment_order_ladder_is_a_nested_monomial_family():
    """Each rung adds the next monomial degree and keeps the lower rungs."""
    assert set(moment_terms(MomentOrder.CENTROID)) == {(0, 0), (1, 0), (0, 1)}
    assert moment_terms(MomentOrder.CENTROID)[0] == (0, 0)  # monopole first
    assert len(moment_terms(MomentOrder.CENTROID)) == 3
    assert len(moment_terms(MomentOrder.QUADRUPOLE)) == 6
    assert len(moment_terms(MomentOrder.OCTUPOLE)) == 10
    centroid = moment_terms(MomentOrder.CENTROID)
    assert moment_terms(MomentOrder.QUADRUPOLE)[: len(centroid)] == centroid


def test_order_below_the_centroid_rung_is_rejected():
    with pytest.raises(ValueError):
        moment_terms(0)


def test_basis_is_candidate_masked_with_a_monopole_first_column():
    read, _ = synthetic_read()
    candidate = np.ones(read.cells.number)
    candidate[: read.cells.number // 2] = 0.0
    basis, labels, scale = build_moment_basis(
        read.cells.r, read.cells.z, candidate, 0.9, order=MomentOrder.QUADRUPOLE
    )
    assert basis.shape == (read.cells.number, 6)
    assert labels[0] == "1"
    assert scale > 0
    np.testing.assert_allclose(basis[:, 0], candidate)
    assert np.all(basis[candidate == 0.0, :] == 0.0)


def test_shape_moments_carry_no_net_current():
    """Zero-sum higher moments land the whole of Ip on the monopole."""
    read, _ = synthetic_read()
    candidate = np.ones(read.cells.number)
    basis, _, _ = build_moment_basis(
        read.cells.r, read.cells.z, candidate, 0.9, order=MomentOrder.OCTUPOLE
    )
    assert np.allclose(basis[:, 1:].sum(axis=0), 0.0, atol=1e-9)


# --- the whitened fit -------------------------------------------------------


def test_recovers_a_current_inside_the_moment_span():
    """A low-order moment current is recovered from its noise-free signature."""
    read, rng = synthetic_read()
    basis, _, _ = build_moment_basis(
        read.cells.r,
        read.cells.z,
        read.cells.candidate,
        read.major_radius,
        order=MomentOrder.OCTUPOLE,
    )
    coefficients = rng.standard_normal(basis.shape[1]) * 5.0e4
    cell_current = basis @ coefficients
    vacuum = rng.standard_normal(read.sensor_coupling.shape[0]) * 1e-3
    slice_data = measurement(read.sensor_coupling, cell_current, vacuum=vacuum)

    inversion = read.fit(
        slice_data, MomentConfig(order=MomentOrder.OCTUPOLE, ridge=1e-12)
    )
    assert inversion.misfit < 1e-6
    np.testing.assert_allclose(
        inversion.cell_current, cell_current, rtol=1e-4, atol=1e-2
    )
    assert inversion.plasma_current_error < 1e-6


def test_the_rogowski_anchor_pins_the_total_current():
    """Out of span, the hard anchor still holds the total current exactly."""
    read, _ = synthetic_read(seed=1)
    blob = np.exp(
        -(((read.cells.r - read.major_radius) / 0.25) ** 2 + (read.cells.z / 0.3) ** 2)
    )
    cell_current = blob / blob.sum() * 6.0e5
    slice_data = measurement(read.sensor_coupling, cell_current, plasma_current=6.0e5)

    anchored = read.fit(slice_data, MomentConfig(order=MomentOrder.OCTUPOLE))
    assert anchored.plasma_current_error < 1e-9
    assert abs(anchored.centroid_r - read.major_radius) < 0.15
    assert abs(anchored.centroid_z) < 0.15

    free = read.fit(
        slice_data, MomentConfig(order=MomentOrder.OCTUPOLE, ip_anchor=False)
    )
    assert free.plasma_current_error >= anchored.plasma_current_error


def test_untrusted_rows_do_not_enter_the_fit():
    """Corrupting masked-out sensor rows must not move the coefficients."""
    read, rng = synthetic_read(seed=2)
    basis, _, _ = build_moment_basis(
        read.cells.r,
        read.cells.z,
        read.cells.candidate,
        read.major_radius,
        order=MomentOrder.QUADRUPOLE,
    )
    cell_current = basis @ (rng.standard_normal(basis.shape[1]) * 3.0e4)
    mask = np.ones(read.sensor_coupling.shape[0], bool)
    mask[::4] = False

    clean = measurement(read.sensor_coupling, cell_current, mask=mask)
    corrupted = SliceMeasurement(
        measured=clean.measured + np.where(mask, 0.0, 1e3),
        vacuum=clean.vacuum,
        mask=mask,
        scale=clean.scale,
        plasma_current=clean.plasma_current,
    )
    config = MomentConfig(order=MomentOrder.QUADRUPOLE)
    np.testing.assert_allclose(
        read.fit(clean, config).coefficients,
        read.fit(corrupted, config).coefficients,
        rtol=1e-8,
        atol=1e-6,
    )


def test_higher_rungs_are_never_worse_on_a_noise_free_signature():
    """Climbing the ladder cannot raise the misfit of an exact signature."""
    read, rng = synthetic_read(seed=5)
    blob = np.exp(-(((read.cells.r - read.major_radius) / 0.3) ** 2))
    cell_current = blob / blob.sum() * 4.0e5
    slice_data = measurement(read.sensor_coupling, cell_current)
    misfit = [
        read.fit(slice_data, MomentConfig(order=order)).misfit for order in MomentOrder
    ]
    assert misfit == sorted(misfit, reverse=True)


# --- the self-sized centroid seed and the over-fit gate ---------------------


def test_uniform_disc_seed_spreads_the_plasma_current_evenly():
    read, _ = synthetic_read()
    cell_current = read.uniform_disc(0.9, 0.0, 0.3, 1.2e5)
    inside = cell_current != 0.0
    assert inside.sum() > 0
    assert np.isclose(cell_current.sum(), 1.2e5)
    assert np.allclose(cell_current[inside], cell_current[inside][0])
    assert np.all(np.hypot(read.cells.r - 0.9, read.cells.z)[inside] < 0.3)


def test_unsupported_seed_disc_carries_its_support_counts():
    read, _ = synthetic_read()

    with pytest.raises(UnsupportedSlice) as caught:
        read.uniform_disc(0.9, 0.0, 0.01, 1.2e5)

    assert caught.value.condition == "seed-disc-insufficient-supported-cells"
    assert caught.value.details["supported_cell_count"] == 1
    assert caught.value.details["minimum_cell_count"] == 5


def test_centroid_outside_limiter_carries_position_and_radial_support():
    read, _ = synthetic_read()
    read.grid = SimpleNamespace(
        limiter_r=np.array([0.2, 1.8, 1.8, 0.2, 0.2]),
        limiter_z=np.array([-1.5, -1.5, 1.5, 1.5, -1.5]),
    )

    with pytest.raises(UnsupportedSlice) as caught:
        read.self_sized_seed(object(), (2.0, 0.0))

    assert caught.value.condition == "current-centroid-outside-limiter"
    assert caught.value.details == {
        "centroid_r_m": 2.0,
        "centroid_z_m": 0.0,
        "limiter_inboard_r_m": 0.2,
        "limiter_outboard_r_m": 1.8,
        "supported_minor_distance_m": pytest.approx(-0.2),
    }


def test_flood_seed_outside_vessel_carries_position_and_grid_index():
    read, _ = synthetic_read()
    read.grid = SimpleNamespace(
        rg=np.array([0.5, 1.0, 1.5]),
        zg=np.array([-0.5, 0.0, 0.5]),
        inside_limiter=np.array(
            [[False, False, False], [False, True, False], [False, False, False]]
        ),
    )

    with pytest.raises(UnsupportedSlice) as caught:
        read.push_out(np.zeros((3, 3)), (1.5, 0.5))

    assert caught.value.condition == "flood-seed-outside-vessel"
    assert caught.value.details == {
        "centroid_r_m": 1.5,
        "centroid_z_m": 0.5,
        "grid_row": 2,
        "grid_column": 2,
        "grid_r_m": 1.5,
        "grid_z_m": 0.5,
    }


def test_limiter_extent_of_a_rectangular_vessel():
    limiter_r = np.array([0.2, 1.8, 1.8, 0.2, 0.2])
    limiter_z = np.array([-1.5, -1.5, 1.5, 1.5, -1.5])
    inboard, outboard = limiter_radial_extent(limiter_r, limiter_z, 0.0)
    assert abs(inboard - 0.2) < 1e-9
    assert abs(outboard - 1.8) < 1e-9
    # above the vessel the polygon bounding radii are the fallback
    inboard, outboard = limiter_radial_extent(limiter_r, limiter_z, 5.0)
    assert inboard <= outboard


def circle(r0, z0, radius, n=64):
    angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)])


def test_ring_shift_vanishes_for_identical_rings():
    ring = circle(0.9, 0.0, 0.5)
    assert ring_shift_rms(ring, ring.copy(), (0.9, 0.0)) < 1e-12


def test_ring_shift_measures_radial_expansion():
    shift = ring_shift_rms(circle(0.9, 0, 0.50), circle(0.9, 0, 0.60), (0.9, 0))
    assert abs(shift - 0.10) < 5e-3


def test_ring_shift_is_infinite_when_a_ring_is_missing():
    ring = circle(0.9, 0.0, 0.5)
    assert ring_shift_rms(ring, None, (0.9, 0.0)) == float("inf")
    assert ring_shift_rms(None, ring, (0.9, 0.0)) == float("inf")


def test_the_gate_rejects_a_boundary_swinging_stage():
    """A stage moving the boundary by more than the gate fraction is rejected."""
    config = MomentConfig()
    radius = 0.6
    small = ring_shift_rms(circle(0.9, 0, 0.50), circle(0.9, 0, 0.54), (0.9, 0))
    large = ring_shift_rms(circle(0.9, 0, 0.50), circle(0.9, 0, 0.65), (0.9, 0))
    assert small / radius < config.gate_shift_fraction
    assert large / radius > config.gate_shift_fraction


# --- the nova.biot coupling -------------------------------------------------


def test_sensor_coupling_projects_onto_the_probe_kind():
    """Flux loops read psi; field probes read the field along their angle."""
    cells = CurrentCells(np.array([1.0]), np.array([0.0]), 0.05, 0.05)
    magnetics = Magnetics(
        r=np.array([1.6, 1.6, 1.6]),
        z=np.array([0.4, 0.4, 0.4]),
        angle=np.array([0.0, 90.0, 0.0]),
        flux_loop=np.array([False, False, True]),
    )
    coupling = sensor_coupling(cells, magnetics)
    from nova.biot.greens import hybrid_greens

    psi, br, bz = hybrid_greens(magnetics.r, magnetics.z, 1.0, 0.0, 0.05, 0.05)
    assert coupling.shape == (3, 1)
    assert np.isclose(coupling[0, 0], br[0])
    assert np.isclose(coupling[1, 0], bz[1])
    assert np.isclose(coupling[2, 0], psi[2])


def test_sensor_coupling_rows_follow_the_magnetics_order():
    cells = CurrentCells(np.array([1.0, 1.2]), np.array([0.0, 0.1]), 0.05, 0.05)
    magnetics = Magnetics(
        r=np.array([0.3, 1.4]),
        z=np.array([0.1, -0.2]),
        angle=np.array([45.0, 0.0]),
        flux_loop=np.array([False, True]),
    )
    coupling = sensor_coupling(cells, magnetics)
    assert coupling.shape == (2, 2)
    assert magnetics.number == 2
