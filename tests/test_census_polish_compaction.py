"""Equivalence pins for compact stationary-point census polishing."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nova.biot.null import Null2D
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.forward_operator import _FixedDesignNull2D
from nova.equilibrium.flux_surface_connectivity import (
    fit_tensor_spline,
    polish_census_stationary_points,
    polish_stationary_points,
)
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes


@pytest.fixture(scope="module", autouse=True)
def _double_precision():
    """Exercise the precision used by production forward maps."""
    configure_dtypes()


def _locator(radial: np.ndarray, vertical: np.ndarray) -> _FixedDesignNull2D:
    """Build the production census wrapper on a tensor-product lattice."""
    lattice = FluxLattice(radial, vertical)
    locator = Null2D.from_coordinates(
        lattice.coordinate,
        hex_stencil(lattice.shape),
        maxsize=30,
    )
    return _FixedDesignNull2D.from_locator(locator)


def _flat_field(values: np.ndarray) -> jnp.ndarray:
    """Flatten values in the forward-grid ordering."""
    return jnp.asarray(values.T.reshape(-1), dtype=jnp.float64)


def _all_origin_census(fixed: _FixedDesignNull2D, psi: jnp.ndarray) -> dict:
    """Evaluate every origin directly as an equivalence authority."""
    radial_count, vertical_count = fixed.spline_shape
    values = (
        jnp.asarray(psi, dtype=fixed.fit_dtype)
        .reshape((radial_count, vertical_count))
        .T
    )
    spline = fit_tensor_spline(fixed.spline_radial, fixed.spline_vertical, values)
    ring_coordinate = fixed.locator.coordinate[fixed.locator.stencil]
    ring_values = spline(
        ring_coordinate[..., 0].astype(fixed.fit_dtype),
        ring_coordinate[..., 1].astype(fixed.fit_dtype),
    )
    crossing_count = fixed.locator.crossing_count(ring_values)
    ring_mask = jnp.stack((crossing_count == 0, crossing_count == 4), axis=0)
    admitted = jnp.any(ring_mask, axis=0)
    seed = fixed.locator.physical_origin.astype(fixed.fit_dtype)
    polish = polish_stationary_points(spline, seed, admitted)
    displacement = jnp.linalg.norm(polish["position_rz"] - seed, axis=1)
    requested_displacement = jnp.where(admitted, displacement, jnp.nan)
    neighbours = ring_coordinate[:, 1:] - ring_coordinate[:, :1]
    cell_width = jnp.max(jnp.linalg.norm(neighbours, axis=-1), axis=1).astype(
        fixed.fit_dtype
    )
    within_cell = displacement < cell_width
    expected_hessian_type = jnp.asarray((1, -1), dtype=jnp.int8)[:, None]
    type_agrees = polish["hessian_type"][None, :] == expected_hessian_type
    polished_mask = ring_mask & polish["converged"][None, :] & within_cell[None, :]
    typed_mask = polished_mask & type_agrees
    domain_scale = jnp.hypot(
        fixed.spline_radial[-1] - fixed.spline_radial[0],
        fixed.spline_vertical[-1] - fixed.spline_vertical[0],
    )
    root_uncertainty = fixed._root_uncertainty(polish, cell_width, domain_scale)
    deduplicated = [
        fixed._deduplicate_type(
            polish["position_rz"], typed_mask[index], root_uncertainty
        )
        for index in range(2)
    ]
    representative_mask = jnp.stack([item[0] for item in deduplicated])
    multiplicity = jnp.stack([item[1] for item in deduplicated])
    representative_index = jnp.stack([item[2] for item in deduplicated])
    hessian = polish["hessian"]
    hessian_determinant = (
        hessian[..., 0, 0] * hessian[..., 1, 1]
        - hessian[..., 0, 1] * hessian[..., 1, 0]
    )
    extremum_kind = -jnp.sign(jnp.trace(hessian, axis1=-2, axis2=-1))
    kind = jnp.where(crossing_count == 4, 0.0, extremum_kind)
    candidates = jnp.column_stack(
        (
            polish["position_rz"].astype(jnp.float64),
            polish["value"].astype(jnp.float64),
            kind.astype(jnp.float64),
        )
    )
    source_origin = fixed.locator.stencil[:, 0].astype(jnp.int32)
    raw_ring_count = jnp.sum(ring_mask, axis=1, dtype=jnp.int32)
    polished_count = jnp.sum(polished_mask, axis=1, dtype=jnp.int32)
    typed_count = jnp.sum(typed_mask, axis=1, dtype=jnp.int32)
    same_root_count = jnp.sum(representative_mask, axis=1, dtype=jnp.int32)
    capacity = jnp.full(same_root_count.shape, fixed.locator.maxsize, dtype=jnp.int32)
    retained_count = jnp.minimum(same_root_count, capacity)
    retained_index = jnp.stack(
        [
            jnp.where(mask, size=fixed.locator.maxsize, fill_value=0)[0]
            for mask in representative_mask
        ]
    )
    retained_slot = jnp.arange(fixed.locator.maxsize)[None, :]
    retained_valid = retained_slot < retained_count[:, None]
    retained_multiplicity = jnp.take_along_axis(multiplicity, retained_index, axis=1)
    retained_multiplicity = jnp.where(retained_valid, retained_multiplicity, 0)

    def retain(values_to_gather, *, trailing=0, fill=0):
        gathered = values_to_gather[retained_index]
        valid_shape = retained_valid.shape + (1,) * trailing
        return jnp.where(
            retained_valid.reshape(valid_shape),
            gathered,
            jnp.asarray(fill, dtype=gathered.dtype),
        )

    return {
        "candidate": candidates,
        "ring_crossing_count": crossing_count,
        "ring_admitted_mask": ring_mask,
        "ring_resolution_limited": crossing_count == 2,
        "polish_converged": polish["converged"],
        "requested_displacement": requested_displacement,
        "cell_width": cell_width,
        "within_cell": within_cell,
        "polish_rejected": ring_mask
        & (~polish["converged"][None, :] | ~within_cell[None, :]),
        "hessian_type": polish["hessian_type"],
        "hessian_type_agrees": type_agrees,
        "typed_mask": typed_mask,
        "representative_mask": representative_mask,
        "representative_index": representative_index,
        "multiplicity": multiplicity,
        "source_origin_index": source_origin,
        "spline_gradient": polish["gradient"],
        "spline_gradient_norm": polish["gradient_norm"],
        "spline_hessian_determinant": hessian_determinant,
        "root_uncertainty": root_uncertainty,
        "raw_ring_count": raw_ring_count,
        "polished_count": polished_count,
        "typed_count": typed_count,
        "same_root_count": same_root_count,
        "retained_count": retained_count,
        "capacity": capacity,
        "overflow": same_root_count > capacity,
        "retained_candidate": retain(candidates, trailing=1),
        "retained_valid": retained_valid,
        "retained_representative_origin_index": retain(source_origin),
        "retained_representative_origin_rz": retain(
            fixed.locator.physical_origin, trailing=1
        ),
        "retained_multiplicity": retained_multiplicity,
        "retained_spline_gradient": retain(polish["gradient"], trailing=1),
        "retained_spline_gradient_norm": retain(polish["gradient_norm"]),
        "retained_spline_hessian_determinant": retain(hessian_determinant),
        "retained_requested_displacement": retain(displacement),
        "retained_root_uncertainty": retain(root_uncertainty),
        "spline_authored": jnp.asarray(True),
    }


def _stationary_fields(radial: np.ndarray, vertical: np.ndarray) -> list[np.ndarray]:
    """Return two smooth maps with different admitted-origin censuses."""
    mesh_r, mesh_z = np.meshgrid(radial, vertical)
    local_r = mesh_r - 1.2
    cubic_pair = local_r**3 / 3.0 - 0.83**2 * local_r + 0.7 * (mesh_z - 0.04) ** 2
    offset_pair = local_r**3 / 3.0 - 0.71**2 * local_r + 0.55 * (mesh_z + 0.07) ** 2
    return [cubic_pair, offset_pair]


# Telemetry keys the publication path adds beside the authority's own keys.
_EXTRA_STATUS_KEYS = {"candidate_count", "truncated", "census_slots_exhausted"}

# Values that are re-derived inside the routed branch and can therefore carry
# one last ULP from the branch-trace context on a marginally-conditioned field,
# even when the census itself runs eagerly.  The structural keys are exact.
_BRANCH_TRACE_KEYS = {
    "requested_displacement",
    "spline_hessian_determinant",
    "root_uncertainty",
    "retained_requested_displacement",
    "retained_spline_hessian_determinant",
    "retained_root_uncertainty",
}

# The raw polish-value arrays.  On a field whose stationary polish converges
# only marginally, the compiled census graph and the eager authority can carry
# a last-ULP difference in these values (the structural keys below are exact).
_POLISH_VALUE_KEYS = {
    "candidate",
    "requested_displacement",
    "spline_gradient",
    "spline_gradient_norm",
    "spline_hessian_determinant",
    "root_uncertainty",
    "retained_candidate",
    "retained_requested_displacement",
    "retained_spline_gradient",
    "retained_spline_gradient_norm",
    "retained_spline_hessian_determinant",
    "retained_root_uncertainty",
}


def _assert_bit_identical(actual, expected, extra_keys, name):
    """Require every authority key to match shape, dtype and storage bytes."""
    assert actual.keys() == expected.keys() | extra_keys, name
    for key, expected_value in expected.items():
        expected_array = np.asarray(expected_value)
        actual_array = np.asarray(actual[key])
        assert actual_array.shape == expected_array.shape, (name, key)
        assert actual_array.dtype == expected_array.dtype, (name, key)
        assert actual_array.tobytes() == expected_array.tobytes(), (name, key)


def test_compacted_polish_is_bit_identical_to_all_origin_census():
    """Admission, candidates, values, and every telemetry slot remain identical."""
    radial = np.linspace(0.2, 2.2, 17)
    vertical = np.linspace(-1.2, 1.2, 19)
    fixed = _locator(radial, vertical)

    for values in _stationary_fields(radial, vertical):
        field = _flat_field(values)
        expected = _all_origin_census(fixed, field)
        actual = fixed.candidate_table_status(field)

        _assert_bit_identical(actual, expected, _EXTRA_STATUS_KEYS, "analytic")
        np.testing.assert_array_equal(
            actual["candidate_count"], expected["same_root_count"]
        )
        np.testing.assert_array_equal(actual["truncated"], expected["overflow"])
        np.testing.assert_array_equal(actual["census_slots_exhausted"], False)


def test_polish_work_slots_are_bounded_by_published_capacity(monkeypatch):
    """Spline polish receives compact slots instead of every complete ring.

    The census routes both the fixed-slot and the all-origin lanes through
    ``lax.cond``, so an ordinary under-capacity field traces (and, eagerly,
    executes) both; the lane that is actually taken is the bounded one.
    """
    radial = np.linspace(0.2, 2.2, 17)
    vertical = np.linspace(-1.2, 1.2, 19)
    fixed = _locator(radial, vertical)
    observed = []

    from nova.equilibrium import forward_operator

    direct_polish = forward_operator.polish_stationary_points

    def record_slots(spline, seed, valid):
        observed.append(seed.shape[0])
        return direct_polish(spline, seed, valid)

    monkeypatch.setattr(forward_operator, "polish_stationary_points", record_slots)
    fixed._structured_census(_flat_field(_stationary_fields(radial, vertical)[0]))

    assert sorted(observed) == sorted(
        [2 * fixed.locator.maxsize, fixed.locator.stencil.shape[0]]
    )
    assert min(observed) == 2 * fixed.locator.maxsize
    assert min(observed) < fixed.locator.stencil.shape[0]


def _noise_field(radial: np.ndarray, vertical: np.ndarray, seed: int) -> jnp.ndarray:
    """One seeded white-noise map admitting many independent stationary rings."""
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 1e-3, size=(radial.size, vertical.size))
    return _flat_field(values)


def _admitted_count(census) -> int:
    """Return how many ring origins any type admits."""
    admitted = np.any(np.asarray(census["ring_admitted_mask"]), axis=0)
    return int(admitted.sum())


def test_over_capacity_census_publishes_the_complete_all_origin_census():
    """Admitted origins beyond the work slots are never silently dropped."""
    radial = np.linspace(0.2, 2.2, 17)
    vertical = np.linspace(-1.2, 1.2, 19)
    fixed = _locator(radial, vertical)
    field = _noise_field(radial, vertical, seed=0)
    expected = _all_origin_census(fixed, field)
    actual = fixed.candidate_table_status(field)

    assert _admitted_count(actual) > 2 * fixed.locator.maxsize
    assert bool(actual["census_slots_exhausted"])
    np.testing.assert_array_equal(actual["truncated"], expected["overflow"])
    np.testing.assert_array_equal(
        actual["candidate_count"], expected["same_root_count"]
    )

    # The routing logic itself: the same over-capacity branch, executed outside
    # the compiled publication wrapper, is byte-identical to the all-origin
    # authority on every key except the branch-trace ULP set above.
    routing = fixed._structured_census(field)
    assert bool(routing["census_slots_exhausted"])
    for key, expected_value in expected.items():
        expected_array = np.asarray(expected_value)
        actual_array = np.asarray(routing[key])
        assert actual_array.shape == expected_array.shape, key
        assert actual_array.dtype == expected_array.dtype, key
        if key in _BRANCH_TRACE_KEYS:
            np.testing.assert_allclose(
                actual_array, expected_array, rtol=0.0, atol=1.0e-9, err_msg=key
            )
        else:
            assert actual_array.tobytes() == expected_array.tobytes(), key

    # The compiled publication path is byte-identical on every structural and
    # telemetry key; its raw polish values reproduce the authority within the
    # compiled-polish precision (see the documented key set above).
    for key, expected_value in expected.items():
        if key in _POLISH_VALUE_KEYS:
            continue
        expected_array = np.asarray(expected_value)
        actual_array = np.asarray(actual[key])
        assert actual_array.shape == expected_array.shape, key
        assert actual_array.dtype == expected_array.dtype, key
        assert actual_array.tobytes() == expected_array.tobytes(), key
    for key in _POLISH_VALUE_KEYS:
        np.testing.assert_allclose(
            np.asarray(actual[key]),
            np.asarray(expected[key]),
            rtol=0.0,
            atol=4.0e-8,
            err_msg=f"{key} must reproduce the all-origin value at compiled precision",
        )


def test_over_capacity_root_uncertainty_reconstruction_is_exact():
    """Admitted origins carry the direct uncertainty, not an offset rebuild."""
    radial = np.linspace(0.2, 2.2, 17)
    vertical = np.linspace(-1.2, 1.2, 19)
    fixed = _locator(radial, vertical)
    field = _noise_field(radial, vertical, seed=1)
    expected = _all_origin_census(fixed, field)
    routing = fixed._structured_census(field)

    assert _admitted_count(routing) > 2 * fixed.locator.maxsize
    assert bool(routing["census_slots_exhausted"])
    expected_uncertainty = np.asarray(expected["root_uncertainty"])
    actual_uncertainty = np.asarray(routing["root_uncertainty"])
    np.testing.assert_allclose(
        actual_uncertainty,
        expected_uncertainty,
        rtol=0.0,
        atol=1.0e-9,
        err_msg="root uncertainty must carry the direct all-origin value",
    )
    np.testing.assert_allclose(
        np.asarray(routing["retained_root_uncertainty"]),
        np.asarray(expected["retained_root_uncertainty"]),
        rtol=0.0,
        atol=1.0e-9,
        err_msg="retained root uncertainty must carry the direct all-origin value",
    )


def test_polish_branches_publish_identical_converged_on_stationary_input():
    """Retain and refine lanes give every slot the same acceptance verdict."""
    radial = np.linspace(-1.0, 1.0, 33)
    vertical = np.linspace(-1.0, 1.0, 33)
    radial_grid, vertical_grid = np.meshgrid(radial, vertical)
    local_r = radial_grid
    # A cubic radial term with a quadratic vertical term: a minimum at
    # (0.53, 0.0) and a saddle at (-0.53, 0.0), both exactly stationary.
    values = local_r**3 / 3.0 - 0.53**2 * local_r + 0.7 * vertical_grid**2

    def flux(position):
        r, z = position
        return r**3 / 3.0 - 0.53**2 * r + 0.7 * z**2

    def row(position):
        point = jnp.asarray(position, dtype=jnp.float64)
        return jnp.r_[point, flux(position), 0.0]

    extremum_seed = row((0.53, 0.0))
    saddle_seed = row((-0.53, 0.0))
    interface = jnp.asarray(flux((-0.53, 0.0)), dtype=jnp.float64)

    def publish(offset):
        return polish_census_stationary_points(
            values,
            radial,
            vertical,
            interface,
            jnp.asarray(-1.0),
            extremum_seed.at[0].add(jnp.asarray(offset)),
            saddle_seed,
        )

    # Both seeds exactly stationary: the retain lane runs and every published
    # convergence qualification holds.
    _extremum, _saddle, retain_receipt = publish(0.0)
    np.testing.assert_array_equal(retain_receipt["seed_stationary"], True)
    converged = np.asarray(retain_receipt["converged"])
    np.testing.assert_array_equal(converged, True)

    # A sub-convergence seed displacement forces the refine lane; its verdict
    # is the same accepted set as the retain lane published.
    _extremum, _saddle, refine_receipt = publish(0.02)
    assert not bool(refine_receipt["seed_stationary"][0])
    np.testing.assert_array_equal(np.asarray(refine_receipt["converged"]), converged)
