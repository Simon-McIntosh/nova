"""Controls for fragment booking and the unchanged seven-cell curve method."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import os
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from scripts.prototypes.membership.booking import classify, reference_control
from scripts.prototypes.membership.hybrid import membership

assert jax.config.jax_enable_x64 is True


def test_collection_polygon_integral_is_not_skipped():
    reference_control()


def test_curved_booking_receipt_reproduces_first_hybrid():
    directory = Path(os.environ["MEMBERSHIP_REPLAYS"])
    with (directory / "diverted-2000-booked-replay.pkl").open("rb") as stream:
        jets, indices, mesh, geometry, fixed, bell, sigma, diverted = pickle.load(
            stream
        )
    level = fixed.jet.value[-1]
    args = jets, indices, mesh, geometry, fixed, level, sigma
    original = jax.jit(membership)(*args, diverted)
    result = jax.jit(lambda *values: classify(*values, curved=True, diverted=True))(
        *args
    )
    np.testing.assert_allclose(result.membership, original[0], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(result.fragment_selected, original[4])
    np.testing.assert_array_equal(result.normal_form_cells, original[3])
    assert int(jnp.sum(result.normal_form_cells)) == 7
    assert result.valid
    stored = np.load(directory.parent / "rows/diverted-2000-panel.npz")
    np.testing.assert_allclose(
        result.membership, stored["membership"], rtol=0, atol=1e-12
    )
    error = np.abs(np.asarray(result.membership) - stored["reference"])
    error *= np.asarray(geometry.full_area) / np.median(geometry.full_area)
    measured = float(np.max(error[stored["near"]]))
    print("FIRST_HYBRID_REPRODUCTION", measured, flush=True)
    assert abs(measured - 0.00043348956278423535) < 1e-12


def test_limited_methods_have_identical_membership():
    directory = Path(os.environ["MEMBERSHIP_REPLAYS"])
    with (directory / "limited-2000-booked-replay.pkl").open("rb") as stream:
        jets, indices, mesh, geometry, fixed, bell, sigma, _ = pickle.load(stream)
    args = jets, indices, mesh, geometry, fixed, bell.boundary_flux, sigma
    results = [
        jax.jit(lambda *values: classify(*values, curved=curved, diverted=False))(*args)
        for curved in (False, True)
    ]
    np.testing.assert_array_equal(results[0].membership, results[1].membership)
    assert all(bool(r.valid) for r in results)
