"""Exact rectangular flux image carried by a production solve receipt."""

from __future__ import annotations

import os
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import efit_forward_parity_slice as parity
from benchmarks import mast_response_carrier_warm as carrier
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from nova.imas.mast_solve_inputs import SHOT_STORE


pytestmark = pytest.mark.slow

CONVERGING_SHOT = 22086


@pytest.fixture(scope="module")
def mast_receipt():
    """Read the persisted state through the frozen MAST response carrier."""
    selected = next(
        row
        for row in parity.select_slices_by_shot(parity.DECOMPOSITION_BANK)
        if int(row[0]["shot"]) == CONVERGING_SHOT
    )
    case, context = parity._mast_case_from_selection(SHOT_STORE, *selected)
    response_cache, _cache_receipt = _persisted_response_cache(
        carrier.DEFAULT_CARRIER, carrier.DEFAULT_RECEIPT
    )
    _case, profile, _policy = parity._passive_inclusive_case(
        case, context, response_cache
    )
    target_current = abs(float(case["reference"]["plasma_current_a"]))
    return (
        profile,
        target_current,
        profile.observe(case["state"], target_current=target_current),
    )


def _assert_tree_identical(left, right) -> None:
    """Require every corresponding receipt leaf to retain identical bytes."""
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for first, second in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_raster_receipt_is_the_exact_current_image(mast_receipt) -> None:
    """The persisted grid is evaluated directly and carries its full topology."""
    profile, _target_current, receipt = mast_receipt
    raster = receipt.raster_flux
    labelled = receipt.labelled_flux
    assert raster is not None
    assert labelled is not None

    np.testing.assert_array_equal(raster.radius, profile.lattice.radius)
    np.testing.assert_array_equal(raster.height, profile.lattice.height)
    np.testing.assert_array_equal(raster.shape, np.asarray([33, 33]))
    assert raster.psi.shape == raster.psi_norm.shape == (33 * 33,)
    assert raster.domain_label.shape == (33 * 33,)
    np.testing.assert_array_equal(raster.domain_label, labelled.domain_label)

    zero = jnp.zeros_like(receipt.cell_current)
    direct = profile.operator.raster_image(
        CellCurrentMoments(receipt.cell_current, zero, zero)
    )
    np.testing.assert_array_equal(raster.psi, direct)
    np.testing.assert_array_equal(
        raster.psi_norm,
        (direct - receipt.topology.axis_flux) / receipt.topology.flux_span,
    )
    count = int(raster.separatrix_vertex_count)
    assert count > 0
    assert count <= len(raster.separatrix)
    assert np.all(np.isfinite(np.asarray(raster.separatrix)[:count]))


def test_raster_postprocessing_keeps_existing_receipt_fields_bit_identical(
    mast_receipt,
) -> None:
    """Adding the raster leaf cannot perturb any previously published value."""
    profile, target_current, receipt = mast_receipt
    with patch.object(ForwardProfile, "_raster_flux", return_value=None):
        without_raster = profile.observe(
            receipt.flux,
            target_current=target_current,
        )
    _assert_tree_identical(receipt._replace(raster_flux=None), without_raster)
