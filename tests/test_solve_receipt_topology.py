"""Terminal labelled-flux receipt on a cached MAST equilibrium arm."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from benchmarks import efit_forward_parity_slice as parity
from benchmarks import mast_response_carrier_warm as carrier
from benchmarks.diiid_forward_gs_match import _margin_graded_newton_krylov
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium.forward import ForwardDomainLabel
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE


pytestmark = pytest.mark.slow

CONVERGING_SHOT = 21985


@pytest.fixture(scope="module")
def converged_mast_receipt():
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
    mapped = profile.flux_map(
        requested_class=TopologyClass.DIVERTED,
        target_current=target_current,
    )
    solved = _margin_graded_newton_krylov(
        mapped,
        profile.operator.topology_margin,
        case["state"],
        newton_steps=parity.NEWTON_STEPS,
        gmres_iterations=parity.GMRES_ITERATIONS,
    )
    solved.state.block_until_ready()
    assert float(solved.residual) <= parity.FIXED_POINT_CRITERION
    return profile, profile.observe(solved.state, target_current=target_current)


def test_terminal_receipt_matches_an_independent_topology_read(
    converged_mast_receipt,
) -> None:
    profile, receipt = converged_mast_receipt
    labelled = receipt.labelled_flux
    assert labelled is not None
    masks, topology = profile.operator.read(receipt.flux)

    np.testing.assert_array_equal(
        np.asarray(labelled.psi),
        np.asarray(receipt.flux)[: profile.lattice.node_count],
    )
    np.testing.assert_array_equal(
        np.asarray(labelled.psi_norm), np.asarray(masks.psi_norm)
    )
    np.testing.assert_array_equal(
        np.asarray(labelled.o_point), np.asarray(topology.axis)
    )
    np.testing.assert_array_equal(
        np.asarray(labelled.primary_x_point), np.asarray(topology.x_point)
    )
    assert labelled.psi.shape == labelled.psi_norm.shape == (33 * 33,)
    assert labelled.domain_label.shape == (33 * 33,)
    assert 0 <= int(labelled.lcfs_vertex_count) <= len(labelled.lcfs)
    assert np.all(
        np.isfinite(np.asarray(labelled.lcfs)[: int(labelled.lcfs_vertex_count)])
    )
    assert np.all(np.isfinite(np.asarray(labelled.strike_points)))


def test_terminal_receipt_uses_only_resolvable_cell_domains(
    converged_mast_receipt,
) -> None:
    _profile, receipt = converged_mast_receipt
    labelled = receipt.labelled_flux
    assert labelled is not None

    assert set(np.unique(np.asarray(labelled.domain_label))) == {
        int(ForwardDomainLabel.CORE),
        int(ForwardDomainLabel.PRIVATE_FLUX),
        int(ForwardDomainLabel.WALL_SHADOW),
        int(ForwardDomainLabel.OPEN),
    }
    assert not hasattr(ForwardDomainLabel, "COMMON_SOL")
