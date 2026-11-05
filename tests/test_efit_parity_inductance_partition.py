from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.efit_parity_inductance_partition import (
    MOMENT_RECEIPT,
    _plot_partition,
    measure,
)


def _current_supports():
    return json.loads(MOMENT_RECEIPT.read_text())["current_supports"]


@pytest.fixture(scope="module")
def result():
    return measure()


def test_banked_terminal_observation_is_the_only_execution(result):
    receipt, _fields = result
    execution = receipt["execution_contract"]
    assert execution["nonlinear_solve_calls"] == 0
    assert execution["equilibrium_observations_from_serialised_state"] == 1
    integrity = receipt["protected_banked_artifacts"]
    assert integrity["verified_digest_count"] == 23
    assert integrity["all_digests_match"] is True


def test_open_separatrix_has_no_enclosed_geometry(result):
    receipt, fields = result
    assert receipt["boundary_status"]["status"] == "open"
    assert receipt["separatrix"]["status"] == "open"
    assert receipt["separatrix"]["poloidal_area_m2"] is None
    assert receipt["separatrix"]["exact_solid_of_revolution_m3"] is None
    contour = fields["contour"]
    assert len(contour) >= 4
    assert np.linalg.norm(contour[0] - contour[-1]) > 0.0


def test_direct_current_closes_without_inventing_branch_partition(result):
    receipt, fields = result
    banked = _current_supports()
    table = receipt["current_partition_table"]
    live = table["confined_core"]
    # The two supports disagree in cell count and in current. Both facts are
    # reported against the independently banked current table; neither pair of
    # numbers is pinned as the expected outcome.
    assert live["cell_count"] != table["banked_confined_core"]["cell_count"]
    assert table["banked_confined_core"]["cell_current_a"] == pytest.approx(
        banked["confined_core"], abs=2.0e-9
    )
    assert live["cell_current_a"] > table["banked_confined_core"]["cell_current_a"]
    divergence = receipt["core_support_divergence"]
    assert divergence["cell_counts_differ"] is True
    assert divergence["live_core_cell_count"] == live["cell_count"]
    assert divergence["banked_core_current_a"] == pytest.approx(
        banked["confined_core"], abs=2.0e-9
    )
    assert divergence["current_agrees"] is False
    for name in ("inside_closed_branch_outside_core", "outside_closed_branch"):
        assert table[name]["status"] == "undefined_open_boundary"
        assert table[name]["cell_count"] is None
        assert table[name]["cell_current_a"] is None
    closure = receipt["current_closure"]
    # The totals are checked against the independently banked current table
    # rather than against the benchmark's own subtraction.
    assert closure["direct_total_current_a"] == pytest.approx(
        banked["all_domain_constrained"], abs=2.0e-9
    )
    assert table["confined_core"]["cell_current_a"] + closure[
        "direct_out_of_core_current_a"
    ] == pytest.approx(banked["all_domain_constrained"], abs=4.0e-9)
    assert closure["banked_out_of_core_current_a"] == pytest.approx(
        banked["outside_confined_core_a"], abs=2.0e-9
    )
    assert closure["summed_partition_current_a"] is None
    assert closure["closure_residual_a"] is None
    assert fields["partition"] is None


def test_boundary_enclosed_rescore_is_undefined(result):
    receipt, _fields = result
    rescore = receipt["boundary_enclosed_moment_rescore"]
    assert rescore["status"] == "undefined_open_boundary"
    assert rescore["reason"] == receipt["boundary_status"]["reason"]


def test_open_contour_disallows_masking_verdict_and_plots_without_closure(
    result, tmp_path
):
    receipt, fields = result
    discriminator = receipt["discriminator_result"]
    assert discriminator["status"] == "undefined_open_boundary"
    assert discriminator["current_location_supported_reading"] is None
    assert discriminator["inside_share_of_out_of_core_current"] is None
    assert discriminator["internal_inductance_mechanism_verdict"] is None
    figure = Path(tmp_path) / "partition.png"
    _plot_partition(fields["contour"], fields["partition"], figure)
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
