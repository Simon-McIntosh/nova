from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.efit_parity_inductance_partition import (
    MOMENT_RECEIPT,
    _plot_partition,
    _cell_partition,
    measure,
)
from nova.jax.config import configure_dtypes


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
    assert table["banked_confined_core"]["cell_current_a"] == pytest.approx(
        banked["confined_core"], abs=2.0e-9
    )
    divergence = receipt["core_support_divergence"]
    assert divergence["live_core_cell_count"] == live["cell_count"]
    assert divergence["banked_core_current_a"] == pytest.approx(
        banked["confined_core"], abs=2.0e-9
    )
    for name in ("inside_closed_branch_outside_core", "outside_closed_branch"):
        assert table[name]["status"] == "undefined_open_boundary"
        assert table[name]["cell_count"] is None
        assert table[name]["cell_current_a"] is None
    closure = receipt["current_closure"]
    assert closure["direct_total_current_a"] == pytest.approx(
        banked["all_domain_constrained"], abs=2.0e-9
    )
    out_of_core_current_from_banked_total = (
        banked["all_domain_constrained"] - live["cell_current_a"]
    )
    assert closure["direct_out_of_core_current_a"] == pytest.approx(
        out_of_core_current_from_banked_total, abs=4.0e-9
    )
    out_of_core_support = ~fields["core"]
    independently_summed_out_of_core = float(
        np.sum(fields["cell_current"][out_of_core_support])
    )
    # Compare cellwise out-of-core current with banked total minus live core.
    assert independently_summed_out_of_core == pytest.approx(
        out_of_core_current_from_banked_total, abs=4.0e-9
    )
    assert closure["banked_out_of_core_current_a"] == pytest.approx(
        banked["outside_confined_core_a"], abs=2.0e-9
    )
    assert closure["summed_partition_current_a"] is None
    assert closure["closure_residual_a"] is None
    assert fields["partition"] is None


def test_outside_contour_core_centroid_is_counted_once():
    configure_dtypes()
    profile = SimpleNamespace(
        lattice=SimpleNamespace(
            coordinate=np.array([[5.0, 5.0], [1.0, 1.0]], dtype=np.float64)
        )
    )
    equilibrium = SimpleNamespace(
        domains=SimpleNamespace(core=np.array([True, False])),
        cell_current=np.array([7.0, 11.0], dtype=np.float64),
    )
    contour = np.array(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
        dtype=np.float64,
    )

    partition = _cell_partition(profile, equilibrium, contour, solved_branch=False)
    core = partition["core"]
    interior_noncore = partition["interior_noncore"]
    exterior = partition["exterior"]
    assert not np.any(core & interior_noncore)
    assert not np.any(core & exterior)
    assert not np.any(interior_noncore & exterior)
    assert np.array_equal(core | interior_noncore | exterior, np.array([True, True]))
    assert partition["records"]["confined_core"]["cell_current_a"] == pytest.approx(7.0)
    assert partition["records"]["inside_closed_branch_outside_core"][
        "cell_current_a"
    ] == pytest.approx(11.0)
    assert partition["records"]["outside_closed_branch"]["cell_current_a"] == 0.0
    partitioned_out_of_core = (
        partition["records"]["inside_closed_branch_outside_core"]["cell_current_a"]
        + partition["records"]["outside_closed_branch"]["cell_current_a"]
    )
    assert partitioned_out_of_core == pytest.approx(
        np.sum(equilibrium.cell_current[~equilibrium.domains.core]), abs=1.0e-12
    )
    assert partition["summed_current_a"] == pytest.approx(
        np.sum(equilibrium.cell_current), abs=1.0e-12
    )


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
