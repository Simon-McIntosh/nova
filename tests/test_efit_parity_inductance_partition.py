from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks.efit_parity_inductance_partition import (
    EXPECTED_MATCHED_SUPPORT_BETA_DEVIATION,
    EXPECTED_MATCHED_SUPPORT_LI_DEVIATION,
    EXPECTED_OUT_OF_CORE_CURRENT_A,
    EXPECTED_SOLVED_BOUNDARY_AREA_M2,
    EXPECTED_SOLVED_BOUNDARY_VOLUME_M3,
    EXPECTED_TOTAL_CURRENT_A,
    _plot_partition,
    measure,
)


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


def test_closed_branch_reproduces_the_banked_geometry(result):
    receipt, _fields = result
    separatrix = receipt["separatrix"]
    assert separatrix["poloidal_area_m2"] == pytest.approx(
        EXPECTED_SOLVED_BOUNDARY_AREA_M2, abs=2.0e-15
    )
    assert separatrix["exact_solid_of_revolution_m3"] == pytest.approx(
        EXPECTED_SOLVED_BOUNDARY_VOLUME_M3, abs=2.0e-14
    )


def test_current_partition_is_disjoint_and_closes_the_pin(result):
    receipt, fields = result
    table = receipt["current_partition_table"]
    assert {name: row["cell_count"] for name, row in table.items()} == {
        "confined_core": 95,
        "inside_closed_branch_outside_core": 139,
        "outside_closed_branch": 855,
    }
    assert table["confined_core"]["cell_current_a"] == pytest.approx(
        416958.25412595563, abs=2.0e-9
    )
    assert table["inside_closed_branch_outside_core"][
        "cell_current_a"
    ] == pytest.approx(515290.2244376731, abs=2.0e-9)
    assert table["outside_closed_branch"]["cell_current_a"] == pytest.approx(
        786.3964363712394, abs=2.0e-9
    )
    closure = receipt["current_closure"]
    assert closure["summed_partition_current_a"] == EXPECTED_TOTAL_CURRENT_A
    assert closure["closure_residual_a"] == 0.0
    assert closure["partitioned_out_of_core_current_a"] == pytest.approx(
        EXPECTED_OUT_OF_CORE_CURRENT_A, abs=2.0e-9
    )
    partition = fields["partition"]
    assert not np.any(partition["core"] & partition["interior_noncore"])
    assert not np.any(partition["core"] & partition["exterior"])
    assert np.all(
        partition["core"] | partition["interior_noncore"] | partition["exterior"]
    )


def test_boundary_enclosed_rescore_preserves_the_surviving_deficit(result):
    receipt, _fields = result
    rescore = receipt["boundary_enclosed_moment_rescore"]
    assert rescore["support"]["cell_count"] == 234
    assert rescore["support"]["current_integral_a"] == pytest.approx(
        932248.4785636287, abs=2.0e-9
    )
    assert rescore["integrals"][
        "poloidal_field_squared_volume_integral_t2_m3"
    ] == pytest.approx(0.28949138263178675, rel=2.0e-15)
    inductance = rescore["internal_inductance"]
    assert inductance["solved"] == pytest.approx(0.4484883501138651, rel=2.0e-15)
    assert inductance["signed_relative_deviation"] == pytest.approx(
        -0.41707821153026325, rel=2.0e-15
    )
    assert inductance["matched_support_estimate_signed_relative_deviation"] == (
        EXPECTED_MATCHED_SUPPORT_LI_DEVIATION
    )
    beta = rescore["poloidal_beta"]
    assert beta["solved"] == pytest.approx(0.32178501123910763, rel=2.0e-15)
    assert beta["signed_relative_deviation"] == pytest.approx(
        -0.03982171826624836, rel=2.0e-15
    )
    assert beta["matched_support_estimate_signed_relative_deviation"] == (
        EXPECTED_MATCHED_SUPPORT_BETA_DEVIATION
    )


def test_partition_supports_masking_but_does_not_close_inductance(result, tmp_path):
    receipt, fields = result
    discriminator = receipt["discriminator_result"]
    assert discriminator["current_location_supported_reading"] == "MASKING_ARTIFACT"
    assert discriminator["inside_share_of_out_of_core_current"] == pytest.approx(
        0.998476202167346, rel=2.0e-15
    )
    assert discriminator["internal_inductance_mechanism_verdict"] == (
        "MASKING_CURRENT_CONFIRMED_BUT_FIELD_DEFICIT_SURVIVES"
    )
    figure = Path(tmp_path) / "partition.png"
    _plot_partition(fields["contour"], fields["partition"], figure)
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
