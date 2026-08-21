"""Contracts for the DIII-D omitted-solenoid inclusion measurement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import diiid_solenoid_inclusion_ladder as ladder
from nova.imas.diiid_description import POLOIDAL_CONDUCTORS


def test_declaration_fixes_ecoila_only_current_authority() -> None:
    declaration = ladder.preregistration()

    assert declaration["cohort"]["frame_count"] >= 5
    assert len({item["shot"] for item in declaration["cohort"]["frames"]}) >= 5
    assert declaration["current_authority"]["nothing_fitted"]
    assert declaration["current_authority"]["no_current_adjusted"]
    assert "no recovered current" in declaration["current_authority"]["operation"]
    assert declaration["solver"]["relative_residual_criterion"] == 1.0e-6
    assert declaration["metrics"]["label_representability_ceiling"] == 0.0429


def test_inclusion_vectors_use_only_shipped_ecoila_and_fixed_scales() -> None:
    shipped = np.arange(24, dtype=float) + 100.0
    shipped[-5:] = 0.0
    ecoila = float(shipped[POLOIDAL_CONDUCTORS.index("ECOILA")])

    currents = ladder.inclusion_currents(shipped, ecoila)

    assert len(currents) == 4
    np.testing.assert_allclose(currents[0], shipped)
    assert currents[1][-5] == pytest.approx(1.0172 * ecoila)
    np.testing.assert_allclose(
        currents[2][-5:],
        [1.0172 * ecoila, 0.9929 * ecoila, 0.9823 * ecoila, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        currents[3][-5:],
        [
            1.0172 * ecoila,
            0.9929 * ecoila,
            0.9823 * ecoila,
            0.9806 * ecoila,
            1.0165 * ecoila,
        ],
    )


def test_ampere_turn_fractions_are_cumulative_and_complete() -> None:
    fractions = ladder.ampere_turn_fractions()

    assert fractions[0]["cumulative_missing_ampere_turn_fraction"] == 0.0
    assert fractions[-1]["cumulative_missing_ampere_turn_fraction"] == pytest.approx(
        1.0
    )
    assert fractions[2]["cumulative_missing_ampere_turn_fraction"] == pytest.approx(
        0.8170494544
    )
    assert sum(
        item["incremental_missing_ampere_turn_fraction"] for item in fractions
    ) == pytest.approx(1.0)
    assert fractions[1]["cumulative_missing_ampere_turn_fraction"] > 0.6


def test_monotonic_classifier_requires_every_rung_to_approach() -> None:
    assert ladder.monotonic_toward_label([0.45, 0.30, 0.20, 0.10])
    assert not ladder.monotonic_toward_label([0.45, 0.30, 0.31, 0.10])
    assert not ladder.monotonic_toward_label([0.45, np.nan, 0.20, 0.10])


def test_generated_receipt_has_five_frames_and_all_four_inclusions() -> None:
    receipt = json.loads((ladder.DEFAULT_OUTPUT / ladder.RECEIPT_NAME).read_text())
    result = receipt["result"]

    assert result["frame_count"] >= 5
    assert result["distinct_shots"] >= 5
    assert result["all_shots_screened_free_of_affected_population"]
    assert result["all_inclusions_derived_from_shipped_ecoila"]
    assert result["label_recovered_current_values_used"] == 0
    assert len(result["per_inclusion"]) == 4
    assert "monotonic" in result["pooled_x_point_migration_verdict"]
    assert (
        result["fully_qualified_trajectory_frames"]
        + result["unqualified_trajectory_frames"]
        == result["frame_count"]
    )
    for frame in result["frames"]:
        assert len(frame["inclusions"]) == 4
        assert frame["same_label_branch_seed_all_inclusions"]
        for inclusion in frame["inclusions"]:
            assert "relative_residual" in inclusion
            assert "iterations" in inclusion
            assert "terminal_topology" in inclusion
            assert "x_point_rz_m" in inclusion
            assert "x_point_separation_m" in inclusion
            assert "lcfs_symmetric_mean_separation_m" in inclusion
            assert "gauge_free_fractional_rms" in inclusion
            assert "qualified_converged_diverted" in inclusion
    assert (ladder.DEFAULT_OUTPUT / ladder.FIGURE_NAME).is_file()


def test_source_keeps_equilibrium_read_only_and_refuses_recovery_values() -> None:
    source = Path(ladder.__file__).read_text()

    assert "nova/equilibrium" not in source
    assert "recovered_currents_a" not in source
    assert "never reads a label-derived current recovery" in source
