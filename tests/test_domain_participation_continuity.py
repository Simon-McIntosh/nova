"""Continuity pin for source participation at a domain-label hand-off."""

from __future__ import annotations

import numpy as np

from benchmarks.domain_participation_jump import measure_participation_jump


def test_non_crossing_cell_participation_is_continuous_through_label_flip():
    """A cell outside the crossing set must not switch its whole source on."""
    result = measure_participation_jump(write_artifacts=False)

    assert result["target_cell"]["polyline_intersects"] is False
    assert result["target_cell"]["labels_across_fine_pair"] == ["COMMON_SOL", "CORE"]

    ratios = np.asarray(result["epsilon_doubling_ratios"]["all_entries"])
    message = (
        "non-crossing centroid-label flip switches the whole target-cell current: "
        f"{result['measured_jump']['target_cell_current_A']:.6f} A, "
        f"{result['measured_jump']['stall_amplitude_fraction']:.3f} of the "
        "reference stall amplitude; continuity requires every current, moment and "
        f"composed-field epsilon-doubling ratio near 2, measured "
        f"[{ratios.min():.6f}, {ratios.max():.6f}]"
    )
    np.testing.assert_allclose(ratios, 2.0, rtol=2.0e-2, atol=0.0, err_msg=message)
