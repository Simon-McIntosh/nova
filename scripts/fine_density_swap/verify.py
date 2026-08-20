"""Validate the banked fine density-swap evidence."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path(__file__).resolve().parent


def main() -> None:
    """Check cache, control, solve, trend, and artifact invariants."""
    report = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
    carrier = report["cache_carrier"]
    fine = report["fine_measurement"]
    coarse = report["coarse_control"]
    trend = report["refinement"]
    assert carrier["warm_hit"] is True
    assert carrier["realised_cells"] == 1069
    assert carrier["arrays_verified"] == carrier["persisted_child_count"] == 31
    assert carrier["persisted_coupling_child_count"] == 20
    assert abs(coarse["density_projection_percent"] - 99.1762605973) < 1.0e-9
    assert abs(coarse["root_response"]["axis_signed_projection_mm"] - 40.065) < 0.001
    assert (
        abs(coarse["root_response"]["flux_signed_peak_percent_of_span"] - 6.386) < 0.001
    )
    assert fine["root_response"]["gmres_info"] == 0
    assert fine["root_response"]["residual_relative_sup"] < 1.0e-12
    assert 0.98 < fine["forcing"]["projection_fraction"] < 1.0
    assert 0.99 < trend["fine_to_coarse_ratio"] < 1.01
    assert abs(trend["estimated_power_against_cell_width"]) < 0.02
    assert report["verdict"]["density_is_h_independent_carrier_estimate"] is True
    figure = OUTPUT / "fine-density-swap.png"
    assert figure.stat().st_size > 100_000
    assert report["artifacts"]["figure_bytes"] == figure.stat().st_size
    print(
        "VERIFIED "
        f"fine_share={100.0 * fine['forcing']['projection_fraction']:.9f}% "
        f"fine_to_coarse={trend['fine_to_coarse_ratio']:.9f} "
        f"width_power={trend['estimated_power_against_cell_width']:.9f}"
    )


if __name__ == "__main__":
    main()
