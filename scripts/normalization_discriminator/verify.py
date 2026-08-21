"""Validate the banked normalization-constant factorial measurement."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path(__file__).resolve().parent


def main() -> None:
    """Check controls, factorial completeness, responses, and classification."""
    report = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
    assert report["fixture"]["cache"]["warm_hit"] is True
    assert report["fixture"]["plasma_cells"] == 566
    assert report["fixture"]["density_evaluation_points_per_cell"] == 672
    control = report["production_control_difference"]
    assert control["production_image_sup_wb"] < 2.0e-13
    assert abs(control["density_projection_fraction"]) < 2.0e-14
    assert abs(control["density_forcing_sup_wb"]) < 1.0e-13
    assert control["normalised_interpolation_identity_sup"] < 2.0e-13

    expected = {
        "production_values_production_constants",
        "exact_values_production_constants",
        "production_values_exact_constants",
        "exact_values_exact_constants",
    }
    assert set(report["arms"]) == expected
    for arm in report["arms"].values():
        assert arm["forcing"]["sup_wb"] >= 0.0
        assert arm["root_response"]["gmres_info"] == 0
        assert arm["root_response"]["residual_relative_sup"] < 1.0e-12
        assert "axis_signed_projection_mm" in arm["root_response"]
        assert "flux_signed_peak_percent_of_span" in arm["root_response"]

    constants = report["normalization_constants"]
    saddle = constants["banked_clip_to_newton_saddle"]
    assert abs(saddle["clip_beyond_saddle_psi_norm"] - 5.530053327573725e-05) < 1e-15
    carrier = report["verdict"]["collapse_carrier"]
    assert carrier in {
        "normalization_constants",
        "in_cell_flux_values",
        "interaction_or_other",
    }
    if carrier == "normalization_constants":
        assert report["verdict"]["exact_values_with_production_constants_near_full"]
        assert report["verdict"]["exact_constants_collapse_with_production_values"]
        assert report["verdict"]["exact_constants_collapse_with_exact_values"]
    figure = OUTPUT / "factorial.png"
    assert figure.stat().st_size > 50_000
    assert report["artifacts"]["figure_bytes"] == figure.stat().st_size
    print(
        "VERIFIED "
        f"carrier={carrier} "
        f"clip_saddle_offset={saddle['clip_beyond_saddle_psi_norm']:.12g}"
    )


if __name__ == "__main__":
    main()
