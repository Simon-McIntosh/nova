"""Validate the banked density forcing ladder."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path(__file__).resolve().parent


def main() -> None:
    """Check controls, ladder arms, responses, costs, and mechanism verdict."""
    report = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
    arms = report["arms"]
    production = arms["seven_sample_quadratic"]
    cubic = arms["thirteen_sample_cubic"]
    exact = arms["exact_flux_at_density_samples"]
    assert report["fixture"]["cache"]["warm_hit"] is True
    assert report["fixture"]["plasma_cells"] == 566
    assert abs(production["forcing"]["sup_wb"] - 1.252292371968979) < 1.0e-14
    projection_difference = (
        production["forcing"]["projection_fraction"] - 0.991762605972982
    )
    assert abs(projection_difference) < 2.0e-14
    assert report["production_control_difference"]["production_image_sup_wb"] < 2.0e-13
    assert cubic["forcing"]["sup_wb"] > production["forcing"]["sup_wb"]
    assert exact["forcing"]["sup_wb"] > production["forcing"]["sup_wb"]
    assert exact["interpolation"]["flux_interpolation_error_wb"]["sup"] == 0.0
    assert exact["interpolation"]["density_error"]["sup"] == 0.0
    assert report["mechanism"]["dominant_stage"] == (
        "density_fit_projection_after_exact_flux_sampling"
    )
    assert report["mechanism"]["nonlinear_rectification_detected"] is False
    assert report["verdict"]["density_forcing_falls_with_cubic_sampling"] is False
    for name, expected_samples in (
        ("seven_sample_quadratic", 7),
        ("thirteen_sample_cubic", 13),
        ("exact_flux_at_density_samples", 672),
    ):
        arm = arms[name]
        assert arm["cost"]["flux_samples_per_cell"] == expected_samples
        assert arm["cost"]["steady_median_microseconds"] > 0.0
        assert arm["root_response"]["gmres_info"] == 0
        assert arm["root_response"]["residual_relative_sup"] < 1.0e-12
    figure = OUTPUT / "density-forcing-ladder.png"
    assert figure.stat().st_size > 100_000
    assert report["artifacts"]["figure_bytes"] == figure.stat().st_size
    mechanism = report["mechanism"]
    print(
        "VERIFIED "
        f"production={production['forcing']['sup_wb']:.9f}Wb "
        f"cubic_ratio={mechanism['cubic_forcing_sup_fraction_of_production']:.9f} "
        f"exact_ratio={mechanism['exact_flux_forcing_sup_fraction_of_production']:.9f}"
    )


if __name__ == "__main__":
    main()
