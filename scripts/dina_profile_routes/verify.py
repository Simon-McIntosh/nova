"""Verify the banked DINA dual-route receipt and figure contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


OUTPUT = Path(__file__).resolve().parent
FIGURES = Path(__file__).resolve().parents[2] / "docs/figures/dina-profile-routes"


def main() -> None:
    """Assert provenance, qualification, caches, and route separation."""
    receipt = json.loads((OUTPUT / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["reference"]["dd_version"] == "3.39.0"
    assert receipt["reference"]["reader"] == "imas.DBEntry"
    assert receipt["reference"]["mapping_bit_identical_to_stored_reader"] is True
    anchors = receipt["anchors"]["normalization_constants"]
    boundary_offset = anchors["offsets"]["boundary_flux_wb"]
    assert abs(boundary_offset - 0.4109918832381485) < 1.0e-12
    qualification = receipt["qualification"]
    assert qualification["reliable_shells"] >= 2
    assert qualification["total_shells"] == 19
    common = receipt["common_profile_base"]
    reliable = np.asarray(common["reliable"], dtype=bool)
    assert reliable.sum() == qualification["reliable_shells"]
    assert len(common["psi_norm_declared"]) == qualification["total_shells"]
    for name, fixture in receipt["reproduction_lane_forcing"].items():
        assert fixture["cache"]["warm_hit"] is True
        assert fixture["cache"]["bitwise_stored_precision"] is True
        assert fixture["cache"]["semantic_key"] in {
            "746fbe1553c4b242",
            "f0f96aa214aa9459",
        }
        difference = fixture["map_extracted_minus_declared_image_wb"]
        assert np.isfinite(difference["sup"])
        assert difference["sup"] > 0.0
    expected = {
        "profile_routes.png",
        "route_deviation.png",
        "primitive_integrals.png",
        "anchor_offsets.png",
        "forcing_routes.png",
    }
    assert expected.issubset({path.name for path in FIGURES.glob("*.png")})
    html = (OUTPUT / "report.html").read_text(encoding="utf-8")
    for name in expected:
        assert f'src="/nova/figures/dina-profile-routes/{name}"' in html
    addendum = json.loads((OUTPUT / "addendum.json").read_text(encoding="utf-8"))
    assert addendum["reference"]["dd_version"] == "3.39.0"
    rotation = addendum["rotation_and_variance"]
    assert len(rotation["shells"]) == 19
    for shell in rotation["shells"]:
        assert shell["signal_rms_rj_phi_a_per_m"] > 0.0
        assert np.isfinite(shell["static_explained_variance_fraction"])
        assert np.isfinite(shell["rotation_column_amplitude"])
        assert np.isfinite(shell["rotation_column_amplitude_uncertainty"])
    constraint = addendum["rotational_pressure_constraint"]
    assert constraint["present_in_dictionary"] is True
    assert constraint["filled"] is False
    assert constraint["item_count"] == 0
    reconciliation = addendum["forcing_reconciliation"]
    assert len(reconciliation["arms"]) == 4
    correction = reconciliation["banked_arm_correction"]
    assert correction["direct_path_control_difference_wb"]["sup"] < 1.0e-10
    assert (
        abs(
            correction["rerun_grid_core_forcing_sup_wb"]
            - correction["banked_grid_core_forcing_sup_wb"]
        )
        < 1.0e-10
    )
    coordinate = addendum["extracted_route_coordinate_control"]
    assert len(coordinate["arms"]) == 3
    assert coordinate["extended_extraction"]["reliable_shells"] > 19
    assert abs(coordinate["controls"]["banked_rerun_difference_wb"]) < 1.0e-10
    assert (FIGURES / "route_controls.png").stat().st_size > 1000
    print(
        "DINA_DUAL_ROUTE_VERIFY_EXIT=0 "
        f"reliable={qualification['reliable_shells']}/{qualification['total_shells']} "
        f"boundary_offset_wb={boundary_offset:.12g}"
    )


if __name__ == "__main__":
    main()
