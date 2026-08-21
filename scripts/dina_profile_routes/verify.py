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
    assert {path.name for path in FIGURES.glob("*.png")} == expected
    html = (OUTPUT / "report.html").read_text(encoding="utf-8")
    for name in expected:
        assert f'src="/nova/figures/dina-profile-routes/{name}"' in html
    print(
        "DINA_DUAL_ROUTE_VERIFY_EXIT=0 "
        f"reliable={qualification['reliable_shells']}/{qualification['total_shells']} "
        f"boundary_offset_wb={boundary_offset:.12g}"
    )


if __name__ == "__main__":
    main()
