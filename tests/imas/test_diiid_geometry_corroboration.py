"""Tests for direct DIII-D geometry corroboration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.diiid_geometry_corroboration import (
    EXPECTED_TABLE_DIGEST,
    SKEWED_CONDUCTORS,
    build_receipt,
    load_competition,
    section_descriptor,
    write_figures,
)
from benchmarks.diiid_ids_machine_description import read_entry


@pytest.fixture(scope="module")
def corroboration() -> tuple[dict, dict]:
    """Build the receipt once from the authoritative artifacts."""
    return build_receipt(load_competition(), read_entry())


def test_section_normalization_ignores_vertex_order_and_bbox_height() -> None:
    competition = np.asarray(
        [
            [0.9345, 1.3876],
            [1.0737, 1.5268],
            [1.0737, 1.6462],
            [0.9345, 1.5070],
        ]
    )
    netcdf = competition[[0, 3, 2, 1]]
    released = section_descriptor(competition)
    stored = section_descriptor(netcdf)
    assert released["width_m"] == pytest.approx(0.1392)
    assert released["height_m"] == pytest.approx(0.1194)
    assert released["skew_deg"] == pytest.approx(45.0)
    assert stored["centre_m"] == pytest.approx(released["centre_m"])
    assert stored["width_m"] == pytest.approx(released["width_m"])
    assert stored["height_m"] == pytest.approx(released["height_m"])
    assert stored["skew_deg"] == pytest.approx(released["skew_deg"])
    assert stored["axis_aligned_extent_m"] == pytest.approx(
        released["axis_aligned_extent_m"]
    )
    assert stored["axis_aligned_extent_m"][1] == pytest.approx(0.2586)


def test_all_competition_conductors_pair_and_agree(corroboration) -> None:
    receipt, _ = corroboration
    assert receipt["sources"]["competition_table_digest"] == EXPECTED_TABLE_DIGEST
    assert receipt["pairing_summary"] == {
        "competition_count": 19,
        "agreed_count": 19,
        "disagreed_count": 0,
        "name_preserving_count": 19,
    }
    assert len(receipt["pairings"]) == 19
    assert all(pairing["verdict"] == "agreed" for pairing in receipt["pairings"])
    assert receipt["method"]["fitting"] is False
    assert receipt["method"]["coordinate_adjustment"] is False


def test_skew_set_and_f5a_discrepancy_are_resolved(corroboration) -> None:
    receipt, _ = corroboration
    skew = receipt["skewed_conductor_set"]
    assert skew["names"] == list(SKEWED_CONDUCTORS)
    assert skew["independently_named_by_both_sources"] is True
    assert skew["agreed_count"] == 6
    resolution = receipt["f5a_discrepancy_resolution"]
    assert resolution["netcdf_axis_aligned_extent_m"] == [0.1392, 0.2586]
    assert resolution["netcdf_first_edge_angle_deg"] == 90.0
    assert resolution["maximum_vertex_set_distance_mm"] == pytest.approx(0.0)
    assert resolution["real_machine_supports"] == "competition physical F5A polygon"


def test_omissions_and_grid_margins_are_quantified(corroboration) -> None:
    receipt, _ = corroboration
    coverage = receipt["coverage"]
    omitted = coverage["netcdf_coils_omitted_by_competition"]
    assert [coil["name"] for coil in omitted] == [
        "ECOILB",
        "E567UP",
        "E567DN",
        "E89DN",
        "E89UP",
    ]
    assert [coil["element_count"] for coil in omitted] == [48, 6, 6, 7, 7]
    assert coverage["omitted_element_count"] == 74
    assert coverage["expected_116_element_claim_supported"] is False
    assert coverage["ampere_turn_carrying_conductors"] == {
        "competition_coil_count": 19,
        "competition_element_count": 19,
        "netcdf_pf_active_coil_count": 24,
        "netcdf_pf_active_element_count": 140,
    }

    grid = receipt["limiter_against_competition_grid"]
    assert grid["limiter_vertex_count"] == 82
    assert grid["grid_shape"] == [65, 65]
    assert grid["grid_encloses_wall"] is True
    assert grid["signed_grid_beyond_wall_margins_m"] == pytest.approx(
        {
            "inboard_m": 0.17730003595352168,
            "outboard_m": 0.18889999389648438,
            "lower_m": 0.2410600185394287,
            "upper_m": 0.23701000213623047,
        }
    )


def test_receipt_and_figures_are_serializable(corroboration, tmp_path: Path) -> None:
    receipt, plot_data = corroboration
    figure_paths = write_figures(plot_data, tmp_path)
    assert len(figure_paths) == 2
    assert all(path.is_file() and path.stat().st_size > 10_000 for path in figure_paths)
    assert json.loads(json.dumps(receipt))["pairing_summary"]["agreed_count"] == 19
