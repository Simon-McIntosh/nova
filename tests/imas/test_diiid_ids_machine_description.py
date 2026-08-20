"""Tests for the DIII-D netCDF static machine-description receipt."""

from pathlib import Path

import pytest

from benchmarks.diiid_ids_machine_description import build_receipt, write_figures


def _source() -> dict:
    return {
        "source_path": "/data/DIII-D/200000.nc",
        "backend": "imas-python netCDF DBEntry",
        "mode": "read-only",
        "occurrences": {
            "wall": [0],
            "pf_active": [0],
            "pf_passive": [],
            "tf": [0],
            "magnetics": [0],
            "equilibrium": [0],
        },
        "dd_versions": {
            "wall": "3.41.0",
            "pf_active": "3.41.0",
            "pf_passive": None,
            "tf": "3.41.0",
            "magnetics": "3.41.0",
            "equilibrium": "3.41.0",
        },
        "contour": {
            "kind": "limiter",
            "r": [1.0, 2.0, 2.0, 1.0],
            "z": [-1.0, -1.0, 1.0, 1.0],
        },
        "pf_active": [
            {
                "name": "coil",
                "identifier": "coil",
                "elements": [
                    {
                        "name": "rectangle",
                        "geometry_type": 2,
                        "r": 1.2,
                        "z": 0.3,
                        "width": 0.1,
                        "height": 0.2,
                    },
                    {
                        "name": "skew",
                        "geometry_type": 1,
                        "r": [1.4, 1.5, 1.6, 1.5],
                        "z": [0.0, 0.1, 0.1, 0.0],
                    },
                ],
            }
        ],
        "pf_passive_loop_count": 0,
        "tf_coil_count": 0,
        "tf": {
            "occurrence_present": True,
            "filled_paths": ["b_field_tor_vacuum_r/data"],
            "static_geometry_present": False,
        },
        "doctrine_fence": {
            "magnetics_occurrence_present": True,
            "equilibrium_occurrence_present": True,
            "equilibrium_time_slice_count": 340,
            "equilibrium_constraints_present": True,
        },
    }


def _competition_grid() -> dict:
    return {
        "shape": [65, 65],
        "r_extent_m": [0.84, 2.54],
        "z_extent_m": [-1.6, 1.6],
        "source_receipt": "machine_description_receipt.json",
        "provenance": {"statement": "released competition coordinate axes"},
    }


def test_receipt_preserves_coils_and_routes_each_element_outline():
    receipt, machine = build_receipt(_source(), _competition_grid())

    active = receipt["quantities"]["pf_active"]
    assert active["coil_count"] == 1
    assert active["element_count"] == 2
    assert active["non_rectangular_element_count"] == 1
    assert active["coils"][0]["element_count"] == 2
    rectangle, outline = active["coils"][0]["elements"]
    assert rectangle["status"] == "read_after_named_change"
    assert rectangle["outline_vertex_count"] == 4
    assert rectangle["centre_m"] == pytest.approx([1.2, 0.3])
    assert rectangle["width_m"] == pytest.approx(0.1)
    assert rectangle["height_m"] == pytest.approx(0.2)
    assert outline["status"] == "read_unmodified"
    assert outline["outline_vertices_m"] == [
        [1.4, 0.0],
        [1.5, 0.1],
        [1.6, 0.1],
        [1.5, 0.0],
    ]
    assert len(machine.active_coils[0].elements) == 2
    assert len(machine.active_sections) == 2


def test_receipt_keeps_absence_cocos_and_content_doctrine_explicit():
    receipt, _ = build_receipt(_source(), _competition_grid())

    assert receipt["dd_versions"]["wall"] == "3.41.0"
    assert receipt["dd_versions"]["pf_passive"] is None
    assert receipt["quantities"]["wall_limiter"]["vertex_count"] == 4
    assert receipt["quantities"]["tf"]["status"] == "cannot_reach"
    assert receipt["quantities"]["pf_passive"]["status"] == "cannot_reach"
    assert receipt["quantities"]["pf_passive"]["occurrence_present"] is False
    assert receipt["cocos"]["ids"]["source_index"] == 11
    assert receipt["cocos"]["ids"]["target_index"] == 17
    assert receipt["cocos"]["ids"]["transform_to_nova"]["psi_like"] == -1.0
    assert receipt["cocos"]["competition_corpus"]["source_index"] == 5
    assert receipt["cocos"]["competition_corpus"]["used_for_this_ids_read"] is False
    doctrine = receipt["doctrine_fence"]
    assert doctrine["additional_entry_content"]["magnetics_ids"] is True
    assert doctrine["additional_entry_content"]["constrained_equilibrium"] is True
    assert doctrine["additional_entry_content"]["equilibrium_time_slice_count"] == 340
    assert doctrine["magnetics_signal_used"] is False
    assert doctrine["equilibrium_label_used"] is False


def test_three_figures_cover_all_elements_outline_detail_and_grid(tmp_path: Path):
    receipt, machine = build_receipt(_source(), _competition_grid())

    figures = write_figures(machine, receipt, tmp_path)

    assert [path.name for path in figures] == [
        "limiter_pf_active_elements.png",
        "non_rectangular_element_outlines.png",
        "limiter_competition_grid_extent.png",
    ]
    assert all(path.stat().st_size > 0 for path in figures)
