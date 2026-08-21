"""Tests for the native latest-DD DIII-D machine-description IDS export."""

import json
from pathlib import Path

import imas

from benchmarks.diiid_machine_ids_export import (
    DEFAULT_OUTPUT,
    SUPERSEDED_LEAF_COUNT,
    SUPERSEDED_SHA256,
    SUPERSEDED_SIZE_BYTES,
    export_machine_ids,
)
from nova.imas.diiid_machine_ids import (
    IDS_NAMES,
    latest_published_dd_version,
    round_trip_leaf_receipt,
)


def test_latest_published_dd_version_is_resolved_semantically():
    assert latest_published_dd_version(["3.42.2", "4.0.0", "4.1.1", "4.1.0"]) == "4.1.1"
    assert latest_published_dd_version().split(".")[0] == "4"


def test_leaf_receipt_requires_exact_geometry_and_reports_zero_differences():
    snapshot = {
        "wall": {"description_2d[0]/limiter/unit[0]/outline/r": [1.0, 2.0]},
        "pf_active": {"coil[0]/element[0]/geometry/outline/r": [1.1, 1.2]},
        "magnetics": {"b_field_pol_probe[0]/position/r": 1.8},
    }

    receipt = round_trip_leaf_receipt(snapshot, snapshot)

    wall = receipt["wall"]["description_2d[0]/limiter/unit[0]/outline/r"]
    element = receipt["pf_active"]["coil[0]/element[0]/geometry/outline/r"]
    assert wall == {"exact_equal": True, "maximum_absolute_difference": 0.0}
    assert element == {
        "exact_equal": True,
        "maximum_absolute_difference": 0.0,
    }


def test_live_export_writes_exact_native_latest_dd(tmp_path: Path):
    output = tmp_path / "diiid_machine_description.nc"

    receipt = export_machine_ids(output)
    written_dd = receipt["native_authoring"]["target_data_dictionary"]

    assert written_dd == latest_published_dd_version()
    assert written_dd.split(".")[0] == "4"
    assert receipt["source"]["data_dictionary"] == "3.41.0"
    assert receipt["source"]["autoconvert"] is False
    assert receipt["native_authoring"]["cross_major_conversion_performed"] is False
    assert receipt["content"] == {
        "ids": ["wall", "pf_active", "magnetics"],
        "wall_limiter_vertices": 82,
        "pf_active_coils": 24,
        "pf_active_elements": 140,
        "b_field_pol_probe_positions": 76,
        "flux_loop_positions": 44,
        "magnetics_signal_arrays": 0,
        "equilibrium_occurrences": 0,
    }
    assert receipt["round_trip"]["verdict"] == "exact"
    assert receipt["round_trip"]["wall_outline_maximum_absolute_difference"] == 0.0
    assert receipt["round_trip"]["element_vertex_maximum_absolute_difference"] == 0.0
    major = receipt["round_trip"]["major_comparison"]
    assert major["comparison_available"] is False
    assert major["superseded_leaf_count"] == SUPERSEDED_LEAF_COUNT
    assert major["published_leaf_count"] == receipt["round_trip"]["total_leaf_count"]
    assert major["leaf_count_difference"] == (
        major["published_leaf_count"] - SUPERSEDED_LEAF_COUNT
    )
    assert major["presence_differences"] == []
    assert major["shape_differences"] == []
    assert receipt["output"]["superseded"] == {
        "data_dictionary": "3.41.0",
        "size_bytes": SUPERSEDED_SIZE_BYTES,
        "sha256": SUPERSEDED_SHA256,
    }
    assert receipt["output"]["size_bytes"] > 0
    assert len(receipt["output"]["sha256"]) == 64
    assert all(
        properties["data_dictionary"] == written_dd
        and properties["source"].endswith("DIII-D/200000.nc")
        and properties["provenance"][0]["sources"][0]
        == "IMAS netCDF source; Data Dictionary 3.41.0"
        for properties in receipt["ids_properties"].values()
    )
    assert [item["quantity"] for item in receipt["declared_absent"]] == [
        "pf_passive",
        "tf static conductor geometry",
        "Thomson scattering line-of-sight endpoints",
    ]

    with imas.DBEntry(output, "r", dd_version=written_dd) as database:
        versions = {
            name: str(
                database.get(
                    name, 0, autoconvert=False
                ).ids_properties.version_put.data_dictionary
            )
            for name in IDS_NAMES
        }
        assert database.list_all_occurrences("pf_passive") == []
        assert database.list_all_occurrences("tf") == []
        assert database.list_all_occurrences("equilibrium") == []
    assert versions == {name: written_dd for name in IDS_NAMES}


def test_published_receipt_names_every_cross_major_schema_change():
    receipt_path = DEFAULT_OUTPUT.with_suffix(".receipt.json")
    receipt = json.loads(receipt_path.read_text())
    major = receipt["round_trip"]["major_comparison"]

    assert receipt["round_trip"]["total_leaf_count"] == 1549
    assert major["comparison_available"] is True
    assert major["superseded_leaf_count"] == SUPERSEDED_LEAF_COUNT
    assert major["published_leaf_count"] == 1549
    assert major["leaf_count_difference"] == -284
    assert len(major["presence_differences"]) == 296
    assert all(
        {"ids", "path", "presence"} <= difference.keys()
        for difference in major["presence_differences"]
    )
    assert major["shape_differences"] == []
