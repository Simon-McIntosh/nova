"""Tests for the native latest-DD DIII-D machine-description IDS export."""

import json
from pathlib import Path

import imas
import pytest

from benchmarks.diiid_ids_machine_description import (
    VERBATIM_ARTIFACT_CONTENT_SHA256,
    write_repaired_artifact_receipt,
)
from nova.imas.diiid_machine_ids import (
    IDS_NAMES,
    SOURCE_PATH,
    _primitive_leaves,
    latest_published_dd_version,
    machine_ids_snapshot,
    round_trip_leaf_receipt,
)
from nova.scripts.diiid_machine_artifact import publish_diiid_machine_artifact


DEFAULT_OUTPUT = Path(
    "docs/figures/diiid-forward-onboarding/ids-set/diiid_machine_description.nc"
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
    if not SOURCE_PATH.is_file():
        pytest.skip("the authoritative DIII-D netCDF source is not mounted")
    output = tmp_path / "diiid_machine_description.nc"
    artifact_receipt_path = tmp_path / "artifact.receipt.json"
    manifest_path = tmp_path / "diiid_machine_description.manifest.json"
    recipe_path = tmp_path / "PUBLISH.md"
    artifact_receipt = publish_diiid_machine_artifact(
        repository="ghcr.io/test-account/diii-d-machine-description",
        cache_directory=tmp_path / "cache",
        source_path=SOURCE_PATH,
        output=output,
        receipt_path=artifact_receipt_path,
        manifest_path=manifest_path,
        recipe_path=recipe_path,
    )
    receipt = write_repaired_artifact_receipt(
        source_path=SOURCE_PATH,
        ids_path=output,
        manifest_path=manifest_path,
        artifact_receipt_path=artifact_receipt_path,
        recipe_path=recipe_path,
        receipt_path=tmp_path / "diiid_machine_description.receipt.json",
    )
    written_dd = receipt["native_authoring"]["target_data_dictionary"]

    assert written_dd == latest_published_dd_version()
    assert written_dd.split(".")[0] == "4"
    assert receipt["native_authoring"]["source_data_dictionary"] == "3.41.0"
    assert receipt["native_authoring"]["cross_major_conversion_performed"] is False
    assert artifact_receipt["round_trip"]["exact_equal"] is True
    assert artifact_receipt["round_trip"]["maximum_absolute_difference"] == 0.0
    assert receipt["limiter"]["source_vertex_count"] == 82
    assert receipt["limiter"]["repair"]["published_vertex_count"] == 84
    assert receipt["artifact"]["content_sha256"] != VERBATIM_ARTIFACT_CONTENT_SHA256
    assert all(receipt["identity_comparison"].values())
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


def _wiring_and_passive_leaf_count(pf_active_ids, pf_passive_ids) -> int:
    """Count the circuit/supply/pf_passive leaves layered onto the core bundle."""

    count = 0
    for circuit in pf_active_ids.circuit:
        count += sum(1 for _ in _primitive_leaves(circuit, ("name", "description")))
        count += 1  # connections
    for supply in pf_active_ids.supply:
        count += sum(1 for _ in _primitive_leaves(supply, ("name", "description")))
    for loop in pf_passive_ids.loop:
        count += sum(
            1
            for _ in _primitive_leaves(
                loop,
                (
                    "name",
                    "description",
                    "resistivity",
                    "resistance",
                    "resistance_error_lower",
                    "resistance_error_upper",
                ),
            )
        )
        for element in loop.element:
            count += sum(
                1
                for _ in _primitive_leaves(
                    element, ("name", "description", "area", "turns_with_sign")
                )
            )
            count += 3  # geometry_type, outline/r, outline/z
    return count


def test_published_receipt_records_repaired_identity_and_native_authoring():
    receipt_path = DEFAULT_OUTPUT.with_suffix(".receipt.json")
    receipt = json.loads(receipt_path.read_text())

    round_trip = receipt["artifact"]["round_trip"]
    published_dd = receipt["native_authoring"]["target_data_dictionary"]
    extended_ids_names = (*IDS_NAMES, "pf_passive")
    with imas.DBEntry(DEFAULT_OUTPUT, "r", dd_version=published_dd) as database:
        published = {
            name: database.get(name, 0, autoconvert=False)
            for name in extended_ids_names
        }
    core = machine_ids_snapshot({name: published[name] for name in IDS_NAMES})
    authored_leaf_count = sum(len(values) for values in core.values())
    authored_leaf_count += _wiring_and_passive_leaf_count(
        published["pf_active"], published["pf_passive"]
    )

    assert round_trip["authored_leaf_count"] == authored_leaf_count
    assert round_trip["exact_equal"] is True
    assert receipt["native_authoring"]["target_data_dictionary"] == (
        latest_published_dd_version()
    )
    assert receipt["native_authoring"]["cross_major_conversion_performed"] is False
    assert receipt["limiter"]["authority"] == "repaired-ring-only"
    assert receipt["limiter"]["source_chain_valid"] is False
    assert receipt["limiter"]["repair"]["valid_material_relative_area_difference"] == (
        pytest.approx(4.1593370133e-05, rel=1.0e-10)
    )
    assert all(receipt["identity_comparison"].values())
    assert receipt["data_dictionary_floor"]["refused"] is True
