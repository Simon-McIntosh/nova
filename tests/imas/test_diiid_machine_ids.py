"""Tests for the static DIII-D machine-description IDS export."""

from pathlib import Path

import imas
import numpy as np

from benchmarks.diiid_machine_ids_export import export_machine_ids
from nova.imas.diiid_machine_ids import (
    DD_VERSION,
    IDS_NAMES,
    _author_magnetics,
    _author_pf_active,
    _author_wall,
    machine_ids_snapshot,
    round_trip_leaf_receipt,
)


def _source_ids() -> dict[str, object]:
    factory = imas.IDSFactory(version=DD_VERSION)
    wall = factory.new("wall")
    wall.description_2d.resize(1)
    wall.description_2d[0].type.index = 1
    wall.description_2d[0].limiter.unit.resize(1)
    limiter = wall.description_2d[0].limiter.unit[0]
    limiter.name = "limiter"
    limiter.outline.r = [1.0, 2.0, 2.0, 1.0]
    limiter.outline.z = [-1.0, -1.0, 1.0, 1.0]

    active = factory.new("pf_active")
    active.coil.resize(1)
    coil = active.coil[0]
    coil.name = "coil"
    coil.identifier = "coil"
    coil.function.resize(1)
    coil.function[0].index = 1
    coil.element.resize(2)
    rectangle = coil.element[0]
    rectangle.name = "rectangle"
    rectangle.turns_with_sign = 58.0
    rectangle.geometry.geometry_type = 2
    rectangle.geometry.rectangle.r = 1.25
    rectangle.geometry.rectangle.z = 0.25
    rectangle.geometry.rectangle.width = 0.2
    rectangle.geometry.rectangle.height = 0.4
    outline = coil.element[1]
    outline.name = "outline"
    outline.turns_with_sign = -3.0
    outline.geometry.geometry_type = 1
    outline.geometry.outline.r = [1.4, 1.5, 1.6, 1.5]
    outline.geometry.outline.z = [0.0, 0.1, 0.1, 0.0]

    magnetics = factory.new("magnetics")
    magnetics.b_field_pol_probe.resize(1)
    probe = magnetics.b_field_pol_probe[0]
    probe.name = "probe"
    probe.identifier = "probe"
    probe.type.index = 1
    probe.position.r = 1.8
    probe.position.z = -0.2
    probe.poloidal_angle = 0.3
    probe.length = 0.04
    magnetics.flux_loop.resize(1)
    loop = magnetics.flux_loop[0]
    loop.name = "loop"
    loop.identifier = "loop"
    loop.type.index = 1
    loop.position.resize(1)
    loop.position[0].r = 1.9
    loop.position[0].z = 0.4
    return {"wall": wall, "pf_active": active, "magnetics": magnetics}


def test_authored_ids_expand_elements_and_keep_only_static_geometry(tmp_path: Path):
    source = _source_ids()
    factory = imas.IDSFactory(version=DD_VERSION)
    ids = {
        "wall": _author_wall(factory, source["wall"], tmp_path / "source.nc"),
        "pf_active": _author_pf_active(
            factory, source["pf_active"], tmp_path / "source.nc"
        ),
        "magnetics": _author_magnetics(
            factory, source["magnetics"], tmp_path / "source.nc"
        ),
    }

    assert tuple(ids) == IDS_NAMES
    assert np.array_equal(
        ids["pf_active"].coil[0].element[0].geometry.outline.r,
        [1.15, 1.35, 1.35, 1.15],
    )
    assert np.allclose(
        ids["pf_active"].coil[0].element[0].geometry.outline.z,
        [0.05, 0.05, 0.45, 0.45],
        rtol=0.0,
        atol=1.0e-15,
    )
    assert float(ids["pf_active"].coil[0].element[0].turns_with_sign) == 58.0
    assert len(ids["magnetics"].b_field_pol_probe) == 1
    assert not ids["magnetics"].b_field_pol_probe[0].field.data.has_value
    assert not ids["magnetics"].flux_loop[0].flux.data.has_value
    assert len(ids["magnetics"].ip) == 0
    assert all(
        str(value.ids_properties.source) == str(tmp_path / "source.nc")
        for value in ids.values()
    )


def test_leaf_receipt_requires_exact_geometry_and_reports_zero_differences(
    tmp_path: Path,
):
    source = _source_ids()
    factory = imas.IDSFactory(version=DD_VERSION)
    ids = {
        "wall": _author_wall(factory, source["wall"], tmp_path / "source.nc"),
        "pf_active": _author_pf_active(
            factory, source["pf_active"], tmp_path / "source.nc"
        ),
        "magnetics": _author_magnetics(
            factory, source["magnetics"], tmp_path / "source.nc"
        ),
    }
    snapshot = machine_ids_snapshot(ids)

    receipt = round_trip_leaf_receipt(snapshot, snapshot)

    wall = receipt["wall"]["description_2d[0]/limiter/unit[0]/outline/r"]
    element = receipt["pf_active"]["coil[0]/element[0]/geometry/outline/r"]
    assert wall == {"exact_equal": True, "maximum_absolute_difference": 0.0}
    assert element == {
        "exact_equal": True,
        "maximum_absolute_difference": 0.0,
    }


def test_live_export_round_trip_has_exact_required_content(tmp_path: Path):
    receipt = export_machine_ids(tmp_path / "diiid_machine_description.nc")

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
    assert receipt["output"]["size_bytes"] > 0
    assert len(receipt["output"]["sha256"]) == 64
    assert all(
        properties["data_dictionary"] == DD_VERSION
        and properties["source"].endswith("DIII-D/200000.nc")
        and properties["provenance"][0]["sources"]
        == ["IMAS netCDF entry; Data Dictionary 3.41.0"]
        for properties in receipt["ids_properties"].values()
    )
    assert [item["quantity"] for item in receipt["declared_absent"]] == [
        "pf_passive",
        "tf static conductor geometry",
        "Thomson scattering line-of-sight endpoints",
    ]
