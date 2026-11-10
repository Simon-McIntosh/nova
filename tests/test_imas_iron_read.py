"""Pinned WEST iron reads and preservation of the script inputs."""

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import imas
import numpy as np
import pytest
import shapely

import nova.imas.machine as machine_module
from nova.imas.machine import ImasGeometryReader, StaticMachineDescription
from scripts.prototypes.iron_magnetisation.magnetisation import Material, mesh

ROOT = Path(__file__).resolve().parents[1]
CALLERS = (
    "scripts/prototypes/iron_magnetisation/west.py",
    "scripts/west_stray_field/solve.py",
    "scripts/sheet_tile_census/measure.py",
)


@pytest.fixture(scope="module")
def west():
    """Open the WEST source at the DD pin declared by its curation manifest."""
    manifest_path = Path(
        os.environ.get(
            "NOVA_WEST_SOURCE_MANIFEST",
            "/home/ITER/mcintos/Code/imas-efit/tests/data-manifest.json",
        )
    )
    if not manifest_path.is_file():
        pytest.skip("WEST source manifest is not mounted")
    manifest = json.loads(manifest_path.read_text())
    row = next(
        row for row in manifest["machine_descriptions"] if row["key"] == "west-dd4-1-0"
    )
    source = manifest_path.parent.parent / row["target"]
    uri = f"imas:hdf5?path={source}"
    with imas.DBEntry(uri, "r", dd_version=row["dd_version"]) as entry:
        assert 0 in entry.list_all_occurrences("iron_core")
        iron = entry.get("iron_core", 0, lazy=False, autoconvert=False)
    return uri, row["dd_version"], iron


def _read_assignments(path, names, namespace):
    """Execute the callers' actual read expressions without their field solves."""
    tree = ast.parse((ROOT / path).read_text())
    for name in names:
        node = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace


def _namespace(west):
    uri, dd_version, _ = west
    assert dd_version == "4.1.0"
    return dict(
        MACHINE=uri.removeprefix("imas:hdf5?path="),
        machine=uri.removeprefix("imas:hdf5?path="),
        StaticMachineDescription=StaticMachineDescription,
        np=np,
        shapely=shapely,
        Material=Material,
    )


def _outlines(iron):
    """The entry-by-entry array expressions used by the displaced readers."""
    return [
        np.column_stack(
            (np.asarray(s.geometry.outline.r), np.asarray(s.geometry.outline.z))
        )
        for s in iron.segment
    ]


def test_west_iron_segments_match_previous_reads(west):
    uri, dd_version, iron = west
    description = StaticMachineDescription.read_iron_core(uri, dd_version=dd_version)
    assert len(description.iron_segments) == len(iron.segment) == 6
    assert [len(s.outline.r) for s in description.iron_segments] == [5, 6, 15, 5, 15, 6]
    for actual, expected in zip(description.iron_segments, iron.segment, strict=True):
        assert actual.name == str(expected.name)
        assert actual.description == str(expected.description)
        np.testing.assert_array_equal(actual.outline.r, expected.geometry.outline.r)
        np.testing.assert_array_equal(actual.outline.z, expected.geometry.outline.z)
        np.testing.assert_array_equal(actual.material.b_field, expected.b_field)
        np.testing.assert_array_equal(
            actual.material.permeability_relative, expected.permeability_relative
        )
        assert len(actual.material.b_field) == 25


def test_prototype_inputs_match_previous_read(west):
    namespace = _read_assignments(
        CALLERS[0], ("machine", "segments", "table", "material"), _namespace(west)
    )
    iron = west[2]
    assert len(namespace["segments"]) == len(iron.segment) == 6
    for actual, expected in zip(namespace["segments"], _outlines(iron), strict=True):
        np.testing.assert_array_equal(
            actual.exterior.coords, shapely.Polygon(expected).exterior.coords
        )
    np.testing.assert_array_equal(
        namespace["material"].b_field, iron.segment[0].b_field
    )
    np.testing.assert_array_equal(
        namespace["material"].mu_r, iron.segment[0].permeability_relative
    )


def test_stray_field_outlines_match_previous_read(west):
    namespace = _read_assignments(
        CALLERS[1], ("machine", "iron_outlines"), _namespace(west)
    )
    expected = [outline for outline in _outlines(west[2]) if len(outline)]
    assert len(namespace["iron_outlines"]) == len(expected) == 6
    for actual, outline in zip(namespace["iron_outlines"], expected, strict=True):
        np.testing.assert_array_equal(actual, outline)


def test_census_mesh_matches_previous_read(west):
    namespace = _namespace(west)
    namespace.update(polygons=[], mesh=mesh, args=SimpleNamespace(size=0.08))
    _read_assignments(CALLERS[2], ("description",), namespace)
    tree = ast.parse((ROOT / CALLERS[2]).read_text())
    loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and ast.unparse(node.iter) == "description.iron_segments"
    )
    exec(
        compile(ast.Module(body=[loop], type_ignores=[]), CALLERS[2], "exec"), namespace
    )
    expected = [
        polygon
        for outline in _outlines(west[2])
        for polygon in mesh(shapely.Polygon(outline).buffer(0), 0.08).polygons
    ]
    actual = namespace["polygons"]
    assert len(actual) == len(expected) == 1696
    for polygon, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(polygon, reference)


def _iron():
    iron = imas.IDSFactory("4.1.0").new("iron_core")
    iron.ids_properties.version_put.data_dictionary = "4.1.0"
    iron.segment.resize(2)
    for index, segment in enumerate(iron.segment):
        segment.name = f"iron {index}"
        segment.description = f"material description {index}"
        segment.geometry.outline.r = [1, 2, 2, 1, 1]
        segment.geometry.outline.z = [0, 0, 1, 1, 0]
        segment.b_field = [0, 1 + index]
        segment.permeability_relative = [215, 500 + index]
    return iron


def test_each_segment_keeps_its_material_and_raw_vertices():
    iron = _iron()
    description = StaticMachineDescription.from_iron_core_ids(iron)
    assert len(description.iron_segments) == 2
    assert (
        description.iron_segments[0].material != description.iron_segments[1].material
    )
    assert description.iron_segments[0].outline.r == (1, 2, 2, 1, 1)
    assert description.iron_segments[1].material.b_field == (0, 2)
    assert description.iron_segments[1].description == "material description 1"
    iron.segment[1].b_field = [0, 9]
    assert description.iron_segments[1].material.b_field == (0, 2)
    direct = ImasGeometryReader(iron.segment[0].geometry).iron_segment(iron.segment[0])
    assert direct == description.iron_segments[0]


def test_empty_outline_segment_is_retained():
    iron = _iron()
    iron.segment[1].geometry.outline.r = []
    iron.segment[1].geometry.outline.z = []
    description = StaticMachineDescription.from_iron_core_ids(iron)
    assert len(description.iron_segments) == 2
    assert description.iron_segments[1].outline.r == ()
    assert description.iron_segments[1].material.b_field == (0, 2)


def test_unpaired_material_table_is_refused():
    iron = _iron()
    iron.segment[0].permeability_relative = [215]
    with pytest.raises(ValueError, match="different lengths"):
        StaticMachineDescription.from_iron_core_ids(iron)


@pytest.mark.parametrize(
    "occurrences,written", [([], "4.1.0"), ([0], "4.0.0"), ([0], "4.1.0")]
)
def test_open_uses_manifest_pin_and_refuses_missing_or_mismatched_ids(
    monkeypatch, occurrences, written
):
    iron = _iron()
    iron.ids_properties.version_put.data_dictionary = written
    calls = []

    class Entry:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append("closed")

        def list_all_occurrences(self, name):
            calls.append(("occurrences", name))
            return occurrences

        def get(self, name, occurrence, *, lazy, autoconvert):
            assert calls[-1] == ("occurrences", "iron_core")
            assert 0 in occurrences
            assert (name, occurrence, lazy, autoconvert) == (
                "iron_core",
                0,
                False,
                False,
            )
            calls.append("get")
            return iron

    def open_entry(source, mode, *, dd_version):
        assert (source, mode, dd_version) == (
            "imas:hdf5?path=description",
            "r",
            "4.1.0",
        )
        return Entry()

    monkeypatch.setattr(machine_module, "open_db_entry", open_entry)
    if not occurrences:
        with pytest.raises(ValueError, match="occurrence 0 is absent"):
            StaticMachineDescription.read_iron_core(
                "imas:hdf5?path=description", dd_version="4.1.0"
            )
        assert "get" not in calls
    elif written != "4.1.0":
        with pytest.raises(ValueError, match="disagrees with manifest"):
            StaticMachineDescription.read_iron_core(
                "imas:hdf5?path=description", dd_version="4.1.0"
            )
    else:
        assert (
            len(
                StaticMachineDescription.read_iron_core(
                    "imas:hdf5?path=description", dd_version="4.1.0"
                ).iron_segments
            )
            == 2
        )
    assert calls[-1] == "closed"


def test_missing_dictionary_pin_is_refused():
    with pytest.raises(ValueError, match="manifest DD version"):
        StaticMachineDescription.read_iron_core(
            "imas:hdf5?path=description", dd_version=""
        )


def test_callers_delegate_iron_read_to_machine_owner():
    for path in CALLERS:
        tree = ast.parse((ROOT / path).read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        assert any(
            isinstance(call.func, ast.Attribute) and call.func.attr == "read_iron_core"
            for call in calls
        )
        assert not any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "get"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == "iron_core"
            for call in calls
        )
