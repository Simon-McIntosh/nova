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
from nova.biot.magnetisation import _fit_law, _mesh as mesh
from nova.imas.machine import ImasGeometryReader, StaticMachineDescription

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
        CALLERS[0], ("machine", "segments", "table"), _namespace(west)
    )
    iron = west[2]
    assert len(namespace["segments"]) == len(iron.segment) == 6
    for actual, expected in zip(namespace["segments"], _outlines(iron), strict=True):
        np.testing.assert_array_equal(
            actual.exterior.coords, shapely.Polygon(expected).exterior.coords
        )
    np.testing.assert_array_equal(namespace["table"].b_field, iron.segment[0].b_field)
    np.testing.assert_array_equal(
        namespace["table"].permeability_relative, iron.segment[0].permeability_relative
    )
    table = namespace["table"]
    actual = _fit_law(table.b_field, table.permeability_relative)
    expected = _fit_law(
        np.asarray(iron.segment[0].b_field),
        np.asarray(iron.segment[0].permeability_relative),
    )
    for got, wanted in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, wanted)


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
    assert len(actual) == len(expected) > 0
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
            isinstance(call.func, ast.Attribute)
            and call.func.attr in ("read_iron_core", "read_west_iron_outline")
            for call in calls
        ), path
        assert not any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "get"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == "iron_core"
            for call in calls
        )


WEST_READERS = (
    "scripts/west_iron_solve/measure.py",
    "scripts/west_iron_solve/probe.py",
    "scripts/west_iron_solve/diagnose.py",
    "scripts/west_iron_solve/diagnostic_figure.py",
    "scripts/west_stray_field/solve.py",
    "scripts/west_iron_galerkin/measure.py",
    "scripts/west_iron_galerkin_pair/measure.py",
    "scripts/west_picard_control/measure.py",
    "scripts/west_root_branch/measure.py",
    "scripts/iron_graded_refinement/measure.py",
    "scripts/sheet_tile_census/measure.py",
    "scripts/magnetisation_law/figure.py",
)

FLAT_OUTLINE_NAMES = ("limb_fraction", "flat_fill", "west_regions", "WEST_JOIN")


def test_contour_moment_is_the_section_per_radian_of_a_box():
    box = ((1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 0.0))
    assert machine_module.contour_moment(box) == pytest.approx(1.5)


def test_west_outline_keeps_each_stored_section_and_the_stored_gap(west):
    uri, dd_version, _ = west
    outline = StaticMachineDescription.read_west_iron_outline(
        uri, dd_version=dd_version
    )
    arms = [region for region in outline.regions if region.kind == "arm"]
    limbs = [region for region in outline.regions if region.kind == "limb"]
    assert len(arms) == 2 and len(limbs) == 1
    assert outline.limb_arm_gap == pytest.approx(0.0021, abs=1e-12)
    assert outline.gap_input == "stored"
    assert outline.cross_field_factor == pytest.approx(0.4)
    for region in (*arms, *limbs):
        assert region.is_rectangular
        assert outline.sections[
            outline.regions.index(region)
        ].relative == pytest.approx(0.0, abs=1e-3)
    assert outline.maximum_relative_section_error(("arm", "limb")) < 1e-3
    for limb in limbs:
        r0, _, r1, _ = limb.bounds
        assert r1 - r0 == pytest.approx(outline.limb_depth)
    for arm in arms:
        _, z0, _, z1 = arm.bounds
        assert z1 - z0 == pytest.approx(outline.limb_depth)
        assert arm.contiguity_radius == pytest.approx(0.7639, abs=1e-9)
        assert arm.fill(arm.contiguity_radius) == pytest.approx(1.0)
        assert arm.fill(4.984) == pytest.approx(arm.contiguity_radius / 4.984, rel=1e-2)


def test_west_outline_refuses_any_other_gap_input(west):
    uri, dd_version, _ = west
    with pytest.raises(ValueError, match="limb_arm_gap 'closed' is not supported"):
        StaticMachineDescription.read_west_iron_outline(
            uri, dd_version=dd_version, limb_arm_gap="closed"
        )


def test_stored_arm_thickness_scaled_at_one_radius_is_refused_by_name(west):
    _, _, iron = west
    description = StaticMachineDescription.from_iron_core_ids(iron)
    segments = list(description.iron_segments)
    arm = segments[2]
    radius = list(arm.outline.r)
    height = list(arm.outline.z)
    indexed = list(zip(radius, height, strict=True))
    face = min(height)
    target = next(
        index
        for index, (r, z) in enumerate(indexed)
        if abs(r - 4.0) < 1e-9 and abs(z - face) > 1e-9
    )
    r, z = indexed[target]
    height[target] = face + (z - face) * 1.05
    segments[2] = machine_module.MachineIronSegment(
        name=arm.name,
        description=arm.description,
        outline=machine_module.MachineContour(
            kind=arm.outline.kind, r=tuple(radius), z=tuple(height)
        ),
        material=arm.material,
    )
    mutated = machine_module.StaticMachineDescription(
        contour=None,
        active_sections=(),
        passive_loop_count=0,
        toroidal_coil_count=0,
        sightlines=(),
        active_coils=(),
        wall_units=(),
        iron_segments=tuple(segments),
    )
    with pytest.raises(ValueError, match=r"segment 2 .*thickness times radius"):
        machine_module.iron_core_outline(mutated)


def test_scoped_west_readers_carry_no_flat_outline_construction():
    for path in WEST_READERS:
        source = (ROOT / path).read_text()
        tree = ast.parse(source)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        assert not names & set(FLAT_OUTLINE_NAMES), path
        assert "read_west_iron_outline" in names, path


def test_region_fill_and_cross_field_are_read_from_the_stored_section(west):
    uri, dd_version, _ = west
    outline = StaticMachineDescription.read_west_iron_outline(
        uri, dd_version=dd_version
    )
    arm = outline.region("upper arm")
    cells = np.array(
        [
            [2.0, 3.0],  # inside the upper arm panel
            [0.3, 0.0],  # inside the central column
            [0.5, 2.401],  # in the stored limb to arm gap: no region covers it
        ]
    )
    assert arm.covered(cells).tolist() == [True, False, False]
    fill = outline.fill(cells)
    assert fill[0] == pytest.approx(arm.fill([2.0])[0])
    assert 0 < fill[0] < 1
    assert fill[1] == pytest.approx(1.0)
    assert fill[2] == pytest.approx(0.0)
    factor = outline.cross_field(cells)
    assert factor[0].tolist() == [0.0, pytest.approx(outline.cross_field_factor)]
    assert factor[1].tolist() == [0.0, 0.0]
    assert factor[2].tolist() == [0.0, 0.0]
