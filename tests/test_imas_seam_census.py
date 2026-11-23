"""Contract tests for the IMAS seam census.

The census is the import-graph scan that the seam plan's machine sections read.
These tests hold the properties that make its coverage claim meaningful: every
censused file carries a row, the classes map to the destinations the plan acts
on, the controls are populated, and the record renders the rows it counted.
"""

from __future__ import annotations

import re  # noqa: F401
from pathlib import Path

import pytest

from scripts import imas_seam_census as census_module

ROOT = Path(__file__).resolve().parents[1]
AMBIX = Path("/home/ITER/mcintos/Code/imas-ambix")


@pytest.fixture(scope="module")
def census():
    return census_module.build_census(ROOT, AMBIX if AMBIX.is_dir() else None)


def test_every_censused_file_carries_a_row(census):
    assert census.counts["candidate_files_unrowed"] == 0
    row_paths = {row.path for row in census.rows}
    for path in sorted((ROOT / "nova" / "imas").glob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        assert rel in row_paths, rel


def test_every_machine_named_module_outside_imas_carries_a_row(census):
    row_paths = {row.path for row in census.rows}
    pattern = re.compile(r"^(mast|diiid|machine)_")
    for path in sorted((ROOT / "nova").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if not pattern.match(path.stem):
            continue
        rel = path.relative_to(ROOT).as_posix()
        assert rel in row_paths, rel


def test_the_scan_reads_the_trees_the_plan_covers(census):
    assert set(census_module.TREE_DIRECTORIES) <= set(census.trees)
    assert census.files_seen > 0


def test_the_row_count_states_the_files_it_covers(census):
    counts = census.counts
    assert counts["rows_total"] == (
        counts["module_rows"] + counts["importer_rows"] + counts["convention_rows"]
    )
    assert counts["files_with_module_rows"] == len({row.path for row in census.rows})


def test_each_destination_follows_its_class(census):
    for row in census.rows:
        assert row.destination == census_module.CLASS_DESTINATION[row.cls]


def test_the_classes_cover_every_row(census):
    for row in census.rows:
        assert row.cls in census_module.CLASS_GLOSSARY


def test_the_importer_rows_name_a_censused_module(census):
    censused = {row.module for row in census.rows}
    assert census.importer_rows
    for row in census.importer_rows:
        assert set(row.imports) & censused
        assert row.destination == census_module.CLASS_DESTINATION[row.cls]


def test_the_dataset_control_reports_known_importers(census):
    assert census.dataset_control
    modules = {site.rsplit(":", 1)[0] for site in census.dataset_control}
    assert "nova.equilibrium.inverse" in modules
    assert "nova.io.egress" in modules


def test_the_convention_control_reports_the_statement_and_the_conversions(census):
    control = dict(census.convention_control)
    assert dict(census.convention_control)["nova/equilibrium/convention.py"] == (
        "convention statement"
    )
    assert control["nova/io/ingest.py"] == "convention conversion"
    assert control["nova/io/signalstore.py"] == "convention conversion"


def test_the_convention_scan_finds_every_converters_import_of_the_algebra(census):
    paths = {row.path for row in census.convention_rows}
    assert "nova/equilibrium/convention.py" in paths
    assert "nova/io/signalstore.py" in paths


def test_duplicate_capabilities_name_both_implementations(census):
    rows = {row.path: row for row in census.rows}
    row = rows["nova/imas/mast_geometry.py"]
    assert row.duplicates
    assert "nova/catalog/mast_geometry.py:169" in row.duplicates
    other = {duplicate.rsplit(":", 1)[0] for duplicate in row.duplicates}
    assert "nova/catalog/mast_geometry.py" in other


def test_the_unreached_rows_are_the_modules_nothing_imports(census):
    unreached = {row.path for row in census.rows if row.cls == "unreached"}
    expected = {
        "nova/imas/decimate.py",
        "nova/imas/fuse.py",
        "nova/imas/transient.py",
        "nova/imas/toroidal_field_coil.py",
        "nova/imas/waveform.py",
    }
    assert expected <= unreached


def test_the_reference_shots_name_a_shot_per_machine(census):
    machines = {shot["machine"]: shot["shot"] for shot in census.reference_shots}
    assert machines["MAST"] == "21978"
    assert machines["DIII-D"]


def test_the_machine_rows_name_their_upstream_map(census):
    rows = {row.path: row for row in census.rows}
    assert rows["nova/imas/mast_geometry.py"].ambix_map
    maps = rows["nova/imas/mast_geometry.py"].ambix_map
    assert all("machine_maps" in path for path in maps)


@pytest.mark.skipif(not AMBIX.is_dir(), reason="imas-ambix checkout is not present")
def test_the_ambix_source_and_tests_appear_as_importers(census):
    ambix_rows = [
        row for row in census.importer_rows if row.path.startswith("imas_ambix/")
    ]
    assert ambix_rows


def test_the_record_renders_the_rows(census):
    page = census_module.render_html(census)
    assert str(census.counts["module_rows"]) in page
    assert "nova/imas/mast_geometry.py" in page
    assert page.startswith("<!DOCTYPE html>")


def test_check_exits_under_the_covering_scan(tmp_path):
    target = tmp_path / "census.html"
    status = census_module.main(
        ["--root", str(ROOT), "--ambix", str(AMBIX), "--html", str(target)]
        if AMBIX.is_dir()
        else [
            "--root",
            str(ROOT),
            "--ambix",
            str(tmp_path / "absent"),
            "--html",
            str(target),
        ]
    )
    assert status == 0
    assert target.is_file()
