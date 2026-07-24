"""Tests for the historical fit-parameter manifest.

The in-repo canonical reference units under ``data/Assembly/sector_modules``
are the golden dataset; the tests never require the off-repo corpus. One test
does rebuild against that corpus when it is present and skips cleanly with a
visible reason otherwise.
"""

from pathlib import Path

import pytest

from nova.assembly.provenance import fithistory, yamlio

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN_DIR = _REPO_ROOT / "data" / "Assembly" / "sector_modules"
_COMMITTED_MANIFEST = _REPO_ROOT / "data" / "Assembly" / "fit_manifest.yaml"

# The digest-pinned canonical snapshot the committed manifest is built from,
# if it has been pulled onto this machine.
_SNAPSHOT_GLOB = ".local/share/norma/canonical-*/canonical-*/sector_modules"


def _snapshot_dirs():
    """Return the off-repo canonical snapshot dirs, or None when absent."""
    matches = sorted(Path.home().glob(_SNAPSHOT_GLOB))
    if not matches:
        return None
    base = matches[-1]
    dirs = [base]
    if (base / "IDM").is_dir():
        dirs.append(base / "IDM")
    return dirs


def _golden_csvs():
    """Return the golden canonical CSV paths."""
    return sorted(_GOLDEN_DIR.glob("*.csv"))


# --- parameter-epoch evolution ---


def test_parameter_epochs_are_ordered_and_chained():
    """Epochs are oldest-first with half-open date ranges that chain."""
    epochs = fithistory.parameter_epochs()
    assert len(epochs) == 10
    dates = [epoch["start_date"] for epoch in epochs]
    assert dates == sorted(dates)
    for older, newer in zip(epochs, epochs[1:]):
        assert older["end_date"] == newer["start_date"]
    assert epochs[-1]["end_date"] is None


def test_parameter_epochs_carry_full_parameter_set():
    """Every epoch pins the full set of fit controls."""
    for epoch in fithistory.parameter_epochs():
        params = epoch["parameters"]
        for key in (
            "method",
            "infer",
            "weights",
            "radial_offset",
            "fiducial_index",
            "transform_dof",
            "gpr",
            "coupled",
        ):
            assert key in params, key
        assert set(params["fiducial_index"]) == {"radial", "toroidal", "vertical"}
        assert len(epoch["commit"]) == 8


def test_parameter_evolution_transitions():
    """The mined transitions land in the recorded epochs."""
    by_commit = {e["commit"]: e["parameters"] for e in fithistory.parameter_epochs()}
    # Radial offset absent before its introduction, then the gap-limit value.
    assert by_commit["c5f52b8e"]["radial_offset"] == 0.0
    assert by_commit["d15ec606"]["radial_offset"] == pytest.approx(
        -0.47109863, abs=1e-8
    )
    # Toroidal constraint set reduces from all fiducials to a subset.
    assert by_commit["d15ec606"]["fiducial_index"]["toroidal"] is None
    assert by_commit["f447a63e"]["fiducial_index"]["toroidal"] == [0, 5, 3, 4]
    # Random seed pinned; vertical weight reduced; coupled switch added.
    assert by_commit["20e72bfc"]["gpr"]["random_state"] == 2025
    assert by_commit["f85e5d90"]["weights"] == [1.0, 1.0, 0.25]
    assert by_commit["093eea55"]["coupled"] is None
    assert by_commit["a18dad0d"]["coupled"] is True


# --- date attribution ---


def test_epoch_for_date_selects_newest_not_after():
    """A date selects the newest epoch on or before it."""
    assert fithistory.epoch_for_date("2020-01-01")["commit"] == "c5f52b8e"
    assert fithistory.epoch_for_date("2024-01-16")["commit"] == "98983c19"
    assert fithistory.epoch_for_date("2024-01-17")["commit"] == "d15ec606"
    assert fithistory.epoch_for_date("2030-01-01")["commit"] == "eac8c052"


def test_attribute_epoch_records_evidence():
    """Attribution is marked inferred and carries its evidence."""
    attribution = fithistory.attribute_epoch("2025-03-01T09:00:00+00:00")
    assert attribution["inferred"] is True
    assert attribution["epoch_commit"] == "20e72bfc"
    evidence = attribution["evidence"]
    assert evidence["basis"] == "source_mtime"
    assert evidence["source_mtime"] == "2025-03-01T09:00:00+00:00"
    assert evidence["epoch_date_range"][0] == "2025-02-21"
    assert "upper-bound" in evidence["note"]


# --- fit extraction from golden units ---


def test_fit_name_signal():
    """The name heuristic flags target/fit sheets, not plain phase names."""
    assert fithistory._fit_name_signal("SSAT target")
    assert fithistory._fit_name_signal("In-pit target")
    assert not fithistory._fit_name_signal("FAT IO")
    assert not fithistory._fit_name_signal("SSAT AL")


def test_iter_fits_recovers_transform_and_attribution():
    """A golden unit's recorded transform and inferred parameters are present."""
    csv = _GOLDEN_DIR / "Sector_Module_#1_CCL_as-built_data_8LMK6A_v8_1.csv"
    fits = fithistory.iter_fits(csv)
    assert {fit["coil"] for fit in fits} == {14, 15}
    fat_io_14 = next(
        fit for fit in fits if fit["sheet"] == "FAT IO" and fit["coil"] == 14
    )
    transform = fat_io_14["recorded"]["transform"]
    assert set(transform) == set(fithistory.TRANSFORM_AXES)
    assert transform["dx"] == pytest.approx(0.05091501080391288)
    assert transform["dy"] == pytest.approx(-0.92942060445795)
    assert fat_io_14["sector"] == 1
    assert fat_io_14["version"] == "8.1"
    assert fat_io_14["inferred"]["inferred"] is True


def test_iter_fits_sorted_by_sheet_then_coil():
    """Fits from one unit come back in a deterministic order."""
    csv = _GOLDEN_DIR / "Sector_Module_#1_CCL_as-built_data_8LMK6A_v8_1.csv"
    fits = fithistory.iter_fits(csv)
    keys = [(fit["sheet"], fit["coil"]) for fit in fits]
    assert keys == sorted(keys)


# --- manifest over the golden corpus ---


def test_build_manifest_golden_counts():
    """The golden corpus yields the expected inventory."""
    manifest = fithistory.build_fit_manifest([_GOLDEN_DIR], "goldens")
    summary = manifest["summary"]
    assert summary["units_scanned"] == len(_golden_csvs())
    assert summary["fits"] == len(manifest["fits"])
    assert summary["fits"] > 0
    assert summary["fits_with_complete_transform"] == summary["fits"]
    # Every fit entry is well-formed.
    for fit in manifest["fits"]:
        assert set(fit["recorded"]["transform"]) == set(fithistory.TRANSFORM_AXES)
        assert fit["inferred"]["inferred"] is True
        assert fit["csv_sha256"].startswith("sha256:")


def test_manifest_fits_globally_sorted():
    """Manifest fits are ordered by unit, then sheet, then coil."""
    manifest = fithistory.build_fit_manifest([_GOLDEN_DIR], "goldens")
    keys = [
        (fit["unit"], fit["sheet"], (fit["coil"] is not None, fit["coil"] or 0))
        for fit in manifest["fits"]
    ]
    assert keys == sorted(keys)


def test_rebuild_byte_identical():
    """Rebuilding over the unchanged golden corpus is byte-identical."""
    first = yamlio.canonical_yaml_bytes(
        fithistory.build_fit_manifest([_GOLDEN_DIR], "goldens")
    )
    second = yamlio.canonical_yaml_bytes(
        fithistory.build_fit_manifest([_GOLDEN_DIR], "goldens")
    )
    assert first == second


def test_write_load_roundtrip(tmp_path):
    """A written manifest loads back equal to the built document."""
    manifest = fithistory.build_fit_manifest([_GOLDEN_DIR], "goldens")
    out = tmp_path / "fit_manifest.yaml"
    fithistory.write_fit_manifest(manifest, out)
    assert fithistory.load_fit_manifest(out) == manifest


def test_header_validation_rejects_foreign_csv(tmp_path):
    """A CSV whose header is not the canonical schema is rejected."""
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    (tmp_path / "bad.meta.yaml").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        fithistory._read_rows(bad)


# --- committed artifact ---


def test_committed_manifest_is_valid():
    """The committed manifest loads and satisfies its own invariants."""
    manifest = fithistory.load_fit_manifest(_COMMITTED_MANIFEST)
    assert manifest["schema"] == fithistory.SCHEMA_ID
    assert len(manifest["parameter_epochs"]) == 10
    assert manifest["summary"]["fits"] == len(manifest["fits"])
    for fit in manifest["fits"]:
        assert set(fit["recorded"]["transform"]) == set(fithistory.TRANSFORM_AXES)
        assert fit["inferred"]["inferred"] is True


def test_committed_manifest_regenerates_byte_identical():
    """Rebuilding from the pinned snapshot reproduces the committed bytes."""
    dirs = _snapshot_dirs()
    if dirs is None:
        pytest.skip("off-repo canonical snapshot not present on this machine")
    committed = _COMMITTED_MANIFEST.read_bytes()
    description = fithistory.load_fit_manifest(_COMMITTED_MANIFEST)["corpus"]
    rebuilt = yamlio.canonical_yaml_bytes(
        fithistory.build_fit_manifest(dirs, description)
    )
    assert rebuilt == committed
