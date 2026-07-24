"""Tests for the top-level workbook <-> canonical transcoder."""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from nova.assembly.canonical import transcode
from nova.assembly.provenance import corpus
from tests.canonical import synthetic

# The measured workbooks live off-repo in the metrology corpus. Resolve a local
# copy if one is staged; when nothing resolves, the real-corpus tests below skip
# on the non-existent path rather than failing.
_CORPUS = corpus.resolve_corpus("appdata") or Path(
    "/mnt/c/Users/mcintos/AppData/Local/nova/sector_modules"
)
_TARGET = _CORPUS / "Sector_Module_#1_CCL_as-built_data_8LMK6A_v8_1.xlsx"
# A workbook carrying cached-value formula cells (uncertainty columns).
_FORMULA_TARGET = _CORPUS / "Sector_Module_#7_CCL_as-built_data_8NR9J7_v9_4.xlsx"


def _nominal(ws):
    synthetic.write_block(
        ws,
        header_row=6,
        header_col=4,
        coil=14,
        points=[
            {"group": "CCL", "name": "A", "xyz": (2713.7, 0.0, -3700.0)},
            {"group": "CCL", "name": "B", "xyz": (2713.7, 0.0, 3700.0)},
        ],
        has_uncertainty=False,
    )
    synthetic.write_block(
        ws,
        header_row=6,
        header_col=17,
        coil=15,
        points=[
            {"group": "CCL", "name": "A", "xyz": (2713.7, 0.0, -3700.0)},
            {"group": "CCL", "name": "B", "xyz": (2713.7, 0.0, 3700.0)},
        ],
        has_uncertainty=False,
    )


def _fat_io(ws):
    synthetic.write_transform(
        ws,
        header_col=4,
        label_row=2,
        components=[
            0.05091501080391288,
            -0.92942060445795,
            0.6158,
            0.001,
            -0.002,
            0.003,
        ],
    )
    synthetic.write_transform(
        ws,
        header_col=17,
        label_row=2,
        components=[-0.378, 0.407, 0.110, -0.012, -0.006, -0.002],
    )
    synthetic.write_block(
        ws,
        header_row=6,
        header_col=4,
        coil=14,
        points=[
            {
                "group": "Mesured ",
                "name": "A",
                "xyz": (2712.879729124093, -0.00294752, 1e-17),
                "u": (0.283, 0.299, 0.0399),
            },
            {
                "group": "best fitted ",
                "name": "F'",
                "xyz": (4284.5, 0.0, -6133.3),
                "u": (0.1, 0.2, 0.3),
            },
        ],
    )
    synthetic.write_block(
        ws,
        header_row=6,
        header_col=17,
        coil=15,
        points=[
            {
                "group": "CCL",
                "name": "A",
                "xyz": (2713.711350641258, -0.4912399936285796, -3697.486427428476),
                "u": (0.449, 0.449, 0.449),
            }
        ],
    )


def _ssat_br(ws):
    synthetic.write_block(
        ws,
        header_row=6,
        header_col=4,
        coil=14,
        points=[
            {"group": "CCL", "name": "A", "xyz": (1.0, 2.0, 3.0), "u": (0.1, 0.1, 0.1)},
            None,
            {
                "group": "Fiducial",
                "name": "A1-AU",
                "xyz": (4.0, 5.0, 6.0),
                "u": (0.2, 0.2, 0.2),
            },
            {
                "group": "ILIS +1 side",
                "name": "P9-OIS",
                "xyz": (7.0, 8.0, 9.0),
                "u": (0.3, 0.3, 0.3),
            },
        ],
    )


def _tfgs_landing(ws):
    # Empty transform (labels only) above a block that carries a formula cell.
    synthetic.write_transform(ws, header_col=4, label_row=2, components=None)
    synthetic.write_block(
        ws,
        header_row=6,
        header_col=4,
        coil=14,
        points=[
            {
                "group": "CCL",
                "name": "A",
                "xyz": ("='In-pit target'!J7+0.1", 5.0, 6.0),
                "u": (0.1, 0.2, 0.3),
            }
        ],
    )


def _full_workbook():
    return synthetic.build_workbook(
        [
            ("Nominal", _nominal),
            ("FAT IO", _fat_io),
            ("SSAT BR", _ssat_br),
            ("TFGS Landing", _tfgs_landing),
        ],
        metadata_text=[(1, 1, "Sector 1"), (3, 2, "traceability note")],
    )


def _save(tmp_path, wb, name):
    path = tmp_path / name
    wb.save(path)
    return path


def test_ingest_produces_rows_and_sidecar(tmp_path):
    path = _save(tmp_path, _full_workbook(), "book.xlsx")
    unit = transcode.ingest_workbook(path)
    assert unit.rows
    assert unit.sidecar["schema"]
    assert unit.digest.startswith("sha256:")
    assert unit.stem == "book"
    # dirty labels normalized, raw retained
    fat = [r for r in unit.rows if r.phase == "FAT IO" and r.record_kind == "fiducial"]
    assert any(
        r.point_group == "Measured" and r.point_group_raw == "Mesured " for r in fat
    )
    assert any(r.point_group == "Best Fitted" and r.feature == "F" for r in fat)


def test_round_trip_is_byte_identical(tmp_path):
    src = _save(tmp_path, _full_workbook(), "src.xlsx")
    unit1 = transcode.ingest_workbook(src)
    transcode.write_unit(unit1, tmp_path / "canon")
    reread = transcode.read_unit(tmp_path / "canon" / "src.csv")
    assert reread.rows == unit1.rows

    egressed = _save(tmp_path, transcode.egress_workbook(reread), "egress.xlsx")
    unit2 = transcode.ingest_workbook(egressed)
    assert unit2.csv_bytes == unit1.csv_bytes
    assert unit2.digest == unit1.digest


def test_float_torture_survives_round_trip(tmp_path):
    # The canonical CSV codec preserves any double exactly (proven directly in
    # test_longformat); here the values are additionally constrained to those
    # openpyxl itself can round-trip, so the workbook stage does not mask the
    # canonical guarantee.
    src = _save(tmp_path, _full_workbook(), "src.xlsx")
    unit1 = transcode.ingest_workbook(src)
    values = {
        r.value
        for r in unit1.rows
        if r.phase == "FAT IO" and r.feature == "A" and r.coil == 14
    }
    assert 2712.879729124093 in values
    assert 1e-17 in values
    assert -0.00294752 in values

    egressed = _save(tmp_path, transcode.egress_workbook(unit1), "egress.xlsx")
    unit2 = transcode.ingest_workbook(egressed)
    assert unit2.csv_bytes == unit1.csv_bytes


def test_formula_cell_captured_not_dropped(tmp_path):
    src = _save(tmp_path, _full_workbook(), "src.xlsx")
    unit = transcode.ingest_workbook(src)
    formula_rows = [r for r in unit.rows if r.phase == "TFGS Landing" and r.is_formula]
    assert formula_rows
    assert formula_rows[0].value is None
    assert any(
        "formula cell without cached value" in n for n in unit.sidecar["anomalies"]
    )


def test_empty_transform_round_trips(tmp_path):
    src = _save(tmp_path, _full_workbook(), "src.xlsx")
    unit1 = transcode.ingest_workbook(src)
    # TFGS Landing transform is empty -> no transform rows, populated False.
    assert not any(
        r.phase == "TFGS Landing" and r.record_kind == "transform" for r in unit1.rows
    )
    tfgs = next(s for s in unit1.sidecar["sheets"] if s["name"] == "TFGS Landing")
    assert tfgs["blocks"][0]["transform"]["populated"] is False


def test_ingest_is_deterministic(tmp_path):
    src = _save(tmp_path, _full_workbook(), "src.xlsx")
    first = transcode.ingest_workbook(src)
    second = transcode.ingest_workbook(src)
    assert first.csv_bytes == second.csv_bytes
    assert first.sidecar == second.sidecar
    assert first.digest == second.digest


def test_filename_metadata_and_revision_prefix(tmp_path):
    wb = _full_workbook()
    name = "_Fprime_Sector_Module_#6_CCL_as-built_data_8NQVKS_v9_0.xlsx"
    path = _save(tmp_path, wb, name)
    meta = transcode.ingest_workbook(path).sidecar["filename_meta"]
    assert meta["sector"] == 6
    assert meta["idm_doc"] == "8NQVKS"
    assert meta["version"] == "9.0"
    assert meta["private"] is True
    assert meta["revision_prefix"] == "_Fprime_"


def test_ingest_tree_skips_locks_and_records_failures(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _save(src, _full_workbook(), "Sector_Module_#1_CCL_as-built_data_8LMK6A_v8_1.xlsx")
    # An Excel lock stub (skipped by name) and a corrupt workbook (recorded).
    (src / "~$Sector_Module_#1_CCL_as-built_data_8LMK6A_v8_1.xlsx").write_bytes(b"lock")
    (src / "broken.xlsx").write_bytes(b"not a zip file")

    summary = transcode.ingest_tree(src, tmp_path / "out")
    stems = {u["stem"] for u in summary["units"]}
    assert "Sector_Module_#1_CCL_as-built_data_8LMK6A_v8_1" in stems
    assert not any("~$" in u["filename"] for u in summary["units"])
    assert [f["filename"] for f in summary["failures"]] == ["broken.xlsx"]
    # written canonical files exist
    assert (tmp_path / "out" / f"{next(iter(stems))}.csv").exists()


@pytest.mark.skipif(not _TARGET.exists(), reason="measured corpus not mounted")
def test_real_corpus_ingest_matches_survey():
    unit = transcode.ingest_workbook(_TARGET)
    sheet_names = [s["name"] for s in unit.sidecar["sheets"]]
    # 8 sheets total in the workbook: Metadata is excluded from measurements.
    assert sheet_names == [
        "Nominal",
        "FAT supplier",
        "FAT IO",
        "SSAT BR",
        "SSAT target",
        "SSAT AL",
        "In-pit target",
    ]
    assert unit.sidecar["metadata_text"]
    coils = {r.coil for r in unit.rows}
    assert coils == {14, 15}
    assert unit.sidecar["filename_meta"]["sector"] == 1
    assert unit.sidecar["filename_meta"]["idm_doc"] == "8LMK6A"

    nominal = {
        (r.coil, r.axis): r.value
        for r in unit.rows
        if r.phase == "Nominal" and r.feature == "A"
    }
    assert nominal[(14, "x")] == 2713.7
    assert nominal[(14, "z")] == -3700.0


@pytest.mark.skipif(not _TARGET.exists(), reason="measured corpus not mounted")
def test_real_corpus_round_trip_and_legacy_parity(tmp_path):
    import pandas

    unit1 = transcode.ingest_workbook(_TARGET)
    egressed = _save(tmp_path, transcode.egress_workbook(unit1), "egress.xlsx")
    unit2 = transcode.ingest_workbook(egressed)
    assert unit2.csv_bytes == unit1.csv_bytes

    # Legacy parity: the pandas reader used by SectorData.read_frame reads the
    # Nominal coil-14 block; its numeric values must agree with the canonical
    # rows exactly (the canonical path must not drop or alter measurements).
    legacy = pandas.read_excel(
        _TARGET,
        sheet_name="Nominal",
        skiprows=5,
        usecols=list(range(3, 9)),
        index_col=[0, 1, 2],
        keep_default_na=False,
        na_values="",
        dtype=dict.fromkeys(["X", "Y", "Z"], float),
    )
    legacy = legacy.rename(columns={c: c.split(".")[0].lower() for c in legacy.columns})
    legacy.index = legacy.index.set_names([n.split(".")[0] for n in legacy.index.names])
    for (_coil, _point, name), record in legacy.iterrows():
        canonical = {
            r.axis: r.value
            for r in unit1.rows
            if r.phase == "Nominal" and r.feature == name and r.coil == 14
        }
        if canonical:
            assert canonical["x"] == float(record["x"])
            assert canonical["z"] == float(record["z"])


@pytest.mark.skipif(not _FORMULA_TARGET.exists(), reason="measured corpus not mounted")
def test_real_corpus_formula_value_captured():
    unit = transcode.ingest_workbook(_FORMULA_TARGET)
    # SSAT AR uncertainty columns are =VLOOKUP-style formulas with cached
    # values; the legacy pandas reader NaN'd and dropped them.
    captured = [
        r
        for r in unit.rows
        if r.phase == "SSAT AR"
        and r.point_group == "CCL"
        and r.feature == "A"
        and r.is_formula
        and r.uncertainty is not None
    ]
    assert captured
    assert any(abs(r.uncertainty - 1.0629560774803892) < 1e-12 for r in captured)


def test_egress_reingest_openpyxl_valid(tmp_path):
    src = _save(tmp_path, _full_workbook(), "src.xlsx")
    unit = transcode.ingest_workbook(src)
    book = transcode.egress_workbook(unit)
    out = _save(tmp_path, book, "egress.xlsx")
    reopened = openpyxl.load_workbook(out, read_only=True)
    assert "FAT IO" in reopened.sheetnames
    reopened.close()
