"""Tests for canonical long-format rows and their CSV encoding."""

from __future__ import annotations

from nova.assembly.canonical import longformat, workbook
from tests.canonical import synthetic


def _parse(tmp_path, sheets, metadata_text=None):
    wb = synthetic.build_workbook(sheets, metadata_text=metadata_text)
    path = tmp_path / "book.xlsx"
    wb.save(path)
    return workbook.parse_workbook(path)


def test_normalize_point_group():
    assert longformat.normalize_point_group("Mesured ") == "Measured"
    assert longformat.normalize_point_group("Measured") == "Measured"
    assert longformat.normalize_point_group("best fitted ") == "Best Fitted"
    assert longformat.normalize_point_group("Best Fit") == "Best Fitted"
    assert longformat.normalize_point_group("CCL") == "CCL"
    assert longformat.normalize_point_group("ILIS +1 side") == "ILIS +1 side"
    assert longformat.normalize_point_group(None) == ""


def test_normalize_feature():
    assert longformat.normalize_feature("F'") == "F"
    assert longformat.normalize_feature("A1-AU") == "A1-AU"
    assert longformat.normalize_feature(" B ") == "B"


def test_fiducial_becomes_three_axis_rows(tmp_path):
    def build(ws):
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=4,
            coil=14,
            points=[
                {
                    "group": "Best Fit",
                    "name": "F'",
                    "xyz": (1.0, 2.0, 3.0),
                    "u": (0.1, 0.2, 0.3),
                }
            ],
        )

    rows = longformat.rows_from_parse(_parse(tmp_path, [("Nominal", build)]))
    assert [r.axis for r in rows] == ["x", "y", "z"]
    assert all(r.record_kind == "fiducial" for r in rows)
    assert rows[0].point_group == "Best Fitted"
    assert rows[0].point_group_raw == "Best Fit"
    assert rows[0].feature == "F"
    assert rows[0].feature_raw == "F'"
    assert rows[0].value == 1.0
    assert rows[0].uncertainty == 0.1
    assert rows[0].units == "mm"
    assert rows[2].value == 3.0


def test_transform_rows_before_fiducials_and_units(tmp_path):
    def build(ws):
        synthetic.write_transform(
            ws,
            header_col=4,
            label_row=2,
            components=[0.1, -0.2, 0.3, 0.001, -0.002, 0.003],
        )
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=4,
            coil=14,
            points=[{"group": "CCL", "name": "A", "xyz": (1.0, 2.0, 3.0)}],
        )

    rows = longformat.rows_from_parse(_parse(tmp_path, [("FAT IO", build)]))
    transform_rows = [r for r in rows if r.record_kind == "transform"]
    assert [r.axis for r in transform_rows] == list(workbook.TRANSFORM_AXES)
    assert transform_rows[0].units == "mm"
    assert transform_rows[3].units == "deg"
    assert transform_rows[0].coil == 14
    # transform rows precede the fiducial rows
    assert rows[: len(transform_rows)] == transform_rows


def test_empty_transform_emits_no_rows(tmp_path):
    def build(ws):
        synthetic.write_transform(ws, header_col=4, label_row=2, components=None)
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=4,
            coil=14,
            points=[{"group": "CCL", "name": "A", "xyz": (1.0, 2.0, 3.0)}],
        )

    rows = longformat.rows_from_parse(_parse(tmp_path, [("TFGS Landing", build)]))
    assert not any(r.record_kind == "transform" for r in rows)


def test_csv_round_trips_rows(tmp_path):
    def build(ws):
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=4,
            coil=14,
            points=[
                {"group": "CCL", "name": "A", "xyz": (1.0, 2.0, 3.0)},
                {"group": "Fiducial", "name": "B", "xyz": (4.0, 5.0, 6.0)},
            ],
        )

    rows = longformat.rows_from_parse(_parse(tmp_path, [("Nominal", build)]))
    data = longformat.rows_to_csv_bytes(rows)
    assert longformat.csv_bytes_to_rows(data) == rows


def test_csv_is_deterministic_and_lf_terminated(tmp_path):
    def build(ws):
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=4,
            coil=14,
            points=[{"group": "CCL", "name": "A", "xyz": (1.0, 2.0, 3.0)}],
        )

    rows = longformat.rows_from_parse(_parse(tmp_path, [("Nominal", build)]))
    first = longformat.rows_to_csv_bytes(rows)
    second = longformat.rows_to_csv_bytes(rows)
    assert first == second
    assert b"\r\n" not in first
    assert first.startswith(",".join(longformat.COLUMNS).encode() + b"\n")


def test_float_torture_survives_exactly():
    torture = [2712.4100000000003, -0.00294752, 1e-17, -3699.619919541942]
    rows = [
        longformat.Row(
            coil=14,
            phase="SSAT BR",
            record_kind="fiducial",
            point_group="CCL",
            point_group_raw="CCL",
            feature="A",
            feature_raw="A",
            axis="x",
            value=value,
            uncertainty=None,
            units="mm",
            is_formula=False,
        )
        for value in torture
    ]
    recovered = longformat.csv_bytes_to_rows(longformat.rows_to_csv_bytes(rows))
    assert [r.value for r in recovered] == torture


def test_formula_flag_preserved_through_csv(tmp_path):
    def build(ws):
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=4,
            coil=14,
            points=[{"group": "CCL", "name": "A", "xyz": ("=1+2", 5.0, 6.0)}],
        )

    rows = longformat.rows_from_parse(_parse(tmp_path, [("SSAT AR", build)]))
    recovered = longformat.csv_bytes_to_rows(longformat.rows_to_csv_bytes(rows))
    assert recovered[0].is_formula is True
    assert recovered[0].value is None
