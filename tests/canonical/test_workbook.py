"""Tests for the structural workbook parser."""

from __future__ import annotations

import openpyxl

from nova.assembly.canonical import workbook
from tests.canonical import synthetic


def _save(tmp_path, wb, name="book.xlsx"):
    path = tmp_path / name
    wb.save(path)
    return path


def test_locates_two_side_by_side_blocks(tmp_path):
    def build(ws):
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=4,
            coil=14,
            points=[{"group": "CCL", "name": "A", "xyz": (1.0, 2.0, 3.0)}],
            has_uncertainty=False,
        )
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=17,
            coil=15,
            points=[{"group": "CCL", "name": "A", "xyz": (4.0, 5.0, 6.0)}],
            has_uncertainty=False,
        )

    path = _save(tmp_path, synthetic.build_workbook([("Nominal", build)]))
    parse = workbook.parse_workbook(path)
    assert len(parse.sheets) == 1
    sheet = parse.sheets[0]
    assert sheet.name == "Nominal"
    assert [b.coil for b in sheet.blocks] == [14, 15]
    assert [b.header_col for b in sheet.blocks] == [4, 17]
    assert sheet.blocks[0].width == workbook.WIDTH_NO_UNCERTAINTY
    assert sheet.blocks[0].has_uncertainty is False
    assert sheet.blocks[0].fiducials[0].coords["x"].value == 1.0
    assert sheet.blocks[1].fiducials[0].coords["z"].value == 6.0


def test_single_block_sheet(tmp_path):
    def build(ws):
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=4,
            coil=8,
            points=[{"group": "CCL", "name": "A", "xyz": (1.0, 2.0, 3.0)}],
        )

    path = _save(tmp_path, synthetic.build_workbook([("FAT IO", build)]))
    parse = workbook.parse_workbook(path)
    assert len(parse.sheets[0].blocks) == 1
    assert parse.sheets[0].blocks[0].width == workbook.WIDTH_WITH_UNCERTAINTY


def test_uncertainty_columns_read(tmp_path):
    def build(ws):
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=4,
            coil=14,
            points=[
                {
                    "group": "CCL",
                    "name": "A",
                    "xyz": (1.0, 2.0, 3.0),
                    "u": (0.1, 0.2, 0.3),
                }
            ],
        )

    path = _save(tmp_path, synthetic.build_workbook([("FAT IO", build)]))
    fid = workbook.parse_workbook(path).sheets[0].blocks[0].fiducials[0]
    assert fid.uncerts["x"].value == 0.1
    assert fid.uncerts["z"].value == 0.3


def test_forward_fill_point_group(tmp_path):
    def build(ws):
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=4,
            coil=14,
            points=[
                {"group": "CCL", "name": "A", "xyz": (1.0, 2.0, 3.0)},
                {"group": None, "name": "B", "xyz": (4.0, 5.0, 6.0)},
                {"group": "Fiducial", "name": "C", "xyz": (7.0, 8.0, 9.0)},
            ],
        )

    path = _save(tmp_path, synthetic.build_workbook([("SSAT BR", build)]))
    fids = workbook.parse_workbook(path).sheets[0].blocks[0].fiducials
    assert [f.point_group_raw for f in fids] == ["CCL", "CCL", "Fiducial"]
    assert [f.name_raw for f in fids] == ["A", "B", "C"]


def test_spacer_between_labelled_groups_is_benign(tmp_path):
    def build(ws):
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=4,
            coil=14,
            points=[
                {"group": "CCL", "name": "A", "xyz": (1.0, 2.0, 3.0)},
                None,
                {"group": "Fiducial", "name": "B", "xyz": (4.0, 5.0, 6.0)},
            ],
        )

    path = _save(tmp_path, synthetic.build_workbook([("SSAT BR", build)]))
    sheet = workbook.parse_workbook(path).sheets[0]
    # spacer dropped, both points kept, no forward-fill leak flagged
    assert len(sheet.blocks[0].fiducials) == 2
    assert not any("forward-filled" in note for note in sheet.anomalies)


def test_spacer_that_forward_fills_group_is_flagged(tmp_path):
    def build(ws):
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=4,
            coil=14,
            points=[
                {"group": "CCL", "name": "A", "xyz": (1.0, 2.0, 3.0)},
                None,
                {"group": None, "name": "B", "xyz": (4.0, 5.0, 6.0)},
            ],
        )

    path = _save(tmp_path, synthetic.build_workbook([("SSAT BR", build)]))
    sheet = workbook.parse_workbook(path).sheets[0]
    fids = sheet.blocks[0].fiducials
    assert len(fids) == 2
    # the unlabelled row inherits CCL across the blank gap
    assert fids[1].point_group_raw == "CCL"
    assert any("forward-filled across blank" in note for note in sheet.anomalies)


def test_formula_cell_without_cache_flagged(tmp_path):
    def build(ws):
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=4,
            coil=14,
            points=[
                {"group": "CCL", "name": "A", "xyz": ("=1+2", 5.0, 6.0)},
            ],
        )

    path = _save(tmp_path, synthetic.build_workbook([("SSAT AR", build)]))
    sheet = workbook.parse_workbook(path).sheets[0]
    cell = sheet.blocks[0].fiducials[0].coords["x"]
    assert cell.is_formula is True
    assert cell.value is None
    assert any("formula cell without cached value" in n for n in sheet.anomalies)


def test_transform_block_parsed_and_tied_to_coil(tmp_path):
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

    path = _save(tmp_path, synthetic.build_workbook([("FAT IO", build)]))
    sheet = workbook.parse_workbook(path).sheets[0]
    assert len(sheet.transforms) == 1
    transform = sheet.transforms[0]
    assert transform.coil == 14
    assert transform.header_col == 4
    assert transform.populated is True
    assert transform.components["dx"].value == 0.1
    assert transform.components["rz"].value == 0.003


def test_empty_transform_block(tmp_path):
    def build(ws):
        synthetic.write_transform(ws, header_col=4, label_row=2, components=None)
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=4,
            coil=14,
            points=[{"group": "CCL", "name": "A", "xyz": (1.0, 2.0, 3.0)}],
        )

    path = _save(tmp_path, synthetic.build_workbook([("TFGS Landing", build)]))
    transform = workbook.parse_workbook(path).sheets[0].transforms[0]
    assert transform.populated is False


def test_metadata_text_captured_and_sheet_excluded(tmp_path):
    def build(ws):
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=4,
            coil=14,
            points=[{"group": "CCL", "name": "A", "xyz": (1.0, 2.0, 3.0)}],
        )

    wb = synthetic.build_workbook(
        [("Nominal", build)],
        metadata_text=[(1, 1, "Sector 1"), (2, 1, "traceability note")],
    )
    path = _save(tmp_path, wb)
    parse = workbook.parse_workbook(path)
    assert [s.name for s in parse.sheets] == ["Nominal"]
    assert (1, 1, "Sector 1") in parse.metadata_text
    assert (2, 1, "traceability note") in parse.metadata_text


def test_read_only_open_does_not_modify_source(tmp_path):
    def build(ws):
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=4,
            coil=14,
            points=[{"group": "CCL", "name": "A", "xyz": (1.0, 2.0, 3.0)}],
        )

    path = _save(tmp_path, synthetic.build_workbook([("Nominal", build)]))
    before = path.read_bytes()
    workbook.parse_workbook(path)
    assert path.read_bytes() == before
    # sanity: file still opens
    openpyxl.load_workbook(path, read_only=True).close()
