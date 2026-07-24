"""Tests for the canonical sidecar builder and serializer."""

from __future__ import annotations

from nova.assembly.canonical import sidecar, workbook
from tests.canonical import synthetic

_SOURCE = {
    "filename": "Sector_Module_#1_CCL_as-built_data_8LMK6A_v8_1.xlsx",
    "sha256": "sha256:deadbeef",
    "size": 1234,
    "mtime": "2026-06-17T08:13:01+00:00",
}
_META = {
    "sector": 1,
    "idm_doc": "8LMK6A",
    "version": "8.1",
    "private": False,
    "revision_prefix": "",
}


def _parse(tmp_path, sheets, metadata_text=None):
    wb = synthetic.build_workbook(sheets, metadata_text=metadata_text)
    path = tmp_path / "book.xlsx"
    wb.save(path)
    return workbook.parse_workbook(path)


def _build(parse):
    return sidecar.build_sidecar(
        parse, source=_SOURCE, filename_meta=_META, csv_sha256="sha256:abc"
    )


def test_records_block_layout(tmp_path):
    def build(ws):
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=4,
            coil=14,
            points=[{"group": "CCL", "name": "A", "xyz": (1.0, 2.0, 3.0)}],
        )
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=17,
            coil=15,
            points=[{"group": "CCL", "name": "A", "xyz": (4.0, 5.0, 6.0)}],
            has_uncertainty=False,
        )

    doc = _build(_parse(tmp_path, [("Nominal", build)]))
    assert doc["schema"] == sidecar.SCHEMA_ID
    assert doc["source"] == _SOURCE
    assert doc["filename_meta"] == _META
    assert doc["csv_sha256"] == "sha256:abc"
    sheet = doc["sheets"][0]
    assert sheet["name"] == "Nominal"
    assert [b["coil"] for b in sheet["blocks"]] == [14, 15]
    assert sheet["blocks"][0]["header_col"] == 4
    assert sheet["blocks"][0]["width"] == 9
    assert sheet["blocks"][1]["width"] == 6
    assert sheet["blocks"][0]["transform"] is None


def test_records_transform_position(tmp_path):
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

    doc = _build(_parse(tmp_path, [("FAT IO", build)]))
    transform = doc["sheets"][0]["blocks"][0]["transform"]
    assert transform["label_row"] == 2
    assert transform["populated"] is True


def test_metadata_text_and_anomalies(tmp_path):
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

    doc = _build(_parse(tmp_path, [("SSAT BR", build)], metadata_text=[(1, 1, "note")]))
    assert {"row": 1, "col": 1, "text": "note"} in doc["metadata_text"]
    assert any("spacer row" in note for note in doc["anomalies"])
    assert all(note.startswith("SSAT BR: ") for note in doc["anomalies"])


def test_emit_is_deterministic_and_round_trips(tmp_path):
    def build(ws):
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=4,
            coil=14,
            points=[{"group": "CCL", "name": "A", "xyz": (1.0, 2.0, 3.0)}],
        )

    doc = _build(_parse(tmp_path, [("Nominal", build)]))
    first = sidecar.emit_sidecar(doc)
    second = sidecar.emit_sidecar(doc)
    assert first == second
    assert sidecar.load_sidecar_bytes(first) == doc


def test_write_and_load_file(tmp_path):
    def build(ws):
        synthetic.write_block(
            ws,
            header_row=6,
            header_col=4,
            coil=14,
            points=[{"group": "CCL", "name": "A", "xyz": (1.0, 2.0, 3.0)}],
        )

    doc = _build(_parse(tmp_path, [("Nominal", build)]))
    path = tmp_path / "book.meta.yaml"
    sidecar.write_sidecar(doc, path)
    assert sidecar.load_sidecar(path) == doc
