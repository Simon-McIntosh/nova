"""Promotion writes measured acquisition blocks into the correction document."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
import yaml

from nova.calibrate.correction_model import CorrectionKind
from nova.calibrate.correction_set import load_correction_set
from nova.imas.mast_block_scale import (
    BlockScale,
    BlockScaleTable,
    CorrectionSetScales,
)
from nova.scripts.mast_acquisition_sweep import promote_acquisition_corrections

BANKED_TABLE = Path(__file__).parent / "data" / "banked_block_scales.json"
MAST_DOCUMENT = (
    Path(__file__).parents[2]
    / "nova"
    / "calibrate"
    / "corrections"
    / "mast"
    / "magnetics.yaml"
)


def correction_document():
    """Return a small valid document with an acquisition ladder and retained gain."""

    return {
        "machine": "mast",
        "diagnostic_system": "magnetics",
        "schema_version": "1.0.0",
        "set_version": "3.4.5",
        "generated_by": "test fixture",
        "generated_at": "2026-08-11",
        "ladders": [
            {
                "name": "acquisition_range",
                "kind": "acquisition_scale",
                "rungs": [0.5, 1.0, 2.0],
                "tolerance": 0.08,
            }
        ],
        "corrections": [
            {
                "channel": "retained_probe",
                "kind": "gain",
                "status": "recorded",
                "value": 0.9,
                "validity": [{}],
                "provenance": {
                    "method": "independent calibration",
                    "evidence_uri": "evidence/retained.json",
                },
            },
            {
                "channel": "superseded_probe",
                "kind": "acquisition_scale",
                "status": "promoted",
                "value": 1.0,
                "measured_value": 0.99,
                "ladder": "acquisition_range",
                "validity": [
                    {
                        "pulse_start": 1,
                        "pulse_end": 2,
                        "measured_pulses": [1, 2],
                    }
                ],
                "provenance": {
                    "method": "earlier sweep",
                    "evidence_uri": "evidence/earlier.json",
                },
            },
        ],
    }


def test_promotion_round_trips_through_the_correction_engine(tmp_path):
    path = tmp_path / "magnetics.yaml"
    path.write_text(yaml.safe_dump(correction_document(), sort_keys=False))
    table = BlockScaleTable.create(
        [
            BlockScale(
                channel="measured_probe",
                scale=1.94,
                rung=2.0,
                shots=(101, 102),
                route="far-field response ratio",
            )
        ],
        route="far-field response ratio",
    )

    written = promote_acquisition_corrections(
        table,
        path,
        evidence_uri="evidence/current-sweep.json",
        fitted_at=date(2026, 8, 12),
    )
    loaded = load_correction_set(path)

    assert written == loaded
    assert loaded.set_version == "3.4.6"
    assert loaded.generated_at == date(2026, 8, 12)
    assert loaded.generated_by == "nova.scripts.mast_acquisition_sweep"
    assert all(row.channel != "superseded_probe" for row in loaded.corrections)
    assert any(row.channel == "retained_probe" for row in loaded.corrections)

    promoted = [
        row
        for row in loaded.corrections
        if row.kind == CorrectionKind.acquisition_scale
    ]
    assert len(promoted) == 1
    assert promoted[0].value == 2.0
    assert promoted[0].measured_value == 1.94
    assert promoted[0].provenance.evidence_uri == "evidence/current-sweep.json"
    assert promoted[0].provenance.fitted_at == date(2026, 8, 12)
    assert promoted[0].provenance.fitted_by == "nova.scripts.mast_acquisition_sweep"
    reader = CorrectionSetScales.create(loaded)
    assert reader.correction("measured_probe", 101).scale == pytest.approx(2.0)


def test_off_ladder_measurements_are_written_as_refused_records(tmp_path):
    path = tmp_path / "magnetics.yaml"
    path.write_text(yaml.safe_dump(correction_document(), sort_keys=False))
    table = BlockScaleTable.create(
        [
            BlockScale(
                channel="measured_probe",
                scale=1.31,
                rung=float("nan"),
                shots=(201, 202),
                route="far-field response ratio",
            )
        ]
    )

    loaded = promote_acquisition_corrections(
        table,
        path,
        evidence_uri="evidence/current-sweep.json",
        fitted_at=date(2026, 8, 12),
    )

    refused = next(
        row
        for row in loaded.corrections
        if row.kind == CorrectionKind.acquisition_scale
    )
    assert refused.status == "refused"
    assert refused.value is None
    assert refused.measured_value == 1.31
    reader = CorrectionSetScales.create(loaded)
    result = reader.correction("measured_probe", 201)
    assert not result.applied
    assert result.candidates == (1.31,)


def test_the_banked_table_stamps_every_promoted_record(tmp_path):
    path = tmp_path / "magnetics.yaml"
    path.write_text(MAST_DOCUMENT.read_text())
    before = load_correction_set(path)
    table = BlockScaleTable.from_dict(json.loads(BANKED_TABLE.read_text()))

    loaded = promote_acquisition_corrections(
        table,
        path,
        evidence_uri="evidence/banked-sweep.json",
        fitted_at=date(2026, 8, 12),
    )

    previous = [int(part) for part in before.set_version.split(".")]
    current = [int(part) for part in loaded.set_version.split(".")]
    assert current == [previous[0], previous[1], previous[2] + 1]
    acquisition = [
        row
        for row in loaded.corrections
        if row.kind == CorrectionKind.acquisition_scale
    ]
    promoted = [row for row in acquisition if row.status == "promoted"]
    refused = [row for row in acquisition if row.status == "refused"]
    assert len(acquisition) == 113
    assert len(promoted) == 108
    assert len(refused) == 5
    assert all(
        row.provenance.evidence_uri == "evidence/banked-sweep.json"
        and row.provenance.fitted_at == date(2026, 8, 12)
        and row.provenance.fitted_by == "nova.scripts.mast_acquisition_sweep"
        for row in promoted
    )
