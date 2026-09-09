"""Tests for the persisted DIII-D vacuum-field comparison contract."""

import json
from pathlib import Path

import pytest


RECEIPT = Path(
    "docs/figures/diiid-vertical-force-balance/vacuum-field-reproduction.json"
)


def test_vacuum_receipt_closes_persisted_ecoila_representation() -> None:
    if not RECEIPT.is_file():
        pytest.skip("the persisted vacuum-field receipt is not available")

    receipt = json.loads(RECEIPT.read_text())

    assert receipt["all_frames_within_roundoff_bound"] is True
    assert receipt["roundoff_bound_relative"] == 1e-12
    assert receipt["device"] == "cpu"
    frame_by_number = {frame["frame"]: frame for frame in receipt["frames"]}
    assert frame_by_number[89]["within_roundoff_bound"] is True
    assert any(frame["frame"] != 89 for frame in receipt["frames"])

    representation = frame_by_number[89]["representation"]
    ecoila = representation["ecoila"]
    assert ecoila["reference"] == "persisted pf_active element tessellation"
    assert ecoila["elements"] == 48
    assert ecoila["signed_turn_sum"] == 48.0
    assert "bulk ECOILA rectangle" in ecoila["reason"]

    provenance = frame_by_number[89]["current_vector"]["provenance"]
    assert "recorded ECOILA current" in provenance
    assert "fixed drive scale" in provenance
