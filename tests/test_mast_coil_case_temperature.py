"""Pin the cited steel conversion and refusal to invent can temperatures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.mast_coil_case_temperature import (
    FIT_CEILING,
    _refuse_unmapped_temperature,
    _temperature_channels,
    crossing_temperature,
    reference_resistivity,
)


@pytest.mark.parametrize(
    ("temperature_k", "expected"),
    [(300.0, 0.811), (400.0, 0.890), (450.0, 0.923)],
)
def test_cited_reference_points(temperature_k: float, expected: float) -> None:
    assert reference_resistivity(temperature_k) == pytest.approx(expected)


def test_synthetic_can_temperature_crosses_the_fit_ceiling() -> None:
    cooler_can_k = 350.0
    warmer_can_k = 450.0
    assert warmer_can_k - cooler_can_k == 100.0
    assert reference_resistivity(cooler_can_k) < FIT_CEILING
    assert reference_resistivity(warmer_can_k) > FIT_CEILING
    assert crossing_temperature(FIT_CEILING) == pytest.approx(415.15151515)


def test_reference_conversion_refuses_extrapolation() -> None:
    with pytest.raises(ValueError, match="outside 300–450 K"):
        reference_resistivity(299.0)
    with pytest.raises(ValueError, match="outside 300–450 K"):
        reference_resistivity(451.0)


def test_plasma_temperature_cannot_be_classified_as_can_evidence() -> None:
    channels = _temperature_channels(
        {
            "act/ss_temperature/.zattrs": {
                "name": "act/ss_temperature",
                "label": "Carbon temperature",
                "units": "eV",
            },
            "case/coil_case_temperature/.zattrs": {
                "name": "case/coil_case_temperature",
                "label": "Coil case temperature",
                "units": "K",
            },
        }
    )
    assert [row["path"] for row in channels["unrelated"]] == ["act/ss_temperature"]
    assert [row["path"] for row in channels["case"]] == ["case/coil_case_temperature"]
    with pytest.raises(ValueError, match="sensor-to-can mapping"):
        _refuse_unmapped_temperature(11884, channels)


def test_receipt_keeps_every_fitted_shot_unknown_without_can_evidence() -> None:
    receipt_path = (
        Path(__file__).resolve().parents[1]
        / "docs/figures/mast-coil-case-temperature/receipt.json"
    )
    if not receipt_path.exists():
        pytest.skip("data pass has not produced the receipt")
    receipt = json.loads(receipt_path.read_text())
    split = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/figures/mast-passive-held-out/split.json"
        ).read_text()
    )
    assert receipt["counts"]["training"] == 131
    assert receipt["counts"]["held_out"] == 37
    assert receipt["positive_control"]["metadata_records_present"] == 168
    assert {row["shot"] for row in receipt["shots"]} == set(
        split["training"] + split["held_out"]
    )
    assert all(row["temperature_status"] == "unknown" for row in receipt["shots"])
    assert all(
        row["implied_resistivity_micro_ohm_m"] is None for row in receipt["shots"]
    )
    assert all(row["source_sha256"] for row in receipt["shots"])
    assert receipt["counts"]["unknown"] == 168
    sliding_joint_shot = next(row for row in receipt["shots"] if row["shot"] == 25722)
    assert "not a PF can reading" in sliding_joint_shot["operations_context"]
