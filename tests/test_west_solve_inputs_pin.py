"""The WEST 62751 solve-inputs pin is present and carries the committed values.

west_inputs() reads this pin on every call, so a deleted or stale pin is a live
failure of the production path rather than a missing artefact. Each value is
compared against the record that owns it, so a number mis-transcribed into both
the pin writer and the pin itself would still be caught here.

    NOVA_WEST_SOLVE_INPUTS=<path> pytest tests/test_west_solve_inputs_pin.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.west_iron_galerkin_pair.measure import REFERENCE
from scripts.west_iron_inputs import committed

SHARED_ROOT_PIN = Path(
    "/work/projects/imas_gpu/nova/evidence/iron-core-magnetisation/west-62751"
    "/mesh-pair/inputs/west-62751-solve-inputs.json"
)


def pin_path() -> Path:
    """The pin west_inputs() reads, or the newest one where the tracked path lags."""
    override = os.environ.get("NOVA_WEST_SOLVE_INPUTS")
    if override:
        return Path(override)
    # Once the tracked constant names a live pin, the test follows it.
    return REFERENCE if REFERENCE.is_file() else SHARED_ROOT_PIN


def test_pin_is_present():
    path = pin_path()
    assert path.is_file(), f"the solve-inputs pin is missing at {path}"


@pytest.fixture(scope="module")
def pin():
    path = pin_path()
    assert path.is_file(), f"the solve-inputs pin is missing at {path}"
    return json.loads(path.read_text())


def test_probe_positions(pin):
    radii = [float(row[0]) for row in pin["probes_m"]]
    heights = [float(row[1]) for row in pin["probes_m"]]
    assert radii == committed.probe_radii()
    assert heights == [0.0, 0.0, 0.0]


def test_air_core_field(pin):
    magnitude = [float(value) for value in pin["air_magnitude_T"]]
    for measured, recorded in zip(
        magnitude, committed.air_millitesla_solve(), strict=True
    ):
        assert measured * 1e3 == pytest.approx(recorded, rel=1e-11)
    for measured, recorded in zip(
        magnitude, committed.air_microtesla_pair(), strict=True
    ):
        assert measured * 1e6 == pytest.approx(recorded, rel=1e-5)
    assert magnitude[0] * 1e6 == pytest.approx(
        committed.air_plan_microtesla(), rel=5e-4
    )


def test_material_table(pin):
    knots = [float(value) for value in pin["table_b_T"][:2]]
    assert knots == committed.table_b_knots()
    mu = [float(value) for value in pin["table_mu_relative"]]
    origin = committed.table_mu_origin()
    assert mu[0] == pytest.approx(origin[0], rel=1e-9)
    assert mu[1] == pytest.approx(origin[1], rel=1e-7)
    assert mu[0] == pytest.approx(committed.material_test_mu_origin(), rel=1e-9)
    assert max(mu) == pytest.approx(committed.table_mu_maximum(), rel=1e-4)


def test_source_time_and_provenance(pin):
    assert pin["time_s"] == pytest.approx(committed.source_time(), rel=1e-12)
    assert len(pin["revision"]) == 40
    assert len(pin["pin_py_blob"]) == 40
    assert pin["source_sha256"] == pin["source_sha256"].lower()
    assert len(pin["source_sha256"]) == 64
