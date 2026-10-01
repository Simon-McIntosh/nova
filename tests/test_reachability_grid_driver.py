"""Contract tests for the reachability-grid driver's MAST state request.

The driver forwards the persisted response carrier identity to the MAST state
request, so a solve cannot be requested under a carrier the bank no longer
honours.  These tests pin that forwarding at the call site without solving.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "docs/figures/primary-xpoint-evidence/reachability_grid_final.py"
)

CASE_CARRIER_IDENTITY = "synthetic-carrier-identity"


def _driver():
    spec = spec_from_file_location("reachability_grid_final_driver", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _passive_case() -> dict[str, Any]:
    return {
        "reference": {"plasma_current_a": -753.0},
        "state": np.zeros(3),
        "carrier": {"semantic_response_identity": CASE_CARRIER_IDENTITY},
    }


def _bind_synthetic_case(monkeypatch, driver, passive_case):
    """Point the driver's MAST reconstruction at a synthesised passive case."""

    monkeypatch.setattr(driver, "_achieved_classes", lambda: {})
    monkeypatch.setattr(driver, "_efit_rows", lambda: {})
    monkeypatch.setattr(
        driver,
        "select_slices_by_shot",
        lambda bank: [({"shot": 21_978, "slice_index": 35}, "qualification")],
    )
    carrier_evidence = {
        "carrier": {"semantic_response_identity": CASE_CARRIER_IDENTITY},
        "loaded_from_persisted_carrier": True,
    }
    monkeypatch.setattr(
        driver,
        "_persisted_response_cache",
        lambda carrier, receipt: ({}, carrier_evidence),
    )
    monkeypatch.setattr(
        driver,
        "_mast_case_from_selection",
        lambda store, row, qualification: ("case", "context"),
    )
    monkeypatch.setattr(
        driver,
        "_passive_inclusive_case",
        lambda case, context, response_cache: (
            passive_case,
            SimpleNamespace(),
            {"section_kernel_evaluations_this_shot": 0},
        ),
    )


def test_mast_state_request_carries_the_persisted_carrier_identity(monkeypatch):
    driver = _driver()
    passive_case = _passive_case()
    _bind_synthetic_case(monkeypatch, driver, passive_case)

    requests: list[dict[str, Any]] = []

    def fake_mast_states(*args, **kwargs):
        requests.append({"args": args, "kwargs": kwargs})
        return {}

    reachability = SimpleNamespace(_mast_states=fake_mast_states)

    panels = driver._mast_panels(object(), reachability)

    assert panels == []
    assert len(requests) == 1, "the MAST state request path did not execute"
    request = requests[0]
    assert "carrier_identity" in request["kwargs"], (
        "the MAST state request omitted carrier_identity"
    )
    assert (
        request["kwargs"]["carrier_identity"]
        == passive_case["carrier"]["semantic_response_identity"]
    )
    assert request["args"][2] == pytest.approx(753.0)


def test_mast_state_request_forwards_the_identity_the_carrier_reports(monkeypatch):
    """The forwarded identity is the carrier's, not a constant baked into the driver."""

    driver = _driver()
    passive_case = _passive_case()
    _bind_synthetic_case(monkeypatch, driver, passive_case)
    rotated = "rotated-carrier-identity"
    passive_case["carrier"]["semantic_response_identity"] = rotated
    monkeypatch.setattr(
        driver,
        "_persisted_response_cache",
        lambda carrier, receipt: (
            {},
            {"carrier": {"semantic_response_identity": rotated}},
        ),
    )

    captured: dict[str, Any] = {}

    def fake_mast_states(*args, **kwargs):
        captured.update(kwargs)
        return {}

    driver._mast_panels(object(), SimpleNamespace(_mast_states=fake_mast_states))

    assert captured["carrier_identity"] == rotated
