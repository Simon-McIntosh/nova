"""The integrated polygon evaluator obeys the retained plasma accuracy bound."""

import json
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def integrated_receipt():
    path = os.environ.get("NOVA_POLYGON_INTEGRATED_RECEIPT")
    if not path:
        pytest.skip("requires the integrated 64-node plasma measurement")
    receipt = json.loads(Path(path).read_text())
    assert receipt["completed"] is True and receipt["count"] == 64
    assert [row["state"] for row in receipt["states"]] == [220, 333, 336]
    return receipt


@pytest.mark.parametrize("index", range(3), ids=("220", "333", "336"))
def test_integrated_plasma_trace(integrated_receipt, index):
    row = integrated_receipt["states"][index]
    assert row["reference_trace"]["relative"] < 1e-8
    assert row["trace"]["relative"] <= 1e-8


@pytest.mark.parametrize("component", ("psi_RR", "psi_RZ", "psi_ZR", "psi_ZZ"))
def test_interior_hessian_reference(integrated_receipt, component):
    for row in integrated_receipt["states"]:
        assert row["components"][component]["reference_bound"] < 1e-8
        assert row["components"][component]["relative"] <= 1e-8
