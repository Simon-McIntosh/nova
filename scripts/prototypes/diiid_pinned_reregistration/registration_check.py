"""Hold a recorded seed preflight to the finite-edge registration.

The record is whatever the benchmark entry point last wrote, so this reads a
measurement rather than recomputing one. Point NOVA_PINNED_SEED_RECORD at
another record to check it instead; the default is the finite-edge head record
under the run's cache root.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import benchmarks.diiid_current_pinned_forward as pinned

TOLERANCE = pinned.COHORT_PREFLIGHT_RELATIVE_TOLERANCE
DECLARED = pinned.REPRESENTATIVE_COHORT[0]
DEFAULT_RECORD = (
    "/home/ITER/mcintos/.cache/nova-diiid-pinned-reregistration/seed-route/head/"
    "seed_preflight.json"
)


def _record():
    path = Path(os.environ.get("NOVA_PINNED_SEED_RECORD", DEFAULT_RECORD))
    return json.loads(path.read_text())


def test_seed_current_reproduces_the_finite_edge_registration():
    record = _record()
    assert record["frame"] == DECLARED["frame"]
    measured = record["unscaled_seed_plasma_current_a"]
    declared = DECLARED["unscaled_source_ip_a"]
    relative = abs(measured - declared) / abs(declared)
    assert relative <= TOLERANCE
    amplitude = record["seed_profile_amplitude"]
    assert amplitude == pytest.approx(DECLARED["seed_lambda"], rel=TOLERANCE)


def test_the_record_carries_the_prescribed_boundary_values():
    edge = _record()["prescribed_profile_edge"]
    assert edge["psi_norm"] == 1.0
    assert edge["p_prime"] < 0.0
    assert edge["ff_prime"] != 0.0
