"""Batched booking preserves each matrix and bounds the device scaling."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import json
import os
from pathlib import Path

import jax
import numpy as np
import pytest

from scripts.prototypes.bank_step_booking_batch.measure import (
    cycle,
    inputs,
    jacobians,
)


@pytest.fixture(scope="module")
def machine():
    assert jax.config.jax_enable_x64
    print(
        "MODULE", cycle.support_read.__file__, "CWD", Path.cwd().resolve(), flush=True
    )
    return cycle.reference.machine("diverted")


@pytest.mark.slow
@pytest.mark.parametrize("lanes", [1, 16])
def test_each_matrix_matches_archived_base(machine, lanes):
    path = os.environ.get("BOOKING_BASE_MATRICES")
    if not path:
        pytest.skip("paired CPU gate supplies the archived matrix receipt")
    bank, _, _, _, _, profile, quadrature, *_ = machine
    states, support, booked = inputs(lanes, machine)
    result = jax.jit(lambda b, m, s, p: jacobians(b, m, s, p, profile, quadrature))(
        bank, states, support, booked
    )
    actual = np.asarray(result)
    with np.load(path) as receipt:
        expected = receipt[f"matrix_{lanes}"]
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    scale = np.linalg.norm(expected.reshape(lanes, -1), axis=1)
    assert np.all(scale > 0)
    error = np.linalg.norm((actual - expected).reshape(lanes, -1), axis=1) / scale
    print("MATRIX_RELATIVE_ERRORS", lanes, error.tolist(), flush=True)
    assert np.all(error <= 1e-12), error


@pytest.mark.slow
def test_geometry_stencil_overflow_is_refused(machine, monkeypatch):
    bank, _, _, _, _, profile, quadrature, *_ = machine
    states, support, booked = inputs(1, machine)
    # The seed uses one triangle; two slots cannot represent its three nodes.
    monkeypatch.setattr(cycle.support_read, "_FRESH_GEOMETRY_NODE_CAPACITY", 2)
    result = jax.jit(lambda b, m, s, p: jacobians(b, m, s, p, profile, quadrature))(
        bank, states, support, booked
    )
    assert np.isnan(np.asarray(result)).all()


def test_sixteen_lane_scaling():
    path = os.environ.get("BOOKING_TIMINGS")
    if not path:
        pytest.skip("paired H200 gate supplies synchronized timing measurements")
    timings = json.loads(Path(path).read_text())
    one = timings["1:assembly"]
    sixteen = timings["16:assembly"]
    assert np.isfinite(one) and one > 0
    assert np.isfinite(sixteen) and sixteen > 0
    print("SCALING", sixteen / one, "ONE_MS", one, "SIXTEEN_MS", sixteen, flush=True)
    assert sixteen <= 4 * one, (one, sixteen, sixteen / one)
