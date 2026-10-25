"""Regression checks for a finite correction and continuous saddle booking."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import json
import math
import sys

import jax
import pytest

from scripts.prototypes.fast_map_production import solver, tie_probe
from tests.equilibrium.test_topology_read import _census_cache, _census_tie_setup


@pytest.mark.slow
def test_first_diverted_correction_is_finite(tmp_path, monkeypatch):
    assert jax.config.jax_enable_x64 is True
    output = tmp_path / "correction.json"
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "solver",
            "--kind",
            "diverted",
            "--cells",
            "550",
            "--method",
            "gmres",
            "--max-steps",
            "1",
            "--line-search",
            "--no-panel",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="cold solve residual"):
        solver.main()
    receipt = json.loads(output.read_text())
    correction = receipt["history"][0]
    assert math.isfinite(receipt["centre_residual"])
    assert correction["accepted"]
    assert math.isfinite(correction["linear_residual"])
    assert math.isfinite(correction["residual"])
    assert correction["residual"] < receipt["centre_residual"]


@pytest.mark.slow
def test_adjacent_saddle_booking_keeps_one_representation():
    assert jax.config.jax_enable_x64 is True
    state, prior, _ = _census_tie_setup(_census_cache())
    sampled = tie_probe.make_sampler(state)
    readings = [
        sampled(offset * state["scale"], prior)
        for offset in (0.16436667480678155, 0.16436667480678158)
    ]
    import numpy as np

    # The pair has to straddle a switch, or the booking claim below is asserted
    # on a state where the switch never happened and passes for that reason.
    states = [np.asarray(reading.booking_cells) for _, reading, *_ in readings]
    differing = [int(index) for index in np.nonzero(states[0] != states[1])[0]]
    assert differing, "the pair does not straddle a booking switch"

    for _, reading, current, _, _, _ in readings:
        assert bool(reading.valid and reading.qualified)
        assert bool(reading.booking_cells[266])
        assert math.isfinite(float(current))
    assert abs(float(readings[1][2] - readings[0][2])) < 1e-8
