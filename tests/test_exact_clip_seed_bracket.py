"""The exact cold seed locates finite current-amplitude bracket endpoints."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import exact_clip_seed_amplitude as benchmark
from benchmarks import solovev_certificate as certificate
from nova.equilibrium.forward_operator import (
    set_support_clip_mode,
    support_clip_mode,
)
from nova.jax.config import configure_dtypes


@pytest.fixture(autouse=True)
def _restore_clip_mode():
    previous = support_clip_mode()
    yield
    set_support_clip_mode(previous)


@pytest.mark.slow
def test_exact_seed_finds_a_finite_bracket_on_the_342_cell_carrier() -> None:
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    set_support_clip_mode("exact")
    case_name = "weak-rotation-reactor-static"
    requested_cells = -300
    (
        machine,
        _exact,
        _analytic,
        operator,
        profile,
        target_current,
        centroid,
        current_receipt,
    ) = benchmark._problem(case_name, requested_cells)

    seed, requested_class, _seed_receipt = certificate._production_seed(
        profile,
        case_name,
        target_current,
        centroid,
        current_receipt,
    )
    moments = operator.cell_current_moments(
        jnp.asarray(seed), requested_class=requested_class
    )
    booked_current = float(jnp.sum(moments.cell_current))
    amplitude = float(target_current / booked_current)

    assert len(machine.node) == 342
    assert np.isfinite(amplitude)
    assert abs(amplitude - 1.0) <= 1.0e-2
