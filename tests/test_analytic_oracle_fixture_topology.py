"""The analytic fixture topology record carries the clip's saddle vertex.

The forward support clip reads ``topology.x_point`` to place the separatrix
saddle in its traced boundary.  The fixture poses that topology from the
analytic reference's own identity, so a diverted reference contributes its
declared X-point while a limited one contributes the same non-finite vertex
pair a production read carries where it admits no saddle.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import solovev_certificate as certificate
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures import measure as fixture


@pytest.fixture(scope="module", autouse=True)
def _binary64_cpu() -> None:
    configure_dtypes()
    assert jax.config.jax_enable_x64
    assert jax.default_backend() == "cpu"


def _fixture_topology(case_name: str):
    """Return one certified case's analytic reference and fixture topology."""
    _carrier, _source, analytic = certificate._case(case_name)
    topology = fixture._analytic_topology(
        analytic,
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jnp.asarray(-1.0),
    )
    return analytic, topology


def test_fixture_topology_carries_the_diverted_saddle():
    """A diverted analytic reference contributes its own X-point vertex."""
    analytic, topology = _fixture_topology("diverted-jump-bearing")

    assert hasattr(topology, "x_point")
    saddle = np.asarray(topology.x_point, dtype=np.float64)
    assert saddle.shape == (2,)
    assert np.all(np.isfinite(saddle))
    np.testing.assert_allclose(saddle, np.asarray(analytic.x_point, dtype=np.float64))

    # The carried vertex is the analytic saddle rather than a copied tag: the
    # flux gradient vanishes there and is far from vanishing away from it.
    saddle_gradient = float(
        np.linalg.norm(np.asarray(analytic.gradient(saddle[None, :]))[0])
    )
    interior = 0.5 * (saddle + np.asarray(analytic.magnetic_axis, dtype=np.float64))
    interior_gradient = float(
        np.linalg.norm(np.asarray(analytic.gradient(interior[None, :]))[0])
    )
    assert saddle_gradient < 1.0e-6 * interior_gradient


def test_fixture_topology_carries_the_absent_saddle_when_limited():
    """A limited analytic reference contributes the absent-saddle vertex pair."""
    analytic, topology = _fixture_topology("moderate-rotation-conventional")

    assert hasattr(topology, "x_point")
    saddle = np.asarray(topology.x_point, dtype=np.float64)
    assert saddle.shape == (2,)
    assert not np.any(np.isfinite(saddle))
    assert not hasattr(analytic, "x_point")
