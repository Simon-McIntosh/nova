"""A sampled flux function carries a supplied boundary value at psi_norm = 1.

EFIT's p' and FF' stay finite on the last closed flux surface, so the profile
representation has to return the boundary value a profile supplies rather than
cap it to zero. The zero-edge form stays available for a source that vanishes
at the separatrix, and the interior samples must not change between the forms.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import ast
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import diiid_forward_gs_match as bench
from nova.equilibrium.solve_request import SampledFluxFunction

assert jax.config.jax_enable_x64 is True

INTERIOR = np.linspace(0.05, 0.95, 19)
INTERIOR_VALUES = -3000.0 - 500.0 * np.cos(np.pi * INTERIOR)
EDGE_VALUE = -2319.18090032498
EXTRACTION = Path(
    "docs/figures/diiid-vertical-force-balance/diiid-finite-edge-flux/extraction.json"
)


def test_finite_edge_returns_the_supplied_value_at_the_boundary():
    finite = SampledFluxFunction(INTERIOR, INTERIOR_VALUES, edge_value=EDGE_VALUE)
    one = jnp.asarray(1.0)
    assert float(finite(one)) == EDGE_VALUE
    assert float(jax.jit(lambda profile, point: profile(point))(finite, one)) == (
        EDGE_VALUE
    )
    beyond = jnp.asarray([1.0, 1.02, 1.4])
    np.testing.assert_array_equal(finite(beyond), np.full(3, EDGE_VALUE))


def test_zero_edge_form_still_returns_zero_at_the_boundary():
    zero = SampledFluxFunction(INTERIOR, INTERIOR_VALUES)
    assert float(zero(jnp.asarray(1.0))) == 0.0
    assert float(zero(jnp.asarray(1.4))) == 0.0


def test_finite_edge_keeps_the_interior_samples():
    zero = SampledFluxFunction(INTERIOR, INTERIOR_VALUES)
    finite = SampledFluxFunction(INTERIOR, INTERIOR_VALUES, edge_value=EDGE_VALUE)
    knots = jnp.asarray(INTERIOR)
    np.testing.assert_array_equal(np.asarray(finite(knots)), INTERIOR_VALUES)
    inside = jnp.linspace(0.0, float(INTERIOR[-1]), 257, dtype=jnp.float64)
    np.testing.assert_array_equal(np.asarray(finite(inside)), np.asarray(zero(inside)))


def test_finite_edge_is_continuous_across_the_last_knot():
    finite = SampledFluxFunction(INTERIOR, INTERIOR_VALUES, edge_value=EDGE_VALUE)
    last = float(INTERIOR[-1])
    step = 1e-7
    at_knot = float(finite(jnp.asarray(last)))
    np.testing.assert_allclose(at_knot, float(INTERIOR_VALUES[-1]), rtol=0, atol=1e-12)
    left = (at_knot - float(finite(jnp.asarray(last - step)))) / step
    right = (float(finite(jnp.asarray(last + step))) - at_knot) / step
    np.testing.assert_allclose(left, right, rtol=1e-5)
    assert left > 0.0


def test_finite_edge_carries_one_value_per_batched_member():
    first = SampledFluxFunction(INTERIOR, INTERIOR_VALUES, edge_value=EDGE_VALUE)
    second = SampledFluxFunction(
        INTERIOR, INTERIOR_VALUES, edge_value=0.0058420389744635714
    )
    batched = jax.tree_util.tree_map(
        lambda one, other: jnp.stack((one, other)), first, second
    )
    boundary = jnp.asarray([1.0, 1.0])
    observed = jax.jit(jax.vmap(lambda profile, point: profile(point)))(
        batched, boundary
    )
    np.testing.assert_array_equal(
        np.asarray(observed), np.asarray([EDGE_VALUE, 0.0058420389744635714])
    )


def test_finite_edge_refuses_a_table_that_reaches_the_boundary():
    with pytest.raises(ValueError, match="inside psi_norm=1"):
        SampledFluxFunction(
            np.linspace(0.1, 1.0, 10), np.linspace(5.0, 1.0, 10), edge_value=1.0
        )


def test_builder_reads_the_boundary_shell_the_record_commits():
    """The builder's psi_norm = 1 value is the committed extraction's value."""

    if not EXTRACTION.is_file():
        pytest.skip("the boundary extraction receipt is not present")
    records = json.loads(EXTRACTION.read_text())["rows"]
    shot = bench.DEFAULT_DATA / "d3d_shot_00000c4a7b.parquet"
    if not shot.is_file():
        pytest.skip("the DIII-D corpus is not readable from this host")
    columns = (
        bench._LABEL_COLUMNS
        + bench._GEOMETRY_COLUMNS
        + bench._CURRENT_COLUMNS
        + bench._PLASMA_CURRENT_COLUMNS
    )
    row = bench._read(shot, columns)
    row["_source_path"] = str(shot)
    record = records["P102"]
    psi_norm, p_prime, ff_prime = bench._profile_edge_values(row, record["frame"])
    assert psi_norm == 1.0
    np.testing.assert_allclose(p_prime, record["p_prime"], rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(ff_prime, record["ff_prime"], rtol=1e-10, atol=0.0)
    _, surfaces, values, _ = bench._label_state(row, record["frame"])
    profile = SampledFluxFunction(surfaces, values, edge_value=p_prime)
    assert float(profile(jnp.asarray(1.0))) == p_prime


DRIVER_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "scripts/prototypes/diiid_finite_edge_flux"
)


@pytest.mark.parametrize(
    "driver", sorted(path.name for path in DRIVER_DIRECTORY.glob("*.py"))
)
def test_the_drivers_parse_at_the_oldest_supported_grammar(driver):
    """A driver the record cites must parse where the project's readers run.

    CPython 3.14 accepts ``except TypeError, ValueError``, so importing the
    module proves nothing about older interpreters and a regression there
    would pass every arm silently. Parsing at the 3.13 grammar keeps the
    driver readable wherever the receipt is regenerated.
    """

    source = (DRIVER_DIRECTORY / driver).read_text()
    ast.parse(source, filename=str(DRIVER_DIRECTORY / driver), feature_version=(3, 13))
