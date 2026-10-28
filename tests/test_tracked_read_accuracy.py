"""The tracked read at the certificate states, against the analytic Newton.

Every reference here is a direct Newton solve on the analytic flux of the
certificate state, seeded from the certificate's own roots and sharing no code
with any read, so no instrument this module exercises decides its own clause.
The trial booking is the one the fast route consumes: a read of the sampled
field, booked on the analytic flux at the sample points.

The read's derivative source is a Bell interpolant of the nodal jets over the
triangle a point occupies. ``field_reference`` solves that same interpolant, so
the distance between the read and it measures the solve, and the distance
between it and the analytic extremum measures the interpolant and the booking.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import fixed_points, support_read

assert jax.config.jax_enable_x64 is True

CELLS = 550
POLICY_RADIUS = 0.054806712567833996
AXIS_BUDGET = 5e-4
BOUNDARY_BUDGET = 1e-4


def _built(kind):
    from scripts.prototypes.tracked_read.measure import build_kind, cache_for

    case = build_kind(kind, CELLS, POLICY_RADIUS)
    operands = support_read.SupportOperands.from_rows(
        case["geometry"],
        case["matrix"],
        case["exterior"],
        case["nodes"],
        case["sample_index"],
        bucket=len(case["nodes"]),
    )
    return case, operands, cache_for(operands)


def field_reference(operands, moments, seed, iterations=60):
    """Newton on the read's own derivative source, from a stated seed."""
    field = fixed_points._CachedField(
        operands.mesh,
        jnp.arange(len(np.asarray(operands.nodes)), dtype=jnp.int32),
        operands.rows,
        jnp.stack(tuple(moments), axis=-1),
        operands.exterior,
    )
    position = np.asarray(seed, dtype=float)
    for _ in range(iterations):
        jet = field.evaluate(jnp.asarray(position))
        step = np.linalg.solve(
            np.asarray(jet.hessian), np.asarray(jet.gradient)
        )
        position = position - step
        if np.linalg.norm(step) < 1e-14:
            break
    return position


@pytest.fixture(scope="module", params=["limited", "diverted"])
def measured(request):
    from scripts.prototypes.tracked_read.measure import newton_reference

    kind = request.param
    case, operands, cache = _built(kind)
    references = [newton_reference(case["analytic"], seed)[0] for seed in case["roots"]]
    minor_radius = float(
        np.max(
            np.linalg.norm(
                np.asarray(case["geometry"].sample_points).reshape(-1, 2)
                - references[0],
                axis=-1,
            )
        )
    )
    span = abs(float(case["analytic_read"].boundary_flux - case["analytic_read"].axis_flux))
    states = {}
    for name, moments in case["states"].items():
        reading = support_read.read(
            operands,
            cache,
            moments,
            jnp.asarray(case["roots"]),
            jnp.asarray(case["roots"]),
            case["convention"],
            case["policy"],
            iterations=24,
        )
        jax.block_until_ready(reading)
        states[name] = dict(
            position=np.asarray(reading.stationary.position),
            valid=bool(reading.valid),
            qualified=bool(reading.reading.qualified),
            status=np.asarray(reading.stationary.status).tolist(),
            boundary_flux=float(reading.reading.boundary_flux),
            boundary_reference=float(case["analytic_read"].boundary_flux),
        )
    return dict(
        kind=kind,
        references=references,
        minor_radius=minor_radius,
        span=span,
        states=states,
    )


@pytest.mark.slow
@pytest.mark.parametrize("state", ["converged", "trial"])
def test_stationary_points_against_the_analytic_newton(measured, state):
    row = measured["states"][state]
    budget = AXIS_BUDGET * measured["minor_radius"]
    assert row["valid"], (measured["kind"], state, row["status"])
    assert row["qualified"], (measured["kind"], state, row["status"])
    axis_error = float(np.linalg.norm(row["position"][0] - measured["references"][0]))
    assert axis_error <= budget, (measured["kind"], state, axis_error, budget)
    if len(measured["references"]) > 1:
        saddle_error = float(
            np.linalg.norm(row["position"][1] - measured["references"][1])
        )
        assert saddle_error <= budget, (measured["kind"], state, saddle_error, budget)


@pytest.mark.slow
@pytest.mark.parametrize("state", ["converged", "trial"])
def test_boundary_flux_against_the_analytic_read(measured, state):
    row = measured["states"][state]
    span = measured["span"]
    assert span > 0.0
    error = abs(row["boundary_flux"] - row["boundary_reference"]) / span
    assert error <= BOUNDARY_BUDGET, (measured["kind"], state, error)