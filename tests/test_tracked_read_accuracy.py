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

import inspect
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import fixed_points, support_read
from scripts.prototypes.tracked_read_batch.reference import analytic_boundary
from scripts.prototypes.tracked_read_batch.reference import psi as analytic_psi

assert jax.config.jax_enable_x64 is True

# The certificate's 550-cell bank is compiled inside the module fixture, which
# costs several minutes on the device and is charged to the first test that
# asks for it. The suite-wide 300 s bound is shorter than that compile, so it
# would report a timeout rather than a result.
pytestmark = pytest.mark.timeout(1800)

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


def _record(record):
    """Append one measurement record to the artifact named by the environment.

    A fixture's stdout reaches a log only inside a captured-output section,
    which pytest prints for a failing test, so a row for a state that passes is
    absent from the log of that run. A named path carries every row whatever
    the test outcomes are, which is what lets a quoted number be traced to the
    run that produced it.
    """
    path = os.environ.get("NOVA_ACCURACY_ROWS")
    if not path:
        return
    with Path(path).open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def _wall_flux_range(case, per_segment=64):
    """The analytic flux range over the wall polygon the case carries.

    The wall is sampled along every segment, so the range is a property of the
    geometry the read is built on rather than of the reference being checked.
    """
    wall = np.asarray(case["geometry"].wall_points, dtype=float).reshape(-1, 2)
    start = wall
    end = np.roll(wall, -1, axis=0)
    fraction = np.linspace(0.0, 1.0, per_segment, endpoint=False)[None, :, None]
    sample = (start[:, None, :] + fraction * (end - start)[:, None, :]).reshape(-1, 2)
    values = np.asarray(
        jax.vmap(lambda point: case["analytic"].evaluate(point).value)(
            jnp.asarray(sample)
        )
    )
    return float(values.max()), float(values.min())


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
        step = np.linalg.solve(np.asarray(jet.hessian), np.asarray(jet.gradient))
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
    boundary_point, boundary_flux_reference = analytic_boundary(case, references)
    axis_flux = analytic_psi(case["analytic"], references[0])
    span = abs(boundary_flux_reference - axis_flux)
    wall_flux_max, wall_flux_min = _wall_flux_range(case)

    def run(moments):
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
        return dict(
            position=np.asarray(reading.stationary.position),
            valid=bool(reading.valid),
            qualified=bool(reading.reading.qualified),
            status=np.asarray(reading.stationary.status).tolist(),
            boundary_flux=float(reading.reading.boundary_flux),
        )

    states = {name: run(moments) for name, moments in case["states"].items()}
    for name, row in states.items():
        row["moments"] = case["states"][name]
    # The same read on the moments booked from the analytic field, so a failing
    # booked-field row can be shown beside the row the analytic field gives.
    substituted = run(case["states"]["converged"])
    budget = AXIS_BUDGET * minor_radius
    for name, row in states.items():
        row["boundary_reference"] = boundary_flux_reference
        source = field_reference(operands, case["states"][name], case["roots"][0])
        record = dict(
            kind=kind,
            state=name,
            axis_error_m=float(np.linalg.norm(row["position"][0] - references[0])),
            saddle_error_m=(
                float(np.linalg.norm(row["position"][1] - references[1]))
                if len(references) > 1
                else None
            ),
            axis_budget_m=budget,
            boundary_error=abs(row["boundary_flux"] - boundary_flux_reference) / span,
            analytic_field_axis_error_m=float(
                np.linalg.norm(substituted["position"][0] - references[0])
            ),
            analytic_field_boundary_error=abs(
                substituted["boundary_flux"] - boundary_flux_reference
            )
            / span,
            interpolant_axis_error_m=float(np.linalg.norm(source - references[0])),
            iterations=row["status"],
        )
        print("ACCURACY_ROW", json.dumps(record), flush=True)
        _record(record)
    boundary_record = dict(
        kind=kind,
        point=np.asarray(boundary_point).tolist(),
        flux=boundary_flux_reference,
        axis_flux=axis_flux,
        span=span,
        minor_radius=minor_radius,
        wall_flux_max=wall_flux_max,
        wall_flux_min=wall_flux_min,
    )
    print("BOUNDARY_REFERENCE", json.dumps(boundary_record), flush=True)
    _record(boundary_record)
    return dict(
        kind=kind,
        mesh=operands.mesh,
        operands=operands,
        references=references,
        minor_radius=minor_radius,
        span=span,
        states=states,
        boundary_flux_reference=boundary_flux_reference,
        wall_flux_max=wall_flux_max,
        wall_flux_min=wall_flux_min,
    )


@pytest.mark.slow
def test_the_cell_selection_has_one_owner():
    """A second nearest-generator rule would drift from the field's own."""
    assert not hasattr(support_read, "stationary_cells")
    assert "nearest_cell" in inspect.getsource(support_read.read)
    assert "nearest_cell" in inspect.getsource(fixed_points._CachedField._location)


@pytest.mark.slow
def test_the_read_and_the_field_choose_the_same_cell(measured):
    """A point equidistant from two generators is where two owners diverge."""
    mesh = measured["mesh"]
    generators = np.asarray(mesh.generators, dtype=float)
    field = fixed_points._CachedField(
        mesh,
        jnp.arange(len(generators), dtype=jnp.int32),
        measured["operands"].rows,
        jnp.stack(tuple(measured["states"]["converged"]["moments"]), axis=-1),
        measured["operands"].exterior,
    )
    distance = np.linalg.norm(generators[:, None, :] - generators[None, :, :], axis=-1)
    first, second = np.unravel_index(
        np.argmin(np.where(distance > 0.0, distance, np.inf)), distance.shape
    )
    edge = jnp.asarray(0.5 * (generators[first] + generators[second]))
    chosen = int(np.asarray(fixed_points.nearest_cell(mesh, edge)))
    located = int(np.asarray(field._location(edge)[0]))
    assert chosen == located, (chosen, located, first, second)


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
def test_the_boundary_reference_is_the_wall_flux_extremum(measured):
    """The limited reference must sit at the wall's largest analytic flux.

    A rule that minimised the wall flux instead, or read a different wall,
    would land outside this window, and the boundary comparison it feeds would
    then be measuring an instrument rather than the read.
    """
    if len(measured["references"]) > 1:
        # A diverted boundary is the certificate's saddle, not a wall point.
        return
    budget = BOUNDARY_BUDGET * measured["span"]
    flux = measured["boundary_flux_reference"]
    maximum = measured["wall_flux_max"]
    minimum = measured["wall_flux_min"]
    assert maximum - minimum > 0.0, (measured["kind"], maximum, minimum)
    assert flux >= maximum - budget, (measured["kind"], flux, maximum, budget)
    assert flux - minimum > 0.5 * (maximum - minimum), (
        measured["kind"],
        flux,
        minimum,
        maximum,
    )


@pytest.mark.slow
@pytest.mark.parametrize("state", ["converged", "trial"])
def test_boundary_flux_against_the_analytic_read(measured, state):
    row = measured["states"][state]
    span = measured["span"]
    assert span > 0.0
    error = abs(row["boundary_flux"] - row["boundary_reference"]) / span
    assert error <= BOUNDARY_BUDGET, (measured["kind"], state, error)
