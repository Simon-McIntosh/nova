"""Exact-mode map fidelity at the fidelity rungs, and the map's JVP.

The map is ``gs.apply`` on the production ``ForwardFluxOperator``: the booked
plasma current imaged through the coupling blocks and added to the exterior
flux.  Fidelity is the relative sup of the difference between that image and the
analytic state over the span between the analytic axis and the analytic zero
level, evaluated once with no solve.  The exterior is posed on the exact
analytic separatrix support, so the closure residual cannot mask the booking
term and a failure from the fixture cannot be read as a map failure.

The map, its booking term and the exact-support exterior come from the
certificate-support gate's own ``build_carrier`` and ``measure_arm``, so the two
gates share one implementation of the recipe instead of each carrying a copy.

The rung requests realise about 2,000, 5,000 and 10,000 solver cells.  The
2,000-cell rung is measured in this test from the analytic state, so a code
regression fails the gate.  The 5,000- and 10,000-cell rungs each cost a carrier
build far beyond a gate's budget, so they are read from the evidence fragment
the driver at the foot of this file publishes; a missing or corrupted row fails
rather than passing on a stale constant.

The map's derivative is checked against central differences through the same
operator the map uses.  The boundary-flux component is the one a padded dummy
saddle once doubled: the wall level was maximised against a null level, so on
the kink autodiff returned one branch's slope while a central difference
returned the average of both.  A select whose tie resolves to the wall branch
reproduces it exactly, and ``NOVA_DUMMY_SADDLE_CLAMP=1`` restores that select
in the boundary-flux test as this file's negative control.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from benchmarks import solovev_certificate as certificate  # noqa: E402
from nova.equilibrium import topology  # noqa: E402
from scripts.analytic_oracle_fixtures import measure as fixture  # noqa: E402
from tests.test_certificate_exterior_support import (  # noqa: E402
    build_carrier,
    measure_arm,
)


LIMITED = "weak-rotation-reactor-static"
DIVERTED = "diverted-single-null"
CASES = (LIMITED, DIVERTED)
# Negative requests target solver cells after boundary qualification; the
# realised mesh is a little larger (2,098 and 5,158 at the first two rungs).
RUNG_REQUESTS = {2000: -2000, 5000: -5000, 10000: -10000}
# The only rung a gate rebuilds; the heavier rungs are published rows.
LIVE_CELLS = 2000
DERIVATIVE_REQUEST = -550
MAP_TOLERANCE = 1.0e-2
DECOMPOSITION_TOLERANCE = 1.0e-10
MAP_JVP_TOLERANCE = 1.0e-6
BOUNDARY_JVP_TOLERANCE = 1.0e-6
JVP_SPAN_FRACTION = 1.0e-6
CLAMP_ENV = "NOVA_DUMMY_SADDLE_CLAMP"
IMAGE_NULL_COLOR = "#7a3ea8"
ROOT = Path(__file__).resolve().parents[1]
FRAGMENT = (
    ROOT
    / "docs/evidence/fragments/converged-forward-solve/cfs-map-fidelity-rungs.html"
)
ROWS_ID = "cfs-map-fidelity-rows"

_MEASUREMENTS: dict[tuple[str, int], dict[str, object]] = {}
_STATES: dict[tuple[str, int], tuple[object, object, float]] = {}
_CARRIERS: dict[tuple[str, int], dict[str, object]] = {}

PUBLISHED_ROW_KEYS = (
    "case",
    "requested_cells",
    "realised_cells",
    "span_wb",
    "map_relative_sup",
    "booking_image_relative_sup",
    "exterior_closure_relative_sup",
    "decomposition_relative_sup",
    "exterior_cache",
)


def _analytic_operator(case_name: str, request: int):
    """Return the production map operator, analytic state and span.

    The operator carries the production support clip. The exact clip that the
    fidelity rungs use selects cells with a branch that is not differentiable
    where a cell boundary sits on a node, so its map tangent disagrees with a
    central difference by 4.0e-3 at this state; the production clip is the map
    a solve differentiates and agrees to 2.9e-8.
    """
    key = (case_name, request)
    if key in _STATES:
        return _STATES[key]
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    carrier, source, exact = certificate._case(case_name, clip_mode="exact")
    machine = certificate._case_machine(case_name, carrier, exact, request)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    state = jnp.asarray(
        certificate._exact_state(case_name, exact, coordinates), dtype=np.float64
    )
    operator = fixture.forward_operator(source, machine)
    span = abs(float(fixture._analytic_axis_flux(exact)))
    _STATES[key] = (operator, state, span)
    return _STATES[key]


def _carrier(case_name: str, cells: int) -> dict[str, object]:
    """Build one rung's carrier through the certificate-support gate's recipe.

    The rung's machine comes from the certificate case and the requested cell
    count; everything above it — the analytic state, the exact-support
    exterior, the operator, the closed-form current target and the analytic
    support override — is built by the shared ``build_carrier``.
    """
    key = (case_name, cells)
    if key in _CARRIERS:
        return _CARRIERS[key]
    carrier_case, _source, exact = certificate._case(case_name, clip_mode="exact")
    machine = certificate._case_machine(
        case_name, carrier_case, exact, RUNG_REQUESTS[cells]
    )
    record = build_carrier(case_name, machine)
    _CARRIERS[key] = record
    return record


def _measure_map(case_name: str, cells: int) -> dict[str, object]:
    """Measure one rung's map, booking and closure terms at the analytic state."""
    key = (case_name, cells)
    if key in _MEASUREMENTS:
        return _MEASUREMENTS[key]
    carrier = _carrier(case_name, cells)
    measured, _built = measure_arm(
        case_name, cells, carrier["machine"], "exact", branch=carrier["branch"]
    )
    row = {name: measured[name] for name in PUBLISHED_ROW_KEYS}
    _MEASUREMENTS[key] = row
    return row


def _receipt_rows() -> dict[tuple[str, int], dict[str, object]]:
    """Read the published measurement rows out of the evidence fragment."""
    if not FRAGMENT.exists():
        pytest.fail(f"map-fidelity evidence fragment is missing: {FRAGMENT}")
    text = FRAGMENT.read_text()
    match = re.search(
        rf'<script type="application/json" id="{ROWS_ID}">(.*?)</script>',
        text,
        re.DOTALL,
    )
    if match is not None:
        rows = json.loads(match.group(1))
        return {(row["case"], int(row["requested_cells"])): row for row in rows}
    pytest.fail(f"no {ROWS_ID} block in {FRAGMENT}")


@pytest.mark.parametrize("case_name", CASES, ids=("limited", "diverted"))
@pytest.mark.parametrize("cells", sorted(RUNG_REQUESTS))
@pytest.mark.timeout(0)
def test_map_reproduces_the_analytic_state_within_the_bound(case_name, cells):
    """The exact-mode map is within the fixed bound at every analytic rung.

    The 2,000-cell rung is measured here from the analytic state, so a code
    regression turns this test red. The 5,000- and 10,000-cell rungs each cost
    a carrier build far beyond a gate's budget, so their rows come from the
    evidence fragment the driver publishes.
    """
    if cells == LIVE_CELLS:
        row = _measure_map(case_name, cells)
    else:
        row = _receipt_rows()[(case_name, int(cells))]
    print("MAP_FIDELITY_ROW " + json.dumps(row, default=str), flush=True)
    assert row["decomposition_relative_sup"] < DECOMPOSITION_TOLERANCE, row
    assert row["exterior_closure_relative_sup"] < row["booking_image_relative_sup"], row
    assert row["map_relative_sup"] <= MAP_TOLERANCE, row


@pytest.fixture(scope="module")
def forcing_direction():
    """The map's own forcing direction at the limited analytic state.

    The forcing is the map's residual, which is the direction the map moves
    the state along towards the fixed point. Along it the map agrees with a
    central difference to 2.9e-8 and the boundary flux to 8.3e-9; along the
    Newton response step those readings are 2.6e-6 and 8.1e-6, because the
    response step points across the wall-node promotion boundary.
    """
    operator, state, span = _analytic_operator(LIMITED, DERIVATIVE_REQUEST)
    mapped = operator.flux_map(requested_class=topology.TopologyClass.LIMITED)(state)
    forcing = np.asarray(mapped) - np.asarray(state)
    direction = jnp.asarray(
        forcing / max(float(np.max(np.abs(forcing))), np.finfo(float).tiny)
    )
    return operator, state, span, direction


@pytest.mark.slow
def test_map_jvp_matches_central_difference(forcing_direction):
    """The map's tangent agrees with a central difference along its own step."""
    operator, state, span, direction = forcing_direction
    map_fn = operator.flux_map(requested_class=topology.TopologyClass.LIMITED)
    _value, tangent = jax.linearize(map_fn, state)
    tangent_value = np.asarray(tangent(direction))
    step = JVP_SPAN_FRACTION * span
    finite = np.asarray(
        (map_fn(state + step * direction) - map_fn(state - step * direction))
        / (2.0 * step)
    )
    error = float(
        np.max(np.abs(finite - tangent_value))
        / max(float(np.max(np.abs(tangent_value))), 1.0e-300)
    )
    print(f"MAP_JVP_RELATIVE_SUP={error:.17g}", flush=True)
    assert np.isfinite(error)
    assert np.max(np.abs(finite)) > 0.0
    assert error < MAP_JVP_TOLERANCE


def _read_level(operator, field):
    _masks, reading = operator.read(field, topology.TopologyClass.LIMITED)
    return reading.boundary_flux


def _boundary_level(operator, field):
    value = _read_level(operator, field)
    if os.environ.get(CLAMP_ENV) == "1":
        # The padded dummy saddle sat at zero flux, and the wall level was
        # selected against it with a tie that resolved to the wall branch.
        zero = jnp.zeros_like(value)
        value = jnp.where(value >= zero, value, zero)
    return value


@pytest.mark.slow
def test_map_boundary_flux_jvp_matches_central_difference(forcing_direction):
    """The boundary-flux component's derivative agrees with a central difference.

    The base is the analytic state shifted by a uniform flux offset so the
    selected wall level sits exactly at zero flux, where the dummy-saddle clamp
    had its kink.  A uniform offset moves every level equally, so it changes no
    topology; it only puts the derivative on the kink the clamp broke.
    """
    operator, state, span, direction = forcing_direction
    level_at = float(_read_level(operator, state))
    base = state - jnp.asarray(level_at, dtype=state.dtype)
    np.testing.assert_allclose(
        float(_boundary_level(operator, base)), 0.0, atol=1.0e-10
    )
    derivative = float(
        jax.jit(
            lambda field: jax.jvp(
                lambda value: _boundary_level(operator, value),
                (field,),
                (direction,),
            )[1]
        )(base)
    )
    compiled = jax.jit(lambda field: _boundary_level(operator, field))
    step = JVP_SPAN_FRACTION * span
    finite = float(
        (compiled(base + step * direction) - compiled(base - step * direction))
        / (2.0 * step)
    )
    error = abs(derivative - finite) / max(abs(finite), 1.0e-300)
    print(
        f"BOUNDARY_JVP={derivative:.17g} CENTRAL={finite:.17g} RELATIVE={error:.17g}",
        flush=True,
    )
    assert np.isfinite(derivative)
    assert np.isfinite(finite)
    assert abs(finite) > 0.0
    assert error < BOUNDARY_JVP_TOLERANCE


def _figure(case_name: str, cells: int = 10000) -> Path:
    """Draw the map image beside the analytic state at one carrier rung.

    Each panel carries BOTH null sets in the null vocabulary — solid triangle
    for the axis, filled cross for the admitted saddle — with the analytic set
    in the default marker colour and the map image's in a second hue, so the
    reader sees whether the map's topology matches the state it came from.
    """
    import matplotlib

    matplotlib.use("Agg")
    from dataclasses import replace

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from nova.media import ink, poloidal

    carrier = _carrier(case_name, cells)
    machine = carrier["machine"]
    analytic = carrier["analytic"]
    active = carrier["active"]
    branch = carrier["branch"]
    image = np.asarray(
        active.flux_map(requested_class=branch, target_current=carrier["target"])(
            analytic
        )
    )
    grid = len(machine.node)
    exact_grid = np.asarray(analytic)[:grid]
    _masks, exact_read = carrier["operator"].read(analytic, branch)
    _masks, image_read = carrier["operator"].read(jnp.asarray(image), branch)
    wall = np.asarray(machine.wall_node)
    levels = poloidal.contour_levels(
        exact_grid,
        14,
        boundary=float(exact_read.boundary_flux),
        axis=float(exact_read.axis_flux),
    )
    style = ink.DEFAULT_INK
    image_style = replace(
        style, axis_color=IMAGE_NULL_COLOR, xpoint_color=IMAGE_NULL_COLOR
    )
    figure, panels = plt.subplots(1, 2, figsize=(14.0, 7.2))
    figure.patch.set_facecolor(style.figure_facecolor)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93, wspace=0.02)
    for panel, values, label in (
        (panels[0], image[:grid], "map image"),
        (panels[1], exact_grid, "analytic state")
    ):
        ink.poloidal_axes(panel, style)
        poloidal.draw_scattered_contours(
            panel,
            machine.node[:, 0],
            machine.node[:, 1],
            values,
            levels,
            wall=wall,
            color=style.flux_color,
            linewidth=0.5,
        )
        poloidal.draw_wall(panel, radius=wall[:, 0], height=wall[:, 1], style=style)
        for reading, null_style in (
            (exact_read, style),
            (image_read, image_style),
        ):
            poloidal.draw_nulls(
                panel,
                magnetic_axis=np.asarray(reading.axis),
                x_points=np.asarray(reading.x_point)[None, :],
                style=null_style,
                contain=wall,
            )
        panel.text(
            0.02,
            0.98,
            label,
            transform=panel.transAxes,
            ha="left",
            va="top",
            fontsize=14)
    handles = (
        Line2D(
            [],
            [],
            marker=style.axis_marker,
            markersize=style.axis_markersize,
            color=style.axis_color,
            linestyle="none",
            label="analytic nulls",
        ),
        Line2D(
            [],
            [],
            marker=style.xpoint_marker,
            markersize=style.xpoint_markersize,
            color=IMAGE_NULL_COLOR,
            linestyle="none",
            label="map-image nulls",
        ),
    )
    figure.legend(handles=handles, loc="upper center", frameon=False, ncol=2)
    directory = ROOT / "docs/figures/converged-forward-solve/cfs-map-fidelity-rungs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{case_name}-map-fidelity-{cells}.svg"
    figure.savefig(
        path, format="svg", facecolor=style.figure_facecolor, metadata={"Date": None}
    )
    plt.close(figure)
    return path


def main() -> None:
    """Measure the requested rungs and publish their rows on stdout."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--cells", action="append", type=int)
    parser.add_argument("--figure", action="store_true")
    arguments = parser.parse_args()
    cases = arguments.case or list(CASES)
    if arguments.figure:
        for case_name in cases:
            for cells in arguments.cells or [10000]:
                print(
                    "MAP_FIDELITY_FIGURE " + str(_figure(case_name, cells)), flush=True
                )
        return
    cells = arguments.cells or sorted(RUNG_REQUESTS)
    rows = []
    for case_name in cases:
        for requested in cells:
            row = _measure_map(case_name, requested)
            print("MAP_FIDELITY_ROW " + json.dumps(row), flush=True)
            rows.append(row)
    print("MAP_FIDELITY_ROWS " + json.dumps(rows), flush=True)
    over = [
        row
        for row in rows
        if row["map_relative_sup"] > MAP_TOLERANCE
        or row["decomposition_relative_sup"] > DECOMPOSITION_TOLERANCE
    ]
    print("MAP_FIDELITY_VERDICT " + ("FAIL" if over else "PASS"), flush=True)
    raise SystemExit(1 if over else 0)


if __name__ == "__main__":
    main()