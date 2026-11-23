"""The certificate exterior must be posed on the exact analytic support.

The prescribed exterior is the analytic state minus the flux image of the
analytic-density plasma current.  When that image is integrated on the
production spline-clipped support, the exterior and the map's own booking
disagree on the same support, and the disagreement appears as a closure
residual that masks the booking error the map-fidelity gate exists to measure.

These tests pose the exterior on the exact analytic-separatrix polygon support
instead, and require the closure residual to fall below the booking term at
every rung.  The certified legacy arm keeps reproducing its recorded map error
under an explicitly named legacy exterior, which is the positive control.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import topology
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.support import geometry
from scripts.prototypes.support import measure as proto


CASE = "diverted-single-null"
# Requested cell counts chosen so the carrier realises about 550, 2,000 and
# 5,000 solver cells after boundary qualification.
RUNG_REQUESTS = {550: -500, 2000: -2000, 5000: -5000}
POSITIVE_CONTROL_MAP_SUP = 0.1973269787182143
# Carriers are loaded from the fixture's default segmented cache and rebuilt
# on a miss.  The certified 5,000-cell carrier and exterior from the reference
# run are staged into that cache before this gate runs.
CERTIFIED_CACHE = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261009T181553786165-proto-support/ordered/rows/cache"
)
DEFAULT_CACHE = Path(os.environ.get("NOVA_EXTERIOR_SUPPORT_CACHE", ".nova"))
BRANCH = topology.TopologyClass.DIVERTED

pytestmark = pytest.mark.slow

_MEASUREMENTS: dict[tuple[int, str], dict[str, object]] = {}


def _cache_dir(cells: int) -> Path:
    del cells
    return DEFAULT_CACHE


def _branch_for(case: str) -> topology.TopologyClass:
    if certificate._is_diverted_case(case):
        return topology.TopologyClass.DIVERTED
    return topology.TopologyClass.LIMITED


def build_carrier(case: str, machine, *, support_arm: str | None = None, branch=None):
    """Build the exact-support map state for one analytic case and carrier
    machine.

    ``support_arm=None`` takes the fixture's default exact-support exterior and
    overrides the operator's support partition with the analytic separatrix
    polygons; a named arm takes that exterior and keeps the operator's own
    partition, which is the legacy positive control.
    """
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    carrier, source, exact = certificate._case(case, clip_mode="exact")
    if branch is None:
        branch = _branch_for(case)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = jnp.asarray(
        certificate._exact_state(case, exact, coordinates), dtype=jnp.float64
    )
    empty = fixture.forward_operator(source, machine)
    span = abs(float(fixture._analytic_axis_flux(exact)))
    centres = np.asarray(empty.moment_geometry.atomic_mesh.centroids)
    physical, exterior, exterior_cache = fixture.cached_fixture_exterior(
        source,
        exact,
        machine,
        empty,
        analytic,
        support_arm="exact" if support_arm is None else support_arm,
    )
    operator = fixture.forward_operator(source, machine, exterior).with_clip_mode(
        "exact"
    )
    target, _centroid, _receipt = certificate._closed_form_current_target(
        case, source, operator, physical
    )
    exact_polygons = geometry.analytic_polygons(exact, machine.cell_polygons)
    true_physical = geometry.analytic_moments(source, exact_polygons, centres)
    reference_image = np.asarray(
        operator.current_moment_image(operator.coupling_current_moments(true_physical))
    )
    partition = jax.jit(lambda state: operator._support_partition(state, branch))(
        analytic
    )
    if support_arm is None:
        exact_support = geometry.pack(exact_polygons, partition[3])
        active = proto.override(
            operator,
            proto.partition(
                operator,
                analytic,
                jnp.asarray(fixture._analytic_axis_flux(exact)),
                jnp.asarray(0.0),
                fixture._analytic_saddle(exact),
                exact_support,
            ),
        )
    else:
        active = operator
    return {
        "carrier": carrier,
        "source": source,
        "exact": exact,
        "machine": machine,
        "analytic": analytic,
        "operator": operator,
        "active": active,
        "target": target,
        "branch": branch,
        "span": span,
        "exterior": exterior,
        "reference_image": reference_image,
        "exterior_cache": exterior_cache,
    }


def measure_arm(
    case: str, cells: int, machine, arm: str, *, branch=None
) -> tuple[dict[str, object], dict[str, object]]:
    """Return one case-rung's booking and closure terms, and its carrier.

    The record is the row this module's gate and the map-fidelity rungs both
    publish; the carrier comes back too, so a caller that draws the state
    reuses the build rather than repeating it.
    """
    built = build_carrier(
        case, machine, support_arm=None if arm == "exact" else arm, branch=branch
    )
    active = built["active"]
    operator = built["operator"]
    analytic = built["analytic"]
    branch_used = built["branch"]
    value = np.asarray(
        active.flux_map(requested_class=branch_used, target_current=built["target"])(
            analytic
        )
    )
    grid = len(built["machine"].node)
    delta = value[:grid] - np.asarray(analytic)[:grid]
    raw = active.cell_current_moments(analytic, branch_used)
    booked = float(jnp.sum(raw.cell_current))
    scaled = active.scaled_current_moments(
        raw, active.current_normalisation_amplitude(built["target"], booked)
    )
    booking_delta = np.asarray(
        operator.current_moment_image(scaled) - built["reference_image"]
    )[:grid]
    exterior_closure = np.asarray(
        built["exterior"] + built["reference_image"] - np.asarray(analytic)
    )[:grid]
    # The full physical map includes the extension at residual-shadow nodes.
    shadow = np.asarray(
        jax.jit(lambda state: operator.residual_shadow_mask(state, branch_used))(
            analytic
        )
    )[:grid]
    span = built["span"]
    result = {
        "case": case,
        "requested_cells": cells,
        "realised_cells": grid,
        "arm": arm,
        "span_wb": span,
        "map_relative_sup": float(np.max(np.abs(delta)) / span),
        "unshadowed_map_relative_sup": float(
            np.max(np.abs(np.where(shadow, 0.0, delta))) / span
        ),
        "booking_image_relative_sup": float(np.max(np.abs(booking_delta)) / span),
        "exterior_closure_relative_sup": float(np.max(np.abs(exterior_closure)) / span),
        "decomposition_relative_sup": float(
            np.max(np.abs(booking_delta + exterior_closure - delta)) / span
        ),
        "exterior_cache": built["exterior_cache"],
    }
    return result, built


def _arm_measurement(cells: int, arm: str) -> dict[str, object]:
    """Return the booking and exterior closure terms for one support arm.

    ``arm="exact"`` takes the fixture's default (exact analytic support)
    exterior, so flipping that default — the declared negative-control
    mutation — turns the closure clause red.  ``arm="legacy"`` names the
    legacy spline-support exterior explicitly.
    """
    key = (cells, arm)
    if key in _MEASUREMENTS:
        return _MEASUREMENTS[key]
    carrier, source, exact = certificate._case(CASE, clip_mode="exact")
    wall = certificate._diverted_wall(exact)
    machine = proto.machine_from_cache(
        carrier, RUNG_REQUESTS[cells], wall, _cache_dir(cells)
    )
    measured, _built = measure_arm(CASE, cells, machine, arm, branch=BRANCH)
    _MEASUREMENTS[key] = measured
    return measured


@pytest.mark.parametrize("cells", sorted(RUNG_REQUESTS))
def test_exterior_closure_falls_below_booking(cells: int) -> None:
    """The exact-support exterior closes below the map's own booking term."""
    measured = _arm_measurement(cells, "exact")
    print("SUPPORT_ARM_ROW " + json.dumps(measured, default=str))
    assert measured["decomposition_relative_sup"] < 1.0e-10, measured
    assert (
        measured["exterior_closure_relative_sup"]
        < measured["booking_image_relative_sup"]
    ), measured


def test_legacy_arm_reproduces_positive_control() -> None:
    """The named legacy exterior keeps the certified 550-cell map error."""
    measured = _arm_measurement(550, "legacy")
    print("SUPPORT_ARM_ROW " + json.dumps(measured, default=str))
    delta = abs(
        measured["unshadowed_map_relative_sup"] / POSITIVE_CONTROL_MAP_SUP - 1.0
    )
    assert delta <= 1.0e-9, (measured, delta)
