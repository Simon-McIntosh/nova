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
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    carrier, source, exact = certificate._case(CASE, clip_mode="exact")
    wall = certificate._diverted_wall(exact)
    machine = proto.machine_from_cache(
        carrier, RUNG_REQUESTS[cells], wall, _cache_dir(cells)
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = jnp.asarray(
        certificate._exact_state(CASE, exact, coordinates), dtype=jnp.float64
    )
    empty = fixture.forward_operator(source, machine)
    span = abs(float(fixture._analytic_axis_flux(exact)))
    centres = np.asarray(empty.moment_geometry.atomic_mesh.centroids)

    if arm == "exact":
        physical, exterior, exterior_cache = fixture.cached_fixture_exterior(
            source, exact, machine, empty, analytic
        )
    else:
        physical, exterior, exterior_cache = fixture.cached_fixture_exterior(
            source, exact, machine, empty, analytic, support_arm="legacy"
        )
    operator = fixture.forward_operator(source, machine, exterior).with_clip_mode(
        "exact"
    )
    target, _centroid, _receipt = certificate._closed_form_current_target(
        CASE, source, operator, physical
    )

    exact_polygons = geometry.analytic_polygons(exact, machine.cell_polygons)
    true_physical = geometry.analytic_moments(source, exact_polygons, centres)
    reference_image = np.asarray(
        operator.current_moment_image(operator.coupling_current_moments(true_physical))
    )
    partition_fn = jax.jit(lambda state: operator._support_partition(state, BRANCH))
    legacy_partition = partition_fn(analytic)
    if arm == "exact":
        exact_support = geometry.pack(exact_polygons, legacy_partition[3])
        exact_partition = proto.partition(
            operator,
            analytic,
            jnp.asarray(fixture._analytic_axis_flux(exact)),
            jnp.asarray(0.0),
            fixture._analytic_saddle(exact),
            exact_support,
        )
        active = proto.override(operator, exact_partition)
    else:
        active = operator

    value = np.asarray(
        active.flux_map(requested_class=BRANCH, target_current=target)(analytic)
    )
    grid = len(machine.node)
    delta = value[:grid] - np.asarray(analytic)[:grid]
    raw = active.cell_current_moments(analytic, BRANCH)
    booked = float(jnp.sum(raw.cell_current))
    scaled = active.scaled_current_moments(
        raw, active.current_normalisation_amplitude(target, booked)
    )
    booking_delta = np.asarray(operator.current_moment_image(scaled) - reference_image)[
        :grid
    ]
    exterior_closure = np.asarray(exterior + reference_image - np.asarray(analytic))[
        :grid
    ]
    # The map leaves residual-shadow cells unevaluated, so both booked terms
    # are zeroed there before the decomposition is required to reconstruct it.
    shadow = np.asarray(
        jax.jit(lambda state: operator.residual_shadow_mask(state, BRANCH))(analytic)
    )[:grid]
    booking_delta = np.where(shadow, 0.0, booking_delta)
    exterior_closure = np.where(shadow, 0.0, exterior_closure)
    result = {
        "cells": cells,
        "realised_cells": len(machine.node),
        "arm": arm,
        "span_wb": span,
        "map_relative_sup": float(np.max(np.abs(delta)) / span),
        "booking_image_relative_sup": float(np.max(np.abs(booking_delta)) / span),
        "exterior_closure_relative_sup": float(
            np.max(np.abs(exterior_closure)) / span
        ),
        "decomposition_relative_sup": float(
            np.max(np.abs(booking_delta + exterior_closure - delta)) / span
        ),
        "exterior_cache": exterior_cache,
    }
    _MEASUREMENTS[key] = result
    return result


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
    delta = abs(measured["map_relative_sup"] / POSITIVE_CONTROL_MAP_SUP - 1.0)
    assert delta <= 1.0e-9, (measured, delta)