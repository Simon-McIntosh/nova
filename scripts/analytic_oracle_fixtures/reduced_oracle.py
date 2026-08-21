"""Seconds-class closed-form oracle for the default recovery lane."""

from __future__ import annotations

from time import perf_counter

import jax.numpy as jnp
import numpy as np

from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures.measure import (
    TOTAL_FLUX_FACTOR,
    _internal_flux_image,
    analytic_case,
    build_machine,
    exact_current_moments,
    exact_state,
    forward_operator,
)


REQUESTED_CELLS = -110
WALL_POINT_COUNT = 41


def measure_reduced_oracle() -> dict[str, object]:
    """Build the reduced carrier and evaluate its production map exactly once."""
    configure_dtypes()
    started = perf_counter()
    case = analytic_case()
    machine = build_machine(case, REQUESTED_CELLS, wall_nodes=WALL_POINT_COUNT)
    coordinates = np.vstack(
        [machine.node, machine.wall_node, machine.sample_coordinates]
    )
    oracle_state = exact_state(case, coordinates)

    zero_exterior = forward_operator(case, machine)
    exact_physical = exact_current_moments(case, zero_exterior, oracle_state)
    exact_coefficients = zero_exterior.coupling_current_moments(exact_physical)
    exact_internal = _internal_flux_image(zero_exterior, exact_coefficients)
    prescribed_exterior = oracle_state - exact_internal
    operator = forward_operator(case, machine, prescribed_exterior)

    map_evaluations = 0
    mapped = np.asarray(operator.flux_map()(jnp.asarray(oracle_state)))
    map_evaluations += 1
    forcing = mapped - oracle_state
    span = TOTAL_FLUX_FACTOR * case.axis_flux
    _, topology = operator.read(jnp.asarray(oracle_state))

    return {
        "construction": "closed-form field, profiles, density, and exact exterior",
        "requested_cells": REQUESTED_CELLS,
        "realised_cells": len(machine.node),
        "wall_rows": len(machine.wall_node),
        "state_size": len(oracle_state),
        "map_evaluations": map_evaluations,
        "forcing_sup_wb": float(np.max(np.abs(forcing))),
        "fixed_point_residual": float(
            np.max(np.abs(forcing)) / np.max(np.abs(oracle_state))
        ),
        "grid_forcing_fraction_of_span": float(
            np.max(np.abs(forcing[: len(machine.node)])) / span
        ),
        "gauge_receipt": {
            "raw_flux_comparison_gauge": "shared_exact_exterior",
            "psi_norm_root_anchors_from": "root_field",
            "psi_norm_oracle_anchors_from": "closed_form_field",
            "reference_gauge_constant_used": False,
            "axis_flux_wb": float(topology.axis_flux),
            "boundary_flux_wb": float(topology.boundary_flux),
        },
        "wall_seconds": perf_counter() - started,
    }


def convergence_clause_passes(
    coarse_deviation: float,
    fine_deviation: float,
    coarse_floor: float,
    fine_floor: float,
    *,
    flat_ratio: float = 0.8,
) -> bool:
    """Reject a non-converging excess regardless of its absolute magnitude."""
    coarse_excess = max(0.0, abs(coarse_deviation) - abs(coarse_floor))
    fine_excess = max(0.0, abs(fine_deviation) - abs(fine_floor))
    if coarse_excess == 0.0 or fine_excess == 0.0:
        return True
    return fine_excess / coarse_excess < flat_ratio
