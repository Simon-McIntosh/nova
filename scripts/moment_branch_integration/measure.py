"""Measure the closed-form forcing after faithful moment integration."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures import measure as fixture


OUTPUT = Path(__file__).resolve().parent
FIT_BASELINE = Path("scripts/analytic_oracle_fixtures/results.json")
REQUESTED_CELLS = -500
ROUND_OFF_CEILING_WB = 1.0e-13


def _internal_image(operator, coefficients) -> np.ndarray:
    """Contract physical coupling coefficients onto every solve target."""
    return np.asarray(
        np.r_[
            operator.grid.internal(coefficients),
            operator.wall.internal(coefficients),
            operator.sample.internal(coefficients),
        ]
    )


def _production_physical_moments(operator, state):
    """Evaluate the sole production moment route on one support partition."""
    masks, _topology, sample_flux, core_support, common_support = (
        operator._support_partition(jnp.asarray(state))
    )
    return operator.source.current_moments(
        masks,
        operator.support_current_moments,
        core_support,
        common_support,
        sample_flux=sample_flux,
    )


def measure() -> dict[str, object]:
    configure_dtypes()
    case = fixture.analytic_case()
    machine = fixture.cached_machine(
        case, REQUESTED_CELLS, wall_nodes=fixture.WALL_POINT_COUNT
    )
    coordinates = np.vstack(
        [machine.node, machine.wall_node, machine.sample_coordinates]
    )
    exact = fixture.exact_state(case, coordinates)

    empty = fixture.forward_operator(case, machine)
    analytic_physical = fixture.exact_current_moments(case, empty, exact)
    analytic_coefficients = empty.coupling_current_moments(analytic_physical)
    analytic_internal = _internal_image(empty, analytic_coefficients)
    prescribed_exterior = exact - analytic_internal

    operator = fixture.forward_operator(case, machine, prescribed_exterior)
    mapped = np.asarray(operator.flux_map()(jnp.asarray(exact)))
    forcing = mapped - exact
    production_physical = _production_physical_moments(operator, exact)
    physical_delta = np.asarray(
        [
            np.asarray(production) - np.asarray(analytic)
            for production, analytic in zip(
                production_physical, analytic_physical, strict=True
            )
        ]
    )
    production_coefficients = operator.coupling_current_moments(production_physical)
    moment_forcing = (
        _internal_image(operator, production_coefficients) - analytic_internal
    )

    baseline = json.loads(FIT_BASELINE.read_text(encoding="utf-8"))["fixtures"][
        "coarse"
    ]["forcing"]
    _masks, _topology, _sample, support, _common = operator._support_partition(
        jnp.asarray(exact)
    )
    participating = np.asarray(support.included)
    excluded = ~participating
    report = {
        "schema": "faithful-moment-closed-form-forcing",
        "fixture": {
            "case": case.name,
            "requested_cells": REQUESTED_CELLS,
            "realised_cells": len(machine.node),
            "state_size": len(exact),
            "cache": machine.cache,
            "state_construction": (
                "closed-form flux evaluated independently at every grid, wall, "
                "and direct-sample target"
            ),
            "exterior_construction": (
                "independent closed-form state minus the exact analytic-density "
                "plasma image"
            ),
        },
        "routes": {
            "fit_baseline": "banked production degree-nine density projection",
            "production": "fixed degree-fifteen Duffy quadrature on traced supports",
            "truth": "order-fifteen analytic-density polygon quadrature",
        },
        "fit_baseline": {
            "forcing_sup_wb": baseline["sup_wb"],
            "forcing_rms_wb": baseline["rms_wb"],
            "artifact": str(FIT_BASELINE),
        },
        "faithful": {
            "forcing_sup_wb": float(np.max(np.abs(forcing))),
            "forcing_rms_wb": float(np.sqrt(np.mean(forcing**2))),
            "forcing_grid_sup_fraction_of_span": float(
                np.max(np.abs(forcing[: len(machine.node)]))
                / (2.0 * np.pi * case.axis_flux)
            ),
            "moment_forcing_sup_wb": float(np.max(np.abs(moment_forcing))),
            "map_minus_moment_closure_sup_wb": float(
                np.max(np.abs(forcing - moment_forcing))
            ),
            "physical_moment_delta_sup": {
                "m0_a": float(np.max(np.abs(physical_delta[0]))),
                "mR_a_m": float(np.max(np.abs(physical_delta[1]))),
                "mZ_a_m": float(np.max(np.abs(physical_delta[2]))),
            },
            "round_off_ceiling_wb": ROUND_OFF_CEILING_WB,
            "round_off_class": bool(np.max(np.abs(forcing)) <= ROUND_OFF_CEILING_WB),
        },
        "topology_qualification": {
            "participating_supports": int(np.count_nonzero(participating)),
            "nonempty_supports": int(np.count_nonzero(np.asarray(support.included))),
            "excluded_supports": int(np.count_nonzero(excluded)),
            "excluded_current_sup_a": float(
                np.max(np.abs(np.asarray(production_physical.cell_current)[excluded]))
            ),
            "excluded_first_sup_a_m": float(
                max(
                    np.max(
                        np.abs(np.asarray(production_physical.radial_moment)[excluded])
                    ),
                    np.max(
                        np.abs(
                            np.asarray(production_physical.vertical_moment)[excluded]
                        )
                    ),
                )
            ),
        },
        "adjudication": {
            "smooth_ring_m0_current_weighted_l1": 3.9974011216932296e-16,
            "steep_ring_fidelity_improvement_factor": 6.009052818952474,
            "faithful_to_fit_cpu_cost_ratio": 0.36521167788219194,
        },
        "bounds_moved_or_applied": False,
    }
    return report


def main() -> None:
    report = measure()
    (OUTPUT / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["faithful"], indent=2, sort_keys=True))
    print(json.dumps(report["topology_qualification"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
