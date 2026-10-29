"""Measure the diverted 300-cell Solovev certificate row without banking it.

The certificate driver's ``_measure`` writes a part receipt into the repository's
figure tree and then runs rendering and scoring. This program stops at the
production solve and prints the fixed-point trace, so a revision comparison
reads the residual ladder rather than only its terminal value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import numpy as np

PATH_ROOT = Path(__file__).resolve().parents[3]
if str(PATH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATH_ROOT))

from benchmarks import solovev_certificate as certificate  # noqa: E402
from nova.equilibrium.stencil_mesh import StencilMesh  # noqa: E402
from nova.jax.config import (  # noqa: E402
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from scripts.analytic_oracle_fixtures import measure as oracle_fixture  # noqa: E402
from scripts.oracle_rebaseline import measure as recovery  # noqa: E402


def _finite(values: object) -> list[float | None]:
    array = np.asarray(values, dtype=np.float64).ravel()
    return [float(value) if np.isfinite(value) else None for value in array]


def measure(case_name: str, requested_cells: int) -> dict[str, object]:
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier_case, exact, requested_cells)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = certificate._exact_state(case_name, exact, coordinates)
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical, fixture_exterior, fixture_cache = (
        oracle_fixture.cached_fixture_exterior(
            source_case, exact, machine, empty_operator, oracle_state
        )
    )
    operator = oracle_fixture.forward_operator(source_case, machine, fixture_exterior)
    mesh = StencilMesh(machine.node, machine.stencil, machine.area)
    profile = certificate.ForwardProfile(
        operator, mesh, newton_steps=recovery.NEWTON_STEPS
    )
    target_current, current_centroid, current_receipt = (
        certificate._closed_form_current_target(
            case_name, source_case, operator, exact_physical
        )
    )
    seed, requested_class, seed_receipt = certificate._production_seed(
        profile, case_name, target_current, current_centroid, current_receipt
    )
    request = certificate._certificate_solve_request(
        profile,
        seed,
        target_current,
        carrier_identity=f"solovev:{case_name}:{requested_cells}",
    )
    receipt = profile.solve(request)
    equilibrium = receipt.equilibrium
    jax.block_until_ready(equilibrium.flux)
    terminal = equilibrium.fixed_point
    residual = float(terminal.residual)
    return {
        "case": case_name,
        "requested_cells": requested_cells,
        "cell_count": int(len(machine.node)),
        "coordinate_count": int(machine.node.shape[0]),
        "cache_directory": str(getattr(cache, "directory", cache)),
        "fixture_cache": {k: str(v) for k, v in dict(fixture_cache).items()},
        "terminal_residual": residual,
        "qualification": (
            "qualified"
            if np.isfinite(residual) and residual <= certificate.TERMINAL_RESIDUAL_BOUND
            else "unqualified"
        ),
        "terminal_bound": float(certificate.TERMINAL_RESIDUAL_BOUND),
        "converged": bool(terminal.converged),
        "termination_reason": int(terminal.termination_reason),
        "trace": _finite(terminal.trace),
        "active_set_iterations": int(terminal.active_set_iterations),
        "active_set_residuals": _finite(terminal.active_set_residuals),
        "attempted_newton_promotions": int(terminal.attempted_newton_promotions),
        "accepted_newton_promotions": int(terminal.accepted_newton_promotions),
        "inner_iteration_residuals_before": _finite(
            terminal.inner_iteration_residuals_before
        ),
        "inner_iteration_residuals_after": _finite(
            terminal.inner_iteration_residuals_after
        ),
        "inner_iteration_accepted": [
            int(value)
            for value in np.asarray(terminal.inner_iteration_accepted).ravel()
        ],
        "krylov_action_qualification": int(terminal.krylov_action_qualification),
        "seed": {
            "requested_class": seed_receipt["requested_class"],
            "construction": seed_receipt["construction"],
            "plasma_current_a": seed_receipt["plasma_current_a"],
            "supported_cell_count": seed_receipt["supported_cell_count"],
        },
        "target_current_a": float(target_current),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="diverted-single-null")
    parser.add_argument("--requested-cells", type=int, default=-300)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    import nova

    print(f"module={nova.__file__}", flush=True)
    row = measure(arguments.case, arguments.requested_cells)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "case": row["case"],
                "requested_cells": row["requested_cells"],
                "terminal_residual": row["terminal_residual"],
                "qualification": row["qualification"],
                "converged": row["converged"],
                "active_set_iterations": row["active_set_iterations"],
                "accepted_newton_promotions": row["accepted_newton_promotions"],
                "trace": row["trace"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
