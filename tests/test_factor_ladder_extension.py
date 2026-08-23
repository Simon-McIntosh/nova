"""Contracts and measurement driver for the fixed backtracking ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from benchmarks import diiid_repaired_solve_remeasure as repaired_solve
    from benchmarks import topology_qualified_mesh_convergence as mesh_convergence
    from nova.equilibrium.fixed_point import (
        KrylovActionQualification,
        kink_aware_newton_krylov,
    )
    from nova.equilibrium.topology import TopologyClass
    from nova.jax.config import configure_dtypes


HERE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    HERE / "docs/figures/diiid-forward-onboarding/factor-ladder-extension.json"
)
FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)


def test_fixed_ladder_reaches_measured_fraction_under_jit_and_vmap():
    """The fixed result shape includes the smallest reachable safe trial."""

    def solve(initial):
        def admitted(candidate):
            return candidate[0] <= initial[0] + 0.04

        return kink_aware_newton_krylov(
            lambda state: jnp.ones_like(state),
            initial,
            strategy="nonmonotone",
            newton_steps=1,
            gmres_iterations=1,
            warmup=0,
            admissibility_fn=admitted,
        )

    result = jax.jit(jax.vmap(solve))(jnp.zeros((3, 1)))
    assert result.candidate_admissibility.shape == (3, 1, len(FACTORS))
    np.testing.assert_array_equal(
        np.asarray(result.candidate_admissibility[:, 0]),
        np.asarray([[False, False, False, False, False, True]] * 3),
    )
    np.testing.assert_allclose(np.asarray(result.accepted_factors), 0.03125)
    np.testing.assert_allclose(np.asarray(result.state), 0.03125)


def _source_commit() -> str:
    """Return the committed source identity used for the measurement."""

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()


def _selected_factors(result) -> list[float]:
    """Return every actually promoted factor in iteration order."""

    factors = np.asarray(result.accepted_factors, dtype=float)
    return [float(factor) for factor in factors if factor > 0.0]


def _selected_candidates_were_admitted(result) -> bool:
    """Verify every promoted factor names an admitted fixed-shape column."""

    admitted = np.asarray(result.candidate_admissibility, dtype=bool)
    factor_to_column = {factor: index for index, factor in enumerate(FACTORS)}
    for iteration, factor in enumerate(
        np.asarray(result.accepted_factors, dtype=float)
    ):
        if factor == 0.0:
            continue
        column = factor_to_column.get(float(factor))
        if column is None or not admitted[iteration, column]:
            return False
    return True


def _signed_relative_change(value: float, comparator: float) -> float:
    """Return a signed ratio whose sign states improvement or regression."""

    return (value - comparator) / comparator


def _measure_frame(
    data: Path,
    case: repaired_solve.FrameCase,
    banked_record: dict[str, Any],
) -> dict[str, Any]:
    """Run one fixed score-blind frame through the extended production route."""

    started = perf_counter()
    row = repaired_solve._read_case(data / case.shot)
    profile, current, target_current_a, time_ms, seed = repaired_solve._prepare_frame(
        row, case.frame
    )
    mapped = profile.flux_map(
        jnp.asarray(current), TopologyClass.DIVERTED, target_current_a
    )

    def remains_diverted(candidate):
        _masks, topology = profile.operator.read(candidate)
        return jnp.all(jnp.isfinite(candidate)) & topology.diverted

    result = kink_aware_newton_krylov(
        mapped,
        seed,
        strategy="nonmonotone",
        newton_steps=repaired_solve.NEWTON_STEPS,
        gmres_iterations=repaired_solve.GMRES_ITERATIONS,
        warmup=0,
        admissibility_fn=remains_diverted,
    )
    state = np.asarray(result.state, dtype=float)
    image = np.asarray(mapped(result.state), dtype=float)
    residual = float(
        np.max(np.abs(image - state)) / max(np.max(np.abs(image)), 1.0e-30)
    )
    _masks, topology = profile.operator.read(result.state)
    x_point = np.asarray(topology.x_point, dtype=float)
    selected_factors = _selected_factors(result)
    repaired_comparator = float(banked_record["terminal_relative_residual"])
    unqualified_comparator = float(case.previous_relative_residual)
    qualification = KrylovActionQualification(
        int(result.krylov_action_qualification)
    ).name
    return {
        "shot": case.shot,
        "frame": case.frame,
        "time_ms": time_ms,
        "target_current_a": float(target_current_a),
        "admitted_promotion_count_of_89": len(selected_factors),
        "banked_admitted_promotion_count_of_89": int(
            banked_record["promoted_iteration_count"]
        ),
        "terminal_relative_residual": residual,
        "banked_repaired_terminal_relative_residual": repaired_comparator,
        "banked_unqualified_plateau_relative_residual": unqualified_comparator,
        "signed_relative_change_vs_banked_repaired": _signed_relative_change(
            residual, repaired_comparator
        ),
        "signed_relative_change_vs_unqualified_plateau": _signed_relative_change(
            residual, unqualified_comparator
        ),
        "terminal_topology_class": (
            "diverted" if bool(topology.diverted) else "limited"
        ),
        "finite_terminal_x_point": bool(np.all(np.isfinite(x_point))),
        "terminal_x_point_rz_m": (
            x_point.tolist() if np.all(np.isfinite(x_point)) else None
        ),
        "krylov_action_qualification": qualification,
        "selected_candidates_were_admitted": _selected_candidates_were_admitted(result),
        "selected_factor_by_admitted_step": selected_factors,
        "selected_factor_counts": {
            str(factor): selected_factors.count(factor) for factor in FACTORS
        },
        "runtime_seconds": perf_counter() - started,
    }


def _measure_favourable_frame(data: Path) -> dict[str, Any]:
    """Repeat the established coarse and native favourable-frame solves."""

    bank = np.load(mesh_convergence.STATE_BANK)
    current = np.asarray(bank["current"], dtype=float)
    seed = np.asarray(bank["seed"], dtype=float)
    row = mesh_convergence._read_case(data / mesh_convergence.SHOT)
    banked = json.loads(mesh_convergence.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    measured_rungs = [
        mesh_convergence._solve_rung(row, rung, current, seed)
        for rung in mesh_convergence.MESH_LADDER
    ]
    records = []
    for measured, comparator in zip(measured_rungs, banked["rungs"], strict=True):
        records.append(
            {
                "name": measured["name"],
                "full_step_admission_count_of_89": measured["solver"][
                    "accepted_factor_counts"
                ]["1.0"],
                "banked_full_step_admission_count_of_89": comparator["solver"][
                    "accepted_factor_counts"
                ]["1.0"],
                "terminal_relative_residual": measured["solver"][
                    "terminal_relative_residual"
                ],
                "banked_terminal_relative_residual": comparator["solver"][
                    "terminal_relative_residual"
                ],
                "terminal_topology_class": measured["terminal_topology"]["class"],
                "finite_terminal_x_point": measured["terminal_topology"][
                    "finite_x_point"
                ],
            }
        )
    return {
        "shot": mesh_convergence.SHOT,
        "frame": mesh_convergence.FRAME,
        "rungs": records,
    }


def measure(data: Path, output: Path) -> dict[str, Any]:
    """Write the extended-ladder five-frame convergence receipt."""

    configure_dtypes()
    repaired_solve._validate_baseline()
    banked = json.loads(repaired_solve.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    banked_by_case = {
        (record["shot"], int(record["frame"])): record
        for record in banked["frame_records"]
    }
    records = [
        _measure_frame(data, case, banked_by_case[(case.shot, case.frame)])
        for case in repaired_solve.COHORT
    ]
    receipt = {
        "artifact": "factor_ladder_extension",
        "source_commit": _source_commit(),
        "solver_contract": {
            "route": "topology-qualified nonmonotone Newton-Krylov",
            "factor_ladder": list(FACTORS),
            "factor_count": len(FACTORS),
            "fixed_shape": True,
            "data_dependent_length": False,
            "jit_and_vmap_contract_tested": True,
            "admission_predicate_changed": False,
            "newton_steps": repaired_solve.NEWTON_STEPS,
            "gmres_iterations": repaired_solve.GMRES_ITERATIONS,
        },
        "comparison_contract": {
            "signed_relative_change": "(new - comparator) / comparator",
            "negative_means_lower_residual": True,
            "positive_means_higher_residual": True,
        },
        "frame_records": records,
        "cohort_counts": {
            "frames": len(records),
            "terminal_diverted": sum(
                record["terminal_topology_class"] == "diverted" for record in records
            ),
            "finite_terminal_x_point": sum(
                record["finite_terminal_x_point"] for record in records
            ),
            "krylov_action_accepted": sum(
                record["krylov_action_qualification"] == "ACCEPTED"
                for record in records
            ),
            "extended_rung_exercised_frames": sum(
                any(
                    factor <= 0.0625
                    for factor in record["selected_factor_by_admitted_step"]
                )
                for record in records
            ),
        },
        "favourable_frame_remeasure": _measure_favourable_frame(data),
        "convergence_gate": {
            "verdict": "DECLINED",
            "reason": (
                "Both the shipped hard-coded 1e-6 bound and the benchmark-registered "
                "1e-5 bound remain untraced."
            ),
            "shipped_bound": repaired_solve.SHIPPED_RESIDUAL_BOUND,
            "registered_bound": repaired_solve.REGISTERED_RESIDUAL_BOUND,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    """Run the quantitative measurement when invoked as a script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=repaired_solve.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = measure(arguments.data, arguments.output)
    print(json.dumps(receipt["cohort_counts"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
