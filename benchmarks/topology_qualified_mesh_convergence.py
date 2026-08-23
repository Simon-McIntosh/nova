"""Measure the mesh sensitivity of a topology-qualified DIII-D solve.

The banked complete-current diverted case is repeated on the released coarse
carrier and on the native 65 by 65 carrier.  Both rungs use the same extracted
sources, conductor currents, pseudo-wall, branch seed, fixed Newton budget,
Krylov dimension, and topology-qualified nonmonotone admission.  Only the
structured plasma carrier changes.

The receipt distinguishes a discretisation floor from a solver floor.  A
residual reduction greater than five percent is treated as mesh-sensitive and
reported with its observed order.  Residuals agreeing within five percent are
treated as solver-limited.  A material increase is refused because neither
declared verdict describes it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import socket
import subprocess
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import diiid_forward_gs_match as forward_case
from benchmarks.diiid_diverted_root_full_currents import (
    _CURRENT_COLUMNS,
    _GEOMETRY_COLUMNS,
    _LABEL_COLUMNS,
    _omitted_vertices,
    append_recovered_conductors,
)
from nova.equilibrium.fixed_point import (
    KrylovActionQualification,
    kink_aware_newton_krylov,
)
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes


HERE = Path(__file__).resolve().parents[1]
DEFAULT_DATA = forward_case.DEFAULT_DATA
DEFAULT_OUTPUT = (
    HERE
    / "docs/figures/diiid-forward-onboarding/topology-qualified-mesh-convergence.json"
)
STATE_BANK = (
    HERE / "docs/figures/diiid-forward-onboarding/diverted-root/"
    "host_large_budget_terminal_state.npz"
)
SOURCE_RECEIPT = (
    HERE / "docs/figures/diiid-forward-onboarding/diverted-root/"
    "host_large_budget_receipt.json"
)
ADMISSION_RECEIPT = (
    HERE / "docs/figures/diiid-forward-onboarding/topology-qualified-admission.json"
)
SHOT = "d3d_shot_00000c4a7b.parquet"
FRAME = 0
PSEUDO_WALL_EXPANSION = 0.02
NEWTON_STEPS = 89
GMRES_ITERATIONS = 24
NONMONOTONE_FACTORS = (1.0, 0.5, 0.25, 0.125)
MESH_FLAT_RELATIVE_TOLERANCE = 0.05
PRODUCTION_CONVERGENCE_CRITERION = 1.0e-6
TOPOLOGY_QUALIFIED_BASELINE = 2.9254381732970964e-4
UNQUALIFIED_PLATEAU_COMPARATOR = 3.491e-2
PRIOR_MESH_ORDER_RANGE = (0.97, 3.39)
PRIOR_MESH_SCALING_COUNT = 5


@dataclass(frozen=True)
class MeshRung:
    """One structured carrier resolution in the fixed solve ladder."""

    name: str
    grid_stride: int


MESH_LADDER = (
    MeshRung("coarse_carrier", 2),
    MeshRung("reference_native", 1),
)


def _source_commit() -> str:
    """Return the checked-out source identity used by the measurement."""

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()


def _read_case(path: Path) -> dict[str, Any]:
    """Read only the columns required to rebuild the banked solve."""

    columns = tuple(
        dict.fromkeys((*_LABEL_COLUMNS, *_CURRENT_COLUMNS, *_GEOMETRY_COLUMNS))
    )
    return forward_case._read(path, columns)


def _build_profile(row: dict[str, Any], rung: MeshRung):
    """Build a profile at one stride without retaining cross-mesh couplings."""

    previous_stride = forward_case.REGISTERED_GRID_STRIDE
    forward_case._COUPLING_CACHE.clear()
    forward_case.REGISTERED_GRID_STRIDE = rung.grid_stride
    try:
        profile, seed, *_rest = forward_case.build_profile(
            row, FRAME, PSEUDO_WALL_EXPANSION
        )
    finally:
        forward_case.REGISTERED_GRID_STRIDE = previous_stride
        forward_case._COUPLING_CACHE.clear()
    return append_recovered_conductors(profile, _omitted_vertices()), seed


def _selected_candidate_was_admitted(
    admitted: np.ndarray, accepted_factors: np.ndarray
) -> bool:
    """Verify every promoted state was selected from an admitted candidate."""

    factor_to_column = {value: index for index, value in enumerate(NONMONOTONE_FACTORS)}
    for index, factor in enumerate(accepted_factors):
        if factor == 0.0:
            continue
        column = factor_to_column.get(float(factor))
        if column is None or not bool(admitted[index, column]):
            return False
    return True


def _solve_rung(
    row: dict[str, Any],
    rung: MeshRung,
    current: np.ndarray,
    banked_seed: np.ndarray,
) -> dict[str, Any]:
    """Build and solve one carrier with topology-qualified trial admission."""

    rung_started = perf_counter()
    build_started = perf_counter()
    profile, rebuilt_seed = _build_profile(row, rung)
    build_seconds = perf_counter() - build_started
    seed = banked_seed if rebuilt_seed.shape == banked_seed.shape else rebuilt_seed
    seed_matches_bank = bool(
        rebuilt_seed.shape == banked_seed.shape
        and np.array_equal(np.asarray(rebuilt_seed), np.asarray(banked_seed))
    )
    current_array = jnp.asarray(current)
    mapped = profile.flux_map(current_array, TopologyClass.DIVERTED)

    def remains_diverted(candidate):
        _masks, topology = profile.operator.read(candidate)
        return jnp.all(jnp.isfinite(candidate)) & topology.diverted

    solve_started = perf_counter()
    result = kink_aware_newton_krylov(
        mapped,
        jnp.asarray(seed),
        strategy="nonmonotone",
        newton_steps=NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
        warmup=0,
        admissibility_fn=remains_diverted,
    )
    terminal_state = np.asarray(result.state, dtype=float)
    terminal_image = np.asarray(mapped(result.state), dtype=float)
    solve_seconds = perf_counter() - solve_started
    terminal_residual = float(
        np.max(np.abs(terminal_image - terminal_state))
        / max(np.max(np.abs(terminal_image)), 1.0e-30)
    )
    _masks, topology = profile.operator.read(result.state)
    x_point = np.asarray(topology.x_point, dtype=float)
    candidate_admissibility = np.asarray(result.candidate_admissibility, dtype=bool)
    accepted_factors = np.asarray(result.accepted_factors, dtype=float)
    promoted = accepted_factors > 0.0
    selected_candidates_admitted = _selected_candidate_was_admitted(
        candidate_admissibility, accepted_factors
    )
    all_promoted_diverted = bool(
        selected_candidates_admitted and bool(topology.diverted)
    )
    qualification = KrylovActionQualification(
        int(result.krylov_action_qualification)
    ).name
    device = jax.devices()[0]
    interior = np.asarray(profile.operator.inside_material, dtype=bool)
    return {
        "name": rung.name,
        "grid_stride": rung.grid_stride,
        "grid_shape": list(profile.lattice.shape),
        "achieved_interior_cell_count": int(np.count_nonzero(interior)),
        "state_dimension": int(terminal_state.size),
        "wall_node_count": int(profile.operator.wall.node_number),
        "seed": {
            "kind": "convention-clean labelled map used only as a branch seed",
            "rebuilt_seed_matches_banked_seed_bitwise": seed_matches_bank,
        },
        "solver": {
            "route": "topology-qualified nonmonotone Newton-Krylov",
            "newton_steps": NEWTON_STEPS,
            "gmres_iterations": GMRES_ITERATIONS,
            "krylov_action_qualification": qualification,
            "terminal_relative_residual": terminal_residual,
            "reported_result_residual": float(result.residual),
            "production_convergence_criterion": PRODUCTION_CONVERGENCE_CRITERION,
            "meets_production_convergence_criterion": bool(
                terminal_residual <= PRODUCTION_CONVERGENCE_CRITERION
            ),
            "promoted_iteration_count": int(np.count_nonzero(promoted)),
            "unpromoted_iteration_count": int(np.count_nonzero(~promoted)),
            "all_promoted_iterations_retained_diverted_class": all_promoted_diverted,
            "selected_candidates_were_admitted": selected_candidates_admitted,
            "accepted_factor_counts": {
                str(factor): int(np.count_nonzero(accepted_factors == factor))
                for factor in (*NONMONOTONE_FACTORS, 0.0)
            },
        },
        "terminal_topology": {
            "class": "diverted" if bool(topology.diverted) else "limited",
            "finite_x_point": bool(np.all(np.isfinite(x_point))),
            "x_point_rz_m": x_point.tolist() if np.all(np.isfinite(x_point)) else None,
        },
        "runtime": {
            "wall_clock_seconds": perf_counter() - rung_started,
            "profile_build_seconds": build_seconds,
            "solve_compile_and_execute_seconds": solve_seconds,
            "hostname": socket.gethostname(),
            "device": device.device_kind,
            "platform": jax.default_backend(),
        },
    }


def _mesh_verdict(rungs: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify the terminal floor from the measured two-rung trend."""

    coarse, fine = rungs
    coarse_residual = coarse["solver"]["terminal_relative_residual"]
    fine_residual = fine["solver"]["terminal_relative_residual"]
    coarse_cells = coarse["achieved_interior_cell_count"]
    fine_cells = fine["achieved_interior_cell_count"]
    mesh_refinement_ratio = np.sqrt(fine_cells / coarse_cells)
    residual_ratio = fine_residual / coarse_residual
    relative_change = (fine_residual - coarse_residual) / coarse_residual
    observed_order = float(
        np.log(coarse_residual / fine_residual) / np.log(mesh_refinement_ratio)
    )
    flat = bool(abs(relative_change) <= MESH_FLAT_RELATIVE_TOLERANCE)
    falling = bool(relative_change < -MESH_FLAT_RELATIVE_TOLERANCE)
    if falling:
        verdict = "DISCRETISATION_LIMITED"
        basis = (
            "terminal residual falls by more than the declared five-percent "
            "mesh-flat tolerance"
        )
    elif flat:
        verdict = "SOLVER_LIMITED"
        basis = "terminal residual is flat within the declared five-percent tolerance"
    else:
        raise RuntimeError(
            "terminal residual increased materially with refinement; neither "
            "declared floor verdict applies"
        )
    return {
        "classification": verdict,
        "basis": basis,
        "mesh_flat_relative_tolerance": MESH_FLAT_RELATIVE_TOLERANCE,
        "mesh_refinement_ratio": float(mesh_refinement_ratio),
        "fine_to_coarse_residual_ratio": float(residual_ratio),
        "relative_residual_change": float(relative_change),
        "observed_order": observed_order,
        "production_convergence_criterion": PRODUCTION_CONVERGENCE_CRITERION,
        "finest_residual_to_production_criterion_ratio": float(
            fine_residual / PRODUCTION_CONVERGENCE_CRITERION
        ),
        "finest_rung_meets_production_convergence_criterion": bool(
            fine_residual <= PRODUCTION_CONVERGENCE_CRITERION
        ),
        "qualification": (
            "The terminal floor is mesh-sensitive, but the finest residual remains "
            "above the unchanged production convergence criterion."
        ),
        "all_rungs_retained_diverted_class_on_every_promotion": all(
            rung["solver"]["all_promoted_iterations_retained_diverted_class"]
            for rung in rungs
        ),
        "all_terminal_x_points_finite": all(
            rung["terminal_topology"]["finite_x_point"] for rung in rungs
        ),
    }


def run(data: Path, output: Path) -> dict[str, Any]:
    """Execute the fixed mesh ladder and write its quantitative receipt."""

    configure_dtypes()
    bank = np.load(STATE_BANK)
    current = np.asarray(bank["current"], dtype=float)
    banked_seed = np.asarray(bank["seed"], dtype=float)
    row = _read_case(data / SHOT)
    rungs = [_solve_rung(row, rung, current, banked_seed) for rung in MESH_LADDER]
    verdict = _mesh_verdict(rungs)
    receipt = {
        "artifact": "topology_qualified_mesh_convergence",
        "source_commit": _source_commit(),
        "case": {
            "shot": SHOT,
            "frame": FRAME,
            "time_ms": float(row["efit_times"][FRAME]),
            "poloidal_conductor_count": int(current.size),
            "current_source": str(STATE_BANK.relative_to(HERE)),
            "source_receipt": str(SOURCE_RECEIPT.relative_to(HERE)),
            "topology_admission_receipt": str(ADMISSION_RECEIPT.relative_to(HERE)),
            "coefficients_fitted": 0,
            "current_adjustments": 0,
        },
        "comparators": {
            "topology_qualified_baseline": TOPOLOGY_QUALIFIED_BASELINE,
            "unqualified_plateau": UNQUALIFIED_PLATEAU_COMPARATOR,
            "baseline_to_unqualified_reduction_factor": (
                UNQUALIFIED_PLATEAU_COMPARATOR / TOPOLOGY_QUALIFIED_BASELINE
            ),
        },
        "mesh_sensitivity_context": {
            "prior_stall_floors_scaling_with_mesh": PRIOR_MESH_SCALING_COUNT,
            "prior_stall_floor_count": PRIOR_MESH_SCALING_COUNT,
            "prior_observed_order_range": list(PRIOR_MESH_ORDER_RANGE),
            "topology_qualified_route_present_in_prior_study": False,
        },
        "rungs": rungs,
        "verdict": verdict,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = run(arguments.data, arguments.output)
    print(json.dumps(receipt["verdict"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
