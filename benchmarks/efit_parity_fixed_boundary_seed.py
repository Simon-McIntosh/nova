"""Measure a declared-boundary interior solve before a free-boundary solve.

Nova has no MAST fixed-boundary profile entry point.  This benchmark assembles
that constraint from the existing rectangular lattice, Delta-star convention,
prescribed-anchor source evaluation, and scalar target-current normalisation.
The fixed solve holds the reference map on every node outside the reference
declared support and solves only the support interior.  Its terminal state is
then available as the initial state for the ordinary current-constrained free
branch.

The two paths through source space and boundary-condition space are distinct.
Source-strength homotopy scales the plasma source from zero.  The measurement
here changes the boundary condition from fixed to free while retaining the
full source strength.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.optimize import NoConvergence, newton_krylov
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from benchmarks.efit_forward_parity_slice import (  # noqa: E402
    DECOMPOSITION_BANK,
    select_slices_by_shot,
)
from benchmarks.efit_parity_boundary_volume import (  # noqa: E402
    _verify_protected_artifacts,
)
from benchmarks.efit_parity_tared_external_field import (  # noqa: E402
    MESH_STRIDES,
    PROTECTED_SOURCE,
    _implied_current,
    _mast_case_at_grid_stride,
)
from nova.equilibrium.convention import (  # noqa: E402
    delta_star_from_current_density,
)
from nova.imas.mast_solve_inputs import SHOT_STORE  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402


TARGET_SHOT = 21978
TARGET_SLICE = 35
OUTPUT_RECEIPT = Path(
    "docs/figures/efit-forward-parity/fixed-boundary-double-seed.json"
)
OUTPUT_FIGURE = Path("docs/figures/efit-forward-parity/fixed-boundary-double-seed.png")
BANKED_COLD_RESIDUALS = {
    "coarse": 1.035680e-2,
    "fine": 5.300e-3,
}
FIXED_BOUNDARY_TOLERANCE = 1.0e-8
NEWTON_PROMOTIONS = 12
KRYLOV_ITERATIONS = 12
PICARD_FINDING = {
    "constant_relaxation": {
        "iterations": 400,
        "relaxation": 0.5,
        "outcome": "stalled",
    },
    "adaptive_relaxation": {
        "iterations": 2000,
        "initial_relaxation": 0.2,
        "minimum_relaxation": 1.0e-6,
        "reduction_interval": 100,
        "terminal_unrelaxed_relative_sup_residual": 7.659068e-4,
        "best_unrelaxed_relative_sup_residual": 7.645682e-4,
        "outcome": "stalled",
    },
    "reading": (
        "Picard is too weak on the imposed-boundary residual; the result does "
        "not establish whether the fixed-boundary equation has a root"
    ),
}


@dataclass(frozen=True)
class DeclaredSupportDirichlet:
    """Sparse Delta-star solve over an arbitrary node-centred support."""

    radius: np.ndarray
    height: np.ndarray
    unknown: np.ndarray
    factor: Any
    boundary_terms: tuple[tuple[int, int, float], ...]

    def solve(self, source: np.ndarray, boundary: np.ndarray) -> np.ndarray:
        """Solve Delta-star(flux) = source while retaining exterior values."""
        source_flat = np.asarray(source, dtype=np.float64).reshape(-1)
        boundary_flat = np.asarray(boundary, dtype=np.float64).reshape(-1)
        right_hand_side = source_flat[self.unknown].copy()
        for row, neighbour, coefficient in self.boundary_terms:
            right_hand_side[row] -= coefficient * boundary_flat[neighbour]
        solved = boundary_flat.copy()
        solved[self.unknown] = self.factor.solve(right_hand_side)
        return solved


def _selected_reference(bank: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the predeclared single frozen reference and its qualification."""
    matches = [
        item
        for item in select_slices_by_shot(bank)
        if (int(item[0]["shot"]), int(item[0]["slice_index"]))
        == (TARGET_SHOT, TARGET_SLICE)
    ]
    if len(matches) != 1:
        raise RuntimeError("the frozen selection does not contain 21978 slice 35")
    return matches[0]


def _dirichlet_operator(profile) -> DeclaredSupportDirichlet:
    """Factor the rectangular Delta-star stencil on declared support nodes."""
    radius = np.asarray(profile.lattice.radius, dtype=np.float64)
    height = np.asarray(profile.lattice.height, dtype=np.float64)
    shape = (radius.size, height.size)
    support = np.asarray(profile.operator.declared_support, dtype=bool).reshape(shape)
    interior = np.asarray(profile.lattice.interior(), dtype=bool).reshape(shape)
    unknown_mask = support & interior
    unknown = np.flatnonzero(unknown_mask.reshape(-1))
    if unknown.size == 0:
        raise RuntimeError("the declared support has no interior stencil nodes")
    row_for_node = np.full(radius.size * height.size, -1, dtype=np.int64)
    row_for_node[unknown] = np.arange(unknown.size)
    radial_step = float(np.diff(radius).mean())
    vertical_step = float(np.diff(height).mean())
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    boundary_terms: list[tuple[int, int, float]] = []

    for row, node in enumerate(unknown):
        radial_index, vertical_index = np.unravel_index(node, shape)
        inverse_radial = 1.0 / (2.0 * radius[radial_index] * radial_step)
        neighbours = (
            (
                radial_index - 1,
                vertical_index,
                1.0 / radial_step**2 + inverse_radial,
            ),
            (
                radial_index + 1,
                vertical_index,
                1.0 / radial_step**2 - inverse_radial,
            ),
            (radial_index, vertical_index - 1, 1.0 / vertical_step**2),
            (radial_index, vertical_index + 1, 1.0 / vertical_step**2),
        )
        rows.append(row)
        columns.append(row)
        values.append(-2.0 / radial_step**2 - 2.0 / vertical_step**2)
        for neighbour_r, neighbour_z, coefficient in neighbours:
            neighbour = np.ravel_multi_index((neighbour_r, neighbour_z), shape)
            column = row_for_node[neighbour]
            if column >= 0:
                rows.append(row)
                columns.append(int(column))
                values.append(coefficient)
            else:
                boundary_terms.append((row, int(neighbour), coefficient))

    matrix = csr_matrix((values, (rows, columns)), shape=(unknown.size, unknown.size))
    return DeclaredSupportDirichlet(
        radius=radius,
        height=height,
        unknown=unknown,
        factor=splu(matrix.tocsc()),
        boundary_terms=tuple(boundary_terms),
    )


def _fixed_boundary_solve(
    profile,
    reference_state: np.ndarray,
    reference_grid: np.ndarray,
    target_current: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Apply benchmark-local Newton-Krylov to the imposed-boundary residual."""
    operator = _dirichlet_operator(profile)
    node_count = profile.lattice.node_count
    boundary = np.asarray(reference_grid, dtype=np.float64).reshape(-1)
    state = np.asarray(reference_state, dtype=np.float64).copy()
    state[:node_count] = boundary
    residual_trace: list[float] = []
    iterate_trace: list[np.ndarray] = []

    def mapped(unknown_state: np.ndarray):
        grid_state = boundary.copy()
        grid_state[operator.unknown] = unknown_state
        trial = state.copy()
        trial[:node_count] = grid_state
        moments, amplitude = profile.operator.normalised_current_moments(
            jnp.asarray(trial), target_current
        )
        cell_current = np.asarray(moments.cell_current, dtype=np.float64)
        current_density = cell_current / np.asarray(
            profile.lattice.cell_area, dtype=np.float64
        )
        source = np.asarray(
            delta_star_from_current_density(
                np.asarray(profile.lattice.node_radius, dtype=np.float64),
                current_density,
            ),
            dtype=np.float64,
        )
        solved = operator.solve(source, boundary)
        return solved, moments, float(amplitude), grid_state

    def relative_residual(unknown_state: np.ndarray) -> np.ndarray:
        solved, _moments, _amplitude, grid_state = mapped(unknown_state)
        scale = max(float(np.max(np.abs(solved[operator.unknown]))), 1.0)
        return (solved[operator.unknown] - grid_state[operator.unknown]) / scale

    def record(unknown_state: np.ndarray, residual: np.ndarray) -> None:
        iterate_trace.append(np.asarray(unknown_state, dtype=np.float64).copy())
        residual_trace.append(float(np.max(np.abs(residual))))

    initial = boundary[operator.unknown]
    solver_returned = True
    try:
        terminal_unknown = np.asarray(
            newton_krylov(
                relative_residual,
                initial,
                method="gmres",
                inner_maxiter=KRYLOV_ITERATIONS,
                maxiter=NEWTON_PROMOTIONS,
                f_tol=FIXED_BOUNDARY_TOLERANCE,
                line_search="armijo",
                callback=record,
            ),
            dtype=np.float64,
        )
    except NoConvergence as error:
        solver_returned = False
        terminal_unknown = np.asarray(error.args[0], dtype=np.float64)

    terminal_vector = relative_residual(terminal_unknown)
    terminal_residual = float(np.max(np.abs(terminal_vector)))
    if iterate_trace:
        best_index = int(np.argmin(residual_trace))
        best_unknown = iterate_trace[best_index]
        best_residual = residual_trace[best_index]
    else:
        best_index = 0
        best_unknown = terminal_unknown
        best_residual = terminal_residual
    if terminal_residual < best_residual:
        best_index = len(residual_trace)
        best_unknown = terminal_unknown
        best_residual = terminal_residual

    converged = bool(best_residual <= FIXED_BOUNDARY_TOLERANCE)
    distribution_unknown = terminal_unknown if converged else best_unknown
    _solved, _moments, amplitude, grid = mapped(distribution_unknown)
    state[:node_count] = grid
    terminal_moments, amplitude = profile.operator.normalised_current_moments(
        jnp.asarray(state), target_current
    )
    cell_current = np.asarray(terminal_moments.cell_current, dtype=np.float64)
    reference_current = _implied_current(profile, reference_grid)
    implied = np.asarray(reference_current["plasma_cell_current"], dtype=np.float64)
    compare = np.asarray(profile.operator.declared_support, dtype=bool) & np.asarray(
        reference_current["valid"], dtype=bool
    )
    denominator_floor = np.finfo(np.float64).eps * max(
        float(np.max(np.abs(implied[compare]))), 1.0
    )
    relative = np.abs(cell_current[compare] - implied[compare]) / np.maximum(
        np.abs(implied[compare]), denominator_floor
    )
    distribution = {
        "definition": (
            "abs(fixed_cell_current - Delta-star-implied_cell_current) / "
            "max(abs(Delta-star-implied_cell_current), eps*support_sup)"
        ),
        "cell_count": int(relative.size),
        "denominator_floor_a": denominator_floor,
        "minimum": float(np.min(relative)),
        "median": float(np.median(relative)),
        "p90": float(np.percentile(relative, 90.0)),
        "supremum": float(np.max(relative)),
    }
    fixed_integral = float(np.sum(cell_current[compare]))
    implied_integral = float(np.sum(implied[compare]))
    record = {
        "entry_point": "benchmark-local Newton-Krylov sparse Dirichlet constraint",
        "nova_fixed_boundary_entry_point_available": False,
        "diagnostic_solver_scope": (
            "benchmark-local; no production solver was added to nova.equilibrium"
        ),
        "constraint_surface": (
            "FluxLattice interior stencil + delta_star_from_current_density + "
            "DeclaredAnchorOperator.normalised_current_moments"
        ),
        "boundary_condition": (
            "reference flux retained outside the declared-support unknowns; "
            "the support-edge neighbours are the discrete Dirichlet boundary"
        ),
        "source": "reference prescribed p_prime and ff_prime",
        "normalisation": (
            "same prescribed anchors and scalar target current as free arm"
        ),
        "target_current_a": target_current,
        "route": "newton_krylov",
        "nonlinear_promotions": NEWTON_PROMOTIONS,
        "krylov_iterations_per_promotion": KRYLOV_ITERATIONS,
        "line_search": "armijo",
        "converged": converged,
        "solver_returned_without_no_convergence": solver_returned,
        "criterion": FIXED_BOUNDARY_TOLERANCE,
        "iterations": len(residual_trace),
        "terminal_relative_sup_residual": terminal_residual,
        "best_relative_sup_residual": best_residual,
        "best_iteration_index": best_index,
        "relative_sup_residual_trace": residual_trace,
        "distribution_state": (
            "converged terminal iterate" if converged else "best recorded iterate"
        ),
        "terminal_current_normalisation_amplitude": float(amplitude),
        "declared_support_node_count": int(
            np.count_nonzero(profile.operator.declared_support)
        ),
        "solved_interior_node_count": int(operator.unknown.size),
        "current_distribution_comparison": distribution,
        "current_integrals_on_same_support_a": {
            "fixed_boundary": fixed_integral,
            "delta_star_implied": implied_integral,
            "fixed_over_implied_minus_one": fixed_integral / implied_integral - 1.0,
            "fixed_over_target_minus_one": fixed_integral / target_current - 1.0,
        },
        "production_surface_if_promoted": {
            "owner_decision_required": True,
            "proposed_entry_point": "ForwardProfile.solve_fixed_boundary",
            "required_inputs": (
                "initial total-flux state, imposed boundary flux, declared support, "
                "requested target current, nonlinear route and tolerance"
            ),
            "required_receipt": (
                "fixed-point residual history, finite-state qualification, exact "
                "current normalisation, terminal current moments and boundary closure"
            ),
        },
    }
    return record, {
        "state": state,
        "cell_current": cell_current,
        "implied_cell_current": implied,
        "comparison_support": compare,
        "moments_cell_current": np.asarray(
            terminal_moments.cell_current, dtype=np.float64
        ),
    }


def _plot_distributions(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot fixed and implied current density plus their pointwise difference."""
    figure, axes = plt.subplots(
        len(rows), 3, figsize=(13.5, 3.8 * len(rows)), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    for row_index, row in enumerate(rows):
        profile = row["profile"]
        fields = row["fields"]
        shape = profile.lattice.shape
        area = np.asarray(profile.lattice.cell_area, dtype=np.float64)
        fixed = fields["cell_current"] / area
        implied = fields["implied_cell_current"] / area
        support = fields["comparison_support"]
        maps = (
            np.where(support, fixed, np.nan).reshape(shape),
            np.where(support, implied, np.nan).reshape(shape),
            np.where(support, fixed - implied, np.nan).reshape(shape),
        )
        titles = (
            "fixed-boundary j_phi",
            "Delta-star-implied j_phi",
            "fixed minus implied j_phi",
        )
        for axis, values, title in zip(axes[row_index], maps, titles, strict=True):
            image = axis.pcolormesh(
                profile.lattice.radius,
                profile.lattice.height,
                values.T,
                shading="nearest",
                cmap="RdBu_r",
            )
            axis.set_title(f"{row['mesh_name']} | {title}")
            axis.set_xlabel("R [m]")
            axis.set_aspect("equal")
            figure.colorbar(image, ax=axis, label="j_phi [A/m2]", shrink=0.82)
        axes[row_index, 0].set_ylabel("Z [m]")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_fixed_boundary_only(
    store: Path = SHOT_STORE,
    bank: Path = DECOMPOSITION_BANK,
    output_path: Path = OUTPUT_RECEIPT,
    figure_path: Path = OUTPUT_FIGURE,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure the imposed-boundary interior solve on both banked meshes."""
    configure_dtypes()
    selected, qualification = _selected_reference(bank)
    protected_before = _verify_protected_artifacts(
        json.loads(PROTECTED_SOURCE.read_text())
    )
    rows = []
    runtime = []
    for mesh_name, stride in MESH_STRIDES.items():
        case, context = _mast_case_at_grid_stride(
            store, selected, qualification, stride
        )
        target_current = abs(float(case["reference"]["plasma_current_a"]))
        fixed, fields = _fixed_boundary_solve(
            context["profile"],
            case["state"],
            context["reference_flux"],
            target_current,
        )
        rows.append(
            {
                "mesh": case["mesh"],
                "fixed_boundary_solve": fixed,
            }
        )
        runtime.append(
            {
                "mesh_name": mesh_name,
                "case": case,
                "context": context,
                "profile": context["profile"],
                "fixed": fixed,
                "fields": fields,
            }
        )
    _plot_distributions(runtime, figure_path)
    protected_after = _verify_protected_artifacts(
        json.loads(PROTECTED_SOURCE.read_text())
    )
    fixed_converged = [item for item in runtime if item["fixed"]["converged"]]
    if fixed_converged:
        fixed_verdict = "FIXED_BOUNDARY_STATE_EXISTS"
        fixed_statement = (
            "At least one imposed-boundary mesh reached the unchanged criterion, "
            "so its terminal state is eligible for the fixed-to-free seed."
        )
    else:
        fixed_verdict = "FIXED_BOUNDARY_NEWTON_STALLS"
        fixed_statement = (
            "Newton-Krylov stalls on both meshes even with the reference boundary "
            "imposed and confinement guaranteed by construction. The difficulty "
            "therefore persists after removing free-boundary motion; this run "
            "cannot attribute it to the free boundary."
        )
    receipt = {
        "receipt": {
            "kind": "fixed_boundary_then_free_seed_measurement",
            "status": (
                "fixed_boundary_complete_free_seed_pending"
                if any(item["fixed"]["converged"] for item in runtime)
                else "fixed_boundary_complete_free_seed_unavailable"
            ),
            "reference": {
                "shot": TARGET_SHOT,
                "slice_index": TARGET_SLICE,
            },
            "single_reference_only": True,
        },
        "selection_basis": {
            "observed_mesh_order": 0.97,
            "weakest_refinement_response_of_six": True,
            "closed_axis_enclosing_flux_surface": False,
            "reading": (
                "the reference lies in both low strata, making boundary "
                "representation distinguishable from interior refinement"
            ),
        },
        "reading_declared_before_measurement": {
            "materially_lower_seeded_residual": (
                "boundary-condition homotopy enters the plasma basin where "
                "source-strength homotopy did not; the low mesh order is "
                "plausibly a boundary-representation limit"
            ),
            "no_material_improvement": (
                "the low order is interior and the fixed-boundary route adds "
                "nothing on this lane"
            ),
        },
        "homotopy_distinction": {
            "banked_negative": (
                "source-strength homotopy scales plasma source from zero and "
                "converged to the dominating limited root"
            ),
            "this_measurement": (
                "boundary-condition homotopy changes fixed to free at full "
                "source strength"
            ),
        },
        "picard_iteration_finding": PICARD_FINDING,
        "fixed_boundary_outcome": {
            "converged_mesh_count": len(fixed_converged),
            "measured_mesh_count": len(runtime),
            "verdict": fixed_verdict,
            "statement": fixed_statement,
            "criterion_unchanged_after_observing_stalls": True,
        },
        "banked_cold_residuals": BANKED_COLD_RESIDUALS,
        "qualification_before_solve": qualification,
        "mesh_results": rows,
        "seeded_stage_eligible_meshes": [item["mesh_name"] for item in fixed_converged],
        "seeded_stage": {
            "status": "pending_after_fixed_boundary_commit"
            if fixed_converged
            else "unrun_no_converged_fixed_boundary_state",
            "passive_inclusive_circuit_background_built": False,
            "reason": (
                "No fixed-boundary state reached the unchanged criterion, so "
                "neither mesh supplied an admissible state and current-moment seed."
            )
            if not fixed_converged
            else "awaiting the mandatory fixed-boundary commit",
        },
        "protected_banked_artifacts": {
            "before": protected_before,
            "after": protected_after,
        },
        "figure": {
            "path": str(figure_path),
            "sha256": hashlib.sha256(figure_path.read_bytes()).hexdigest(),
            "content": (
                "fixed-boundary and Delta-star-implied current-density fields "
                "with their spatial difference"
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt, runtime


def main() -> None:
    """Run the fixed-boundary stage; the free seed is added after it is banked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--bank", type=Path, default=DECOMPOSITION_BANK)
    parser.add_argument("--output", type=Path, default=OUTPUT_RECEIPT)
    parser.add_argument("--figure", type=Path, default=OUTPUT_FIGURE)
    args = parser.parse_args()
    receipt, _runtime = run_fixed_boundary_only(
        args.store, args.bank, args.output, args.figure
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
