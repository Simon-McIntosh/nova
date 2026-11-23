"""Adjudicate raised operator order against mesh refinement on banked evidence.

The floor and cost comparison is a receipt-only synthesis.  The one live check
rebuilds the qualified analytic stable control twice on the same source tree:
once with the centroid current rule that produced the banked eigenvalue and once
with the degree-three-exact cubic-cross cell average.  This paired construction
isolates operator order when deciding whether the near-marginal pole moves.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.coupled_map_mode_identification import (
    BASELINE_LEADING_EIGENVALUE,
    _build_qualified_control,
    _dominant_eigenpairs,
)
from benchmarks.passive_closure_trace import (
    _build_analytic_control,
    _require_control_branch,
    _solve_analytic_control,
)
from nova.equilibrium.fixed_point import KrylovActionQualification
from nova.jax.config import configure_dtypes


OUTPUT = Path(
    "docs/figures/forward-operator-refinement/order-vs-mesh-adjudication.json"
)
FIGURE = Path("docs/figures/forward-operator-refinement/order-vs-mesh-adjudication.png")
MESH_RECEIPT = Path(
    "docs/figures/diiid-forward-onboarding/topology-qualified-mesh-convergence.json"
)
ORDER_RECEIPT = Path(
    "docs/figures/forward-operator-refinement/operator-order-floor-receipt.json"
)
MEMORY_RECEIPT = Path(
    "docs/figures/forward-operator-refinement/reference-native-resolution-default.json"
)
CRITERION_RECEIPT = Path(
    "docs/figures/forward-operator-refinement/criterion-family.json"
)
MODE_RECEIPT = Path(
    "docs/figures/forward-operator-refinement/coupled-map-mode-identification.json"
)
MESH_MEASUREMENT_COMMIT = "7a424c1236e8b69fd65e1575320ab5133ed37c90"
ORDER_MEASUREMENT_COMMIT = "862e712ae8c67db7955938da1cd1e4c785f2c386"
MATERIAL_LAMBDA_MOVEMENT_FRACTION = 0.1


def _git(*arguments: str) -> str:
    """Return one git identity fact."""

    return subprocess.check_output(
        ["git", *arguments], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _digest(path: Path) -> str:
    """Return the SHA-256 digest of one authority input."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    """Load one banked JSON receipt."""

    return json.loads(path.read_text())


def _strict(value: Any) -> Any:
    """Return finite built-ins suitable for an attributable JSON receipt."""

    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("adjudication receipt contains a non-finite scalar")
    return value


def _tree_for_commit(commit: str) -> str:
    """Resolve the frozen source tree named by a banked measurement."""

    return _git("rev-parse", f"{commit}^{{tree}}")


def _matrix_memory_bytes(cells: int, wall_nodes: int, sources: int) -> int:
    """Return float64 bytes in the four interaction matrices quoted by order."""

    values = cells * cells + wall_nodes * cells + cells * sources + wall_nodes * sources
    return 8 * values


def _matrix_digests(operator) -> dict[str, str]:
    """Digest the interaction matrices whose identity isolates operator order."""

    arrays = {
        "plasma_to_grid": operator.grid.plasma_target,
        "plasma_to_wall": operator.wall.plasma_target,
        "source_to_grid": operator.grid.source_target,
        "source_to_wall": operator.wall.source_target,
    }
    return {
        name: hashlib.sha256(np.asarray(array).tobytes()).hexdigest()
        for name, array in arrays.items()
    }


def _qualification_name(solved) -> str:
    """Return the structured Krylov qualification as a stable enum name."""

    value = int(np.asarray(solved.fixed_point.krylov_action_qualification))
    return KrylovActionQualification(value).name


def _control_mode(
    control,
    label: str,
    *,
    solve_passes: int = 1,
) -> dict[str, Any]:
    """Solve, qualify and measure the leading stable-control eigenvalue."""

    started = perf_counter()
    seed = control.seed_flux
    solve_receipts = []
    for pass_index in range(solve_passes):
        solved, solve_receipt = _solve_analytic_control(
            control.without_passive,
            seed,
            f"{label}_solve_pass_{pass_index + 1}",
        )
        solve_receipts.append(solve_receipt)
        seed = solved.flux
    _require_control_branch(
        control.reference_axis,
        control.material_boundary,
        solve_receipt,
    )
    qualification = _qualification_name(solved)
    if qualification != "ACCEPTED":
        raise RuntimeError(f"{label} Krylov action is {qualification}, not ACCEPTED")
    operator = control.without_passive.operator
    root = solved.flux
    mapped, tangent = jax.linearize(operator.flux_map(), root)
    drive = (
        control.with_passive.operator.external()
        - control.without_passive.operator.external()
    )
    values, _vectors, residuals = _dominant_eigenpairs(tangent, drive, count=2)
    leading = values[0]
    root_residual = float(
        np.max(np.abs(np.asarray(mapped - root)))
        / max(float(np.max(np.abs(np.asarray(root)))), np.finfo(float).tiny)
    )
    return {
        "label": label,
        "leading_eigenvalue": {
            "real": float(np.real(leading)),
            "imaginary": float(np.imag(leading)),
            "magnitude": float(abs(leading)),
        },
        "leading_ritz_residual_l2": float(residuals[0]),
        "root_relative_map_residual": root_residual,
        "krylov_action_qualification": qualification,
        "physical_branch_qualified": bool(
            solve_receipt["control_branch_qualification"]["qualified"]
        ),
        "direct_branch_receipt": solve_receipt["direct_branch_receipt"],
        "solve_passes": [
            {
                "pass": index,
                "terminal_relative_fixed_point_residual": item[
                    "terminal_relative_fixed_point_residual"
                ],
                "elapsed_seconds": item["elapsed_seconds"],
                "solver": item["solver"],
            }
            for index, item in enumerate(solve_receipts, start=1)
        ],
        "vertical_conditioning": {
            key: solve_receipt["vertical_conditioning"][key]
            for key in (
                "definition",
                "decay_index",
                "open_stability_window",
                "stable",
            )
        },
        "cell_average": {
            "enabled": operator.cell_average_stencil is not None,
            "stencil_shape": (
                None
                if operator.cell_average_stencil is None
                else list(operator.cell_average_stencil.shape)
            ),
            "weights": (
                None
                if operator.cell_average_weight is None
                else np.asarray(operator.cell_average_weight)
            ),
        },
        "interaction_matrix_sha256": _matrix_digests(operator),
        "elapsed_seconds": perf_counter() - started,
    }


def _lambda_check() -> dict[str, Any]:
    """Run the centroid-versus-raised-order check on one current source tree."""

    centroid = _control_mode(_build_qualified_control(), "centroid_cell_current")
    raised = _control_mode(
        _build_analytic_control(),
        "cubic_cross_cell_average",
        solve_passes=4,
    )
    if centroid["cell_average"]["enabled"]:
        raise RuntimeError(
            "the centroid comparator unexpectedly enables cell averaging"
        )
    if not raised["cell_average"]["enabled"]:
        raise RuntimeError("the raised-order control lacks its cell-average stencil")
    matrices_identical = (
        centroid["interaction_matrix_sha256"] == raised["interaction_matrix_sha256"]
    )
    if not matrices_identical:
        raise RuntimeError(
            "operator-order controls rebuilt different interaction matrices"
        )

    centroid_value = centroid["leading_eigenvalue"]["real"]
    raised_value = raised["leading_eigenvalue"]["real"]
    paired_movement = raised_value - centroid_value
    banked_movement = raised_value - BASELINE_LEADING_EIGENVALUE
    contraction_margin = 1.0 - BASELINE_LEADING_EIGENVALUE
    material_threshold = MATERIAL_LAMBDA_MOVEMENT_FRACTION * contraction_margin
    material = abs(paired_movement) >= material_threshold
    verdict = (
        "ORDER_MOVES_NEAR_MARGINAL_MODE"
        if material
        else "INTRINSIC_MAP_PROPERTY_CONFIRMED"
    )
    return {
        "source_commit": _git("rev-parse", "HEAD"),
        "source_tree": _git("rev-parse", "HEAD^{tree}"),
        "banked_leading_eigenvalue": BASELINE_LEADING_EIGENVALUE,
        "centroid_control": centroid,
        "raised_order_control": raised,
        "centroid_movement_from_banked": centroid_value - BASELINE_LEADING_EIGENVALUE,
        "raised_order_movement_from_banked": banked_movement,
        "paired_raised_minus_centroid_movement": paired_movement,
        "absolute_paired_movement": abs(paired_movement),
        "baseline_contraction_margin": contraction_margin,
        "material_movement_threshold": material_threshold,
        "movement_fraction_of_contraction_margin": abs(paired_movement)
        / contraction_margin,
        "interaction_matrices_byte_identical": matrices_identical,
        "verdict": verdict,
        "interpretation": (
            "The raised cell average does not materially move the physical "
            "near-marginal pole; it changes the achieved discretisation floor "
            "without rehabilitating or removing the intrinsic map mode."
            if not material
            else "The raised cell average materially moves the near-marginal pole; "
            "the intrinsic-map classification must be amended for operator order."
        ),
    }


def _adjudication_table(
    mesh: dict[str, Any],
    order: dict[str, Any],
    criterion_status: str,
) -> list[dict[str, Any]]:
    """Assemble the decision axes from their banked receipts."""

    coarse, fine = mesh["rungs"]
    order_floor = order["native_carrier_floor"]
    matrix_invariance = order["matrix_and_build_invariance"]
    sources = int(mesh["case"]["poloidal_conductor_count"])
    wall_nodes = int(coarse["wall_node_count"])
    coarse_memory = _matrix_memory_bytes(
        int(coarse["achieved_interior_cell_count"]), wall_nodes, sources
    )
    fine_memory = _matrix_memory_bytes(
        int(fine["achieved_interior_cell_count"]), wall_nodes, sources
    )
    step_fields = {
        "differencing_step": "not-applicable-on-exact-route",
        "step_selection_rule": "not-applicable-on-exact-route",
        "step_invariance.second_step": "not-applicable-on-exact-route",
        "step_invariance.floor_delta": "not-applicable-on-exact-route",
        "step_invariance.bound": "not-applicable-on-exact-route",
        "step_invariance.verdict": "not-applicable-on-exact-route",
        "reason": "the exact autodiff tangent has no differencing-step parameter",
    }
    return [
        {
            "axis": "mesh_refinement",
            "intervention": "33x33/1089 cells to 65x65/4225 cells",
            "floor": {
                "before": coarse["solver"]["terminal_relative_residual"],
                "after": fine["solver"]["terminal_relative_residual"],
                "relative_movement": mesh["verdict"]["relative_residual_change"],
                "percent_movement": 100.0 * mesh["verdict"]["relative_residual_change"],
                "classification": mesh["verdict"]["classification"],
            },
            "criterion_verdict": criterion_status,
            "build_cost": {
                "before_seconds": coarse["runtime"]["profile_build_seconds"],
                "after_seconds": fine["runtime"]["profile_build_seconds"],
                "after_over_before": fine["runtime"]["profile_build_seconds"]
                / coarse["runtime"]["profile_build_seconds"],
                "rebuild": "full interaction-matrix rebuild at each mesh",
            },
            "memory": {
                "scope": (
                    "four float64 interaction matrices quoted by the order receipt"
                ),
                "before_bytes": coarse_memory,
                "after_bytes": fine_memory,
                "after_over_before": fine_memory / coarse_memory,
                "excludes": "derivative and moment-companion matrices",
            },
            "krylov_action_qualification": [
                coarse["solver"]["krylov_action_qualification"],
                fine["solver"]["krylov_action_qualification"],
            ],
            "measurement_identity": {
                "commit": MESH_MEASUREMENT_COMMIT,
                "tree": _tree_for_commit(MESH_MEASUREMENT_COMMIT),
                "receipt_source_commit": mesh["source_commit"],
            },
            "step_invariance": step_fields,
        },
        {
            "axis": "operator_order",
            "intervention": (
                "centroid current rule to degree-three-exact five-node "
                "cubic-cross cell average on the unchanged 65x65 carrier"
            ),
            "floor": {
                "before": order_floor["banked_centroid_terminal_floor"],
                "after": order_floor["raised_order_terminal_floor"],
                "relative_movement": order_floor["relative_floor_movement"],
                "percent_movement": 100.0 * order_floor["relative_floor_movement"],
                "classification": "FLOOR_MOVED_MODESTLY",
            },
            "criterion_verdict": criterion_status,
            "build_cost": {
                "before_seconds": matrix_invariance["centroid_build_wall_seconds"],
                "after_seconds": matrix_invariance["raised_order_build_wall_seconds"],
                "relative_change": matrix_invariance["relative_build_wall_change"],
                "rebuild": "none; all four matrices byte-identical",
            },
            "memory": {
                "scope": "four float64 interaction matrices quoted by this receipt",
                "before_bytes": fine_memory,
                "after_bytes": fine_memory,
                "relative_change": 0.0,
                "excludes": "derivative and moment-companion matrices",
            },
            "krylov_action_qualification": order_floor["krylov_action_qualification"],
            "measurement_identity": {
                "commit": ORDER_MEASUREMENT_COMMIT,
                "tree": _tree_for_commit(ORDER_MEASUREMENT_COMMIT),
            },
            "step_invariance": step_fields,
        },
    ]


def _render(receipt: dict[str, Any], path: Path) -> None:
    """Plot decision-relevant floor, cost and eigenvalue comparisons."""

    table = receipt["adjudication_table"]
    lambda_check = receipt["raised_order_lambda_check"]
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)

    labels = ["mesh\nbefore", "mesh\nafter", "order\nbefore", "order\nafter"]
    floors = [
        table[0]["floor"]["before"],
        table[0]["floor"]["after"],
        table[1]["floor"]["before"],
        table[1]["floor"]["after"],
    ]
    colours = ["0.55", "tab:blue", "0.55", "tab:orange"]
    axes[0].bar(labels, floors, color=colours)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("terminal relative residual")
    axes[0].set_title("Achieved floor movement")

    build_ratios = [
        table[0]["build_cost"]["after_over_before"],
        1.0 + table[1]["build_cost"]["relative_change"],
    ]
    memory_ratios = [
        table[0]["memory"]["after_over_before"],
        1.0 + table[1]["memory"]["relative_change"],
    ]
    positions = np.arange(2)
    width = 0.34
    axes[1].bar(positions - width / 2, build_ratios, width, label="build")
    axes[1].bar(positions + width / 2, memory_ratios, width, label="memory")
    axes[1].set_xticks(positions, ["mesh", "order"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("after / before")
    axes[1].set_title("Resource multiplier")
    axes[1].legend(frameon=False)

    centroid = lambda_check["centroid_control"]["leading_eigenvalue"]["real"]
    raised = lambda_check["raised_order_control"]["leading_eigenvalue"]["real"]
    values = np.asarray([BASELINE_LEADING_EIGENVALUE, centroid, raised])
    movement = (values - BASELINE_LEADING_EIGENVALUE) * 1.0e6
    axes[2].bar(["banked", "paired\ncentroid", "raised\norder"], movement)
    axes[2].axhline(0.0, color="black", lw=0.8)
    axes[2].set_ylabel("eigenvalue movement from banked (ppm)")
    axes[2].set_title("Stable-control leading mode")

    for axis in axes:
        axis.grid(True, axis="y", alpha=0.25)
    figure.suptitle("Order versus mesh: floor movement is not route conditioning")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def measure(output: Path, figure: Path) -> dict[str, Any]:
    """Run the missing lambda check and bank the full adjudication."""

    configure_dtypes()
    mesh = _load(MESH_RECEIPT)
    order = _load(ORDER_RECEIPT)
    memory = _load(MEMORY_RECEIPT)
    criterion = _load(CRITERION_RECEIPT)
    mode = _load(MODE_RECEIPT)
    banked_mode = mode["stable_control"]["dominant_eigenpairs"][0]["eigenvalue"]["real"]
    if banked_mode != BASELINE_LEADING_EIGENVALUE:
        raise RuntimeError("the banked stable-control eigenvalue changed")
    gate = criterion["criterion_family"]["diiid_forward_gate"]
    criterion_status = gate["criterion_status"]
    if criterion_status != "NO_DEFENSIBLE_DIIID_TOLERANCE_DERIVED":
        raise RuntimeError("the gate-owner notification state changed")

    table = _adjudication_table(mesh, order, criterion_status)
    lambda_check = _lambda_check()
    mesh_memory_multiplier = table[0]["memory"]["after_over_before"]
    ruling = (
        "Mesh refinement moves the achieved exact-route residual floor by "
        "72.89%, versus 7.70% from raised order: mesh is the decisive floor "
        "lever, but it requires a full rebuild and multiplies the four quoted "
        f"matrix bytes by {mesh_memory_multiplier:.2f}, while order costs zero "
        "rebuild and zero added "
        "matrix memory. Neither result rehabilitates Picard or supplies a "
        "DIII-D tolerance: the map remains non-contractive and the banked floors "
        "are achieved solver residuals, not independent discretisation-error "
        "estimates. Use the raised order because its modest floor reduction is "
        "free on an existing carrier; use mesh when the larger floor movement "
        "justifies rebuild cost; treat route conditioning separately."
    )
    receipt = {
        "schema": "nova-order-vs-mesh-adjudication/1.0",
        "status": "banked",
        "completed_utc": datetime.now(UTC).isoformat(),
        "source_identity": {
            "commit": _git("rev-parse", "HEAD"),
            "tree": _git("rev-parse", "HEAD^{tree}"),
            "backend": jax.default_backend(),
            "device": jax.devices()[0].device_kind,
            "runtime_mitigations": {
                "XLA_FLAGS": os.environ["XLA_FLAGS"],
                "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ[
                    "XLA_PYTHON_CLIENT_PREALLOCATE"
                ],
            },
        },
        "authority_inputs": {
            str(path): {"sha256": _digest(path)}
            for path in (
                MESH_RECEIPT,
                ORDER_RECEIPT,
                MEMORY_RECEIPT,
                CRITERION_RECEIPT,
                MODE_RECEIPT,
            )
        },
        "adjudication_table": table,
        "raised_order_lambda_check": lambda_check,
        "non_contractivity_constraint": {
            "pinned_picard_spectral_radius": 1.1455670310089587,
            "unpinned_picard_spectral_radius": 1.2576631175347157,
            "resolution_claim_boundary": (
                "floor movement is credited to discretisation; Newton-Krylov "
                "conditioning and seed-direction sensitivity remain solver-route "
                "properties and no resolution intervention rehabilitates Picard"
            ),
        },
        "ruling": ruling,
        "diiid_gate_owner_notification": {
            "recipient": "owner of benchmarks/diiid_forward_gs_match.py",
            "notification_written": True,
            "adjudicated_state": criterion_status,
            "selected_relative_residual_bound": None,
            "gate_rerun_authorised_now": False,
            "reason": (
                "No blessed number is available: both 1e-5 and 1e-6 are "
                "unpassable on the banked qualified DIII-D route, and selecting "
                "a bound below the achieved floor would be circular."
            ),
            "reinstatement_prerequisite": gate["required_correction"][
                "independent_object_needed"
            ],
            "admissible_future_form": gate["required_correction"][
                "admissible_future_form"
            ],
        },
        "banked_memory_cross_check": {
            "same_4225_cell_carrier_active_coupling_gb": memory["resolution_rows"][
                "fresh_65_point"
            ]["coupling_memory"]["active_coupling_gb_decimal"],
            "same_4225_cell_carrier_resident_coupling_gb": memory["resolution_rows"][
                "fresh_65_point"
            ]["coupling_memory"]["resident_coupling_gb_decimal"],
            "qualification": (
                "This independent same-size carrier includes derivative arrays "
                "and zero linear-moment companions; it is a memory cross-check, "
                "not substituted for the table's four-matrix DIII-D accounting."
            ),
        },
    }
    strict_receipt = _strict(receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(strict_receipt, indent=2) + "\n")
    _render(strict_receipt, figure)
    return strict_receipt


def parser() -> argparse.ArgumentParser:
    """Return the benchmark command line."""

    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=OUTPUT)
    result.add_argument("--figure", type=Path, default=FIGURE)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    result = measure(arguments.output, arguments.figure)
    lambda_result = result["raised_order_lambda_check"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "lambda_verdict": lambda_result["verdict"],
                "paired_lambda_movement": lambda_result[
                    "paired_raised_minus_centroid_movement"
                ],
                "gate_owner_state": result["diiid_gate_owner_notification"][
                    "adjudicated_state"
                ],
            },
            indent=2,
        )
    )
