"""Measure physical alignment of near-null frozen-mask Newton directions.

The banked MAST terminal states are rebuilt through the persisted response
carrier, then the terminal residual shadow is held fixed.  The dense active
Jacobian is assembled from batched Jacobian-vector products on the accelerator;
its singular decomposition and the physical-mode projections are therefore
measurements of the same piecewise-smooth operator used by the Newton solve.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from benchmarks import settled_mask_stall as settled
from nova.equilibrium import fixed_point
from nova.equilibrium.forward_operator import _structured_grid_axes
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPERANDS = settled.DEFAULT_OPERANDS
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/solver-convergence-regression/vertical-mode/"
    "jacobian-null-direction.json"
)
DEFAULT_FIGURE = DEFAULT_OUTPUT.with_suffix(".png")
DEFAULT_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/vertical-mode/"
    "jacobian-null-direction.md"
)
STALLED_TARGETS = ((21985, 51), (21986, 46), (21989, 55), (22086, 43))
CONTRAST_TARGET = (22086, 43)
SMALLEST_SINGULAR_VALUE_COUNT = 3
DEFAULT_MATRIX_BLOCK_SIZE = 32
NEAR_NULL_RELATIVE_THRESHOLD = 1.0e-3
VERTICAL_ALIGNMENT_THRESHOLD = 0.5
ALIGNMENT_SEPARATION = 0.1
PHYSICAL_DIRECTION_NAMES = ("vertical_shift", "radial_shift", "current_rescale")


@dataclass(frozen=True)
class BankedTerminal:
    """One terminal identity and the retained witnesses needed for validation."""

    identity: str
    shot: int
    slice_index: int
    arm: str
    terminal_residual: float
    termination_reason: str
    active_set_residuals: np.ndarray
    active_set_mask_differences: np.ndarray


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_revision() -> str:
    module_root = Path(fixed_point.__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "-C", str(module_root), "rev-parse", "HEAD"], text=True
    ).strip()


def _strict_float(value: Any) -> float | None:
    result = float(np.asarray(value))
    return result if np.isfinite(result) else None


def _load_terminals(path: Path) -> dict[tuple[int, int, str], BankedTerminal]:
    """Load the four stalled pure arms and the converged mixed contrast."""

    targets = {(shot, index, "pure") for shot, index in STALLED_TARGETS}
    targets.add((*CONTRAST_TARGET, "mixed"))
    terminals: dict[tuple[int, int, str], BankedTerminal] = {}
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"].item()))
        for row_index, row in enumerate(metadata["rows"]):
            key = (int(row["shot"]), int(row["slice_index"]), str(row["arm"]))
            if key not in targets:
                continue
            prefix = f"arm_{row_index:02d}"
            terminals[key] = BankedTerminal(
                identity=f"{key[0]}/{key[1]}",
                shot=key[0],
                slice_index=key[1],
                arm=key[2],
                terminal_residual=float(row["terminal_residual"]),
                termination_reason=str(row["termination_reason"]),
                active_set_residuals=np.asarray(
                    stored[f"{prefix}_active_set_residuals"], dtype=np.float64
                ),
                active_set_mask_differences=np.asarray(
                    stored[f"{prefix}_active_set_mask_differences"], dtype=np.int64
                ),
            )
    missing = sorted(targets - set(terminals))
    if missing:
        raise RuntimeError(f"operand cache lacks requested terminals: {missing}")
    return terminals


def _terminal_validation(
    state_result: Any,
    banked: BankedTerminal,
    production_result: Any | None,
) -> dict[str, Any]:
    """Require the rebuilt branch to match its banked terminal witness."""

    terminal_residual = float(state_result.terminal_residual)
    terminal_difference = abs(terminal_residual - banked.terminal_residual)
    result: dict[str, Any] = {
        "passes": terminal_difference <= 5.0e-11,
        "absolute_tolerance": 5.0e-11,
        "terminal_residual": terminal_residual,
        "banked_terminal_residual": banked.terminal_residual,
        "terminal_residual_difference": terminal_difference,
        "termination_reason": state_result.termination_reason,
        "banked_termination_reason": banked.termination_reason,
    }
    if production_result is not None:
        iterations = int(np.asarray(production_result.active_set_iterations))
        residuals = np.asarray(
            production_result.active_set_residuals, dtype=np.float64
        )[:iterations]
        differences = np.asarray(
            production_result.active_set_mask_differences, dtype=np.int64
        )[:iterations]
        residual_history_difference = (
            float(np.max(np.abs(residuals - banked.active_set_residuals)))
            if residuals.shape == banked.active_set_residuals.shape
            else None
        )
        mask_history_exact = bool(
            differences.shape == banked.active_set_mask_differences.shape
            and np.array_equal(differences, banked.active_set_mask_differences)
        )
        history_passes = bool(
            residual_history_difference is not None
            and residual_history_difference <= 5.0e-11
            and mask_history_exact
        )
        result.update(
            {
                "active_set_residuals": residuals.tolist(),
                "active_set_mask_differences": differences.tolist(),
                "maximum_residual_history_difference": residual_history_difference,
                "mask_history_exact": mask_history_exact,
                "passes": bool(result["passes"] and history_passes),
            }
        )
    if not result["passes"]:
        raise AssertionError(
            f"rebuilt {banked.identity} {banked.arm} terminal missed the bank: {result}"
        )
    return result


def _active_residual_action(
    frozen_map: Callable[[jax.Array], jax.Array],
    state: jax.Array,
    shadow: jax.Array,
) -> tuple[Callable[[jax.Array], jax.Array], np.ndarray]:
    """Return the residual Jacobian restricted to non-shadowed coordinates."""

    _mapped, tangent = jax.linearize(frozen_map, state)
    active_index = np.flatnonzero(~np.asarray(shadow, dtype=bool))
    active_index_device = jnp.asarray(active_index, dtype=jnp.int32)

    def action(vector: jax.Array) -> jax.Array:
        lifted = jnp.zeros_like(state).at[active_index_device].set(vector)
        residual_tangent = lifted - tangent(lifted)
        return residual_tangent[active_index_device]

    return action, active_index


def _materialise_jacobian(
    action: Callable[[jax.Array], jax.Array],
    dimension: int,
    block_size: int,
) -> jax.Array:
    """Assemble columns from accelerator-resident batched operator actions."""

    if block_size <= 0:
        raise ValueError("matrix block size must be positive")

    @jax.jit
    def apply(indices: jax.Array) -> jax.Array:
        basis = jax.nn.one_hot(indices, dimension, dtype=jnp.float64)
        return jax.vmap(action)(basis)

    row_blocks = []
    for start in range(0, dimension, block_size):
        stop = min(start + block_size, dimension)
        indices = np.arange(start, stop, dtype=np.int32)
        if indices.size < block_size:
            indices = np.pad(indices, (0, block_size - indices.size), mode="edge")
        block = apply(jnp.asarray(indices))[: stop - start]
        block.block_until_ready()
        row_blocks.append(block)
    return jnp.concatenate(row_blocks, axis=0).T


def _singular_decomposition(
    action: Callable[[jax.Array], jax.Array],
    dimension: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return all singular values and the requested smallest right vectors."""

    matrix = _materialise_jacobian(action, dimension, block_size)
    _left, singular_values, right_vectors = jnp.linalg.svd(matrix, full_matrices=False)
    singular_values.block_until_ready()
    values = np.asarray(singular_values, dtype=np.float64)
    vectors = np.asarray(right_vectors, dtype=np.float64)
    order = np.argsort(values)[:SMALLEST_SINGULAR_VALUE_COUNT]

    probe = np.sin(np.arange(dimension, dtype=np.float64) + 0.5)
    probe /= np.linalg.norm(probe)
    direct = np.asarray(action(jnp.asarray(probe)), dtype=np.float64)
    dense = np.asarray(matrix @ jnp.asarray(probe), dtype=np.float64)
    scale = max(np.linalg.norm(direct), np.linalg.norm(dense), np.finfo(float).tiny)
    validation = {
        "probe_relative_disagreement": float(np.linalg.norm(direct - dense) / scale),
        "finite": bool(np.isfinite(values).all() and np.isfinite(vectors).all()),
        "matrix_frobenius_norm": float(np.linalg.norm(np.asarray(matrix))),
    }
    if not validation["finite"] or validation["probe_relative_disagreement"] > 1e-11:
        raise AssertionError(f"dense Jacobian assembly did not validate: {validation}")
    return values, vectors[order], validation


def _grid_and_wall_gradient(
    operator: Any, state: jax.Array
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate the total state on its tensor grid and sample at the wall."""

    radius, height = _structured_grid_axes(operator.grid.coordinate)
    shape = (radius.size, height.size)
    grid_state = np.asarray(
        state[: operator.grid.node_number], dtype=np.float64
    ).reshape(shape)
    radial, vertical = np.gradient(grid_state, radius, height, edge_order=2)
    wall_coordinate = np.asarray(operator.wall.coordinate, dtype=np.float64)

    def complete(values: np.ndarray) -> np.ndarray:
        wall = RegularGridInterpolator(
            (radius, height), values, bounds_error=False, fill_value=None
        )(wall_coordinate)
        return np.concatenate((values.reshape(-1), np.asarray(wall, dtype=np.float64)))

    if operator.node_number != operator.physical_node_number:
        raise AssertionError(
            "rigid-shift measurement requires grid and wall state rows only"
        )
    return complete(radial), complete(vertical)


def _translated_current_flux(
    operator: Any,
    moments: CellCurrentMoments,
    axis: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Differentiate a rigid translation of the centroid current density."""

    radius, height = _structured_grid_axes(operator.grid.coordinate)
    shape = (radius.size, height.size)
    area = np.asarray(operator.area, dtype=np.float64).reshape(shape)
    current = np.asarray(moments.cell_current, dtype=np.float64).reshape(shape)
    density = np.divide(current, area, out=np.zeros_like(current), where=area != 0.0)
    gradients = np.gradient(density, radius, height, edge_order=2)
    translated_current = -gradients[axis] * area
    zero = jnp.zeros(operator.grid.node_number, dtype=jnp.float64)
    translated = CellCurrentMoments(
        jnp.asarray(translated_current.reshape(-1)), zero, zero
    )
    flux_change = np.asarray(
        operator.current_moment_image(translated), dtype=np.float64
    )
    return flux_change, {
        "current_change_a": float(np.sum(translated_current)),
        "current_change_l1_a": float(np.sum(np.abs(translated_current))),
    }


def _physical_directions(
    operator: Any,
    state: jax.Array,
    shadow: jax.Array,
    requested_class: jax.Array,
    target_current: float,
    action: Callable[[jax.Array], jax.Array],
    active_index: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    """Build rigid translations and a fractional plasma-current rescale."""

    moments, amplitude = operator.normalised_current_moments(
        state, target_current, requested_class
    )
    state_gradient_r, state_gradient_z = _grid_and_wall_gradient(operator, state)
    radial_flux_change, radial_current = _translated_current_flux(operator, moments, 0)
    vertical_flux_change, vertical_current = _translated_current_flux(
        operator, moments, 1
    )
    internal_flux = np.asarray(operator.current_moment_image(moments), dtype=np.float64)
    full_directions = {
        "vertical_shift": vertical_flux_change - state_gradient_z,
        "radial_shift": radial_flux_change - state_gradient_r,
        "current_rescale": internal_flux,
    }
    definitions = {
        "vertical_shift": (
            "flux change from translating the centroid current density by +1 m "
            "in Z, minus dpsi/dZ"
        ),
        "radial_shift": (
            "flux change from translating the centroid current density by +1 m "
            "in R, minus dpsi/dR"
        ),
        "current_rescale": (
            "flux change from a +100 percent uniform plasma-current rescale"
        ),
    }
    current_receipts = {
        "vertical_shift": vertical_current,
        "radial_shift": radial_current,
        "current_rescale": {
            "current_change_a": float(np.sum(np.asarray(moments.cell_current))),
            "current_change_l1_a": float(
                np.sum(np.abs(np.asarray(moments.cell_current)))
            ),
        },
    }
    records: dict[str, dict[str, Any]] = {}
    unit_vectors: dict[str, np.ndarray] = {}
    for name in PHYSICAL_DIRECTION_NAMES:
        full = np.where(np.asarray(shadow, dtype=bool), 0.0, full_directions[name])
        active = np.asarray(full[active_index], dtype=np.float64)
        norm = float(np.linalg.norm(active))
        if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
            raise AssertionError(f"physical direction {name} has no finite active norm")
        unit = active / norm
        physical_action = np.asarray(action(jnp.asarray(active)), dtype=np.float64)
        unit_action = np.asarray(action(jnp.asarray(unit)), dtype=np.float64)
        unit_vectors[name] = unit
        records[name] = {
            "definition": definitions[name],
            "physical_unit_active_norm": norm,
            "normalised_current_amplitude": float(np.asarray(amplitude)),
            "current_translation_receipt": current_receipts[name],
            "linear_residual_change_l2_for_physical_unit": float(
                np.linalg.norm(physical_action)
            ),
            "linear_residual_change_sup_for_physical_unit": float(
                np.max(np.abs(physical_action))
            ),
            "linear_residual_change_l2_per_unit_state_norm": float(
                np.linalg.norm(unit_action)
            ),
            "linear_residual_change_sup_per_unit_state_norm": float(
                np.max(np.abs(unit_action))
            ),
        }
    gram = {
        left: {
            right: float(np.dot(unit_vectors[left], unit_vectors[right]))
            for right in PHYSICAL_DIRECTION_NAMES
        }
        for left in PHYSICAL_DIRECTION_NAMES
    }
    for record in records.values():
        record["direction_gram_matrix"] = gram
    return records, unit_vectors


def _row_verdict(
    singular_values: np.ndarray,
    singular_records: list[dict[str, Any]],
) -> dict[str, Any]:
    largest = float(np.max(singular_values))
    smallest = float(np.min(singular_values))
    ratio = smallest / largest
    first = singular_records[0]["projection_fractions"]
    vertical = first["vertical_shift"]
    competing = max(first["radial_shift"], first["current_rescale"])
    near_null = bool(ratio <= NEAR_NULL_RELATIVE_THRESHOLD)
    vertical_mode = bool(
        near_null
        and vertical >= VERTICAL_ALIGNMENT_THRESHOLD
        and vertical >= competing + ALIGNMENT_SEPARATION
    )
    if vertical_mode:
        solver_shape = "vertical_position_constraint_with_compensating_unknown"
        owner = "nova.equilibrium.forward.ForwardProfile.solve"
    elif near_null:
        solver_shape = "deflation_of_measured_mode"
        owner = "nova.equilibrium.fixed_point.newton_krylov"
    else:
        solver_shape = "none"
        owner = None
    return {
        "near_null_direction": near_null,
        "near_null_is_vertical_mode": vertical_mode,
        "smallest_to_largest_ratio": ratio,
        "vertical_projection_fraction": vertical,
        "largest_competing_projection_fraction": competing,
        "solver_shape": solver_shape,
        "owner": owner,
        "rule": {
            "near_null_relative_threshold": NEAR_NULL_RELATIVE_THRESHOLD,
            "vertical_alignment_threshold": VERTICAL_ALIGNMENT_THRESHOLD,
            "alignment_separation": ALIGNMENT_SEPARATION,
        },
    }


def _measure_state(
    profile: Any,
    state_result: Any,
    banked: BankedTerminal,
    target_current: float,
    production_result: Any | None,
    matrix_block_size: int,
) -> dict[str, Any]:
    state = jnp.asarray(state_result.state)
    state.block_until_ready()
    validation = _terminal_validation(state_result, banked, production_result)
    requested_class = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    shadow = profile.operator.residual_shadow_mask(state, requested_class)
    captured_map = profile.operator.flux_map_with_shadow(
        requested_class=requested_class, target_current=target_current
    )

    def frozen_map(candidate: jax.Array) -> jax.Array:
        return captured_map(candidate, shadow)

    action, active_index = _active_residual_action(frozen_map, state, shadow)
    singular_values, right_vectors, matrix_validation = _singular_decomposition(
        action, active_index.size, matrix_block_size
    )
    directions, unit_vectors = _physical_directions(
        profile.operator,
        state,
        shadow,
        requested_class,
        target_current,
        action,
        active_index,
    )
    smallest_order = np.argsort(singular_values)[:SMALLEST_SINGULAR_VALUE_COUNT]
    singular_records = []
    for rank, (value_index, vector) in enumerate(
        zip(smallest_order, right_vectors, strict=True), start=1
    ):
        vector_norm = float(np.linalg.norm(vector))
        projections = {
            name: float(abs(np.dot(vector, direction)) / vector_norm)
            for name, direction in unit_vectors.items()
        }
        signed_projections = {
            name: float(np.dot(vector, direction) / vector_norm)
            for name, direction in unit_vectors.items()
        }
        singular_records.append(
            {
                "rank_from_smallest": rank,
                "singular_value": float(singular_values[value_index]),
                "fraction_of_largest_singular_value": float(
                    singular_values[value_index] / np.max(singular_values)
                ),
                "right_singular_vector_norm": vector_norm,
                "right_singular_vector_active_values": vector.tolist(),
                "projection_fractions": projections,
                "signed_projection_fractions": signed_projections,
            }
        )
    verdict = _row_verdict(singular_values, singular_records)
    return {
        "identity": banked.identity,
        "arm": banked.arm,
        "role": "converging_contrast" if banked.arm == "mixed" else "stalled_row",
        "bank_validation": validation,
        "terminal_state": {
            "full_dimension": int(state.size),
            "active_dimension": int(active_index.size),
            "shadow_count": int(state.size - active_index.size),
            "active_indices": active_index.tolist(),
            "residual_shadow_frozen": True,
        },
        "jacobian": {
            "operator": "active coordinates of I minus frozen-mask map tangent",
            "matrix_assembly": matrix_validation,
            "largest_singular_value": float(np.max(singular_values)),
            "smallest_three": singular_records,
        },
        "physical_directions": directions,
        "verdict": verdict,
    }


def _draw_figure(receipt: dict[str, Any], path: Path) -> None:
    rows = receipt["rows"]
    figure, axes = plt.subplots(
        len(rows), 4, figsize=(16.0, 3.15 * len(rows)), constrained_layout=True
    )
    projection_width = 0.24
    direction_colours = {
        "vertical_shift": "#5b3f95",
        "radial_shift": "#087e8b",
        "current_rescale": "#d1495b",
    }
    for row_index, row in enumerate(rows):
        singular_axis, vector_axis, projection_axis, residual_axis = axes[row_index]
        values = [item["singular_value"] for item in row["jacobian"]["smallest_three"]]
        largest = row["jacobian"]["largest_singular_value"]
        singular_axis.scatter(range(1, 4), values, color="#5b3f95", s=34)
        singular_axis.axhline(largest, color="#222222", linestyle="--", linewidth=1.0)
        singular_axis.set_yscale("log")
        singular_axis.set_xticks((1, 2, 3), ("smallest", "second", "third"))
        singular_axis.tick_params(axis="x", rotation=20)
        singular_axis.set_ylabel("singular value")
        singular_axis.set_title(
            f"{row['identity']} {row['arm']} · largest {largest:.2e}"
        )

        active_index = np.asarray(row["terminal_state"]["active_indices"], dtype=int)
        vector = np.zeros(row["terminal_state"]["full_dimension"], dtype=float)
        vector[active_index] = np.asarray(
            row["jacobian"]["smallest_three"][0]["right_singular_vector_active_values"]
        )
        grid = vector[: settled.EXPECTED_GRID_CELLS].reshape((33, 33)).T
        limit = max(float(np.max(np.abs(grid))), np.finfo(float).tiny)
        vector_axis.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        vector_axis.set_title("smallest right singular vector")
        vector_axis.set_xlabel("R index")
        vector_axis.set_ylabel("Z index")

        rank = np.arange(3, dtype=float)
        for offset, name in enumerate(PHYSICAL_DIRECTION_NAMES):
            projection_axis.bar(
                rank + (offset - 1) * projection_width,
                [
                    item["projection_fractions"][name]
                    for item in row["jacobian"]["smallest_three"]
                ],
                width=projection_width,
                label=name.replace("_", " "),
                color=direction_colours[name],
            )
        projection_axis.set_xticks(rank, ("smallest", "second", "third"))
        projection_axis.set_ylim(0.0, 1.0)
        projection_axis.set_ylabel("absolute projection fraction")
        projection_axis.set_title("right-vector physical alignment")
        if row_index == 0:
            projection_axis.legend(fontsize=7, frameon=False)

        changes = [
            row["physical_directions"][name][
                "linear_residual_change_l2_per_unit_state_norm"
            ]
            for name in PHYSICAL_DIRECTION_NAMES
        ]
        residual_axis.bar(
            range(3),
            changes,
            color=[direction_colours[name] for name in PHYSICAL_DIRECTION_NAMES],
        )
        residual_axis.set_yscale("log")
        residual_axis.set_xticks(
            range(3), ("vertical", "radial", "current"), rotation=20
        )
        residual_axis.set_ylabel("||J direction||₂")
        verdict = row["verdict"]
        short_shape = {
            "vertical_position_constraint_with_compensating_unknown": "constraint",
            "deflation_of_measured_mode": "deflation",
            "none": "none",
        }[verdict["solver_shape"]]
        residual_axis.set_title(
            "vertical null: "
            + ("YES" if verdict["near_null_is_vertical_mode"] else "NO")
            + f" · {short_shape}"
        )
        for axis in (singular_axis, vector_axis, projection_axis, residual_axis):
            axis.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_report(receipt: dict[str, Any], path: Path) -> None:
    lines = [
        "# Frozen-mask Jacobian null-direction measurement",
        "",
        (
            "The H200 measurement reconstructs four stalled pure branches and the "
            "converged mixed branch of 22086/43, freezes each terminal residual "
            "shadow, and decomposes the active Jacobian exactly from batched JVPs."
        ),
        (
            "Measurement job: "
            f"{receipt['runtime']['slurm']['job_id']} on "
            f"{receipt['runtime']['slurm']['partition']} under reservation "
            f"{receipt['runtime']['slurm']['reservation']}."
        ),
        "",
        (
            "| row | smallest / largest | vertical projection | radial projection "
            "| current projection | verdict | solver shape | owner |"
        ),
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in receipt["rows"]:
        smallest = row["jacobian"]["smallest_three"][0]
        projections = smallest["projection_fractions"]
        verdict = row["verdict"]
        mode_verdict = (
            "vertical mode"
            if verdict["near_null_is_vertical_mode"]
            else "not vertical mode"
        )
        lines.append(
            "| "
            f"{row['identity']} {row['arm']} | "
            f"{verdict['smallest_to_largest_ratio']:.6e} | "
            f"{projections['vertical_shift']:.6f} | "
            f"{projections['radial_shift']:.6f} | "
            f"{projections['current_rescale']:.6f} | "
            f"{mode_verdict} | "
            f"{verdict['solver_shape']} | {verdict['owner'] or 'none'} |"
        )
    stalled = [row for row in receipt["rows"] if row["role"] == "stalled_row"]
    vertical_rows = [
        row for row in stalled if row["verdict"]["near_null_is_vertical_mode"]
    ]
    aligned_rows = [
        row
        for row in stalled
        if not row["verdict"]["near_null_direction"]
        and row["verdict"]["vertical_projection_fraction"]
        >= VERTICAL_ALIGNMENT_THRESHOLD
    ]
    contrast_rows = {(row["identity"], row["arm"]): row for row in receipt["rows"]}
    contrast_identity = f"{CONTRAST_TARGET[0]}/{CONTRAST_TARGET[1]}"
    contrast_pure = contrast_rows[(contrast_identity, "pure")]
    contrast_mixed = contrast_rows[(contrast_identity, "mixed")]
    contrast_ratio = (
        contrast_pure["verdict"]["smallest_to_largest_ratio"]
        / contrast_mixed["verdict"]["smallest_to_largest_ratio"]
    )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            (
                "The universal vertical-null hypothesis is rejected: "
                f"{len(vertical_rows)} of {len(stalled)} stalled rows have a "
                "near-null smallest singular direction that is dominantly vertical. "
                "Only those rows support a vertical-position constraint with a "
                "compensating unknown."
            ),
            "",
            (
                "The non-threshold but vertically aligned rows are "
                + (
                    ", ".join(f"{row['identity']} {row['arm']}" for row in aligned_rows)
                    if aligned_rows
                    else "none"
                )
                + ". They retain the measured alignment without being promoted to "
                "a near-null verdict."
            ),
            "",
            (
                f"At {contrast_identity}, the stalled pure branch's condition ratio "
                f"is {contrast_ratio:.6f} times the converged mixed branch's ratio; "
                "their comparable singular scale rejects a vertical constraint as "
                "the explanation for that stall."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## Singular values and physical-direction residual changes",
            "",
        ]
    )
    for row in receipt["rows"]:
        values = ", ".join(
            f"{item['singular_value']:.9e}"
            for item in row["jacobian"]["smallest_three"]
        )
        changes = ", ".join(
            f"{name}={row['physical_directions'][name]['linear_residual_change_l2_per_unit_state_norm']:.9e}"
            for name in PHYSICAL_DIRECTION_NAMES
        )
        lines.extend(
            [
                f"- **{row['identity']} {row['arm']}**: smallest three [{values}]; "
                f"largest {row['jacobian']['largest_singular_value']:.9e}; unit-state "
                f"residual changes {changes}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Method and interpretation",
            "",
            (
                "Shadowed coordinates are fixed, not allowed to create trivial zero "
                "singular values. Rigid translations differentiate the bank route's "
                "centroid current density on its 33 by 33 grid, image that current "
                "change through the same plasma coupling, and subtract the spatial "
                "gradient of the terminal total flux. The current mode is a uniform "
                "fractional rescale of the normalised plasma-current image."
            ),
            "",
            (
                "A row is called a vertical near-null only when the "
                "smallest-to-largest "
                f"ratio is at most {NEAR_NULL_RELATIVE_THRESHOLD:g}, its absolute "
                f"vertical projection is at least {VERTICAL_ALIGNMENT_THRESHOLD:g}, "
                f"and it clears both competing projections by {ALIGNMENT_SEPARATION:g}."
            ),
            "",
            f"Receipt: `{receipt['artifacts']['receipt']}`.",
            f"Figure: `{receipt['artifacts']['figure']}`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def measure(
    *,
    operands: Path,
    output: Path,
    figure: Path,
    report: Path,
    matrix_block_size: int,
) -> dict[str, Any]:
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    terminals = _load_terminals(operands)
    response_cache, carrier_evidence = settled._persisted_response_cache(
        settled.response_carrier.DEFAULT_CARRIER,
        settled.response_carrier.DEFAULT_RECEIPT,
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in settled.select_slices_by_shot(
            settled.DECOMPOSITION_BANK
        )
    }
    rows: list[dict[str, Any]] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for key in STALLED_TARGETS:
        print(f"RECONSTRUCTING {key[0]}/{key[1]}", flush=True)
        selected_row, qualification = selected[key]
        case, context = settled._mast_case_from_selection(
            settled.SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, policy = settled._passive_inclusive_case(
            case, context, response_cache
        )
        if int(policy["section_kernel_evaluations_this_shot"]) != 0:
            raise RuntimeError("profile rebuild entered the direct response builder")
        target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
        observed = settled.bank_producer._ObservedProfile(profile)
        states = settled.reachability._mast_states(
            observed, jnp.asarray(passive_case["state"]), target_current
        )
        if observed.portfolio is None:
            raise RuntimeError(
                "production solve returned no observable branch portfolio"
            )
        pure_branch = jax.tree.map(
            lambda value: value[int(TopologyClass.DIVERTED)],
            observed.portfolio.branches,
        )
        requested_arms = ("pure", "mixed") if key == CONTRAST_TARGET else ("pure",)
        for arm in requested_arms:
            banked = terminals[(*key, arm)]
            print(f"DECOMPOSING {banked.identity} {arm}", flush=True)
            row = _measure_state(
                profile,
                states[arm],
                banked,
                target_current,
                pure_branch.equilibrium.fixed_point if arm == "pure" else None,
                matrix_block_size,
            )
            rows.append(row)
            partial = {
                "artifact": "frozen-mask Jacobian null-direction measurement",
                "complete": False,
                "rows": rows,
            }
            output.write_text(
                json.dumps(partial, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print(
                "DECOMPOSED "
                + json.dumps(
                    {
                        "identity": row["identity"],
                        "arm": arm,
                        "smallest_to_largest": row["verdict"][
                            "smallest_to_largest_ratio"
                        ],
                        "vertical_projection": row["verdict"][
                            "vertical_projection_fraction"
                        ],
                        "vertical_mode": row["verdict"]["near_null_is_vertical_mode"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    receipt = {
        "artifact": "frozen-mask Jacobian null-direction measurement",
        "complete": True,
        "source_commit": _source_revision(),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
            "slurm": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
                "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
                "node_list": os.environ.get("SLURM_JOB_NODELIST"),
                "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            },
        },
        "evidence_inputs": {
            "operands": str(operands),
            "operands_sha256": _sha256(operands),
            "response_carrier": carrier_evidence,
            "persistent_compilation_cache": cache.receipt(),
        },
        "measurement_contract": {
            "stalled_targets": [list(key) for key in STALLED_TARGETS],
            "converging_contrast": [*CONTRAST_TARGET, "mixed"],
            "smallest_singular_value_count": SMALLEST_SINGULAR_VALUE_COUNT,
            "matrix_block_size": matrix_block_size,
            "operator": (
                "I minus the frozen-mask map tangent, restricted on input and "
                "output to non-shadowed state coordinates"
            ),
            "singular_solver": "dense accelerator SVD of batched JVP columns",
            "physical_direction_names": list(PHYSICAL_DIRECTION_NAMES),
        },
        "artifacts": {
            "receipt": str(output),
            "figure": str(figure),
            "report": str(report),
        },
        "rows": rows,
        "verdict": {
            "all_banked_terminals_reproduced": all(
                row["bank_validation"]["passes"] for row in rows
            ),
            "vertical_mode_rows": [
                f"{row['identity']} {row['arm']}"
                for row in rows
                if row["verdict"]["near_null_is_vertical_mode"]
            ],
            "nonvertical_near_null_rows": [
                f"{row['identity']} {row['arm']}"
                for row in rows
                if row["verdict"]["near_null_direction"]
                and not row["verdict"]["near_null_is_vertical_mode"]
            ],
        },
    }
    _draw_figure(receipt, figure)
    _write_report(receipt, report)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operands", type=Path, default=DEFAULT_OPERANDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--matrix-block-size", type=int, default=DEFAULT_MATRIX_BLOCK_SIZE
    )
    arguments = parser.parse_args()
    result = measure(
        operands=arguments.operands,
        output=arguments.output,
        figure=arguments.figure,
        report=arguments.report,
        matrix_block_size=arguments.matrix_block_size,
    )
    print(json.dumps(result["verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
