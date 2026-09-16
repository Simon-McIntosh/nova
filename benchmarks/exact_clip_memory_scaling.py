"""Measure and attribute exact-clip solve memory across cell-count rungs.

The production solve is compiled without execution.  Each rung is persisted as
soon as its executable and optimized HLO census are available, so a later
compiler failure does not erase an earlier measurement.  Predicate signatures
whose shape carries the realised cell count more than once are reported
separately: those are the all-cell pairwise constructions that cannot satisfy
the linear-memory contract of a per-cell topology or clipping operation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import clip_quadrature
from nova.equilibrium import forward_operator
from nova.equilibrium.forward_operator import set_support_clip_mode, support_clip_mode
from nova.jax.config import configure_dtypes


SCHEMA = "nova.exact-clip-memory-scaling"
JVP_RELATIVE_ERROR_BOUND = 1.0e-8


def scaling_exponent(
    first_bytes: int,
    second_bytes: int,
    first_cells: int,
    second_cells: int,
) -> float:
    """Return the power-law exponent between two positive measurements."""
    values = (first_bytes, second_bytes, first_cells, second_cells)
    if any(value <= 0 for value in values):
        raise ValueError("scaling measurements must be positive")
    return math.log(second_bytes / first_bytes) / math.log(second_cells / first_cells)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace one receipt only after its complete JSON is on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _cell_axes(shape: list[int], realised_cells: int) -> list[int]:
    """Return axes whose dimension is the rung's realised cell count."""
    return [axis for axis, size in enumerate(shape) if size == realised_cells]


def _pairwise_predicates(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return predicate results carrying two independent all-cell axes."""
    realised = int(row["realised_cells"])
    candidates = []
    for signature in row.get("qualifying_array_signatures", []):
        if signature["dtype"] != "pred":
            continue
        axes = _cell_axes(signature["shape"], realised)
        if len(axes) < 2:
            continue
        candidates.append(signature | {"realised_cell_axes": axes})
    return sorted(
        candidates,
        key=lambda item: (
            item["logical_size_in_bytes"],
            item["instruction_count"],
        ),
        reverse=True,
    )


def _scaling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit temporary and pairwise-predicate growth over completed rungs."""
    ordered = sorted(rows, key=lambda row: row["realised_cells"])
    adjacent = []
    for first, second in zip(ordered, ordered[1:], strict=False):
        adjacent.append(
            {
                "first_requested_cells": abs(first["requested_cells"]),
                "second_requested_cells": abs(second["requested_cells"]),
                "temporary_bytes_exponent": scaling_exponent(
                    first["memory_analysis"]["temp_size_in_bytes"],
                    second["memory_analysis"]["temp_size_in_bytes"],
                    first["realised_cells"],
                    second["realised_cells"],
                ),
            }
        )
    return {
        "independent_variable": "realised atomic cell count",
        "adjacent_rungs": adjacent,
    }


def _compile_memory_only(
    case_name: str,
    requested_cells: int,
    *,
    arm: str,
) -> dict[str, Any]:
    """Compile one solve and read memory without serializing multi-GiB HLO.

    Accelerator programs at the failing resolution can exceed protobuf's
    two-GiB serialization limit even though the compiled executable exposes a
    valid memory analysis. Keeping that receipt independent from the optional
    text census prevents the diagnostic export from erasing the gate it was
    meant to measure.
    """
    started = perf_counter()
    profile, seed, request, dimensions = certificate._certificate_compile_problem(
        case_name, requested_cells
    )
    mapped = profile.flux_map(
        request.current,
        target_current=request.target_current,
        prescribed_current=request.prescribed_current,
    )
    shadowed_map = profile.operator.flux_map_with_shadow(
        request.current,
        target_current=request.target_current,
        prescribed_current=request.prescribed_current,
    )

    def shadow_mask(state):
        return profile.operator.residual_shadow_mask(state)

    def promoted_shadow_mask(state, previous):
        return profile.operator.residual_shadow_mask(state, previous_shadow=previous)

    def solve_program(initial):
        result = certificate.recovery.fixed_point.newton_krylov(
            mapped,
            initial,
            shadow_mask_fn=shadow_mask,
            promoted_shadow_mask_fn=promoted_shadow_mask,
            shadowed_map_fn=shadowed_map,
            **request.policy.kernel_options(),
        )
        return result.state, result.residual, result.converged

    compiled = (
        jax.jit(solve_program).lower(jnp.asarray(seed, dtype=jnp.float64)).compile()
    )
    analysis = certificate._compiled_memory_fields(compiled.memory_analysis())
    return {
        "case": case_name,
        "arm": arm,
        "requested_cells": requested_cells,
        "realised_cells": dimensions["realised_cells"],
        "dimensions": dimensions,
        "compile_wall_seconds": perf_counter() - started,
        "memory_analysis": analysis,
        "largest_array_intermediates": [],
        "largest_predicate_intermediates": [],
        "qualifying_array_signatures": [],
        "executed": False,
        "method": (
            "jax.jit(solve_program).lower(seed).compile().memory_analysis(); "
            "optimized HLO serialization deliberately skipped"
        ),
    }


def measure(
    output: Path,
    compiler_root: Path,
    requested_cells: list[int],
    *,
    part_root: Path | None = None,
    capture_hlo: bool = True,
) -> dict:
    """Compile exact-clip rungs and persist each result before continuing.

    Every rung receives an immutable, independently readable part receipt in
    addition to the cumulative receipt.  A later compilation failure therefore
    cannot obscure which earlier executable and memory analysis actually
    landed.
    """
    configure_dtypes()
    if not hasattr(certificate.observation, "_UNIT_NODE"):
        certificate.observation._UNIT_NODE = clip_quadrature._UNIT_NODE
    original_mode = support_clip_mode()
    rows: list[dict[str, Any]] = []
    source_revision = certificate._source_revision()
    lane = certificate._lane()
    if part_root is None:
        part_root = output.parent / f"{output.stem}-parts"
    try:
        set_support_clip_mode("exact")
        for requested in requested_cells:
            if capture_hlo:
                row = certificate._compile_solve_memory(
                    "weak-rotation-reactor-static",
                    -abs(requested),
                    arm=f"exact-{abs(requested)}",
                    compiler_artifact_root=compiler_root,
                )
            else:
                row = _compile_memory_only(
                    "weak-rotation-reactor-static",
                    -abs(requested),
                    arm=f"exact-{abs(requested)}",
                )
            row["pairwise_predicate_candidates"] = _pairwise_predicates(row)
            _atomic_json(
                part_root / f"requested-{abs(requested)}.json",
                {
                    "schema": SCHEMA,
                    "source_revision": source_revision,
                    "lane": lane,
                    "row": row,
                },
            )
            rows.append(row)
            receipt = {
                "schema": SCHEMA,
                "source_revision": source_revision,
                "lane": lane,
                "rows": rows,
                "scaling": _scaling(rows),
                "completed": len(rows) == len(requested_cells),
            }
            _atomic_json(output, receipt)
            temporary_gib = row["memory_analysis"]["temp_size_in_bytes"] / 2**30
            print(
                "EXACT_CLIP_MEMORY_RUNG "
                f"requested={abs(requested)} realised={row['realised_cells']} "
                f"temporary_gib={temporary_gib:.6f} "
                f"pairwise_predicates={len(row['pairwise_predicate_candidates'])}",
                flush=True,
            )
    finally:
        set_support_clip_mode(original_mode)
    return json.loads(output.read_text(encoding="utf-8"))


def solve_and_measure(
    output: Path,
    figure_root: Path,
    part_root: Path,
    requested_cells: int,
) -> dict[str, Any]:
    """Run one exact-clip production solve and persist its allocation peak.

    The certificate driver owns the production ``profile.solve`` call and its
    line-contour panel.  This wrapper only redirects those durable outputs and
    reads the accelerator allocator after the terminal state is ready.  A
    positive byte counter proves the allocator instrument saw the live solve;
    an empty or uniformly zero report is rejected rather than presented as a
    low-memory measurement.
    """
    configure_dtypes()
    if not hasattr(certificate.observation, "_UNIT_NODE"):
        certificate.observation._UNIT_NODE = clip_quadrature._UNIT_NODE
    device = jax.devices()[0]
    original_mode = support_clip_mode()
    original_figure_root = certificate.FIGURE_ROOT
    original_part_root = certificate.PART_ROOT
    certificate.FIGURE_ROOT = figure_root
    certificate.PART_ROOT = part_root
    requested_cells = -abs(requested_cells)
    try:
        set_support_clip_mode("exact")
        certificate._measure(
            "weak-rotation-reactor-static",
            requested_cells,
        )
        row_path = certificate._part_path(
            "weak-rotation-reactor-static", requested_cells
        )
        figure_path = certificate._figure_path(
            "weak-rotation-reactor-static", requested_cells
        )
        row = json.loads(row_path.read_text(encoding="utf-8"))
        raw_stats = device.memory_stats() or {}
    finally:
        certificate.FIGURE_ROOT = original_figure_root
        certificate.PART_ROOT = original_part_root
        set_support_clip_mode(original_mode)

    statistics = {
        str(name): int(value) if isinstance(value, int | float) else str(value)
        for name, value in raw_stats.items()
    }
    byte_counters = {
        name: value
        for name, value in statistics.items()
        if "bytes" in name and isinstance(value, int)
    }
    observed_bytes = max(byte_counters.values(), default=0)
    if observed_bytes <= 0:
        raise RuntimeError("accelerator allocation counters did not see the solve")
    peak_bytes = statistics.get("peak_bytes_in_use")
    if not isinstance(peak_bytes, int) or peak_bytes <= 0:
        raise RuntimeError("accelerator allocator did not report a positive peak")

    receipt = {
        "schema": "nova.exact-clip-production-solve",
        "source_revision": certificate._source_revision(),
        "lane": certificate._lane(),
        "requested_cells": requested_cells,
        "row_receipt": str(row_path),
        "figure": {
            "filesystem_path": str(figure_path),
            "project_absolute_src": row["figure"]["project_absolute_src"],
            "sha256": row["figure"]["sha256"],
        },
        "allocator": {
            "device": str(device),
            "statistics": statistics,
            "instrument_check": {
                "byte_counter_count": len(byte_counters),
                "largest_observed_byte_counter": observed_bytes,
                "positive_peak_bytes_in_use": True,
            },
            "peak_bytes_in_use": peak_bytes,
            "peak_gib": peak_bytes / 2**30,
        },
        "row": row,
    }
    _atomic_json(output, receipt)
    print(
        "EXACT_CLIP_SOLVE "
        f"requested={abs(requested_cells)} realised={row['realised_cells']} "
        f"peak_gib={peak_bytes / 2**30:.6f} "
        f"residual={row['solver']['terminal_fixed_point_residual']}",
        flush=True,
    )
    return receipt


def _terminal_state(path: Path) -> np.ndarray:
    """Read one explicitly named terminal state and prove it is populated."""
    if path.suffix == ".npz":
        with np.load(path) as stored:
            state = np.asarray(stored["flux"], dtype=np.float64)
    else:
        stored = json.loads(path.read_text(encoding="utf-8"))
        state = np.asarray(stored["render_data"]["terminal_flux_wb"], dtype=np.float64)
    if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
        raise RuntimeError(f"terminal state is empty or nonfinite: {path}")
    return state


def _normalise_direction(direction: jax.Array) -> jax.Array:
    """Return a finite max-unit direction or refuse an empty instrument."""
    direction = jnp.asarray(direction, dtype=jnp.float64)
    scale = jnp.max(jnp.abs(direction))
    if not np.isfinite(float(scale)) or float(scale) == 0.0:
        raise RuntimeError("JVP direction is empty or nonfinite")
    return direction / scale


def _fixed_clip_map(profile, request, state):
    """Return the smooth map and clip primal selected by one terminal read."""
    operator = profile.operator
    masks, topology, _sample_psi_norm, support = operator._support_partition(state)
    participation = jnp.asarray(support.vertex_count) >= 3
    shadow = operator.residual_shadow_mask(state)
    external = operator.external(request.current, request.prescribed_current)

    def partitioned(candidate):
        physical = jnp.asarray(candidate)[: operator.physical_node_number]
        grid_flux, _wall_flux = operator.topology.split_flux_map(physical)
        psi_norm = operator.topology.normalize(
            topology.axis_flux,
            topology.boundary_flux,
            grid_flux,
        )
        candidate_masks = masks._replace(psi_norm=psi_norm)
        sample_flux = operator.sample_node_flux(candidate)
        sample_psi_norm = (sample_flux - topology.axis_flux) / topology.flux_span
        candidate_support = operator._profile_support(
            candidate_masks,
            topology,
            physical,
            sample_psi_norm,
            fixed_participation=participation,
        )
        partition = (
            candidate_masks,
            topology,
            sample_psi_norm,
            candidate_support,
        )
        return candidate_support, partition

    def mapped(candidate):
        _candidate_support, partition = partitioned(candidate)
        moments = operator._partitioned_current_moments(partition)
        if request.target_current is not None:
            amplitude = operator.current_normalisation_amplitude(
                request.target_current,
                jnp.sum(moments.cell_current),
            )
            moments = operator.scaled_current_moments(moments, amplitude)
        image = external + operator.current_moment_image(moments)
        return operator._exclude_shadow_residual(
            candidate,
            image,
            shadow=shadow,
        )

    def clip_primal(candidate):
        candidate_support, _partition = partitioned(candidate)
        return candidate_support.support_vertices

    return (
        mapped,
        clip_primal,
        {
            "fixed_participating_cells": int(jnp.sum(participation)),
            "fixed_shadowed_carriers": int(jnp.sum(shadow)),
        },
    )


def _clip_branch_levels(profile, topology):
    """Return continuous values whose signs select exact-clip packing branches."""
    operator = profile.operator

    def levels(candidate):
        physical = jnp.asarray(candidate)[: operator.physical_node_number]
        grid_flux, _wall_flux = operator.topology.split_flux_map(physical)
        psi_norm = operator.topology.normalize(
            topology.axis_flux,
            topology.boundary_flux,
            grid_flux,
        )
        sample_flux = operator.sample_node_flux(candidate)
        sample_psi_norm = (sample_flux - topology.axis_flux) / topology.flux_span
        masks = forward_operator.DomainMasks(
            label=jnp.zeros_like(psi_norm, dtype=jnp.int32),
            psi_norm=psi_norm,
        )
        coefficient = operator.support_flux_coefficients(
            masks.psi_norm,
            sample_psi_norm,
        )
        inside_coefficient = (-coefficient).at[:, 0].add(1.0)
        coordinate = jnp.asarray(operator.grid.coordinate, dtype=psi_norm.dtype)
        surface_value = psi_norm[None, :]
        surface = forward_operator.fit_split_spline(
            coordinate[None, :, 0],
            coordinate[None, :, 1],
            surface_value,
            surface_value - 1.0,
            order=6,
            regularization=1.0e-14,
        )
        evaluator = forward_operator._ExactClipLevel(
            surface,
            inside_coefficient,
            operator._support_curve_centre,
            operator._support_curve_scale,
        )
        atomic_mesh = operator.moment_geometry.atomic_mesh
        vertices = jnp.asarray(atomic_mesh.node_coordinates)[
            jnp.asarray(atomic_mesh.cell_nodes)
        ]
        shared_level = operator.polarity * (
            operator.shared_node_flux(physical) - topology.boundary_flux
        )
        return jnp.concatenate((shared_level.ravel(), evaluator(vertices).ravel()))

    return jax.jit(levels)


def measure_jvp_accuracy(
    output: Path,
    part_root: Path,
    state_paths: dict[int, Path],
    requested_cells: tuple[int, ...] = (110, 300),
) -> dict[str, Any]:
    """Check implicit exact-clip tangents against central primal differences."""
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("exact-clip JVP validation requires binary64")
    if not hasattr(certificate.observation, "_UNIT_NODE"):
        certificate.observation._UNIT_NODE = clip_quadrature._UNIT_NODE
    original_mode = support_clip_mode()
    rows: list[dict[str, Any]] = []
    try:
        set_support_clip_mode("exact")
        for requested in requested_cells:
            state_path = state_paths[requested]
            state = jnp.asarray(_terminal_state(state_path), dtype=jnp.float64)
            profile, _seed, request, dimensions = (
                certificate._certificate_compile_problem(
                    "weak-rotation-reactor-static", -requested
                )
            )
            if state.shape != (dimensions["solve_state_size"],):
                raise RuntimeError(
                    f"terminal state size {state.size} does not match "
                    f"compiled size {dimensions['solve_state_size']}"
                )
            mapped, clip_primal, branch = _fixed_clip_map(profile, request, state)
            _masks, topology, _sample_psi_norm, _support = (
                profile.operator._support_partition(state)
            )
            branch_levels = _clip_branch_levels(profile, topology)
            base_levels = jax.block_until_ready(branch_levels(state))
            base_sign = base_levels > 0.0
            branch |= {
                "packing_level_count": int(base_levels.size),
                "packing_exact_zero_count": int(jnp.sum(base_levels == 0.0)),
                "packing_minimum_absolute_level": float(jnp.min(jnp.abs(base_levels))),
            }
            compiled_map = jax.jit(mapped)
            compiled_clip_primal = jax.jit(clip_primal)
            mapped_state, tangent_action = jax.linearize(compiled_map, state)
            residual = mapped_state - state

            def linear_action(vector):
                return vector - tangent_action(vector)

            newton_direction, gmres_info = jax.scipy.sparse.linalg.gmres(
                linear_action,
                residual,
                tol=1.0e-12,
                maxiter=request.policy.gmres_iterations,
                restart=request.policy.gmres_iterations,
                solve_method="batched",
            )
            newton_direction = _normalise_direction(
                jax.block_until_ready(newton_direction)
            )
            generator = np.random.default_rng(20260915 + requested)
            directions = [("newton", newton_direction)]
            for index in range(3):
                directions.append(
                    (
                        f"random-{index + 1}",
                        _normalise_direction(
                            jnp.asarray(generator.standard_normal(state.shape))
                        ),
                    )
                )

            state_scale = float(jnp.maximum(jnp.max(jnp.abs(state)), 1.0))
            difference_step = float(np.cbrt(np.finfo(np.float64).eps) * state_scale)
            comparisons = []
            for name, direction in directions:
                _primal, map_tangent = jax.jvp(
                    compiled_map,
                    (state,),
                    (direction,),
                )
                _clip, clip_tangent = jax.jvp(
                    compiled_clip_primal,
                    (state,),
                    (direction,),
                )
                _level_primal, level_tangent = jax.jvp(
                    branch_levels,
                    (state,),
                    (direction,),
                )
                upper = compiled_map(state + difference_step * direction)
                lower = compiled_map(state - difference_step * direction)
                map_central = (upper - lower) / (2.0 * difference_step)
                clip_upper = compiled_clip_primal(state + difference_step * direction)
                clip_lower = compiled_clip_primal(state - difference_step * direction)
                clip_central = (clip_upper - clip_lower) / (2.0 * difference_step)
                upper_levels = branch_levels(state + difference_step * direction)
                lower_levels = branch_levels(state - difference_step * direction)
                level_central = (upper_levels - lower_levels) / (2.0 * difference_step)
                map_tangent, map_central, clip_tangent, clip_central = (
                    jax.block_until_ready(
                        (map_tangent, map_central, clip_tangent, clip_central)
                    )
                )
                level_tangent, level_central = jax.block_until_ready(
                    (level_tangent, level_central)
                )
                clip_error = clip_tangent - clip_central
                clip_tangent_norm = float(jnp.linalg.norm(clip_tangent))
                clip_central_norm = float(jnp.linalg.norm(clip_central))
                clip_error_norm = float(jnp.linalg.norm(clip_error))
                clip_scale = max(
                    clip_tangent_norm,
                    clip_central_norm,
                    np.finfo(np.float64).tiny,
                )
                map_error_norm = float(jnp.linalg.norm(map_tangent - map_central))
                map_tangent_norm = float(jnp.linalg.norm(map_tangent))
                map_central_norm = float(jnp.linalg.norm(map_central))
                level_error_norm = float(jnp.linalg.norm(level_tangent - level_central))
                level_tangent_norm = float(jnp.linalg.norm(level_tangent))
                level_central_norm = float(jnp.linalg.norm(level_central))
                step_probes = []
                for multiplier in (0.25, 1.0, 4.0, 16.0, 64.0, 256.0):
                    probe_step = difference_step * multiplier
                    probe_upper = compiled_map(state + probe_step * direction)
                    probe_lower = compiled_map(state - probe_step * direction)
                    probe_central = (probe_upper - probe_lower) / (2.0 * probe_step)
                    upper_levels = branch_levels(state + probe_step * direction)
                    lower_levels = branch_levels(state - probe_step * direction)
                    probe_central, upper_levels, lower_levels = jax.block_until_ready(
                        (probe_central, upper_levels, lower_levels)
                    )
                    probe_error = float(jnp.linalg.norm(map_tangent - probe_central))
                    probe_norm = float(jnp.linalg.norm(probe_central))
                    step_probes.append(
                        {
                            "multiplier": multiplier,
                            "step": probe_step,
                            "relative_error": probe_error
                            / max(
                                map_tangent_norm,
                                probe_norm,
                                np.finfo(np.float64).tiny,
                            ),
                            "upper_sign_changes": int(
                                jnp.sum((upper_levels > 0.0) != base_sign)
                            ),
                            "lower_sign_changes": int(
                                jnp.sum((lower_levels > 0.0) != base_sign)
                            ),
                            "opposed_probe_signs": int(
                                jnp.sum((upper_levels > 0.0) != (lower_levels > 0.0))
                            ),
                        }
                    )
                comparisons.append(
                    {
                        "direction": name,
                        "finite_difference_rule": (
                            "cuberoot(binary64 epsilon) times terminal infinity scale "
                            "for a max-unit direction"
                        ),
                        "finite_difference_step": difference_step,
                        "clip_tangent_l2": clip_tangent_norm,
                        "clip_central_difference_l2": clip_central_norm,
                        "clip_error_l2": clip_error_norm,
                        "relative_error": clip_error_norm / clip_scale,
                        "clip_error_linf": float(jnp.max(jnp.abs(clip_error))),
                        "production_map_relative_error": map_error_norm
                        / max(
                            map_tangent_norm,
                            map_central_norm,
                            np.finfo(np.float64).tiny,
                        ),
                        "packing_level_jvp_relative_error": level_error_norm
                        / max(
                            level_tangent_norm,
                            level_central_norm,
                            np.finfo(np.float64).tiny,
                        ),
                        "step_probes": step_probes,
                    }
                )
            row = {
                "requested_cells": requested,
                "realised_cells": dimensions["realised_cells"],
                "terminal_state": str(state_path),
                "terminal_state_size": int(state.size),
                "fixed_point_residual_l2": float(jnp.linalg.norm(residual)),
                "gmres_info": int(gmres_info),
                "branch": branch,
                "relative_error_bound": JVP_RELATIVE_ERROR_BOUND,
                "directions": comparisons,
                "passed": all(
                    item["relative_error"] <= JVP_RELATIVE_ERROR_BOUND
                    for item in comparisons
                ),
            }
            _atomic_json(
                part_root / f"requested-{requested}.json",
                {
                    "schema": "nova.exact-clip-jvp-accuracy",
                    "source_revision": certificate._source_revision(),
                    "lane": certificate._lane(),
                    "row": row,
                },
            )
            rows.append(row)
            receipt = {
                "schema": "nova.exact-clip-jvp-accuracy",
                "source_revision": certificate._source_revision(),
                "lane": certificate._lane(),
                "rows": rows,
                "completed": len(rows) == 2,
                "passed": len(rows) == 2 and all(item["passed"] for item in rows),
            }
            _atomic_json(output, receipt)
            print(
                "EXACT_CLIP_JVP "
                f"requested={requested} "
                + " ".join(
                    f"{item['direction']}={item['relative_error']:.3e}"
                    for item in comparisons
                ),
                flush=True,
            )
    finally:
        set_support_clip_mode(original_mode)
    if not receipt["passed"]:
        raise RuntimeError("exact-clip JVP central-difference gate failed")
    return receipt


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler-root", type=Path)
    parser.add_argument("--part-root", type=Path)
    parser.add_argument("--figure-root", type=Path)
    parser.add_argument("--execute-solve", action="store_true")
    parser.add_argument("--check-jvp", action="store_true")
    parser.add_argument("--state-110", type=Path)
    parser.add_argument("--state-300", type=Path)
    parser.add_argument("--skip-hlo", action="store_true")
    parser.add_argument("--cells", type=int, nargs="+", default=[110, 300, 500])
    return parser.parse_args()


def main() -> None:
    arguments = _parse()
    if arguments.execute_solve:
        if len(arguments.cells) != 1:
            raise ValueError("the production solve accepts exactly one cell count")
        solve_and_measure(
            arguments.output,
            arguments.figure_root or arguments.output.parent / "solve-panels",
            arguments.part_root or arguments.output.parent / "solve-parts",
            arguments.cells[0],
        )
        return
    if arguments.check_jvp:
        if arguments.state_110 is None or arguments.state_300 is None:
            raise ValueError("--state-110 and --state-300 are required for JVP checks")
        requested = tuple(cell for cell in arguments.cells if cell in (110, 300))
        if not requested:
            raise ValueError("JVP checks require cell count 110 or 300")
        measure_jvp_accuracy(
            arguments.output,
            arguments.part_root or arguments.output.parent / "jvp-parts",
            {110: arguments.state_110, 300: arguments.state_300},
            requested,
        )
        return
    if arguments.compiler_root is None:
        raise ValueError("--compiler-root is required for memory analysis")
    receipt = measure(
        arguments.output,
        arguments.compiler_root,
        arguments.cells,
        part_root=arguments.part_root,
        capture_hlo=not arguments.skip_hlo,
    )
    print(
        json.dumps(
            {
                "completed": receipt["completed"],
                "rungs": len(receipt["rows"]),
                "scaling": receipt["scaling"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
