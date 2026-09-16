"""Bank bounded centroid-row evidence on the analytic fixture."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import oracle_start_newton_probe as oracle_probe
from benchmarks import solovev_certificate as certificate
from nova.equilibrium import ForwardProfile, fixed_point
from nova.equilibrium.constraint import assemble_augmented_system
from nova.equilibrium.forward_operator import set_support_clip_mode, support_clip_mode
from nova.equilibrium.solve_request import default_forward_compilation_cache_root
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
)
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from scripts.analytic_oracle_fixtures import measure as oracle_fixture
from scripts.analytic_oracle_fixtures.centroid_row import (
    DEFAULT_FIELD_BOUND_T,
    DEFAULT_FIELD_SCALE_T,
    centroid_constraint_pair,
    exterior_field_identity,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-codex/centroid-row"
)
DEFAULT_FIGURE = ROOT / (
    "docs/figures/centroid-constrained-oracle-solve/weak-displaced-control.png"
)
ROWS = (
    ("weak-rotation-reactor-static", -110),
    ("moderate-rotation-conventional-static", -110),
    ("strong-rotation-compact-static", -110),
    ("diverted-single-null", -110),
)
DISPLACEMENT_M = np.asarray((0.020, 0.0), dtype=np.float64)


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value), dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _lane(required: str) -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    device = jax.devices()[0]
    if job_id is None:
        raise RuntimeError("the centroid receipt requires one scheduler allocation")
    if required == "cpu":
        if os.environ.get("SLURM_JOB_PARTITION") != "all_debug":
            raise RuntimeError("the compile probe requires the all_debug partition")
        if device.platform != "cpu" or os.environ.get("JAX_PLATFORMS") != "cpu":
            raise RuntimeError("the compile probe requires JAX_PLATFORMS=cpu")
    elif required == "h200":
        if os.environ.get("SLURM_JOB_PARTITION") != "betelgeuse":
            raise RuntimeError("the scientific receipt requires betelgeuse")
        if os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA":
            raise RuntimeError("the scientific receipt requires gpu_0003_grpA")
        if device.platform != "gpu" or "H200" not in device.device_kind:
            raise RuntimeError(
                f"the scientific receipt requires one H200, got {device}"
            )
        if os.environ.get("JAX_PLATFORMS") != "cuda,cpu":
            raise RuntimeError("the scientific receipt requires JAX_PLATFORMS=cuda,cpu")
    else:
        raise ValueError(f"unsupported lane requirement {required!r}")
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("TMPDIR must be /tmp inside the allocation")
    return {
        "job_id": int(job_id),
        "partition": os.environ["SLURM_JOB_PARTITION"],
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "allocated_cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        "memory_mb": int(os.environ.get("SLURM_MEM_PER_NODE", "0")),
        "device": device.device_kind,
        "platform": device.platform,
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "tmpdir": os.environ["TMPDIR"],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


def _revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _stablehlo_instruction_count(module: str) -> int:
    """Count result-defining StableHLO operations in one lowered module."""
    return sum(" = stablehlo." in line for line in module.splitlines())


def _translated_state(context: dict[str, Any]) -> np.ndarray:
    shifted = context["coordinates"] - DISPLACEMENT_M[None, :]
    return np.asarray(
        certificate._exact_state(context["case_name"], context["exact"], shifted),
        dtype=np.float64,
    )


def _context(case_name: str, requested_cells: int) -> dict[str, Any]:
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier_case, exact, requested_cells)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = certificate._exact_state(case_name, exact, coordinates)
    empty = oracle_fixture.forward_operator(source_case, machine)
    exact_moments, baseline, cache = oracle_fixture.cached_fixture_exterior(
        source_case, exact, machine, empty, analytic
    )
    operator = oracle_fixture.forward_operator(source_case, machine, baseline)
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=certificate.recovery.NEWTON_STEPS,
    )
    target_current, centroid, current_receipt = certificate._closed_form_current_target(
        case_name, source_case, operator, exact_moments
    )
    seed, requested_class, seed_receipt = certificate._production_seed(
        profile, case_name, target_current, centroid, current_receipt
    )
    pitch = float(np.sqrt(np.median(np.asarray(machine.area))))
    response = np.asarray(operator.prescribed_current_field.response)
    if response.shape[1] != 2 or not np.all(np.max(np.abs(response), axis=0) > 0.0):
        raise RuntimeError("the uniform exterior response failed its positive control")
    return {
        "case_name": case_name,
        "requested_cells": requested_cells,
        "source_case": source_case,
        "exact": exact,
        "machine": machine,
        "coordinates": coordinates,
        "analytic": np.asarray(analytic, dtype=np.float64),
        "profile": profile,
        "target_current": float(target_current),
        "centroid": np.asarray(centroid, dtype=np.float64),
        "pitch": pitch,
        "seed": np.asarray(seed, dtype=np.float64),
        "requested_class": int(requested_class),
        "seed_receipt": seed_receipt,
        "cache": cache,
        "baseline": np.asarray(baseline, dtype=np.float64),
        "response": response,
    }


def _solve(
    context: dict[str, Any], seed: np.ndarray, *, constrained: bool
) -> tuple[dict[str, Any], np.ndarray]:
    pair = centroid_constraint_pair(
        context["centroid"],
        pitch=context["pitch"],
    )
    request = certificate._certificate_solve_request(
        context["profile"],
        seed,
        context["target_current"],
        carrier_identity=":".join(
            (
                "centroid-fixture",
                context["case_name"],
                str(context["requested_cells"]),
                "constrained" if constrained else "control",
            )
        ),
    )
    if constrained:
        request = replace(request, constraint_pairs=(pair,))
    started = perf_counter()
    receipt = context["profile"].solve(request)
    equilibrium = receipt.equilibrium
    state = np.asarray(jax.block_until_ready(equilibrium.flux), dtype=np.float64)
    topology = oracle_probe._topology(context["profile"].operator, state)
    if constrained:
        record = equilibrium.constraints[0]
        observed = np.asarray(record.observed, dtype=np.float64)
        physical = np.asarray(record.physical_unknown, dtype=np.float64)
        scaled_residual = np.asarray(record.scaled_residual, dtype=np.float64)
        qualified = bool(np.asarray(record.qualified).all())
    else:
        observation = context["profile"].current_moment_observation(
            jnp.asarray(state), target_current=context["target_current"]
        )
        observed = np.asarray(
            (observation.centroid_r, observation.centroid_z), dtype=np.float64
        )
        physical = np.full(2, np.nan)
        scaled_residual = (observed - context["centroid"]) / context["pitch"]
        qualified = False
    return (
        {
            "constrained": constrained,
            "qualified": bool(np.asarray(receipt.qualified)),
            "row_qualified": qualified,
            "terminal_residual": float(equilibrium.fixed_point.residual),
            "centroid_target_m": context["centroid"],
            "centroid_observed_m": observed,
            "centroid_error_m": observed - context["centroid"],
            "centroid_error_pitches": float(
                np.linalg.norm(observed - context["centroid"]) / context["pitch"]
            ),
            "row_scaled_residual_sup": float(np.max(np.abs(scaled_residual))),
            "compensating_field_t": physical,
            "field_bound_t": DEFAULT_FIELD_BOUND_T,
            "field_scale_t": DEFAULT_FIELD_SCALE_T,
            "wall_seconds": perf_counter() - started,
            "topology": topology,
            "state_sha256_binary64": _digest(state),
        },
        state,
    )


def _compile_probe_program(
    context: dict[str, Any], *, constrained: bool
) -> tuple[Any, tuple[jax.Array, ...]]:
    """Return the production solve program and explicit array arguments."""
    profile = context["profile"]
    request = certificate._certificate_solve_request(
        profile,
        context["seed"],
        context["target_current"],
        carrier_identity=f"centroid-compile:{'bounded' if constrained else 'control'}",
    )
    options = request.policy.kernel_options()
    if not constrained:
        program = profile._accelerated_history_program(
            request.route,
            requested_class=None,
            target_current=request.target_current,
            **options,
        )
        external = profile.operator.external(
            request.current, request.prescribed_current
        )
        return program, (jnp.asarray(context["seed"]), external)

    pair = centroid_constraint_pair(
        context["centroid"],
        pitch=context["pitch"],
    )
    mapped = profile.flux_map(
        request.current,
        None,
        request.target_current,
        request.prescribed_current,
    )
    shadowed = profile.operator.flux_map_with_shadow(
        request.current,
        None,
        request.target_current,
        request.prescribed_current,
    )

    def shadow_mask(state):
        return profile.operator.residual_shadow_mask(state, None)

    def promoted_shadow_mask(state, previous):
        return profile.operator.residual_shadow_mask(
            state, None, previous_shadow=previous
        )

    system = assemble_augmented_system(
        profile,
        jnp.asarray(context["seed"]),
        (pair,),
        base_map=mapped,
        base_shadow_mask=shadow_mask,
        base_promoted_shadow_mask=promoted_shadow_mask,
        base_shadowed_map=shadowed,
        requested_class=None,
        target_current=jnp.asarray(context["target_current"]),
    )

    def solve(initial):
        return fixed_point.newton_krylov(
            system.map_fn,
            initial,
            shadow_mask_fn=system.shadow_mask_fn,
            promoted_shadow_mask_fn=system.promoted_shadow_mask_fn,
            shadowed_map_fn=system.shadowed_map_fn,
            row_jvp_observers=system.row_jvp_observers,
            **options,
        )

    return jax.jit(solve), (system.initial,)


def compile_probe_arm(output_root: Path, arm: str) -> dict[str, Any]:
    """Lower and compile one weak-fixture solve arm with durable checkpoints."""
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_forward_compilation_cache_root()
    )
    lane = _lane("cpu")
    previous_mode = support_clip_mode()
    set_support_clip_mode("exact")
    started = perf_counter()
    try:
        context = _context("weak-rotation-reactor-static", -110)
        constructed = perf_counter()
        program, arguments = _compile_probe_program(
            context, constrained=arm == "constrained"
        )
        lower_started = perf_counter()
        lowered = program.lower(*arguments)
        lower_seconds = perf_counter() - lower_started
        stablehlo = lowered.as_text(dialect="stablehlo")
        stablehlo_path = output_root / f"compile-{arm}.stablehlo"
        stablehlo_path.parent.mkdir(parents=True, exist_ok=True)
        stablehlo_path.write_text(stablehlo, encoding="utf-8")
        partial = {
            "schema": "nova.centroid-constraint-compile-probe",
            "arm": arm,
            "source_revision": _revision(),
            "lane": lane,
            "cache_directory": str(cache.directory),
            "construction_seconds": constructed - started,
            "lower_seconds": lower_seconds,
            "stablehlo_instruction_count": _stablehlo_instruction_count(stablehlo),
            "stablehlo_sha256": hashlib.sha256(stablehlo.encode()).hexdigest(),
            "stablehlo_path": str(stablehlo_path),
            "completed": False,
        }
        state_path = output_root / f"compile-{arm}.json"
        _write_json(state_path, partial)
        compile_started = perf_counter()
        lowered.compile()
        partial["backend_compile_seconds"] = perf_counter() - compile_started
        partial["total_seconds"] = perf_counter() - started
        partial["completed"] = True
        _write_json(state_path, partial)
        return partial
    finally:
        set_support_clip_mode(previous_mode)


def _row(case_name: str, requested_cells: int) -> tuple[dict[str, Any], dict[str, Any]]:
    context = _context(case_name, requested_cells)
    result, state = _solve(context, context["seed"], constrained=True)
    row = {
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": len(context["machine"].node),
        "characteristic_pitch_m": context["pitch"],
        "baseline_sha256_binary64": _digest(context["baseline"]),
        "fixture_exterior_cache": context["cache"],
        "response_column_sup_wb_per_t": np.max(np.abs(context["response"]), axis=0),
        "field_identity": exterior_field_identity(),
        "seed": context["seed_receipt"],
        "solve": result,
    }
    return row, {"context": context, "state": state}


def _bound_refusal() -> dict[str, Any]:
    pair = centroid_constraint_pair(np.asarray((1.0, 0.0)), pitch=0.1)
    trial = np.asarray((1.01 * DEFAULT_FIELD_BOUND_T / DEFAULT_FIELD_SCALE_T, 0.0))
    try:
        pair.unknown.require_within_bound(jnp.asarray(trial))
    except ValueError as error:
        return {
            "fired": True,
            "trial_normalized": trial,
            "trial_physical_t": trial * DEFAULT_FIELD_SCALE_T,
            "declared_bound_t": DEFAULT_FIELD_BOUND_T,
            "message": str(error),
        }
    raise RuntimeError("the exterior-field bound accepted an out-of-bound trial")


def _draw_control(
    context: dict[str, Any], constrained: np.ndarray, control: np.ndarray, path: Path
) -> dict[str, Any]:
    wall = np.asarray(context["machine"].wall_node, dtype=np.float64)
    radial, height, analytic_field = certificate._raster_field(
        context["coordinates"], context["analytic"], wall
    )
    _, _, constrained_field = certificate._raster_field(
        context["coordinates"], constrained, wall
    )
    _, _, control_field = certificate._raster_field(
        context["coordinates"], control, wall
    )
    levels = poloidal.contour_levels(analytic_field, count=12)
    analytic_topology = oracle_probe._topology(
        context["profile"].operator, context["analytic"]
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), constrained_layout=True)
    for axis, state, field, title, color in (
        (
            axes[0],
            constrained,
            constrained_field,
            "bounded centroid row",
            "#cc7722",
        ),
        (
            axes[1],
            control,
            control_field,
            "same displaced seed, no row",
            "#a23b72",
        ),
    ):
        topology = oracle_probe._topology(context["profile"].operator, state)
        poloidal.draw_flux_contours(
            axis, radial, height, analytic_field, levels, color="#3366cc"
        )
        poloidal.draw_flux_contours(axis, radial, height, field, levels, color=color)
        poloidal.draw_wall(axis, units=(wall,))
        poloidal.draw_nulls(
            axis,
            magnetic_axis=analytic_topology["axis_rz_m"],
            x_points=analytic_topology["x_point_rz_m"],
            style=DEFAULT_INK.variant(
                axis_marker="^", axis_color="#3366cc", xpoint_color="#3366cc"
            ),
            contain=(wall,),
        )
        poloidal.draw_nulls(
            axis,
            magnetic_axis=topology["axis_rz_m"],
            x_points=topology["x_point_rz_m"],
            style=DEFAULT_INK.variant(
                axis_marker="^", axis_color=color, xpoint_color=color
            ),
            contain=(wall,),
        )
        poloidal_axes(axis)
        axis.set_title(f"{title}\nanalytic blue / terminal coloured", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return {
        "filesystem_path": str(path),
        "project_absolute_src": (
            "/nova/figures/centroid-constrained-oracle-solve/weak-displaced-control.png"
        ),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def measure(output_root: Path, figure_path: Path) -> dict[str, Any]:
    configure_dtypes()
    configure_persistent_compilation_cache(default_forward_compilation_cache_root())
    lane = _lane("h200")
    previous_mode = support_clip_mode()
    set_support_clip_mode("exact")
    rows = []
    weak_artifacts = None
    try:
        for case_name, requested_cells in ROWS:
            row, artifacts = _row(case_name, requested_cells)
            rows.append(row)
            _write_json(
                output_root / f"{case_name}-cells-{abs(requested_cells)}.json", row
            )
            if case_name == "weak-rotation-reactor-static":
                weak_artifacts = artifacts
        if weak_artifacts is None:
            raise RuntimeError("the weak positive control was not measured")
        context = weak_artifacts["context"]
        displaced = _translated_state(context)
        positive, positive_state = _solve(context, displaced, constrained=True)
        negative, negative_state = _solve(context, displaced, constrained=False)
        controls = {
            "displacement_m": DISPLACEMENT_M,
            "displaced_seed_sha256_binary64": _digest(displaced),
            "positive": positive,
            "negative": negative,
        }
        _write_json(output_root / "weak-displaced-controls.json", controls)
        figure = _draw_control(context, positive_state, negative_state, figure_path)
    finally:
        set_support_clip_mode(previous_mode)
    verdict = {
        "all_rows_terminal_residual_at_or_below_1e_12": all(
            row["solve"]["terminal_residual"] <= 1.0e-12 for row in rows
        ),
        "all_rows_row_residual_at_or_below_1e_12": all(
            row["solve"]["row_scaled_residual_sup"] <= 1.0e-12 for row in rows
        ),
        "positive_centroid_within_tenth_pitch": (
            controls["positive"]["centroid_error_pitches"] <= 0.1
        ),
        "negative_remains_outside_tenth_pitch": (
            controls["negative"]["centroid_error_pitches"] > 0.1
        ),
    }
    verdict["passed"] = all(verdict.values())
    report = {
        "schema": "nova.centroid-constrained-analytic-fixture",
        "source_revision": _revision(),
        "support_clip_mode": "exact",
        "lane": lane,
        "field_identity": exterior_field_identity(),
        "bound_refusal": _bound_refusal(),
        "rows": rows,
        "controls": controls,
        "figure": figure,
        "verdict": verdict,
    }
    _write_json(output_root / "receipt.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--compile-probe-arm", choices=("unconstrained", "constrained"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.compile_probe_arm is not None:
        result = compile_probe_arm(arguments.output_root, arguments.compile_probe_arm)
        print(
            "CENTROID_COMPILE_PROBE "
            f"arm={result['arm']} lower={result['lower_seconds']:.6f}s "
            f"compile={result['backend_compile_seconds']:.6f}s "
            f"stablehlo_instructions={result['stablehlo_instruction_count']}",
            flush=True,
        )
        return
    report = measure(arguments.output_root, arguments.figure)
    print(
        f"CENTROID_CONSTRAINED_FIXTURE_EXIT={0 if report['verdict']['passed'] else 1}",
        flush=True,
    )
    if not report["verdict"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
