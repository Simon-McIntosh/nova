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
from nova.equilibrium.constraint import (
    ConstraintContext,
    ConstraintPair,
    assemble_augmented_system,
)
from nova.equilibrium.forward_operator import set_support_clip_mode, support_clip_mode
from nova.equilibrium.observation import MomentIntegralSupport
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
    DEFAULT_LEVEL_SCALE_WB,
    DEFAULT_STEP_LIMIT,
    centroid_constraint_pair,
    exterior_field_identity,
    fixture_constraint_pairs,
    reader_identity,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-local/centroid-step"
)
DEFAULT_FIGURE = ROOT / (
    "docs/figures/centroid-constrained-oracle-solve/weak-displaced-control.png"
)
DEFAULT_SOURCE_ROOT = DEFAULT_OUTPUT_ROOT
DEFAULT_GAUGE_ROOT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-local/centroid-gauge"
)
# A converged boundary level must sit inside a thousandth of the analytic span.
LEVEL_TOLERANCE_OF_SPAN = 1.0e-3
ROWS = (("weak-rotation-reactor-static", -110),)
DISPLACEMENT_M = np.asarray((0.020, 0.0), dtype=np.float64)

# One-map response of the fixture centroid to a uniform exterior field,
# measured with both signs at 1 mT and 10 mT with the exterior held fixed.
PROBE_RADIAL_M_PER_T = 1.627789
PROBE_VERTICAL_M_PER_T = 4.916664


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
    operator = oracle_fixture.forward_operator(
        source_case, machine, baseline, compensation=True
    )
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
    response = np.asarray(operator.prescribed_current_field.response, dtype=np.float64)
    if response.shape[1] != len(oracle_fixture.EXTERIOR_COMPENSATION_COLUMNS):
        raise RuntimeError("the compensation response lost a declared column")
    if not np.all(np.max(np.abs(response), axis=0) > 0.0):
        raise RuntimeError("the uniform exterior response failed its positive control")
    if not np.all(response[:, 2] == response[0, 2]):
        raise RuntimeError("the level column is not a uniform flux offset")
    # the level row is declared at the analytic magnetic axis, where both
    # solenoidal field columns read exactly zero, so the row reads the level
    axis_point = np.asarray(exact.magnetic_axis, dtype=np.float64).reshape(1, 2)
    level_target = float(
        np.asarray(certificate._exact_state(case_name, exact, axis_point))[0]
    )
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
        "axis_point": axis_point,
        "level_target_wb": level_target,
    }


def _certificate_request(context: dict[str, Any]) -> Any:
    return certificate._certificate_solve_request(
        context["profile"],
        context["seed"],
        context["target_current"],
        carrier_identity="centroid-fixture-certificate",
    )


def _certificate_pairs(
    context: dict[str, Any], *, level: bool
) -> tuple[ConstraintPair, ...]:
    """Return this fixture's pairs, with or without the flux-level row.

    Dropping the level pair leaves the centroid pair exactly as declared, so
    two programs built from these lists differ by the level row alone.
    """
    pairs = fixture_constraint_pairs(
        jnp.asarray(context["centroid"]),
        level_point=context["axis_point"],
        level_target=jnp.asarray((context["level_target_wb"],)),
        pitch=context["pitch"],
    )
    return pairs if level else (pairs[0],)


def _augmented_system(
    context: dict[str, Any], request: Any, pairs: tuple[ConstraintPair, ...]
):
    """Assemble the fixture's augmented system for one pair list."""
    profile = context["profile"]
    return assemble_augmented_system(
        profile,
        jnp.asarray(context["seed"]),
        pairs,
        base_map=profile.flux_map(
            request.current,
            None,
            request.target_current,
            request.prescribed_current,
        ),
        base_shadow_mask=lambda state: profile.operator.residual_shadow_mask(
            state, None
        ),
        base_promoted_shadow_mask=lambda state, previous: (
            profile.operator.residual_shadow_mask(state, None, previous_shadow=previous)
        ),
        base_shadowed_map=profile.operator.flux_map_with_shadow(
            request.current,
            None,
            request.target_current,
            request.prescribed_current,
        ),
        requested_class=None,
        target_current=jnp.asarray(context["target_current"]),
    )


def _solve(
    context: dict[str, Any], seed: np.ndarray, *, constrained: bool
) -> tuple[dict[str, Any], np.ndarray]:
    pairs = _certificate_pairs(context, level=True)
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
        request = replace(request, constraint_pairs=pairs)
    started = perf_counter()
    receipt = context["profile"].solve(request)
    equilibrium = receipt.equilibrium
    state = np.asarray(jax.block_until_ready(equilibrium.flux), dtype=np.float64)
    topology = oracle_probe._topology(context["profile"].operator, state)
    if constrained:
        centroid_record, level_record = equilibrium.constraints
        observed = np.asarray(centroid_record.observed, dtype=np.float64)
        amplitudes = np.concatenate(
            (
                np.asarray(centroid_record.physical_unknown, dtype=np.float64),
                np.asarray(level_record.physical_unknown, dtype=np.float64),
            )
        )
        field = amplitudes[:2]
        level_amplitude = float(amplitudes[2])
        centroid_residual = np.asarray(
            centroid_record.scaled_residual, dtype=np.float64
        )
        level_residual = float(
            np.max(np.abs(np.asarray(level_record.scaled_residual, dtype=np.float64)))
        )
        scaled_residual = np.concatenate(
            (
                centroid_residual,
                np.asarray(level_record.scaled_residual, dtype=np.float64),
            )
        )
        qualified = bool(
            np.asarray(centroid_record.qualified).all()
            and np.asarray(level_record.qualified).all()
        )
        bound_refusal = [
            bool(value)
            for record in (centroid_record, level_record)
            if record.bound_refusal is not None
            for value in np.asarray(record.bound_refusal).reshape(-1)
        ] or None
    else:
        observation = context["profile"].current_moment_observation(
            jnp.asarray(state),
            support=MomentIntegralSupport.ALL_DOMAIN,
            target_current=context["target_current"],
        )
        observed = np.asarray(
            (observation.centroid_r, observation.centroid_z), dtype=np.float64
        )
        field = np.full(2, np.nan)
        amplitudes = np.full(3, np.nan)
        level_amplitude = float("nan")
        level_residual = float("nan")
        scaled_residual = (observed - context["centroid"]) / context["pitch"]
        qualified = False
        bound_refusal = None
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
            "level_row_scaled_residual": level_residual,
            "bound_refusal": bound_refusal,
            "compensating_field_t": field,
            "compensating_field_t_abs_sup": float(np.max(np.abs(field))),
            "compensating_amplitudes": amplitudes,
            "level_amplitude_wb": level_amplitude,
            "level_target_wb": context["level_target_wb"],
            "level_error_wb": level_amplitude - context["level_target_wb"],
            "field_bound_t": DEFAULT_FIELD_BOUND_T,
            "field_scale_t": DEFAULT_FIELD_SCALE_T,
            "level_scale_wb": DEFAULT_LEVEL_SCALE_WB,
            "wall_seconds": perf_counter() - started,
            "topology": topology,
            "state_sha256_binary64": _digest(state),
        },
        state,
    )


def _compile_probe_program(
    context: dict[str, Any], *, constrained: bool, level: bool = True
) -> tuple[Any, tuple[jax.Array, ...]]:
    """Return the production solve program and explicit array arguments.

    ``level=False`` assembles the same bounded solve with the flux-level pair
    dropped, so the two programs differ by the level row alone and their build
    walls are a paired measurement of what the row adds to the compiled program.
    """
    profile = context["profile"]
    request = _certificate_request(context)
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

    pairs = _certificate_pairs(context, level=level)
    system = _augmented_system(context, request, pairs)

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


def _callback_counts(text: str) -> dict[str, int]:
    """Count the host-callback tokens a traced or compiled program carries."""
    return {
        "pure_callback": text.count("pure_callback"),
        "callback": text.count("callback"),
    }


def _wall_per_evaluation(function, argument, *, repeats: int = 50) -> dict[str, float]:
    """Return the blocked wall per evaluation of one callable.

    The call is blocked on every repeat, so the figure is the dispatch-plus-
    compute wall of one evaluation rather than the time to queue it.
    """
    jax.block_until_ready(function(argument))
    started = perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(function(argument))
    total = perf_counter() - started
    return {
        "repeats": repeats,
        "total_seconds": total,
        "seconds_per_evaluation": total / repeats,
    }


def reader_facts(output_root: Path) -> dict[str, Any]:
    """Record what the level row's point read is built from and what it costs.

    Three facts, each measured rather than asserted: the reader's static
    construction (host-solved per-node weights over the owning cell's own
    polygon, with source lines); the absence of a host callback in the traced
    row and in the compiled program; and the row's own cost -- the observed()
    wall per evaluation beside one augmented map evaluation, and the build wall
    of the program with the level row against the same program without it.
    """
    configure_dtypes()
    configure_persistent_compilation_cache(default_forward_compilation_cache_root())
    lane = _lane("h200")
    previous_mode = support_clip_mode()
    set_support_clip_mode("exact")
    try:
        context = _context("weak-rotation-reactor-static", -110)
        profile = context["profile"]
        pairs = _certificate_pairs(context, level=True)
        centroid_pair, level_pair = pairs
        binding = level_pair.binding
        flux = jnp.asarray(context["seed"], dtype=jnp.float64)

        def read(flux_state, pair):
            ctx = ConstraintContext(
                flux=flux_state,
                requested_class=None,
                target_current=jnp.asarray(context["target_current"]),
                shadow=None,
            )
            return pair.functional.residual(
                profile,
                ctx,
                jnp.zeros(1, dtype=jnp.float64),
                binding.payload,
                binding.target,
                binding.scale,
            )

        row_jaxpr = jax.make_jaxpr(lambda state: read(state, level_pair))(flux)
        centroid_jaxpr = jax.make_jaxpr(lambda state: read(state, centroid_pair))(flux)
        row_text = str(row_jaxpr)
        centroid_text = str(centroid_jaxpr)
        row_callbacks = _callback_counts(row_text)

        row_observed = jax.jit(lambda state: read(state, level_pair))
        observed_wall = _wall_per_evaluation(row_observed, flux)

        request = _certificate_request(context)
        systems = {
            level: _augmented_system(
                context, request, _certificate_pairs(context, level=level)
            )
            for level in (True, False)
        }
        steps = {level: jax.jit(system.map_fn) for level, system in systems.items()}
        initial = np.asarray(systems[True].initial, dtype=np.float64)
        initial_state = jnp.asarray(initial)
        newton_step_wall = _wall_per_evaluation(steps[True], initial_state)
        # the same map without the level row, measured on the same state so the
        # pair differs by the row alone rather than by the point it is asked at
        newton_step_wall_without = _wall_per_evaluation(steps[False], initial_state)

        builds = {}
        for level in (True, False):
            program, arguments = _compile_probe_program(
                context, constrained=True, level=level
            )
            lowered = program.lower(*arguments)
            text = lowered.as_text(dialect="stablehlo")
            started = perf_counter()
            lowered.compile()
            builds["with_level_row" if level else "without_level_row"] = {
                "compile_seconds": perf_counter() - started,
                "stablehlo_instruction_count": _stablehlo_instruction_count(text),
                "stablehlo_callback_counts": _callback_counts(text),
                "stablehlo_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }

        observed_seconds = observed_wall["seconds_per_evaluation"]
        step_seconds = newton_step_wall["seconds_per_evaluation"]
        level_cost = {
            "observed_seconds_per_evaluation": observed_seconds,
            "observed_repeats": observed_wall["repeats"],
            "one_newton_step_seconds": step_seconds,
            "one_newton_step_repeats": newton_step_wall["repeats"],
            "observed_to_step_ratio": observed_seconds / step_seconds,
            "one_newton_step_seconds_without_level_row": newton_step_wall_without[
                "seconds_per_evaluation"
            ],
            "one_newton_step_repeats_without_level_row": newton_step_wall_without[
                "repeats"
            ],
            "compile_seconds_with_level_row": builds["with_level_row"][
                "compile_seconds"
            ],
            "compile_seconds_without_level_row": builds["without_level_row"][
                "compile_seconds"
            ],
            "compile_seconds_delta": builds["with_level_row"]["compile_seconds"]
            - builds["without_level_row"]["compile_seconds"],
            "stablehlo_instruction_delta": builds["with_level_row"][
                "stablehlo_instruction_count"
            ]
            - builds["without_level_row"]["stablehlo_instruction_count"],
        }
        receipt = {
            "schema": "nova.centroid-flux-level-reader-facts",
            "source_revision": _revision(),
            "lane": lane,
            "support_clip_mode": "exact",
            "case": "weak-rotation-reactor-static",
            "requested_cells": -110,
            "carrier": type(profile.lattice).__name__,
            "reader_identity": reader_identity(),
            "row_jaxpr_callback_counts": row_callbacks,
            "row_jaxpr_has_host_callback": bool(
                row_callbacks["pure_callback"] or row_callbacks["callback"]
            ),
            "centroid_row_jaxpr_callback_counts": _callback_counts(centroid_text),
            "compiled_program": builds,
            "level_cost": level_cost,
            "level_amplitude_slot_wb": float(DEFAULT_LEVEL_SCALE_WB),
        }
        _write_json(output_root / "reader-facts.json", receipt)
        return receipt
    finally:
        set_support_clip_mode(previous_mode)


def measure_first_step(output_root: Path) -> dict[str, Any]:
    """Record the row's first undamped Newton step on the weak fixture.

    The unknown starts at zero, so the first map evaluation's unknown step is
    the undamped Newton step of the row: ``-scaled_residual`` on the normalised
    unknown, mapped to tesla by the field scale.  The receipt puts its size and
    sign beside what the measured one-map response predicts for the seed's own
    centroid offset.
    """
    configure_dtypes()
    configure_persistent_compilation_cache(default_forward_compilation_cache_root())
    lane = _lane("h200")
    previous_mode = support_clip_mode()
    set_support_clip_mode("exact")
    try:
        context = _context("weak-rotation-reactor-static", -110)
        observation = context["profile"].current_moment_observation(
            jnp.asarray(context["seed"]),
            support=MomentIntegralSupport.ALL_DOMAIN,
            target_current=context["target_current"],
        )
        observed = np.asarray(
            (observation.centroid_r, observation.centroid_z), dtype=np.float64
        )
        target = context["centroid"]
        pitch = context["pitch"]
        row_scaled_residual = (observed - target) / pitch
        normalised_step = -row_scaled_residual
        field_step = DEFAULT_FIELD_SCALE_T * normalised_step
        probe_field = -row_scaled_residual[0] * pitch / PROBE_RADIAL_M_PER_T
        receipt = {
            "schema": "nova.centroid-first-newton-step",
            "source_revision": _revision(),
            "support_clip_mode": "exact",
            "lane": lane,
            "case": "weak-rotation-reactor-static",
            "requested_cells": -110,
            "realised_cells": len(context["machine"].node),
            "characteristic_pitch_m": pitch,
            "analytic_centroid_m": target,
            "seed_centroid_observed_m": observed,
            "seed_centroid_offset_m": observed - target,
            "row_scaled_residual": row_scaled_residual,
            "first_step_normalized": normalised_step,
            "first_step_field_t": field_step,
            "first_step_pitches": normalised_step,
            "probe_radial_response_m_per_t": PROBE_RADIAL_M_PER_T,
            "probe_field_for_seed_offset_t": np.asarray(
                (probe_field, np.nan), dtype=np.float64
            ),
            "probe_field_for_seed_offset_pitches": np.asarray(
                (probe_field * PROBE_RADIAL_M_PER_T / pitch, np.nan),
                dtype=np.float64,
            ),
            "first_step_to_probe_ratio": float(field_step[0] / probe_field),
            "field_bound_t": DEFAULT_FIELD_BOUND_T,
            "step_limit": DEFAULT_STEP_LIMIT,
            "first_step_exceeds_declared_bound": bool(
                np.any(np.abs(field_step) > DEFAULT_FIELD_BOUND_T)
            ),
            "first_step_cap_binds": bool(
                np.any(np.abs(normalised_step) > DEFAULT_STEP_LIMIT)
            ),
        }
        _write_json(output_root / "weak-first-step.json", receipt)
        return receipt
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
        "response_column_sup_wb_per_t": np.max(
            np.abs(context["response"][:, :2]), axis=0
        ),
        "level_column_flux_per_wb": context["response"][0, 2],
        "field_identity": exterior_field_identity(),
        "seed": context["seed_receipt"],
        "solve": result,
    }
    return row, {"context": context, "state": state}


def _bound_refusal() -> dict[str, Any]:
    """Show the declared bound refusing a step, with its in-bound control."""
    pair = centroid_constraint_pair(np.asarray((1.0, 0.0)), pitch=0.1)
    bound_normalized = DEFAULT_FIELD_BOUND_T / DEFAULT_FIELD_SCALE_T
    # Both components beyond the bound, so the refusal must hold componentwise
    # and a per-component flag cannot pass the check by being in bound.
    over = np.full(2, 1.01 * bound_normalized)
    accepted_state = np.asarray((0.5 * bound_normalized, 0.0))

    try:
        pair.unknown.require_within_bound(jnp.asarray(over))
    except ValueError as error:
        raised = {"fired": True, "message": str(error)}
    else:
        raise RuntimeError("the exterior-field bound accepted an out-of-bound trial")

    refused_step, refused = pair.unknown.damped_step(
        jnp.asarray(over), jnp.asarray((0.0, 0.0))
    )
    accepted_step, accepted_refusal = pair.unknown.damped_step(
        jnp.asarray(accepted_state), jnp.asarray((0.5, 0.0))
    )
    if not bool(np.asarray(refused).all()):
        raise RuntimeError("damped step control did not refuse an out-of-bound state")
    if bool(np.asarray(accepted_refusal).any()):
        raise RuntimeError("damped step control refused an in-bound state")
    return {
        "raised_route": raised,
        "damped_route": {
            "fired": True,
            "state_normalized": over,
            "state_physical_t": over * DEFAULT_FIELD_SCALE_T,
            "declared_bound_t": DEFAULT_FIELD_BOUND_T,
            "step_normalized": np.asarray(refused_step, dtype=np.float64).tolist(),
            "refusal_recorded": np.asarray(refused).tolist(),
        },
        "in_bound_control": {
            "fired": False,
            "state_normalized": accepted_state,
            "physical_t": accepted_state * DEFAULT_FIELD_SCALE_T,
            "step_normalized": np.asarray(accepted_step, dtype=np.float64).tolist(),
            "refusal_recorded": np.asarray(accepted_refusal).tolist(),
        },
        "step_limit": DEFAULT_STEP_LIMIT,
    }


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


def _draw_state(
    context: dict[str, Any],
    state: np.ndarray,
    path: Path,
    *,
    title: str,
    color: str,
    project_src: str,
) -> dict[str, Any]:
    """Draw one terminal state beside the analytic field under the plotting rules."""
    wall = np.asarray(context["machine"].wall_node, dtype=np.float64)
    radial, height, analytic_field = certificate._raster_field(
        context["coordinates"], context["analytic"], wall
    )
    _, _, field = certificate._raster_field(context["coordinates"], state, wall)
    levels = poloidal.contour_levels(analytic_field, count=12)
    analytic_topology = oracle_probe._topology(
        context["profile"].operator, context["analytic"]
    )
    topology = oracle_probe._topology(context["profile"].operator, state)
    figure, axis = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
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
        "project_absolute_src": project_src,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _control_verdict(
    rows: list[dict[str, Any]], positive: dict[str, Any], negative: dict[str, Any]
) -> dict[str, Any]:
    verdict = {
        "all_rows_terminal_residual_at_or_below_1e_12": all(
            row["solve"]["terminal_residual"] <= 1.0e-12 for row in rows
        ),
        "all_rows_row_residual_at_or_below_1e_12": all(
            row["solve"]["row_scaled_residual_sup"] <= 1.0e-12 for row in rows
        ),
        "all_rows_field_within_declared_bound": all(
            row["solve"]["compensating_field_t_abs_sup"] <= DEFAULT_FIELD_BOUND_T
            for row in rows
        ),
        "all_rows_bound_never_engaged": all(
            row["solve"]["bound_refusal"] is None
            or not any(row["solve"]["bound_refusal"])
            for row in rows
        ),
        "positive_row_residual_at_or_below_1e_12": (
            positive["row_scaled_residual_sup"] <= 1.0e-12
        ),
        "positive_field_within_declared_bound": (
            positive["compensating_field_t_abs_sup"] <= DEFAULT_FIELD_BOUND_T
        ),
        "positive_centroid_within_tenth_pitch": (
            positive["centroid_error_pitches"] <= 0.1
        ),
        "positive_level_row_at_or_below_1e_12": (
            positive["level_row_scaled_residual"] <= 1.0e-12
        ),
        "positive_level_amplitude_is_finite": bool(
            np.isfinite(positive["level_amplitude_wb"])
        ),
        "negative_remains_outside_tenth_pitch": (
            negative["centroid_error_pitches"] > 0.1
        ),
    }
    verdict["passed"] = all(verdict.values())
    return verdict


def control_arm(output_root: Path, figure_path: Path, arm: str) -> dict[str, Any]:
    """Solve one displaced-seed control and draw its own panel.

    The positive arm imposes the centroid row on a seed displaced from the
    analytic centroid; the negative arm runs the same displaced seed with no
    row. Each arm is its own job because one H200 job does not hold the four
    programs of a full receipt inside an hour.
    """
    if arm not in ("positive", "negative"):
        raise ValueError(f"unknown control arm {arm!r}")
    configure_dtypes()
    configure_persistent_compilation_cache(default_forward_compilation_cache_root())
    lane = _lane("h200")
    constrained = arm == "positive"
    previous_mode = support_clip_mode()
    set_support_clip_mode("exact")
    try:
        context = _context("weak-rotation-reactor-static", -110)
        displaced = _translated_state(context)
        result, state = _solve(context, displaced, constrained=constrained)
        figure = _draw_state(
            context,
            state,
            figure_path,
            title=(
                "bounded centroid row on a displaced seed"
                if constrained
                else "same displaced seed, no row"
            ),
            color="#cc7722" if constrained else "#a23b72",
            project_src=(
                f"/nova/figures/centroid-constrained-oracle-solve/control-{arm}.png"
            ),
        )
    finally:
        set_support_clip_mode(previous_mode)
    control = {
        "schema": "nova.centroid-displaced-control",
        "arm": arm,
        "constrained": constrained,
        "source_revision": _revision(),
        "support_clip_mode": "exact",
        "lane": lane,
        "field_identity": exterior_field_identity(),
        "case": "weak-rotation-reactor-static",
        "requested_cells": -110,
        "realised_cells": len(context["machine"].node),
        "characteristic_pitch_m": context["pitch"],
        "centroid_target_m": context["centroid"],
        "displacement_m": DISPLACEMENT_M,
        "displaced_seed_sha256_binary64": _digest(displaced),
        "terminal_state_sha256_binary64": _digest(state),
        "solve": result,
        "figure": figure,
    }
    _write_json(output_root / f"control-{arm}.json", control)
    return control


def merge_controls(output_root: Path) -> dict[str, Any]:
    """Assemble the receipt from the split row and control jobs.

    A pure data merge: it reads the row receipt and both control receipts and
    applies the same verdict, so it runs on the login node without a lane.
    """
    row_path = output_root / "weak-rotation-reactor-static-cells-110.json"
    positive_path = output_root / "control-positive.json"
    negative_path = output_root / "control-negative.json"
    for path in (row_path, positive_path, negative_path):
        if not path.exists():
            raise FileNotFoundError(f"the split receipt is incomplete: missing {path}")
    row = json.loads(row_path.read_text())
    positive_receipt = json.loads(positive_path.read_text())
    negative_receipt = json.loads(negative_path.read_text())
    positive = positive_receipt["solve"]
    negative = negative_receipt["solve"]
    controls = {
        "displacement_m": np.asarray(
            positive_receipt["displacement_m"], dtype=np.float64
        ),
        "displaced_seed_sha256_binary64": positive_receipt[
            "displaced_seed_sha256_binary64"
        ],
        "positive": positive,
        "negative": negative,
    }
    if (
        negative_receipt["displaced_seed_sha256_binary64"]
        != controls["displaced_seed_sha256_binary64"]
    ):
        raise RuntimeError("the two control arms solved different displaced seeds")
    report = {
        "schema": "nova.centroid-constrained-analytic-fixture",
        "source_revision": _revision(),
        "support_clip_mode": "exact",
        "split_jobs": {
            "row": row_path.name,
            "positive": positive_path.name,
            "negative": negative_path.name,
            "arms": [
                {"arm": key, "lane": value["lane"], "job": value["lane"]["job_id"]}
                for key, value in (
                    ("positive", positive_receipt),
                    ("negative", negative_receipt),
                )
            ],
        },
        "field_identity": exterior_field_identity(),
        "bound_refusal": _bound_refusal(),
        "rows": [row],
        "controls": controls,
        "figures": {
            "positive": positive_receipt["figure"],
            "negative": negative_receipt["figure"],
        },
        "verdict": _control_verdict([row], positive, negative),
    }
    _write_json(output_root / "receipt.json", report)
    return report


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
    verdict = _control_verdict(rows, controls["positive"], controls["negative"])
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


def reread_gauge_readings(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = DEFAULT_GAUGE_ROOT,
) -> dict[str, Any]:
    """Re-read a banked control's flux offset as a gauge-free span difference.

    No solve runs and no field is recomputed: the banked receipt already
    carries the axis and boundary points, the level at each, and the
    compensating field, and the fixture's closed form supplies the authored
    levels those two points should carry.  The comparison the row's fixed-point
    clause is judged on is then the span between the magnetic axis and the
    boundary on both sides, which no additive constant in the flux can move.

    The compensator's own contribution to the span is tabled beside the
    difference, so the part of an offset that is the actuator's own field is
    separated from the part that is the equilibrium the row reached.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    arms: list[dict[str, Any]] = []
    for arm in ("positive", "negative"):
        path = source_root / f"control-{arm}.json"
        if not path.exists():
            raise FileNotFoundError(f"banked control receipt absent: {path}")
        stored = json.loads(path.read_text())
        case_name = str(stored["case"])
        topology = stored["solve"]["topology"]
        field = np.asarray(stored["solve"]["compensating_field_t"], dtype=np.float64)
        if field.shape != (2,) or not np.all(np.isfinite(field)):
            field = np.zeros(2, dtype=np.float64)
        _carrier, _source, exact = certificate._case(case_name)
        reading = oracle_fixture.gauge_free_flux_read(
            exact,
            np.asarray(topology["axis_rz_m"], dtype=np.float64),
            np.asarray(topology["boundary_rz_m"], dtype=np.float64),
            float(topology["axis_flux_wb"]),
            float(topology["boundary_flux_wb"]),
            field,
        )
        row_residual = float(stored["solve"]["row_scaled_residual_sup"])
        constrained = bool(stored["solve"]["constrained"])
        row_clause = row_residual <= oracle_probe.FIXED_POINT_TOLERANCE
        level_clause = (
            abs(reading["gauge_free_flux_offset_of_span"]) <= LEVEL_TOLERANCE_OF_SPAN
        )
        if not constrained:
            verdict = "not_applicable"
        elif row_clause and level_clause:
            verdict = "holds"
        else:
            verdict = "fails"
        arms.append(
            {
                "arm": arm,
                "case": case_name,
                "constrained": constrained,
                "source_receipt": str(path),
                "source_revision": stored.get("source_revision"),
                "row_scaled_residual_sup": row_residual,
                "fixed_point_clause_at_or_below_1e-12": row_clause,
                "level_clause_within_1e-3_of_span": level_clause,
                "fixed_point_verdict": verdict,
                "compensating_field_t": field.tolist(),
                **reading,
            }
        )
    report = {
        "schema": "nova.centroid-gauge-free-offset",
        "source_root": str(source_root),
        "level_tolerance_of_span": LEVEL_TOLERANCE_OF_SPAN,
        "row_tolerance": oracle_probe.FIXED_POINT_TOLERANCE,
        "gauge": (
            "solved and analytic flux levels are compared only as the span "
            "between the magnetic axis and the boundary; the compensator "
            "columns are anchored at the magnetic axis, so a level read against "
            "the authored zero is confounded by an additive constant"
        ),
        "arms": arms,
    }
    _write_json(output_root / "gauge-reread.json", report)
    (output_root / "gauge-reread.md").write_text(_gauge_markdown(report))
    return report


def _gauge_markdown(report: dict[str, Any]) -> str:
    lines = [
        "| arm | gauge-free offset (Wb) | offset / span | compensator span (Wb) | "
        "offset less compensator (Wb) | compensator at contact R (Wb) | "
        "row residual | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in report["arms"]:
        lines.append(
            "| {arm} | {gauge_free_flux_offset_wb:+.9e} | "
            "{gauge_free_flux_offset_of_span:+.9e} | "
            "{compensator_span_contribution_wb:+.9e} | "
            "{gauge_free_flux_offset_less_compensator_wb:+.9e} | "
            "{compensator_flux_at_contact_radius_wb:+.9e} | "
            "{row_scaled_residual_sup:+.9e} | {fixed_point_verdict} |".format(**arm)
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--compile-probe-arm", choices=("unconstrained", "constrained"))
    parser.add_argument("--first-step", action="store_true")
    parser.add_argument("--control-arm", choices=("positive", "negative"))
    parser.add_argument("--merge-controls", action="store_true")
    parser.add_argument("--reread-gauge", action="store_true")
    parser.add_argument(
        "--reader-facts",
        action="store_true",
        help="record the level row's reader construction, purity and cost",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="root holding the banked control receipts the re-read consumes",
    )
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
    if arguments.first_step:
        receipt = measure_first_step(arguments.output_root)
        radial_step = float(receipt["first_step_field_t"][0])
        probe_field = float(receipt["probe_field_for_seed_offset_t"][0])
        print(
            "CENTROID_FIRST_STEP "
            f"radial_field_step_t={radial_step:+.9e} "
            f"radial_step_pitches={float(receipt['first_step_pitches'][0]):+.9e} "
            f"probe_field_t={probe_field:+.9e} "
            f"probe_measured_ratio={radial_step / probe_field:+.6f} "
            f"exceeds_bound={bool(receipt['first_step_exceeds_declared_bound'])} "
            f"cap_binds={bool(receipt['first_step_cap_binds'])}",
            flush=True,
        )
        return
    if arguments.reader_facts:
        receipt = reader_facts(arguments.output_root)
        cost = receipt["level_cost"]
        print(
            "CENTROID_READER_FACTS "
            f"carrier={receipt['carrier']} "
            f"host_callback_in_row={receipt['row_jaxpr_has_host_callback']} "
            f"observed_us={cost['observed_seconds_per_evaluation'] * 1.0e6:.3f} "
            f"newton_step_us={cost['one_newton_step_seconds'] * 1.0e6:.3f} "
            f"newton_step_us_without_level_row="
            f"{cost['one_newton_step_seconds_without_level_row'] * 1.0e6:.3f} "
            f"compile_s_with_level_row={cost['compile_seconds_with_level_row']:.3f} "
            f"compile_s_without_level_row="
            f"{cost['compile_seconds_without_level_row']:.3f} "
            f"stablehlo_instruction_delta={cost['stablehlo_instruction_delta']}",
            flush=True,
        )
        return
        report = reread_gauge_readings(
            arguments.source_root,
            arguments.output_root,
        )
        for arm in report["arms"]:
            print(
                "CENTROID_GAUGE_REREAD "
                f"arm={arm['arm']} "
                f"gauge_free_offset_wb={arm['gauge_free_flux_offset_wb']:+.9e} "
                f"of_span={arm['gauge_free_flux_offset_of_span']:+.9e} "
                f"compensator_span_wb={arm['compensator_span_contribution_wb']:+.9e} "
                f"row_scaled_residual_sup={arm['row_scaled_residual_sup']:+.9e} "
                f"verdict={arm['fixed_point_verdict']}",
                flush=True,
            )
        return
    if arguments.merge_controls:
        report = merge_controls(arguments.output_root)
        print(
            f"CENTROID_MERGED_CONTROLS passed={report['verdict']['passed']}",
            flush=True,
        )
        return
    if arguments.control_arm is not None:
        arm_figure = arguments.figure.with_name(f"control-{arguments.control_arm}.png")
        control = control_arm(arguments.output_root, arm_figure, arguments.control_arm)
        solve = control["solve"]
        print(
            "CENTROID_CONTROL "
            f"arm={control['arm']} constrained={solve['constrained']} "
            f"centroid_error_pitches={solve['centroid_error_pitches']:+.9e} "
            f"row_scaled_residual_sup={solve['row_scaled_residual_sup']:+.9e} "
            f"terminal_residual={solve['terminal_residual']:+.9e} "
            f"field_t_abs_sup={solve['compensating_field_t_abs_sup']:+.9e} "
            f"bound_refusal={solve['bound_refusal']} "
            f"qualified={solve['qualified']} "
            f"wall_seconds={solve['wall_seconds']:.3f}",
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
