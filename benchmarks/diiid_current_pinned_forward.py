"""Measure whether an exact plasma-current constraint changes the forward map.

The experiment keeps Nova's absolute-source production policy untouched and
wraps the existing free-boundary map in closed-form elimination of one common
profile amplitude.  The recorded plasma current is a prescribed input of the
same class as the coil currents; it is neither fitted nor inferred inside the
equilibrium solve.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from benchmarks.diiid_boundary_current_recovery import (
    CHECKPOINT_NAME as RECOVERY_CHECKPOINT_NAME,
    OMITTED_COILS,
    POLARITY_RECEIPT,
    RECEIPT_NAME as RECOVERY_RECEIPT_NAME,
    DEFAULT_OUTPUT as RECOVERY_OUTPUT,
)
from benchmarks.diiid_diverted_root_full_currents import (
    POLARITY_AFFECTED_SHOT_COUNT,
    FrameInput,
    _omitted_vertices,
    append_recovered_conductors,
    current_arms,
)
from benchmarks.diiid_forward_gs_match import (
    DEFAULT_DATA,
    _CURRENT_COLUMNS,
    _GEOMETRY_COLUMNS,
    _LABEL_COLUMNS,
    _read,
    build_profile,
)
from nova.equilibrium import fixed_point
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes


DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/current-pinned")
PREREGISTRATION_NAME = "current_pinned_forward_preregistration.json"
CHECKPOINT_NAME = "current_pinned_forward_frames.jsonl"
RECEIPT_NAME = "current_pinned_forward_receipt.json"
FIGURE_NAME = "current_pinned_forward.png"
ARM_NAMES = ("unpinned", "pinned_eliminated")
LAMBDA_BAND = (1.0e-6, 1.0e6)
RELATIVE_RESIDUAL_CRITERION = 1.0e-6
CURRENT_CONSTRAINT_CRITERION = 1.0e-10
UNPINNED_PLATEAU_CONTROL = 3.491124178554655e-2
SHIPPED_ONLY_PLATEAU_CONTROL = 0.1233879
UNPINNED_CONTROL_ABSOLUTE_TOLERANCE = 2.0e-7
HOST_OUTER_ITERATIONS = 100
HOST_INNER_ITERATIONS = 40
POWER_ITERATIONS = 16
POWER_RELATIVE_STEP = 1.0e-6
PSEUDO_WALL_EXPANSION = 0.02
PLASMA_CURRENT_COLUMNS = (
    "magnetics_plasma_current",
    "magnetics_plasma_current_times",
)
REPRESENTATIVE_CURRENT_FLOOR_A = 200_000.0
COHORT_PREFLIGHT_RELATIVE_TOLERANCE = 2.0e-6
LOW_CURRENT_CONTROL = ("d3d_shot_00000c4a7b.parquet", 0)
REPRESENTATIVE_COHORT = (
    {
        "shot": "d3d_shot_00000c4a7b.parquet",
        "frame": 102,
        "time_ms": 2200.0,
        "recorded_ip_a": 1283617.6680326462,
        "unscaled_source_ip_a": 1137702.226779481,
        "seed_lambda": 1.1282545096762369,
    },
    {
        "shot": "d3d_shot_0003ff34e7.parquet",
        "frame": 89,
        "time_ms": 1980.0,
        "recorded_ip_a": 1745309.6601366997,
        "unscaled_source_ip_a": 1826694.2826122313,
        "seed_lambda": 0.9554470481184465,
    },
    {
        "shot": "d3d_shot_001270afa9.parquet",
        "frame": 215,
        "time_ms": 5100.0,
        "recorded_ip_a": 990789.0240550041,
        "unscaled_source_ip_a": 666293.0624446461,
        "seed_lambda": 1.4870168697536397,
    },
    {
        "shot": "d3d_shot_001554e054.parquet",
        "frame": 41,
        "time_ms": 1160.0,
        "recorded_ip_a": 1006624.5902478695,
        "unscaled_source_ip_a": 1010288.9124805233,
        "seed_lambda": 0.9963729956971843,
    },
    {
        "shot": "d3d_shot_001cbcc9e6.parquet",
        "frame": 252,
        "time_ms": 5600.0,
        "recorded_ip_a": 998801.5315532684,
        "unscaled_source_ip_a": 801715.452065907,
        "seed_lambda": 1.2458304613931273,
    },
)


class LambdaOutOfBand(RuntimeError):
    """Report a profile amplitude outside the declared admissible band."""

    def __init__(self, value: float):
        self.value = float(value)
        super().__init__(
            f"profile amplitude {self.value:.12g} is outside {LAMBDA_BAND}"
        )


class _CriterionReached(Exception):
    """Stop a host root immediately after both declared criteria are met."""

    def __init__(self, state: np.ndarray):
        self.state = np.asarray(state, dtype=float)
        super().__init__("declared residual criteria reached")


@dataclass(frozen=True)
class MapEvaluation:
    """One constrained image and its current-amplitude diagnostics."""

    image: np.ndarray
    amplitude: float
    unscaled_current_a: float
    achieved_current_a: float


def preregistration() -> dict[str, Any]:
    """Return the complete declaration fixed before any corpus score."""

    return {
        "measurement": "plasma-current constrained free-boundary map",
        "selection": {
            "frames": len(REPRESENTATIVE_COHORT),
            "source": str(RECOVERY_OUTPUT / RECOVERY_CHECKPOINT_NAME),
            "rule": (
                "one maximum-absolute-current banked diverted frame per distinct "
                "shot, absolute recorded Ip at least 200 kA, retained only when "
                "target Ip and extracted unscaled source Ip have the same sign"
            ),
            "absolute_recorded_ip_floor_a": REPRESENTATIVE_CURRENT_FLOOR_A,
            "polarity_screen": (
                "every selected shot is absent from the landed 603-shot population"
            ),
            "cohort_declared_before_solver_scoring": list(REPRESENTATIVE_COHORT),
            "preflight_reproducibility_relative_tolerance": (
                COHORT_PREFLIGHT_RELATIVE_TOLERANCE
            ),
            "low_current_control_fixture": {
                "shot": LOW_CURRENT_CONTROL[0],
                "frame": LOW_CURRENT_CONTROL[1],
                "time_ms": 160.0,
                "recorded_ip_a": -3465.503291954519,
                "role": (
                    "selection-defect control only; excluded from every cohort "
                    "statistic and constrained solve because seed lambda is negative"
                ),
            },
        },
        "shared_inputs": {
            "seed": "the same convention-clean labelled branch seed in every arm",
            "poloidal_conductors": 24,
            "current_set": (
                "nineteen shipped poloidal currents plus five recovered currents; "
                "the shipped bcoil channel supplies the toroidal-field function"
            ),
            "coefficients_fitted": 0,
            "currents_adjusted": 0,
        },
        "target_current": {
            "source": "magnetics_plasma_current interpolated at the labelled time",
            "unit_crossing": "recorded kA multiplied by 1000 exactly once",
            "status": "declared constraint, not a fitted coefficient",
            "inference_availability": (
                "admissible competition input and may also be supplied by a partner "
                "transport solve"
            ),
        },
        "arms": {
            "unpinned": {
                "route": "accelerated newton_krylov",
                "newton_steps": 24,
                "gmres_iterations": 24,
                "warmup": 8,
                "relaxation": 0.5,
                "step_cap": 10.0,
                "low_current_full_24_plateau_control": UNPINNED_PLATEAU_CONTROL,
                "low_current_shipped_20_plateau_control": (
                    SHIPPED_ONLY_PLATEAU_CONTROL
                ),
                "control_absolute_tolerance": UNPINNED_CONTROL_ABSOLUTE_TOLERANCE,
                "representative_current_comparison": (
                    "remeasure shipped-20 and full-24 from the same seed per frame"
                ),
            },
            "pinned_eliminated": {
                "definition": (
                    "lambda = target Ip / unscaled clipped-support Ip at every "
                    "map evaluation; lambda multiplies all current moments"
                ),
                "unknowns_and_rows": "N flux unknowns and N flux residual rows",
                "route": "host Jacobian-free Newton-Krylov with Armijo search",
            },
        },
        "dropped_comparison_arm": {
            "name": "pinned_augmented",
            "status": "not applicable and not scored",
            "alpha_stability": "not applicable",
            "reason": (
                "an explicit current-residual row requires a clipped-support "
                "observation primitive the public ForwardOperator does not expose; "
                "closed-form elimination satisfies Ip exactly without that primitive, "
                "keeps the system square and needs no residual weight"
            ),
        },
        "host_budget": {
            "maximum_outer_iterations": HOST_OUTER_ITERATIONS,
            "maximum_inner_iterations": HOST_INNER_ITERATIONS,
        },
        "lambda_guard": {
            "inclusive_band": list(LAMBDA_BAND),
            "policy": (
                "no clipping: evaluation raises LambdaOutOfBand and the arm records "
                "a loud guard termination"
            ),
        },
        "qualification": {
            "relative_flux_residual": RELATIVE_RESIDUAL_CRITERION,
            "relative_current_error": CURRENT_CONSTRAINT_CRITERION,
            "requires_terminal_diverted": True,
        },
        "spectral_estimate": {
            "method": (
                "central-difference map action inside fixed-count power iteration"
            ),
            "iterations": POWER_ITERATIONS,
            "relative_step": POWER_RELATIVE_STEP,
            "comparison_band": [1.25, 1.40],
            "interpretation": (
                "estimate the Picard-map spectral radius; rho below one after "
                "pinning would rehabilitate damped fixed-point routes, while rho "
                "above one does not make the Newton root singular because the "
                "Newton residual Jacobian eigenvalue is 1-rho"
            ),
        },
        "nova_equilibrium_modified": False,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_preregistration(output: Path, *, replace: bool = False) -> Path:
    """Write the declaration, requiring an explicit action to replace it."""

    output.mkdir(parents=True, exist_ok=True)
    path = output / PREREGISTRATION_NAME
    encoded = json.dumps(preregistration(), indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded and not replace:
        raise RuntimeError("on-disk current-pinned preregistration differs")
    path.write_text(encoded)
    return path


def _recovery_inputs(affected_shots: set[str]) -> tuple[list[FrameInput], FrameInput]:
    """Load the preregistered frames from the landed incremental recovery bank."""

    path = RECOVERY_OUTPUT / RECOVERY_CHECKPOINT_NAME
    bank = {
        (item["shot"], int(item["frame"])): item
        for item in (
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        )
    }

    def frame_input(shot: str, frame: int) -> FrameInput:
        if shot in affected_shots:
            raise RuntimeError(f"selected shot {shot} is polarity affected")
        try:
            item = bank[(shot, frame)]
        except KeyError as error:
            raise RuntimeError(f"recovery bank lacks {shot}:{frame}") from error
        currents = item["recovered_currents_a"]
        return FrameInput(
            shot=shot,
            frame=frame,
            recovered_currents_a=tuple(float(currents[name]) for name in OMITTED_COILS),
        )

    cohort = [
        frame_input(str(item["shot"]), int(item["frame"]))
        for item in REPRESENTATIVE_COHORT
    ]
    if len({item.shot for item in cohort}) != len(REPRESENTATIVE_COHORT):
        raise RuntimeError("representative-current cohort must use distinct shots")
    low = frame_input(*LOW_CURRENT_CONTROL)
    return cohort, low


def _target_current(row: dict[str, Any], time_ms: float) -> float:
    """Return the recorded target current after its single kA-to-A crossing."""

    value = float(
        np.interp(
            time_ms,
            np.asarray(row["magnetics_plasma_current_times"], dtype=float),
            np.asarray(row["magnetics_plasma_current"], dtype=float),
        )
    )
    target = 1000.0 * value
    if not np.isfinite(target) or abs(target) <= np.finfo(float).tiny:
        raise RuntimeError(f"target plasma current {target} A is not qualified")
    return target


def _scaled_moments(
    moments: CellCurrentMoments, amplitude: jax.Array
) -> CellCurrentMoments:
    """Scale the common p-prime and FF-prime amplitude at current-image level."""

    return CellCurrentMoments(*(amplitude * value for value in moments))


def _lambda_value(target_current_a: float, unscaled_current_a: float) -> float:
    """Return the eliminated amplitude or raise on the declared guard."""

    amplitude = float(target_current_a / unscaled_current_a)
    if not np.isfinite(amplitude) or not (
        LAMBDA_BAND[0] <= amplitude <= LAMBDA_BAND[1]
    ):
        raise LambdaOutOfBand(amplitude)
    return amplitude


def eliminated_map(
    profile: ForwardProfile, current: np.ndarray, target_current_a: float
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[np.ndarray], MapEvaluation]]:
    """Return the exact-amplitude map and a guarded host evaluation wrapper."""

    operator = profile.operator
    external = operator.external(jnp.asarray(current))

    def traced(state: jax.Array):
        moments = operator.cell_current_moments(state, TopologyClass.DIVERTED)
        unscaled = jnp.sum(moments.cell_current)
        amplitude = jnp.asarray(target_current_a, dtype=unscaled.dtype) / unscaled
        image = external + operator.current_moment_image(
            _scaled_moments(moments, amplitude)
        )
        return image, amplitude, unscaled

    compiled = jax.jit(traced)

    def mapped(state: jax.Array) -> jax.Array:
        return traced(state)[0]

    def evaluated(state: np.ndarray) -> MapEvaluation:
        image, amplitude, unscaled = compiled(jnp.asarray(state))
        amplitude_value = _lambda_value(target_current_a, float(unscaled))
        return MapEvaluation(
            image=np.asarray(image, dtype=float),
            amplitude=amplitude_value,
            unscaled_current_a=float(unscaled),
            achieved_current_a=amplitude_value * float(unscaled),
        )

    return mapped, evaluated


def _relative_sup(image: np.ndarray, state: np.ndarray) -> float:
    return float(
        np.max(np.abs(np.asarray(image) - np.asarray(state)))
        / max(np.max(np.abs(image)), 1.0e-30)
    )


def _topology(profile: ForwardProfile, state: np.ndarray) -> tuple[str, np.ndarray]:
    _masks, topology = profile.operator.read(jnp.asarray(state))
    name = "diverted" if bool(topology.diverted) else "limited"
    return name, np.asarray(topology.x_point, dtype=float)


def solve_eliminated(
    profile: ForwardProfile,
    seed: np.ndarray,
    current: np.ndarray,
    target_current_a: float,
) -> dict[str, Any]:
    """Solve the square eliminated system and retain every accepted residual."""

    mapped, evaluate = eliminated_map(profile, current, target_current_a)
    history: list[float] = []
    amplitude_history: list[float] = []
    maximum_current_error = 0.0
    evaluations = 0

    def residual(state: np.ndarray) -> np.ndarray:
        nonlocal evaluations, maximum_current_error
        item = evaluate(state)
        evaluations += 1
        error = abs(item.achieved_current_a - target_current_a) / abs(target_current_a)
        maximum_current_error = max(maximum_current_error, error)
        return item.image - state

    initial = np.asarray(seed, dtype=float)
    terminal = initial
    termination = "outer iteration ceiling exhausted"
    guard_value: float | None = None

    def record(state: np.ndarray, value: np.ndarray) -> None:
        item = evaluate(state)
        relative = _relative_sup(item.image, state)
        history.append(relative)
        amplitude_history.append(item.amplitude)
        if relative <= RELATIVE_RESIDUAL_CRITERION:
            raise _CriterionReached(state)

    try:
        terminal = scipy.optimize.newton_krylov(
            residual,
            initial,
            method="gmres",
            inner_maxiter=HOST_INNER_ITERATIONS,
            maxiter=HOST_OUTER_ITERATIONS,
            f_tol=0.0,
            line_search="armijo",
            callback=record,
        )
        termination = "host solver returned"
    except _CriterionReached as reached:
        terminal = reached.state
        termination = "declared flux criterion reached"
    except scipy.optimize.NoConvergence as error:
        terminal = np.asarray(error.args[0], dtype=float)
    except LambdaOutOfBand as error:
        guard_value = error.value
        termination = str(error)

    try:
        final = evaluate(terminal)
        achieved = _relative_sup(final.image, terminal)
        amplitude = final.amplitude
        current_error = abs(final.achieved_current_a - target_current_a) / abs(
            target_current_a
        )
    except LambdaOutOfBand as error:
        guard_value = error.value
        achieved = float("inf")
        amplitude = error.value
        current_error = float("inf")
    topology, x_point = _topology(profile, terminal)
    return {
        "state": terminal,
        "relative_residual": achieved,
        "current_relative_error": current_error,
        "current_constraint_required": True,
        "amplitude": amplitude,
        "iterations": len(history),
        "map_evaluations": evaluations,
        "residual_history": history,
        "amplitude_history": amplitude_history,
        "maximum_map_current_relative_error": maximum_current_error,
        "topology": topology,
        "x_point_rz_m": x_point,
        "termination": termination,
        "lambda_guard_triggered": guard_value is not None,
        "lambda_guard_value": guard_value,
        "mapped": mapped,
    }


def power_iteration(
    mapped: Callable[[jax.Array], jax.Array], state: np.ndarray
) -> dict[str, Any]:
    """Estimate the dominant map eigenvalue with finite-difference actions."""

    compiled = jax.jit(mapped)
    state = np.asarray(state, dtype=float)
    generator = np.random.default_rng(11)
    vector = generator.normal(size=state.shape)
    vector /= np.linalg.norm(vector)
    state_scale = max(float(np.max(np.abs(state))), 1.0)
    growth: list[float] = []

    def action(direction: np.ndarray) -> np.ndarray:
        direction_scale = max(float(np.max(np.abs(direction))), 1.0e-300)
        delta = POWER_RELATIVE_STEP * state_scale / direction_scale
        plus = np.asarray(compiled(jnp.asarray(state + delta * direction)), dtype=float)
        minus = np.asarray(
            compiled(jnp.asarray(state - delta * direction)), dtype=float
        )
        return (plus - minus) / (2.0 * delta)

    finite = True
    for _ in range(POWER_ITERATIONS):
        image = action(vector)
        norm = float(np.linalg.norm(image))
        if not np.isfinite(norm) or norm <= 1.0e-300:
            finite = False
            break
        growth.append(norm)
        vector = image / norm
    if finite:
        final = action(vector)
        rayleigh = float(np.dot(vector, final))
    else:
        rayleigh = float("nan")
    return {
        "method": "central-difference map action in fixed-count power iteration",
        "iterations": len(growth),
        "relative_step": POWER_RELATIVE_STEP,
        "rayleigh_quotient": rayleigh,
        "absolute_dominant_eigenvalue_estimate": abs(rayleigh),
        "last_five_norm_growth_estimates": growth[-5:],
        "finite": finite and np.isfinite(rayleigh),
        "banked_diverted_state_comparison_band": [1.25, 1.40],
    }


def _serialise_arm(result: dict[str, Any]) -> dict[str, Any]:
    """Remove runtime arrays and attach the simultaneous qualification."""

    serial = {
        key: value for key, value in result.items() if key not in {"state", "mapped"}
    }
    for key in ("relative_residual", "current_relative_error"):
        value = float(serial[key])
        serial[key] = value if np.isfinite(value) else None
    serial["x_point_rz_m"] = (
        np.asarray(serial["x_point_rz_m"], dtype=float).tolist()
        if np.all(np.isfinite(serial["x_point_rz_m"]))
        else None
    )
    current_ok = bool(
        not serial.get("current_constraint_required", False)
        or (
            serial["current_relative_error"] is not None
            and serial["current_relative_error"] <= CURRENT_CONSTRAINT_CRITERION
        )
    )
    serial["simultaneously_meets_1e-6_and_diverted"] = bool(
        serial["relative_residual"] is not None
        and serial["relative_residual"] <= RELATIVE_RESIDUAL_CRITERION
        and current_ok
        and serial["topology"] == "diverted"
        and not serial["lambda_guard_triggered"]
    )
    return serial


def solve_unpinned(
    profile: ForwardProfile,
    seed: np.ndarray,
    current: np.ndarray,
    target_current_a: float,
) -> dict[str, Any]:
    """Run the fixed-budget accelerated control on one current vector."""

    mapped = profile.flux_map(jnp.asarray(current), TopologyClass.DIVERTED)
    accelerated = fixed_point.newton_krylov(
        mapped,
        jnp.asarray(seed),
        newton_steps=24,
        gmres_iterations=24,
        warmup=8,
        relaxation=0.5,
        step_cap=10.0,
    )
    state = np.asarray(accelerated.state, dtype=float)
    image = np.asarray(jax.jit(mapped)(accelerated.state), dtype=float)
    topology, x_point = _topology(profile, state)
    achieved_current = float(
        np.sum(
            np.asarray(
                profile.operator.cell_current(accelerated.state, TopologyClass.DIVERTED)
            )
        )
    )
    return {
        "state": state,
        "mapped": mapped,
        "relative_residual": _relative_sup(image, state),
        "current_relative_error": abs(achieved_current - target_current_a)
        / abs(target_current_a),
        "current_constraint_required": False,
        "amplitude": 1.0,
        "achieved_current_a": achieved_current,
        "iterations": 24,
        "map_evaluations": int(np.count_nonzero(np.isfinite(accelerated.trace))),
        "residual_history": [
            float(value)
            for value in np.asarray(accelerated.trace)
            if np.isfinite(value)
        ],
        "topology": topology,
        "x_point_rz_m": x_point,
        "termination": "fixed accelerated budget completed",
        "lambda_guard_triggered": False,
        "lambda_guard_value": None,
    }


def solve_frame(
    row: dict[str, Any],
    frame_input: FrameInput,
    geometry: dict[str, tuple[tuple[np.ndarray, float], ...]],
    declared: dict[str, Any],
) -> dict[str, Any]:
    """Run every registered arm from one identical labelled branch seed."""

    profile, seed, _label, _wall, _reliable, _statement = build_profile(
        row, frame_input.frame, PSEUDO_WALL_EXPANSION
    )
    profile = append_recovered_conductors(profile, geometry)
    shipped_current, current = current_arms(profile, frame_input.recovered_currents_a)
    time_ms = float(row["efit_times"][frame_input.frame])
    target = _target_current(row, time_ms)
    seed_unscaled = float(
        np.sum(np.asarray(profile.operator.cell_current(seed, TopologyClass.DIVERTED)))
    )
    seed_amplitude = _lambda_value(target, seed_unscaled)
    measured = (time_ms, target, seed_unscaled, seed_amplitude)
    registered = (
        float(declared["time_ms"]),
        float(declared["recorded_ip_a"]),
        float(declared["unscaled_source_ip_a"]),
        float(declared["seed_lambda"]),
    )
    if not np.allclose(
        measured,
        registered,
        rtol=COHORT_PREFLIGHT_RELATIVE_TOLERANCE,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            f"pre-solve cohort qualification drifted for {frame_input.shot}:"
            f"{frame_input.frame}: measured={measured}, registered={registered}"
        )
    if abs(target) < REPRESENTATIVE_CURRENT_FLOOR_A or target * seed_unscaled <= 0.0:
        raise RuntimeError("representative-current qualification failed before scoring")
    shipped_unpinned = solve_unpinned(profile, seed, shipped_current, target)
    unpinned = solve_unpinned(profile, seed, current, target)
    eliminated = solve_eliminated(profile, seed, current, target)

    for result in (unpinned, eliminated):
        result["dominant_map_eigenvalue"] = power_iteration(
            result["mapped"], result["state"]
        )
    record = {
        "shot": frame_input.shot,
        "frame": frame_input.frame,
        "time_ms": time_ms,
        "screened_out_of_affected_polarity_population": True,
        "target_plasma_current_a": target,
        "absolute_target_plasma_current_a": abs(target),
        "unscaled_seed_plasma_current_a": seed_unscaled,
        "seed_profile_amplitude": seed_amplitude,
        "target_and_unscaled_source_same_sign": bool(target * seed_unscaled > 0.0),
        "target_current_role": (
            "declared admissible input, not fitted; available in the competition "
            "input set or from a partner transport solve"
        ),
        "poloidal_conductor_count": 24,
        "same_label_branch_seed_all_arms": True,
        "coefficients_fitted": 0,
        "currents_adjusted": 0,
        "arms": {
            ARM_NAMES[0]: _serialise_arm(unpinned),
            ARM_NAMES[1]: _serialise_arm(eliminated),
        },
        "unconstrained_current_controls": {
            "shipped_20": _serialise_arm(shipped_unpinned),
            "full_24": _serialise_arm(unpinned),
            "shipped_to_full_residual_ratio": (
                shipped_unpinned["relative_residual"]
                / max(unpinned["relative_residual"], 1.0e-300)
            ),
        },
        "alpha_stability": {
            "status": "not applicable",
            "reason": "the explicit-row augmented comparison arm was dropped",
        },
    }
    return record


def solve_low_current_control(
    row: dict[str, Any],
    frame_input: FrameInput,
    geometry: dict[str, tuple[tuple[np.ndarray, float], ...]],
) -> dict[str, Any]:
    """Reproduce the ramp-start fixture without pooling or reversing its source."""

    profile, seed, _label, _wall, _reliable, _statement = build_profile(
        row, frame_input.frame, PSEUDO_WALL_EXPANSION
    )
    profile = append_recovered_conductors(profile, geometry)
    shipped_current, full_current = current_arms(
        profile, frame_input.recovered_currents_a
    )
    time_ms = float(row["efit_times"][frame_input.frame])
    target = _target_current(row, time_ms)
    unscaled = float(
        np.sum(np.asarray(profile.operator.cell_current(seed, TopologyClass.DIVERTED)))
    )
    attempted_amplitude = target / unscaled
    shipped = solve_unpinned(profile, seed, shipped_current, target)
    full = solve_unpinned(profile, seed, full_current, target)
    return {
        "role": (
            "low-current selection-defect control fixture; excluded from the "
            "representative-current scoring cohort"
        ),
        "shot": frame_input.shot,
        "frame": frame_input.frame,
        "time_ms": time_ms,
        "recorded_plasma_current_a": target,
        "absolute_recorded_plasma_current_a": abs(target),
        "unscaled_seed_plasma_current_a": unscaled,
        "attempted_seed_lambda": attempted_amplitude,
        "positive_lambda_guard_retained": list(LAMBDA_BAND),
        "constrained_arms_scored": False,
        "constrained_arm_refusal": (
            "negative lambda would reverse both extracted source terms and is not "
            "a current normalization"
        ),
        "shipped_20_unpinned": _serialise_arm(shipped),
        "full_24_unpinned": _serialise_arm(full),
        "shipped_to_full_residual_ratio": shipped["relative_residual"]
        / max(full["relative_residual"], 1.0e-300),
        "historical_full_24_plateau": UNPINNED_PLATEAU_CONTROL,
        "historical_full_24_plateau_reproduced": bool(
            abs(full["relative_residual"] - UNPINNED_PLATEAU_CONTROL)
            <= UNPINNED_CONTROL_ABSOLUTE_TOLERANCE
        ),
    }


def summarize(
    records: list[dict[str, Any]], low_current_control: dict[str, Any]
) -> dict[str, Any]:
    """Return cohort medians and the pinning verdict without hiding failures."""

    def values(arm: str, key: str) -> list[float]:
        return [
            float(item["arms"][arm][key])
            for item in records
            if item["arms"][arm][key] is not None
        ]

    arms = {}
    for arm in ARM_NAMES:
        residuals = values(arm, "relative_residual")
        radii = [
            float(
                item["arms"][arm]["dominant_map_eigenvalue"][
                    "absolute_dominant_eigenvalue_estimate"
                ]
            )
            for item in records
            if item["arms"][arm]["dominant_map_eigenvalue"]["finite"]
        ]
        amplitudes = values(arm, "amplitude")
        arms[arm] = {
            "median_relative_residual": (
                float(np.median(residuals)) if residuals else None
            ),
            "median_dominant_map_eigenvalue": (
                float(np.median(radii)) if radii else None
            ),
            "median_amplitude": float(np.median(amplitudes)) if amplitudes else None,
            "simultaneously_converged_and_diverted_frames": int(
                sum(
                    item["arms"][arm]["simultaneously_meets_1e-6_and_diverted"]
                    for item in records
                )
            ),
            "lambda_guard_terminations": int(
                sum(item["arms"][arm]["lambda_guard_triggered"] for item in records)
            ),
        }
    shipped_residuals = [
        float(item["unconstrained_current_controls"]["shipped_20"]["relative_residual"])
        for item in records
    ]
    full_residuals = [
        float(item["unconstrained_current_controls"]["full_24"]["relative_residual"])
        for item in records
    ]
    ratios = [
        float(item["unconstrained_current_controls"]["shipped_to_full_residual_ratio"])
        for item in records
    ]
    pin_passes = arms[ARM_NAMES[1]]["simultaneously_converged_and_diverted_frames"]
    return {
        "frame_count": len(records),
        "distinct_shots": len({item["shot"] for item in records}),
        "all_shots_screened_free_of_affected_population": all(
            item["screened_out_of_affected_polarity_population"] for item in records
        ),
        "all_frames_absolute_recorded_ip_at_least_200ka": all(
            item["absolute_target_plasma_current_a"] >= REPRESENTATIVE_CURRENT_FLOOR_A
            for item in records
        ),
        "all_frames_target_and_source_same_sign": all(
            item["target_and_unscaled_source_same_sign"] for item in records
        ),
        "representative_current_unpinned_comparison": {
            "median_shipped_20_relative_residual": float(np.median(shipped_residuals)),
            "median_full_24_relative_residual": float(np.median(full_residuals)),
            "median_paired_shipped_to_full_ratio": float(np.median(ratios)),
            "historical_low_current_ratio": (
                SHIPPED_ONLY_PLATEAU_CONTROL / UNPINNED_PLATEAU_CONTROL
            ),
            "historical_3_53x_holds_at_representative_current": bool(
                np.median(ratios) >= 3.5
            ),
        },
        "low_current_control_fixture": low_current_control,
        "arms": arms,
        "alpha_stability": {
            "status": "not applicable",
            "reason": (
                "the dropped explicit-row comparison would require a new public "
                "primitive, while the recommended eliminated construction does not"
            ),
        },
        "eigenvalue_interpretation": (
            "The reported value estimates the Picard-map spectral radius. Pinning "
            "below one would additionally rehabilitate damped fixed-point routes; "
            "a radius of 1.25 to 1.40 corresponds to a Newton residual-Jacobian "
            "eigenvalue of -0.25 to -0.40 and is not a singular Newton root."
        ),
        "current_pinning_removes_vacuum_root_and_orbiting_plateau": bool(
            pin_passes == len(records)
        ),
        "frames": records,
    }


def _figure(summary: dict[str, Any], path: Path) -> None:
    """Plot residual, source amplitude and spectral-radius comparisons."""

    records = summary["frames"]
    labels = [f"{item['shot'][9:17]}:{item['frame']}" for item in records]
    x = np.arange(len(records))
    width = 0.34
    colors = ("#4477aa", "#228833")
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    for offset, (arm, color) in enumerate(zip(ARM_NAMES, colors, strict=True)):
        residual = [
            item["arms"][arm]["relative_residual"] or np.nan for item in records
        ]
        radius = [
            item["arms"][arm]["dominant_map_eigenvalue"][
                "absolute_dominant_eigenvalue_estimate"
            ]
            for item in records
        ]
        amplitude = [item["arms"][arm]["amplitude"] for item in records]
        shift = (offset - 0.5) * width
        axes[0].bar(x + shift, residual, width, color=color, label=arm)
        axes[1].bar(x + shift, amplitude, width, color=color)
        axes[2].bar(x + shift, radius, width, color=color)
    axes[0].set_yscale("log")
    axes[0].axhline(RELATIVE_RESIDUAL_CRITERION, color="black", linestyle="--")
    axes[0].set_ylabel("terminal relative residual")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].set_ylabel("profile amplitude lambda")
    axes[2].axhspan(1.25, 1.40, color="#bbbbbb", alpha=0.35)
    axes[2].axhline(1.0, color="black", linestyle="--")
    axes[2].set_ylabel("dominant map eigenvalue magnitude")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=35, ha="right")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(data: Path, output: Path) -> dict[str, Any]:
    """Run the fixed cohort and publish incremental, receipt and plot artifacts."""

    configure_dtypes()
    declaration = write_preregistration(output)
    recovery_path = RECOVERY_OUTPUT / RECOVERY_RECEIPT_NAME
    polarity = json.loads(POLARITY_RECEIPT.read_text())["full_corpus_census"]
    affected = set(polarity["affected_shots"])
    if len(affected) != POLARITY_AFFECTED_SHOT_COUNT:
        raise RuntimeError("polarity authority is not the landed 603-shot population")
    selected, low_input = _recovery_inputs(affected)
    geometry = _omitted_vertices()
    columns = tuple(
        dict.fromkeys(
            (
                *_LABEL_COLUMNS,
                *_CURRENT_COLUMNS,
                *_GEOMETRY_COLUMNS,
                *PLASMA_CURRENT_COLUMNS,
            )
        )
    )
    checkpoint = output / CHECKPOINT_NAME
    checkpoint.write_text("")
    low_path = data / low_input.shot
    low_row = _read(low_path, columns)
    low_row["_source_path"] = str(low_path)
    low_control = solve_low_current_control(low_row, low_input, geometry)
    print(
        "LOW_CURRENT_CONTROL "
        f"{low_input.shot}:{low_input.frame} "
        f"Ip={low_control['recorded_plasma_current_a']:.9f} "
        f"lambda={low_control['attempted_seed_lambda']:.12g} "
        "excluded_from_cohort=true",
        flush=True,
    )
    records = []
    for number, (frame_input, declared) in enumerate(
        zip(selected, REPRESENTATIVE_COHORT, strict=True), start=1
    ):
        path = data / frame_input.shot
        row = _read(path, columns)
        row["_source_path"] = str(path)
        record = solve_frame(row, frame_input, geometry, declared)
        records.append(record)
        with checkpoint.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        print(
            f"SOLVED {number}/{len(selected)} {frame_input.shot}:{frame_input.frame} "
            + " ".join(
                f"{arm}={record['arms'][arm]['relative_residual']}/"
                f"{record['arms'][arm]['topology']}"
                for arm in ARM_NAMES
            ),
            flush=True,
        )
    result = summarize(records, low_control)
    receipt = {
        "preregistration": preregistration(),
        "preregistration_path": str(declaration),
        "preregistration_sha256": _sha256(declaration),
        "authorities": {
            "recovery_receipt": str(recovery_path),
            "recovery_receipt_sha256": _sha256(recovery_path),
            "recovery_incremental_bank": str(
                RECOVERY_OUTPUT / RECOVERY_CHECKPOINT_NAME
            ),
            "recovery_incremental_bank_sha256": _sha256(
                RECOVERY_OUTPUT / RECOVERY_CHECKPOINT_NAME
            ),
            "polarity_receipt": str(POLARITY_RECEIPT),
            "polarity_receipt_sha256": _sha256(POLARITY_RECEIPT),
            "affected_shot_count": len(affected),
        },
        "interpretation": (
            "Prescribed Ip is a declared admissible input of the same class as the "
            "coil currents, available in the competition inputs or from a partner "
            "transport solve; it is not fitted inside this equilibrium solve. The "
            "representative seed amplitudes 0.955 to 1.487 are positive and near "
            "unity, independently confirming that the ramp-start fixture's "
            "-0.00712 amplitude was a frame-selection defect."
        ),
        "design_finding": (
            "The arm requiring a new public clipped-support primitive is the "
            "explicit-row comparison that was not recommended; closed-form "
            "elimination keeps the system square and enforces current exactly "
            "without that primitive or an alpha weight."
        ),
        "result": result,
    }
    (output / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _figure(result, output / FIGURE_NAME)
    if not result["low_current_control_fixture"][
        "historical_full_24_plateau_reproduced"
    ]:
        raise RuntimeError(
            "the separated low-current plateau control did not reproduce"
        )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preregister-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.preregister_only:
        print(f"PREREGISTERED {write_preregistration(arguments.output, replace=True)}")
        return
    receipt = run(arguments.data, arguments.output)
    headline = dict(receipt["result"])
    headline.pop("frames", None)
    print(json.dumps(headline, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
