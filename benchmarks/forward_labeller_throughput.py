"""Measure the forward solve as a training labeller on the H200.

One MAST shot's efm slices are stepped in time order through the constrained
reduced-Newton route, warm-started slice to slice from the previous
equilibrium and unknown, in two arms: free, and with the vertical
current-centroid row pinned to that slice's EFIT centre height.  Every slice
records the trips, Newton steps, wall, converged and qualified flags, the
achieved centroid against the target, the compensating current, and the
conditioning flag with its target source, persisted as each slice lands.

The operator for the shot is built once (from the calibrated keyframe slice);
each slice supplies its own fitted circuit currents as the ``prescribed``
current argument and, in the conditioned arm, its own EFIT centre-height
target.  These are traced kernel arguments, so one reduced program serves
the arm's slices while the receipt still states the first compile wall beside
the warm steady-state figure.

The receipt states slices per second and per GPU-hour per arm, the converged
and qualified fractions, and the extrapolation to the 1,341,435-slice census
on one and on eight H200s, and carries the two records this section stipulates
beside the numbers: the per-slice conditioning flag with its target source,
and the caveat that every conditioned label inherits one reconstruction scalar
per slice on the axis position.

The vertical centre-height pin is a scoped exception authorised for the demo
only: it is removed when a Thomson forward model can supply the plasma height
instead of the reconstruction.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.interpolate import RectBivariateSpline

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.efit_forward_parity_slice import (
    FIXED_POINT_CRITERION,
    GRID_STRIDE,
    TOTAL_FLUX_FACTOR,
    _mast_case_from_selection,
    _passive_inclusive_case,
)
from benchmarks.efit_topology_boundary_score import _live_flux_map, _stored_x_points
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium import reduced_newton
from nova.equilibrium.constraint import (
    ConstraintBinding,
    ConstraintMultiplier,
    ConstraintPair,
    CurrentCentroidConstraint,
    compensator_rule_name,
)
from nova.equilibrium.observation import MomentIntegralSupport
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


ROOT = Path(__file__).resolve().parents[1]
SHOT = 22086
#: The calibrated keyframe used to build the shot's one operator.
KEYFRAME_SLICE = 43
#: The shot holds 60 reconstructed efm slices at a 5 ms cadence; rows 1..57
#: carry finite EFIT centre heights and fitted currents, so that is the
#: consecutive time-ordered chain both arms step.  Rows 0, 58 and 59 lack the
#: centre-height/current references and are recorded as excluded.
SLICES = tuple(range(1, 58))
DEFAULT_OUTPUT = (
    ROOT
    / "docs/figures/playable-forward-solve/labeller/forward-labeller-throughput.json"
)
DEFAULT_FIGURE = (
    ROOT
    / "docs/figures/playable-forward-solve/labeller/forward-labeller-throughput.png"
)
DEFAULT_LAUNCHER = (
    ROOT / "docs/figures/playable-forward-solve/labeller/run_labeller_h200.sh"
)
#: Newton budget per trip (the constrained keyframe driver's figure).
NEWTON_STEPS = 24
#: The decoder corpus census this section's extrapolation quotes.
CORPUS_CENSUS = 1_341_435
#: The training-side decode rate the online question is read against, one
#: decoded frame per second per process, per the imas-ambix consumer facts.
DECODE_FRAMES_PER_SECOND_PER_PROCESS = 8


def _source_revision() -> str:
    """Return the revision this measurement runs from."""
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _strict_float(value: Any) -> float | None:
    """Return one finite host float, or None where the value is not finite."""
    result = float(np.asarray(value))
    return result if np.isfinite(result) else None


def _centroid(profile, flux, target_current) -> float:
    """Return the vertical current centroid [m] of one flux state."""
    return float(
        np.asarray(
            profile.current_moment_observation(
                jnp.asarray(flux),
                support=MomentIntegralSupport.ALL_DOMAIN,
                target_current=target_current,
            ).centroid_z
        )
    )


def _circuit_names(policy) -> dict[int, str]:
    """Return the zero-based circuit index of every named active family."""
    return {
        int(item["stored_circuit"]) - 1: str(item["family"])
        for item in policy["active_mapping"]
    }


def _centroid_pair(profile, flux, *, target, unknown, target_current, requested, names):
    """Return the centroid row with a matrix-led compensating direction.

    The seed pair carries a multiplier rather than a circuit, so the
    derivation supplies both the direction and the current amplitude that
    moves the row by one declared scale.  ``unknown`` is the normalised
    compensation the previous slice settled on, which is what makes a warm
    start warm in the unknowns as well as in the flux.

    The vertical centre-height pin applied here is a scoped exception
    authorised for the demo only; it is removed when a Thomson forward model
    can supply the plasma height instead of the reconstruction.
    """
    scale = float(np.ptp(np.asarray(profile.lattice.height)))
    seeded = ConstraintPair(
        functional=CurrentCentroidConstraint(
            components=("centroid_z",),
            support=MomentIntegralSupport.ALL_DOMAIN,
        ),
        unknown=ConstraintMultiplier(multiplier_scale=jnp.asarray([1.0])),
        binding=ConstraintBinding(
            target=jnp.asarray([target]),
            tolerance=jnp.asarray([1.0e-6]),
            scale=jnp.asarray([scale]),
            initial_unknown=jnp.asarray([0.0]),
        ),
    )
    (derived,), selection = profile.derived_constraint_pairs(
        (seeded,),
        jnp.asarray(flux),
        requested_class=requested,
        target_current=target_current,
        circuits=sorted(names),
    )
    rescaled = ConstraintPair(
        functional=derived.functional,
        unknown=derived.unknown,
        binding=ConstraintBinding(
            target=derived.binding.target,
            tolerance=derived.binding.tolerance,
            scale=derived.binding.scale,
            initial_unknown=jnp.asarray(
                np.atleast_1d(
                    np.asarray(0.0 if unknown is None else unknown, dtype=float)
                )
            ),
        ),
    )
    return rescaled, selection


def _walls(result) -> dict[str, Any]:
    """Return the per-trip wall breakdown one solve measured."""
    warm = result.trip_wall_per_trip[1:]
    return {
        "trip_count": len(result.trip_wall_per_trip),
        "trip_wall_per_trip_s": result.trip_wall_per_trip,
        "first_trip_wall_s": (
            result.trip_wall_per_trip[0] if result.trip_wall_per_trip else None
        ),
        "median_warm_trip_wall_s": float(np.median(warm)) if warm else None,
        "warm_wall_s": float(np.sum(warm)) if warm else None,
    }


def _slice_table(
    group: zarr.Group, slices: tuple[int, ...]
) -> dict[int, dict[str, Any]]:
    """Return per-row fitted currents, centre heights and time references."""
    table: dict[int, dict[str, Any]] = {}
    for row in slices:
        table[row] = {
            "time_s": float(group["time"][row]),
            "current_a": np.asarray(group["fcoil_c"][row], dtype=np.float64).tolist(),
            "centre_height_m": float(group["current_centrd_z"][row]),
            "axis_z_m": float(group["magnetic_axis_z"][row]),
            "reference_plasma_current_a": float(group["plasma_current_c"][row]),
        }
    return table


def _requested_class(group, row: int) -> int:
    """Return the topology class the slice's own reconstruction carries.

    Early discharge rows are limited plasmas without an X-point, so pinning
    every slice to the diverted class fails the topology read on them; each
    slice requests the class its stored X-point set declares instead.
    """
    return int(
        TopologyClass.DIVERTED
        if len(_stored_x_points(group, row))
        else TopologyClass.LIMITED
    )


def _slices_seed(group: zarr.Group, row: int, full_r, full_z) -> np.ndarray:
    """Return the reference-flux seed one slice induces, without the operator.

    The benchmark grid subsamples the stored 65-point axes by ``GRID_STRIDE``
    and the wall nodes continue the reference through a third-order spline,
    exactly as ``build_profile`` constructs the slice-43 seed; only the
    geometry-independent response matrices are skipped, because they do not
    change across the slice.
    """
    reference_full = TOTAL_FLUX_FACTOR * _live_flux_map(group, row, len(full_r)).T
    reference = reference_full[::GRID_STRIDE, ::GRID_STRIDE]
    spline = RectBivariateSpline(full_r, full_z, reference_full, kx=3, ky=3, s=0.0)
    limiter = np.column_stack(
        [
            np.asarray(group["limiterr"], dtype=float),
            np.asarray(group["limiterz"], dtype=float),
        ]
    )
    return np.r_[reference.ravel(), spline.ev(limiter[:, 0], limiter[:, 1])]


def _draw_figure(receipt: dict[str, Any], output: Path) -> None:
    """Draw per-slice walls and the converged/qualified fractions per arm."""
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))
    arms = ("free", "conditioned")
    colours = {"free": "#3b6ea5", "conditioned": "#a53b3b"}
    for index, arm in enumerate(arms):
        records = [item for item in receipt["arms"][arm]["slices"] if item]
        if not records:
            continue
        axis = axes[index]
        times = [item["time_s"] for item in records]
        axis.plot(
            times,
            [1.0e3 * item["solve_wall_s"] for item in records],
            "-",
            color=colours[arm],
            lw=1.1,
            label="as-built (with compile)",
        )
        axis.plot(
            times,
            [1.0e3 * item["steady_wall_s"] for item in records],
            "--",
            color=colours[arm],
            lw=1.1,
            label="warm steady state",
        )
        axis.set_yscale("log")
        axis.set_ylabel(f"{arm} slice wall [ms]")
        axis.set_xlabel("time [s]")
        axis.legend(frameon=False, fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    axis = axes[2]
    labels = []
    walls = []
    for arm in arms:
        summary = receipt["arms"][arm]["summary"]
        labels.append(arm)
        walls.append(summary["median_as_built_wall_per_slice_s"])
    axis.bar(
        np.arange(len(labels)), [1.0e3 * value for value in walls], color="#3b6ea5"
    )
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_ylabel("median as-built wall per slice [ms]")
    axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.2)
    title = (
        f"Forward labeller throughput on {receipt['identity']} — "
        f"{receipt['source_commit'][:8]}"
    )
    figure.suptitle(title, y=0.97)
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.13, top=0.84, wspace=0.32)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_json(receipt: dict[str, Any], output: Path) -> None:
    """Persist the receipt so far, creating its directory once."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def _vmap_probe(operator, coordinates, external, requested, target_current, seed):
    """Probe a batched entry over 8 and 16 slices of the fused trip close.

    The route's fused tunnelled kernel is the trip boundary (``boundary``),
    which closes one trip inside a single compiled program; its Newton ladder
    is a host loop with data-dependent backtracking, so a vmap can only batch
    the fused boundary, not the whole trip.  The probe times one vmapped call
    against the same number of serial calls on the same program, and records
    the refusal reason if the vmap cannot trace.
    """
    kernels = reduced_newton._reduced_kernels(
        operator, coordinates, external, requested, target_current
    )
    kernels = reduced_newton._bind_dynamic_arguments(
        kernels, external, jnp.asarray(target_current), requested
    )
    reduced = np.asarray(kernels["initial_gather"](jnp.asarray(seed)), dtype=float)
    shadow = np.ravel(
        np.asarray(
            operator.residual_shadow_mask(jnp.asarray(seed), requested), dtype=bool
        )
    )
    result: dict[str, Any] = {
        "probe": (
            "jax.vmap of the fused trip-boundary kernel over a batch of states "
            "on one program"
        ),
        "batches": {},
    }
    for batch in (8, 16):
        states = [seed.copy() for _ in range(batch)]
        states[-1] = seed
        serial_started = time.perf_counter()
        for _ in range(3):
            for item in states:
                kernels["boundary"](
                    jnp.asarray(reduced), jnp.asarray(shadow), jnp.asarray(item)
                )
        serial_wall = (time.perf_counter() - serial_started) / 3
        try:
            batched = jax.vmap(kernels["boundary"])
            batched_started = time.perf_counter()
            for _ in range(3):
                batched(
                    jnp.stack([jnp.asarray(reduced) for _ in states]),
                    jnp.stack([jnp.asarray(shadow) for _ in states]),
                    jnp.stack([jnp.asarray(item) for item in states]),
                )
            batched_wall = (time.perf_counter() - batched_started) / 3
            result["batches"][str(batch)] = {
                "serial_calls_wall_s": serial_wall,
                "vmap_wall_s": batched_wall,
                "serial_per_slice_s": serial_wall / batch,
                "vmap_per_slice_s": batched_wall / batch,
                "vmap_over_serial": batched_wall / serial_wall,
            }
        except Exception as error:  # noqa: BLE001 - the probe records the refusal
            result["batches"][str(batch)] = {
                "serial_calls_wall_s": serial_wall,
                "vmap_refused": f"{type(error).__name__}: {error}",
            }
    result["route_refusal_reason"] = (
        "a full batched entry over distinct-current slices does not exist in the "
        "route: the reduced program now accepts external fields, prescribed "
        "currents and row leaves as traced arguments, but the Newton ladder is a "
        "host loop whose backtracking selection is data-dependent; only the "
        "fused trip-boundary kernel admits a vmap, and only over states of one "
        "program"
    )
    return result, kernels


def _write_launcher(
    launcher: Path,
    *,
    revision: str,
    output: Path,
    figure: Path,
) -> None:
    """Persist the working H200 harness used for this receipt."""
    del output, figure
    launcher.parent.mkdir(parents=True, exist_ok=True)
    wrap = (
        "export TMPDIR=/tmp JAX_PLATFORMS=cuda,cpu "
        'JAX_ENABLE_COMPILATION_CACHE=1 PYTHONPATH="$H200_LABELLER_ROOT"; '
        'echo "H200_LABELLER_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; '
        'echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"; '
        'echo "SOURCE_REVISION=$(git -C "$H200_LABELLER_ROOT" rev-parse HEAD)"; '
        'echo "PYTHONPATH=$PYTHONPATH"; '
        '"$H200_LABELLER_ROOT/.venv/bin/python" -m '
        "benchmarks.forward_labeller_throughput --output "
        '"$H200_LABELLER_ROOT/docs/figures/playable-forward-solve/labeller/'
        'forward-labeller-throughput.json" --figure '
        '"$H200_LABELLER_ROOT/docs/figures/playable-forward-solve/labeller/'
        'forward-labeller-throughput.png"; echo "H200_LABELLER_EXIT=$?"'
    )
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "# Submission harness for the forward-labeller throughput receipt.\n"
        f"# source revision {revision}\n"
        "set -euo pipefail\n"
        'ROOT="$(git -C "$(dirname "$(realpath -e -- "${BASH_SOURCE[0]}")")" '
        'rev-parse --show-toplevel)"\n'
        'OUT="${1:?missing log path}"\n'
        'LOG_DIR="$(dirname "$(realpath -m -- "${OUT}")")"\n'
        'if [[ -e "${OUT}" ]]; then echo "refusing to overwrite ${OUT}" >&2; '
        "exit 2; fi\n"
        'mkdir -p -- "${LOG_DIR}"\n'
        "sbatch --parsable --job-name=nova-labeller-throughput \\\n"
        "  --partition=betelgeuse \\\n"
        "  --reservation=gpu_0003_grpA \\\n"
        "  --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus=h200:1 --mem=64G \\\n"
        '  --time=00:55:00 --chdir="${ROOT}" --output="${OUT}" '
        '--error="${OUT}" \\\n'
        '  --export="ALL,H200_LABELLER_ROOT=${ROOT}" \\\n'
        f"  --wrap='{wrap}'\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)


def _compiled_flag(item: dict[str, Any]) -> bool:
    """Return whether one slice's first trip paid a program compile or load.

    A warm trip on the millisecond route costs tens of milliseconds; a fresh
    program compile or persistent-cache load costs seconds, so a first trip
    above a quarter second is unambiguously a new program for that solve.
    """
    first = item.get("first_trip_wall_s")
    warm = item.get("median_warm_trip_wall_s")
    if first is None:
        return False
    if warm is not None:
        return bool(first > 5.0 * warm)
    return bool(first > 0.25)


def _program_cache_sizes(program) -> dict[str, int]:
    """Return the compiled-entry count for every kernel in one program."""
    if program is None:
        return {}
    return {
        name: int(kernel._cache_size())
        for name, kernel in program.kernels.items()
        if hasattr(kernel, "_cache_size")
    }


def _warm_slice_wall(item: dict[str, Any], global_trip_wall: float | None) -> float:
    """Return one slice's warm steady-state cost, every trip at the warm rate.

    A slice that settled in one trip has no within-solve warm reference, so
    its single trip is billed at the arm's median warm-trip wall instead of
    the zero the raw tally would report; a multi-trip slice is the product of
    its own trip count and its own median warm-trip wall.  This is the rate
    the same loop costs once the programs are hot and the currents are traced
    arguments rather than trace constants.
    """
    trips = int(item.get("trip_count") or 0)
    if trips <= 0:
        return 0.0
    median = item.get("median_warm_trip_wall_s")
    if median is not None:
        return trips * float(median)
    if global_trip_wall is not None:
        return trips * global_trip_wall
    return 0.0


def _arm_rate_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse one arm's slice records into rate statistics."""
    measured = [item for item in records if item is not None]
    solved = [item for item in measured if item["solved"]]
    converged = [item for item in solved if item["converged"]]
    conditioned = any(bool(item["conditioning_flag"]) for item in solved)
    qualified = (
        [item for item in converged if bool(item["row_qualified"])]
        if conditioned
        else None
    )
    total_wall = float(np.sum([item["solve_wall_s"] for item in measured]))
    as_built_walls = [item["solve_wall_s"] for item in measured]
    reference_trips = [
        float(item["median_warm_trip_wall_s"])
        for item in measured
        if item.get("median_warm_trip_wall_s") is not None
    ]
    global_trip_wall = float(np.median(reference_trips)) if reference_trips else None
    warm_walls = [_warm_slice_wall(item, global_trip_wall) for item in measured]
    warm_wall = float(np.sum(warm_walls))
    per_second = len(measured) / total_wall if total_wall > 0 else 0.0
    warm_per_second = len(measured) / warm_wall if warm_wall > 0 else 0.0
    return {
        "slices_measured": len(measured),
        "slices_solved": len(solved),
        "slices_converged": len(converged),
        "slices_qualified": len(qualified) if qualified is not None else None,
        "solved_fraction": len(solved) / len(measured) if measured else 0.0,
        "converged_fraction": len(converged) / len(solved) if solved else 0.0,
        "qualified_fraction": (
            len(qualified) / len(converged) if qualified and converged else None
        ),
        "total_wall_s": total_wall,
        "median_as_built_wall_per_slice_s": float(np.median(as_built_walls))
        if as_built_walls
        else None,
        "median_warm_wall_per_slice_s": float(np.median(warm_walls))
        if warm_walls
        else None,
        "median_warm_trip_wall_s_global": global_trip_wall,
        "as_built_slices_per_second": per_second,
        "warm_slices_per_second": warm_per_second,
        "as_built_slices_per_gpu_hour": per_second * 3600.0,
        "warm_slices_per_gpu_hour": warm_per_second * 3600.0,
        "compiles_inferred": int(sum(1 for item in measured if _compiled_flag(item))),
    }


def measure(*, output: Path, figure: Path, cache_root: Path | None = None):
    """Step the shot's slices through both arms and write the receipt."""
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
        if cache_root is None
        else cache_root
    )
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    group = zarr.open_group(str(SHOT_STORE / f"{SHOT}.zarr"), mode="r")["efm"]
    table = _slice_table(group, SLICES)
    full_r = np.asarray(group["gridr"], dtype=np.float64)
    full_z = np.asarray(group["gridz"], dtype=np.float64)

    case, context = _mast_case_from_selection(
        SHOT_STORE,
        {"shot": SHOT, "slice_index": KEYFRAME_SLICE},
        {"note": "throughput keyframe; not a decomposition-bank selection"},
    )
    passive_case, profile, policy = _passive_inclusive_case(
        case, context, response_cache
    )
    operator = profile.operator
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
    names = _circuit_names(policy)

    seed_43 = jnp.asarray(case["state"])
    identity = f"{SHOT}/{KEYFRAME_SLICE}-keyframe"
    receipt: dict[str, Any] = {
        "artifact": "forward labeller throughput on one MAST shot",
        "identity": identity,
        "source_commit": _source_revision(),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
            "scheduler": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "node": os.environ.get("SLURMD_NODENAME"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
                "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
            },
        },
        "evidence_inputs": {
            "response_carrier": carrier_evidence,
            "persistent_compilation_cache": cache.receipt(),
            "shot": SHOT,
            "keyframe_slice": KEYFRAME_SLICE,
            "slices": list(SLICES),
            "slice_cadence_s": 5.0e-3,
            "excluded_slices": {
                "note": (
                    "rows 0, 58, 59 lack finite EFIT centre height or current reference"
                )
            },
        },
        "measurement_contract": {
            "route_free": "reduced_newton.solve_reduced_newton",
            "route_conditioned": (
                "reduced_newton.solve_constrained_reduced_newton with the "
                "vertical current-centroid row targeted on efm/current_centrd_z"
            ),
            "decision": (
                "labeller-vertical-constraint: pin the centre height to the "
                "reconstruction value per slice; the flag and target source "
                "record which labels carry the pin"
            ),
            "height_pin_gate": (
                "scoped exception for the demo only; removed when a Thomson "
                "forward model can supply the plasma height instead of the "
                "reconstruction"
            ),
            "warm_start": (
                "each slice re-solves from the equilibrium and, in the "
                "conditioned arm, the normalised compensation the previous "
                "slice settled on"
            ),
            "currents_source": "efm/fcoil_c per slice passed as prescribed_current",
            "operator": f"built once at {SHOT}/{KEYFRAME_SLICE}",
            "convergence_tolerance": FIXED_POINT_CRITERION,
            "newton_steps": NEWTON_STEPS,
            "requested_class": "diverted",
            "conditional_label_caveat": (
                "every conditioned slice inherits one reconstruction scalar "
                "per slice on the axis position; a held-out result scored by "
                "the EFIT referee on a corpus built with these labels carries "
                "a one-scalar-per-slice leak on exactly that number"
            ),
        },
        "arms": {
            "free": {
                "conditioning": {"conditioned": False, "target_source": None},
                "slices": [None] * len(SLICES),
            },
            "conditioned": {
                "conditioning": {
                    "conditioned": True,
                    "target_source": "efm/current_centrd_z",
                },
                "slices": [None] * len(SLICES),
            },
        },
        "vmap_probe": None,
    }
    _write_json(receipt, output)

    def persist_arm(arm: str, index: int, record: dict[str, Any]):
        receipt["arms"][arm]["slices"][index] = record
        _write_json(receipt, output)

    seeds = {row: _slices_seed(group, row, full_r, full_z) for row in SLICES}

    # --- Free arm ---
    state = jnp.asarray(seeds[SLICES[0]])
    program = None
    for index, row in enumerate(SLICES):
        current = table[row]["current_a"]
        requested = jnp.asarray(_requested_class(group, row), dtype=jnp.int8)
        target_current = abs(float(table[row]["reference_plasma_current_a"]))
        print(f"LABELLER slice {row} free {identity}", flush=True)
        started = time.perf_counter()
        item: dict[str, Any] = {"row": row, **table[row]}
        item["converged"] = False
        try:
            result = reduced_newton.solve_reduced_newton(
                operator,
                state,
                requested_class=requested,
                target_current=target_current,
                prescribed_current=jnp.asarray(current),
                tolerance=FIXED_POINT_CRITERION,
                newton_steps=NEWTON_STEPS,
                program=program,
                stream=False,
            )
            item["solved"] = True
            item.update(
                {
                    "route": "free",
                    "solve_wall_s": time.perf_counter() - started,
                    "terminal_residual": result.terminal_residual,
                    "active_set_trips": result.active_set_iterations,
                    "newton_step_count": int(sum(result.newton_steps_per_trip)),
                    "converged": bool(result.converged),
                    "termination": result.termination_name,
                    "reduced_dimension": result.reduced_dimension,
                    "off_support_leakage": result.off_support_leakage,
                }
            )
            item.update(_walls(result))
            item["conditioning_flag"] = False
            item["target_source"] = None
            item["row_qualified"] = None
            item["compensating_current_a"] = 0.0
            item["compensator_rule"] = None
            state = result.state
            program = result.program
            try:
                achieved = _centroid(profile, result.state, target_current)
                item["achieved_centroid_m"] = achieved
                item["centroid_error_against_efit_target_m"] = (
                    achieved - item["centre_height_m"]
                )
            except Exception as error:  # noqa: BLE001 - the solved state is the
                # label; a failed moment read only denies the achieved number
                item["achieved_centroid_m"] = None
                item["centroid_error_against_efit_target_m"] = None
                item["centroid_read_failure"] = f"{type(error).__name__}: {error}"
        except Exception as error:  # noqa: BLE001 - one bad slice must not kill the chain
            item["solved"] = False
            item["solve_wall_s"] = time.perf_counter() - started
            item["failure"] = f"{type(error).__name__}: {error}"
            # A converged state whose magnetic axis cannot be re-read would
            # strand the whole chain at the next slice, so a failed slice is
            # re-entered from its own reconstruction seed; the re-seed is the
            # stated cost of the warm-start break.
            state = jnp.asarray(seeds[row])
        item["program_compiled"] = _compiled_flag(item)
        item["steady_wall_s"] = _warm_slice_wall(item, None)
        persist_arm("free", index, item)
        print(
            "LABELLER-DONE "
            + json.dumps(
                {
                    k: item.get(k)
                    for k in ("row", "solved", "converged", "solve_wall_s")
                },
                sort_keys=True,
            ),
            flush=True,
        )

    # --- Conditioned arm ---
    free_program = program
    state = jnp.asarray(seeds[SLICES[0]])
    unknown = None
    program = None
    for index, row in enumerate(SLICES):
        current = table[row]["current_a"]
        target = table[row]["centre_height_m"]
        requested = jnp.asarray(_requested_class(group, row), dtype=jnp.int8)
        target_current = abs(float(table[row]["reference_plasma_current_a"]))
        print(f"LABELLER slice {row} conditioned {identity}", flush=True)
        started = time.perf_counter()
        item: dict[str, Any] = {"row": row, **table[row]}
        item["converged"] = False
        try:
            pair, selection = _centroid_pair(
                profile,
                state,
                target=target,
                unknown=unknown,
                target_current=target_current,
                requested=requested,
                names=names,
            )
            result = reduced_newton.solve_constrained_reduced_newton(
                profile,
                state,
                constraint_pairs=(pair,),
                requested_class=requested,
                target_current=target_current,
                prescribed_current=jnp.asarray(current),
                tolerance=FIXED_POINT_CRITERION,
                newton_steps=NEWTON_STEPS,
                program=program,
                stream=False,
            )
            record = result.constraints[0]
            item["solved"] = True
            item.update(
                {
                    "route": "conditioned",
                    "solve_wall_s": time.perf_counter() - started,
                    "terminal_residual": result.terminal_residual,
                    "active_set_trips": result.active_set_iterations,
                    "newton_step_count": int(sum(result.newton_steps_per_trip)),
                    "converged": bool(result.converged),
                    "termination": result.termination_name,
                    "reduced_dimension": result.reduced_dimension,
                    "off_support_leakage": result.off_support_leakage,
                    "conditioning_flag": True,
                    "target_source": "efm/current_centrd_z",
                    "target_centre_height_m": target,
                    "row_error_m": _strict_float(record.physical_residual[0]),
                    "row_qualified": bool(np.asarray(record.qualified)[0]),
                    "compensator_rule": compensator_rule_name(record.compensator_rule),
                    "normalised_unknown": _strict_float(record.normalized_unknown[0]),
                    "compensating_current_a": _strict_float(record.physical_unknown[0]),
                    "compensating_current_norm_a": float(
                        np.abs(record.physical_unknown[0])
                    ),
                    "direction_authority_row_scales_per_ampere": [
                        float(value)
                        for value in np.asarray(selection.direction_authority)
                    ],
                }
            )
            item.update(_walls(result))
            state = result.state
            program = result.program
            unknown = float(np.asarray(result.compensating_unknown)[0])
            try:
                achieved = _centroid(profile, result.state, target_current)
                item["achieved_centroid_m"] = achieved
                item["achieved_against_target_m"] = achieved - target
            except Exception as error:  # noqa: BLE001 - the solved state is the
                # label; a failed moment read only denies the achieved number
                item["achieved_centroid_m"] = None
                item["achieved_against_target_m"] = None
                item["centroid_read_failure"] = f"{type(error).__name__}: {error}"
        except Exception as error:  # noqa: BLE001 - one bad slice must not kill the chain
            item["solved"] = False
            item["solve_wall_s"] = time.perf_counter() - started
            item["failure"] = f"{type(error).__name__}: {error}"
            # re-enter the chain from the slice's own seed (see the free arm)
            state = jnp.asarray(seeds[row])
            unknown = None
        item["program_compiled"] = _compiled_flag(item)
        item["steady_wall_s"] = _warm_slice_wall(item, None)
        persist_arm("conditioned", index, item)
        print(
            "LABELLER-DONE "
            + json.dumps(
                {
                    k: item.get(k)
                    for k in ("row", "solved", "converged", "solve_wall_s")
                },
                sort_keys=True,
            ),
            flush=True,
        )

    # --- vmap probe over the keyframe program ---
    conditioned_program = program
    if KEYFRAME_SLICE not in table:
        receipt["vmap_probe"] = {
            "skipped": f"keyframe slice {KEYFRAME_SLICE} outside the driven window",
        }
    else:
        fixed_current = jnp.asarray(table[KEYFRAME_SLICE]["current_a"])
        external = operator.external(None, fixed_current)
        fixed_requested = jnp.asarray(
            _requested_class(group, KEYFRAME_SLICE), dtype=jnp.int8
        )
        fixed_target_current = abs(
            float(table[KEYFRAME_SLICE]["reference_plasma_current_a"])
        )
        coordinates = reduced_newton.reduced_coordinates(
            operator,
            seed_43,
            requested_class=fixed_requested,
            target_current=fixed_target_current,
        )
        probe, _kernels = _vmap_probe(
            operator,
            coordinates,
            external,
            fixed_requested,
            fixed_target_current,
            np.asarray(seed_43),
        )
        receipt["vmap_probe"] = probe
        _write_json(receipt, output)

    for arm in ("free", "conditioned"):
        slices = receipt["arms"][arm]["slices"]
        summary = _arm_rate_summary(slices)
        program_handle = free_program if arm == "free" else conditioned_program
        cache_sizes = _program_cache_sizes(program_handle)
        summary["program_cache_sizes"] = cache_sizes
        summary["program_compile_count"] = max(cache_sizes.values(), default=0)
        receipt["arms"][arm]["summary"] = summary
        trip_wall = summary["median_warm_trip_wall_s_global"]
        for item in (slice_ for slice_ in slices if slice_ is not None):
            item["steady_wall_s"] = _warm_slice_wall(item, trip_wall)

    free = receipt["arms"]["free"]["summary"]
    cond = receipt["arms"]["conditioned"]["summary"]

    def census_hours(per_second: float) -> dict[str, Any]:
        hours_one = CORPUS_CENSUS / (per_second * 3600.0) if per_second > 0 else None
        return {
            "census_slices": CORPUS_CENSUS,
            "hours_on_one_h200": hours_one,
            "hours_on_eight_h200": hours_one / 8.0 if hours_one is not None else None,
        }

    receipt["extrapolation"] = {
        "free_as_built": census_hours(free["as_built_slices_per_second"]),
        "free_warm": census_hours(free["warm_slices_per_second"]),
        "conditioned_as_built": census_hours(cond["as_built_slices_per_second"]),
        "conditioned_warm": census_hours(cond["warm_slices_per_second"]),
    }
    receipt["online_verdict"] = {
        "decode_requirement": (
            f"{DECODE_FRAMES_PER_SECOND_PER_PROCESS} decoded frames per second "
            "per process needs a label no slower than that many per process"
        ),
        "free_warm_slices_per_second": free["warm_slices_per_second"],
        "conditioned_warm_slices_per_second": cond["warm_slices_per_second"],
        "answer": (
            "the warm steady-state rate feeds training online only if decode "
            "demand stays below the label rate; the as-built rate with per-slice "
            "program compiles cannot, so the corpus is produced once ahead of "
            "training and written to file per shot"
        ),
    }
    receipt["verdict"] = {
        "slices_measured": len(SLICES),
        "free": {
            "converged_fraction": free["converged_fraction"],
            "qualified_fraction": free["qualified_fraction"],
            "as_built_slices_per_gpu_hour": free["as_built_slices_per_gpu_hour"],
            "slices_converged": free["slices_converged"],
        },
        "conditioned": {
            "converged_fraction": cond["converged_fraction"],
            "qualified_fraction": cond["qualified_fraction"],
            "as_built_slices_per_gpu_hour": cond["as_built_slices_per_gpu_hour"],
            "slices_converged": cond["slices_converged"],
        },
    }
    _write_json(receipt, output)
    _draw_figure(receipt, figure)
    _write_launcher(
        DEFAULT_LAUNCHER,
        revision=receipt["source_commit"],
        output=output,
        figure=figure,
    )
    return receipt


def main() -> None:
    """Parse the caller's operands and run the measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--cache-root", type=Path, default=None)
    arguments = parser.parse_args()
    measure(
        output=arguments.output,
        figure=arguments.figure,
        cache_root=arguments.cache_root,
    )


if __name__ == "__main__":
    main()
