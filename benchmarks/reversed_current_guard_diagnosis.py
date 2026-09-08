#!/usr/bin/env python3
"""Diagnose one reversed-plasma-current slice refused by the labeller.

On an ordinary positive-current shot the labeller's branch guard compares the
free solve's achieved current-centroid height against the EFIT reconstructed
height and accepts the slice when the signed difference stays inside a 50 mm
band.  On the reversed-current block every slice is recorded as non-converged
and non-qualified with the guard flag false, and the failure needs the test
that distinguishes three mechanisms before any repair is proposed:

* the solve fails to contract (the residual never reaches the convergence
  criterion),
* the solve converges to a different equilibrium (a different branch, so a
  different topology class or a centroid far from the EFIT target), or
* the solve contracts to the intended equilibrium but the guard measures it
  wrongly (the achieved-centroid read is refused or biased under the reversed
  current sign).

This driver builds the same shared operator, seed and slice inputs the batch
labeller uses, solves one slice freely, then reports — for the one slice —
the achieved centroid against the EFIT target with its signed error, the
residual trace across Newton trips, whether the achieved state admits the
requested topology class (and when refused, the O-point candidate census the
qualification consumed), whether the conditioned re-solve is entered and what
it does, and an explicit check of the guard's sign convention under the
reversed current.  No repair is made.

Run on a debug partition or the reserved card with a fresh TMPDIR:

    export TMPDIR=/tmp JAX_PLATFORMS=cpu
    python benchmarks/reversed_current_guard_diagnosis.py --shot 22475 --row 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np
import zarr

from benchmarks.efit_forward_parity_slice import FIXED_POINT_CRITERION
from benchmarks.forward_labeller_throughput import (
    NEWTON_STEPS,
    SHOT_STORE,
    _centroid_pair,
    _circuit_names,
    _requested_class,
    _slices_seed,
)
from nova.equilibrium import fixed_point, reduced_newton
from nova.equilibrium.observation import MomentIntegralSupport
from nova.equilibrium.topology import NoQualifiedAxisError, TopologyClass
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from scripts.labeller_batch.shard import (
    BRANCH_GUARD_TOLERANCE_M,
    _forward_receipt,
    _internal_geometry,
    _prepared_with_polarity,
    _shot_polarity,
    _slice_inputs,
    prepare_labeller,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/playable-forward-solve/reversed-current-guard/diagnosis.json"
)


def _host_float(value: Any) -> float | None:
    """Return one finite host float, or None where the value is not finite."""
    result = float(np.asarray(value))
    return result if np.isfinite(result) else None


def _source_revision() -> str:
    """Return the revision that supplied this driver."""
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _census_rows(block: Any) -> list[dict[str, Any]]:
    """Return host-printable (R, Z, flux, kind) rows from an O or X block."""
    array = np.asarray(block)
    rows = []
    for entry in array:
        if not np.all(np.isfinite(entry)):
            continue
        rows.append(
            {
                "r_m": round(float(entry[0]), 5),
                "z_m": round(float(entry[1]), 5),
                "flux_wb": round(float(entry[2]), 6),
                "kind": int(entry[3]),
            }
        )
    return rows


def _annotate_distances(
    rows: list[dict[str, Any]], point: Sequence[float]
) -> list[dict[str, Any]]:
    """Annotate census candidates with distance to one R-Z reference."""
    target = np.asarray(point, dtype=float)
    annotated = []
    for entry in rows:
        delta = np.asarray([entry["r_m"], entry["z_m"]], dtype=float) - target
        annotated.append(
            {**entry, "distance_to_reference_m": float(np.linalg.norm(delta))}
        )
    return annotated


def candidate_census(operator, state, reference: Sequence[float]) -> dict[str, Any]:
    """Return every O/X candidate the axis qualification consumes.

    Rows are (R, Z, flux, extremum kind), with the retained-valid flag the
    composed read filters on, so the reader can tell whether the finder placed
    any candidate near the reference at all, or whether the qualifier rejected
    the candidates it found.
    """
    physical = np.asarray(state)[: operator.physical_node_number]
    topology = operator._fixed_design_topology
    grid_flux, _wall_flux = topology.split_flux_map(jnp.asarray(physical))
    (_vmap_o, _vmap_x), census = topology.grid.read_census(grid_flux)
    candidate = np.asarray(census["retained_candidate"])
    valid = np.asarray(census["retained_valid"], dtype=int)
    return {
        "o_candidates": _annotate_distances(_census_rows(candidate[0]), reference),
        "o_retained_valid": [int(value) for value in valid[0]],
        "x_candidates": _annotate_distances(_census_rows(candidate[1]), reference),
        "x_retained_valid": [int(value) for value in valid[1]],
        "candidate_slots": int(candidate.shape[1]),
        "overflow": [
            bool(value) for value in np.asarray(census["overflow"]).reshape(-1)
        ],
    }


def _guard_convention(achieved_z: float | None, target_z: float) -> dict[str, Any]:
    """Check the branch guard's sign convention under reversed current.

    The writer's guard is ``abs(achieved - reference) <= 0.05``, which treats
    an error of either sign at equal tolerance; a sign error would have to
    enter through the achieved centroid itself, so that value is checked
    separately against the positive-weight centroid.
    """
    base = {
        "target_z_m": float(target_z),
        "guard_uses_absolute_value": True,
        "tolerance_m": BRANCH_GUARD_TOLERANCE_M,
    }
    if achieved_z is None or not np.isfinite(achieved_z):
        return {
            **base,
            "achieved_z_m": None,
            "error_m": None,
            "guard_ok": False,
            "guard_signed_lower": False,
            "guard_signed_upper": False,
            "note": "achieved centroid unavailable; the guard was never evaluated",
        }
    error = float(achieved_z - target_z)
    return {
        **base,
        "achieved_z_m": float(achieved_z),
        "error_m": error,
        "guard_ok": bool(abs(error) <= BRANCH_GUARD_TOLERANCE_M),
        "guard_signed_lower": bool(error >= -BRANCH_GUARD_TOLERANCE_M),
        "guard_signed_upper": bool(error <= BRANCH_GUARD_TOLERANCE_M),
        "note": (
            "the guard computes abs(error)<=tol, admitting both signs at equal "
            "tolerance; a reversed sign would have to enter through the "
            "achieved centroid, reported separately"
        ),
    }


def _read_admission(operator, state, requested_value: int | None) -> dict[str, Any]:
    """Return whether the achieved state admits one topology class.

    ``None`` requests the emergent read (no pinned class); the produced read
    is the same composed admission the centroid read and the writer's frame
    assembly consume.
    """
    try:
        masks, topology = operator.read(
            jnp.asarray(state), requested_class=requested_value
        )
        axis = np.asarray(topology.axis, dtype=float)
        x_point = np.asarray(topology.x_point, dtype=float)
        return {
            "admitted": True,
            "requested": None if requested_value is None else int(requested_value),
            "axis_r_m": _host_float(axis[0]),
            "axis_z_m": _host_float(axis[1]),
            "boundary_flux_wb": _host_float(topology.boundary_flux),
            "x_point_r_m": _host_float(x_point[0]),
            "x_point_z_m": _host_float(x_point[1]),
            "class_margin": (
                None if requested_value is None else _host_float(topology.class_margin)
            ),
            "core_cells": int(np.count_nonzero(np.asarray(masks.core))),
        }
    except Exception as error:
        return {
            "admitted": False,
            "requested": None if requested_value is None else int(requested_value),
            "error": f"{type(error).__name__}: {error}",
        }


def _frame_path_outcome(
    prepared,
    result,
    *,
    requested_value: int,
    target_current: float,
    applied_current,
    solve_wall_seconds: float,
) -> dict[str, Any]:
    """Report whether the writer's own frame construction accepts the result.

    The writer builds the steering frame from ``_forward_receipt`` followed by
    ``_internal_geometry``; a converged state that the read admits can still
    be dropped there when the boundary does not close inside the lattice, so
    this attempts the exact same two stages.
    """
    try:
        receipt = _forward_receipt(
            prepared,
            result,
            requested_class=jnp.asarray(requested_value, dtype=jnp.int8),
            target_current=target_current,
            prescribed_current=applied_current,
            solve_wall_seconds=solve_wall_seconds,
        )
        equilibrium = receipt.terminal_state
        geometry = _internal_geometry(
            prepared,
            equilibrium,
            diverted=requested_value == int(TopologyClass.DIVERTED),
        )
        return {
            "constructed": True,
            "boundary_flux_wb": _host_float(equilibrium.topology.boundary_flux),
            "n_surface": int(np.asarray(geometry.surface_psi_norm).size),
        }
    except Exception as error:
        return {
            "constructed": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _conditioning_description(conditioning: dict[str, Any]) -> str:
    """Describe in words what the conditioned re-solve did or why it stopped."""
    if "pair_exception" in conditioning:
        return (
            "the centroid-pin pair could not be derived from the seed "
            f"({conditioning['pair_exception']}), so the constrained solve "
            "never began"
        )
    if "solve_exception" in conditioning:
        return f"the conditioned solve raised ({conditioning['solve_exception']})"
    if conditioning.get("converged"):
        gap = conditioning.get("signed_error_m")
        clause = (
            f"and reached the EFIT centroid to within {abs(gap):.4f} m"
            if gap is not None
            else "but reports no finite achieved centroid"
        )
        return f"the conditioned solve converged {clause}"
    return (
        "the conditioned solve did not contract to the convergence criterion "
        f"(terminal residual {conditioning.get('terminal_residual')}, "
        f"termination {conditioning.get('termination')})"
    )


def conclusion(free: dict[str, Any]) -> str:
    """State which mechanism one slice exhibits."""
    guard = free["guard"]
    topology = free["topology"]
    conditioning = free["conditioning"]
    if not free["free_converged"]:
        base = (
            "the free solve fails to contract: terminal residual "
            f"{free['free_terminal_residual']}, termination "
            f"{free['free_termination']} after {free['free_trips']} trips"
        )
    elif not topology["requested"].get("admitted"):
        base = (
            "the free solve converges but its achieved state is refused by the "
            "topology read at the requested class "
            f"({topology['requested'].get('error')}); the guard is never evaluated"
        )
    elif guard["achieved_z_m"] is None:
        base = (
            "the free solve converges and the requested class is admitted, but "
            "the achieved centroid is not finite, so the guard is not evaluated"
        )
    elif not guard["guard_ok"]:
        base = (
            f"the free solve converges to an achieved centroid height "
            f"{guard['achieved_z_m']:+.4f} m against the EFIT target "
            f"{guard['target_z_m']:+.4f} m: a signed error of "
            f"{guard['error_m']:+.4f} m, outside the "
            f"{BRANCH_GUARD_TOLERANCE_M} m band"
        )
    else:
        base = "the free solve converges and the branch guard passes"
    parts = []
    if conditioning.get("entered"):
        parts.append(
            "the conditioned re-solve was entered and "
            + _conditioning_description(conditioning)
        )
    if topology["emergent"].get("admitted"):
        parts.append(
            "the achieved state also admits an emergent class at "
            f"({topology['emergent'].get('axis_r_m')}, "
            f"{topology['emergent'].get('axis_z_m')})"
        )
    return base + ("; " + "; ".join(parts) if parts else "") + "."


def _draw_figure(payload: dict[str, Any], output: Path, revision: str) -> None:
    """Compose the free-solve residual, centroid and census evidence panels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    free = payload["free"]
    trace = list(free["free_residual_trace"])
    axis = axes[0]
    if trace:
        axis.semilogy(
            range(1, len(trace) + 1), trace, "o-", color="#3b6ea5", ms=4, lw=1.1
        )
        axis.axhline(
            free["convergence_criterion"],
            color="black",
            ls="--",
            lw=0.9,
            label="convergence criterion",
        )
        axis.set_xlabel("active-set trip")
        axis.set_ylabel("relative residual")
        axis.legend(frameon=False, fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    else:
        axis.text(
            0.5,
            0.5,
            "no residual trace",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    axis = axes[1]
    target = free["efm"]["target_centroid_z_m"]
    achieved = free["guard"]["achieved_z_m"]
    tolerance = free["guard_tolerance_m"]
    axis.axhspan(target - tolerance, target + tolerance, color="#d9ead9", alpha=0.8)
    axis.axhline(target, color="#2e7d32", lw=1.2, label="EFIT target centroid z")
    if achieved is not None:
        axis.axhline(
            achieved,
            color="#a33b3b",
            lw=1.2,
            label=f"achieved centroid z ({achieved:+.4f} m)",
        )
    axis.set_xlim(0, 1)
    low = min(target - tolerance, achieved if achieved is not None else target)
    high = max(target + tolerance, achieved if achieved is not None else target)
    axis.set_ylim(low - 0.02, high + 0.02)
    axis.set_xticks([])
    axis.set_ylabel("height z [m]")
    axis.set_title(
        f"achieved vs EFIT target z: {_error_label(achieved, target)}", fontsize=9
    )
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    axis.grid(alpha=0.2)

    axis = axes[2]
    census = free.get("candidate_census")
    if census:
        rows = census["o_candidates"]
        if rows:
            axis.scatter(
                [item["r_m"] for item in rows],
                [item["z_m"] for item in rows],
                c="#6b6b6b",
                s=46,
                marker="D",
                label="O-point candidates",
            )
        efm = free["efm"]
        axis.scatter(
            [efm["efm_axis_r_m"]],
            [efm["efm_axis_z_m"]],
            c="#2e7d32",
            s=80,
            marker="*",
            edgecolors="black",
            linewidths=0.6,
            label="EFIT magnetic axis",
        )
        axis.set_xlabel("R [m]")
        axis.set_ylabel("Z [m]")
        axis.set_title("O-point census on the converged state", fontsize=9)
        axis.legend(frameon=False, fontsize=7, loc="best")
        axis.grid(alpha=0.2)
    else:
        axis.text(
            0.5,
            0.5,
            "census not published\n(centroid read succeeded)",
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=9,
        )
    figure.suptitle(
        f"Reversed-current guard diagnosis — MAST {payload['identity']} "
        f"@{revision[:8]}",
        y=0.98,
    )
    figure.subplots_adjust(left=0.06, right=0.99, bottom=0.14, top=0.84, wspace=0.34)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _error_label(achieved: float | None, target: float) -> str:
    """Return one signed-centroid-error label for the figure."""
    if achieved is None:
        return "unavailable"
    return f"{achieved - target:+.4f} m"


def _centroid_observation(
    profile, state, *, requested_value: int, target_current: float
) -> dict[str, Any]:
    """Read the achieved current centroid exactly as the writer does."""
    try:
        observation = profile.current_moment_observation(
            jnp.asarray(state),
            support=MomentIntegralSupport.ALL_DOMAIN,
            requested_class=requested_value,
            target_current=target_current,
        )
        return {
            "achieved_r_m": _host_float(observation.centroid_r),
            "achieved_z_m": _host_float(observation.centroid_z),
            "plasma_current_A": _host_float(observation.plasma_current),
        }
    except Exception as error:
        return {"exception": f"{type(error).__name__}: {error}"}


def _absolute_weighted_centroid(
    profile, state, *, requested_value: int, target_current: float
) -> dict[str, Any]:
    """Return the positive-weight centroid used as the guard cross-check.

    A single-signed plasma current makes the signed and absolute-weighted
    centroids coincide; any gap reveals mixed-sign current inside the support
    that the signed read blends, which is the guard measuring the wrong value.
    """
    try:
        state_data = profile._integral_state(
            jnp.asarray(state),
            requested_class=requested_value,
            target_current=target_current,
        )[0]
        cell = np.asarray(state_data.cell_current, dtype=np.float64)
        coordinate = np.asarray(profile.operator.grid.coordinate, dtype=np.float64)
        total = np.abs(cell).sum()
        if total <= 0.0:
            return {"exception": "zero absolute plasma current"}
        centroid = np.sum(np.abs(cell)[:, None] * coordinate, axis=0) / total
        return {"r_m": float(centroid[0]), "z_m": float(centroid[1])}
    except Exception as error:
        return {"exception": f"{type(error).__name__}: {error}"}


def main(argv: Sequence[str] | None = None) -> int:
    """Drive one reversed-current slice diagnosis without repairing anything."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, default=22475)
    parser.add_argument("--row", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path)
    arguments = parser.parse_args(argv)
    output = arguments.output.resolve()
    figure_output = (
        arguments.figure.resolve()
        if arguments.figure is not None
        else output.with_suffix(".png")
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    setup_started = time.perf_counter()
    prepared_base = prepare_labeller()
    prepared = _prepared_with_polarity(
        prepared_base, int(_shot_polarity(arguments.shot))
    )
    setup_wall = time.perf_counter() - setup_started
    profile = prepared.profile
    operator = profile.operator

    group = zarr.open_group(str(SHOT_STORE / f"{arguments.shot}.zarr"), mode="r")["efm"]
    full_r = np.asarray(group["gridr"], dtype=np.float64)
    full_z = np.asarray(group["gridz"], dtype=np.float64)
    inputs = _slice_inputs(group, arguments.row)
    if inputs is None:
        raise ValueError("selected slice carries no finite reconstruction")
    seed = _slices_seed(group, arguments.row, full_r, full_z)
    requested_value = _requested_class(group, arguments.row)
    requested = jnp.asarray(requested_value, dtype=jnp.int8)
    target_current = abs(inputs["reference_plasma_current"])
    current = np.asarray(inputs["current"], dtype=np.float64)
    efm_reference = (
        float(np.asarray(group["magnetic_axis_r"][arguments.row])),
        float(np.asarray(group["magnetic_axis_z"][arguments.row])),
    )

    # 1. The free solve, with the reversed-polarity operator.
    free_started = time.perf_counter()
    free_result = reduced_newton.solve_reduced_newton(
        operator,
        seed,
        requested_class=requested,
        target_current=target_current,
        prescribed_current=current,
        tolerance=FIXED_POINT_CRITERION,
        newton_steps=NEWTON_STEPS,
        program=None,
        stream=False,
    )
    free_wall = time.perf_counter() - free_started
    free = {
        "free_converged": bool(free_result.converged),
        "free_terminal_residual": _host_float(free_result.terminal_residual),
        "free_termination": fixed_point.FixedPointTerminationReason(
            int(free_result.termination_reason)
        ).name,
        "free_trips": int(free_result.active_set_iterations),
        "free_newton_steps_per_trip": [
            int(value) for value in np.asarray(free_result.newton_steps_per_trip)
        ],
        "free_residual_trace": [
            float(value) for value in np.asarray(free_result.active_set_residuals)
        ],
        "free_wall_seconds": free_wall,
        "requested_class": int(requested_value),
        "requested_class_name": TopologyClass(requested_value).name,
        "convergence_criterion": FIXED_POINT_CRITERION,
        "guard_tolerance_m": BRANCH_GUARD_TOLERANCE_M,
        "efm": {
            "target_centroid_z_m": float(inputs["target_centroid_z"]),
            "efm_axis_r_m": efm_reference[0],
            "efm_axis_z_m": efm_reference[1],
            "reference_plasma_current_A": float(inputs["reference_plasma_current"]),
        },
    }

    # 2. The achieved centroid, the guard its sign convention.
    centroid = _centroid_observation(
        profile,
        free_result.state,
        requested_value=requested_value,
        target_current=target_current,
    )
    free["centroid_read"] = centroid
    if "exception" in centroid:
        reference = (
            float(np.asarray(group["current_centrd_r"][arguments.row])),
            float(inputs["target_centroid_z"]),
        )
        free["candidate_census"] = candidate_census(
            operator, free_result.state, reference
        )
        free["guard"] = _guard_convention(None, float(inputs["target_centroid_z"]))
    else:
        free["guard"] = _guard_convention(
            centroid["achieved_z_m"], float(inputs["target_centroid_z"])
        )
        absolute = _absolute_weighted_centroid(
            profile,
            free_result.state,
            requested_value=requested_value,
            target_current=target_current,
        )
        free["absolute_weighted_centroid"] = absolute
        if "exception" not in absolute:
            signed = np.asarray([centroid["achieved_r_m"], centroid["achieved_z_m"]])
            absolute_rz = np.asarray([absolute["r_m"], absolute["z_m"]])
            free["guard"]["absolute_weighted_centroid_gap_m"] = float(
                np.linalg.norm(signed - absolute_rz)
            )
            free["guard"]["signed_minus_absolute_z_gap_m"] = float(
                centroid["achieved_z_m"] - absolute["z_m"]
            )

    # 3. The achieved topology class against the requested one.
    free["topology"] = {
        "requested": _read_admission(operator, free_result.state, requested_value),
        "emergent": _read_admission(operator, free_result.state, None),
    }

    # 4. The conditioned re-solve, entered exactly when the writer enters it.
    guard_ok = free["guard"]["guard_ok"]
    should_condition = not bool(free_result.converged) or not bool(guard_ok)
    conditioning: dict[str, Any] = {
        "entered": False,
        "why": None,
    }
    if should_condition:
        conditioning["entered"] = True
        conditioning["why"] = (
            "free solve not converged"
            if not bool(free_result.converged)
            else f"free branch guard false ({free['guard']['note']})"
        )
        names = _circuit_names(prepared.policy_evidence)
        try:
            pair, selection = _centroid_pair(
                profile,
                seed,
                target=float(inputs["target_centroid_z"]),
                unknown=None,
                target_current=target_current,
                requested=requested_value,
                names=names,
            )
        except NoQualifiedAxisError as error:
            conditioning["pair_exception"] = f"NoQualifiedAxisError: {error}"
            pair = None
        except Exception as error:
            conditioning["pair_exception"] = f"{type(error).__name__}: {error}"
            pair = None
        if pair is not None:
            t0 = time.perf_counter()
            try:
                conditioned = reduced_newton.solve_constrained_reduced_newton(
                    profile,
                    seed,
                    constraint_pairs=(pair,),
                    requested_class=requested,
                    target_current=target_current,
                    prescribed_current=current,
                    tolerance=FIXED_POINT_CRITERION,
                    newton_steps=NEWTON_STEPS,
                    program=None,
                    stream=False,
                )
                conditioning["conditioned_wall_seconds"] = time.perf_counter() - t0
                conditioning["converged"] = bool(conditioned.converged)
                conditioning["terminal_residual"] = _host_float(
                    conditioned.terminal_residual
                )
                conditioning["termination"] = fixed_point.FixedPointTerminationReason(
                    int(conditioned.termination_reason)
                ).name
                conditioning["trips"] = int(conditioned.active_set_iterations)
                conditioning["residual_trace"] = [
                    float(value)
                    for value in np.asarray(conditioned.active_set_residuals)
                ]
                conditioned_centroid = _centroid_observation(
                    profile,
                    conditioned.state,
                    requested_value=requested_value,
                    target_current=target_current,
                )
                conditioning["centroid_read"] = conditioned_centroid
                if "achieved_z_m" in conditioned_centroid:
                    gap = conditioned_centroid["achieved_z_m"] - float(
                        inputs["target_centroid_z"]
                    )
                    conditioning["signed_error_m"] = float(gap)
                    conditioning["guard_ok"] = bool(
                        abs(gap) <= BRANCH_GUARD_TOLERANCE_M
                    )
                conditioning["topology_admission"] = _read_admission(
                    operator, conditioned.state, requested_value
                )
            except Exception as error:
                conditioning["solve_exception"] = f"{type(error).__name__}: {error}"
    free["conditioning"] = conditioning

    # 5. The writer's own frame path on the selected result.
    applied_current = (
        np.asarray(free_result.prescribed_current)
        if getattr(free_result, "prescribed_current", None) is not None
        else current
    )
    free["frame_path"] = _frame_path_outcome(
        prepared,
        free_result,
        requested_value=requested_value,
        target_current=target_current,
        applied_current=applied_current,
        solve_wall_seconds=free_wall,
    )

    payload = {
        "identity": f"{arguments.shot}/{arguments.row}",
        "polarity": int(_shot_polarity(arguments.shot)),
        "target_centroid_source": "efm/current_centrd_z",
        "setup_wall_seconds": setup_wall,
        "compilation_cache_directory": str(cache.directory),
        "revision": _source_revision(),
        "host": subprocess.run(
            ["hostname"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "conclusion": conclusion(free),
        "free": free,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _draw_figure(payload, figure_output, _source_revision())
    print("=== DIAGNOSIS RECEIPT ===")
    print(json.dumps(payload, indent=2, default=str))
    print("=== END ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
