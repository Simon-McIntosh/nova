"""Measure the reduced-state plain-Newton prototype on the four bank rows.

Each row is solved from the same persisted pure-arm seed through three
prototype routes: the batched-tail ladder behind a fused trip boundary, the
serial per-grade ladder behind the same boundary, and the eager ladder behind
the dispatched boundary the first prototype measured.  The eager route is what
the banked receipt beside this one recorded, so it re-derives that receipt's
terminal flux in the same job and the fast route's terminal flux is compared
against it row by row; the serial route is the route the batched tail
replaces, so the two stand side by side in the same job and the batched tail
is judged a strict improvement or not on the rows that refuse grades.
Optionally each row is also solved through the production public route,
``ForwardProfile.solve_branch(newton_krylov)``.

Every arm reports the terminal residual, the trip count, the Newton steps and
map evaluations per trip, and the wall per step, per boundary and per trip, so
the prototype's distance to a millisecond fixed point is read against the
route it replaces rather than against a projection.  Every row is written to
the receipt as it finishes, so a job that runs out of wall clock still
delivers the rows it measured.
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
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    FIXED_POINT_CRITERION,
    NEWTON_STEPS,
    RELAXATION,
    STEP_CAP,
    WARMUP_SWEEPS,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium import reduced_newton
from nova.equilibrium.fixed_point import (
    FixedPointTerminationReason,
    _relative_residual,
)
from nova.equilibrium.forward import PerturbedSeedPolicy
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ((21985, 51), (21986, 46), (21989, 55), (22086, 43))
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/millisecond-converged-solve/reduced-newton/four-rows.json"
)
PRODUCTION_RECEIPT = (
    ROOT
    / "docs/figures/solver-convergence-regression"
    / "settled-mask-stall/fallback-carry/four-rows-after.json"
)
#: The receipt this measurement is read against, copied from the revision that
#: banked it so a rewrite of the live receipt cannot move the comparison.
PREVIOUS_RECEIPT = (
    ROOT
    / "docs/figures/millisecond-converged-solve/reduced-newton/four-rows-before.json"
)
#: Terminal-flux agreement required between the two prototype routes, as a
#: fraction of the flux span.  A converged row sits at the map's fixed point
#: and the routes may differ only by the arithmetic the fusion reassociates; a
#: settled row stops on an unmoved mask at a finite residual, where the same
#: reassociation is carried through the trips that follow it.
CONVERGED_FLUX_AGREEMENT = 1.0e-9
SETTLED_FLUX_AGREEMENT = 1.0e-6
#: Every prototype route, named by the mechanism each measures, in the order
#: they are solved: a job that runs out of wall clock delivers the batched
#: tail and the route it replaces before it reaches the eager reference.
PROTOTYPE_ROUTES = {
    "prototype": {
        "ladder_scoring": reduced_newton.LADDER_SCORING,
        "trip_boundary": reduced_newton.TRIP_BOUNDARY,
    },
    "serial_prototype": {
        "ladder_scoring": reduced_newton.SERIAL_LADDER_SCORING,
        "trip_boundary": reduced_newton.TRIP_BOUNDARY,
    },
    "eager_prototype": {
        "ladder_scoring": reduced_newton.EAGER_LADDER_SCORING,
        "trip_boundary": reduced_newton.DISPATCHED_TRIP_BOUNDARY,
    },
}
PUBLIC_ROUTE_POLICY = PerturbedSeedPolicy()


def _source_revision() -> str:
    """Return the revision this measurement runs from."""
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _banked_production_rows() -> dict[str, dict[str, Any]]:
    """Return the merged production receipt this prototype is measured beside."""
    if not PRODUCTION_RECEIPT.exists():
        return {}
    receipt = json.loads(PRODUCTION_RECEIPT.read_text())
    return {row["identity"]: row["public_route"] for row in receipt["rows"]}


def _previous_prototype_rows() -> dict[str, dict[str, Any]]:
    """Return the prototype figures the receipt beside this one banked."""
    if not PREVIOUS_RECEIPT.exists():
        return {}
    receipt = json.loads(PREVIOUS_RECEIPT.read_text())
    return {row["identity"]: row["prototype"] for row in receipt["rows"]}


def _route_agreement(fast, eager, previous) -> dict[str, Any]:
    """Compare the two prototype routes, and the slow one against its receipt.

    The eager route re-derives what the previous receipt measured, so its
    terminal residual and trip census standing beside the banked ones say
    whether the reference reproduced; the fast route's terminal flux is then
    compared against that reference rather than against a projection.
    """
    fast_flux = jnp.asarray(fast["flux"])
    eager_flux = jnp.asarray(eager["flux"])
    span = float(jnp.max(jnp.abs(eager_flux)))
    difference = float(jnp.max(jnp.abs(fast_flux - eager_flux)))
    fraction = difference / max(span, 1.0e-30)
    required = (
        CONVERGED_FLUX_AGREEMENT if eager["converged"] else SETTLED_FLUX_AGREEMENT
    )
    reproduced = None
    if previous is not None:
        banked = float(previous["terminal_residual"])
        reproduced = {
            "banked_terminal_residual": banked,
            "measured_terminal_residual": eager["terminal_residual"],
            "relative_residual_difference": abs(eager["terminal_residual"] - banked)
            / max(abs(banked), 1.0e-30),
            "banked_active_set_iterations": previous["active_set_iterations"],
            "measured_active_set_iterations": eager["active_set_iterations"],
            "banked_newton_steps_per_trip": previous["newton_steps_per_trip"],
            "measured_newton_steps_per_trip": eager["newton_steps_per_trip"],
            "identical_trip_census": (
                previous["active_set_iterations"] == eager["active_set_iterations"]
                and previous["newton_steps_per_trip"] == eager["newton_steps_per_trip"]
                and previous["active_set_mask_differences"]
                == eager["active_set_mask_differences"]
            ),
        }
    return {
        "reference_route": "eager ladder behind the dispatched boundary",
        "sup_flux_difference_wb": difference,
        "sup_flux_difference_fraction_of_span": fraction,
        "required_fraction_of_span": required,
        "flux_agrees": fraction <= required,
        "bitwise_identical_flux": bool(jnp.array_equal(fast_flux, eager_flux)),
        "terminal_residual_fast": fast["terminal_residual"],
        "terminal_residual_reference": eager["terminal_residual"],
        "identical_trip_census": (
            fast["active_set_iterations"] == eager["active_set_iterations"]
            and fast["newton_steps_per_trip"] == eager["newton_steps_per_trip"]
            and fast["active_set_mask_differences"]
            == eager["active_set_mask_differences"]
        ),
        "previous_receipt_reproduced": reproduced,
    }


def _median(values) -> float | None:
    """Return the median of one sample, or None where the sample is empty."""
    return float(np.median(values)) if len(values) else None


def _difference(left, right) -> float | None:
    """Return one signed wall difference, or None where a figure is absent."""
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _tail_comparison(fast, serial, eager) -> dict[str, Any]:
    """Read the batched refused tail against the routes it stands between.

    A refused step is where the batching acts: the serial route dispatches
    every remaining grade and the eager route scores them all in one
    ``lax.map``, so the batched tail is a strict improvement only if its
    refused step lands at the eager route's wall while its accepting step
    keeps the serial route's.  Every decision is compared against the serial
    route for exact equality, because the tail must move round trips and
    nothing else.
    """
    fast_flux = jnp.asarray(fast["flux"])
    serial_flux = jnp.asarray(serial["flux"])
    return {
        "reference_route": "serial per-grade ladder behind the fused boundary",
        "refused_step_wall_s": fast["median_warm_refused_step_wall_s"],
        "serial_refused_step_wall_s": serial["median_warm_refused_step_wall_s"],
        "eager_refused_step_wall_s": eager["median_warm_refused_step_wall_s"],
        "refused_step_wall_over_eager_s": _difference(
            fast["median_warm_refused_step_wall_s"],
            eager["median_warm_refused_step_wall_s"],
        ),
        "refused_step_wall_recovered_from_serial_s": _difference(
            serial["median_warm_refused_step_wall_s"],
            fast["median_warm_refused_step_wall_s"],
        ),
        "accepting_step_wall_s": fast["median_warm_accepting_step_wall_s"],
        "serial_accepting_step_wall_s": serial["median_warm_accepting_step_wall_s"],
        "accepting_step_wall_over_serial_s": _difference(
            fast["median_warm_accepting_step_wall_s"],
            serial["median_warm_accepting_step_wall_s"],
        ),
        "warm_refused_step_count": fast["warm_refused_step_count"],
        "warm_accepting_step_count": fast["warm_accepting_step_count"],
        "identical_decisions": (
            fast["active_set_iterations"] == serial["active_set_iterations"]
            and fast["newton_steps_per_trip"] == serial["newton_steps_per_trip"]
            and fast["active_set_mask_differences"]
            == serial["active_set_mask_differences"]
            and fast["rejected_steps_per_trip"] == serial["rejected_steps_per_trip"]
            and fast["termination_reason"] == serial["termination_reason"]
            and [step["accepted_factor"] for step in fast["steps"]]
            == [step["accepted_factor"] for step in serial["steps"]]
        ),
        "bitwise_identical_flux": bool(jnp.array_equal(fast_flux, serial_flux)),
        "sup_flux_difference_wb": float(jnp.max(jnp.abs(fast_flux - serial_flux))),
        "identical_terminal_residual": (
            fast["terminal_residual"] == serial["terminal_residual"]
        ),
    }


def _speedup(fast, eager) -> dict[str, Any]:
    """Rank what the two repairs removed from the warm step and the trip."""

    def ratio(numerator, denominator):
        """Return one before-after ratio, or None where a figure is absent."""
        if not numerator or not denominator:
            return None
        return float(numerator) / float(denominator)

    return {
        "warm_step_wall_s": fast["median_warm_step_wall_s"],
        "reference_warm_step_wall_s": eager["median_warm_step_wall_s"],
        "warm_step_speedup": ratio(
            eager["median_warm_step_wall_s"], fast["median_warm_step_wall_s"]
        ),
        "warm_trip_wall_s": fast["median_warm_trip_wall_s"],
        "reference_warm_trip_wall_s": eager["median_warm_trip_wall_s"],
        "warm_trip_speedup": ratio(
            eager["median_warm_trip_wall_s"], fast["median_warm_trip_wall_s"]
        ),
        "warm_boundary_wall_s": fast["median_warm_boundary_wall_s"],
        "reference_warm_boundary_wall_s": eager["median_warm_boundary_wall_s"],
        "warm_newton_work_per_trip_s": fast["median_warm_newton_work_per_trip_s"],
        "reference_warm_newton_work_per_trip_s": eager[
            "median_warm_newton_work_per_trip_s"
        ],
        "warm_step_map_evaluations": fast["median_warm_step_map_evaluations"],
        "reference_warm_step_map_evaluations": eager[
            "median_warm_step_map_evaluations"
        ],
        "millisecond_target_s": 1.0e-3,
        "warm_step_walls_of_target": ratio(fast["median_warm_step_wall_s"], 1.0e-3),
    }


def _production_arm(profile, seed, target_current) -> dict[str, Any]:
    """Time one production public-route solve of a bank seed."""
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    started = time.perf_counter()
    branch = profile.solve_branch(
        seed,
        requested,
        route="newton_krylov",
        target_current=target_current,
        tolerance=FIXED_POINT_CRITERION,
        newton_steps=NEWTON_STEPS,
        gmres_iterations=PUBLIC_ROUTE_POLICY.gmres_iterations,
        warmup=WARMUP_SWEEPS,
        relaxation=RELAXATION,
        step_cap=STEP_CAP,
    )
    branch.equilibrium.flux.block_until_ready()
    wall = time.perf_counter() - started
    result = branch.equilibrium.fixed_point
    trips = int(np.asarray(result.active_set_iterations))
    return {
        "route": "ForwardProfile.solve_branch(newton_krylov)",
        "gmres_iterations": PUBLIC_ROUTE_POLICY.gmres_iterations,
        "newton_steps": NEWTON_STEPS,
        "wall_s": wall,
        "wall_per_trip_s": wall / max(trips, 1),
        "terminal_residual": float(np.asarray(result.residual)),
        "active_set_iterations": trips,
        "converged": bool(np.asarray(result.converged)),
        "termination_reason": FixedPointTerminationReason(
            int(np.asarray(result.termination_reason))
        ).name.lower(),
        "active_set_residuals": [
            float(value) for value in np.asarray(result.active_set_residuals)[:trips]
        ],
        "active_set_mask_differences": [
            int(value)
            for value in np.asarray(result.active_set_mask_differences)[:trips]
        ],
        "flux": np.asarray(branch.equilibrium.flux),
    }


def _prototype_arm(
    operator, seed, target_current, policy: str, route: dict[str, str]
) -> dict[str, Any]:
    """Time one reduced-state plain-Newton solve of the same bank seed."""
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    started = time.perf_counter()
    result = reduced_newton.solve_reduced_newton(
        operator,
        seed,
        requested_class=requested,
        target_current=target_current,
        tolerance=FIXED_POINT_CRITERION,
        newton_steps=NEWTON_STEPS,
        support_policy=policy,
        stream=True,
        **route,
    )
    wall = time.perf_counter() - started
    accepted = [step for step in result.steps if step.accepted_factor is not None]
    step_walls = [step.wall_s for step in result.steps]
    warm_steps = [step for step in result.steps if step.trip > 0]
    warm_trips = result.trip_wall_per_trip[1:]
    warm_boundaries = result.boundary_wall_per_trip[1:]
    warm_newton = result.newton_wall_per_trip[1:]
    warm_accepting = [step for step in warm_steps if step.accepted_factor is not None]
    warm_refused = [step for step in warm_steps if step.accepted_factor is None]
    return {
        "route": "reduced_newton.solve_reduced_newton",
        "ladder_scoring": route["ladder_scoring"],
        "trip_boundary": route["trip_boundary"],
        "support_policy": policy,
        "reduced_dimension": result.reduced_dimension,
        "support_cells": result.support_cells,
        "off_support_leakage": result.off_support_leakage,
        "wall_s": wall,
        "wall_per_trip_s": wall / max(result.active_set_iterations, 1),
        "terminal_residual": result.terminal_residual,
        "active_set_iterations": result.active_set_iterations,
        "converged": result.converged,
        "termination_reason": result.termination_name,
        "active_set_residuals": result.active_set_residuals,
        "active_set_mask_differences": result.active_set_mask_differences,
        "newton_steps_per_trip": result.newton_steps_per_trip,
        "jacobian_builds_per_trip": result.jacobian_builds_per_trip,
        "rejected_steps_per_trip": result.rejected_steps_per_trip,
        "map_evaluations_per_trip": result.map_evaluations_per_trip,
        "jacobian_wall_per_trip_s": result.jacobian_wall_per_trip,
        "newton_wall_per_trip_s": result.newton_wall_per_trip,
        "boundary_wall_per_trip_s": result.boundary_wall_per_trip,
        "trip_wall_per_trip_s": result.trip_wall_per_trip,
        "accepted_step_count": len(accepted),
        "rejected_step_count": len(result.steps) - len(accepted),
        "first_step_wall_s": step_walls[0] if step_walls else None,
        "median_warm_step_wall_s": (
            float(np.median([step.wall_s for step in warm_steps]))
            if warm_steps
            else None
        ),
        "first_jacobian_wall_s": (
            result.jacobian_wall_per_trip[0] if result.jacobian_wall_per_trip else None
        ),
        "median_warm_jacobian_wall_s": (
            float(np.median(result.jacobian_wall_per_trip[1:]))
            if len(result.jacobian_wall_per_trip) > 1
            else None
        ),
        "median_warm_trip_wall_s": (
            float(np.median(warm_trips)) if warm_trips else None
        ),
        "median_warm_boundary_wall_s": (
            float(np.median(warm_boundaries)) if warm_boundaries else None
        ),
        "median_warm_newton_work_per_trip_s": (
            float(np.median(warm_newton)) if warm_newton else None
        ),
        "median_warm_step_map_evaluations": (
            float(np.median([step.map_evaluations for step in warm_steps]))
            if warm_steps
            else None
        ),
        "warm_accepting_step_count": len(warm_accepting),
        "warm_refused_step_count": len(warm_refused),
        "median_warm_accepting_step_wall_s": _median(
            [step.wall_s for step in warm_accepting]
        ),
        "median_warm_refused_step_wall_s": _median(
            [step.wall_s for step in warm_refused]
        ),
        "median_warm_refused_step_map_evaluations": _median(
            [float(step.map_evaluations) for step in warm_refused]
        ),
        "steps": [
            {
                "trip": step.trip,
                "step": step.step,
                "flux_residual": step.flux_residual,
                "reduced_residual": step.reduced_residual,
                "merit": step.merit,
                "accepted_factor": step.accepted_factor,
                "grades_tried": step.grades_tried,
                "map_evaluations": step.map_evaluations,
                "jacobian_refreshed": step.jacobian_refreshed,
                "wall_s": step.wall_s,
            }
            for step in result.steps
        ],
        "flux": np.asarray(result.state),
    }


def _fixed_point_agreement(operator, production, prototype, target_current):
    """Compare both terminal fluxes and both fixed-point residuals."""
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    mapped = operator.flux_map(None, requested, target_current, None)
    production_flux = jnp.asarray(production["flux"])
    prototype_flux = jnp.asarray(prototype["flux"])
    span = float(jnp.max(jnp.abs(production_flux)))
    difference = float(jnp.max(jnp.abs(prototype_flux - production_flux)))
    return {
        "sup_flux_difference_wb": difference,
        "sup_flux_difference_fraction_of_span": difference / max(span, 1.0e-30),
        "production_free_map_residual": float(
            _relative_residual(mapped(production_flux), production_flux)
        ),
        "prototype_free_map_residual": float(
            _relative_residual(mapped(prototype_flux), prototype_flux)
        ),
        "both_converged": bool(production["converged"] and prototype["converged"]),
    }


def _draw(rows: list[dict[str, Any]], figure_path: Path) -> None:
    """Plot the warm step, the warm trip and the map evaluations, before/after."""
    figure, axes = plt.subplots(1, 4, figsize=(18.0, 4.4))
    identities = [row["identity"] for row in rows]
    index = np.arange(len(rows))
    before = "#c0504d"
    middle = "#f0a04b"
    after = "#4f81bd"

    def series(name, field_name):
        """Return one per-row figure of one prototype route."""
        return [row[name].get(field_name) or np.nan for row in rows]

    axes[0].bar(
        index - 0.2,
        series("eager_prototype", "median_warm_step_wall_s"),
        0.4,
        label="eager ladder, dispatched boundary",
        color=before,
    )
    axes[0].bar(
        index + 0.2,
        series("prototype", "median_warm_step_wall_s"),
        0.4,
        label="batched tail, fused boundary",
        color=after,
    )
    axes[0].axhline(1.0e-3, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("median warm Newton step wall [s]")
    axes[0].set_title("per Newton step against 1 ms")
    axes[0].legend(fontsize=7)

    axes[1].bar(
        index - 0.2,
        series("eager_prototype", "median_warm_trip_wall_s"),
        0.4,
        label="dispatched boundary",
        color=before,
    )
    axes[1].bar(
        index + 0.2,
        series("prototype", "median_warm_trip_wall_s"),
        0.4,
        label="fused boundary",
        color=after,
    )
    axes[1].bar(
        index + 0.2,
        series("prototype", "median_warm_newton_work_per_trip_s"),
        0.4,
        label="Newton work inside the fused trip",
        color="#9bbb59",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("median warm trip wall [s]")
    axes[1].set_title("per active-set trip")
    axes[1].legend(fontsize=7)

    axes[2].bar(
        index - 0.2,
        series("eager_prototype", "median_warm_step_map_evaluations"),
        0.4,
        label="eager ladder",
        color=before,
    )
    axes[2].bar(
        index + 0.2,
        series("prototype", "median_warm_step_map_evaluations"),
        0.4,
        label="batched tail",
        color=after,
    )
    axes[2].set_ylabel("map evaluations per warm Newton step")
    axes[2].set_title("map evaluations a step pays for")
    axes[2].legend(fontsize=7)

    for offset, name, label, colour in (
        (-0.27, "eager_prototype", "eager ladder", before),
        (0.0, "serial_prototype", "serial per-grade ladder", middle),
        (0.27, "prototype", "batched tail", after),
    ):
        axes[3].bar(
            index + offset,
            series(name, "median_warm_refused_step_wall_s"),
            0.26,
            label=label,
            color=colour,
        )
    axes[3].plot(
        index,
        series("prototype", "median_warm_accepting_step_wall_s"),
        "k_",
        markersize=14,
        label="batched tail, accepting step",
    )
    axes[3].axhline(1.0e-3, color="black", linestyle="--", linewidth=1.0)
    axes[3].set_yscale("log")
    axes[3].set_ylabel("median warm refused step wall [s]")
    axes[3].set_title("a refused grade ladder, per route")
    axes[3].legend(fontsize=7)

    for axis in axes:
        axis.set_xticks(index)
        axis.set_xticklabels(identities, rotation=20, fontsize=8)
        axis.grid(True, axis="y", alpha=0.3)
    figure.suptitle(
        "Reduced-state plain Newton: a batched refused grade tail behind a fused trip"
        " boundary"
    )
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)


def _write(receipt: dict[str, Any], output: Path) -> None:
    """Persist the receipt so far, atomically, after every row."""
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = output.with_suffix(".json.partial")
    scratch.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    scratch.replace(output)


def measure(
    *,
    output: Path,
    policy: str,
    production: bool,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Solve the four bank rows through both routes and persist as they land."""
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        cache_root
        if cache_root is not None
        else default_persistent_compilation_cache_root()
    )
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    }
    banked_production = _banked_production_rows()
    previous_prototype = _previous_prototype_rows()
    rows: list[dict[str, Any]] = []
    indexed: dict[str, dict[str, Any]] = {}
    terminal_flux: dict[str, np.ndarray] = {}
    receipt = {
        "artifact": "reduced-state plain-Newton prototype on the four bank rows",
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
            "production_receipt": str(PRODUCTION_RECEIPT),
            "previous_receipt": str(PREVIOUS_RECEIPT),
        },
        "measurement_contract": {
            "targets": [list(key) for key in TARGETS],
            "seed": "persisted pure-arm bank seed for each row",
            "production_route": "ForwardProfile.solve_branch(newton_krylov)",
            "prototype_route": "reduced_newton.solve_reduced_newton",
            "prototype_routes": PROTOTYPE_ROUTES,
            "support_policy": policy,
            "convergence_tolerance": FIXED_POINT_CRITERION,
            "newton_steps": NEWTON_STEPS,
            "production_arm_measured_here": production,
            "reference_arm": (
                "the eager ladder behind the dispatched boundary is the route "
                "the previous receipt measured, re-derived in this job so the "
                "fast route's terminal flux is compared against a terminal "
                "flux measured here rather than against a stored scalar; the "
                "serial per-grade ladder behind the same fused boundary is "
                "the route the batched tail replaces, measured beside it so "
                "the refused step's recovered wall and the unmoved decisions "
                "are read against a route measured in the same job"
            ),
            "residual_norm_source": (
                "the reduced residual sup norm each step reports is returned "
                "by the scoring kernel that already holds the residual, not "
                "reduced in a dispatch of its own"
            ),
            "pass_order": (
                "both prototype routes of a row land before the next row, and "
                "every production row lands after all of them, so a job that "
                "runs out of wall clock still delivers the arms it was "
                "dispatched to measure"
            ),
        },
        "rows": rows,
    }
    _write(receipt, output)

    def rebuild(key):
        """Rebuild one bank row's passive-inclusive profile and seed."""
        selected_row, qualification = selected[key]
        case, context = _mast_case_from_selection(
            SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, _policy = _passive_inclusive_case(
            case, context, response_cache
        )
        target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
        return profile, jnp.asarray(passive_case["state"]), target_current

    def publish(headline):
        """Persist the receipt so far and announce the row that landed."""
        receipt["rows"] = [
            {
                name: value
                for name, value in item.items()
                if not name.endswith("_terminal_flux")
            }
            for item in rows
        ]
        _write(receipt, output)
        print("REDUCED-ROW-DONE " + json.dumps(headline, sort_keys=True), flush=True)

    for key in TARGETS:
        identity = f"{key[0]}/{key[1]}"
        profile, seed, target_current = rebuild(key)
        row = {"identity": identity, "arm": "pure"}
        row["banked_production"] = banked_production.get(identity)
        row["previous_prototype"] = previous_prototype.get(identity)
        indexed[identity] = row
        rows.append(row)
        arms: dict[str, dict[str, Any]] = {}
        for name, route in PROTOTYPE_ROUTES.items():
            print(f"REDUCED-ROW {name} {identity}", flush=True)
            measured = _prototype_arm(
                profile.operator, seed, target_current, policy, route
            )
            arms[name] = measured
            row[f"{name}_terminal_flux"] = measured["flux"]
            row[name] = {
                field_name: value
                for field_name, value in measured.items()
                if field_name != "flux"
            }
            publish(
                {
                    "identity": identity,
                    "route": name,
                    "residual": measured["terminal_residual"],
                    "trips": measured["active_set_iterations"],
                    "wall_s": measured["wall_s"],
                    "median_warm_step_wall_s": measured["median_warm_step_wall_s"],
                    "median_warm_trip_wall_s": measured["median_warm_trip_wall_s"],
                }
            )
        row["route_agreement"] = _route_agreement(
            arms["prototype"], arms["eager_prototype"], row["previous_prototype"]
        )
        row["repair"] = _speedup(row["prototype"], row["eager_prototype"])
        row["batched_tail"] = _tail_comparison(
            arms["prototype"], arms["serial_prototype"], arms["eager_prototype"]
        )
        terminal_flux[f"{identity} fast"] = np.asarray(arms["prototype"]["flux"])
        terminal_flux[f"{identity} serial"] = np.asarray(
            arms["serial_prototype"]["flux"]
        )
        terminal_flux[f"{identity} reference"] = np.asarray(
            arms["eager_prototype"]["flux"]
        )
        np.savez_compressed(
            output.with_name(output.stem + "-terminal-flux.npz"), **terminal_flux
        )
        publish(
            {
                "identity": identity,
                "flux_agrees": row["route_agreement"]["flux_agrees"],
                "sup_flux_difference_fraction_of_span": row["route_agreement"][
                    "sup_flux_difference_fraction_of_span"
                ],
                "warm_step_speedup": row["repair"]["warm_step_speedup"],
                "warm_trip_speedup": row["repair"]["warm_trip_speedup"],
                "refused_step_wall_s": row["batched_tail"]["refused_step_wall_s"],
                "refused_step_wall_over_eager_s": row["batched_tail"][
                    "refused_step_wall_over_eager_s"
                ],
                "accepting_step_wall_s": row["batched_tail"]["accepting_step_wall_s"],
                "batched_tail_decisions_identical": row["batched_tail"][
                    "identical_decisions"
                ],
                "batched_tail_flux_bitwise_identical": row["batched_tail"][
                    "bitwise_identical_flux"
                ],
            }
        )

    if production:
        for key in TARGETS:
            identity = f"{key[0]}/{key[1]}"
            print(f"REDUCED-ROW production {identity}", flush=True)
            profile, seed, target_current = rebuild(key)
            row = indexed[identity]
            measured = _production_arm(profile, seed, target_current)
            row["agreement"] = _fixed_point_agreement(
                profile.operator,
                measured,
                {
                    "flux": row["prototype_terminal_flux"],
                    "converged": row["prototype"]["converged"],
                },
                target_current,
            )
            measured.pop("flux")
            row["production"] = measured
            publish(
                {
                    "identity": identity,
                    "production_wall_s": measured["wall_s"],
                    "production_trips": measured["active_set_iterations"],
                    "sup_flux_difference_wb": row["agreement"][
                        "sup_flux_difference_wb"
                    ],
                }
            )

    published = receipt["rows"]
    figure_path = output.with_suffix(".png")
    _draw(published, figure_path)
    receipt["figure"] = str(figure_path)
    receipt["verdict"] = {
        "row_count": len(published),
        "prototype_converged_count": sum(
            row["prototype"]["converged"] for row in published
        ),
        "maximum_off_support_leakage": max(
            (row["prototype"]["off_support_leakage"] for row in published),
            default=0.0,
        ),
        "any_rejected_newton_step": any(
            row["prototype"]["rejected_step_count"] > 0 for row in published
        ),
        "minimum_warm_step_wall_s": min(
            (
                row["prototype"]["median_warm_step_wall_s"]
                for row in published
                if row["prototype"]["median_warm_step_wall_s"] is not None
            ),
            default=None,
        ),
        "maximum_warm_step_wall_s": max(
            (
                row["prototype"]["median_warm_step_wall_s"]
                for row in published
                if row["prototype"]["median_warm_step_wall_s"] is not None
            ),
            default=None,
        ),
        "every_row_flux_agrees": all(
            row["route_agreement"]["flux_agrees"] for row in published
        ),
        "every_row_trip_census_identical": all(
            row["route_agreement"]["identical_trip_census"] for row in published
        ),
        "every_row_batched_tail_decisions_identical": all(
            row["batched_tail"]["identical_decisions"]
            for row in published
            if "batched_tail" in row
        ),
        "every_row_batched_tail_flux_bitwise_identical": all(
            row["batched_tail"]["bitwise_identical_flux"]
            for row in published
            if "batched_tail" in row
        ),
        "maximum_refused_step_wall_over_eager_s": max(
            (
                row["batched_tail"]["refused_step_wall_over_eager_s"]
                for row in published
                if row.get("batched_tail", {}).get("refused_step_wall_over_eager_s")
                is not None
            ),
            default=None,
        ),
        "maximum_accepting_step_wall_s": max(
            (
                row["batched_tail"]["accepting_step_wall_s"]
                for row in published
                if row.get("batched_tail", {}).get("accepting_step_wall_s") is not None
            ),
            default=None,
        ),
        "previous_receipt_reproduced": all(
            (row["route_agreement"]["previous_receipt_reproduced"] or {}).get(
                "identical_trip_census", False
            )
            for row in published
        ),
        "terminal_flux": str(output.with_name(output.stem + "-terminal-flux.npz")),
    }
    _write(receipt, output)
    return receipt


def main() -> None:
    """Run the four-row prototype measurement from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--support-policy",
        choices=("participation", "active"),
        default=reduced_newton.SUPPORT_POLICY,
    )
    parser.add_argument("--no-production", action="store_true")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="persistent compilation cache root this launch selects",
    )
    arguments = parser.parse_args()
    measure(
        output=arguments.output,
        policy=arguments.support_policy,
        production=not arguments.no_production,
        cache_root=arguments.cache_root,
    )


if __name__ == "__main__":
    main()
