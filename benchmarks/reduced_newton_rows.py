"""Measure the reduced-state plain-Newton prototype on the four bank rows.

Each row is solved twice from the same persisted pure-arm seed: once through
the production public route, ``ForwardProfile.solve_branch(newton_krylov)``,
and once through the reduced-amplitude plain-Newton prototype.  Both arms
report the terminal residual, the trip count, the Newton steps per trip and
the wall per step and per trip, so the prototype's distance to a millisecond
fixed point is read against the ladder it replaces rather than against a
projection.  Every row is written to the receipt as it finishes, so a job that
runs out of wall clock still delivers the rows it measured.
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


def _prototype_arm(operator, seed, target_current, policy: str) -> dict[str, Any]:
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
    )
    wall = time.perf_counter() - started
    accepted = [step for step in result.steps if step.accepted_factor is not None]
    step_walls = [step.wall_s for step in result.steps]
    warm_steps = [step for step in result.steps if step.trip > 0]
    return {
        "route": "reduced_newton.solve_reduced_newton",
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
        "jacobian_wall_per_trip_s": result.jacobian_wall_per_trip,
        "newton_wall_per_trip_s": result.newton_wall_per_trip,
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
        "steps": [
            {
                "trip": step.trip,
                "step": step.step,
                "flux_residual": step.flux_residual,
                "reduced_residual": step.reduced_residual,
                "merit": step.merit,
                "accepted_factor": step.accepted_factor,
                "grades_tried": step.grades_tried,
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
    """Plot per-row wall and per-step wall for both arms."""
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    identities = [row["identity"] for row in rows]
    index = np.arange(len(rows))
    production = [row["production"]["wall_s"] for row in rows]
    prototype = [row["prototype"]["wall_s"] for row in rows]
    axes[0].bar(index - 0.2, production, 0.4, label="production", color="#c0504d")
    axes[0].bar(index + 0.2, prototype, 0.4, label="reduced Newton", color="#4f81bd")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("solve wall [s]")
    axes[0].set_title("whole solve")
    axes[0].legend(fontsize=8)

    warm = [row["prototype"]["median_warm_step_wall_s"] or np.nan for row in rows]
    first = [row["prototype"]["first_step_wall_s"] or np.nan for row in rows]
    axes[1].bar(index - 0.2, first, 0.4, label="first step (cold)", color="#9bbb59")
    axes[1].bar(index + 0.2, warm, 0.4, label="median warm step", color="#4f81bd")
    axes[1].axhline(1.0e-3, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Newton step wall [s]")
    axes[1].set_title("per Newton step against 1 ms")
    axes[1].legend(fontsize=8)

    axes[2].bar(
        index - 0.2,
        [row["production"]["terminal_residual"] for row in rows],
        0.4,
        label="production",
        color="#c0504d",
    )
    axes[2].bar(
        index + 0.2,
        [row["prototype"]["terminal_residual"] for row in rows],
        0.4,
        label="reduced Newton",
        color="#4f81bd",
    )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("terminal relative residual")
    axes[2].set_title("terminal residual")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.set_xticks(index)
        axis.set_xticklabels(identities, rotation=20, fontsize=8)
        axis.grid(True, axis="y", alpha=0.3)
    figure.suptitle(
        "Reduced-state plain Newton beside the production Newton-Krylov ladder"
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


def measure(*, output: Path, policy: str, production: bool) -> dict[str, Any]:
    """Solve the four bank rows through both arms and persist as they land."""
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    }
    banked_production = _banked_production_rows()
    rows: list[dict[str, Any]] = []
    indexed: dict[str, dict[str, Any]] = {}
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
        },
        "measurement_contract": {
            "targets": [list(key) for key in TARGETS],
            "seed": "persisted pure-arm bank seed for each row",
            "production_route": "ForwardProfile.solve_branch(newton_krylov)",
            "prototype_route": "reduced_newton.solve_reduced_newton",
            "support_policy": policy,
            "convergence_tolerance": FIXED_POINT_CRITERION,
            "newton_steps": NEWTON_STEPS,
            "production_arm_measured_here": production,
            "pass_order": (
                "every prototype row lands before the first production row, so "
                "a job that runs out of wall clock still delivers the arm it "
                "was dispatched to measure"
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

    measurement_passes = ["prototype"] + (["production"] if production else [])
    for measurement_pass in measurement_passes:
        for key in TARGETS:
            identity = f"{key[0]}/{key[1]}"
            print(f"REDUCED-ROW {measurement_pass} {identity}", flush=True)
            profile, seed, target_current = rebuild(key)
            if measurement_pass == "prototype":
                measured = _prototype_arm(
                    profile.operator, seed, target_current, policy
                )
                row = {"identity": identity, "arm": "pure"}
                row["banked_production"] = banked_production.get(identity)
                row["prototype_terminal_flux"] = measured["flux"]
                row["prototype"] = {
                    field_name: value
                    for field_name, value in measured.items()
                    if field_name != "flux"
                }
                indexed[identity] = row
                rows.append(row)
                headline = {
                    "identity": identity,
                    "prototype_residual": measured["terminal_residual"],
                    "prototype_trips": measured["active_set_iterations"],
                    "prototype_wall_s": measured["wall_s"],
                    "reduced_dimension": measured["reduced_dimension"],
                }
            else:
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
                headline = {
                    "identity": identity,
                    "production_wall_s": measured["wall_s"],
                    "production_trips": measured["active_set_iterations"],
                    "sup_flux_difference_wb": row["agreement"][
                        "sup_flux_difference_wb"
                    ],
                }
            receipt["rows"] = [
                {
                    name: value
                    for name, value in item.items()
                    if name != "prototype_terminal_flux"
                }
                for item in rows
            ]
            _write(receipt, output)
            print(
                "REDUCED-ROW-DONE " + json.dumps(headline, sort_keys=True),
                flush=True,
            )

    rows = receipt["rows"]
    if production and all("production" in row for row in rows):
        figure_path = output.with_suffix(".png")
        _draw(rows, figure_path)
        receipt["figure"] = str(figure_path)
    receipt["verdict"] = {
        "row_count": len(rows),
        "prototype_converged_count": sum(row["prototype"]["converged"] for row in rows),
        "maximum_off_support_leakage": max(
            (row["prototype"]["off_support_leakage"] for row in rows), default=0.0
        ),
        "any_rejected_newton_step": any(
            row["prototype"]["rejected_step_count"] > 0 for row in rows
        ),
        "minimum_warm_step_wall_s": min(
            (
                row["prototype"]["median_warm_step_wall_s"]
                for row in rows
                if row["prototype"]["median_warm_step_wall_s"] is not None
            ),
            default=None,
        ),
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
    arguments = parser.parse_args()
    measure(
        output=arguments.output,
        policy=arguments.support_policy,
        production=not arguments.no_production,
    )


if __name__ == "__main__":
    main()
