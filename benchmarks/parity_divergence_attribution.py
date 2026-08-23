"""Attribute compiled-versus-eager divergence in the profile solve.

The measurement separates three mutually exclusive explanations.  It repeats
the frozen six-case solve on a source snapshot preceding the candidate repair
commits, compares one map application at an identical state, and measures the
compiled-versus-eager flux separation after each nonlinear update.  The
committed receipt retains all discriminating measurements and the quantity
clusters from the gate that triggered this diagnosis.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import zarr

from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    _stored_lcfs,
    build_profile,
    select_slices_by_shot,
)
from nova.equilibrium import fixed_point as fixed_point_module
from nova.equilibrium import forward_operator as forward_operator_module
from nova.equilibrium.forward import _lattice_cells
from nova.equilibrium.stencil_mesh import (
    MomentGeometry,
    StencilMesh,
)
from nova.geometry.hexstencil import hex_stencil
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes


OUTPUT = Path("docs/figures/mast-catalog-gpu-solve/parity-divergence-attribution.json")
PARITY_RECEIPT = Path(
    "docs/figures/mast-catalog-gpu-solve/jitted-eager-parity-gate.json"
)
SPINE_BENCH = Path("/home/ITER/mcintos/Code/imas-ambix/imas_ambix/spine_bench")
SHOTSET_MODULE = SPINE_BENCH / "shots.py"
SHOTSET_VERSION = "v0-mast-heldout-6"
HISTORICAL_COMMIT = "83dc40fbb1bff3c37ce97730b9eaa9bba0344117"
CANDIDATE_REPAIR_COMMITS = (
    "6143221d7a2dc5c3ad2a74b822bf1a026c05eed3",
    "06b09f5bfdcf184a69066e829a6e7ee16fdd3a2b",
    "257af9dcbe5dd19313708f862a8bb3a383f917b2",
    "7e33c4d000131ce0bb76b9b3c7ad2b71ce668d36",
)
REGISTERED_TOLERANCE = 1.0e-10
MATERIAL_REDUCTION_FACTOR = 0.1
PINNED_MAP_SPECTRAL_RADIUS = 1.1455670310089587
UNPINNED_MAP_SPECTRAL_RADIUS = 1.2576631175347157
TRAJECTORY_CASE = (21985, 51)
SOLVE_OPTIONS = {
    "route": "newton_krylov",
    "gmres_iterations": 12,
    "warmup": 0,
}
NONLINEAR_UPDATES = 12


def _git(*arguments: str) -> str:
    """Return one source-tree identity fact."""

    return subprocess.check_output(
        ["git", *arguments], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _utc_now() -> str:
    """Return a JSON-stable UTC timestamp."""

    return datetime.now(UTC).isoformat()


def _digest(path: Path) -> str:
    """Return the SHA-256 digest of one authority artifact."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, receipt: dict[str, Any]) -> None:
    """Write a human-readable receipt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=False) + "\n")


def _with_moment_geometry(profile):
    """Complete the frozen-case builder with production moment geometry."""

    if profile.operator.moment_geometry is not None:
        return profile
    lattice = profile.lattice
    mesh = StencilMesh(
        coordinate=lattice.coordinate,
        stencil=hex_stencil(lattice.shape),
        area=lattice.cell_area,
    )
    operator = replace(
        profile.operator,
        moment_geometry=MomentGeometry.from_cells(mesh, _lattice_cells(lattice)),
        prescribed_current_field=profile.operator.prescribed_field,
    )
    return replace(profile, operator=operator)


def _case_rows(store: Path) -> list[tuple[int, int]]:
    """Return the immutable reference slice for every frozen held-out shot."""

    selected = select_slices_by_shot(DECOMPOSITION_BANK)
    rows = [(int(row["shot"]), int(row["slice_index"])) for row, _ in selected]
    if len(rows) != 6 or len({shot for shot, _ in rows}) != 6:
        raise RuntimeError("the frozen held-out cohort must contain six distinct shots")
    missing = [shot for shot, _ in rows if not (store / f"{shot}.zarr").is_dir()]
    if missing:
        raise FileNotFoundError(f"held-out shot stores are absent: {missing}")
    return rows


def _difference(reference: Any, candidate: Any) -> dict[str, float | list[int]]:
    """Return scale-aware sup-norm separation for two numeric values."""

    left = np.asarray(reference)
    right = np.asarray(candidate)
    if left.shape != right.shape:
        raise RuntimeError(
            f"shape mismatch: eager={left.shape}, compiled={right.shape}"
        )
    finite = np.isfinite(left) & np.isfinite(right)
    if not np.any(finite):
        return {
            "shape": list(left.shape),
            "maximum_absolute_difference": 0.0,
            "maximum_relative_difference": 0.0,
        }
    absolute = np.abs(right[finite] - left[finite])
    maximum_absolute = float(np.max(absolute))
    scale = max(float(np.max(np.abs(left[finite]))), np.finfo(float).tiny)
    return {
        "shape": list(left.shape),
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_absolute / scale,
    }


def _build_case(store: Path, shot: int, slice_index: int):
    """Build one profile, its production seed, and declared current."""

    group = zarr.open_group(str(store / f"{shot}.zarr"), mode="r")["efm"]
    profile, _reference_seed, _reference, _provenance = build_profile(
        group, shot, slice_index, "fcoil_c"
    )
    profile = _with_moment_geometry(profile)
    boundary = _stored_lcfs(group, slice_index)
    target_current = abs(float(group["plasma_current_c"][slice_index]))
    seed = profile.moment_seed(boundary, target_current)
    return profile, seed, target_current, float(group["time"][slice_index])


def _solve_pair(profile, initial_flux, target_current, nonlinear_updates: int):
    """Run the production solve eagerly and as one compiled program."""

    def solve(state):
        return profile.solve(
            state,
            target_current=target_current,
            newton_steps=nonlinear_updates,
            **SOLVE_OPTIONS,
        )

    eager = solve(initial_flux)
    compiled = jax.jit(solve)(initial_flux)
    jax.block_until_ready(compiled)
    return eager, compiled


def _trajectory(profile, initial_flux, target_current) -> list[dict[str, Any]]:
    """Measure separation after every production nonlinear-update count."""

    rows: list[dict[str, Any]] = []
    preceding = None
    for nonlinear_updates in range(1, NONLINEAR_UPDATES + 1):
        eager, compiled = _solve_pair(
            profile, initial_flux, target_current, nonlinear_updates
        )
        flux = _difference(eager.flux, compiled.flux)
        absolute = float(flux["maximum_absolute_difference"])
        growth = None if preceding in (None, 0.0) else absolute / preceding
        rows.append(
            {
                "nonlinear_update": nonlinear_updates,
                **flux,
                "growth_from_preceding_update": growth,
                "eager_residual": float(eager.fixed_point.residual),
                "compiled_residual": float(compiled.fixed_point.residual),
            }
        )
        preceding = absolute
    return rows


def _measure_source(store: Path, *, include_trajectory: bool) -> dict[str, Any]:
    """Measure the six cases using whichever source tree this process imported."""

    configure_dtypes()
    cases = []
    trajectory = None
    for shot, slice_index in _case_rows(store):
        profile, seed, target_current, time_s = _build_case(store, shot, slice_index)
        mapped = profile.flux_map(target_current=target_current)
        eager_map = mapped(seed.flux)
        compiled_map = jax.jit(mapped)(seed.flux)
        jax.block_until_ready(compiled_map)
        single_application = _difference(eager_map, compiled_map)

        eager, compiled = _solve_pair(
            profile, seed.flux, target_current, NONLINEAR_UPDATES
        )
        solved_flux = _difference(eager.flux, compiled.flux)
        cases.append(
            {
                "shot": shot,
                "slice_index": slice_index,
                "time_s": time_s,
                "single_map_application": single_application,
                "solved_flux": solved_flux,
                "eager_residual": float(eager.fixed_point.residual),
                "compiled_residual": float(compiled.fixed_point.residual),
            }
        )
        if include_trajectory and (shot, slice_index) == TRAJECTORY_CASE:
            trajectory = _trajectory(profile, seed.flux, target_current)

    if include_trajectory and trajectory is None:
        raise RuntimeError(f"trajectory case {TRAJECTORY_CASE} was not measured")
    return {
        "imported_source_modules": {
            "nova/equilibrium/fixed_point.py": _digest(
                Path(fixed_point_module.__file__)
            ),
            "nova/equilibrium/forward_operator.py": _digest(
                Path(forward_operator_module.__file__)
            ),
        },
        "backend": {
            "platform": jax.default_backend(),
            "device": jax.devices()[0].device_kind,
            "jax_version": jax.__version__,
            "precision": "float64",
        },
        "cases": cases,
        "trajectory": trajectory,
    }


def _maximum(rows: list[dict[str, Any]], section: str, metric: str) -> float:
    """Return one maximum metric across the frozen cases."""

    return max(float(row[section][metric]) for row in rows)


def _historical_measurement(store: Path, script: Path) -> dict[str, Any]:
    """Run this measurement against an extracted pre-repair source snapshot."""

    with tempfile.TemporaryDirectory(prefix="nova-parity-history-") as directory:
        root = Path(directory)
        archive = root / "source.tar"
        with archive.open("wb") as stream:
            subprocess.run(
                ["git", "archive", "--format=tar", HISTORICAL_COMMIT],
                check=True,
                stdout=stream,
            )
        with tarfile.open(archive) as bundle:
            bundle.extractall(root, filter="data")
        output = root / "measurement.json"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(root)
        command = [
            sys.executable,
            str(script),
            "--measurement-only",
            "--store",
            str(store),
            "--output",
            str(output),
        ]
        subprocess.run(command, check=True, env=environment)
        return json.loads(output.read_text())


def _quantity_clusters(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Group every gate quantity by the mechanism its value represents."""

    quantities = receipt["comparisons"]["profile_solve"]["quantities"]
    failing = {name: row for name, row in quantities.items() if not row["passes"]}
    passing = {name: row for name, row in quantities.items() if row["passes"]}

    def take(rows: Mapping[str, Any], predicate) -> list[str]:
        return sorted(name for name in rows if predicate(name))

    failing_clusters = {
        "state_domain_topology_and_solver_trace": take(
            failing,
            lambda name: (
                name in {"flux", "cell_current"}
                or name.startswith(("domains.", "topology.", "fixed_point."))
            ),
        ),
        "integrated_moments_and_current_ledger": take(
            failing, lambda name: name.startswith(("moments.", "ledger."))
        ),
        "conservation_and_source_normalisation": take(
            failing,
            lambda name: name.startswith(("conservation.", "normalisation.")),
        ),
    }
    passing_clusters = {
        "continuous_values_at_relative_roundoff": take(
            passing,
            lambda name: float(passing[name]["maximum_absolute_difference"]) > 0.0,
        ),
        "exact_categorical_policy_and_structural_invariants": take(
            passing,
            lambda name: float(passing[name]["maximum_absolute_difference"]) == 0.0,
        ),
    }
    if sum(map(len, failing_clusters.values())) != len(failing):
        raise RuntimeError("failing quantity classification is incomplete")
    if sum(map(len, passing_clusters.values())) != len(passing):
        raise RuntimeError("passing quantity classification is incomplete")
    return {
        "failing_quantity_count": len(failing),
        "passing_quantity_count": len(passing),
        "failing_clusters": {
            name: {"count": len(names), "quantities": names}
            for name, names in failing_clusters.items()
        },
        "passing_clusters": {
            name: {"count": len(names), "quantities": names}
            for name, names in passing_clusters.items()
        },
    }


def _attribution(
    current: Mapping[str, Any], historical: Mapping[str, Any]
) -> dict[str, Any]:
    """Choose exactly one cause from the preregistered discriminators."""

    current_cases = current["cases"]
    historical_cases = historical["cases"]
    current_flux = _maximum(current_cases, "solved_flux", "maximum_absolute_difference")
    historical_flux = _maximum(
        historical_cases, "solved_flux", "maximum_absolute_difference"
    )
    historical_present = historical_flux > REGISTERED_TOLERANCE
    historical_materially_smaller = (
        historical_flux <= MATERIAL_REDUCTION_FACTOR * current_flux
    )
    regression_supported = (not historical_present) or historical_materially_smaller

    single_relative = _maximum(
        current_cases, "single_map_application", "maximum_relative_difference"
    )
    trajectory = current["trajectory"]
    final_trajectory = float(trajectory[-1]["maximum_absolute_difference"])
    first_nonzero = next(
        (
            float(row["maximum_absolute_difference"])
            for row in trajectory
            if float(row["maximum_absolute_difference"]) > 0.0
        ),
        0.0,
    )
    cumulative_growth = (
        float("inf") if first_nonzero == 0.0 else final_trajectory / first_nonzero
    )
    roundoff_scale = 16.0 * np.finfo(np.float64).eps
    amplification_supported = (
        not regression_supported
        and single_relative <= roundoff_scale
        and final_trajectory > REGISTERED_TOLERANCE
        and cumulative_growth > 1.0
        and PINNED_MAP_SPECTRAL_RADIUS > 1.0
    )

    if regression_supported:
        cause = "REGRESSION_FROM_TODAYS_REPAIRS"
    elif amplification_supported:
        cause = "AMPLIFIED_REPRESENTATION_DIFFERENCE"
    else:
        cause = "SEMANTIC_DIFFERENCE"

    historical_relation = "present"
    if not historical_present:
        historical_relation = "absent"
    elif historical_materially_smaller:
        historical_relation = "smaller"

    return {
        "selected_cause": cause,
        "exactly_one_cause_selected": True,
        "causes": {
            "REGRESSION_FROM_TODAYS_REPAIRS": {
                "selected": cause == "REGRESSION_FROM_TODAYS_REPAIRS",
                "discriminator": (
                    "Repeat the same six cases at a source tree preceding all four "
                    "candidate repair commits; classify the prior divergence as "
                    "present, absent, or materially smaller."
                ),
                "historical_commit": HISTORICAL_COMMIT,
                "candidate_repair_commits": list(CANDIDATE_REPAIR_COMMITS),
                "current_maximum_flux_absolute_difference": current_flux,
                "historical_maximum_flux_absolute_difference": historical_flux,
                "historical_relation": historical_relation,
                "materially_smaller_definition": (
                    f"historical <= {MATERIAL_REDUCTION_FACTOR} * current"
                ),
            },
            "AMPLIFIED_REPRESENTATION_DIFFERENCE": {
                "selected": cause == "AMPLIFIED_REPRESENTATION_DIFFERENCE",
                "discriminator": (
                    "At the identical seeded state, compare one eager and compiled "
                    "map application, then measure separation after each nonlinear "
                    "update on a map whose spectral radius is already measured."
                ),
                "maximum_single_map_relative_difference": single_relative,
                "float64_epsilon": float(np.finfo(np.float64).eps),
                "roundoff_band_upper_bound": float(roundoff_scale),
                "trajectory_case": {
                    "shot": TRAJECTORY_CASE[0],
                    "slice_index": TRAJECTORY_CASE[1],
                },
                "first_nonzero_trajectory_absolute_difference": first_nonzero,
                "final_trajectory_absolute_difference": final_trajectory,
                "cumulative_growth": cumulative_growth,
                "pinned_map_spectral_radius": PINNED_MAP_SPECTRAL_RADIUS,
                "unpinned_map_spectral_radius": UNPINNED_MAP_SPECTRAL_RADIUS,
                "trajectory": trajectory,
            },
            "SEMANTIC_DIFFERENCE": {
                "selected": cause == "SEMANTIC_DIFFERENCE",
                "discriminator": (
                    "If a one-application difference exceeds the float64 roundoff "
                    "band without a repair-tree regression, identify the first "
                    "behaviourally different operation by source location."
                ),
                "first_diverging_operation": None,
                "interpretation": (
                    "No semantic operation is named when the selected evidence "
                    "shows only roundoff-scale one-application separation followed "
                    "by iterative amplification."
                    if cause != "SEMANTIC_DIFFERENCE"
                    else "A source-level operation must be localized before closure."
                ),
            },
        },
    }


def measure(output: Path, store: Path) -> dict[str, Any]:
    """Run both source snapshots and commit the attribution receipt."""

    current_commit = _git("rev-parse", "HEAD")
    current_tree = _git("rev-parse", "HEAD^{tree}")
    ancestry = {
        commit: subprocess.run(
            ["git", "merge-base", "--is-ancestor", HISTORICAL_COMMIT, commit],
            check=False,
        ).returncode
        == 0
        for commit in CANDIDATE_REPAIR_COMMITS
    }
    if not all(ancestry.values()):
        raise RuntimeError("the historical source does not precede every repair commit")

    current = _measure_source(store, include_trajectory=True)
    historical = _historical_measurement(store, Path(__file__).resolve())
    gate = json.loads(PARITY_RECEIPT.read_text())
    attribution = _attribution(current, historical)
    selected = attribution["selected_cause"]
    requirement_rederived = selected == "AMPLIFIED_REPRESENTATION_DIFFERENCE"
    receipt = {
        "schema": "nova-parity-divergence-attribution/1.0",
        "status": "attributed",
        "completed_utc": _utc_now(),
        "source_identity": {
            "current_commit": current_commit,
            "current_tree": current_tree,
            "historical_commit": HISTORICAL_COMMIT,
            "historical_precedes_candidate_repairs": ancestry,
        },
        "held_out_set": {
            "version": SHOTSET_VERSION,
            "shotset_module": str(SHOTSET_MODULE),
            "shotset_module_sha256": _digest(SHOTSET_MODULE),
            "case_count": len(current["cases"]),
            "cases": [
                {"shot": row["shot"], "slice_index": row["slice_index"]}
                for row in current["cases"]
            ],
        },
        "registered_parity_requirement": {
            "absolute_tolerance": REGISTERED_TOLERANCE,
            "relative_tolerance": REGISTERED_TOLERANCE,
            "triggering_flux_absolute_difference": gate["comparisons"]["profile_solve"][
                "quantities"
            ]["flux"]["maximum_absolute_difference"],
            "triggering_flux_relative_difference": gate["comparisons"]["profile_solve"][
                "quantities"
            ]["flux"]["maximum_relative_difference"],
            "trigger_receipt": str(PARITY_RECEIPT),
            "trigger_receipt_sha256": _digest(PARITY_RECEIPT),
        },
        "attribution": attribution,
        "current_source_measurement": current,
        "historical_source_measurement": historical,
        "quantity_clusters": _quantity_clusters(gate),
        "fixed_point_parity_disposition": {
            "achievable_as_registered": not requirement_rederived,
            "must_be_rederived": requirement_rederived,
            "reason": (
                "A non-contractive map with measured spectral radius above one "
                "amplified a float64-roundoff one-application difference beyond "
                "the registered fixed-point bound. Cross-compilation equality of "
                "the terminal state is therefore not a stable requirement; bind "
                "parity to one map application and separately bound physically "
                "meaningful terminal observables."
                if requirement_rederived
                else "The selected cause does not establish impossibility of the bound."
            ),
        },
    }
    _write(output, receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    """Return the benchmark command-line interface."""

    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=OUTPUT)
    result.add_argument("--store", type=Path, default=SHOT_STORE)
    result.add_argument(
        "--measurement-only", action="store_true", help=argparse.SUPPRESS
    )
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.measurement_only:
        result = _measure_source(arguments.store, include_trajectory=False)
        _write(arguments.output, result)
    else:
        result = measure(arguments.output, arguments.store)
    print(
        json.dumps(
            {
                "status": result.get("status", "measured"),
                "selected_cause": result.get("attribution", {}).get("selected_cause"),
                "case_count": len(
                    result.get("held_out_set", {}).get("cases", result.get("cases", []))
                ),
            },
            indent=2,
        )
    )
