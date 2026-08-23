"""Measure conditioned convergence and registered terminal-observable parity.

The DIII-D half repeats the established five score-blind solves and the
favourable coarse/native carrier on the checked-out source tree.  The MAST half
repeats the six aligned eager/compiled terminal solves and applies the frozen
per-observable envelopes from the committed criterion-family receipt.

No DIII-D residual tolerance is selected here.  The residual measurements are
reported only against their pre-conditioning comparators.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import zarr

from benchmarks import diiid_repaired_solve_remeasure as repaired_solve
from benchmarks import jitted_eager_parity_gate as parity
from benchmarks import topology_qualified_mesh_convergence as mesh_convergence
from nova.equilibrium.fixed_point import (
    KrylovActionQualification,
    kink_aware_newton_krylov,
)
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes


HERE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    HERE / "docs/figures/diiid-forward-onboarding/"
    "conditioned-convergence-and-observables.json"
)
CRITERION_SOURCE = (
    HERE / "docs/figures/forward-operator-refinement/criterion-family.json"
)
PARITY_ATTRIBUTION_SOURCE = (
    HERE / "docs/figures/mast-catalog-gpu-solve/parity-divergence-attribution.json"
)
PRECONDITIONING_SOURCE = (
    HERE / "docs/figures/diiid-forward-onboarding/factor-ladder-extension.json"
)
EXPECTED_FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)


def _git(*arguments: str) -> str:
    """Return one git identity for the measured source tree."""

    return subprocess.check_output(
        ["git", *arguments], cwd=HERE, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _sha256(path: Path) -> str:
    """Return the content identity of one evidence input."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read a committed JSON evidence input."""

    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write stable, finite JSON at the assigned receipt path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _utc_now() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(UTC).isoformat()


def _selected_factors(result: Any) -> list[float]:
    """Return every promoted factor in iteration order."""

    factors = np.asarray(result.accepted_factors, dtype=float)
    return [float(factor) for factor in factors if factor > 0.0]


def _measure_diiid_frame(
    data: Path,
    case: repaired_solve.FrameCase,
    comparator: dict[str, Any],
) -> dict[str, Any]:
    """Repeat one score-blind topology-qualified DIII-D solve."""

    started = perf_counter()
    row = repaired_solve._read_case(data / case.shot)
    profile, current, target_current_a, time_ms, seed = repaired_solve._prepare_frame(
        row, case.frame
    )
    mapped = profile.flux_map(
        jnp.asarray(current), TopologyClass.DIVERTED, target_current_a
    )

    def remains_diverted(candidate):
        _masks, topology = profile.operator.read(candidate)
        return jnp.all(jnp.isfinite(candidate)) & topology.diverted

    result = kink_aware_newton_krylov(
        mapped,
        seed,
        strategy="nonmonotone",
        newton_steps=repaired_solve.NEWTON_STEPS,
        gmres_iterations=repaired_solve.GMRES_ITERATIONS,
        warmup=0,
        admissibility_fn=remains_diverted,
    )
    state = np.asarray(result.state, dtype=float)
    image = np.asarray(mapped(result.state), dtype=float)
    residual = float(
        np.max(np.abs(image - state)) / max(np.max(np.abs(image)), 1.0e-30)
    )
    _masks, topology = profile.operator.read(result.state)
    x_point = np.asarray(topology.x_point, dtype=float)
    factors = _selected_factors(result)
    preconditioning_residual = float(
        comparator["banked_repaired_terminal_relative_residual"]
    )
    residual_ratio = residual / preconditioning_residual
    return {
        "shot": case.shot,
        "frame": case.frame,
        "time_ms": time_ms,
        "target_current_a": float(target_current_a),
        "terminal_relative_residual": residual,
        "preconditioning_terminal_relative_residual": preconditioning_residual,
        "residual_ratio_vs_preconditioning": residual_ratio,
        "signed_relative_change_vs_preconditioning": residual_ratio - 1.0,
        "admitted_promotion_count_of_89": len(factors),
        "preconditioning_admitted_promotion_count_of_89": int(
            comparator["admitted_promotion_count_of_89"]
        ),
        "terminal_topology_class": (
            "diverted" if bool(topology.diverted) else "limited"
        ),
        "finite_terminal_x_point": bool(np.all(np.isfinite(x_point))),
        "krylov_action_qualification": KrylovActionQualification(
            int(result.krylov_action_qualification)
        ).name,
        "krylov_conditioning_count": int(result.krylov_conditioning_count),
        "maximum_projected_krylov_condition": float(
            result.maximum_projected_krylov_condition
        ),
        "selected_factor_by_admitted_step": factors,
        "runtime_seconds": perf_counter() - started,
    }


def _measure_favourable_frame(data: Path) -> dict[str, Any]:
    """Repeat the established coarse and reference-native carrier solves."""

    bank = np.load(mesh_convergence.STATE_BANK)
    current = np.asarray(bank["current"], dtype=float)
    seed = np.asarray(bank["seed"], dtype=float)
    row = mesh_convergence._read_case(data / mesh_convergence.SHOT)
    comparators = _read_json(PRECONDITIONING_SOURCE)["favourable_frame_remeasure"][
        "rungs"
    ]
    measured = [
        mesh_convergence._solve_rung(row, rung, current, seed)
        for rung in mesh_convergence.MESH_LADDER
    ]
    records = []
    for result, comparator in zip(measured, comparators, strict=True):
        if result["name"] != comparator["name"]:
            raise RuntimeError("favourable-frame rung identity changed")
        residual = float(result["solver"]["terminal_relative_residual"])
        preconditioning_residual = float(
            comparator["banked_terminal_relative_residual"]
        )
        ratio = residual / preconditioning_residual
        if 0.99 <= ratio <= 1.01:
            comparison_disposition = "REPRODUCED_WITHIN_ONE_PERCENT"
        elif ratio < 0.99:
            comparison_disposition = "DISPLACED_FAVOURABLY"
        else:
            comparison_disposition = "DISPLACED_UNFAVOURABLY"
        records.append(
            {
                "name": result["name"],
                "grid_shape": result["grid_shape"],
                "terminal_relative_residual": residual,
                "preconditioning_terminal_relative_residual": (
                    preconditioning_residual
                ),
                "residual_ratio_vs_preconditioning": ratio,
                "signed_relative_change_vs_preconditioning": ratio - 1.0,
                "comparison_disposition": comparison_disposition,
                "full_step_admission_count_of_89": int(
                    result["solver"]["accepted_factor_counts"]["1.0"]
                ),
                "preconditioning_full_step_admission_count_of_89": int(
                    comparator["full_step_admission_count_of_89"]
                ),
                "krylov_conditioning_count": int(
                    result["solver"].get("krylov_conditioning_count", 0)
                ),
                "maximum_projected_krylov_condition": result["solver"].get(
                    "maximum_projected_krylov_condition"
                ),
                "terminal_topology_class": result["terminal_topology"]["class"],
                "finite_terminal_x_point": bool(
                    result["terminal_topology"]["finite_x_point"]
                ),
            }
        )
    return {
        "shot": mesh_convergence.SHOT,
        "frame": mesh_convergence.FRAME,
        "rungs": records,
    }


def _finite_difference(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Return absolute and reference-scaled relative differences."""

    finite = np.isfinite(left) & np.isfinite(right)
    if not np.array_equal(np.isnan(left), np.isnan(right)) or not np.any(finite):
        if np.array_equal(left, right, equal_nan=True):
            return 0.0, 0.0
        raise RuntimeError("terminal observable carries an unmatched non-finite value")
    absolute = float(
        np.max(
            np.abs(right[finite].astype(np.float64) - left[finite].astype(np.float64))
        )
    )
    reference_scale = max(
        float(np.max(np.abs(left[finite]))), np.finfo(np.float64).tiny
    )
    return absolute, absolute / reference_scale


def _bound_ratio(value: float, bound: float) -> float:
    """Return finite envelope utilisation, including an exact-zero bound."""

    if bound > 0.0:
        return value / bound
    return 0.0 if value == 0.0 else float("inf")


def _score_observable(
    eager: Any, compiled: Any, registration: dict[str, Any]
) -> dict[str, Any]:
    """Apply one frozen terminal-observable registration to one case."""

    left = np.asarray(eager)
    right = np.asarray(compiled)
    if left.shape != right.shape:
        return {
            "passes": False,
            "maximum_absolute_difference": None,
            "maximum_relative_difference": None,
            "maximum_bound_ratio": None,
            "reason": "shape mismatch",
        }
    absolute, relative = _finite_difference(left, right)
    if registration["criterion_kind"] == "exact_equality":
        passes = bool(np.array_equal(left, right, equal_nan=True))
        return {
            "passes": passes,
            "maximum_absolute_difference": absolute,
            "maximum_relative_difference": relative,
            "maximum_bound_ratio": 0.0 if passes else None,
            "reason": "exact equality" if passes else "exact equality mismatch",
        }
    absolute_bound = float(registration["absolute_bound"])
    relative_bound = float(registration["relative_bound"])
    absolute_ratio = _bound_ratio(absolute, absolute_bound)
    relative_ratio = _bound_ratio(relative, relative_bound)
    maximum_ratio = max(absolute_ratio, relative_ratio)
    if not np.isfinite(maximum_ratio):
        raise RuntimeError("a zero terminal-observable envelope was exceeded")
    return {
        "passes": bool(absolute <= absolute_bound and relative <= relative_bound),
        "maximum_absolute_difference": absolute,
        "maximum_relative_difference": relative,
        "absolute_bound_ratio": absolute_ratio,
        "relative_bound_ratio": relative_ratio,
        "maximum_bound_ratio": maximum_ratio,
        "reason": "dual absolute and relative envelope",
    }


def _initial_observable_record(registration: dict[str, Any]) -> dict[str, Any]:
    """Return an empty six-case accumulator for one registered observable."""

    result = {
        "observable": registration["observable"],
        "criterion_kind": registration["criterion_kind"],
        "dtype": registration["dtype"],
        "shape": registration["shape"],
        "pass_count": 0,
        "fail_count": 0,
        "maximum_absolute_difference": 0.0,
        "maximum_relative_difference": 0.0,
        "maximum_bound_ratio": 0.0,
        "largest_difference_case": None,
        "failing_cases": [],
    }
    if registration["criterion_kind"] == "banked_dual_envelope":
        result["absolute_bound"] = float(registration["absolute_bound"])
        result["relative_bound"] = float(registration["relative_bound"])
    return result


def _accumulate_observable(
    aggregate: dict[str, Any], measured: dict[str, Any], case_identity: dict[str, int]
) -> None:
    """Accumulate one case score into one observable receipt row."""

    aggregate["pass_count" if measured["passes"] else "fail_count"] += 1
    if not measured["passes"]:
        aggregate["failing_cases"].append(case_identity)
    absolute = measured["maximum_absolute_difference"]
    relative = measured["maximum_relative_difference"]
    if absolute is not None and absolute >= aggregate["maximum_absolute_difference"]:
        aggregate["maximum_absolute_difference"] = absolute
        aggregate["largest_difference_case"] = case_identity
    if relative is not None:
        aggregate["maximum_relative_difference"] = max(
            aggregate["maximum_relative_difference"], relative
        )
    ratio = measured["maximum_bound_ratio"]
    if ratio is not None:
        aggregate["maximum_bound_ratio"] = max(aggregate["maximum_bound_ratio"], ratio)


def _cluster_summary(
    observables: dict[str, dict[str, Any]], cluster_source: dict[str, Any]
) -> dict[str, Any]:
    """Report registered-bound outcomes for the three iterated clusters."""

    clusters = cluster_source["quantity_clusters"]["failing_clusters"]
    result = {}
    for name, cluster in clusters.items():
        quantities = cluster["quantities"]
        rows = [observables[quantity] for quantity in quantities]
        failed = [row for row in rows if row["fail_count"] > 0]
        worst = max(rows, key=lambda row: row["maximum_bound_ratio"])
        result[name] = {
            "registered_observable_count": len(rows),
            "expected_observable_count": int(cluster["count"]),
            "passing_observable_count": len(rows) - len(failed),
            "failing_observable_count": len(failed),
            "largest_violation_ratio": (
                max(row["maximum_bound_ratio"] - 1.0 for row in failed)
                if failed
                else 0.0
            ),
            "largest_violation_observable": (
                max(failed, key=lambda row: row["maximum_bound_ratio"])["observable"]
                if failed
                else None
            ),
            "largest_bound_utilisation": {
                "observable": worst["observable"],
                "ratio": worst["maximum_bound_ratio"],
            },
        }
    return result


def _measure_terminal_observables(
    store: Path,
    criterion: dict[str, Any],
    attribution: dict[str, Any],
) -> dict[str, Any]:
    """Repeat six held-out solves and apply all registered terminal bounds."""

    registration = criterion["criterion_family"]["terminal_compiled_parity"][
        "terminal_observable_registration"
    ]
    bounds = registration["bounds"]
    if len(bounds) != 69:
        raise RuntimeError("terminal-observable registration no longer has 69 bounds")
    by_name = {row["observable"]: row for row in bounds}
    observables = {
        name: _initial_observable_record(row) for name, row in by_name.items()
    }
    cases = []
    unregistered = set()
    for shot, slice_index, _row in parity._case_rows(store):
        group = zarr.open_group(str(store / f"{shot}.zarr"), mode="r")["efm"]
        profile, _reference_seed, _reference, _provenance = parity.build_profile(
            group, shot, slice_index, "fcoil_c"
        )
        profile = parity._with_moment_geometry(profile)
        boundary = parity._stored_lcfs(group, slice_index)
        target_current = abs(float(group["plasma_current_c"][slice_index]))
        seed = profile.moment_seed(boundary, target_current)

        def solve(state):
            return profile.solve(
                state, target_current=target_current, **parity.SOLVE_OPTIONS
            )

        eager = solve(seed.flux)
        compiled = jax.jit(solve)(seed.flux)
        jax.block_until_ready(compiled)
        eager_leaves = parity._leaves(parity._named_tree(eager))
        compiled_leaves = parity._leaves(parity._named_tree(compiled))
        if eager_leaves.keys() != compiled_leaves.keys():
            raise RuntimeError("eager and compiled terminal result trees differ")
        missing = sorted(by_name.keys() - eager_leaves.keys())
        if missing:
            raise RuntimeError(f"registered terminal observables are absent: {missing}")
        unregistered.update(eager_leaves.keys() - by_name.keys())
        case_identity = {"shot": int(shot), "slice_index": int(slice_index)}
        case_failures = []
        for name, bound in by_name.items():
            measured = _score_observable(
                eager_leaves[name], compiled_leaves[name], bound
            )
            _accumulate_observable(observables[name], measured, case_identity)
            if not measured["passes"]:
                case_failures.append(name)
        cases.append(
            {
                **case_identity,
                "time_s": float(group["time"][slice_index]),
                "registered_observable_pass_count": len(bounds) - len(case_failures),
                "registered_observable_fail_count": len(case_failures),
                "failed_observables": sorted(case_failures),
                "eager_krylov_conditioning_count": int(
                    eager.fixed_point.krylov_conditioning_count
                ),
                "compiled_krylov_conditioning_count": int(
                    compiled.fixed_point.krylov_conditioning_count
                ),
                "eager_maximum_projected_krylov_condition": float(
                    eager.fixed_point.maximum_projected_krylov_condition
                ),
                "compiled_maximum_projected_krylov_condition": float(
                    compiled.fixed_point.maximum_projected_krylov_condition
                ),
            }
        )
    observable_pass_count = sum(row["fail_count"] == 0 for row in observables.values())
    case_evaluation_pass_count = sum(row["pass_count"] for row in observables.values())
    one_map = attribution["attribution"]["causes"][
        "AMPLIFIED_REPRESENTATION_DIFFERENCE"
    ]
    one_map_bound = criterion["criterion_family"]["terminal_compiled_parity"][
        "one_map_bound"
    ]
    return {
        "cohort": {
            "identity": registration["calibration_cohort"],
            "case_count": len(cases),
            "seed_alignment": "exact shared seed per eager/compiled case",
            "cases": cases,
        },
        "registration": {
            "criterion_source": str(CRITERION_SOURCE.relative_to(HERE)),
            "criterion_source_sha256": _sha256(CRITERION_SOURCE),
            "observable_count": len(bounds),
            "exact_equality_count": int(registration["exact_equality_count"]),
            "dual_envelope_count": int(registration["dual_envelope_count"]),
            "calibration_limit": registration["calibration_limit"],
        },
        "per_observable": [observables[name] for name in sorted(observables)],
        "counts": {
            "observable_pass_count": observable_pass_count,
            "observable_fail_count": len(bounds) - observable_pass_count,
            "case_observable_evaluation_pass_count": case_evaluation_pass_count,
            "case_observable_evaluation_fail_count": (
                len(bounds) * len(cases) - case_evaluation_pass_count
            ),
        },
        "iterated_quantity_clusters": _cluster_summary(observables, attribution),
        "unregistered_terminal_diagnostics": sorted(unregistered),
        "already_measured_one_map_parity": {
            "maximum_relative_difference": float(
                one_map["maximum_single_map_relative_difference"]
            ),
            "registered_sixteen_epsilon_bound": float(one_map_bound["relative_bound"]),
            "registered_sixteen_epsilon_bound_decimal": ("3.5527136788005009e-15"),
            "passes": bool(
                one_map["maximum_single_map_relative_difference"]
                <= one_map_bound["relative_bound"]
            ),
            "bound_over_measurement": float(
                one_map_bound["relative_bound"]
                / one_map["maximum_single_map_relative_difference"]
            ),
            "measurement_source": str(PARITY_ATTRIBUTION_SOURCE.relative_to(HERE)),
            "measurement_source_sha256": _sha256(PARITY_ATTRIBUTION_SOURCE),
        },
    }


def _convergence_verdict(ratios: list[float]) -> str:
    """Classify the measured residual ratios under the fixed three-way rule."""

    if all(0.99 <= ratio <= 1.01 for ratio in ratios):
        return "REPRODUCIBILITY_ONLY"
    if all(ratio < 0.99 for ratio in ratios):
        return "BOTH_MOVED_FAVOURABLY"
    if any(ratio > 1.01 for ratio in ratios):
        return "REPRODUCIBILITY_AT_CONVERGENCE_COST"
    raise RuntimeError("residual ratios do not satisfy any registered verdict rule")


def measure(data: Path, store: Path, output: Path) -> dict[str, Any]:
    """Run the complete convergence-and-observable measurement once."""

    configure_dtypes()
    started = perf_counter()
    source_commit = _git("rev-parse", "HEAD")
    source_tree = _git("rev-parse", "HEAD^{tree}")
    preconditioning = _read_json(PRECONDITIONING_SOURCE)
    factors = tuple(preconditioning["solver_contract"]["factor_ladder"])
    if factors != EXPECTED_FACTORS:
        raise RuntimeError("the pre-conditioning factor ladder changed")
    comparator_by_case = {
        (row["shot"], int(row["frame"])): row
        for row in preconditioning["frame_records"]
    }
    frame_records = [
        _measure_diiid_frame(data, case, comparator_by_case[(case.shot, case.frame)])
        for case in repaired_solve.COHORT
    ]
    favourable = _measure_favourable_frame(data)
    residual_ratios = [
        row["residual_ratio_vs_preconditioning"] for row in frame_records
    ] + [row["residual_ratio_vs_preconditioning"] for row in favourable["rungs"]]
    criterion = _read_json(CRITERION_SOURCE)
    attribution = _read_json(PARITY_ATTRIBUTION_SOURCE)
    terminal_observables = _measure_terminal_observables(store, criterion, attribution)
    verdict = _convergence_verdict(residual_ratios)
    receipt = {
        "artifact": "conditioned_convergence_and_observables",
        "status": "complete",
        "completed_utc": _utc_now(),
        "source_identity": {
            "commit_sha": source_commit,
            "tree_sha": source_tree,
            "driver_sha256": _sha256(Path(__file__)),
        },
        "backend": {
            "platform": jax.default_backend(),
            "device": jax.devices()[0].device_kind,
            "jax_version": jax.__version__,
            "precision": "float64",
        },
        "diiid_conditioned_convergence": {
            "solver_contract": {
                "route": "topology-qualified nonmonotone Newton-Krylov",
                "factor_ladder": list(EXPECTED_FACTORS),
                "newton_steps": repaired_solve.NEWTON_STEPS,
                "gmres_iterations": repaired_solve.GMRES_ITERATIONS,
                "preconditioning_source": str(PRECONDITIONING_SOURCE.relative_to(HERE)),
                "preconditioning_source_sha256": _sha256(PRECONDITIONING_SOURCE),
            },
            "score_blind_frames": frame_records,
            "favourable_frame": favourable,
            "residual_ratio_range": [min(residual_ratios), max(residual_ratios)],
            "conditioning_intervention_count": sum(
                row["krylov_conditioning_count"] for row in frame_records
            )
            + sum(int(row["krylov_conditioning_count"]) for row in favourable["rungs"]),
            "conditioning_interpretation": (
                "The DIII-D carriers use 24 GMRES iterations, outside the calibrated "
                "12-vector damping route, and recorded zero conditioning "
                "interventions. Their residual movement is a final-tree "
                "reproducibility outcome and is not attributed to step damping."
            ),
            "residual_verdict": verdict,
            "diiid_residual_gate_verdict": "DECLINED",
            "diiid_residual_gate_reason": (
                "No defensible DIII-D terminal relative-residual tolerance exists; "
                "the shipped 1e-6 and registered 1e-5 readings remain untraced."
            ),
        },
        "terminal_observable_parity": terminal_observables,
        "runtime_seconds": perf_counter() - started,
        "verdict": verdict,
    }
    _write_json(output, receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    """Return the command-line interface."""

    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, default=repaired_solve.DEFAULT_DATA)
    result.add_argument("--store", type=Path, default=parity.SHOT_STORE)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    result = measure(arguments.data, arguments.store, arguments.output)
    print(
        json.dumps(
            {
                "source_commit": result["source_identity"]["commit_sha"],
                "residual_verdict": result["diiid_conditioned_convergence"][
                    "residual_verdict"
                ],
                "observable_counts": result["terminal_observable_parity"]["counts"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
