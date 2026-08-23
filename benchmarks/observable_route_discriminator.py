"""Separate terminal-label computation differences from state inheritance.

Each frozen MAST case is solved through the eager and compiled routes to retain
the observed state and label differences.  The eager terminal flux is then
held fixed and evaluated once through ``ForwardProfile.observe`` and once
through a jitted leading-axis ``vmap`` of that same entry point.  Exact
disagreement on the shared input assigns ``COMPUTATION_DIFFERS``; exact
agreement assigns ``STATE_INHERITED``.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import mu_0
import zarr

from benchmarks import jitted_eager_parity_gate as parity
from nova.equilibrium.conservation import (
    STENCIL_MARGIN,
    _axisymmetric_divergence,
)
from nova.equilibrium.observation import declared_field_function_squared
from nova.jax.config import configure_dtypes


HERE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    HERE / "docs/figures/derived-observable-parity/computation-discriminator.json"
)
CRITERION_SOURCE = (
    HERE / "docs/figures/forward-operator-refinement/criterion-family.json"
)
CONDITIONED_SOURCE = (
    HERE / "docs/figures/diiid-forward-onboarding/"
    "conditioned-convergence-and-observables.json"
)
TARGET_OBSERVABLES = (
    "conservation.divergence_j",
    "moments.major_radius",
    "moments.volume",
)
MOMENT_OPERATION_ORDER = (
    "support_partition.domain_labels",
    "support_partition.flux_span",
    "support_partition.closed_branch",
    "clipped_support.included_cells",
    "clipped_support.area",
    "clipped_support.radial_first_moment",
    "clipped_support.radial_second_moment",
    "clipped_measure.per_cell_volume",
    "clipped_measure.per_cell_radial_volume",
    "observe_moments.volume_reduction",
    "observe_moments.radial_volume_reduction",
    "observe_moments.major_radius_division",
)
VOLUME_OPERATION_ORDER = (
    "support_partition.domain_labels",
    "support_partition.flux_span",
    "support_partition.closed_branch",
    "clipped_support.included_cells",
    "clipped_support.area",
    "clipped_support.radial_first_moment",
    "clipped_measure.per_cell_volume",
    "observe_moments.volume_reduction",
)
CONSERVATION_OPERATION_ORDER = (
    "conservation.declared_support",
    "conservation.eroded_checked_support",
    "conservation.declared_field_function_squared",
    "conservation.guarded_square_root",
    "conservation.field_function_gradient",
    "conservation.poloidal_current_components",
    "conservation.axisymmetric_current_divergence",
    "conservation.checked_cell_sup_reduction",
    "conservation.current_gradient_scale_reduction",
)
CELL_CENTERED_MOMENT_OPERATION_ORDER = (
    "support_partition.domain_labels",
    "support_partition.flux_span",
    "cell_centered_support.core_mask",
    "cell_centered_measure.area",
    "cell_centered_measure.per_cell_volume",
    "cell_centered_measure.per_cell_radial_volume",
    "observe_moments.volume_reduction",
    "observe_moments.radial_volume_reduction",
    "observe_moments.major_radius_division",
)
CELL_CENTERED_VOLUME_OPERATION_ORDER = (
    "support_partition.domain_labels",
    "support_partition.flux_span",
    "cell_centered_support.core_mask",
    "cell_centered_measure.area",
    "cell_centered_measure.per_cell_volume",
    "observe_moments.volume_reduction",
)


def _git(*arguments: str) -> str:
    """Return one identity from the measured source tree."""

    return subprocess.check_output(
        ["git", *arguments], cwd=HERE, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _sha256(path: Path) -> str:
    """Return the content identity of one evidence input."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one committed JSON evidence input."""

    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write stable finite JSON at the assigned receipt path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _utc_now() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(UTC).isoformat()


def _scalar(value: Any) -> float:
    """Return one scalar array as a Python float."""

    array = np.asarray(value)
    if array.size != 1:
        raise RuntimeError(f"expected one scalar value, received shape {array.shape}")
    return float(array.reshape(()))


def _difference(left: Any, right: Any) -> dict[str, Any]:
    """Return exact and scale-normalised differences between numeric arrays."""

    eager = np.asarray(left)
    transformed = np.asarray(right)
    if eager.shape != transformed.shape:
        raise RuntimeError(
            f"comparison shape changed from {eager.shape} to {transformed.shape}"
        )
    if not np.all(np.isfinite(eager)) or not np.all(np.isfinite(transformed)):
        raise RuntimeError("comparison carries a non-finite value")
    absolute = float(
        np.max(np.abs(transformed.astype(np.float64) - eager.astype(np.float64)))
    )
    scale = max(float(np.max(np.abs(eager))), np.finfo(np.float64).tiny)
    return {
        "exactly_equal": bool(np.array_equal(eager, transformed)),
        "maximum_absolute_difference": absolute,
        "maximum_relative_difference": absolute / scale,
    }


def _ranks(values: np.ndarray) -> np.ndarray:
    """Return average ranks so tied measurements retain a stable correlation."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _coefficient(left: np.ndarray, right: np.ndarray) -> float | None:
    """Return a Pearson coefficient, or none when either vector is constant."""

    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _correlations(state_difference: list[float], observable_difference: list[float]):
    """Return raw Pearson and tied-rank Spearman coefficients."""

    state = np.asarray(state_difference, dtype=float)
    observable = np.asarray(observable_difference, dtype=float)
    if state.shape != (6,) or observable.shape != (6,):
        raise RuntimeError("correlation requires the complete six-case cohort")
    pearson = _coefficient(state, observable)
    spearman = _coefficient(_ranks(state), _ranks(observable))
    return {
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "case_count": 6,
        "interpretation": (
            "undefined because at least one six-case vector is constant"
            if pearson is None or spearman is None
            else "finite correlation across all six cases"
        ),
    }


def _moment_trace(profile, flux, target_current) -> dict[str, jax.Array]:
    """Expose the ordered clipped-support operations used by both moments."""

    partition = profile.operator._support_partition(flux)
    masks, topology, _sample, core_support, _common_support = partition
    closed_branch = masks.core | masks.common_sol
    measure = profile.operator._clipped_integral_measure(partition)
    volume = jnp.sum(measure.volume)
    radial_volume = jnp.sum(measure.radial_volume)
    safe_volume = jnp.where(volume > 0.0, volume, 1.0)
    return {
        "support_partition.domain_labels": masks.label,
        "support_partition.flux_span": topology.flux_span,
        "support_partition.closed_branch": closed_branch,
        "clipped_support.included_cells": core_support.included,
        "clipped_support.area": core_support.area,
        "clipped_support.radial_first_moment": core_support.first_area_moment[:, 0],
        "clipped_support.radial_second_moment": (
            core_support.second_area_moment[:, 0, 0]
        ),
        "clipped_measure.per_cell_volume": measure.volume,
        "clipped_measure.per_cell_radial_volume": measure.radial_volume,
        "observe_moments.volume_reduction": volume,
        "observe_moments.radial_volume_reduction": radial_volume,
        "observe_moments.major_radius_division": radial_volume / safe_volume,
    }


def _cell_centered_moment_trace(profile, flux, target_current) -> dict[str, jax.Array]:
    """Expose ordered cell-centred operations used by the frozen profiles."""

    _current, measure, masks, topology, _amplitude = profile._integral_state(
        flux, target_current=target_current
    )
    volume = jnp.sum(measure.volume)
    radial_volume = jnp.sum(measure.radial_volume)
    safe_volume = jnp.where(volume > 0.0, volume, 1.0)
    return {
        "support_partition.domain_labels": masks.label,
        "support_partition.flux_span": topology.flux_span,
        "cell_centered_support.core_mask": masks.core,
        "cell_centered_measure.area": measure.area,
        "cell_centered_measure.per_cell_volume": measure.volume,
        "cell_centered_measure.per_cell_radial_volume": measure.radial_volume,
        "observe_moments.volume_reduction": volume,
        "observe_moments.radial_volume_reduction": radial_volume,
        "observe_moments.major_radius_division": radial_volume / safe_volume,
    }


def _conservation_trace(profile, flux, target_current) -> dict[str, jax.Array]:
    """Expose the ordered operations producing current-divergence receipts."""

    _current, _integrals, masks, topology, _amplitude = profile._integral_state(
        flux, target_current=target_current
    )
    mesh = profile.lattice
    radius = jnp.asarray(mesh.node_radius)
    support = profile.operator.source.declared_support(masks)
    checked = mesh.erode(support, STENCIL_MARGIN) & mesh.interior()
    squared = declared_field_function_squared(
        profile.operator.source, masks, topology.flux_span
    )
    field_function = jnp.sqrt(jnp.maximum(squared, 0.0))
    function_radial, function_vertical = mesh.gradient(field_function)
    radial_current = -function_vertical / (mu_0 * radius)
    vertical_current = function_radial / (mu_0 * radius)
    divergence = _axisymmetric_divergence(mesh, radial_current, vertical_current)
    checked_sup = jnp.max(jnp.where(checked, jnp.abs(divergence), 0.0))
    radial_gradient = mesh.gradient(radial_current)[1]
    vertical_gradient = mesh.gradient(vertical_current)[0]
    current_scale = jnp.maximum(
        jnp.max(jnp.where(checked, jnp.abs(radial_gradient), 0.0)),
        jnp.max(jnp.where(checked, jnp.abs(vertical_gradient), 0.0)),
    )
    return {
        "conservation.declared_support": support,
        "conservation.eroded_checked_support": checked,
        "conservation.declared_field_function_squared": squared,
        "conservation.guarded_square_root": field_function,
        "conservation.field_function_gradient": jnp.stack(
            [function_radial, function_vertical]
        ),
        "conservation.poloidal_current_components": jnp.stack(
            [radial_current, vertical_current]
        ),
        "conservation.axisymmetric_current_divergence": divergence,
        "conservation.checked_cell_sup_reduction": checked_sup,
        "conservation.current_gradient_scale_reduction": current_scale,
    }


def _first_trace_difference(
    scalar_trace: dict[str, Any],
    batched_trace: dict[str, Any],
    operation_order: tuple[str, ...],
) -> dict[str, Any] | None:
    """Name the earliest ordered intermediate that changes under batching."""

    if scalar_trace.keys() != batched_trace.keys():
        raise RuntimeError("localisation trace structure changed under batching")
    for operation in operation_order:
        difference = _difference(scalar_trace[operation], batched_trace[operation])
        if not difference["exactly_equal"]:
            return {"operation": operation, **difference}
    return None


def _localise(profile, flux, target_current, observable: str) -> dict[str, Any]:
    """Name the first numerical operation changed by the transformed route."""

    if observable.startswith("moments."):
        if profile.operator.use_linear_moments:
            trace = _moment_trace
            operation_order = (
                VOLUME_OPERATION_ORDER
                if observable == "moments.volume"
                else MOMENT_OPERATION_ORDER
            )
        else:
            trace = _cell_centered_moment_trace
            operation_order = (
                CELL_CENTERED_VOLUME_OPERATION_ORDER
                if observable == "moments.volume"
                else CELL_CENTERED_MOMENT_OPERATION_ORDER
            )
    else:
        trace = _conservation_trace
        operation_order = CONSERVATION_OPERATION_ORDER
    scalar_trace = trace(profile, flux, target_current)
    transformed_trace = jax.jit(
        jax.vmap(lambda state, target: trace(profile, state, target))
    )(flux[jnp.newaxis, ...], jnp.asarray([target_current]))
    jax.block_until_ready(transformed_trace)
    batched_trace = jax.tree.map(lambda value: value[0], transformed_trace)
    first = _first_trace_difference(scalar_trace, batched_trace, operation_order)
    if first is None:
        return {
            "operation": "ForwardProfile.observe transformed output assembly",
            "qualification": (
                "The retained intermediate trace is exact; the difference appears "
                "only when the complete observation result is compiled as one tree."
            ),
        }
    return {
        **first,
        "qualification": "first non-identical value in production operation order",
    }


def _case_measurement(store: Path, shot: int, slice_index: int) -> dict[str, Any]:
    """Measure route and shared-state differences for one frozen case."""

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
    jax.block_until_ready((eager, compiled))
    eager_leaves = parity._leaves(parity._named_tree(eager))
    compiled_leaves = parity._leaves(parity._named_tree(compiled))
    if eager_leaves.keys() != compiled_leaves.keys():
        raise RuntimeError("eager and compiled solve result trees differ")

    shared_flux = eager.flux
    scalar_observation = profile.observe(shared_flux, target_current=target_current)
    transformed_observation = jax.jit(
        jax.vmap(lambda state, target: profile.observe(state, target_current=target))
    )(shared_flux[jnp.newaxis, ...], jnp.asarray([target_current]))
    jax.block_until_ready((scalar_observation, transformed_observation))
    scalar_leaves = parity._leaves(parity._named_tree(scalar_observation))
    transformed_leaves = parity._leaves(parity._named_tree(transformed_observation))
    case = {
        "shot": shot,
        "slice_index": slice_index,
        "time_s": float(group["time"][slice_index]),
    }
    state_difference = _difference(eager.flux, compiled.flux)
    observables = {}
    for observable in TARGET_OBSERVABLES:
        route_difference = _difference(
            eager_leaves[observable], compiled_leaves[observable]
        )
        scalar_value = scalar_leaves[observable]
        transformed_value = transformed_leaves[observable][0]
        shared_difference = _difference(scalar_value, transformed_value)
        localisation = None
        if not shared_difference["exactly_equal"]:
            localisation = _localise(profile, shared_flux, target_current, observable)
        observables[observable] = {
            "route_observable_difference": route_difference,
            "shared_state_evaluation": {
                "shared_state_source": "eager terminal flux",
                "scalar_forward_profile_observe": _scalar(scalar_value),
                "jitted_vmap_forward_profile_observe": _scalar(transformed_value),
                **shared_difference,
            },
            "first_structurally_differing_operation": localisation,
        }
    return {
        **case,
        "terminal_state_difference": state_difference,
        "observables": observables,
    }


def _observable_receipt(observable: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one observable's six-case evidence and assign its verdict."""

    rows = [case["observables"][observable] for case in cases]
    shared_differences = [
        row["shared_state_evaluation"]["maximum_absolute_difference"] for row in rows
    ]
    state_differences = [
        case["terminal_state_difference"]["maximum_relative_difference"]
        for case in cases
    ]
    route_differences = [
        row["route_observable_difference"]["maximum_absolute_difference"]
        for row in rows
    ]
    computation_differs = any(value > 0.0 for value in shared_differences)
    localisations = [
        {
            "shot": case["shot"],
            "slice_index": case["slice_index"],
            **row["first_structurally_differing_operation"],
        }
        for case, row in zip(cases, rows, strict=True)
        if row["first_structurally_differing_operation"] is not None
    ]
    first_operation = localisations[0]["operation"] if localisations else None
    return {
        "observable": observable,
        "verdict": "COMPUTATION_DIFFERS" if computation_differs else "STATE_INHERITED",
        "verdict_rule": (
            "COMPUTATION_DIFFERS iff scalar observe and jitted vmap observe are "
            "not exactly equal on any shared terminal state; otherwise STATE_INHERITED"
        ),
        "shared_state_case_count": len(cases),
        "shared_state_maximum_absolute_difference": max(shared_differences),
        "first_structurally_differing_operation": first_operation,
        "localisation_by_differing_case": localisations,
        "difference_against_state_agreement": {
            "state_metric": (
                "max(abs(compiled_flux - eager_flux)) / max(abs(eager_flux))"
            ),
            "observable_metric": (
                "absolute difference between eager-solve and compiled-solve leaf"
            ),
            "correlation": _correlations(state_differences, route_differences),
            "cases": [
                {
                    "shot": case["shot"],
                    "slice_index": case["slice_index"],
                    "terminal_state_maximum_relative_difference": state_difference,
                    "route_observable_maximum_absolute_difference": (
                        observable_difference
                    ),
                }
                for case, state_difference, observable_difference in zip(
                    cases, state_differences, route_differences, strict=True
                )
            ],
        },
        "shared_state_evaluations": [
            {
                "shot": case["shot"],
                "slice_index": case["slice_index"],
                **row["shared_state_evaluation"],
            }
            for case, row in zip(cases, rows, strict=True)
        ],
    }


def _platform_snapshot(receipt: dict[str, Any]) -> dict[str, Any]:
    """Retain one complete platform measurement without nesting comparisons."""

    return {
        "status": receipt["status"],
        "completed_utc": receipt["completed_utc"],
        "source_identity": receipt["source_identity"],
        "backend": receipt["backend"],
        "observables": receipt["observables"],
        "cases": receipt["cases"],
        "verdict_counts": receipt["verdict_counts"],
    }


def _cross_platform_comparison(
    platform_measurements: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Place each platform's decisive observable evidence side by side."""

    rows = []
    for observable in TARGET_OBSERVABLES:
        platforms = {}
        for platform, measurement in platform_measurements.items():
            by_name = {row["observable"]: row for row in measurement["observables"]}
            row = by_name[observable]
            platforms[platform] = {
                "verdict": row["verdict"],
                "shared_state_maximum_absolute_difference": row[
                    "shared_state_maximum_absolute_difference"
                ],
                "first_structurally_differing_operation": row[
                    "first_structurally_differing_operation"
                ],
                "correlation": row["difference_against_state_agreement"]["correlation"],
                "shared_state_evaluations": row["shared_state_evaluations"],
            }
        comparison = {
            "observable": observable,
            "platforms": platforms,
        }
        if observable == "moments.volume" and "cpu" in platforms:
            differences = {
                platform: row["shared_state_maximum_absolute_difference"]
                for platform, row in platforms.items()
            }
            comparison["exact_equality_bound_evidence"] = {
                "cpu_shared_state_difference_is_exactly_zero": (
                    differences["cpu"] == 0.0
                ),
                "shared_state_difference_by_platform": differences,
                "interpretation": (
                    "Exact equality is an execution-platform claim for this "
                    "floating-point observable, not a discrete-type guarantee."
                ),
            }
        rows.append(comparison)
    return rows


def measure(store: Path, output: Path) -> dict[str, Any]:
    """Run the six-case discriminator and persist its complete receipt."""

    configure_dtypes()
    previous_receipt = None
    if output.exists():
        candidate = _read_json(output)
        if candidate.get("artifact") == "observable_route_discriminator":
            previous_receipt = candidate
    criterion = _read_json(CRITERION_SOURCE)
    registration = criterion["criterion_family"]["terminal_compiled_parity"][
        "terminal_observable_registration"
    ]
    registered_names = {row["observable"] for row in registration["bounds"]}
    if not set(TARGET_OBSERVABLES) <= registered_names:
        raise RuntimeError("a target observable is absent from the frozen registration")
    conditioned = _read_json(CONDITIONED_SOURCE)
    failing_names = {
        row["observable"]
        for row in conditioned["terminal_observable_parity"]["per_observable"]
        if row["fail_count"] > 0
    }
    if failing_names != set(TARGET_OBSERVABLES):
        raise RuntimeError(
            "conditioned receipt no longer identifies exactly three target failures"
        )

    cases = [
        _case_measurement(store, shot, slice_index)
        for shot, slice_index, _row in parity._case_rows(store)
    ]
    observables = [
        _observable_receipt(observable, cases) for observable in TARGET_OBSERVABLES
    ]
    measured_platform = jax.default_backend()
    source_platform = conditioned["backend"]["platform"]
    backend_matches_source = measured_platform == source_platform
    receipt = {
        "artifact": "observable_route_discriminator",
        "status": (
            "complete" if backend_matches_source else "provisional_backend_mismatch"
        ),
        "completed_utc": _utc_now(),
        "source_identity": {
            "commit_sha": _git("rev-parse", "HEAD"),
            "tree_sha": _git("rev-parse", "HEAD^{tree}"),
            "driver_sha256": _sha256(Path(__file__)),
        },
        "backend": {
            "platform": measured_platform,
            "device": jax.devices()[0].device_kind,
            "jax_version": jax.__version__,
            "precision": "float64",
        },
        "evidence_sources": {
            "criterion": {
                "path": str(CRITERION_SOURCE.relative_to(HERE)),
                "sha256": _sha256(CRITERION_SOURCE),
            },
            "conditioned_failures": {
                "path": str(CONDITIONED_SOURCE.relative_to(HERE)),
                "sha256": _sha256(CONDITIONED_SOURCE),
                "banked_observable_pass_count": 66,
                "banked_case_observable_evaluation_pass_count": 407,
            },
        },
        "measurement_contract": {
            "case_count": len(cases),
            "shared_terminal_state": "eager terminal flux for each frozen case",
            "scalar_path": "ForwardProfile.observe(shared_flux)",
            "transformed_path": (
                "jax.jit(jax.vmap(ForwardProfile.observe))(shared_flux[None])"
            ),
            "no_repair_attempted": True,
            "backend_alignment": {
                "failing_evidence_platform": source_platform,
                "measurement_platform": measured_platform,
                "matches": backend_matches_source,
                "qualification": (
                    "Verdicts adjudicate the banked failures only when the "
                    "measurement platform matches their source platform."
                ),
            },
        },
        "observables": observables,
        "cases": cases,
        "verdict_counts": {
            "COMPUTATION_DIFFERS": sum(
                row["verdict"] == "COMPUTATION_DIFFERS" for row in observables
            ),
            "STATE_INHERITED": sum(
                row["verdict"] == "STATE_INHERITED" for row in observables
            ),
        },
    }
    platform_measurements = {}
    if previous_receipt is not None:
        retained = previous_receipt.get("platform_measurements")
        if retained is not None:
            platform_measurements.update(retained)
        else:
            previous_platform = previous_receipt["backend"]["platform"]
            platform_measurements[previous_platform] = _platform_snapshot(
                previous_receipt
            )
    platform_measurements[measured_platform] = _platform_snapshot(receipt)
    receipt["platform_measurements"] = platform_measurements
    receipt["cross_platform_observable_comparison"] = _cross_platform_comparison(
        platform_measurements
    )
    _write_json(output, receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    """Return the command-line interface."""

    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--store", type=Path, default=parity.SHOT_STORE)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    result = measure(arguments.store, arguments.output)
    print(
        json.dumps(
            {
                "verdict_counts": result["verdict_counts"],
                "observables": {
                    row["observable"]: row["verdict"] for row in result["observables"]
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
