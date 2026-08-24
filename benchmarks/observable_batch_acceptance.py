"""Run registered label acceptance at distinct forward-solve batch sizes.

Each held-out case supplies one scalar-route reference.  The same aligned seed
is then repeated through ``ForwardProfile.solve_batch`` at every requested
width, and the production acceptance scores all 69 registered terminal labels.
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
import jaxlib
import numpy as np
import zarr

from benchmarks import jitted_eager_parity_gate as parity
from nova.equilibrium.observable_acceptance import (
    evaluate_observable_bound_acceptance,
)
from nova.jax.config import configure_dtypes


HERE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    HERE / "docs/figures/derived-observable-parity/integrated-acceptance.json"
)
DEFAULT_STATE_OUTPUT = (
    HERE / "docs/figures/same-device-label-determinism/state-label-reproducibility.json"
)
DEFAULT_STATE_ARRAY_OUTPUT = DEFAULT_STATE_OUTPUT.with_suffix(".npz")
BANKED_ACCEPTANCE_SOURCE = (
    HERE / "docs/figures/derived-observable-parity/batch-acceptance.json"
)
CRITERION_SOURCE = (
    HERE / "docs/figures/forward-operator-refinement/criterion-family.json"
)
CONDITIONED_SOURCE = (
    HERE / "docs/figures/diiid-forward-onboarding/"
    "conditioned-convergence-and-observables.json"
)
DEFAULT_BATCH_SIZES = (1, 4)
DEFAULT_REPETITIONS = 3
ACCEPTANCE_ENTRY_POINT = (
    "nova.equilibrium.observable_acceptance.evaluate_observable_bound_acceptance"
)
BANKED_ACCEPTANCE_COUNTS = {
    1: {
        "observable_pass_count": 65,
        "case_observable_evaluation_pass_count": 407,
    },
    4: {
        "observable_pass_count": 66,
        "case_observable_evaluation_pass_count": 408,
    },
}


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


def _array_sha256(value: Any) -> str:
    """Return an identity that includes one array's shape, dtype and bytes."""

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _bitwise_unequal_element_count(left: Any, right: Any) -> int:
    """Count elements whose stored byte patterns differ."""

    first = np.ascontiguousarray(np.asarray(left))
    second = np.ascontiguousarray(np.asarray(right))
    if first.shape != second.shape or first.dtype != second.dtype:
        raise ValueError(
            "bitwise comparison requires identical shapes and dtypes; "
            f"received {first.shape}/{first.dtype} and "
            f"{second.shape}/{second.dtype}"
        )
    if first.size == 0:
        return 0
    first_bytes = first.view(np.uint8).reshape(first.size, first.dtype.itemsize)
    second_bytes = second.view(np.uint8).reshape(second.size, second.dtype.itemsize)
    return int(np.count_nonzero(np.any(first_bytes != second_bytes, axis=1)))


def _repetition_difference(values: Any) -> dict[str, Any]:
    """Compare every later repetition bitwise and numerically to the first."""

    arrays = np.asarray(values)
    if arrays.ndim < 1 or arrays.shape[0] < 2:
        raise ValueError("repetition comparison requires at least two arrays")
    reference = arrays[0]
    comparisons = []
    for repetition_index in range(1, arrays.shape[0]):
        candidate = arrays[repetition_index]
        unequal = _bitwise_unequal_element_count(reference, candidate)
        numeric_reference = reference.astype(np.float64)
        numeric_candidate = candidate.astype(np.float64)
        finite = np.isfinite(numeric_reference) & np.isfinite(numeric_candidate)
        if not np.array_equal(
            np.isfinite(numeric_reference), np.isfinite(numeric_candidate)
        ):
            raise RuntimeError("repetitions carry different non-finite patterns")
        if np.any(finite):
            maximum_absolute = float(
                np.max(np.abs(numeric_candidate[finite] - numeric_reference[finite]))
            )
            scale = max(
                float(np.max(np.abs(numeric_reference[finite]))),
                np.finfo(np.float64).tiny,
            )
            maximum_relative = maximum_absolute / scale
        else:
            maximum_absolute = 0.0
            maximum_relative = 0.0
        comparisons.append(
            {
                "repetition": repetition_index + 1,
                "bitwise_unequal_element_count": unequal,
                "maximum_absolute_difference": maximum_absolute,
                "maximum_relative_difference": maximum_relative,
            }
        )
    return {
        "reference_repetition": 1,
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "maximum_bitwise_unequal_element_count": max(
            row["bitwise_unequal_element_count"] for row in comparisons
        ),
        "maximum_absolute_difference": max(
            row["maximum_absolute_difference"] for row in comparisons
        ),
        "maximum_relative_difference": max(
            row["maximum_relative_difference"] for row in comparisons
        ),
    }


def _repeat_reference(value: Any, batch_size: int) -> np.ndarray:
    """Repeat one scalar-route leaf over the measured batch axis."""

    array = np.asarray(value)
    return np.broadcast_to(array, (batch_size, *array.shape)).copy()


def _case_measurement(
    store: Path,
    shot: int,
    slice_index: int,
    batch_sizes: tuple[int, ...],
    registered_names: set[str],
) -> dict[str, Any]:
    """Measure scalar and batched terminal labels for one held-out case."""

    group = zarr.open_group(str(store / f"{shot}.zarr"), mode="r")["efm"]
    profile, _reference_seed, _reference, _provenance = parity.build_profile(
        group, shot, slice_index, "fcoil_c"
    )
    profile = parity._with_moment_geometry(profile)
    boundary = parity._stored_lcfs(group, slice_index)
    target_current = abs(float(group["plasma_current_c"][slice_index]))
    seed = profile.moment_seed(boundary, target_current)

    scalar = profile.solve(
        seed.flux, target_current=target_current, **parity.SOLVE_OPTIONS
    )
    jax.block_until_ready(scalar)
    scalar_leaves = parity._leaves(parity._named_tree(scalar))
    missing = sorted(registered_names - scalar_leaves.keys())
    if missing:
        raise RuntimeError(f"scalar solve omits registered observables: {missing}")

    batches = {}
    for batch_size in batch_sizes:
        initial_flux = jnp.broadcast_to(seed.flux, (batch_size, *seed.flux.shape))
        targets = jnp.full((batch_size,), target_current, dtype=seed.flux.dtype)

        def solve_batch(states, currents):
            return profile.solve_batch(
                states,
                target_current=currents,
                **parity.SOLVE_OPTIONS,
            )

        transformed = jax.jit(solve_batch)(initial_flux, targets)
        jax.block_until_ready(transformed)
        transformed_leaves = parity._leaves(parity._named_tree(transformed))
        missing = sorted(registered_names - transformed_leaves.keys())
        if missing:
            raise RuntimeError(
                f"batch size {batch_size} omits registered observables: {missing}"
            )
        batches[batch_size] = {
            name: np.asarray(transformed_leaves[name]) for name in registered_names
        }

    return {
        "case_id": f"{shot}/{slice_index}",
        "shot": shot,
        "slice_index": slice_index,
        "time_s": float(group["time"][slice_index]),
        "reference": {
            name: np.asarray(scalar_leaves[name]) for name in registered_names
        },
        "batches": batches,
    }


def _stack_inputs(
    cases: list[dict[str, Any]],
    names: set[str],
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Stack case-local results into the acceptance's two leading axes."""

    reference = {
        name: np.stack(
            [_repeat_reference(case["reference"][name], batch_size) for case in cases]
        )
        for name in names
    }
    candidate = {
        name: np.stack([case["batches"][batch_size][name] for case in cases])
        for name in names
    }
    return reference, candidate


def _batch_dependence(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """State explicitly whether each registered verdict changes with width."""

    by_size = {
        result["batch_size"]: {
            row["observable"]: row for row in result["per_observable"]
        }
        for result in results
    }
    names = set.intersection(*(set(rows) for rows in by_size.values()))
    if any(set(rows) != names for rows in by_size.values()):
        raise RuntimeError("batch acceptances do not cover the same observables")
    rows = []
    for name in sorted(names):
        pass_status = {
            str(size): by_size[size][name]["passes"] for size in sorted(by_size)
        }
        case_status = {
            str(size): {
                row["case_id"]: row["passes"] for row in by_size[size][name]["cases"]
            }
            for size in sorted(by_size)
        }
        case_names = set.intersection(*(set(status) for status in case_status.values()))
        if any(set(status) != case_names for status in case_status.values()):
            raise RuntimeError(
                f"batch acceptances do not cover the same cases for {name}"
            )
        aggregate_depends = len(set(pass_status.values())) > 1
        case_depends = any(
            len({case_status[str(size)][case_id] for size in sorted(by_size)}) > 1
            for case_id in case_names
        )
        rows.append(
            {
                "observable": name,
                "pass_status_by_batch_size": pass_status,
                "case_pass_status_by_batch_size": case_status,
                "aggregate_pass_status_depends_on_batch_size": aggregate_depends,
                "case_pass_status_depends_on_batch_size": case_depends,
                "pass_status_depends_on_batch_size": (
                    aggregate_depends or case_depends
                ),
                "case_pass_count_by_batch_size": {
                    str(size): by_size[size][name]["case_pass_count"]
                    for size in sorted(by_size)
                },
                "maximum_absolute_difference_by_batch_size": {
                    str(size): by_size[size][name]["maximum_absolute_difference"]
                    for size in sorted(by_size)
                },
            }
        )
    return rows


def _repetition_snapshot(receipt: dict[str, Any]) -> dict[str, Any]:
    """Retain the verdict-bearing subset of one complete device measurement."""

    return {
        "completed_utc": receipt["completed_utc"],
        "source_identity": receipt["source_identity"],
        "backend": receipt["backend"],
        "batch_results": [
            {
                "batch_size": result["batch_size"],
                "observable_pass_count": result["observable_pass_count"],
                "observable_fail_count": result["observable_fail_count"],
                "case_observable_evaluation_pass_count": result[
                    "case_observable_evaluation_pass_count"
                ],
                "case_observable_evaluation_fail_count": result[
                    "case_observable_evaluation_fail_count"
                ],
                "per_observable": [
                    {
                        "observable": row["observable"],
                        "passes": row["passes"],
                        "case_pass_count": row["case_pass_count"],
                        "case_pass_status": {
                            case["case_id"]: case["passes"] for case in row["cases"]
                        },
                        "maximum_absolute_difference": row[
                            "maximum_absolute_difference"
                        ],
                        "maximum_relative_difference": row[
                            "maximum_relative_difference"
                        ],
                        "maximum_bound_ratio": row["maximum_bound_ratio"],
                    }
                    for row in result["per_observable"]
                ],
            }
            for result in receipt["batch_results"]
        ],
    }


def _repetition_stability(repetitions: list[dict[str, Any]]) -> dict[str, Any]:
    """Report run-to-run verdict stability separately from batch dependence."""

    source_identities = [repetition["source_identity"] for repetition in repetitions]
    if any(identity != source_identities[0] for identity in source_identities[1:]):
        raise RuntimeError("repeated measurements do not share one source identity")

    sizes = {
        result["batch_size"]
        for repetition in repetitions
        for result in repetition["batch_results"]
    }
    summaries = []
    for size in sorted(sizes):
        results = [
            next(
                result
                for result in repetition["batch_results"]
                if result["batch_size"] == size
            )
            for repetition in repetitions
        ]
        by_repetition = [
            {row["observable"]: row for row in result["per_observable"]}
            for result in results
        ]
        names = set.intersection(*(set(rows) for rows in by_repetition))
        if any(set(rows) != names for rows in by_repetition):
            raise RuntimeError("repeated measurements cover different observables")
        changing = sorted(
            name
            for name in names
            if len({rows[name]["passes"] for rows in by_repetition}) > 1
        )
        changing_cases = []
        for name in sorted(names):
            case_ids = set.intersection(
                *(set(rows[name]["case_pass_status"]) for rows in by_repetition)
            )
            if any(
                set(rows[name]["case_pass_status"]) != case_ids
                for rows in by_repetition
            ):
                raise RuntimeError(
                    f"repeated measurements cover different cases for {name}"
                )
            for case_id in sorted(case_ids):
                statuses = {
                    rows[name]["case_pass_status"][case_id] for rows in by_repetition
                }
                if len(statuses) > 1:
                    changing_cases.append({"observable": name, "case_id": case_id})
        pass_counts = [result["observable_pass_count"] for result in results]
        case_pass_counts = [
            result["case_observable_evaluation_pass_count"] for result in results
        ]
        summaries.append(
            {
                "batch_size": size,
                "observable_pass_count_by_repetition": pass_counts,
                "observable_pass_count_is_stable": len(set(pass_counts)) == 1,
                "case_observable_evaluation_pass_count_by_repetition": (
                    case_pass_counts
                ),
                "case_observable_evaluation_pass_count_is_stable": (
                    len(set(case_pass_counts)) == 1
                ),
                "pass_status_changing_observables": changing,
                "pass_status_changing_observable_count": len(changing),
                "case_pass_status_changes": changing_cases,
                "case_pass_status_change_count": len(changing_cases),
            }
        )
    return {
        "repetition_count": len(repetitions),
        "same_source_required": True,
        "batch_sizes": summaries,
        "all_observable_pass_counts_stable": all(
            row["observable_pass_count_is_stable"] for row in summaries
        ),
        "all_case_observable_evaluation_pass_counts_stable": all(
            row["case_observable_evaluation_pass_count_is_stable"] for row in summaries
        ),
        "all_aggregate_counts_stable": all(
            row["observable_pass_count_is_stable"]
            and row["case_observable_evaluation_pass_count_is_stable"]
            for row in summaries
        ),
        "qualification": (
            "run-to-run variation is reported independently from batch-size "
            "variation and prevents interpreting one device run as a certificate"
        ),
    }


def _failure_verdict(observable: str) -> tuple[str, str]:
    """Return the integrated discriminator verdict and reason for one failure."""

    if observable in {"moments.major_radius", "moments.volume"}:
        return (
            "STATE_INHERITED",
            "The integrated shared-state discriminator is bitwise equal for this "
            "observable on all six held-out H200 cases. Any remaining scalar-solve "
            "versus batched-solve difference is inherited from their terminal "
            "states, so neither the repaired computation nor its registered bound "
            "is moved by this acceptance run.",
        )
    if observable == "conservation.divergence_j":
        return (
            "COMPUTATION_DIFFERS_REPAIR_REFUSED",
            "The integrated shared-state discriminator still localises a route "
            "difference to the fitted field-function gradient. Fixed-association "
            "experiments moved the scalar answer while retaining a route difference, "
            "so that change was refused; the registered bound remains unchanged.",
        )
    if observable == "conservation.divergence_b":
        return (
            "UNADJUDICATED_ACCEPTANCE_FAILURE",
            "This observable was not one of the three discriminator failures and "
            "appeared only in the reusable acceptance measurements. It remains a "
            "registered acceptance failure with no computation-path conviction "
            "authorising a repair or bound change.",
        )
    return (
        "UNADJUDICATED_ACCEPTANCE_FAILURE",
        "The registered bound fails in the integrated acceptance measurement, but "
        "the discriminator evidence does not identify an authorised computation or "
        "bound change for this observable.",
    )


def _remaining_failures(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retain one reasoned verdict for every bound failing at any measured width."""

    rows_by_observable: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for result in results:
        for row in result["per_observable"]:
            if not row["passes"]:
                rows_by_observable.setdefault(row["observable"], []).append(
                    (result["batch_size"], row)
                )

    failures = []
    for observable, rows in sorted(rows_by_observable.items()):
        verdict, reason = _failure_verdict(observable)
        failures.append(
            {
                "observable": observable,
                "verdict": verdict,
                "reason": reason,
                "batch_results": [
                    {
                        "batch_size": batch_size,
                        "case_pass_count": row["case_pass_count"],
                        "case_fail_count": row["case_fail_count"],
                        "failing_case_ids": [
                            case["case_id"]
                            for case in row["cases"]
                            if not case["passes"]
                        ],
                        "maximum_absolute_difference": row[
                            "maximum_absolute_difference"
                        ],
                        "maximum_relative_difference": row[
                            "maximum_relative_difference"
                        ],
                        "maximum_bound_ratio": row["maximum_bound_ratio"],
                    }
                    for batch_size, row in rows
                ],
            }
        )
    return failures


def _repeated_remaining_failures(
    repetitions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Adjudicate the union of failures retained across repeated runs."""

    occurrences: dict[str, list[dict[str, Any]]] = {}
    for repetition_index, repetition in enumerate(repetitions, start=1):
        for result in repetition["batch_results"]:
            for row in result["per_observable"]:
                if row["passes"]:
                    continue
                occurrences.setdefault(row["observable"], []).append(
                    {
                        "repetition": repetition_index,
                        "batch_size": result["batch_size"],
                        "case_pass_count": row["case_pass_count"],
                        "case_fail_count": len(row["case_pass_status"])
                        - row["case_pass_count"],
                        "failing_case_ids": sorted(
                            case_id
                            for case_id, passes in row["case_pass_status"].items()
                            if not passes
                        ),
                        "maximum_absolute_difference": row[
                            "maximum_absolute_difference"
                        ],
                        "maximum_relative_difference": row[
                            "maximum_relative_difference"
                        ],
                        "maximum_bound_ratio": row["maximum_bound_ratio"],
                    }
                )

    failures = []
    latest_repetition = len(repetitions)
    for observable, rows in sorted(occurrences.items()):
        verdict, reason = _failure_verdict(observable)
        failures.append(
            {
                "observable": observable,
                "verdict": verdict,
                "reason": reason,
                "failed_in_latest_repetition": any(
                    row["repetition"] == latest_repetition for row in rows
                ),
                "occurrences": rows,
            }
        )
    return failures


def _case_repetition_measurement(
    store: Path,
    shot: int,
    slice_index: int,
    batch_sizes: tuple[int, ...],
    registered_names: set[str],
    repetitions: int,
) -> dict[str, Any]:
    """Repeat each compiled case/width solve without rebuilding its executable."""

    group = zarr.open_group(str(store / f"{shot}.zarr"), mode="r")["efm"]
    profile, _reference_seed, _reference, _provenance = parity.build_profile(
        group, shot, slice_index, "fcoil_c"
    )
    profile = parity._with_moment_geometry(profile)
    boundary = parity._stored_lcfs(group, slice_index)
    target_current = abs(float(group["plasma_current_c"][slice_index]))
    seed = profile.moment_seed(boundary, target_current)

    scalar = profile.solve(
        seed.flux, target_current=target_current, **parity.SOLVE_OPTIONS
    )
    jax.block_until_ready(scalar)
    scalar_leaves = parity._leaves(parity._named_tree(scalar))
    missing = sorted(registered_names - scalar_leaves.keys())
    if missing:
        raise RuntimeError(f"scalar solve omits registered observables: {missing}")

    batches = {}
    for batch_size in batch_sizes:
        initial_flux = np.broadcast_to(
            np.asarray(seed.flux), (batch_size, *seed.flux.shape)
        ).copy()
        targets = np.full(
            (batch_size,), target_current, dtype=np.asarray(seed.flux).dtype
        )

        def solve_batch(states, currents):
            return profile.solve_batch(
                states,
                target_current=currents,
                **parity.SOLVE_OPTIONS,
            )

        compiled_solve = jax.jit(solve_batch)
        repeated_flux = []
        repeated_observables = {name: [] for name in registered_names}
        input_identities = []
        for _repetition_index in range(repetitions):
            states_copy = initial_flux.copy()
            targets_copy = targets.copy()
            input_identities.append(
                {
                    "initial_flux_sha256": _array_sha256(states_copy),
                    "target_current_sha256": _array_sha256(targets_copy),
                }
            )
            transformed = compiled_solve(
                jnp.asarray(states_copy), jnp.asarray(targets_copy)
            )
            jax.block_until_ready(transformed)
            transformed_leaves = parity._leaves(parity._named_tree(transformed))
            missing = sorted(registered_names - transformed_leaves.keys())
            if missing:
                raise RuntimeError(
                    f"batch size {batch_size} omits registered observables: {missing}"
                )
            repeated_flux.append(np.asarray(transformed.flux))
            for name in registered_names:
                repeated_observables[name].append(np.asarray(transformed_leaves[name]))
        if any(identity != input_identities[0] for identity in input_identities[1:]):
            raise RuntimeError("repetition inputs are not byte-identical")
        batches[batch_size] = {
            "input_identity": input_identities[0],
            "flux": np.stack(repeated_flux),
            "observables": {
                name: np.stack(values) for name, values in repeated_observables.items()
            },
        }

    return {
        "case_id": f"{shot}/{slice_index}",
        "shot": shot,
        "slice_index": slice_index,
        "time_s": float(group["time"][slice_index]),
        "reference": {
            name: np.asarray(scalar_leaves[name]) for name in registered_names
        },
        "batches": batches,
    }


def _repetition_acceptance(
    cases: list[dict[str, Any]],
    names: set[str],
    bounds: list[dict[str, Any]],
    batch_sizes: tuple[int, ...],
    repetitions: int,
) -> list[dict[str, Any]]:
    """Score every retained repetition through the registered acceptance."""

    case_ids = [case["case_id"] for case in cases]
    results = []
    for repetition_index in range(repetitions):
        batch_results = []
        for batch_size in batch_sizes:
            reference = {
                name: np.stack(
                    [
                        _repeat_reference(case["reference"][name], batch_size)
                        for case in cases
                    ]
                )
                for name in names
            }
            candidate = {
                name: np.stack(
                    [
                        case["batches"][batch_size]["observables"][name][
                            repetition_index
                        ]
                        for case in cases
                    ]
                )
                for name in names
            }
            batch_results.append(
                evaluate_observable_bound_acceptance(
                    reference=reference,
                    candidate=candidate,
                    registration=bounds,
                    case_ids=case_ids,
                    batch_size=batch_size,
                )
            )
        results.append(
            {
                "repetition": repetition_index + 1,
                "batch_results": batch_results,
            }
        )
    return results


def _acceptance_case(
    acceptance: list[dict[str, Any]],
    repetition: int,
    batch_size: int,
    observable: str,
    case_id: str,
) -> dict[str, Any]:
    """Select one case verdict from the retained acceptance tree."""

    repeated = next(row for row in acceptance if row["repetition"] == repetition)
    batch = next(
        row for row in repeated["batch_results"] if row["batch_size"] == batch_size
    )
    measured = next(
        row for row in batch["per_observable"] if row["observable"] == observable
    )
    return next(row for row in measured["cases"] if row["case_id"] == case_id)


def _case_reproducibility(
    cases: list[dict[str, Any]],
    acceptance: list[dict[str, Any]],
    bounds: list[dict[str, Any]],
    batch_sizes: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarise state and all-label repetition differences per case and width."""

    rows = []
    pass_status_changes = []
    by_name = {row["observable"]: row for row in bounds}
    for case in cases:
        for batch_size in batch_sizes:
            state = _repetition_difference(case["batches"][batch_size]["flux"])
            observables = []
            first_differing_leaf = None
            for name in sorted(by_name):
                values = case["batches"][batch_size]["observables"][name]
                difference = _repetition_difference(values)
                statuses = [
                    _acceptance_case(
                        acceptance,
                        repetition,
                        batch_size,
                        name,
                        case["case_id"],
                    )["passes"]
                    for repetition in range(1, values.shape[0] + 1)
                ]
                if (
                    first_differing_leaf is None
                    and difference["maximum_bitwise_unequal_element_count"] > 0
                ):
                    first_differing_leaf = name
                observable_row = {
                    "observable": name,
                    **difference,
                    "case_pass_status_by_repetition": statuses,
                    "pass_status_changes": len(set(statuses)) > 1,
                }
                observables.append(observable_row)
                if len(set(statuses)) > 1:
                    registration = by_name[name]
                    flattened = values.reshape(values.shape[0], -1)
                    change = {
                        "case_id": case["case_id"],
                        "batch_size": batch_size,
                        "observable": name,
                        "criterion_kind": registration["criterion_kind"],
                        "case_pass_status_by_repetition": statuses,
                        "maximum_absolute_value_by_repetition": [
                            float(np.max(np.abs(value.astype(np.float64))))
                            for value in flattened
                        ],
                        "reference_maximum_absolute_value": float(
                            np.max(
                                np.abs(
                                    np.asarray(case["reference"][name]).astype(
                                        np.float64
                                    )
                                )
                            )
                        ),
                        "acceptance_members_by_repetition": [
                            _acceptance_case(
                                acceptance,
                                repetition,
                                batch_size,
                                name,
                                case["case_id"],
                            )["members"]
                            for repetition in range(1, values.shape[0] + 1)
                        ],
                    }
                    if flattened.shape[1] <= 16:
                        change["values_by_repetition"] = flattened.tolist()
                    if registration["criterion_kind"] == "banked_dual_envelope":
                        change.update(
                            absolute_bound=float(registration["absolute_bound"]),
                            relative_bound=float(registration["relative_bound"]),
                        )
                    pass_status_changes.append(change)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "batch_size": batch_size,
                    "state_verdict": (
                        "STATE_REPRODUCIBLE"
                        if state["maximum_bitwise_unequal_element_count"] == 0
                        else "STATE_VARIES"
                    ),
                    "state": state,
                    "labels_move": first_differing_leaf is not None,
                    "first_differing_leaf": first_differing_leaf,
                    "observables": observables,
                }
            )
    return rows, pass_status_changes


def _write_repetition_arrays(
    path: Path,
    cases: list[dict[str, Any]],
    names: set[str],
    batch_sizes: tuple[int, ...],
) -> dict[str, Any]:
    """Write lossless arrays with explicit repetition and case axes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "case_ids": np.asarray([case["case_id"] for case in cases]),
        "batch_sizes": np.asarray(batch_sizes, dtype=np.int64),
        "repetitions": np.arange(
            1,
            next(iter(cases[0]["batches"].values()))["flux"].shape[0] + 1,
            dtype=np.int64,
        ),
    }
    manifest: dict[str, Any] = {
        "terminal_flux": {},
        "observables": {},
        "references": {},
    }
    for batch_size in batch_sizes:
        flux_key = f"terminal_flux_width_{batch_size}"
        arrays[flux_key] = np.stack(
            [case["batches"][batch_size]["flux"] for case in cases], axis=1
        )
        manifest["terminal_flux"][str(batch_size)] = {
            "key": flux_key,
            "axis_order": [
                "repetition",
                "case",
                "batch_member",
                *[
                    f"flux_dimension_{index}"
                    for index in range(arrays[flux_key].ndim - 3)
                ],
            ],
            "shape": list(arrays[flux_key].shape),
            "dtype": arrays[flux_key].dtype.name,
        }
    for index, name in enumerate(sorted(names)):
        reference_key = f"reference_observable_{index:02d}"
        arrays[reference_key] = np.stack([case["reference"][name] for case in cases])
        manifest["references"][name] = {
            "key": reference_key,
            "axis_order": [
                "case",
                *[
                    f"observable_dimension_{dimension}"
                    for dimension in range(arrays[reference_key].ndim - 1)
                ],
            ],
            "shape": list(arrays[reference_key].shape),
            "dtype": arrays[reference_key].dtype.name,
        }
        manifest["observables"][name] = {}
        for batch_size in batch_sizes:
            key = f"observable_{index:02d}_width_{batch_size}"
            arrays[key] = np.stack(
                [case["batches"][batch_size]["observables"][name] for case in cases],
                axis=1,
            )
            manifest["observables"][name][str(batch_size)] = {
                "key": key,
                "axis_order": [
                    "repetition",
                    "case",
                    "batch_member",
                    *[
                        f"observable_dimension_{dimension}"
                        for dimension in range(arrays[key].ndim - 3)
                    ],
                ],
                "shape": list(arrays[key].shape),
                "dtype": arrays[key].dtype.name,
            }
    np.savez_compressed(path, **arrays)
    return manifest


def _runtime_identity() -> dict[str, Any]:
    """Return the allocation, executable environment and selected device."""

    flag_names = (
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "JAX_COMPILATION_CACHE_DIR",
        "JAX_ENABLE_X64",
        "JAX_PLATFORMS",
        "NOVA_COMPILATION_CACHE",
        "NVIDIA_TF32_OVERRIDE",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    )
    cache_directory = jax.config.jax_compilation_cache_dir
    gpu_identity = None
    try:
        gpu_identity = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except FileNotFoundError, subprocess.CalledProcessError:
        pass
    return {
        "backend": {
            "platform": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "device_kind": jax.devices()[0].device_kind,
            "gpu_identity": gpu_identity,
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "precision": "float64",
        },
        "allocation": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_reservation": os.environ.get("SLURM_JOB_RESERVATION"),
            "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "process_id": os.getpid(),
        },
        "environment_flags": {name: os.environ.get(name) for name in flag_names},
        "compilation_cache": {
            "directory": str(cache_directory) if cache_directory else None,
            "enabled": cache_directory is not None,
            "nova_setting": os.environ.get("NOVA_COMPILATION_CACHE"),
        },
    }


def measure_state_label_reproducibility(
    store: Path,
    output: Path,
    array_output: Path,
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
    repetitions: int = DEFAULT_REPETITIONS,
) -> dict[str, Any]:
    """Retain and compare repeated terminal states and all registered labels."""

    configure_dtypes()
    batch_sizes = tuple(sorted(set(batch_sizes)))
    if set(batch_sizes) != set(DEFAULT_BATCH_SIZES):
        raise ValueError("state reproducibility requires both registered widths")
    if repetitions < 3:
        raise ValueError("state reproducibility requires at least three repetitions")
    criterion = _read_json(CRITERION_SOURCE)
    registration = criterion["criterion_family"]["terminal_compiled_parity"][
        "terminal_observable_registration"
    ]
    bounds = registration["bounds"]
    registered_names = {row["observable"] for row in bounds}
    if len(bounds) != 69 or len(registered_names) != 69:
        raise RuntimeError("terminal-observable registration no longer has 69 bounds")

    cases = [
        _case_repetition_measurement(
            store,
            shot,
            slice_index,
            batch_sizes,
            registered_names,
            repetitions,
        )
        for shot, slice_index, _row in parity._case_rows(store)
    ]
    if len(cases) != 6:
        raise RuntimeError("state reproducibility requires all six held-out cases")
    acceptance = _repetition_acceptance(
        cases,
        registered_names,
        bounds,
        batch_sizes,
        repetitions,
    )
    case_results, pass_status_changes = _case_reproducibility(
        cases, acceptance, bounds, batch_sizes
    )
    array_manifest = _write_repetition_arrays(
        array_output, cases, registered_names, batch_sizes
    )
    runtime = _runtime_identity()
    backend_matches = (
        runtime["backend"]["platform"] == "gpu"
        and "H200" in runtime["backend"]["device_kind"]
    )
    first_differing_leaf = next(
        (
            row["first_differing_leaf"]
            for row in case_results
            if row["first_differing_leaf"] is not None
        ),
        None,
    )
    receipt = {
        "artifact": "state_label_reproducibility",
        "status": "complete" if backend_matches else "provisional_backend_mismatch",
        "completed_utc": _utc_now(),
        "source_identity": {
            "commit_sha": _git("rev-parse", "HEAD"),
            "tree_sha": _git("rev-parse", "HEAD^{tree}"),
            "driver_sha256": _sha256(Path(__file__)),
            "acceptance_sha256": _sha256(
                HERE / "nova/equilibrium/observable_acceptance.py"
            ),
            "criterion_sha256": _sha256(CRITERION_SOURCE),
        },
        **runtime,
        "measurement_contract": {
            "case_count": len(cases),
            "registered_observable_count": len(registered_names),
            "batch_sizes": list(batch_sizes),
            "repetition_count": repetitions,
            "process_count": 1,
            "allocation_count": 1,
            "solve_callable_creation": "once per case and batch width",
            "comparison_reference": "repetition 1",
            "relative_difference_scale": (
                "maximum absolute value of repetition 1, floored at float64 tiny"
            ),
            "input_identity": (
                "fresh host copies with matching dtype, shape and byte SHA-256"
            ),
            "result_blocking": "every repeated solve is blocked before retention",
        },
        "array_artifact": {
            "path": str(array_output.relative_to(HERE)),
            "sha256": _sha256(array_output),
            "format": "numpy savez compressed, lossless",
            "allow_pickle_required": False,
            "manifest": array_manifest,
        },
        "cases": [
            {
                "case_id": case["case_id"],
                "shot": case["shot"],
                "slice_index": case["slice_index"],
                "time_s": case["time_s"],
                "input_identity_by_batch_size": {
                    str(size): case["batches"][size]["input_identity"]
                    for size in batch_sizes
                },
            }
            for case in cases
        ],
        "case_results": case_results,
        "acceptance_repetitions": acceptance,
        "pass_status_changes": pass_status_changes,
        "pass_status_change_count": len(pass_status_changes),
        "first_differing_leaf": first_differing_leaf,
        "state_reproducible_case_width_count": sum(
            row["state_verdict"] == "STATE_REPRODUCIBLE" for row in case_results
        ),
        "state_varies_case_width_count": sum(
            row["state_verdict"] == "STATE_VARIES" for row in case_results
        ),
        "label_movement_case_width_count": sum(
            row["labels_move"] for row in case_results
        ),
    }
    _write_json(output, receipt)
    return receipt


def measure(
    store: Path,
    output: Path,
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
) -> dict[str, Any]:
    """Measure and persist observable-bound acceptance at each batch size."""

    configure_dtypes()
    previous_receipt = None
    if output.exists():
        candidate = _read_json(output)
        if candidate.get("artifact") == "observable_batch_acceptance":
            previous_receipt = candidate
    if len(set(batch_sizes)) < 2 or any(size < 1 for size in batch_sizes):
        raise ValueError("measurement requires at least two distinct positive sizes")
    batch_sizes = tuple(sorted(set(batch_sizes)))
    criterion = _read_json(CRITERION_SOURCE)
    registration = criterion["criterion_family"]["terminal_compiled_parity"][
        "terminal_observable_registration"
    ]
    bounds = registration["bounds"]
    if len(bounds) != 69:
        raise RuntimeError("terminal-observable registration no longer has 69 bounds")
    registered_names = {row["observable"] for row in bounds}
    if len(registered_names) != len(bounds):
        raise RuntimeError("terminal-observable registration contains duplicates")

    cases = [
        _case_measurement(
            store,
            shot,
            slice_index,
            batch_sizes,
            registered_names,
        )
        for shot, slice_index, _row in parity._case_rows(store)
    ]
    case_ids = [case["case_id"] for case in cases]
    batch_results = []
    for batch_size in batch_sizes:
        reference, candidate = _stack_inputs(cases, registered_names, batch_size)
        batch_results.append(
            evaluate_observable_bound_acceptance(
                reference=reference,
                candidate=candidate,
                registration=bounds,
                case_ids=case_ids,
                batch_size=batch_size,
            )
        )
    if {row["acceptance_entry_point"] for row in batch_results} != {
        ACCEPTANCE_ENTRY_POINT
    }:
        raise RuntimeError("throughput rungs invoked different acceptance entries")
    measured_sizes = {result["batch_size"] for result in batch_results}
    if measured_sizes != set(BANKED_ACCEPTANCE_COUNTS):
        raise RuntimeError(
            "integrated acceptance must measure the two banked batch widths"
        )

    dependence = _batch_dependence(batch_results)
    conditioned = _read_json(CONDITIONED_SOURCE)
    source_platform = conditioned["backend"]["platform"]
    measured_platform = jax.default_backend()
    backend_matches = source_platform == measured_platform
    receipt = {
        "artifact": "observable_batch_acceptance",
        "status": "complete" if backend_matches else "provisional_backend_mismatch",
        "completed_utc": _utc_now(),
        "source_identity": {
            "commit_sha": _git("rev-parse", "HEAD"),
            "tree_sha": _git("rev-parse", "HEAD^{tree}"),
            "driver_sha256": _sha256(Path(__file__)),
            "acceptance_sha256": _sha256(
                HERE / "nova/equilibrium/observable_acceptance.py"
            ),
        },
        "backend": {
            "platform": measured_platform,
            "device": jax.devices()[0].device_kind,
            "jax_version": jax.__version__,
            "precision": "float64",
            "matches_banked_failure_platform": backend_matches,
        },
        "measurement_contract": {
            "acceptance_entry_point": ACCEPTANCE_ENTRY_POINT,
            "throughput_rung_call": (
                "each measured batch size calls the same acceptance entry point"
            ),
            "scalar_reference_path": "ForwardProfile.solve",
            "batch_candidate_path": "jax.jit(ForwardProfile.solve_batch)",
            "seed_alignment": (
                "each case's scalar seed is repeated exactly over the batch axis"
            ),
            "case_count": len(cases),
            "registered_bound_count": len(bounds),
            "batch_sizes": list(batch_sizes),
            "batch_size_dependence_rule": (
                "true iff a bound's aggregate pass or any held-out case pass "
                "status differs between measured sizes"
            ),
        },
        "evidence_sources": {
            "criterion": {
                "path": str(CRITERION_SOURCE.relative_to(HERE)),
                "sha256": _sha256(CRITERION_SOURCE),
                "calibration_limit": registration["calibration_limit"],
            },
            "banked_failures": {
                "path": str(CONDITIONED_SOURCE.relative_to(HERE)),
                "sha256": _sha256(CONDITIONED_SOURCE),
                "platform": source_platform,
            },
            "banked_acceptance": {
                "path": str(BANKED_ACCEPTANCE_SOURCE.relative_to(HERE)),
                "sha256": _sha256(BANKED_ACCEPTANCE_SOURCE),
            },
        },
        "cases": [
            {
                "case_id": case["case_id"],
                "shot": case["shot"],
                "slice_index": case["slice_index"],
                "time_s": case["time_s"],
            }
            for case in cases
        ],
        "batch_results": batch_results,
        "banked_acceptance_comparison": [
            {
                "batch_size": result["batch_size"],
                "registered_bound_count": result["registered_bound_count"],
                "case_observable_evaluation_count": (
                    result["case_observable_evaluation_pass_count"]
                    + result["case_observable_evaluation_fail_count"]
                ),
                "integrated_observable_pass_count": result["observable_pass_count"],
                "banked_observable_pass_count": BANKED_ACCEPTANCE_COUNTS[
                    result["batch_size"]
                ]["observable_pass_count"],
                "integrated_case_observable_evaluation_pass_count": result[
                    "case_observable_evaluation_pass_count"
                ],
                "banked_case_observable_evaluation_pass_count": (
                    BANKED_ACCEPTANCE_COUNTS[result["batch_size"]][
                        "case_observable_evaluation_pass_count"
                    ]
                ),
            }
            for result in batch_results
        ],
        "remaining_failures": _remaining_failures(batch_results),
        "per_observable_batch_dependence": dependence,
        "batch_dependent_bound_count": sum(
            row["pass_status_depends_on_batch_size"] for row in dependence
        ),
        "acceptance_passes_all_measured_batch_sizes": all(
            result["passes"] for result in batch_results
        ),
    }
    receipt["verdict"] = (
        "PASS" if receipt["acceptance_passes_all_measured_batch_sizes"] else "FAIL"
    )
    repetitions = []
    if previous_receipt is not None:
        repetitions.extend(previous_receipt.get("measurement_repetitions", []))
        if not repetitions:
            repetitions.append(_repetition_snapshot(previous_receipt))
        repetitions = [
            repetition
            for repetition in repetitions
            if repetition["source_identity"] == receipt["source_identity"]
        ]
    repetitions.append(_repetition_snapshot(receipt))
    receipt["measurement_repetitions"] = repetitions
    receipt["repetition_stability"] = _repetition_stability(repetitions)
    receipt["remaining_failures"] = _repeated_remaining_failures(repetitions)
    _write_json(output, receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    """Return the command-line interface."""

    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--store", type=Path, default=parity.SHOT_STORE)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
    )
    result.add_argument(
        "--state-output",
        type=Path,
        help="write the repeated terminal-state and label receipt",
    )
    result.add_argument(
        "--state-array-output",
        type=Path,
        default=DEFAULT_STATE_ARRAY_OUTPUT,
        help="write every repeated state and registered label losslessly",
    )
    result.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help="same-process repetitions for the state and label measurement",
    )
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.state_output is None:
        result = measure(
            arguments.store,
            arguments.output,
            tuple(arguments.batch_sizes),
        )
    else:
        result = measure_state_label_reproducibility(
            arguments.store,
            arguments.state_output,
            arguments.state_array_output,
            tuple(arguments.batch_sizes),
            arguments.repetitions,
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "artifact": result["artifact"],
                "output": str(arguments.state_output or arguments.output),
                **(
                    {
                        "verdict": result["verdict"],
                        "batch_dependent_bound_count": result[
                            "batch_dependent_bound_count"
                        ],
                        "batch_results": [
                            {
                                "batch_size": row["batch_size"],
                                "observable_pass_count": row["observable_pass_count"],
                                "registered_bound_count": row["registered_bound_count"],
                            }
                            for row in result["batch_results"]
                        ],
                    }
                    if result["artifact"] == "observable_batch_acceptance"
                    else {
                        "state_reproducible_case_width_count": result[
                            "state_reproducible_case_width_count"
                        ],
                        "state_varies_case_width_count": result[
                            "state_varies_case_width_count"
                        ],
                        "label_movement_case_width_count": result[
                            "label_movement_case_width_count"
                        ],
                        "first_differing_leaf": result["first_differing_leaf"],
                    }
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
