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
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    result = measure(
        arguments.store,
        arguments.output,
        tuple(arguments.batch_sizes),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "verdict": result["verdict"],
                "batch_dependent_bound_count": result["batch_dependent_bound_count"],
                "batch_results": [
                    {
                        "batch_size": row["batch_size"],
                        "observable_pass_count": row["observable_pass_count"],
                        "registered_bound_count": row["registered_bound_count"],
                    }
                    for row in result["batch_results"]
                ],
                "output": str(arguments.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
