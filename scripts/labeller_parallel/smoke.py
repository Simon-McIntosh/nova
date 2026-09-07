#!/usr/bin/env python3
"""Compare the real corpus scheduler with the sequential shard writer."""

from __future__ import annotations

import argparse
import builtins
from contextlib import redirect_stdout
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Sequence

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nova.equilibrium.steering_frames import SESSION_GROUP  # noqa: E402
from scripts.labeller_batch import shard as sequential_shard  # noqa: E402
from scripts.labeller_batch.shard import (  # noqa: E402
    DEFAULT_COHORT_REPORT,
    DEFAULT_MANIFEST,
    LabellerPrograms,
    decoder_corpus,
    label_shot,
    prepare_labeller,
)
from scripts.labeller_parallel.scheduler import (  # noqa: E402
    CorpusScheduler,
    SequentialCompiledEngine,
    SourceIdentity,
    admitted_quartile_rows,
    load_shot,
)

SHOT_COUNT = 16
MANIFEST_TIMING_FIELDS = {
    "wall_seconds",
    "free_wall_seconds",
    "conditioned_wall_seconds",
}
SESSION_TIMING_VARIABLES = {"wall_seconds"}
MANIFEST_VOLATILE_FIELDS = {"setup_wall_seconds", "shot_wall_seconds"}
MANIFEST_PATH_FIELDS = {"session", "companion"}
DECLARED_MANIFEST_ADDITIONS = {
    "declared_additions",
    "nova_equilibrium_tree",
    "labeller_batch_tree",
}
DECLARED_SLICE_ADDITIONS = {"requested_class"}
EXCLUDED_SLICE_FIELDS = {"newton_steps"}
FLOAT_RTOL = 1e-12
FLOAT_ATOL = 1e-14
EXACT_FLOAT_FIELDS = {"time"}


def _maximum_relative_difference(expected: Any, actual: Any) -> float:
    """Return the largest relative difference, treating equal zeros as zero."""
    left = np.asarray(expected, dtype=np.float64)
    right = np.asarray(actual, dtype=np.float64)
    valid = ~(np.isnan(left) & np.isnan(right))
    if not np.any(valid):
        return 0.0
    left = left[valid]
    right = right[valid]
    finite = np.isfinite(left) & np.isfinite(right)
    if np.any(~finite & (left != right)):
        return float("inf")
    difference = np.abs(left[finite] - right[finite])
    denominator = np.abs(left[finite])
    if not difference.size:
        return 0.0
    relative = np.divide(
        difference,
        denominator,
        out=np.full_like(difference, np.inf),
        where=denominator > 0.0,
    )
    relative[(denominator == 0.0) & (difference == 0.0)] = 0.0
    return float(np.max(relative, initial=0.0))


def _arrays_differ(
    expected: np.ndarray, actual: np.ndarray, *, float_field: bool
) -> bool:
    """Apply the authored exact or floating-array comparison contract."""
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return True
    if not float_field:
        return not bool(np.array_equal(expected, actual))
    try:
        np.testing.assert_allclose(
            expected,
            actual,
            rtol=FLOAT_RTOL,
            atol=FLOAT_ATOL,
            equal_nan=True,
        )
    except AssertionError:
        return True
    return False


def _stable_carrier_evidence(value: Any) -> tuple[Any, bool]:
    """Remove elapsed measurements while retaining carrier identity evidence."""
    stable = json.loads(json.dumps(value))
    try:
        warm_load = stable["carrier"].pop("warm_load_seconds")
        check_stdout = stable["named_cache_only_check"]["stdout"]
    except KeyError, TypeError:
        return stable, False
    timings_valid = (
        isinstance(warm_load, int | float)
        and isinstance(check_stdout, str)
        and re.search(r"\bwarm_seconds=[0-9.]+\b", check_stdout) is not None
    )
    if not timings_valid:
        return stable, False
    stable["named_cache_only_check"]["stdout"] = re.sub(
        r"\bwarm_seconds=[0-9.]+\b", "warm_seconds=<elapsed>", check_stdout
    )
    return stable, timings_valid


def _compare(
    reference: Path,
    candidate: Path,
    shots: Sequence[int],
    requested_classes: dict[int, dict[int, int]],
) -> dict[str, Any]:
    differences: list[str] = []
    manifest_differences: dict[str, int] = {}
    npz_differences: dict[str, int] = {}
    session_differences: dict[str, int] = {}
    addition_occurrences: dict[str, int] = {}
    excluded_occurrences: dict[str, int] = {}
    revision_transitions: list[dict[str, Any]] = []
    maximum_relative_differences: dict[str, float] = {}

    def observe(counts: dict[str, int], name: str, differs: bool) -> None:
        counts.setdefault(name, 0)
        if differs:
            counts[name] += 1

    def observe_float(name: str, expected: Any, actual: Any) -> bool:
        left = np.asarray(expected)
        right = np.asarray(actual)
        maximum = _maximum_relative_difference(left, right)
        maximum_relative_differences[name] = max(
            maximum_relative_differences.get(name, 0.0), maximum
        )
        return _arrays_differ(left, right, float_field=True)

    for shot in shots:
        reference_manifest = json.loads(
            (reference / f"{shot}.manifest.json").read_text(encoding="utf-8")
        )
        candidate_manifest = json.loads(
            (candidate / f"{shot}.manifest.json").read_text(encoding="utf-8")
        )
        manifest_keys = sorted(set(reference_manifest) | set(candidate_manifest))
        for name in manifest_keys:
            if name == "slices":
                continue
            if name in DECLARED_MANIFEST_ADDITIONS:
                addition_occurrences[name] = addition_occurrences.get(name, 0) + 1
                continue
            expected_value = reference_manifest.get(name)
            actual_value = candidate_manifest.get(name)
            if name in MANIFEST_VOLATILE_FIELDS:
                differs = not isinstance(expected_value, int | float) or not isinstance(
                    actual_value, int | float
                )
            elif name in MANIFEST_PATH_FIELDS:
                differs = Path(str(expected_value)).name != Path(str(actual_value)).name
            elif name == "carrier":
                expected_carrier, expected_timing_valid = _stable_carrier_evidence(
                    expected_value
                )
                actual_carrier, actual_timing_valid = _stable_carrier_evidence(
                    actual_value
                )
                differs = (
                    not expected_timing_valid
                    or not actual_timing_valid
                    or expected_carrier != actual_carrier
                )
            elif name == "nova_revision" and expected_value != actual_value:
                valid_revisions = all(
                    isinstance(value, str)
                    and re.fullmatch(r"[0-9a-f]{40}", value) is not None
                    for value in (expected_value, actual_value)
                )
                differs = not valid_revisions
                if valid_revisions:
                    revision_transitions.append(
                        {
                            "shot": shot,
                            "reference_process_revision": expected_value,
                            "scheduler_process_revision": actual_value,
                        }
                    )
            else:
                differs = expected_value != actual_value
            observe(manifest_differences, name, differs)
            if differs:
                differences.append(f"manifest:{shot}:{name}")

        expected_rows = reference_manifest["slices"]
        actual_rows = candidate_manifest["slices"]
        if len(expected_rows) != len(actual_rows):
            observe(manifest_differences, "slices.length", True)
            differences.append(f"manifest:{shot}:slice_count")
        for expected_row, actual_row in zip(expected_rows, actual_rows):
            keys = set(expected_row) | set(actual_row)
            for name in sorted(keys):
                counter = f"slices.{name}"
                if name in EXCLUDED_SLICE_FIELDS:
                    excluded_occurrences[counter] = (
                        excluded_occurrences.get(counter, 0) + 1
                    )
                    continue
                if name == "requested_class":
                    addition_occurrences[counter] = (
                        addition_occurrences.get(counter, 0) + 1
                    )
                    differs = actual_row.get(name) != requested_classes[shot].get(
                        int(expected_row["row"])
                    )
                elif name in DECLARED_SLICE_ADDITIONS:
                    addition_occurrences[counter] = (
                        addition_occurrences.get(counter, 0) + 1
                    )
                    continue
                elif name in MANIFEST_TIMING_FIELDS:
                    differs = not isinstance(
                        expected_row.get(name), int | float
                    ) or not isinstance(actual_row.get(name), int | float)
                elif (
                    name not in EXACT_FLOAT_FIELDS
                    and isinstance(expected_row.get(name), float | np.floating)
                    and isinstance(actual_row.get(name), float | np.floating)
                ):
                    differs = observe_float(
                        f"manifest:{counter}",
                        expected_row[name],
                        actual_row[name],
                    )
                else:
                    differs = expected_row.get(name) != actual_row.get(name)
                observe(manifest_differences, counter, differs)
                if differs:
                    differences.append(f"manifest:{shot}:{expected_row['row']}:{name}")

        with np.load(reference / f"{shot}.npz") as expected_npz:
            with np.load(candidate / f"{shot}.npz") as actual_npz:
                names = sorted(set(expected_npz.files) | set(actual_npz.files))
                for name in names:
                    if name not in expected_npz or name not in actual_npz:
                        observe(npz_differences, name, True)
                        differences.append(f"npz:{shot}:{name}")
                    else:
                        float_field = (
                            expected_npz[name].dtype.kind in "fc"
                            and name not in EXACT_FLOAT_FIELDS
                        )
                        if float_field:
                            maximum_relative_differences[f"npz:{name}"] = max(
                                maximum_relative_differences.get(f"npz:{name}", 0.0),
                                _maximum_relative_difference(
                                    expected_npz[name], actual_npz[name]
                                ),
                            )
                        differs = _arrays_differ(
                            expected_npz[name],
                            actual_npz[name],
                            float_field=float_field,
                        )
                        observe(npz_differences, name, differs)
                    if name in expected_npz and name in actual_npz and differs:
                        differences.append(f"npz:{shot}:{name}")

        with xr.open_dataset(reference / f"{shot}.nc", group=SESSION_GROUP) as expected:
            with xr.open_dataset(
                candidate / f"{shot}.nc", group=SESSION_GROUP
            ) as actual:
                names = sorted(set(expected.variables) | set(actual.variables))
                for name in names:
                    if name not in expected.variables or name not in actual.variables:
                        observe(session_differences, name, True)
                        differences.append(f"session:{shot}:{name}")
                        continue
                    left, right = expected[name], actual[name]
                    if name in SESSION_TIMING_VARIABLES:
                        differs = (
                            left.dims != right.dims
                            or left.shape != right.shape
                            or left.dtype != right.dtype
                        )
                    else:
                        float_field = left.dtype.kind in "fc" and name not in (
                            EXACT_FLOAT_FIELDS
                        )
                        if float_field:
                            maximum_relative_differences[f"session:{name}"] = max(
                                maximum_relative_differences.get(
                                    f"session:{name}", 0.0
                                ),
                                _maximum_relative_difference(left.values, right.values),
                            )
                        differs = left.dims != right.dims or _arrays_differ(
                            left.values,
                            right.values,
                            float_field=float_field,
                        )
                    observe(session_differences, name, differs)
                    if differs:
                        differences.append(f"session:{shot}:{name}")
    return {
        "differing_field_count": len(differences),
        "differing_fields": differences,
        "differing_fields_by_surface": {
            "manifest": dict(sorted(manifest_differences.items())),
            "npz": dict(sorted(npz_differences.items())),
            "session": dict(sorted(session_differences.items())),
        },
        "declared_additions": {
            "fields": sorted(DECLARED_MANIFEST_ADDITIONS)
            + [f"slices.{name}" for name in sorted(DECLARED_SLICE_ADDITIONS)],
            "occurrences": dict(sorted(addition_occurrences.items())),
        },
        "accepted_process_revision_transitions": revision_transitions,
        "known_exclusions": {
            "fields": sorted(EXCLUDED_SLICE_FIELDS),
            "occurrences": dict(sorted(excluded_occurrences.items())),
            "reason": (
                "compiled trip counter defect at "
                "nova/equilibrium/reduced_newton.py:1345-1352"
            ),
        },
        "maximum_relative_difference_by_float_field": dict(
            sorted(maximum_relative_differences.items())
        ),
        "comparison_policy": {
            "discrete_values": "exact equality",
            "named_discrete_fields": [
                "converged",
                "qualified",
                "conditioned",
                "free_branch_guard_ok",
                "conditioned_branch_guard_ok",
                "termination",
                "trips",
                "excluded",
                "exclusion",
                "time",
                "row",
                "requested_class",
                "topology labels",
                "domain labels",
            ],
            "float_values": {
                "method": "numpy.testing.assert_allclose",
                "rtol": FLOAT_RTOL,
                "atol": FLOAT_ATOL,
                "equal_nan": True,
            },
            "manifest_wall_clock_fields": sorted(
                MANIFEST_TIMING_FIELDS | MANIFEST_VOLATILE_FIELDS
            ),
            "session_wall_clock_fields": sorted(SESSION_TIMING_VARIABLES),
            "wall_clock_comparison": "presence, numeric type, dimensions and dtype",
            "carrier_timing_comparison": (
                "warm-load fields have numeric/string shape; remaining evidence exact"
            ),
            "artifact_path_comparison": "matching basename",
        },
    }


def _replace_directory(path: Path, *, replace_existing: bool) -> None:
    if path.exists():
        if not replace_existing:
            raise FileExistsError(f"output exists; pass --replace: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _run_reference(
    output: Path,
    shots: Sequence[int],
    selected_rows: dict[int, tuple[int, ...]],
    *,
    condition_on_guard_failure: bool,
    log_path: Path,
) -> None:
    """Run the shard's label and writer route over selected physical rows."""
    prepared = prepare_labeller()
    programs = LabellerPrograms()
    setup_unassigned = prepared.setup_wall_seconds
    missing = object()
    inherited_range = getattr(sequential_shard, "range", missing)
    with log_path.open("w", encoding="utf-8") as log:
        with redirect_stdout(log):
            try:
                for shot in shots:
                    rows = selected_rows[shot]
                    group = sequential_shard.zarr.open_group(
                        str(sequential_shard.SHOT_STORE / f"{shot}.zarr"), mode="r"
                    )["efm"]
                    physical_row_count = int(group["time"].shape[0])

                    def selected_range(*arguments, rows=rows):
                        if len(arguments) == 1 and arguments[0] == physical_row_count:
                            return rows
                        return builtins.range(*arguments)

                    sequential_shard.range = selected_range
                    print(
                        json.dumps(
                            {"reference_shot": shot, "selected_rows": rows},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    programs, record = label_shot(
                        prepared,
                        shot,
                        output,
                        programs=programs,
                        include_raster=False,
                        condition_on_guard_failure=condition_on_guard_failure,
                        setup_wall_seconds=setup_unassigned,
                        max_slices=None,
                    )
                    if record["status"] != "skipped":
                        setup_unassigned = 0.0
            finally:
                if inherited_range is missing:
                    del sequential_shard.range
                else:
                    sequential_shard.range = inherited_range


def run_smoke(
    output: Path,
    *,
    devices: int,
    batch_per_device: int,
    host_workers: int,
    max_slices: int,
    run_reference: bool,
    replace_existing: bool,
    condition_on_guard_failure: bool,
    source_identity: SourceIdentity,
) -> dict[str, Any]:
    """Run one real arm and compare all persisted fields with the shard."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "h200-one-device").mkdir(parents=True, exist_ok=True)
    corpus = decoder_corpus(DEFAULT_MANIFEST, DEFAULT_COHORT_REPORT)
    shots = [item.shot for item in corpus[:SHOT_COUNT]]
    selected_rows = {
        shot: admitted_quartile_rows(shot, count=max_slices) for shot in shots
    }
    shot_list = output / "shot-list.txt"
    shot_list.write_text("".join(f"{shot}\n" for shot in shots), encoding="utf-8")
    reference = output / "reference"
    if run_reference:
        _replace_directory(reference, replace_existing=replace_existing)
        _run_reference(
            reference,
            shots,
            selected_rows,
            condition_on_guard_failure=condition_on_guard_failure,
            log_path=output / "h200-one-device" / "reference.log",
        )
    elif not reference.is_dir():
        raise FileNotFoundError(f"sequential reference is absent: {reference}")

    arm_name = "one-device" if devices == 1 else f"device-arm-{devices}"
    arm_root = output / arm_name
    _replace_directory(arm_root, replace_existing=replace_existing)
    prepared = prepare_labeller()
    ranked = [
        load_shot(shot, max_slices=None, selected_rows=selected_rows[shot])
        for shot in shots
    ]
    requested_classes = {
        work.shot: {item.row: item.requested_class for item in work.slices}
        for work in ranked
    }
    engine = SequentialCompiledEngine(
        prepared,
        device_count=devices,
        condition_on_guard_failure=condition_on_guard_failure,
    )
    scheduler = CorpusScheduler(
        engine=engine,
        device_count=devices,
        batch_per_device=batch_per_device,
        host_workers=host_workers,
        condition_on_guard_failure=condition_on_guard_failure,
    )
    performance = scheduler.run(
        ranked,
        arm_root,
        prepared=prepared,
        source_identity=source_identity,
    )
    identity = _compare(reference, arm_root, shots, requested_classes)
    result = {
        "schema": "nova-forward-labeller-parallel-smoke",
        "corpus_source": str(DEFAULT_MANIFEST),
        "cohort_source": str(DEFAULT_COHORT_REPORT),
        "shot_count": SHOT_COUNT,
        "max_admitted_slices_per_shot": max_slices,
        "shot_ids": shots,
        "engine": "sequential-compiled",
        "reference_driver": "scripts/labeller_batch/shard.py:label_shot",
        "sampling": {
            "method": "admitted-row quartiles",
            "selected_rows_by_shot": {
                str(shot): list(rows) for shot, rows in selected_rows.items()
            },
        },
        "source_identity": {
            "nova_revision": source_identity.nova_revision,
            "nova_equilibrium_tree": source_identity.nova_equilibrium_tree,
            "labeller_batch_tree": source_identity.labeller_batch_tree,
        },
        "declared_writer_additions": [
            "manifest.nova_equilibrium_tree",
            "manifest.labeller_batch_tree",
            "manifest.declared_additions",
            "manifest.slices.requested_class",
        ],
        "performance": performance,
        "identity": identity,
        "passed": identity["differing_field_count"] == 0,
    }
    receipt = output / "h200-one-device" / "identity-receipt.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--batch-per-device", type=int, default=1)
    parser.add_argument("--host-workers", type=int, default=4)
    parser.add_argument("--max-slices", type=int, default=1)
    parser.add_argument("--run-reference", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--condition-on-guard-failure", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    source_identity = SourceIdentity.capture()
    if (
        min(
            arguments.devices,
            arguments.batch_per_device,
            arguments.host_workers,
            arguments.max_slices,
        )
        < 1
    ):
        raise ValueError("device, batch, worker and slice counts must be positive")
    if arguments.prepare_only:
        corpus = decoder_corpus(DEFAULT_MANIFEST, DEFAULT_COHORT_REPORT)
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "corpus_shots": len(corpus),
                    "smoke_shots": [item.shot for item in corpus[:SHOT_COUNT]],
                    "source_identity": source_identity.__dict__,
                },
                sort_keys=True,
            )
        )
        return 0
    result = run_smoke(
        arguments.output.resolve(),
        devices=arguments.devices,
        batch_per_device=arguments.batch_per_device,
        host_workers=arguments.host_workers,
        max_slices=arguments.max_slices,
        run_reference=arguments.run_reference,
        replace_existing=arguments.replace,
        condition_on_guard_failure=arguments.condition_on_guard_failure,
        source_identity=source_identity,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
