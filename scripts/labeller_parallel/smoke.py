#!/usr/bin/env python3
"""Compare the real corpus scheduler with the sequential shard writer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nova.equilibrium.steering_frames import SESSION_GROUP  # noqa: E402
from scripts.labeller_batch.shard import (  # noqa: E402
    DEFAULT_COHORT_REPORT,
    DEFAULT_MANIFEST,
    decoder_corpus,
    prepare_labeller,
)
from scripts.labeller_parallel.scheduler import (  # noqa: E402
    CorpusScheduler,
    SequentialCompiledEngine,
    SourceIdentity,
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


def _equal(expected: np.ndarray, actual: np.ndarray) -> bool:
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    if expected.dtype.kind in "fc":
        return bool(np.array_equal(expected, actual, equal_nan=True))
    return bool(np.array_equal(expected, actual))


def _compare(reference: Path, candidate: Path, shots: Sequence[int]) -> dict[str, Any]:
    differences: list[str] = []
    manifest_differences: dict[str, int] = {}
    npz_differences: dict[str, int] = {}
    session_differences: dict[str, int] = {}
    addition_occurrences: dict[str, int] = {}

    def observe(counts: dict[str, int], name: str, differs: bool) -> None:
        counts.setdefault(name, 0)
        if differs:
            counts[name] += 1

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
                if name in DECLARED_SLICE_ADDITIONS:
                    addition_occurrences[counter] = (
                        addition_occurrences.get(counter, 0) + 1
                    )
                    continue
                if name in MANIFEST_TIMING_FIELDS:
                    differs = not isinstance(
                        expected_row.get(name), int | float
                    ) or not isinstance(actual_row.get(name), int | float)
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
                        differs = not _equal(expected_npz[name], actual_npz[name])
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
                        differs = left.dims != right.dims or not _equal(
                            left.values, right.values
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
        "comparison_policy": {
            "exact_values": "all fields except independent wall-clock values",
            "manifest_wall_clock_fields": sorted(
                MANIFEST_TIMING_FIELDS | MANIFEST_VOLATILE_FIELDS
            ),
            "session_wall_clock_fields": sorted(SESSION_TIMING_VARIABLES),
            "wall_clock_comparison": "presence, numeric type, dimensions and dtype",
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
    shot_list: Path,
    *,
    max_slices: int,
    condition_on_guard_failure: bool,
    log_path: Path,
) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/labeller_batch/shard.py"),
        str(output),
        "--shot-list",
        str(shot_list),
        "--max-slices",
        str(max_slices),
    ]
    if condition_on_guard_failure:
        command.append("--condition-on-guard-failure")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"sequential shard exited {completed.returncode}; see {log_path}"
        )


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
    shot_list = output / "shot-list.txt"
    shot_list.write_text("".join(f"{shot}\n" for shot in shots), encoding="utf-8")
    reference = output / "reference"
    if run_reference:
        _replace_directory(reference, replace_existing=replace_existing)
        _run_reference(
            reference,
            shot_list,
            max_slices=max_slices,
            condition_on_guard_failure=condition_on_guard_failure,
            log_path=output / "h200-one-device" / "reference.log",
        )
    elif not reference.is_dir():
        raise FileNotFoundError(f"sequential reference is absent: {reference}")

    arm_name = "one-device" if devices == 1 else f"device-arm-{devices}"
    arm_root = output / arm_name
    _replace_directory(arm_root, replace_existing=replace_existing)
    prepared = prepare_labeller()
    ranked = [load_shot(shot, max_slices=max_slices) for shot in shots]
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
    identity = _compare(reference, arm_root, shots)
    result = {
        "schema": "nova-forward-labeller-parallel-smoke",
        "corpus_source": str(DEFAULT_MANIFEST),
        "cohort_source": str(DEFAULT_COHORT_REPORT),
        "shot_count": SHOT_COUNT,
        "max_admitted_slices_per_shot": max_slices,
        "shot_ids": shots,
        "engine": "sequential-compiled",
        "reference_driver": "scripts/labeller_batch/shard.py",
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
