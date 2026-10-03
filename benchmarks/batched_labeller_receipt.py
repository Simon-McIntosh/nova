"""Measure the sharded forward labeller on a fixed MAST corpus sample.

The allocation-level command launches a fresh child process for every device
and local-batch arm. Each child sees only its requested cards, constructs the
shared MAST operator, loads one admitted-row quartile from each sampled shot,
cycling evenly over the four quartile positions, and measures the same compiled
program with those slices repeated evenly.
Compilation and warmed execution are reported separately. A lightweight
``nvidia-smi`` sampler records accelerator utilisation throughout every arm.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/playable-forward-solve/batched-labeller/"
    "h200-throughput-receipt.json"
)
DEFAULT_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/playable/"
    "batched-labeller-receipt.md"
)
DEFAULT_ACCEPTANCE_OUTPUT = (
    ROOT / "docs/figures/playable-forward-solve/batched-labeller/"
    "h200-acceptance-receipt.json"
)
CORPUS_MANIFEST_ROOT = Path(
    "/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29"
)
DEVICE_COUNTS = (1, 2, 3)
BATCH_PER_DEVICE = (16, 64, 256)
CORPUS_SHOT_COUNT = 48
QUARTILE_COUNT = 4
MINIMUM_CONVERGED_SLICES = 20
TIMING_REPEATS = 3
UTILISATION_INTERVAL_SECONDS = 1.0
CONDITIONED_CENTROID_ERROR_LIMIT_M = 5.0e-3
MATERIAL_CENTROID_IMPROVEMENT_M = 1.0e-6
ACCEPTANCE_BATCH_PER_DEVICE = 16


def _strict(value: Any) -> Any:
    """Convert arrays and non-finite values to strict JSON data."""
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one strict, human-readable receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_revision() -> str:
    """Return the exact checkout revision measured by the allocation."""
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _visible_device_tokens(required: int = 1) -> list[str]:
    """Return every accelerator granted to this allocation."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not value:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is absent")
    tokens = [item.strip() for item in value.split(",") if item.strip()]
    if len(tokens) < required:
        raise RuntimeError(
            f"{required} granted accelerators are required, found {len(tokens)}"
        )
    return tokens


class _UtilisationSampler:
    """Sample GPU utilisation without retaining an unbounded raw stream."""

    def __init__(self, device_tokens: Sequence[str]):
        self.device_tokens = tuple(device_tokens)
        self.phase = "setup"
        self.samples: list[dict[str, Any]] = []
        self.failures: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, _type, _value, _traceback):
        self._stop.set()
        self._thread.join(timeout=5.0)

    def set_phase(self, phase: str) -> None:
        """Tag subsequent samples with their measurement phase."""
        self.phase = phase

    def _sample(self) -> None:
        while not self._stop.is_set():
            command = [
                "nvidia-smi",
                "-i",
                ",".join(self.device_tokens),
                "--query-gpu=index,uuid,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            stamp = time.time()
            if completed.returncode:
                self.failures.append(completed.stderr.strip()[-500:])
            else:
                for line in completed.stdout.splitlines():
                    fields = [item.strip() for item in line.split(",")]
                    if len(fields) != 4:
                        continue
                    self.samples.append(
                        {
                            "phase": self.phase,
                            "timestamp": stamp,
                            "index": fields[0],
                            "uuid": fields[1],
                            "utilisation_percent": float(fields[2]),
                            "memory_used_mib": float(fields[3]),
                        }
                    )
            self._stop.wait(UTILISATION_INTERVAL_SECONDS)

    def summary(self) -> dict[str, Any]:
        """Return phase and card aggregates for the receipt."""
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for sample in self.samples:
            grouped.setdefault((sample["phase"], sample["uuid"]), []).append(sample)
        rows = []
        for (phase, uuid), values in sorted(grouped.items()):
            utilisation = [item["utilisation_percent"] for item in values]
            memory = [item["memory_used_mib"] for item in values]
            rows.append(
                {
                    "phase": phase,
                    "uuid": uuid,
                    "sample_count": len(values),
                    "mean_utilisation_percent": float(np.mean(utilisation)),
                    "maximum_utilisation_percent": float(np.max(utilisation)),
                    "maximum_memory_used_mib": float(np.max(memory)),
                }
            )
        return {
            "interval_seconds": UTILISATION_INTERVAL_SECONDS,
            "rows": rows,
            "sample_failures": self.failures,
        }


def _admitted_quartile_rows(group, *, count: int = QUARTILE_COUNT) -> tuple[int, ...]:
    """Select the same admitted-row quartiles as the scheduler smoke."""
    from scripts.labeller_batch import shard

    admitted = [
        row
        for row in range(int(group["time"].shape[0]))
        if shard._slice_inputs(group, row) is not None  # noqa: SLF001
    ]
    if len(admitted) < count:
        raise RuntimeError(
            f"shot has {len(admitted)} admitted rows, fewer than {count}"
        )
    positions = [
        round((len(admitted) - 1) * slot / count) for slot in range(1, count + 1)
    ]
    selected = tuple(admitted[position] for position in positions)
    if len(set(selected)) != count:
        raise RuntimeError(f"quartile selection is not unique: {selected}")
    return selected


def _select_inputs(
    *, shot_limit: int = CORPUS_SHOT_COUNT, slice_limit: int | None = None
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Select manifest-qualified shots and their admitted-row quartiles."""
    import zarr

    from scripts.labeller_batch import shard

    selected: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    selected_shots = 0
    corpus = shard.decoder_corpus(shard.DEFAULT_MANIFEST, shard.DEFAULT_COHORT_REPORT)
    for work in corpus:
        if 22_475 <= int(work.shot) <= 22_626:
            continue
        manifest_path = CORPUS_MANIFEST_ROOT / f"{work.shot}.manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        converged_slices = int(manifest.get("converged_slice_count", 0))
        if converged_slices < MINIMUM_CONVERGED_SLICES:
            continue
        path = shard.SHOT_STORE / f"{work.shot}.zarr"
        if not path.exists():
            continue
        group = zarr.open_group(str(path), mode="r")["efm"]
        full_r = np.asarray(group["gridr"], dtype=np.float64)
        full_z = np.asarray(group["gridz"], dtype=np.float64)
        quartile = selected_shots % QUARTILE_COUNT + 1
        row = _admitted_quartile_rows(group)[quartile - 1]
        inputs = shard._slice_inputs(group, row)  # noqa: SLF001
        if inputs is None:
            raise RuntimeError(f"selected row {work.shot}/{row} is not admitted")
        seed = shard._slices_seed(group, row, full_r, full_z)  # noqa: SLF001
        centroid_r = float(group["current_centrd_r"][row])
        requested = shard._requested_class(group, row)  # noqa: SLF001
        if not np.all(np.isfinite(seed)) or not np.isfinite(centroid_r):
            raise RuntimeError(f"selected row {work.shot}/{row} is not finite")
        selected.append(
            {
                "initial": np.asarray(seed, dtype=np.float64),
                "prescribed_current": np.asarray(inputs["current"], dtype=np.float64),
                "target_current": abs(inputs["reference_plasma_current"]),
                "requested_class": int(requested),
                "reference_centroid": np.asarray(
                    [centroid_r, inputs["target_centroid_z"]], dtype=np.float64
                ),
                "centroid_target": np.asarray(
                    [inputs["target_centroid_z"]], dtype=np.float64
                ),
            }
        )
        evidence.append(
            {
                "shot": int(work.shot),
                "row": int(row),
                "quartile": quartile,
                "admitted_row_count": int(manifest["admitted_slice_count"]),
                "source_converged_slice_count": converged_slices,
                "source_manifest": str(manifest_path),
            }
        )
        selected_shots += 1
        if slice_limit is not None and len(selected) == slice_limit:
            break
        if selected_shots == shot_limit:
            break
    expected_slices = slice_limit if slice_limit is not None else shot_limit
    if selected_shots != shot_limit or len(selected) != expected_slices:
        raise RuntimeError(
            f"selected {selected_shots} shots and {len(selected)} slices; expected "
            f"{shot_limit} shots and {expected_slices} slices"
        )
    keys = tuple(selected[0])
    arrays = {key: np.stack([item[key] for item in selected]) for key in keys}
    return arrays, evidence


def _prepare_inputs(
    *, shot_limit: int = CORPUS_SHOT_COUNT, slice_limit: int | None = None
) -> tuple[Any, dict[str, np.ndarray], list[dict[str, Any]]]:
    """Construct the shared operator and one quartile from 48 corpus shots."""
    from nova.equilibrium.solve_request import default_forward_compilation_cache_root
    from nova.jax.config import configure_persistent_compilation_cache
    from scripts.labeller_batch import shard

    arrays, evidence = _select_inputs(
        shot_limit=shot_limit,
        slice_limit=slice_limit,
    )
    prepared = shard.prepare_labeller()
    cache = configure_persistent_compilation_cache(
        default_forward_compilation_cache_root()
    )
    arrays["cache_directory"] = np.asarray(str(cache.directory))
    return prepared.profile, arrays, evidence


def _sequential_reference(profile, inputs: dict[str, np.ndarray]) -> dict[str, Any]:
    """Run each distinct corpus slice through the scalar compiled route."""
    import jax.numpy as jnp

    from nova.equilibrium import reduced_newton
    from nova.equilibrium.batched_labeller import _centroid_pair  # noqa: PLC2701
    from nova.equilibrium.observation import MomentIntegralSupport

    free_program = None
    conditioned_program = None
    rows = []
    first_exception_traceback = None
    for index in range(len(inputs["initial"])):
        initial = jnp.asarray(inputs["initial"][index])
        target_current = jnp.asarray(inputs["target_current"][index])
        requested = jnp.asarray(inputs["requested_class"][index])
        current = jnp.asarray(inputs["prescribed_current"][index])
        free = None
        conditioned_result = None
        free_exception = None
        conditioned_exception = None
        guard_exception = None
        guard = False
        free_centroid = None
        try:
            free = reduced_newton.solve_reduced_newton_compiled(
                profile.operator,
                initial,
                prescribed_current=current,
                target_current=target_current,
                requested_class=requested,
                program=free_program,
            )
            free_program = free.program
            try:
                centroid = profile.current_moment_observation(
                    free.state,
                    support=MomentIntegralSupport.ALL_DOMAIN,
                    target_current=target_current,
                    requested_class=requested,
                ).stack()[1:]
                free_centroid = np.asarray(centroid)
                guard = bool(
                    np.linalg.norm(
                        np.asarray(centroid) - inputs["reference_centroid"][index]
                    )
                    <= 5.0e-2
                )
            except Exception as error:
                guard_exception = {
                    "stage": "guard",
                    "exception_class": type(error).__name__,
                    "exception_message": str(error),
                }
                if first_exception_traceback is None:
                    first_exception_traceback = traceback.format_exc()
        except Exception as error:
            free_exception = {
                "stage": "free",
                "exception_class": type(error).__name__,
                "exception_message": str(error),
            }
            if first_exception_traceback is None:
                first_exception_traceback = traceback.format_exc()
        conditioned = free is None or not bool(free.converged) or not guard
        if conditioned:
            try:
                target_pair = _centroid_pair(
                    profile,
                    initial,
                    inputs["centroid_target"][index, 0],
                    requested_class=requested,
                    target_current=target_current,
                )
                conditioned_result = (
                    reduced_newton.solve_constrained_reduced_newton_compiled(
                        profile,
                        initial if free is None else free.state,
                        constraint_pairs=(target_pair,),
                        prescribed_current=current,
                        target_current=target_current,
                        requested_class=requested,
                        program=conditioned_program,
                    )
                )
                conditioned_program = conditioned_result.program
            except Exception as error:
                conditioned_exception = {
                    "stage": "conditioned",
                    "exception_class": type(error).__name__,
                    "exception_message": str(error),
                }
                if first_exception_traceback is None:
                    first_exception_traceback = traceback.format_exc()
        result = conditioned_result if conditioned else free
        exceptions = [
            item
            for item in (free_exception, guard_exception, conditioned_exception)
            if item is not None
        ]
        rows.append(
            {
                "index": index,
                "converged": bool(result.converged) if result is not None else False,
                "guard": guard,
                "conditioned": conditioned,
                "free_centroid": free_centroid,
                "exception_classes": [item["exception_class"] for item in exceptions],
                "exceptions": exceptions,
            }
        )
    return {
        "slice_count": len(rows),
        "converged_count": sum(item["converged"] for item in rows),
        "guard_count": sum(item["guard"] for item in rows),
        "conditioned_count": sum(item["conditioned"] for item in rows),
        "converged_fraction": float(np.mean([item["converged"] for item in rows])),
        "guard_fraction": float(np.mean([item["guard"] for item in rows])),
        "conditioned_fraction": float(np.mean([item["conditioned"] for item in rows])),
        "exception_slice_count": sum(bool(item["exception_classes"]) for item in rows),
        "first_exception_traceback": first_exception_traceback,
        "slice_results": rows,
    }


def _expanded_steps(
    inputs: dict[str, np.ndarray], total_batch: int
) -> dict[str, np.ndarray]:
    """Repeat all selected slices evenly into whole fixed-size device steps."""
    input_count = len(inputs["initial"])
    measured = math.lcm(input_count, total_batch)
    indices = np.arange(measured) % input_count
    steps = measured // total_batch
    return {
        key: values[indices].reshape((steps, total_batch) + values.shape[1:])
        for key, values in inputs.items()
        if key != "cache_directory"
    }


def _padded_steps(
    inputs: dict[str, np.ndarray], total_batch: int
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Pad a distinct-slice census without repeating active observations."""
    input_count = len(inputs["initial"])
    step_count = math.ceil(input_count / total_batch)
    padded_count = step_count * total_batch
    indices = np.minimum(np.arange(padded_count), input_count - 1)
    active = np.arange(padded_count) < input_count
    steps = {
        key: values[indices].reshape((step_count, total_batch) + values.shape[1:])
        for key, values in inputs.items()
        if key != "cache_directory"
    }
    return steps, active.reshape(step_count, total_batch)


def _measure_one(batch_per_device: int, device_count: int) -> dict[str, Any]:
    """Measure one isolated local-batch and card-count arm."""
    import jax

    from nova.equilibrium.batched_labeller import BatchedLabeller

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU backend required, got {jax.default_backend()!r}")
    if len(jax.devices()) != device_count:
        raise RuntimeError(
            f"requested {device_count} devices, JAX discovered {len(jax.devices())}"
        )
    profile, inputs, evidence = _prepare_inputs()
    total_batch = batch_per_device * device_count
    steps = _expanded_steps(inputs, total_batch)
    labeller = BatchedLabeller(profile)
    device_tokens = _visible_device_tokens()[:device_count]
    solve_exceptions = []
    first_solve_exception_traceback = None

    def solve_step(step: int, *, phase: str, repeat: int | None):
        nonlocal first_solve_exception_traceback
        try:
            return labeller.solve(
                steps["initial"][step],
                prescribed_current=steps["prescribed_current"][step],
                target_current=steps["target_current"][step],
                requested_class=steps["requested_class"][step],
                reference_centroid=steps["reference_centroid"][step],
                centroid_target=steps["centroid_target"][step],
            )
        except Exception as error:
            full_traceback = traceback.format_exc()
            if first_solve_exception_traceback is None:
                first_solve_exception_traceback = full_traceback
            solve_exceptions.append(
                {
                    "phase": phase,
                    "repeat": repeat,
                    "step": step,
                    "exception_class": type(error).__name__,
                    "exception_message": str(error),
                    "slice_count": total_batch,
                }
            )
            return {
                "converged": np.zeros(total_batch, dtype=bool),
                "guard": np.zeros(total_batch, dtype=bool),
                "conditioned": np.ones(total_batch, dtype=bool),
            }

    def result_flag(result, name: str) -> np.ndarray:
        if isinstance(result, dict):
            return np.asarray(result[name])
        return np.asarray(getattr(result, name))

    with _UtilisationSampler(device_tokens) as sampler:
        sampler.set_phase("compile_and_first_execution")
        first_started = time.perf_counter()
        first = solve_step(0, phase="compile_and_first_execution", repeat=None)
        first_wall = time.perf_counter() - first_started
        sampler.set_phase("warmed_execution")
        timings = []
        last_results = []
        for repeat in range(TIMING_REPEATS):
            started = time.perf_counter()
            current_results = []
            for step in range(steps["initial"].shape[0]):
                current_results.append(
                    solve_step(step, phase="warmed_execution", repeat=repeat)
                )
            timings.append(time.perf_counter() - started)
            last_results = current_results
    utilisation = sampler.summary()
    measured_slices = int(steps["initial"].shape[0] * total_batch)
    median_wall = float(np.median(timings))
    converged = np.concatenate(
        [result_flag(result, "converged") for result in last_results]
    )
    guard = np.concatenate([result_flag(result, "guard") for result in last_results])
    conditioned = np.concatenate(
        [result_flag(result, "conditioned") for result in last_results]
    )
    converged_count = int(np.count_nonzero(converged))
    first_execution_estimate = median_wall / steps["initial"].shape[0]
    attempted_slices_per_second = measured_slices / median_wall
    return {
        "device_count": device_count,
        "batch_per_device": batch_per_device,
        "total_batch_per_step": total_batch,
        "step_count_per_repeat": int(steps["initial"].shape[0]),
        "measured_slices_per_repeat": measured_slices,
        "corpus_shots": evidence,
        "first_call_wall_seconds": first_wall,
        "compile_wall_seconds_estimate": max(
            0.0, first_wall - first_execution_estimate
        ),
        "compile_wall_definition": (
            "first call including compilation minus median warmed wall per step"
        ),
        "warmed_wall_seconds": {
            "samples": timings,
            "median": median_wall,
            "minimum": float(np.min(timings)),
            "maximum": float(np.max(timings)),
        },
        "slices_per_second": (
            attempted_slices_per_second if converged_count > 0 else None
        ),
        "attempted_slices_per_second": attempted_slices_per_second,
        "throughput_valid": converged_count > 0,
        "converged_count": converged_count,
        "guard_count": int(np.count_nonzero(guard)),
        "conditioned_count": int(np.count_nonzero(conditioned)),
        "converged_fraction": float(np.mean(converged)),
        "guard_fraction": float(np.mean(guard)),
        "conditioned_fraction": float(np.mean(conditioned)),
        "first_step_converged_fraction": float(
            np.mean(result_flag(first, "converged"))
        ),
        "solve_exception_event_count": len(solve_exceptions),
        "solve_exception_classes": sorted(
            {item["exception_class"] for item in solve_exceptions}
        ),
        "first_solve_exception_traceback": first_solve_exception_traceback,
        "solve_exceptions": solve_exceptions,
        "nvidia_smi": utilisation,
        "cache_directory": str(inputs["cache_directory"]),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }


def _diagnose_two() -> dict[str, Any]:
    """Let the smallest conditioned batch fail with its complete traceback."""
    import jax

    from nova.equilibrium.batched_labeller import BatchedLabeller

    profile, inputs, evidence = _prepare_inputs(shot_limit=2, slice_limit=2)
    if len(jax.devices()) != 2:
        raise RuntimeError(
            f"two host devices are required, JAX discovered {len(jax.devices())}"
        )
    result = BatchedLabeller(profile).solve(
        inputs["initial"],
        prescribed_current=inputs["prescribed_current"],
        target_current=inputs["target_current"],
        requested_class=inputs["requested_class"],
        reference_centroid=inputs["reference_centroid"],
        centroid_target=inputs["centroid_target"],
    )
    return {
        "status": "unexpected-success",
        "selected_slices": evidence,
        "converged": np.asarray(result.converged),
        "guard": np.asarray(result.guard),
        "conditioned": np.asarray(result.conditioned),
    }


def _acceptance(slice_count: int, output: Path) -> dict[str, Any]:
    """Pair scalar and batched acceptance on one accelerator allocation."""
    import jax

    from nova.equilibrium.batched_labeller import BatchedLabeller

    if not 1 <= slice_count <= CORPUS_SHOT_COUNT:
        raise ValueError(
            f"slice_count must be between 1 and {CORPUS_SHOT_COUNT}, got {slice_count}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU backend required, got {jax.default_backend()!r}")
    profile, inputs, evidence = _prepare_inputs(
        shot_limit=slice_count, slice_limit=slice_count
    )
    device_count = len(jax.devices())
    total_batch = ACCEPTANCE_BATCH_PER_DEVICE * device_count
    steps, active = _padded_steps(inputs, total_batch)
    receipt: dict[str, Any] = {
        "status": "prepared",
        "source_revision": _source_revision(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slice_count": slice_count,
        "selection": evidence,
        "batch": {
            "device_count": device_count,
            "elements_per_device": ACCEPTANCE_BATCH_PER_DEVICE,
            "total_elements_per_step": total_batch,
            "step_count": int(active.shape[0]),
            "active_slice_count": int(np.count_nonzero(active)),
            "padded_slice_count": int(active.size - np.count_nonzero(active)),
        },
        "cache_directory": str(inputs["cache_directory"]),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
    }
    _write_json(output, receipt)

    print(f"REFERENCE starts slice_count={slice_count}", flush=True)
    reference_started = time.perf_counter()
    reference = _sequential_reference(profile, inputs)
    reference_wall = time.perf_counter() - reference_started
    print(f"REFERENCE finishes wall_seconds={reference_wall:.6f}", flush=True)
    receipt.update(
        {
            "status": "reference-complete",
            "reference_wall_seconds": reference_wall,
            "sequential": reference,
        }
    )
    _write_json(output, receipt)

    free_labeller = BatchedLabeller(profile)
    conditioned_labeller = BatchedLabeller(profile)
    free_centroids = []
    conditioned_centroids = []
    converged = []
    guards = []
    conditioned_flags = []
    print(
        f"BATCHED starts slice_count={slice_count} device_count={device_count} "
        f"elements_per_device={ACCEPTANCE_BATCH_PER_DEVICE}",
        flush=True,
    )
    batched_started = time.perf_counter()
    try:
        for step in range(steps["initial"].shape[0]):
            common = {
                "prescribed_current": steps["prescribed_current"][step],
                "target_current": steps["target_current"][step],
                "requested_class": steps["requested_class"][step],
                "active": active[step],
            }
            free = free_labeller.solve(steps["initial"][step], **common)
            result = conditioned_labeller.solve(
                steps["initial"][step],
                reference_centroid=steps["reference_centroid"][step],
                centroid_target=steps["centroid_target"][step],
                **common,
            )
            active_count = int(np.count_nonzero(active[step]))
            free_centroids.append(np.asarray(free.achieved_centroid)[:active_count])
            conditioned_centroids.append(
                np.asarray(result.achieved_centroid)[:active_count]
            )
            converged.append(np.asarray(result.converged)[:active_count])
            guards.append(np.asarray(result.guard)[:active_count])
            conditioned_flags.append(np.asarray(result.conditioned)[:active_count])
    except Exception as error:
        receipt.update(
            {
                "status": "failed-batched-exception",
                "batched_exception": {
                    "exception_class": type(error).__name__,
                    "exception_message": str(error),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        _write_json(output, receipt)
        raise
    batched_wall = time.perf_counter() - batched_started
    print(f"BATCHED finishes wall_seconds={batched_wall:.6f}", flush=True)

    free_centroid = np.concatenate(free_centroids)
    achieved_centroid = np.concatenate(conditioned_centroids)
    converged_values = np.concatenate(converged)
    guard_values = np.concatenate(guards)
    conditioned_values = np.concatenate(conditioned_flags)
    targets = np.asarray(inputs["centroid_target"])[:, 0]
    sequential_conditioned = np.asarray(
        [row["conditioned"] for row in reference["slice_results"]], dtype=bool
    )
    sequential_completed = np.asarray(
        [not bool(row["exception_classes"]) for row in reference["slice_results"]],
        dtype=bool,
    )
    conditioned_indices = np.flatnonzero(sequential_conditioned)
    qualified_conditioned_indices = np.flatnonzero(
        sequential_conditioned & sequential_completed
    )
    excluded_conditioned_indices = np.flatnonzero(
        sequential_conditioned & ~sequential_completed
    )
    free_error = np.abs(free_centroid[:, 1] - targets)
    achieved_error = np.abs(achieved_centroid[:, 1] - targets)
    selected_free_error = free_error[qualified_conditioned_indices]
    selected_achieved_error = achieved_error[qualified_conditioned_indices]

    converged_count = int(np.count_nonzero(converged_values))
    guard_count = int(np.count_nonzero(guard_values))
    conditioned_count = int(np.count_nonzero(conditioned_values))
    sequential_conditioned_executed = bool(
        np.all(conditioned_values[conditioned_indices])
    )
    has_conditioned_rows = bool(qualified_conditioned_indices.size)
    finite = bool(
        has_conditioned_rows
        and np.all(np.isfinite(selected_free_error))
        and np.all(np.isfinite(selected_achieved_error))
    )
    materially_improved = bool(
        finite
        and np.all(
            selected_achieved_error
            <= selected_free_error - MATERIAL_CENTROID_IMPROVEMENT_M
        )
    )
    under_limit = bool(
        finite and np.all(selected_achieved_error <= CONDITIONED_CENTROID_ERROR_LIMIT_M)
    )
    no_coincident_rows = bool(
        finite
        and not np.any(
            np.isclose(
                selected_achieved_error,
                selected_free_error,
                rtol=0.0,
                atol=MATERIAL_CENTROID_IMPROVEMENT_M,
            )
        )
    )
    no_worse_rows = bool(
        finite and np.all(selected_achieved_error <= selected_free_error)
    )
    fraction_parity = {
        "denominator": slice_count,
        "sequential": {
            "converged_count": reference["converged_count"],
            "converged_fraction": reference["converged_fraction"],
            "guard_count": reference["guard_count"],
            "guard_fraction": reference["guard_fraction"],
            "conditioned_count": reference["conditioned_count"],
            "conditioned_fraction": reference["conditioned_fraction"],
        },
        "batched": {
            "converged_count": converged_count,
            "converged_fraction": converged_count / slice_count,
            "guard_count": guard_count,
            "guard_fraction": guard_count / slice_count,
            "conditioned_count": conditioned_count,
            "conditioned_fraction": conditioned_count / slice_count,
        },
        "converged_fraction_matches": (converged_count == reference["converged_count"]),
        "guard_fraction_matches": guard_count == reference["guard_count"],
    }
    conditioning = {
        "sequential_conditioned_count": int(np.count_nonzero(sequential_conditioned)),
        "sequential_conditioned_indices": conditioned_indices,
        "batched_conditioned_count": conditioned_count,
        "sequential_conditioned_rows_executed": sequential_conditioned_executed,
        "free_centroid_error_m": selected_free_error,
        "conditioned_centroid_error_m": selected_achieved_error,
        "median_free_centroid_error_m": (
            float(np.median(selected_free_error)) if has_conditioned_rows else None
        ),
        "median_conditioned_centroid_error_m": (
            float(np.median(selected_achieved_error)) if has_conditioned_rows else None
        ),
        "maximum_conditioned_centroid_error_m": (
            float(np.max(selected_achieved_error)) if has_conditioned_rows else None
        ),
        "materially_improved": materially_improved,
        "under_half_centimetre": under_limit,
        "no_coincident_rows": no_coincident_rows,
        "no_worse_rows": no_worse_rows,
        "excluded_conditioned_rows": [
            {
                "index": int(excluded_index),
                "shot": int(evidence[excluded_index]["shot"]),
                "row": int(evidence[excluded_index]["row"]),
                "reference_exception_classes": list(
                    reference["slice_results"][excluded_index]["exception_classes"]
                ),
                "reference_exceptions": reference["slice_results"][excluded_index][
                    "exceptions"
                ],
            }
            for excluded_index in map(int, excluded_conditioned_indices)
        ],
    }
    receipt.update(
        {
            "status": "passed",
            "batched_wall_seconds": batched_wall,
            "fraction_parity": fraction_parity,
            "conditioning": conditioning,
        }
    )
    checks = {
        "slice denominator": len(converged_values) == slice_count,
        "converged fraction parity": fraction_parity["converged_fraction_matches"],
        "guard fraction parity": fraction_parity["guard_fraction_matches"],
        "sequential conditioned rows executed": sequential_conditioned_executed,
        "finite centroid errors": finite,
        "material conditioning improvement": materially_improved,
        "conditioned error limit": under_limit,
        "no coincident free and conditioned rows": no_coincident_rows,
        "no conditioned row worse than free": no_worse_rows,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        receipt["status"] = "failed"
        receipt["failed_checks"] = failed
        _write_json(output, receipt)
        print(json.dumps(_strict(receipt), sort_keys=True, allow_nan=False), flush=True)
        raise AssertionError("accelerator acceptance failed: " + ", ".join(failed))
    _write_json(output, receipt)
    return receipt


def _child(
    mode: str, batch_per_device: int | None, device_tokens: Sequence[str]
) -> dict[str, Any]:
    """Launch one isolated measurement and parse its terminal JSON line."""
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(device_tokens)
    environment["JAX_PLATFORMS"] = "cuda,cpu"
    environment["JAX_ENABLE_X64"] = "true"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    command = [sys.executable, str(Path(__file__).resolve()), mode]
    if batch_per_device is not None:
        command.extend(
            [
                "--batch-per-device",
                str(batch_per_device),
                "--device-count",
                str(len(device_tokens)),
            ]
        )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"child {mode} failed with {completed.returncode}: "
            f"{completed.stderr[-4000:]}\n{completed.stdout[-2000:]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try:
        return json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"child {mode} returned no JSON: {completed.stdout[-4000:]}"
        ) from error


def _write_report(path: Path, receipt: dict[str, Any]) -> None:
    """Write the decision-facing table and name the fastest configuration."""
    arms = receipt["arms"]
    successful = [
        item
        for item in arms
        if item.get("status") == "complete"
        and item.get("throughput_valid") is True
        and item.get("slices_per_second") is not None
    ]
    lines = [
        "# Batched labeller throughput",
        "",
        f"Revision: `{receipt['source_revision']}`. SLURM job: "
        f"`{receipt['slurm_job_id']}`. Corpus sample: "
        f"{receipt['corpus_shot_count']} shots and "
        f"{receipt['corpus_slice_count']} admitted-row quartile slices.",
        "",
        "| Devices | Batch/device | Slices/s | Compile estimate (s) | "
        "Converged | Guard | Mean GPU utilisation |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in successful:
        warmed = [
            row["mean_utilisation_percent"]
            for row in arm["nvidia_smi"]["rows"]
            if row["phase"] == "warmed_execution"
        ]
        mean_utilisation = float(np.mean(warmed)) if warmed else float("nan")
        lines.append(
            f"| {arm['device_count']} | {arm['batch_per_device']} | "
            f"{arm['slices_per_second']:.3f} | "
            f"{arm['compile_wall_seconds_estimate']:.3f} | "
            f"{arm['converged_fraction']:.6f} | "
            f"{arm['guard_fraction']:.6f} | {mean_utilisation:.1f}% |"
        )
    lines.extend(["", "## Best configuration", ""])
    if successful:
        best = max(successful, key=lambda item: item["slices_per_second"])
        lines.extend(
            [
                f"The fastest measured arm used **{best['device_count']} devices "
                f"with {best['batch_per_device']} elements per device**, reaching "
                f"**{best['slices_per_second']:.3f} slices/s**.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "No arm produced a nonzero converged count, so no "
                "slices-per-second result or fastest configuration is reported.",
                "",
            ]
        )
    lines.extend(
        [
            "The JSON receipt retains every timing sample, per-card utilisation "
            "aggregate, corpus shot and row, cache directory, and the scalar "
            "compiled-route convergence and guard fractions.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(output: Path, report: Path) -> dict[str, Any]:
    """Run the scalar reference and all nine accelerator arms."""
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("a SLURM allocation is required")
    tokens = _visible_device_tokens(max(DEVICE_COUNTS))
    reference_started = time.perf_counter()
    print("REFERENCE child starting scalar compiled route", flush=True)
    reference = _child("reference", None, tokens[:1])
    reference_wall = time.perf_counter() - reference_started
    print(f"REFERENCE child finished wall_seconds={reference_wall:.6f}", flush=True)
    receipt: dict[str, Any] = {
        "artifact": "batched forward labeller throughput",
        "status": "working",
        "source_revision": _source_revision(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "corpus_shot_count": CORPUS_SHOT_COUNT,
        "corpus_slice_count": reference["slice_count"],
        "reference_wall_seconds": reference_wall,
        "sampling": {
            "method": "admitted-row quartiles",
            "quartile_positions": QUARTILE_COUNT,
            "quartile_assignment": "round-robin across the sampled shot list",
            "minimum_source_converged_slices": MINIMUM_CONVERGED_SLICES,
            "source_manifest_root": str(CORPUS_MANIFEST_ROOT),
            "excluded_shot_range": [22_475, 22_626],
        },
        "batch_per_device_values": list(BATCH_PER_DEVICE),
        "device_count_values": list(DEVICE_COUNTS),
        "sequential_compiled_reference": reference,
        "arms": [],
    }
    _write_json(output, receipt)
    for device_count in DEVICE_COUNTS:
        for batch_per_device in BATCH_PER_DEVICE:
            print(
                f"ARM starting devices={device_count} "
                f"batch_per_device={batch_per_device}",
                flush=True,
            )
            try:
                arm = _child("measure-one", batch_per_device, tokens[:device_count])
            except Exception as error:
                receipt["arms"].append(
                    {
                        "device_count": device_count,
                        "batch_per_device": batch_per_device,
                        "status": "failed",
                        "failure": f"{type(error).__name__}: {error}",
                    }
                )
                receipt["status"] = "failed"
                _write_json(output, receipt)
                raise
            arm["status"] = (
                "complete" if arm["throughput_valid"] else "failed-zero-convergence"
            )
            arm["converged_fraction_matches_sequential"] = (
                arm["converged_count"] * reference["slice_count"]
                == reference["converged_count"] * arm["measured_slices_per_repeat"]
            )
            arm["guard_fraction_matches_sequential"] = (
                arm["guard_count"] * reference["slice_count"]
                == reference["guard_count"] * arm["measured_slices_per_repeat"]
            )
            receipt["arms"].append(arm)
            _write_json(output, receipt)
            if arm["throughput_valid"]:
                print(
                    f"ARM complete devices={device_count} "
                    f"batch_per_device={batch_per_device} "
                    f"slices_per_second={arm['slices_per_second']:.3f}",
                    flush=True,
                )
            else:
                print(
                    f"ARM invalid devices={device_count} "
                    f"batch_per_device={batch_per_device} "
                    "reason=zero-converged-slices",
                    flush=True,
                )
    receipt["status"] = "complete"
    if not all(
        arm["converged_fraction_matches_sequential"]
        and arm["guard_fraction_matches_sequential"]
        for arm in receipt["arms"]
    ):
        receipt["status"] = "failed-fraction-parity"
    _write_json(output, receipt)
    _write_report(report, receipt)
    return receipt


# ---------------------------------------------------------------------------
# Completion-receipt figures.
#
# The three figures drawn here re-derive every number from the committed
# completion receipt: they share one header stating the corpus, each count
# annotation names the receipt path it reads, every fraction drawn lies in
# [0, 1], and any fraction over a subset of the corpus carries its numerator
# and denominator paths so the population it is over is explicit.
# ---------------------------------------------------------------------------

DEFAULT_FIGURE_DIR = ROOT / "docs/figures/playable-forward-solve/labeller-receipt"
DEFAULT_COMPLETION_RECEIPT = DEFAULT_FIGURE_DIR / "receipt.json"

#: Displacement classes ordered from the far-negative open end to the far
#: positive one.
_PIN_CLASSES = (
    ("below_-50_mm", "< -50 mm"),
    ("-50_to_0_mm", "-50 to 0 mm"),
    ("0_to_+50_mm", "0 to +50 mm"),
    ("above_+50_mm", "> +50 mm"),
)
_PIN_COLORS = ("#4c72b0", "#55a868", "#c44e52", "#8172b2")
_ACCENT = "#4c72b0"
_NEUTRAL = "#6b6b6b"
_MUTED = "#9a9a9a"


def _at(receipt: dict[str, Any], path: str) -> Any:
    """Return the value at a dotted receipt path, indexing lists by integer."""
    value: Any = receipt
    for key in path.split("."):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def _decl(
    label: str,
    kind: str,
    value: float,
    *,
    json_path: str | None = None,
    numerator: str | None = None,
    denominator: str | None = None,
    denominator_paths: tuple[str, ...] | None = None,
    sum_paths: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Record one printed number together with where the receipt carries it.

    A count read directly carries ``json_path``; a count that is the sum of
    several paths carries ``sum_json_paths``; a fraction over a subset carries
    ``numerator_json_path`` and either one denominator path or the several
    count paths that sum to its denominator.
    """
    item: dict[str, Any] = {"label": label, "kind": kind, "value": float(value)}
    if json_path is not None:
        item["json_path"] = json_path
    if numerator is not None:
        item["numerator_json_path"] = numerator
    if denominator is not None:
        item["denominator_json_path"] = denominator
    if denominator_paths is not None:
        item["denominator_json_paths"] = list(denominator_paths)
    if sum_paths is not None:
        item["sum_json_paths"] = list(sum_paths)
    return item


def _decile_paths(decile: str) -> tuple[str, ...]:
    """Return the four class-slice paths whose sum is a decile's written slices."""
    return tuple(
        f"pin_displacement_crosstab.by_decile.{decile}.{key}.slices"
        for key, _ in _PIN_CLASSES
    )


def _decile_total(receipt: dict[str, Any], decile: str) -> int:
    """Sum a decile's class slices to its written-slices total."""
    return int(sum(_at(receipt, path) for path in _decile_paths(decile)))


def _count_decl(receipt: dict[str, Any], label: str, json_path: str) -> dict[str, Any]:
    """Record a printed count read directly from one receipt path."""
    return _decl(label, "count", _at(receipt, json_path), json_path=json_path)


def _corpus(receipt: dict[str, Any]) -> dict[str, Any]:
    """Return the one corpus every figure header states."""
    return {
        "completed_shots": int(_at(receipt, "completed_shots")),
        "slices": int(_at(receipt, "slices")),
        "shots_json_path": "completed_shots",
        "slices_json_path": "slices",
    }


def _corpus_title(corpus: dict[str, Any]) -> str:
    """Compose the shared one-corpus header."""
    return (
        f"Forward labeller receipt: {corpus['completed_shots']} completed shots, "
        f"{corpus['slices']} written slices"
    )


def _despine(axis) -> None:
    """Draw a panel with no gridlines and no top or right spine."""
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _annotate_bars(axis, bars, values, *, fmt: str = "{:d}", dy: float = 3.0) -> None:
    """Write each bar's value above the bar centre."""
    for bar, value in zip(bars, values):
        axis.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, bar.get_height() + dy),
            ha="center",
            va="bottom",
            fontsize=12,
        )


def _draw_adjudication(
    receipt: dict[str, Any], path: Path, title: str
) -> dict[str, Any]:
    """Slice census, guard agreement, displacement classes and decile trend."""
    import matplotlib.pyplot as plt

    declarations: list[dict[str, Any]] = []
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    census = (
        ("admitted", "admitted_slices"),
        ("converged", "converged_slices"),
        ("qualified", "qualified_slices"),
        ("unconverged", "unconverged_slices"),
    )
    labels = [item[0] for item in census]
    values = [_at(receipt, json_path) for _, json_path in census]
    for label, json_path in census:
        declarations.append(_count_decl(receipt, label, json_path))
    axis = axes[0][0]
    _annotate_bars(axis, axis.bar(labels, values, color=_ACCENT), values)
    axis.set_ylabel("slices")
    axis.set_title("Slice census", fontsize=13)
    _despine(axis)

    axis = axes[0][1]
    guard = (
        (
            "branch",
            "branch_guard_agreement_slices",
            "branch_guard_evaluated_slices",
            "branch_guard_agreement_fraction",
        ),
        (
            "conditioned",
            "conditioned_guard_agreement_slices",
            "conditioned_guard_evaluated_slices",
            "conditioned_guard_agreement_fraction",
        ),
    )
    positions = np.arange(len(guard))
    width = 0.36
    for position, (name, agree, evaluated, fraction) in enumerate(guard):
        evaluated_value = _at(receipt, evaluated)
        agreement_value = _at(receipt, agree)
        fraction_value = _at(receipt, fraction)
        declarations.append(_count_decl(receipt, f"{name} evaluated", evaluated))
        declarations.append(_count_decl(receipt, f"{name} agreement", agree))
        declarations.append(
            _decl(
                f"{name} agreement fraction",
                "fraction",
                fraction_value,
                json_path=fraction,
            )
        )
        axis.bar(position - width / 2, evaluated_value, width, color=_MUTED)
        axis.bar(position + width / 2, agreement_value, width, color=_ACCENT)
        axis.annotate(
            f"{fraction_value:.4f}",
            (position + width / 2, agreement_value),
            ha="center",
            va="bottom",
            fontsize=12,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels([item[0] for item in guard])
    axis.set_ylabel("slices")
    axis.set_title("Guard agreement (fraction over evaluated slices)", fontsize=13)
    _despine(axis)

    axis = axes[1][0]
    class_slices = []
    class_labels = []
    for key, label in _PIN_CLASSES:
        json_path = f"pin_displacement_crosstab.overall.{key}.slices"
        class_slices.append(_at(receipt, json_path))
        class_labels.append(label)
        declarations.append(_count_decl(receipt, f"displacement {label}", json_path))
    _annotate_bars(
        axis,
        axis.bar(class_labels, class_slices, color=_PIN_COLORS),
        class_slices,
    )
    axis.set_ylabel("slices")
    axis.set_xlabel("each open-ended end bin drawn as its own bar")
    axis.set_title("Centroid pin displacement by class", fontsize=13)
    _despine(axis)

    axis = axes[1][1]
    deciles = _at(receipt, "conditioned_by_time_in_shot.deciles.bins")
    xs = []
    fractions = []
    counts = []
    for index in range(len(deciles)):
        prefix = f"conditioned_by_time_in_shot.deciles.bins.{index}"
        xs.append(int(_at(receipt, f"{prefix}.decile")))
        fractions.append(_at(receipt, f"{prefix}.fraction"))
        counts.append(_at(receipt, f"{prefix}.slices"))
        declarations.append(
            _decl(
                f"conditioned fraction decile {index}",
                "fraction",
                _at(receipt, f"{prefix}.fraction"),
                json_path=f"{prefix}.fraction",
            )
        )
    axis.step(xs, fractions, where="mid", color=_ACCENT, linewidth=2.6)
    for x, fraction, count in zip(xs, fractions, counts):
        axis.annotate(
            f"n={count}",
            (x, fraction),
            ha="center",
            va="bottom",
            fontsize=9,
            color=_NEUTRAL,
        )
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("time-in-shot decile")
    axis.set_ylabel("conditioned fraction")
    axis.set_title("Conditioned fraction by time-in-shot decile", fontsize=13)
    _despine(axis)

    figure.suptitle(title, y=0.985, fontsize=15)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(path, dpi=100, metadata={"Title": title})
    plt.close(figure)
    return {"png": str(path), "title": title, "declarations": declarations}


def _draw_pin_tail(receipt: dict[str, Any], path: Path, title: str) -> dict[str, Any]:
    """Pin displacement cross-tabulation over time-in-shot deciles."""
    import matplotlib.pyplot as plt

    declarations: list[dict[str, Any]] = []
    figure, axes = plt.subplots(1, 3, figsize=(14, 5.4))

    axis = axes[0]
    totals = []
    labels = []
    for key, label in _PIN_CLASSES:
        json_path = f"pin_displacement_crosstab.overall.{key}.slices"
        totals.append(_at(receipt, json_path))
        labels.append(label)
        declarations.append(_count_decl(receipt, f"class total {label}", json_path))
    _annotate_bars(axis, axis.bar(labels, totals, color=_PIN_COLORS), totals)
    axis.set_ylabel("slices")
    axis.set_title("Pin displacement class totals", fontsize=13)
    _despine(axis)

    axis = axes[1]
    by_decile = _at(receipt, "pin_displacement_crosstab.by_decile")
    decile_keys = list(by_decile)
    xs = np.arange(len(decile_keys))
    bottoms = np.zeros(len(decile_keys))
    for key, color in zip([item[0] for item in _PIN_CLASSES], _PIN_COLORS):
        shares = []
        for decile in decile_keys:
            numerator = f"pin_displacement_crosstab.by_decile.{decile}.{key}.slices"
            share = _at(receipt, numerator) / _decile_total(receipt, decile)
            shares.append(share)
            declarations.append(
                _decl(
                    f"decile {decile} {key} share",
                    "fraction",
                    share,
                    numerator=numerator,
                    denominator_paths=_decile_paths(decile),
                )
            )
        shares_array = np.asarray(shares)
        axis.bar(xs, shares_array, bottom=bottoms, color=color, width=0.72)
        bottoms += shares_array
    for position, decile in enumerate(decile_keys):
        total = _decile_total(receipt, decile)
        declarations.append(
            _decl(
                f"decile {decile} written slices",
                "count",
                total,
                sum_paths=_decile_paths(decile),
            )
        )
        axis.annotate(
            f"n={total}",
            (xs[position], 1.0),
            ha="center",
            va="bottom",
            fontsize=9,
            color=_NEUTRAL,
        )
    axis.set_ylim(0.0, 1.12)
    axis.set_xticks(xs)
    axis.set_xticklabels(decile_keys)
    axis.set_xlabel("time-in-shot decile")
    axis.set_ylabel("class fraction within decile")
    axis.set_title("Class composition within each decile", fontsize=13)
    _despine(axis)

    axis = axes[2]
    series_key, series_label = _PIN_CLASSES[2]
    shares = []
    counts = []
    for decile in decile_keys:
        numerator = f"pin_displacement_crosstab.by_decile.{decile}.{series_key}.slices"
        share = _at(receipt, numerator) / _decile_total(receipt, decile)
        shares.append(share)
        counts.append(_at(receipt, numerator))
        declarations.append(
            _decl(
                f"{series_label} share decile {decile}",
                "fraction",
                share,
                numerator=numerator,
                denominator_paths=_decile_paths(decile),
            )
        )
    axis.plot(xs, shares, marker="o", color=_PIN_COLORS[2], linewidth=2.6)
    for index, (x, share, count) in enumerate(zip(xs, shares, counts)):
        axis.annotate(
            f"n={count}",
            (x, share),
            ha="center",
            va="bottom" if index % 2 else "top",
            fontsize=9,
            color=_NEUTRAL,
        )
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(xs)
    axis.set_title(f"{series_label} share by decile", fontsize=13)
    _despine(axis)

    figure.suptitle(title, y=0.99, fontsize=15)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(path, dpi=100, metadata={"Title": title})
    plt.close(figure)
    return {"png": str(path), "title": title, "declarations": declarations}


def _draw_signed_error(
    receipt: dict[str, Any], path: Path, title: str
) -> dict[str, Any]:
    """Signed error summaries with one wrapped title and no covering legend."""
    import matplotlib.pyplot as plt

    declarations: list[dict[str, Any]] = []
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    distributions = (
        ("free_error", "free_error_mm.distribution"),
        ("conditioned_error", "conditioned_error_mm.distribution"),
        ("absolute_reduction", "absolute_error_reduction_mm.distribution"),
        ("pin_correction", "pin_correction_mm.overall"),
    )
    axis = axes[0]
    rows = []
    for name, root in distributions:
        median = _at(receipt, f"{root}.median")
        p90 = _at(receipt, f"{root}.p90")
        maximum = _at(receipt, f"{root}.maximum")
        count = _at(receipt, f"{root}.count")
        for stat, value in (
            ("median", median),
            ("p90", p90),
            ("maximum", maximum),
        ):
            declarations.append(
                _decl(f"{name} {stat}", "value", value, json_path=f"{root}.{stat}")
            )
        declarations.append(_count_decl(receipt, f"{name} count", f"{root}.count"))
        rows.append((name, median, p90, maximum, count))
    for row_index, (name, median, p90, maximum, count) in enumerate(rows):
        axis.barh(
            row_index,
            max(p90, 0.0),
            left=min(0.0, median),
            color=_PIN_COLORS[row_index % len(_PIN_COLORS)],
        )
        axis.plot(
            [median, median],
            [row_index - 0.32, row_index + 0.32],
            color="black",
            linewidth=2.0,
        )
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels(
        [
            f"{name}\nmedian {median:.3g}, p90 {p90:.3g}, max {maximum:.3g}, n={count}"
            for name, median, p90, maximum, count in rows
        ],
        fontsize=11,
    )
    axis.set_xlabel("mm (bar spans 0 to p90, black tick at median)")
    axis.set_title("Signed error summaries", fontsize=13)
    _despine(axis)

    axis = axes[1]
    root = "absolute_error_reduction_mm.distribution"
    median = _at(receipt, f"{root}.median")
    p90 = _at(receipt, f"{root}.p90")
    maximum = _at(receipt, f"{root}.maximum")
    count = _at(receipt, f"{root}.count")
    bars = axis.bar(
        ("median", "p90", "maximum"),
        (median, p90, maximum),
        color=_ACCENT,
    )
    _annotate_bars(axis, bars, (median, p90, maximum), fmt="{:.3g}")
    axis.set_xlabel("mm")
    axis.set_title(
        "Absolute error reduction |free| - |conditioned|\n"
        "positive means conditioning\nreduced the centroid error",
        fontsize=13,
    )
    axis.annotate(
        f"n={count}",
        (0.5, 0.92),
        xycoords="axes fraction",
        ha="center",
        fontsize=11,
        color=_NEUTRAL,
    )
    _despine(axis)

    figure.suptitle(title, y=0.985, fontsize=15)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(path, dpi=100, metadata={"Title": title})
    plt.close(figure)
    return {"png": str(path), "title": title, "declarations": declarations}


def _figure_paths(figure_dir: Path) -> dict[str, Path]:
    """Return the three served figure paths beside the receipt."""
    return {
        "labeller-receipt-adjudication": figure_dir
        / "labeller-receipt-adjudication.png",
        "pin-tail-crosstab": figure_dir / "pin-tail-crosstab.png",
        "signed-error-summaries": figure_dir / "signed-error-summaries.png",
    }


def render_figures(figure_dir: Path) -> dict[str, Any]:
    """Regenerate the three completion-receipt figures from their receipt."""
    receipt = json.loads((figure_dir / "receipt.json").read_text(encoding="utf-8"))
    corpus = _corpus(receipt)
    title = _corpus_title(corpus)
    paths = _figure_paths(figure_dir)
    figures = [
        _draw_adjudication(receipt, paths["labeller-receipt-adjudication"], title),
        _draw_pin_tail(receipt, paths["pin-tail-crosstab"], title),
        _draw_signed_error(receipt, paths["signed-error-summaries"], title),
    ]
    payload = {
        "schema": "nova-labeller-receipt-figures",
        "source_receipt": "receipt.json",
        "corpus": corpus,
        "figures": [{"name": name, **figure} for name, figure in zip(paths, figures)],
    }
    (figure_dir / "render-receipt.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    measure = subparsers.add_parser("measure-one")
    measure.add_argument("--batch-per-device", type=int, required=True)
    measure.add_argument("--device-count", type=int, required=True)
    subparsers.add_parser("prepare-only")
    subparsers.add_parser("diagnose-two")
    acceptance = subparsers.add_parser("acceptance")
    acceptance.add_argument("--slice-count", type=int, default=CORPUS_SHOT_COUNT)
    acceptance.add_argument("--output", type=Path, default=DEFAULT_ACCEPTANCE_OUTPUT)
    subparsers.add_parser("reference")
    render = subparsers.add_parser("render")
    render.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the allocation parent or one isolated child."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "measure-one":
        print(
            json.dumps(
                _strict(
                    _measure_one(arguments.batch_per_device, arguments.device_count)
                ),
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    if arguments.command == "reference":
        profile, inputs, evidence = _prepare_inputs()
        result = _sequential_reference(profile, inputs)
        result["corpus_shots"] = evidence
        print(json.dumps(_strict(result), sort_keys=True, allow_nan=False))
        return 0
    if arguments.command == "prepare-only":
        inputs, evidence = _select_inputs(shot_limit=1, slice_limit=1)
        print(
            json.dumps(
                _strict(
                    {
                        "status": "prepared",
                        "selected_slice_count": len(inputs["initial"]),
                        "state_shape": inputs["initial"].shape,
                        "selection": evidence,
                    }
                ),
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    if arguments.command == "diagnose-two":
        print(json.dumps(_strict(_diagnose_two()), sort_keys=True, allow_nan=False))
        return 0
    if arguments.command == "acceptance":
        print(
            json.dumps(
                _strict(_acceptance(arguments.slice_count, arguments.output)),
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    if arguments.command == "render":
        payload = render_figures(arguments.figure_dir)
        print(json.dumps(_strict(payload), sort_keys=True, allow_nan=False))
        return 0
    receipt = _run(arguments.output, arguments.report)
    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
