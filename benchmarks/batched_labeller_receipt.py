"""Measure the sharded forward labeller on a fixed MAST corpus sample.

The allocation-level command launches a fresh child process for every device
and local-batch arm. Each child sees only its requested cards, constructs the
shared MAST operator, loads one finite reconstruction slice from each sampled
shot, and measures the same compiled program with those slices repeated evenly.
Compilation and warmed execution are reported separately. A lightweight
``nvidia-smi`` sampler records accelerator utilisation throughout every arm.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/playable-forward-solve/batched-labeller/"
    "h200-throughput-receipt.json"
)
DEFAULT_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/playable/batched-labeller.md"
)
DEVICE_COUNTS = (1, 2, 3)
BATCH_PER_DEVICE = (16, 64, 256)
CORPUS_SHOT_COUNT = 48
TIMING_REPEATS = 3
UTILISATION_INTERVAL_SECONDS = 1.0


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


def _prepare_inputs() -> tuple[Any, dict[str, np.ndarray], list[dict[str, int]]]:
    """Construct the shared operator and one finite slice from 48 corpus shots."""
    import zarr

    from nova.equilibrium.solve_request import default_forward_compilation_cache_root
    from nova.jax.config import configure_persistent_compilation_cache
    from scripts.labeller_batch import shard

    prepared = shard.prepare_labeller()
    cache = configure_persistent_compilation_cache(
        default_forward_compilation_cache_root()
    )
    selected: list[dict[str, Any]] = []
    evidence: list[dict[str, int]] = []
    corpus = shard.decoder_corpus(shard.DEFAULT_MANIFEST, shard.DEFAULT_COHORT_REPORT)
    for work in corpus:
        path = shard.SHOT_STORE / f"{work.shot}.zarr"
        if not path.exists():
            continue
        group = zarr.open_group(str(path), mode="r")["efm"]
        full_r = np.asarray(group["gridr"], dtype=np.float64)
        full_z = np.asarray(group["gridz"], dtype=np.float64)
        for row in range(int(group["time"].shape[0])):
            inputs = shard._slice_inputs(group, row)  # noqa: SLF001
            if inputs is None:
                continue
            seed = shard._slices_seed(group, row, full_r, full_z)  # noqa: SLF001
            centroid_r = float(group["current_centrd_r"][row])
            requested = shard._requested_class(group, row)  # noqa: SLF001
            if not np.all(np.isfinite(seed)) or not np.isfinite(centroid_r):
                continue
            selected.append(
                {
                    "initial": np.asarray(seed, dtype=np.float64),
                    "prescribed_current": np.asarray(
                        inputs["current"], dtype=np.float64
                    ),
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
            evidence.append({"shot": int(work.shot), "row": row})
            break
        if len(selected) == CORPUS_SHOT_COUNT:
            break
    if len(selected) != CORPUS_SHOT_COUNT:
        raise RuntimeError(
            f"found {len(selected)} finite corpus shots, expected {CORPUS_SHOT_COUNT}"
        )
    keys = tuple(selected[0])
    arrays = {key: np.stack([item[key] for item in selected]) for key in keys}
    arrays["cache_directory"] = np.asarray(str(cache.directory))
    return prepared.profile, arrays, evidence


def _sequential_reference(profile, inputs: dict[str, np.ndarray]) -> dict[str, Any]:
    """Run each distinct corpus slice through the scalar compiled route."""
    import jax.numpy as jnp

    from nova.equilibrium import reduced_newton
    from nova.equilibrium.batched_labeller import _centroid_pair  # noqa: PLC2701
    from nova.equilibrium.observation import MomentIntegralSupport

    pair = _centroid_pair(
        profile, inputs["initial"][0], inputs["centroid_target"][0, 0]
    )
    free_program = None
    conditioned_program = None
    rows = []
    centroid_observation_failures = 0
    for index in range(CORPUS_SHOT_COUNT):
        initial = jnp.asarray(inputs["initial"][index])
        target_current = jnp.asarray(inputs["target_current"][index])
        requested = jnp.asarray(inputs["requested_class"][index])
        current = jnp.asarray(inputs["prescribed_current"][index])
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
            ).stack()[1:]
            guard = bool(
                np.linalg.norm(
                    np.asarray(centroid) - inputs["reference_centroid"][index]
                )
                <= 5.0e-2
            )
        except Exception:
            centroid_observation_failures += 1
            guard = False
        conditioned = not bool(free.converged) or not guard
        result = free
        if conditioned:
            target_pair = dataclasses.replace(
                pair,
                binding=dataclasses.replace(
                    pair.binding,
                    target=jnp.asarray(inputs["centroid_target"][index]),
                ),
            )
            result = reduced_newton.solve_constrained_reduced_newton_compiled(
                profile,
                free.state,
                constraint_pairs=(target_pair,),
                prescribed_current=current,
                target_current=target_current,
                requested_class=requested,
                program=conditioned_program,
            )
            conditioned_program = result.program
        rows.append(
            {
                "converged": bool(result.converged),
                "guard": guard,
                "conditioned": conditioned,
            }
        )
    return {
        "shot_count": len(rows),
        "converged_count": sum(item["converged"] for item in rows),
        "guard_count": sum(item["guard"] for item in rows),
        "conditioned_count": sum(item["conditioned"] for item in rows),
        "converged_fraction": float(np.mean([item["converged"] for item in rows])),
        "guard_fraction": float(np.mean([item["guard"] for item in rows])),
        "conditioned_fraction": float(np.mean([item["conditioned"] for item in rows])),
        "centroid_observation_failure_count": centroid_observation_failures,
    }


def _expanded_steps(
    inputs: dict[str, np.ndarray], total_batch: int
) -> dict[str, np.ndarray]:
    """Repeat all 48 shots evenly into whole fixed-size device steps."""
    measured = math.lcm(CORPUS_SHOT_COUNT, total_batch)
    indices = np.arange(measured) % CORPUS_SHOT_COUNT
    steps = measured // total_batch
    return {
        key: values[indices].reshape((steps, total_batch) + values.shape[1:])
        for key, values in inputs.items()
        if key != "cache_directory"
    }


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
    with _UtilisationSampler(device_tokens) as sampler:
        sampler.set_phase("compile_and_first_execution")
        first_started = time.perf_counter()
        first = labeller.solve(
            steps["initial"][0],
            prescribed_current=steps["prescribed_current"][0],
            target_current=steps["target_current"][0],
            requested_class=steps["requested_class"][0],
            reference_centroid=steps["reference_centroid"][0],
            centroid_target=steps["centroid_target"][0],
        )
        first_wall = time.perf_counter() - first_started
        sampler.set_phase("warmed_execution")
        timings = []
        last_results = []
        for _repeat in range(TIMING_REPEATS):
            started = time.perf_counter()
            current_results = []
            for step in range(steps["initial"].shape[0]):
                current_results.append(
                    labeller.solve(
                        steps["initial"][step],
                        prescribed_current=steps["prescribed_current"][step],
                        target_current=steps["target_current"][step],
                        requested_class=steps["requested_class"][step],
                        reference_centroid=steps["reference_centroid"][step],
                        centroid_target=steps["centroid_target"][step],
                    )
                )
            timings.append(time.perf_counter() - started)
            last_results = current_results
    utilisation = sampler.summary()
    measured_slices = int(steps["initial"].shape[0] * total_batch)
    median_wall = float(np.median(timings))
    converged = np.concatenate(
        [np.asarray(result.converged) for result in last_results]
    )
    guard = np.concatenate([np.asarray(result.guard) for result in last_results])
    conditioned = np.concatenate(
        [np.asarray(result.conditioned) for result in last_results]
    )
    first_execution_estimate = median_wall / steps["initial"].shape[0]
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
        "slices_per_second": measured_slices / median_wall,
        "converged_count": int(np.count_nonzero(converged)),
        "guard_count": int(np.count_nonzero(guard)),
        "conditioned_count": int(np.count_nonzero(conditioned)),
        "converged_fraction": float(np.mean(converged)),
        "guard_fraction": float(np.mean(guard)),
        "conditioned_fraction": float(np.mean(conditioned)),
        "first_step_converged_fraction": float(np.mean(np.asarray(first.converged))),
        "nvidia_smi": utilisation,
        "cache_directory": str(inputs["cache_directory"]),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }


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
        timeout=1_800,
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
    successful = [item for item in arms if item.get("status") == "complete"]
    best = max(successful, key=lambda item: item["slices_per_second"])
    lines = [
        "# Batched labeller throughput",
        "",
        f"Revision: `{receipt['source_revision']}`. SLURM job: "
        f"`{receipt['slurm_job_id']}`. Corpus sample: "
        f"{receipt['corpus_shot_count']} shots.",
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
    lines.extend(
        [
            "",
            "## Best configuration",
            "",
            f"The fastest measured arm used **{best['device_count']} devices with "
            f"{best['batch_per_device']} elements per device**, reaching "
            f"**{best['slices_per_second']:.3f} slices/s**.",
            "",
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
    print("REFERENCE starting scalar compiled route", flush=True)
    reference = _child("reference", None, tokens[:1])
    print("REFERENCE complete", flush=True)
    receipt: dict[str, Any] = {
        "artifact": "batched forward labeller throughput",
        "status": "working",
        "source_revision": _source_revision(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "corpus_shot_count": CORPUS_SHOT_COUNT,
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
            arm["status"] = "complete"
            arm["converged_fraction_matches_sequential"] = (
                arm["converged_count"] * reference["shot_count"]
                == reference["converged_count"] * arm["measured_slices_per_repeat"]
            )
            arm["guard_fraction_matches_sequential"] = (
                arm["guard_count"] * reference["shot_count"]
                == reference["guard_count"] * arm["measured_slices_per_repeat"]
            )
            receipt["arms"].append(arm)
            _write_json(output, receipt)
            print(
                f"ARM complete devices={device_count} "
                f"batch_per_device={batch_per_device} "
                f"slices_per_second={arm['slices_per_second']:.3f}",
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    measure = subparsers.add_parser("measure-one")
    measure.add_argument("--batch-per-device", type=int, required=True)
    measure.add_argument("--device-count", type=int, required=True)
    subparsers.add_parser("reference")
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
    receipt = _run(arguments.output, arguments.report)
    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
