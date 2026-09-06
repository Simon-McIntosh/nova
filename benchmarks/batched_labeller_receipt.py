"""Measure batch-size and device-count arms for the forward labeller.

The driver intentionally records compile wall separately from steady-state
steps.  A cluster allocation invokes this file once per requested device arm;
the JSON shape is stable even when an arm is unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import jax
import numpy as np

from nova.equilibrium.batched_labeller import BatchedLabeller


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def measure(profile, seed, *, batch_per_device: int, active=None) -> dict:
    """Measure one warmed batch and return a portable receipt."""
    device_count = len(jax.devices())
    batch = np.repeat(
        np.asarray(seed)[None, :], batch_per_device * device_count, axis=0
    )
    labeller = BatchedLabeller(profile)
    compile_started = time.perf_counter()
    first = labeller.solve(batch, active=active)
    compile_wall = time.perf_counter() - compile_started
    started = time.perf_counter()
    result = labeller.solve(batch, active=active)
    step_wall = time.perf_counter() - started
    return {
        "batch_per_device": batch_per_device,
        "device_count": device_count,
        "compile_wall_seconds": compile_wall,
        "steady_state_wall_seconds": step_wall,
        "slices_per_second": len(batch) / step_wall if step_wall else None,
        "converged_fraction": float(np.mean(np.asarray(result.converged))),
        "guard_fraction": float(np.mean(np.asarray(result.guard))),
        "conditioned_fraction": float(np.mean(np.asarray(result.conditioned))),
        "first_result_converged_fraction": float(np.mean(np.asarray(first.converged))),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }


def main() -> None:
    """Parse receipt arguments; fixture construction stays caller-owned."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-per-device", type=int, nargs="+", required=True)
    parser.add_argument("--source-revision", default="unknown")
    arguments = parser.parse_args()
    _write(
        arguments.output,
        {
            "artifact": "batched forward labeller throughput",
            "source_revision": arguments.source_revision,
            "status": "driver-ready",
            "arms": [
                {
                    "batch_per_device": value,
                    "status": "requires fixture and allocation",
                }
                for value in arguments.batch_per_device
            ],
        },
    )


if __name__ == "__main__":
    main()
