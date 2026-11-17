"""Read the pinned frame through the benchmark's profile and current owners."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import jax
import numpy as np

from nova.jax.config import configure_dtypes


def identity(value):
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def main():
    configure_dtypes()
    from benchmarks import diiid_current_pinned_forward as pinned

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=102)
    arguments = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"REVISION={revision} MODULE={pinned.__file__} CWD={Path.cwd()}", flush=True)
    print(f"DEVICES={jax.devices()} X64={jax.config.jax_enable_x64}", flush=True)
    declared = next(
        item
        for item in pinned.REPRESENTATIVE_COHORT
        if item["frame"] == arguments.frame
    )
    path = pinned.DEFAULT_DATA / declared["shot"]
    columns = tuple(
        dict.fromkeys(
            (
                *pinned._LABEL_COLUMNS,
                *pinned._CURRENT_COLUMNS,
                *pinned._GEOMETRY_COLUMNS,
                *pinned.PLASMA_CURRENT_COLUMNS,
            )
        )
    )
    row = pinned._read(path, columns)
    row["_source_path"] = str(path)
    started = time.monotonic()
    profile, seed, label, wall, reliable, statement = pinned.build_profile(
        row, arguments.frame, pinned.PSEUDO_WALL_EXPANSION
    )
    print(f"BUILT seconds={time.monotonic() - started:.3f}", flush=True)
    time_ms = float(row["efit_times"][arguments.frame])
    profile, shipped, current, authority = pinned.description_driven_currents(
        row, profile, time_ms
    )
    target = pinned._target_current(row, time_ms)
    moments = profile.operator.cell_current(seed, pinned.TopologyClass.DIVERTED)
    unscaled = float(np.sum(np.asarray(moments)))
    amplitude = float(
        profile.operator.current_normalisation_amplitude(target, unscaled)
    )
    result = {
        "revision": revision,
        "module": pinned.__file__,
        "cwd": str(Path.cwd()),
        "shot": declared["shot"],
        "frame": arguments.frame,
        "time_ms": time_ms,
        "unscaled_seed_plasma_current_a": unscaled,
        "registered_seed_current_a": declared["unscaled_source_ip_a"],
        "probe_seed_current_a": 1119994.165,
        "seed_amplitude": amplitude,
        "target_current_a": target,
        "current_authority": authority,
        "seed_identity": identity(seed),
        "label_identity": identity(label),
        "wall_identity": identity(wall),
        "cell_current_identity": identity(moments),
        "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "use_linear_moments": bool(profile.operator.use_linear_moments),
        "reliable_surfaces": reliable,
        "wall_statement": statement,
        "relative_registration_error": unscaled / declared["unscaled_source_ip_a"]
        - 1.0,
        "elapsed_seconds": time.monotonic() - started,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    # Reuse the constructed inputs so the benchmark's own qualification executes.
    pinned.build_profile = lambda *args, **kwargs: (
        profile,
        seed,
        label,
        wall,
        reliable,
        statement,
    )
    pinned.description_driven_currents = lambda *args: (
        profile,
        shipped,
        current,
        authority,
    )
    pinned.solve_frame(
        row, pinned.FrameInput(declared["shot"], arguments.frame), declared
    )


if __name__ == "__main__":
    main()
