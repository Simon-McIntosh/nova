"""Measure the playable app's keyframe cost through its own solver on the H200.

Drives :class:`~apps.playable.session.PlayableSession` with
:class:`apps.playable.production.ProductionSolver` on the MAST 22086/43
machine through a scripted chain of twenty key presses, in one foreground
sbatch on the reserved H200, recording per press the wall, trips, reuse flag,
the solve wall against the receipt-build wall, and for the first moved key the
split between trace/compile, persistent-cache load and dispatch (from
``jax_log_compiles`` events plus a count of persistent-cache artifacts loaded).

The human-response-time fences are the median warm keyframe near a hundred
milliseconds and the first moved key within one second; a seconds-scale first
moved key is attributed to trace, cache load or dispatch rather than bounded.

Every press is persisted to the receipt as it lands, so a job that runs out of
wall clock still delivers the presses it measured.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium.observation import MomentIntegralSupport
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)

from apps.playable.production import ForwardMachine, ProductionSolver
from apps.playable.session import PlayableSession
from apps.playable.shape import PlasmaShape

ROOT = Path(__file__).resolve().parents[1]
TARGET = (22086, 43)
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/playable-forward-solve/keyframes/h200-keyframes.json"
)
DEFAULT_FIGURE = (
    ROOT / "docs/figures/playable-forward-solve/keyframes/h200-keyframes.png"
)
#: One press against each sign of the vertical bulk control, ten round trips:
#: the chain stays near the free centroid while exercising both moved
#: targets.  The horizontal +R centroid move is deliberately excluded: on
#: 22086/43 it drives the solved state off the qualified magnetic axis
#: (measured NoQualifiedAxisError on the moved key), so the vertical direction
#: is the one the steering rows hold on this machine and the cost of a warm
#: moved keyframe is what the receipt reads.
KEY_CHAIN = ("bulk_z+", "bulk_z-") * 10
#: A compile event this fast was served by the persistent cache, not built.
CACHE_SERVED_COMPILE_SECONDS = 0.5


def _source_revision() -> str:
    """Return the revision this measurement runs from."""
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@contextlib.contextmanager
def _stderr_tee(path: Path):
    """Captured compile events go to ``path`` with the driver's own markers."""
    saved = os.dup(2)
    with open(path, "w") as target:
        os.dup2(target.fileno(), 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def _write(receipt: dict[str, Any], output: Path) -> None:
    """Persist the receipt so far, creating its directory once."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def _draw(receipt: dict[str, Any], figure: Path) -> None:
    """Draw the wall per press beside the fences."""
    presses = [press for press in receipt["presses"] if press["press"] is not None]
    if not presses:
        return
    index = [press["index"] for press in presses]
    wall = [press["wall"] for press in presses]
    prime = next(
        (press for press in receipt["presses"] if press["press"] is None), None
    )
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar(index, [1000.0 * second for second in wall], color="steelblue")
    plt.axhline(100.0, color="tab:green", linestyle="--", label="100 ms warm fence")
    plt.axhline(1000.0, color="tab:red", linestyle="--", label="1 s first-moved fence")
    if prime is not None:
        plt.axhline(
            1000.0 * prime["wall"], color="tab:orange", linestyle=":", label="prime"
        )
    plt.xlabel("key press")
    plt.ylabel("keyframe wall / ms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure, dpi=150)
    plt.close()


def _atime_entries(directory: Path) -> dict[str, float]:
    """Return every persistent-cache entry key with its atime-file mtime."""
    if not directory.is_dir():
        return {}
    entries: dict[str, float] = {}
    for artifact in directory.rglob("*-atime"):
        try:
            entries[str(artifact).removesuffix("-atime")] = artifact.stat().st_mtime
        except OSError:
            continue
    return entries


def _loaded_since(before: dict[str, float], after: dict[str, float]) -> int:
    """Count cache entries whose atime file advanced (loaded) in the window."""
    return sum(1 for key, mtime in after.items() if before.get(key, 0.0) < mtime)


def _compiles_in(events: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    """Return the compile events recorded between this phase's markers."""
    within = []
    active = False
    for event in events:
        if event["kind"] == "phase" and event["name"] == phase:
            active = True
            continue
        if event["kind"] == "phase":
            active = False
            continue
        if active:
            within.append(event)
    return within


def measure(
    *, output: Path, figure: Path, cache_root: Path | None = None
) -> dict[str, Any]:
    """Drive the playable session over the MAST 22086/43 machine on the H200."""
    import sys

    compile_log = output.parent / "h200-keyframes-compiles.log"

    def mark(phase: str) -> None:
        """Write a phase boundary into the same stream as the compile events."""
        sys.stderr.write(f"==PHASE {phase}==\n")
        sys.stderr.flush()

    configure_dtypes()
    jax.config.update("jax_log_compiles", True)
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
        if cache_root is None
        else cache_root
    )
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    }
    identity = f"{TARGET[0]}/{TARGET[1]}"

    receipt: dict[str, Any] = {
        "artifact": "playable app keyframes on the constrained reduced route, H200",
        "identity": identity,
        "source_commit": _source_revision(),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
            "scheduler": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "node": os.environ.get("SLURMD_NODENAME"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
                "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
            },
        },
        "evidence_inputs": {
            "response_carrier": carrier_evidence,
            "persistent_compilation_cache": cache.receipt(),
            "key_chain": list(KEY_CHAIN),
        },
        "presses": [],
        "first_moved": None,
    }
    _write(receipt, output)

    with _stderr_tee(compile_log):
        mark("build")
        selected_row, qualification = selected[TARGET]
        case, context = _mast_case_from_selection(
            SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, policy = _passive_inclusive_case(
            case, context, response_cache
        )
        seed = jnp.asarray(passive_case["state"])
        centre = np.asarray(
            profile.current_moment_observation(
                seed, support=MomentIntegralSupport.ALL_DOMAIN
            ).stack()
        )[1:3]
        machine = ForwardMachine(
            profile=profile,
            seed=seed,
            wall=np.asarray(profile.operator.wall.coordinate),
            identity="mast-22086/43",
        )
        solver = ProductionSolver(machine)

        # Per-press solve-vs-receipt split, measured from outside the session:
        # the instance methods the solver's call dispatches through are wrapped
        # with timers, leaving the solver itself untouched.
        solve_walls: list[float] = []
        receipt_walls: list[float] = []
        _reduced = solver._reduced
        _receipt = solver._reduced_receipt

        def timed_reduce(profile_, flux, commanded, program=None):
            started = time.perf_counter()
            result = _reduced(profile_, flux, commanded, program)
            solve_walls.append(time.perf_counter() - started)
            return result

        def timed_receipt(profile_, result):
            started = time.perf_counter()
            equilibrium = _receipt(profile_, result)
            receipt_walls.append(time.perf_counter() - started)
            return equilibrium

        solver._reduced = timed_reduce
        solver._reduced_receipt = timed_receipt

        command = PlasmaShape(
            axis_r=float(centre[0]),
            axis_z=float(centre[1]),
            minor_radius=0.4,
            elongation=1.6,
            triangularity_upper=0.05,
            triangularity_lower=0.1,
            x_point_r=float(centre[0]) - 0.1,
            x_point_z=float(centre[1]) - 0.6,
            inner_gap=0.02,
            outer_gap=0.02,
        )
        session = PlayableSession(solver=solver, shape=command, machine="mast-22086/43")

        mark("prime")
        prime = session.prime()
        receipt["prime"] = {
            "wall": prime.wall,
            "trips": prime.trips,
            "reused": prime.reused,
            "solve_wall": solve_walls[-1],
            "receipt_wall": receipt_walls[-1],
            "seed_centroid_m": centre.tolist(),
        }
        receipt["presses"].append(
            {
                "index": 0,
                "press": None,
                "key": "prime",
                "wall": prime.wall,
                "trips": prime.trips,
                "reused": prime.reused,
                "solve_wall": solve_walls[-1],
                "receipt_wall": receipt_walls[-1],
            }
        )
        _write(receipt, output)
        mark("prime-done")

        mark("first-moved")
        moved_cache_before = _atime_entries(cache.directory)
        for index, key in enumerate(KEY_CHAIN, start=1):
            press = session.step(key)
            entry = {
                "index": index,
                "press": key,
                "parameter": press.parameter,
                "delta": press.delta,
                "wall": press.wall,
                "trips": press.trips,
                "reused": press.reused,
                "solve_wall": solve_walls[-1],
                "receipt_wall": receipt_walls[-1],
            }
            receipt["presses"].append(entry)
            _write(receipt, output)
            print(f"PRESS-DONE {json.dumps(entry, sort_keys=True)}", flush=True)
            if index == 1:
                moved_cache_after = _atime_entries(cache.directory)
                mark("first-moved-done")

    # Attribute the first moved key's wall to trace/compile, cache load or
    # dispatch from the compile events between its phase markers and the cache
    # entries loaded in the same window.
    events = [
        _event_from_line(line)
        for line in compile_log.read_text(encoding="utf-8").splitlines()
    ]
    moved_events = _compiles_in(events, "first-moved")
    fresh = [
        event
        for event in moved_events
        if event["kind"] == "compile"
        and event.get("seconds", float("inf")) > CACHE_SERVED_COMPILE_SECONDS
    ]
    cached = [
        event
        for event in moved_events
        if event["kind"] == "compile"
        and event.get("seconds", float("inf")) <= CACHE_SERVED_COMPILE_SECONDS
    ]
    receipt["first_moved"] = {
        "press": receipt["presses"][1],
        "compile_events": len(moved_events),
        "fresh_compiles": len(fresh),
        "fresh_compile_seconds": sum(event["seconds"] for event in fresh),
        "cache_served_compiles": len(cached),
        "cache_served_compile_seconds": sum(event["seconds"] for event in cached),
        "cache_artifacts_loaded": _loaded_since(
            moved_cache_before, moved_cache_after
        ),
        "dispatch_and_host_seconds": receipt["presses"][1]["wall"]
        - sum(event["seconds"] for event in fresh)
        - sum(event["seconds"] for event in cached),
    }
    receipt["compile_events_total"] = sum(
        1 for event in events if event["kind"] == "compile"
    )

    warm = [press["wall"] for press in receipt["presses"][2:]]
    receipt["verdict"] = {
        "presses_measured": len(receipt["presses"]) - 1,
        "median_warm_keyframe_s": float(np.median(warm)),
        "mean_warm_keyframe_s": float(np.mean(warm)),
        "first_moved_keyframe_s": receipt["presses"][1]["wall"],
        "prime_wall_s": receipt["prime"]["wall"],
        "all_reused_after_prime": all(
            press["reused"] for press in receipt["presses"][1:]
        ),
        "median_warm_vs_100ms": float(np.median(warm) / 0.100),
        "first_moved_vs_1s": receipt["presses"][1]["wall"] / 1.0,
    }
    _write(receipt, output)
    _draw(receipt, figure)
    return receipt


def _event_from_line(line: str) -> dict[str, Any]:
    """Parse one jax_log_compiles line into a compile or phase event."""
    if line.startswith("==PHASE"):
        return {"kind": "phase", "name": line.strip().split(" ", 1)[1].strip("=")}
    if "Finished XLA compilation" in line:
        seconds = 0.0
        if " in " in line:
            try:
                seconds = float(line.rsplit(" in ", 1)[1].strip().removesuffix(" sec"))
            except ValueError:
                seconds = float("inf")
        return {"kind": "compile", "seconds": seconds}
    if line.startswith("Finished tracing") or line.startswith("Compiling jit"):
        return {"kind": "trace"}
    return {"kind": "other"}


def main() -> None:
    """Parse the caller's operands and run the measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--cache-root", type=Path, default=None)
    arguments = parser.parse_args()
    status = 0
    try:
        measure(
            output=arguments.output,
            figure=arguments.figure,
            cache_root=arguments.cache_root,
        )
    except Exception as error:  # keep the exit-status line honest
        import traceback

        traceback.print_exc()
        print(f"DRIVER-ERROR {type(error).__name__}: {error}", flush=True)
        status = 1
    print(f"DRIVER_EXIT_STATUS={status}", flush=True)


if __name__ == "__main__":
    main()
