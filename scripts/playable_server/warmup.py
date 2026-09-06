"""Warm the resident MAST operator, carrier and keyframe solve once at start.

Runs inside the serving job before the Bokeh server starts.  It loads the
22086/43 forward profile (the MAST operator reconstructed from the shot store
with the frozen-six response carrier), then drives a short warm-start chain of
keyframes through the same constrained solver the app serves.  Every line is
the serving log's receipt: the operator loaded once, each keyframe's wall and
trips, and — with ``JAX_LOG_COMPILES=1`` — the persistent-compilation-cache
hit lines the second keyframe emits for the shared operator kernels.

A keyframe that fails to converge is recorded and the warm-up continues: the
serving objective never depends on a novel solve outcome, only on the load.
"""

from __future__ import annotations

import os
from pathlib import Path
import signal
import sys

#: The checkout root; resolved from this file so the payload can run it with
#: nothing on ``sys.path`` but the interpreter's own stdlib.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402  (needs the checkout root on sys.path)

from benchmarks.receipt_raster_check import _profile_and_seed  # noqa: E402
from nova.jax.config import (  # noqa: E402
    configure_dtypes,
    configure_persistent_compilation_cache,
)


def _warm_keyframes(profile, seed) -> None:
    """Run the constrained warm-start chain the served app uses."""
    from apps.playable.production import ForwardMachine, ProductionSolver
    from apps.playable.shape import PlasmaShape

    operator = profile.operator
    machine = ForwardMachine(
        profile=profile,
        seed=seed,
        wall=np.asarray(operator.wall.coordinate, dtype=np.float64),
        identity="mast-22086-43",
    )
    solver = ProductionSolver(machine)
    print(
        "PLAYABLE_SOLVER_BUILT prescribed_circuits=%s frozen_direction=derived"
        % machine.circuit_count,
        flush=True,
    )

    # Prime from the seed, then one keyframe per step.  On this route a moved
    # constraint target descends from a fresh trace, so the second keyframe
    # re-solves the prime command warm and the persistent cache reports its
    # hits there; a moved command follows to prove the loop still steps.  The
    # reduced route's traced-target fix removes that recompile, which is the
    # swap the playable session sequence lands behind this protocol.
    chain: list[tuple[str, str | None, float | None]] = [
        ("prime", None, None),
        ("reprime", None, None),
        ("bulk_r", "bulk_r", 0.02),
    ]

    # A novel solve path must never block the serve: the whole keyframe chain
    # runs on a wall budget, and any keyframe that fails or exceeds its share
    # is recorded and skipped.  ``BaseException`` catches the timing-out
    # alarm's ``TimeoutError`` as well as solve failures; the alarm is re-armed
    # per keyframe against the same shared deadline.  The payload's own
    # timeout around this module is the process-level backstop.
    import time as _time

    budget = float(os.environ.get("PLAYABLE_KEYFRAME_BUDGET_SECONDS", "3600"))
    deadline = _time.perf_counter() + budget

    def _armed_timeout(_signum, _frame):
        raise TimeoutError("keyframe chain exceeded its wall budget")

    signal.signal(signal.SIGALRM, _armed_timeout)

    def _arm():
        remaining = deadline - _time.perf_counter()
        signal.alarm(max(1, int(remaining)))

    previous = None
    commanded = PlasmaShape()
    for index, (label, parameter, delta) in enumerate(chain, start=1):
        action = (parameter, delta) if parameter is not None else None
        next_shape = (
            commanded
            if action is None
            else commanded.apply(parameter, delta if delta is not None else 0.0)
        )
        _arm()
        try:
            result = solver(previous, next_shape, action=action)
            print(
                "PLAYABLE_KEYFRAME %d parameter=%s wall=%.3fs trips=%d status=ok"
                % (index, label, result.wall, result.trips),
                flush=True,
            )
            previous = result.equilibrium
            commanded = next_shape
        except BaseException as error:  # keep serving even if a keyframe is novel
            print(
                "PLAYABLE_KEYFRAME %d parameter=%s status=skipped error=%s: %s"
                % (index, label, type(error).__name__, error),
                flush=True,
            )
    signal.alarm(0)


def main() -> None:
    """Load the operator and carrier once, warm the keyframes, and report."""
    configure_dtypes()
    cache_root = Path(
        os.environ.get("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache"))
    ).expanduser()
    configure_persistent_compilation_cache(cache_root, minimum_compile_seconds=0.0)

    import time

    loaded_started = time.perf_counter()
    case, profile, target_current, _cache_receipt, _policy = _profile_and_seed()
    load_wall = time.perf_counter() - loaded_started
    operator = profile.operator
    seed = np.asarray(case["state"])
    print(
        "PLAYABLE_OPERATOR_LOADED shot=22086 time=43 grid_nodes=%d "
        "prescribed_circuits=%d wall=%.1fs"
        % (
            int(operator.grid.node_number),
            len(profile.operator.prescribed_current_field.current),
            load_wall,
        ),
        flush=True,
    )
    print(
        "PLAYABLE_TARGET_CURRENT_A=%.3f" % abs(float(target_current)),
        flush=True,
    )
    _warm_keyframes(profile, seed)


if __name__ == "__main__":
    main()
