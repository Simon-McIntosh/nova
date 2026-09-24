"""Trace the diverted certificate rung under one of two Krylov stream bodies.

Arguments: arm (``slice1`` or ``exit``), output directory, and optionally
``instrument``. The ``slice1`` arm serves ``nova.equilibrium.fixed_point``
from the capacity-scan revision's source file through an import hook, so
everything else (benchmarks, data, caches) is this tree's. With
``instrument``, every qualified Krylov call and every operator application
inside it reports to the host: the call's residual vector, step and
diagnostics, and each application's input and output, in execution order.
The solver receipt (per-trip live residuals and per-inner-iteration records)
is written in both modes.
"""

import hashlib
import importlib.abc
import importlib.util
import json
import sys
import time
from pathlib import Path

arm, output = sys.argv[1], Path(sys.argv[2])
instrument = len(sys.argv) > 3 and sys.argv[3] == "instrument"
output.mkdir(parents=True, exist_ok=True)
SLICE1_SOURCE = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20260923T163850406961-fsri-single-site-krylov-vmap-exit/mechanism/"
    "fixed_point_e62c8110e.py"
)


class _SliceOneFixedPoint(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "nova.equilibrium.fixed_point":
            return importlib.util.spec_from_file_location(name, SLICE1_SOURCE)
        return None


if arm == "slice1":
    sys.meta_path.insert(0, _SliceOneFixedPoint())
elif arm != "exit":
    raise SystemExit(f"unknown arm {arm}")

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax
import jax.numpy as jnp
import numpy as np

assert jax.config.jax_enable_x64 is True
from nova.equilibrium import fixed_point

print(
    f"ARM {arm} fixed_point={fixed_point.__file__} instrument={instrument}", flush=True
)
assert ("e62c8110e" in fixed_point.__file__) == (arm == "slice1")

events = []


def _digest(array):
    return hashlib.sha1(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


if instrument:
    original = fixed_point._qualified_krylov_step

    def _record_application(vector, action):
        events.append(
            dict(
                kind="apply",
                input=np.array(vector, dtype=np.float64),
                output=np.array(action, dtype=np.float64),
            )
        )

    def _record_call(
        residual_vector,
        nonlinear_residual,
        step,
        unconditioned,
        reduction,
        condition,
        qualification,
    ):
        events.append(
            dict(
                kind="call",
                residual_vector=np.array(residual_vector, dtype=np.float64),
                nonlinear_residual=float(nonlinear_residual),
                step=np.array(step, dtype=np.float64),
                unconditioned_step=np.array(unconditioned, dtype=np.float64),
                achieved_reduction=float(reduction),
                projected_condition=float(condition),
                qualification=int(qualification),
            )
        )

    def traced_step(linear_action, residual_vector, nonlinear_residual, **kwargs):
        def counted(vector):
            action = linear_action(vector)
            jax.debug.callback(_record_application, vector, action, ordered=False)
            return action

        result = original(counted, residual_vector, nonlinear_residual, **kwargs)
        jax.debug.callback(
            _record_call,
            residual_vector,
            nonlinear_residual,
            result.step,
            result.unconditioned_step,
            result.achieved_reduction,
            result.projected_condition,
            result.qualification,
        )
        return result

    fixed_point._qualified_krylov_step = traced_step

from benchmarks import solovev_certificate

scratch = output / "scratch"
solovev_certificate.PART_ROOT = scratch / "parts"
solovev_certificate.FIGURE_ROOT = scratch / "figures"
solovev_certificate.DIAGNOSTIC_ROOT = scratch / "diagnostics"
started = time.perf_counter()
row = solovev_certificate._measure("diverted-single-null", -300)
elapsed = time.perf_counter() - started
solver = row["solver"]
print(
    f"TERMINAL arm={arm} residual={solver['terminal_fixed_point_residual']!r} "
    f"seconds={elapsed:.1f}",
    flush=True,
)
(output / "solver.json").write_text(json.dumps(solver, indent=2, default=str) + "\n")
if instrument:
    calls = [e for e in events if e["kind"] == "call"]
    applications = [e for e in events if e["kind"] == "apply"]
    np.savez_compressed(
        output / "events.npz",
        kinds=np.array([e["kind"] for e in events]),
        **{
            f"e{i}_{k}": v
            for i, e in enumerate(events)
            for k, v in e.items()
            if k != "kind"
        },
    )
    summary = dict(
        calls=len(calls),
        applications=len(applications),
        event_digests=[
            (e["kind"], _digest(e["input"]) + ":" + _digest(e["output"]))
            if e["kind"] == "apply"
            else ("call", _digest(e["residual_vector"]) + ":" + _digest(e["step"]))
            for e in events
        ],
    )
    (output / "events.json").write_text(json.dumps(summary) + "\n")
    print(f"EVENTS calls={len(calls)} applications={len(applications)}", flush=True)
