"""Exercise the certificate receipt on one reduced-Newton terminal state."""

import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import jax
import numpy as np

from benchmarks import solovev_certificate as certificate
from benchmarks.exact_clip_seed_amplitude import _problem
from nova.equilibrium.forward_operator import set_support_clip_mode
from nova.equilibrium.solve_request import ExplicitSolveSeed, ForwardSolveRequest
from nova.jax.config import configure_dtypes


ROOT = Path(__file__).resolve().parent


def main():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"revision={revision} tree={Path.cwd()} command={' '.join(sys.argv)}",
        flush=True,
    )
    configure_dtypes()
    assert jax.config.jax_enable_x64
    assert jax.default_backend() == "cpu"
    set_support_clip_mode("chord")
    case = "weak-rotation-reactor-static"
    started = perf_counter()
    machine, exact, analytic, operator, profile, target, centroid, current = _problem(
        case, -300
    )
    seed, _, _ = certificate._production_seed(profile, case, target, centroid, current)
    request = ForwardSolveRequest.from_defaults(
        carrier_identity="receipt:weak-rotation-reactor-static:300:chord",
        source_profile=profile.source,
        seed_policy=ExplicitSolveSeed(seed),
        policy_overrides={
            "route": "reduced_newton",
            "newton_steps": certificate.recovery.NEWTON_STEPS,
            "gmres_iterations": certificate.recovery.KRYLOV_ITERATIONS,
            "warmup": 0,
            "kernel_tolerance": certificate.TERMINAL_RESIDUAL_BOUND,
            "qualification_tolerance": certificate.TERMINAL_RESIDUAL_BOUND,
        },
        target_current=target,
    )
    equilibrium = profile.solve(request).equilibrium
    jax.block_until_ready(equilibrium.flux)
    history = equilibrium.fixed_point
    print(
        f"SOLVED cells={len(machine.node)} residual={float(history.residual)} "
        f"converged={bool(history.converged)} wall={perf_counter() - started}",
        flush=True,
    )
    shapes = {name: np.shape(getattr(history, name)) for name in history._fields}
    print(f"HISTORY_SHAPES {json.dumps(shapes, sort_keys=True)}", flush=True)
    production = certificate._production_solver_receipt(equilibrium)
    payload = {
        "revision": revision,
        "route": "reduced_newton",
        "case": case,
        "requested_cells": 300,
        "realised_cells": len(machine.node),
        "clip_mode": "chord",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "platform": jax.default_backend(),
        "x64": jax.config.jax_enable_x64,
        "history_shapes": shapes,
        "residual": float(history.residual),
        "converged": bool(history.converged),
        "production_solver": production,
        "elapsed_seconds": perf_counter() - started,
    }
    certificate._write_json(ROOT / "receipt.json", payload)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    terminal = np.asarray(equilibrium.flux)
    points, gradient, hessian = certificate._quadratic_derivatives(
        profile.lattice, terminal[: len(machine.node)]
    )
    exact_gradient, exact_hessian = certificate._exact_derivatives(case, exact, points)
    figure = certificate._plot(
        coordinates,
        terminal,
        analytic,
        points,
        {
            "psi": terminal[: len(machine.node)] - analytic[: len(machine.node)],
            "gradient": gradient - exact_gradient,
            "hessian": hessian - exact_hessian,
        },
        certificate._boundary(case, exact),
        machine.wall_node,
        certificate._topology(operator, terminal),
        certificate._topology(operator, analytic),
        ROOT / "terminal.png",
        f"Reduced Newton: residual={float(history.residual):.6g}; "
        f"converged={bool(history.converged)}",
    )
    certificate.plt.close(figure)
    print("RECEIPT_WRITTEN receipt.json", flush=True)


if __name__ == "__main__":
    main()
