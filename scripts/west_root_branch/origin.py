"""Check whether a finite first drive step stays on the local zero-state branch."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import subprocess
import time

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402
from scripts.west_root_branch.measure import PRIOR, SAVED, write_json  # noqa: E402


def audit_origin(run: Path):
    """Resolve the origin in small steps and compare the first accepted root."""
    assert jax.config.jax_enable_x64
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"# origin revision: {revision} module: {iron.__file__}", flush=True)
    record = json.loads((run / "measurement.json").read_text())
    receipt = json.loads((SAVED / "receipt.json").read_text())
    with (SAVED / "solution.pkl").open("rb") as stream:
        seed = pickle.load(stream)
    gram = np.load(SAVED / "gram.npy")
    external = np.load(PRIOR / "external.npy")
    law = tuple(
        jnp.asarray(item)
        for item in iron._fit_law(receipt["table_b_T"], receipt["table_mu_relative"])
    )
    knots, coefficients, _ = iron._fit_law(
        receipt["table_b_T"], receipt["table_mu_relative"]
    )
    derivative_values = []
    for width, (a, b, c, _) in zip(np.diff(knots), coefficients.T, strict=True):
        points = [0.0, width]
        if a != 0 and 0 < -b / (3 * a) < width:
            points.append(-b / (3 * a))
        derivative_values.extend(
            3 * a * np.asarray(points) ** 2 + 2 * b * np.asarray(points) + c
        )
    operands = (jnp.asarray(gram), jnp.zeros(external.size), *law)
    zero_jacobian = jax.jit(jax.jacfwd(iron._residual))(
        jnp.zeros(external.size), operands
    )
    forcing = jax.jvp(
        lambda drive: iron._residual(
            jnp.zeros(external.size),
            (operands[0], drive * jnp.asarray(external.T.reshape(-1)), *law),
        ),
        (jnp.asarray(0.0),),
        (jnp.asarray(1.0),),
    )[1]
    tangent = np.asarray(jnp.linalg.solve(zero_jacobian, -forcing)).reshape(2, -1).T
    iron._mesh = lambda region, size: seed._mesh
    iron._interaction = lambda cells, points, plan: gram
    current = np.zeros_like(external)
    drive, increment = 0.0, 1e-6
    target_drive = record["steps"][1]["drive"]
    audit = dict(
        revision=revision,
        steps=[],
        target_drive=target_drive,
        tangent_max_A_per_m=float(np.linalg.norm(tangent, axis=1).max()),
        law_min_dH_dB=float(min(derivative_values)),
        law_max_dH_dB=float(max(derivative_values)),
        vacuum_reluctivity=1 / iron._MU0,
    )
    # A tangent check ties the first solved state to the implicit-function
    # branch at zero, independently of the coarse ramp's nonlinear solve.
    while drive < target_drive - 1e-14:
        target = min(target_drive, drive + increment)
        start = time.monotonic()
        solution = iron.solve_magnetisation(
            None,
            lambda points: target * external,
            receipt["table_b_T"],
            receipt["table_mu_relative"],
            cell_size=receipt["cell_size_m"],
            tolerance=1e-10,
            max_iterations=400,
            initial_magnetisation=current,
        )
        row = dict(
            drive=target,
            from_drive=drive,
            iterations=solution.iterations,
            residual=solution.residual_history[-1],
            converged=solution.converged,
            regularised_steps=int(np.count_nonzero(solution.regularisation_history)),
            max_M_A_per_m=float(np.linalg.norm(solution.magnetisation, axis=1).max()),
            wall_seconds=time.monotonic() - start,
        )
        print("ORIGIN_ATTEMPT " + json.dumps(row), flush=True)
        audit.setdefault("attempts", []).append(row)
        if not solution.converged or (solution.iterations > 10 and increment > 1e-6):
            increment /= 2
            write_json(run / "origin-audit.json", audit)
            if increment < 1e-8:
                audit.update(
                    complete=False, termination="minimum-drive-step", last_drive=drive
                )
                write_json(run / "origin-audit.json", audit)
                print(
                    "ORIGIN_UNRESOLVED " + json.dumps(audit["attempts"][-1]), flush=True
                )
                return audit
            continue
        current, drive = solution.magnetisation, target
        audit["steps"].append(row)
        if len(audit["steps"]) == 1:
            audit["first_tangent_relative_error"] = float(
                np.linalg.norm(current / drive - tangent) / np.linalg.norm(tangent)
            )
            assert audit["first_tangent_relative_error"] < 0.01
        np.save(run / "origin-latest.npy", current)
        write_json(run / "origin-audit.json", audit)
        if solution.iterations < 7:
            increment = min(0.005, increment * 1.6)
    # The coarse first root is replayed separately because the main driver
    # retains only its full-drive state. Start from its exact same zero seed.
    coarse = iron.solve_magnetisation(
        None,
        lambda points: target_drive * external,
        receipt["table_b_T"],
        receipt["table_mu_relative"],
        cell_size=receipt["cell_size_m"],
        tolerance=1e-10,
        max_iterations=400,
        initial_magnetisation=np.zeros_like(current),
    )
    assert coarse.converged
    audit.update(
        complete=True,
        endpoint_difference_A_per_m=float(
            np.linalg.norm(current - coarse.magnetisation, axis=1).max()
        ),
        endpoint_relative_difference=float(
            np.linalg.norm(current - coarse.magnetisation) / np.linalg.norm(current)
        ),
        coarse_iterations=coarse.iterations,
        coarse_max_M_A_per_m=float(np.linalg.norm(coarse.magnetisation, axis=1).max()),
    )
    write_json(run / "origin-audit.json", audit)
    print(
        "ORIGIN_SUMMARY "
        + json.dumps(
            {k: v for k, v in audit.items() if k not in ("steps", "attempts")}
        ),
        flush=True,
    )
    return audit
