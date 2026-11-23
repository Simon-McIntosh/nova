"""Compare the in-place Cholesky and LU Newton routes on a persisted operator.

Both routes solve the identical piecewise-linear magnetisation state from one
persisted interaction, in one process, so any difference between them is the
route and not the mesh, the operator or a fresh-process nondeterminism. The
reported agreement is the interaction-derived coefficients and the probe fields,
each required within 1e-12 relative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import numpy as np  # noqa: E402
from nova.biot import magnetisation as iron  # noqa: E402
from scripts.iron_graded_refinement.measure import west_inputs  # noqa: E402
from scripts.west_iron_linear.measure import (  # noqa: E402
    coefficient_agreement,
    mesh_size,
    probe_agreement,
    solve,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--mesh", default="uniform-0.08")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    assert jax.config.jax_enable_x64
    jax.config.update("jax_enable_compilation_cache", False)
    print(f"device {jax.devices()[0].platform} x64 {jax.config.jax_enable_x64}")
    print(f"module {iron.__file__}")

    inputs = west_inputs()
    size = mesh_size(args.mesh, inputs)
    interaction = iron.load_magnetisation_interaction(
        args.operator, region=inputs["region"], cell_size=size, basis="linear"
    )
    row = dict(
        mesh=args.mesh,
        operator=str(args.operator),
        geometry_digest=interaction.geometry_digest,
        unknowns=int(interaction.matrix.shape[0]),
        tolerance=args.tolerance,
        completed=False,
    )
    coefficients = {}
    for route in ("combined", "cholesky"):

        def solution(result, route=route):
            coefficients[route] = np.asarray(result.linear_coefficients).copy()

        row[route] = solve(
            inputs,
            size,
            interaction,
            factorisation=route,
            observe_solution=solution,
            tolerance=args.tolerance,
        )
        print(
            f"{route}: converged {row[route]['converged']} "
            f"iterations {row[route]['iterations']} "
            f"residual {row[route]['residual_history'][-1]:.3e}",
            flush=True,
        )
        args.output.write_text(json.dumps(row, indent=2) + "\n")
        assert row[route]["converged"], row[route]["termination"]
    reference = np.asarray(row["combined"]["field_T"])
    candidate = np.asarray(row["cholesky"]["field_T"])
    row["probe_relative"] = (
        np.linalg.norm(candidate - reference, axis=1)
        / np.linalg.norm(reference, axis=1)
    ).tolist()
    expected = np.asarray(coefficients["combined"])
    actual = np.asarray(coefficients["cholesky"])
    row["coefficient_relative"] = dict(
        frobenius_relative=float(
            np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        ),
        maximum_entry_relative=float(
            np.abs(actual - expected).max() / np.abs(expected).max()
        ),
    )
    args.output.write_text(json.dumps(row, indent=2) + "\n")
    print(
        "AGREEMENT "
        + json.dumps(
            dict(
                probe=row["probe_relative"],
                coefficient=row["coefficient_relative"],
            )
        ),
        flush=True,
    )
    row["probe_pass"] = max(row["probe_relative"]) <= 1e-12
    row["coefficient_pass"] = max(row["coefficient_relative"].values()) <= 1e-12
    row["completed"] = bool(row["probe_pass"] and row["coefficient_pass"])
    args.output.write_text(json.dumps(row, indent=2) + "\n")
    print(
        f"PASS probe {row['probe_pass']} coefficient {row['coefficient_pass']}",
        flush=True,
    )
    if not row["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()