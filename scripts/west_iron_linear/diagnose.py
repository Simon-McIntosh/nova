"""Check full-size Cholesky and the weighted Newton matrix independently."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.sparse.linalg import LinearOperator, eigsh  # noqa: E402
from nova.biot import magnetisation as iron  # noqa: E402
from scripts.iron_graded_refinement.measure import provenance, west_inputs  # noqa: E402
from scripts.west_iron_linear.measure import mesh_size, solve  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--blob", required=True)
    args = parser.parse_args()
    row = dict(
        **provenance(Path(__file__).resolve(), args.revision, args.blob),
        completed=False,
    )
    assert jax.config.jax_enable_x64 and jax.devices()[0].platform == "gpu"
    jax.config.update("jax_enable_compilation_cache", False)
    path = args.output / "cholesky-diagnostic.json"

    def save():
        path.write_text(json.dumps(row, indent=2) + "\n")

    programs = iron._linear_cholesky_programs(jax.devices()[0])
    column, _, factor, _ = programs
    row["known_spd"] = {}
    for dimension in (35184, 70248):
        matrix = jax.jit(
            lambda: jnp.eye(dimension, dtype=jnp.float64), out_shardings=column
        )()
        before = matrix.unsafe_buffer_pointer()
        triangle = factor(matrix)
        diagonal = np.asarray(jnp.diag(triangle))
        row["known_spd"][str(dimension)] = dict(
            finite=bool(np.isfinite(diagonal).all()),
            diagonal_error=float(np.max(np.abs(diagonal - 1))),
            donated=triangle.unsafe_buffer_pointer() == before,
            memory=jax.devices()[0].memory_stats(),
        )
        save()
        assert row["known_spd"][str(dimension)]["finite"]
        assert row["known_spd"][str(dimension)]["diagonal_error"] == 0
        del triangle
    inputs = west_inputs()
    size = mesh_size("graded-16-80", inputs)
    interaction = iron.load_magnetisation_interaction(
        args.output.parent / "graded-16-80-operator.npy",
        region=inputs["region"],
        cell_size=size,
        basis="linear",
    )
    local_owner = iron._linear_cholesky_local
    captured = {}

    def local(state, operands, metric):
        result = local_owner(state, operands, metric)
        captured.update(
            operands=operands, local=result[0], root=result[1], transform=result[2]
        )
        return result

    class Captured(RuntimeError):
        pass

    def inspect(matrix):
        dimension = matrix.shape[0]
        multiply = jax.jit(
            lambda matrix, vector: matrix @ vector, in_shardings=(column, None)
        )
        random = np.random.default_rng(437)
        x, y = (jnp.asarray(random.normal(size=dimension)) for _ in range(2))
        bx, by = multiply(matrix, x), multiply(matrix, y)
        direct = x - iron._local_apply(
            captured["root"],
            iron._linear_response(
                iron._local_apply(captured["transform"], x), captured["operands"]
            ),
        )
        row["matrix"] = dict(
            diagonal_min=float(jnp.min(jnp.diag(matrix))),
            diagonal_max=float(jnp.max(jnp.diag(matrix))),
            direct_action_relative=float(
                jnp.linalg.norm(bx - direct) / jnp.linalg.norm(direct)
            ),
            bilinear_asymmetry=float(
                jnp.abs(x @ by - y @ bx) / (jnp.linalg.norm(x) * jnp.linalg.norm(by))
            ),
        )
        save()
        operator = LinearOperator(
            (dimension, dimension),
            matvec=lambda v: np.asarray(multiply(matrix, jnp.asarray(v))),
            dtype=np.float64,
        )
        values, vectors = eigsh(
            operator,
            k=1,
            which="SA",
            tol=1e-6,
            maxiter=300,
            v0=random.normal(size=dimension),
            ncv=40,
        )
        witness = vectors[:, 0]
        action = operator @ witness
        row["matrix"].update(
            minimum_eigenvalue_estimate=float(values[0]),
            witness_rayleigh=float(witness @ action / (witness @ witness)),
            witness_residual=float(np.linalg.norm(action - values[0] * witness)),
        )
        np.save(args.output / "cholesky-rayleigh-witness.npy", witness)
        row["completed"] = True
        save()
        raise Captured("weighted matrix captured before Cholesky")

    def diagnostic_programs(device):
        return programs[0], programs[1], inspect, programs[3]

    with (
        patch.object(iron, "_linear_cholesky_local", local),
        patch.object(iron, "_linear_cholesky_programs", diagnostic_programs),
    ):
        try:
            solve(inputs, size, interaction, factorisation="cholesky", tolerance=1e-12)
        except Captured:
            pass
    assert row["completed"]
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
