"""Measure dense Newton linear routes on identical, archived batch systems."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
from statistics import median
import subprocess
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.scipy import linalg
from jax._src.scipy.sparse.linalg import _kth_arnoldi_iteration, _lstsq
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_solve_profile import fixture


def emit(label, **row):
    print(label + " " + json.dumps(row), flush=True)


def relative(value, reference):
    return jnp.linalg.norm(value - reference) / jnp.linalg.norm(reference)


def counted_gmres(
    matrix, right, original, raw_right, column, condition, physical_bound=False
):
    """Restarted left-preconditioned GMRES with exact Arnoldi step telemetry."""
    cells = right.size // 3
    indices = jnp.arange(cells)[:, None] + cells * jnp.arange(3)[None, :]
    blocks = matrix[indices[:, :, None], indices[:, None, :]]
    inverse = jnp.linalg.inv(blocks)

    def precondition(vector):
        return jnp.einsum("cij,cj->ci", inverse, vector[indices]).T.reshape(-1)

    def error(solution):
        balanced = condition[1] * relative(matrix @ solution, right)
        if not physical_bound:
            return balanced
        physical = condition[0] * relative(original @ (column * solution), raw_right)
        return jnp.maximum(balanced, physical)

    def cycle(carried):
        solution, cycles, steps = carried
        residual = precondition(right - matrix @ solution)
        norm = jnp.linalg.norm(residual)
        basis = jnp.zeros((right.size, 21)).at[:, 0].set(residual / norm)
        hessenberg = jnp.eye(20, 21)

        def advance(state):
            vectors, h, _, count = state
            vectors, h, breakdown = _kth_arnoldi_iteration(
                count, lambda v: matrix @ v, precondition, vectors, h
            )
            return vectors, h, breakdown, count + 1

        basis, hessenberg, _, count = jax.lax.while_loop(
            lambda state: (state[3] < 20) & ~state[2],
            advance,
            (basis, hessenberg, jnp.asarray(False), jnp.asarray(0)),
        )
        target = jnp.zeros(21).at[0].set(norm)
        solution = solution + basis[:, :-1] @ _lstsq(hessenberg.T, target)
        return solution, cycles + 1, steps + count

    solution, _, steps = jax.lax.while_loop(
        lambda state: (state[1] < 20) & (error(state[0]) > 1e-8),
        cycle,
        (jnp.zeros_like(right), jnp.asarray(0), jnp.asarray(0)),
    )
    return column * solution, steps


def make_arm(name, refinements=0):
    def one(matrix, right, row, column):
        balanced = matrix * column[None, :] / row[:, None]
        rhs = right / row
        if name.startswith("gmres"):
            condition = jnp.stack(
                tuple(
                    support_read._krylov_condition_estimate(lambda v: a @ v, right.size)
                    for a in (
                        (matrix, balanced)
                        if name == "gmres-strict"
                        else (balanced, balanced)
                    )
                )
            )
            return counted_gmres(
                balanced,
                rhs,
                matrix,
                right,
                column,
                condition,
                physical_bound=name == "gmres-strict",
            )
        working = balanced.astype(jnp.float32) if name == "refined" else balanced
        lu, pivots = linalg.lu_factor(working)
        solution = linalg.lu_solve((lu, pivots), rhs.astype(working.dtype)).astype(
            jnp.float64
        )
        if name == "refined":

            def correct(_, solution):
                residual = rhs - balanced @ solution
                correction = linalg.lu_solve((lu, pivots), residual.astype(jnp.float32))
                return solution + correction.astype(jnp.float64)

            solution = jax.lax.fori_loop(0, refinements, correct, solution)
        return column * solution, jnp.asarray(refinements)

    if name != "stacked":
        return jax.jit(jax.vmap(one))

    @jax.jit
    def stacked(matrix, right, row, column):
        balanced = matrix * column[:, None, :] / row[:, :, None]
        lu, pivots, _ = jax.lax.linalg.lu(balanced)
        solution = linalg.lu_solve((lu, pivots), (right / row)[..., None])[..., 0]
        return column * solution, jnp.zeros(right.shape[0], dtype=jnp.int32)

    return stacked


def kernel_count(program, arguments, directory):
    with jax.profiler.trace(str(directory), create_perfetto_link=False):
        jax.block_until_ready(program(*arguments))
    path = next(directory.rglob("*.xplane.pb"))
    data = jax.profiler.ProfileData.from_file(str(path))
    events = [
        event
        for plane in data.planes
        if "GPU" in plane.name
        for line in plane.lines
        if line.name.lower().startswith("stream")
        for event in line.events
        if event.duration_ns
        and "memcpy" not in event.name.lower()
        and "memset" not in event.name.lower()
    ]
    assert events, "device trace must see the known-present solve kernels"
    return len(events)


def extract(batch, directory):
    def reading(state):
        return support_read.read_batch(
            batch.operands,
            batch.cache,
            state,
            batch.seeds_batch,
            batch.previous_batch,
            batch.convention,
            batch.policy,
        )

    @jax.jit
    def systems(state, support):
        def one(moments, read):
            step = support_read.step_residual(
                batch.operands, moments, moments, read, batch.profile, batch.quadrature
            )
            matrix = support_read.step_jacobian(
                batch.operands,
                moments,
                read,
                batch.profile,
                fixture.gradient,
                batch.quadrature,
                booked=step.booked,
                fresh_geometry=True,
            )
            _, row, column = support_read._equilibrate(matrix, moments, step.booked)
            return matrix, -jnp.concatenate(step.residual), row, column

        return jax.vmap(one)(state, support)

    @jax.jit
    def solve(state):
        return support_read.solve_batch(
            batch.operands,
            batch.cache,
            state,
            batch.seeds_batch,
            batch.previous_batch,
            batch.convention,
            batch.policy,
            batch.profile,
            fixture.gradient,
            batch.quadrature,
        )

    for name in ("trial", "converged"):
        if name == "trial":
            states = batch.states
            support = jax.jit(reading)(states)
        else:
            result = jax.block_until_ready(solve(batch.states))
            emit(
                "SOLVED",
                converged=np.asarray(result.converged).tolist(),
                residual=np.asarray(result.residual).tolist(),
                steps=np.asarray(result.steps).tolist(),
            )
            assert bool(jnp.all(result.converged))
            states = result.moments
            support = jax.jit(reading)(states)
        arrays = jax.block_until_ready(systems(states, support))
        assert all(bool(jnp.all(jnp.isfinite(value))) for value in arrays)
        np.savez(
            directory / f"{name}.npz",
            **dict(
                zip(
                    ("matrix", "right", "row", "column"),
                    map(np.asarray, arrays),
                    strict=True,
                )
            ),
        )
        emit("EXTRACTED", state=name, shape=list(arrays[0].shape))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--systems", type=Path)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    emit(
        "HEADER",
        revision=revision,
        tree=str(root),
        cwd=str(Path.cwd()),
        nova=nova.__file__,
        module=support_read.__file__,
        jax=jax.__version__,
        devices=str(jax.devices()),
        x64=jax.config.jax_enable_x64,
    )
    assert jax.config.jax_enable_x64 and jax.default_backend() == "gpu"
    assert args.repeats >= 10
    args.output.mkdir(parents=True, exist_ok=True)
    systems = args.systems or args.output
    if args.systems is None:
        extract(fixture.build(16), systems)
    emit("SYSTEMS", path=str(systems))
    receipt = []
    arms = [
        ("present", 0),
        ("refined", 0),
        ("refined", 1),
        ("refined", 2),
        ("refined", 3),
        ("gmres", 0),
        ("gmres-strict", 0),
        ("stacked", 0),
    ]
    for state in ("trial", "converged"):
        with np.load(systems / f"{state}.npz") as archive:
            full = tuple(
                jnp.asarray(archive[key])
                for key in ("matrix", "right", "row", "column")
            )
        for lanes in (1, 16):
            arguments = tuple(value[:lanes] for value in full)
            matrix, right, row, column = arguments
            reference = jax.jit(jax.vmap(jnp.linalg.solve))(matrix, right)
            balanced = matrix * column[:, None, :] / row[:, :, None]
            estimate = jax.jit(
                jax.vmap(
                    lambda a: support_read._krylov_condition_estimate(
                        lambda v: a @ v, a.shape[0]
                    )
                )
            )
            conditions = [np.asarray(estimate(a)).tolist() for a in (matrix, balanced)]
            reference_residual = np.asarray(
                jax.vmap(relative)(jnp.einsum("bij,bj->bi", matrix, reference), right)
            ).tolist()
            for name, refinements in arms:
                label = f"{name}-{refinements}" if name == "refined" else name
                program = make_arm(name, refinements)
                update, count = jax.block_until_ready(program(*arguments))
                samples = []
                for _ in range(args.repeats):
                    start = perf_counter()
                    jax.block_until_ready(program(*arguments))
                    samples.append(1000 * (perf_counter() - start))
                errors = np.asarray(jax.vmap(relative)(update, reference))
                residuals = np.asarray(
                    jax.vmap(relative)(jnp.einsum("bij,bj->bi", matrix, update), right)
                )
                balanced_residuals = np.asarray(
                    jax.vmap(relative)(
                        jnp.einsum("bij,bj->bi", balanced, update / column), right / row
                    )
                )
                path = args.output / f"trace-{state}-{lanes}-{label}"
                launches = kernel_count(program, arguments, path)
                bound = np.maximum(
                    np.asarray(conditions[0]) * residuals,
                    np.asarray(conditions[1]) * balanced_residuals,
                )
                if name == "gmres":
                    bound = np.asarray(conditions[1]) * balanced_residuals
                passed = bool(
                    np.all(errors <= 1e-8)
                    and (not name.startswith("gmres") or np.all(bound <= 1e-8))
                )
                record = dict(
                    state=state,
                    lanes=lanes,
                    arm=label,
                    median_ms=median(samples),
                    samples_ms=samples,
                    count=np.asarray(count).tolist(),
                    kernels=launches,
                    errors=errors.tolist(),
                    condition_original=conditions[0],
                    condition_balanced=conditions[1],
                    residuals=residuals.tolist(),
                    balanced_residuals=balanced_residuals.tolist(),
                    bound=bound.tolist(),
                    reference_residual=reference_residual,
                    passed=passed,
                )
                receipt.append(record)
                (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))
                emit("ARM", **record)
    emit(
        "SUMMARY",
        arms=len(receipt),
        precision_passes=sum(r["passed"] for r in receipt),
        fastest_batch_ms=min(
            r["median_ms"] for r in receipt if r["lanes"] == 16 and r["passed"]
        ),
    )


if __name__ == "__main__":
    main()
