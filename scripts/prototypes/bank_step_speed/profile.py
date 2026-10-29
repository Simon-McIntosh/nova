"""Time the bank step's independently compiled stages on one device."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.scipy import linalg

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_step.measure import machine


def emit(kind, **data):
    print(kind, json.dumps(data, sort_keys=True, default=float), flush=True)


def measure(name, function, *args):
    compiled = jax.jit(function)
    result = compiled(*args)
    jax.block_until_ready(result)
    walls = []
    for _ in range(3):
        started = perf_counter()
        result = compiled(*args)
        jax.block_until_ready(result)
        walls.append(1e3 * (perf_counter() - started))
    emit("STAGE", name=name, walls_ms=walls, median_ms=sorted(walls)[1])
    return result


def main():
    root = Path(__file__).resolve().parents[3]
    emit(
        "HEADER",
        revision=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        tree=str(root),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        module=support_read.__file__,
        backend=jax.default_backend(),
    )
    assert jax.config.jax_enable_x64
    bank, cache, roots, convention, policy, profile, quadrature, moments, _ = machine(
        "diverted"
    )
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    support = measure(
        "read",
        lambda b, c, m: support_read.read(b, c, m, seeds, roots, convention, policy),
        bank,
        cache,
        moments,
    )
    residual = measure(
        "residual_and_booking",
        lambda b, m, s: support_read.step_residual(b, m, m, s, profile, quadrature),
        bank,
        moments,
        support,
    )
    booked = residual.booked

    def gradient(radius, flux):
        return jnp.zeros_like(radius)

    tangent = measure(
        "local_booking_tangent",
        lambda b, m, s, p: support_read._local_booking_sensitivities(
            b, m, s, profile, gradient, quadrature, p
        ),
        bank,
        moments,
        support,
        booked,
    )
    emit("SHAPE", local_tangent=list(tangent.shape), cells=moments[0].size)
    geometry = measure(
        "geometry_tangent_one_direction",
        lambda b, m, s, p: support_read._normal_geometry_booking_tangent(
            b,
            m,
            support_read._directional_field(b, tuple(jnp.ones_like(a) for a in m))[0],
            s,
            profile,
            quadrature,
            p,
        ),
        bank,
        moments,
        support,
        booked,
    )
    emit("GEOMETRY", norm=float(jnp.linalg.norm(jnp.concatenate(tuple(geometry)))))
    frozen = measure(
        "jacobian_without_geometry",
        lambda b, m, s, p: support_read.step_jacobian(
            b, m, s, profile, gradient, quadrature, p
        ),
        bank,
        moments,
        support,
        booked,
    )
    matrix = measure(
        "jacobian_with_geometry",
        lambda b, m, s, p: support_read.step_jacobian(
            b, m, s, profile, gradient, quadrature, p, fresh_geometry=True
        ),
        bank,
        moments,
        support,
        booked,
    )
    emit("MATRIX", difference=float(jnp.linalg.norm(matrix - frozen)))
    balanced, row, column = measure(
        "equilibration",
        lambda a, m, p: support_read._equilibrate(a, m, p),
        matrix,
        moments,
        booked,
    )
    factor = measure("lu_factor", linalg.lu_factor, balanced)
    right = -jnp.concatenate(residual.residual)
    solution = measure(
        "lu_solve",
        lambda f, v, r, c: c * linalg.lu_solve(f, v / r),
        factor,
        right,
        row,
        column,
    )
    emit(
        "SOLVE",
        linear_residual=float(
            jnp.linalg.norm(matrix @ solution - right) / jnp.linalg.norm(right)
        ),
    )
    whole = jax.jit(
        lambda b, m, s: support_read.newton_step(
            b, m, s, profile, gradient, quadrature, fresh_geometry=True
        )
    )
    measure("whole_step", whole, bank, moments, support)
    jax.block_until_ready(whole(bank, moments, support))
    trace = Path(os.environ["BANK_STEP_TRACE_DIR"])
    jax.profiler.start_trace(str(trace), create_perfetto_link=False)
    for _ in range(2):
        result = whole(bank, moments, support)
        jax.block_until_ready(result)
    jax.profiler.stop_trace()
    emit("TRACE", path=str(trace))


if __name__ == "__main__":
    main()
