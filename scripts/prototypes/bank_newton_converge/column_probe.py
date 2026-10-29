"""Compare assembled bank columns with the exact directional booking tangent."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_step.measure import machine


def emit(name, **values):
    print(name + " " + json.dumps(values, sort_keys=True, default=float), flush=True)


def run():
    operands, cache, roots, convention, policy, profile, quadrature, moments, _ = (
        machine("diverted")
    )
    seeds = roots + 0.03 * jnp.median(operands.geometry.pitch)
    support = jax.jit(
        lambda bank, window, state: support_read.read(
            bank, window, state, seeds, roots, convention, policy
        )
    )(operands, cache, moments)
    assert bool(support.valid)
    cells = moments[0].size

    def matvec(vector, geometry):
        direction = tuple(jnp.split(vector, (cells, 2 * cells)))
        result = support_read.step_jacobian_vector(
            operands,
            moments,
            direction,
            support,
            profile,
            lambda radius, flux: jnp.zeros_like(radius),
            quadrature,
            fresh_geometry=geometry,
        )
        return jnp.concatenate(result)

    matrices = {}
    for geometry in (False, True):
        label = "fresh" if geometry else "frozen"
        matrices[label] = jax.jit(
            lambda bank, state, read: support_read.step_jacobian(
                bank,
                state,
                read,
                profile,
                lambda radius, flux: jnp.zeros_like(radius),
                quadrature,
                fresh_geometry=geometry,
            )
        )(operands, moments, support)
        jax.block_until_ready(matrices[label])

    indices = (0, 183, 184, 507, cells, cells + 184, 2 * cells, 2 * cells + 184)
    seed = np.random.default_rng(7102)
    vectors = [jax.nn.one_hot(i, 3 * cells) for i in indices]
    vectors.append(jnp.asarray(seed.normal(size=3 * cells)))
    for index, vector in enumerate(vectors):
        label = "random" if index == len(indices) else str(indices[index])
        dense_frozen = matrices["frozen"] @ vector
        dense_fresh = matrices["fresh"] @ vector
        mapped_frozen = matvec(vector, False)
        mapped_fresh = matvec(vector, True)
        for mode, dense, mapped in (
            ("frozen", dense_frozen, mapped_frozen),
            ("fresh", dense_fresh, mapped_fresh),
            ("geometry", dense_fresh - dense_frozen, mapped_fresh - mapped_frozen),
        ):
            difference = np.asarray(dense - mapped)
            worst = int(np.argmax(np.abs(difference)))
            emit(
                "COLUMN",
                input=label,
                mode=mode,
                absolute=float(np.linalg.norm(difference)),
                relative=float(
                    np.linalg.norm(difference) / max(np.linalg.norm(mapped), 1e-30)
                ),
                worst_row=worst,
                worst_block=worst // cells,
                worst_error=float(difference[worst]),
            )


if __name__ == "__main__":
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
    run()
