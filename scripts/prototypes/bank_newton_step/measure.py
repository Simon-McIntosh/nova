"""Measure the bank-backed booking tangent at a certificate state."""

# ruff: noqa: E402
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from nova.database import content_store
from nova.equilibrium import current, fixed_points, support_read, topology
from nova.equilibrium.solve_request import TopologyPolicy


ROOT = Path(__file__).resolve().parents[3]
STORE = Path("/work/projects/imas_gpu/sophelio/nova-assembled-step-bank")


def emit(name, **values):
    print(name + " " + json.dumps(values, sort_keys=True, default=float), flush=True)


def machine():
    import tests.equilibrium.test_topology_read as fixture
    from benchmarks import solovev_certificate as certificate
    from scripts.analytic_oracle_fixtures.measure import analytic_profile

    receipt = json.loads((STORE / "receipt.json").read_text())
    digest = receipt["digest"]
    oracle, analytic, wall, axis, saddle = fixture._analytic_inputs("diverted")
    geometry = fixture._realised_hex_geometry(wall, 550)
    object_root = content_store.verified_object_root(STORE, create=False)
    destination = content_store.verified_destination(
        object_root, content_store.digest_hex(digest)
    )
    with np.load(destination / "bank.npz", allow_pickle=False) as archive:
        nodes = archive["nodes"]
        sample_index = archive["sample_index"]
    mesh = fixed_points.build_mesh(nodes, sample_index, geometry)
    operands = support_read.read_bank(STORE, digest, geometry, mesh)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy(normal_form_radius=0.054806712567833996)
    _, source, _ = certificate._case("diverted-single-null", clip_mode="exact")
    profile = analytic_profile(source)
    quadrature = current.Quadrature(sigma=convention.sigma, segments=64)
    roots = jnp.asarray([axis, saddle])
    psi = jax.vmap(jax.vmap(analytic.value))(geometry.sample_points)
    analytic_read = jax.jit(topology.read)(analytic, geometry, convention, policy)
    booked = current.book(psi, analytic_read, geometry, profile, quadrature)
    moments = booked[:3]
    cache = fixed_points.RowCache(
        mesh, operands.nodes, jnp.arange(len(nodes)), operands.rows, operands.exterior
    )
    return operands, cache, roots, convention, policy, profile, quadrature, moments


def main():
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    print(
        f"REVISION={revision} TREE={ROOT} CWD={Path.cwd().resolve()} "
        f"MODULE={support_read.__file__} NOVA={__import__('nova').__file__}",
        flush=True,
    )
    emit(
        "DEVICE", backend=jax.default_backend(), devices=[str(d) for d in jax.devices()]
    )
    print("STAGE load", flush=True)
    operands, cache, roots, convention, policy, profile, quadrature, moments = machine()
    jax.block_until_ready(moments)
    emit("BANK", rows=list(operands.rows.shape), bucket=operands.bucket)
    seed = roots + 0.03 * jnp.median(operands.geometry.pitch)

    @jax.jit
    def read_state(operands, cache, moments):
        return support_read.read(
            operands, cache, moments, seed, roots, convention, policy
        )

    print("STAGE read", flush=True)
    support = read_state(operands, cache, moments)
    jax.block_until_ready(support)
    emit(
        "SUPPORT",
        valid=bool(support.valid),
        read_valid=bool(support.reading.valid),
        qualified=bool(support.reading.qualified),
        cut=int(
            jnp.sum((support.reading.membership > 0) & (support.reading.membership < 1))
        ),
        normal=int(jnp.sum(support.reading.normal_form_cells)),
    )
    direction = tuple(jnp.linspace(0.1, 0.4, value.shape[0]) for value in moments)

    @jax.jit
    def residual(operands, base_moments, moments_in, support):
        return support_read.step_residual(
            operands, base_moments, moments_in, support, profile, quadrature
        )

    @jax.jit
    def applied(operands, moments, direction, support):
        return support_read.step_jacobian_vector(
            operands,
            moments,
            direction,
            support,
            profile,
            lambda radius, flux: jnp.zeros_like(radius),
            quadrature,
        )

    print("STAGE tangent", flush=True)
    product = applied(operands, moments, direction, support)
    jax.block_until_ready(product)
    emit("TANGENT", finite=bool(jnp.all(jnp.isfinite(jnp.concatenate(product)))))
    for step in (1e-5, 1e-6, 1e-7, 1e-8):
        plus = tuple(a + step * d for a, d in zip(moments, direction, strict=True))
        minus = tuple(a - step * d for a, d in zip(moments, direction, strict=True))
        high = residual(operands, moments, plus, support)
        low = residual(operands, moments, minus, support)
        jax.block_until_ready((high, low))
        difference_parts = tuple(
            (a - b) / (2 * step)
            for a, b in zip(high.residual, low.residual, strict=True)
        )
        difference = jnp.concatenate(difference_parts)
        vector = jnp.concatenate(product)
        error = jnp.linalg.norm(vector - difference) / jnp.linalg.norm(difference)
        direction_flat = jnp.concatenate(direction)
        booking_error = jnp.linalg.norm(vector - difference) / jnp.linalg.norm(
            difference - direction_flat
        )
        per_cell = np.asarray(
            jnp.linalg.norm(
                jnp.stack(
                    tuple(
                        a - b for a, b in zip(product, difference_parts, strict=True)
                    ),
                    axis=-1,
                ),
                axis=-1,
            )
        )
        worst = np.argsort(per_cell)[-8:][::-1]
        emit(
            "DIFFERENCE",
            step=step,
            relative=float(error),
            booking_relative=float(booking_error),
            worst_cells=worst.tolist(),
            worst_errors=per_cell[worst].tolist(),
            worst_normal=np.asarray(support.reading.normal_form_cells)[worst].tolist(),
            high_valid=bool(high.valid),
            low_valid=bool(low.valid),
            finite=bool(jnp.all(jnp.isfinite(difference))),
        )


if __name__ == "__main__":
    main()
