"""Capture coarse moving-field jets and reproduce wrongly accepted axis roots."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()
import argparse
from dataclasses import replace
from functools import partial
from pathlib import Path
import json
import jax
import jax.numpy as jnp
import numpy as np
from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures.measure import analytic_profile
from scripts.prototypes.fixed_points.solve import (
    CachedField,
    build_rows,
    select_cache_nodes,
    solve_adaptive,
)
from scripts.prototypes.fixed_points.cache import solve_with_refresh
from scripts.prototypes.read_route.bell import build_mesh
from scripts.prototypes.read_route.check_element import stack_jet
from tests.equilibrium import test_topology_read as fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    oracle, analytic, wall, axis, saddle = fixture._analytic_inputs("diverted")
    geometry = fixture._realised_hex_geometry(wall, 550)
    pitch = jnp.median(geometry.pitch)
    positions = np.asarray([axis, saddle])
    shifts = np.linspace(-2, 2, 20) * float(pitch)
    truth = positions[None] + np.stack((np.zeros(20), shifts), -1)[:, None]
    fields = [replace(analytic, shift=jnp.asarray((0.0, s))) for s in shifts]
    samples = np.asarray(geometry.sample_points)
    nodes, inverse = np.unique(
        np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
    )
    indices = inverse.reshape(samples.shape[:-1])
    mesh = build_mesh(nodes, indices, geometry)
    selected = select_cache_nodes(mesh, truth.reshape(-1, 2), float(pitch))
    lookup = np.full(len(nodes), -1, dtype=np.int32)
    lookup[selected] = np.arange(len(selected))
    print("BUILD_ROWS", len(selected), flush=True)
    reference = fixture._kernel_backed_field("diverted", oracle, analytic, geometry)
    rows = build_rows(nodes[selected], reference.coupling)
    exact = jnp.stack(tuple(reference.moments), -1)
    reference_jets = jnp.einsum("ncmq,cm->nq", rows, exact)
    _, source, _ = certificate._case("diverted-single-null", clip_mode="exact")
    profile = analytic_profile(source)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy(normal_form_radius=0.054806712567833996)
    reader = jax.jit(topology.read)
    book = jax.jit(
        lambda psi, reading: current.book(
            psi,
            reading,
            geometry,
            profile,
            current.Quadrature(sigma=convention.sigma, segments=64),
        )
    )
    nodal = jax.jit(
        lambda field: stack_jet(jax.vmap(field.evaluate)(jnp.asarray(nodes[selected])))
    )
    sample = jax.jit(
        lambda field: jax.vmap(jax.vmap(field.value))(geometry.sample_points)
    )
    executable = jax.jit(partial(solve_adaptive, iterations=24))
    seed = jnp.asarray(truth[0] + 0.07 * float(pitch))
    saved = {}
    record = []
    for index, field in enumerate(fields):
        reading = reader(field, geometry, convention, policy)
        booked = book(sample(field), reading)
        moments = jnp.stack(tuple(booked[:3]), -1)
        jets = nodal(field) - reference_jets
        cached = CachedField(
            mesh, jnp.asarray(lookup), rows, moments, jets, jnp.asarray(1.0)
        )
        result, updated, events = solve_with_refresh(
            executable, cached, seed, pitch, nodes, reference.coupling, exact, field
        )
        eigen = np.linalg.eigvalsh(np.asarray(result.jet.hessian))
        print(
            "STATE",
            index,
            "valid",
            np.asarray(result.valid),
            "eigen",
            eigen.tolist(),
            flush=True,
        )
        if index in (0, 16, 17, 18):
            total = updated.exterior + jnp.einsum(
                "ncmq,cm->nq", updated.rows, updated.moments
            )
            saved[f"jets_{index}"] = np.asarray(total)
            saved[f"lookup_{index}"] = np.asarray(updated.lookup)
            saved[f"seed_{index}"] = np.asarray(seed)
            saved[f"position_{index}"] = np.asarray(result.position)
            saved[f"previous_{index}"] = truth[index]
            record.append(
                dict(
                    state=index,
                    valid=np.asarray(result.valid).tolist(),
                    eigen=eigen.tolist(),
                )
            )
        seed = jnp.where(result.valid[:, None], result.position, seed)
    np.savez_compressed(
        args.output,
        nodes=nodes,
        indices=indices,
        pitch=np.asarray(geometry.pitch),
        **saved,
    )
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2))
    assert len(record) == 4 and all(
        r["valid"][0] and r["eigen"][0][0] * r["eigen"][0][1] < 0 for r in record
    ), record
    print(
        "REPRODUCED: unguarded adaptive Newton accepts saddle as axis "
        "on states 0,16,17,18",
        flush=True,
    )


if __name__ == "__main__":
    main()
