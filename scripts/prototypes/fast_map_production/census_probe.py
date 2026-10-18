"""Inspect the production Newton root contract at a moving-field tie."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import fixed_points, topology
from scripts.prototypes.fast_map_production import tie_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    import nova

    print(
        f"nova={nova.__file__} fixed_points={fixed_points.__file__} "
        f"cwd={Path.cwd().resolve()} backend={jax.default_backend()}",
        flush=True,
    )
    state = tie_probe.build("diverted", 250, args.cache)
    sampled = tie_probe.make_sampler(state)
    prior, warm_count = tie_probe.warm(state, sampled, -0.2, 0.16, 0.01)

    def count(offset):
        roots, *_ = sampled(jnp.asarray(offset * state["scale"]), prior)
        return int(jnp.sum(roots.valid))

    left, right = 0.17, 0.18
    endpoint_counts = (count(left), count(right))
    if endpoint_counts[0] <= endpoint_counts[1]:
        raise RuntimeError(
            f"census departure absent at scan endpoints: {endpoint_counts}"
        )
    while np.nextafter(left, right) < right:
        middle = (left + right) / 2
        if middle in (left, right):
            break
        if count(middle) == endpoint_counts[0]:
            left = middle
        else:
            right = middle
    offsets = (left, right)

    def candidate_evidence(field, target):
        geometry = state["geometry"]
        policy = state["policy"]
        coefficients = field.sampled_coefficient(geometry)
        hessian = jnp.stack(
            (
                jnp.stack((2 * coefficients[:, 3], coefficients[:, 4]), axis=-1),
                jnp.stack((coefficients[:, 4], 2 * coefficients[:, 5]), axis=-1),
            ),
            axis=1,
        )
        conditioned, safe = jax.vmap(topology._conditioned_hessian, in_axes=(0, None))(
            hessian, policy.hessian_tolerance
        )
        displacement = -jnp.linalg.solve(safe, coefficients[:, 1:3, None])[..., 0]
        seeds = geometry.centre + geometry.pitch[:, None] * displacement
        slot = jnp.arange(geometry.vertices.shape[1])
        following = jnp.where(
            slot[None] + 1 < geometry.vertex_count[:, None], slot[None] + 1, 0
        )
        end = jnp.take_along_axis(geometry.vertices, following[..., None], axis=1)
        cross = topology._cross_plane(
            end - geometry.vertices, seeds[:, None] - geometry.vertices
        )
        inside = jnp.all(
            jnp.where(slot[None] < geometry.vertex_count[:, None], cross >= 0, True),
            axis=1,
        )
        candidate = conditioned & inside & jnp.all(jnp.isfinite(seeds), axis=1)
        seed_np = np.asarray(seeds)
        centre_np = np.asarray(geometry.centre)
        candidate_np = np.asarray(candidate)
        seed_capacity = max(4 * policy.null_capacity, prior.shape[0])
        selected = jnp.nonzero(candidate, size=seed_capacity, fill_value=0)[0]
        all_seeds = jnp.concatenate((prior, seeds[selected]))
        active = jnp.concatenate(
            (
                jnp.ones(prior.shape[0], bool),
                jnp.arange(seed_capacity) < jnp.sum(candidate),
            )
        )
        polished = jax.vmap(topology.stationary_read, in_axes=(None, 0, None, None))(
            field, all_seeds, jnp.median(geometry.pitch), policy
        )
        inside_wall = jax.vmap(
            lambda point: jnp.any(topology._inside_cells(geometry, point))
        )(polished.position)
        live = active & polished.valid & inside_wall
        separation = jnp.linalg.norm(
            polished.position[:, None] - polished.position[None], axis=-1
        )
        earlier = jnp.arange(len(live))[None] < jnp.arange(len(live))[:, None]
        duplicate = jnp.any(
            earlier
            & live[None]
            & (separation < 8 * policy.position_tolerance * jnp.min(geometry.pitch)),
            axis=1,
        )
        position_np = np.asarray(polished.position)
        all_seed_np = np.asarray(all_seeds)
        gradient_np = np.asarray(polished.jet.gradient)
        hessian_np = np.asarray(polished.jet.hessian)
        near_polished = np.flatnonzero(
            np.asarray(active)
            & (
                (np.linalg.norm(all_seed_np - target, axis=1) < 0.2)
                | (np.linalg.norm(position_np - target, axis=1) < 0.2)
            )
        )
        order = np.argsort(np.linalg.norm(seed_np - target, axis=1))[:6]
        owner = np.flatnonzero(np.asarray(topology._inside_cells(geometry, target)))
        indices = np.unique(np.concatenate((order, owner)))
        rank = np.cumsum(candidate_np) - 1
        return {
            "candidate_count": int(candidate_np.sum()),
            "owner_cells": owner.tolist(),
            "polished_near_target": [
                {
                    "index": int(index),
                    "cell": int(selected[index - len(prior)])
                    if index >= len(prior)
                    else None,
                    "seed": all_seed_np[index].tolist(),
                    "position": position_np[index].tolist(),
                    "valid": bool(polished.valid[index]),
                    "reason": int(polished.reason[index]),
                    "inside_wall": bool(inside_wall[index]),
                    "duplicate": bool(duplicate[index]),
                    "gradient_norm": float(np.linalg.norm(gradient_np[index])),
                    "correction_norm": float(
                        np.linalg.norm(
                            np.linalg.pinv(hessian_np[index]) @ gradient_np[index]
                        )
                    ),
                    "determinant": float(np.linalg.det(hessian_np[index])),
                }
                for index in near_polished
            ],
            "near_target": [
                {
                    "cell": int(index),
                    "centre": centre_np[index].tolist(),
                    "seed": seed_np[index].tolist(),
                    "distance": float(np.linalg.norm(seed_np[index] - target)),
                    "conditioned": bool(conditioned[index]),
                    "inside": bool(inside[index]),
                    "candidate": bool(candidate_np[index]),
                    "rank": int(rank[index]) if candidate_np[index] else None,
                }
                for index in indices
            ],
        }

    rows = []
    for offset in offsets:
        shifted = jnp.asarray(offset * state["scale"])
        roots, reading, _, _, _, _ = sampled(shifted, prior)
        jets = replace(
            state["jets"],
            exterior=state["jets"].exterior + shifted * state["jet_slope"],
        )
        window = fixed_points.RowCache(
            jets.mesh,
            jets.nodes,
            jnp.arange(len(jets.nodes), dtype=jnp.int32),
            jets.rows,
            jets.exterior,
        )
        seeds = roots.position if not rows else rows[0]["seeds"]
        axis_sign = jnp.sign(roots.jet.hessian[0, 0, 0])
        solved = fixed_points.solve(
            window,
            state["reference"],
            seeds,
            seeds,
            jnp.median(state["geometry"].pitch),
            axis_sign,
            iterations=12,
        )
        jax.block_until_ready(solved)
        hessian = np.asarray(solved.jet.hessian)
        gradient = np.asarray(solved.jet.gradient)
        correction = np.einsum("nij,nj->ni", np.linalg.pinv(hessian), gradient)
        cached = fixed_points._CachedField(
            window.mesh, window.lookup, window.rows, state["reference"], window.exterior
        )
        contained = jax.vmap(cached.contains)(solved.position)
        target = np.asarray(
            roots.position[2] if not rows else rows[0]["census_positions"][2]
        )
        rows.append(
            {
                "offset": offset,
                "pitch": float(jnp.median(state["geometry"].pitch)),
                "position_tolerance": state["policy"].position_tolerance,
                "seeds": np.asarray(seeds),
                "census_valid": np.asarray(roots.valid).tolist(),
                "census_positions": np.asarray(roots.position).tolist(),
                "census_determinants": np.linalg.det(
                    np.asarray(roots.jet.hessian)
                ).tolist(),
                "solved_positions": np.asarray(solved.position).tolist(),
                "solved_valid": np.asarray(solved.valid).tolist(),
                "solved_status": np.asarray(solved.status).tolist(),
                "solved_iterations": np.asarray(solved.iterations).tolist(),
                "solved_gradient": np.linalg.norm(gradient, axis=-1).tolist(),
                "solved_correction": np.linalg.norm(correction, axis=-1).tolist(),
                "solved_hessian": hessian.tolist(),
                "solved_determinants": np.linalg.det(hessian).tolist(),
                "solved_contained": np.asarray(contained).tolist(),
                "reading_class": int(reading.boundary_class),
                "candidate_evidence": candidate_evidence(
                    jets.field(state["reference"]), target
                ),
            }
        )
    rows[0]["seeds"] = rows[0]["seeds"].tolist()
    rows[1]["seeds"] = rows[1]["seeds"].tolist()
    result = {
        "cells": len(state["geometry"].centre),
        "warm_count": warm_count,
        "endpoint_counts": endpoint_counts,
        "adjacent": bool(np.nextafter(left, right) == right),
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
