"""Instrument the prototype census per seed at the 282-cell saddle departure."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scripts.prototypes.fast_map_production import tie_probe
from nova.equilibrium import topology
from scripts.prototypes.read_robustness import census as census_module


def stages(field, geometry, convention, policy, previous):
    """Mirror scripts.prototypes.read_robustness.census, returning each mask."""
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
    offset = -jnp.linalg.solve(safe, coefficients[:, 1:3, None])[..., 0]
    seeds = geometry.centre + geometry.pitch[:, None] * offset
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
    seed_capacity = max(4 * policy.null_capacity, previous.shape[0])
    selected = jnp.nonzero(candidate, size=seed_capacity, fill_value=0)[0]
    live_seed = jnp.arange(seed_capacity) < jnp.sum(candidate)
    all_seeds = jnp.concatenate((previous, seeds[selected]))
    active = jnp.concatenate((jnp.ones(previous.shape[0], bool), live_seed))
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
        earlier & live[None] & (separation < 8 * policy.position_tolerance * jnp.min(geometry.pitch)),
        axis=1,
    )
    deduped = live & ~duplicate
    eigenvalues = jnp.linalg.eigvalsh(convention.sigma * polished.jet.hessian)
    axes = deduped & jnp.all(eigenvalues < 0, axis=1)
    saddles = deduped & (eigenvalues[:, 0] < 0) & (eigenvalues[:, 1] > 0)
    cap = policy.null_capacity - 1
    saddle_count = jnp.sum(saddles, dtype=jnp.int32)
    saddle_slots = jnp.nonzero(saddles, size=cap, fill_value=0)[0]
    distance = jnp.linalg.norm(polished.position - previous[0], axis=-1)
    axis_index = jnp.argmin(jnp.where(axes, distance, jnp.inf))
    return {
        "candidate_count": int(jnp.sum(candidate)),
        "seed_capacity": int(seed_capacity),
        "selected": np.asarray(selected).tolist(),
        "active": np.asarray(active).tolist(),
        "valid": np.asarray(polished.valid).tolist(),
        "reason": np.asarray(polished.reason).tolist(),
        "inside_wall": np.asarray(inside_wall).tolist(),
        "duplicate": np.asarray(duplicate).tolist(),
        "deduped": np.asarray(deduped).tolist(),
        "axes": np.asarray(axes).tolist(),
        "saddles": np.asarray(saddles).tolist(),
        "saddle_count": int(saddle_count),
        "saddle_slots": np.asarray(saddle_slots).tolist(),
        "axis_index": int(axis_index),
        "seeds": np.asarray(all_seeds).tolist(),
        "position": np.asarray(polished.position).tolist(),
        "eigenvalues": np.asarray(eigenvalues).tolist(),
        "det": np.linalg.det(np.asarray(polished.jet.hessian)).tolist(),
        "gradient_norm": np.linalg.norm(np.asarray(polished.jet.gradient), axis=-1).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    import nova

    print(f"nova={nova.__file__} cwd={Path.cwd().resolve()} backend={jax.default_backend()}", flush=True)
    state = tie_probe.build("diverted", 250, args.cache)
    sampled = tie_probe.make_sampler(state)
    prior, warm_count = tie_probe.warm(state, sampled, -0.2, 0.16, 0.01)
    geometry = state["geometry"]
    convention = state["convention"]
    policy = state["policy"]

    def census_count(offset):
        roots, *_ = sampled(jnp.asarray(offset * state["scale"]), prior)
        eigen = np.linalg.eigvalsh(
            np.asarray(convention.sigma) * np.asarray(roots.jet.hessian)
        )
        live = np.asarray(roots.valid) & (eigen[:, 0] < 0) & (eigen[:, 1] > 0)
        return int(np.asarray(roots.valid).sum()), int(live.sum())

    left, right = 0.17, 0.18
    counts = (census_count(left)[0], census_count(right)[0])
    if counts[0] <= counts[1]:
        raise RuntimeError(f"departure absent: {counts}")
    while np.nextafter(left, right) < right:
        middle = (left + right) / 2
        if middle in (left, right):
            break
        if census_count(middle)[0] == counts[0]:
            left = middle
        else:
            right = middle
    offsets = (left, right)

    def mirror_arrays(field, geometry, convention, policy, previous):
        """Same stages as the mirror, returning jax arrays only so it can jit."""
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
        seed_capacity = max(4 * policy.null_capacity, previous.shape[0])
        selected = jnp.nonzero(candidate, size=seed_capacity, fill_value=0)[0]
        live_seed = jnp.arange(seed_capacity) < jnp.sum(candidate)
        all_seeds = jnp.concatenate((previous, seeds[selected]))
        active = jnp.concatenate((jnp.ones(previous.shape[0], bool), live_seed))
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
        deduped = live & ~duplicate
        eigenvalues = jnp.linalg.eigvalsh(convention.sigma * polished.jet.hessian)
        axes = deduped & jnp.all(eigenvalues < 0, axis=1)
        saddles = deduped & (eigenvalues[:, 0] < 0) & (eigenvalues[:, 1] > 0)
        return (
            deduped,
            saddles,
            polished.position,
            eigenvalues,
            all_seeds,
            candidate,
            conditioned,
            inside,
            cross,
            seeds,
        )

    def jitted_sampled_mirror(off):
        varied = replace(
            state["jets"], exterior=state["jets"].exterior + off * state["jet_slope"]
        )
        field = varied.field(state["reference"])
        return mirror_arrays(field, geometry, convention, policy, prior)

    rows = []
    for offset in offsets:
        shifted = jnp.asarray(offset * state["scale"])
        from dataclasses import replace

        jets = replace(state["jets"], exterior=state["jets"].exterior + shifted * state["jet_slope"])
        field = jets.field(state["reference"])
        recorded = census_count(offset)
        detail = stages(field, geometry, convention, policy, prior)
        eager = census_module.census(field, geometry, convention, policy, prior)
        jax.block_until_ready(eager.roots.position)
        jitted = jax.jit(lambda f: census_module.census(f, geometry, convention, policy, prior))(field)
        jax.block_until_ready(jitted.roots.position)
        jm = jax.jit(jitted_sampled_mirror)(shifted)
        jax.block_until_ready(jm[0])
        jm_deduped, jm_saddles, jm_pos, jm_eig, jm_seeds, jm_cand, jm_cond, jm_inside, jm_cross, jm_cseed = jm
        JMCROSS = np.asarray(jm_cross)[263]
        JMSEED = np.asarray(jm_cseed)[263]
        print("  JMIRROR cell263 seed", [float(x) for x in JMSEED], flush=True)
        print("  JMIRROR cell263 cross", [float(x) for x in JMCROSS], flush=True)
        print("  JMIRROR deduped", np.asarray(jm_deduped)[:6].tolist(),
              "saddles", np.asarray(jm_saddles)[:6].tolist(), flush=True)
        print("  JMIRROR pos4", [float(x) for x in np.asarray(jm_pos)[4]],
              "seeds4", [float(x) for x in np.asarray(jm_seeds)[4]], flush=True)
        cand_np = np.asarray(jm_cand)
        cond_np = np.asarray(jm_cond)
        inside_np = np.asarray(jm_inside)
        print("  JMIRROR candidate_count", int(cand_np.sum()),
              "cells", np.flatnonzero(cand_np).tolist(), flush=True)
        for index in np.flatnonzero(cond_np & ~inside_np)[:8]:
            print("    cell", int(index), "conditioned", True, "inside", False,
                  flush=True)
        rows.append(
            {
                "offset": offset,
                "recorded": recorded,
                "stages": detail,
                "eager_valid": np.asarray(eager.roots.valid).tolist(),
                "eager_positions": np.asarray(eager.roots.position).tolist(),
                "eager_saddle_count": int(eager.saddles),
                "eager_candidate": int(eager.candidates),
                "jfield_valid": np.asarray(jitted.roots.valid).tolist(),
                "jfield_saddle_count": int(jitted.saddles),
            }
        )
    result = {
        "cells": len(geometry.centre),
        "warm_count": warm_count,
        "adjacent": bool(np.nextafter(left, right) == right),
        "null_capacity": int(policy.null_capacity),
        "hessian_tolerance": float(policy.hessian_tolerance),
        "position_tolerance": float(policy.position_tolerance),
        "min_pitch": float(jnp.min(geometry.pitch)),
        "median_pitch": float(jnp.median(geometry.pitch)),
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps({k: result[k] for k in ("cells", "adjacent")}), flush=True)
    for row in rows:
        print("OFFSET", repr(row["offset"]), "recorded(valid,saddles)", row["recorded"])
        for key in ("active", "valid", "inside_wall", "duplicate", "deduped", "saddles"):
            print("   ", key, row["stages"][key])
        print("    det", [f"{d:.4e}" for d in row["stages"]["det"]])
        print("    eig", [[f"{v:.4e}" for v in pair] for pair in row["stages"]["eigenvalues"]])
        print("    eager_valid", row["eager_valid"])
        print("    eager_saddle_count", row["eager_saddle_count"], "candidates", row["eager_candidate"])
        print("    jfield_valid", row["jfield_valid"])
        print("    jfield_saddle_count", row["jfield_saddle_count"])
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()