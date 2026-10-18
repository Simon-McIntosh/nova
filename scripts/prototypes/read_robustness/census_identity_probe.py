"""Stage-by-stage receipt for a root that leaves the census without folding.

At each state the probe prints the census masks in order, so a root that stops
being counted can be attributed to the stage that refused it: the cell never
proposed it, its Hessian was rejected as unreadable, its polish failed, it fell
outside the wall, it duplicated an earlier root, or the capacity cut it. The
field is built inside the compiled trace, as the census sampler builds it.
"""

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

from nova.equilibrium import topology
from scripts.prototypes.fast_map_production import tie_probe


def stages(field, geometry, convention, policy, previous):
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
    gradient = coefficients[:, 1:3]
    offset = -jnp.linalg.solve(safe, gradient[..., None])[..., 0]
    seeds = geometry.centre + geometry.pitch[:, None] * offset
    slot = jnp.arange(geometry.vertices.shape[1])
    following = jnp.where(
        slot[None] + 1 < geometry.vertex_count[:, None], slot[None] + 1, 0
    )
    end = jnp.take_along_axis(geometry.vertices, following[..., None], axis=1)
    edge = end - geometry.vertices
    relative = seeds[:, None] - geometry.vertices
    cross = topology._cross_plane(edge, relative)
    epsilon = jnp.finfo(seeds.dtype).eps
    inverse = jnp.linalg.inv(safe)
    hessian_scale = jnp.sum(jnp.abs(hessian), axis=(-2, -1))
    displacement_scale = jnp.sum(jnp.abs(inverse), axis=(-2, -1)) * (
        hessian_scale * jnp.sum(jnp.abs(offset), axis=-1)
        + jnp.sum(jnp.abs(gradient), axis=-1)
    )
    seed_uncertainty = epsilon * jnp.abs(geometry.pitch) * displacement_scale
    rounding = (
        4.0
        * epsilon
        * (jnp.abs(edge[..., 0]) + jnp.abs(edge[..., 1]))
        * (jnp.abs(relative[..., 0]) + jnp.abs(relative[..., 1]))
        + (jnp.abs(edge[..., 0]) + jnp.abs(edge[..., 1])) * seed_uncertainty[:, None]
    )
    inside = jnp.all(
        jnp.where(
            slot[None] < geometry.vertex_count[:, None], cross >= -rounding, True
        ),
        axis=1,
    )
    in_domain = jax.vmap(
        lambda point: jnp.any(topology._inside_cells(geometry, point))
    )(seeds)
    candidate = conditioned & inside & jnp.all(jnp.isfinite(seeds), axis=1)
    domain_candidate = conditioned & in_domain & jnp.all(jnp.isfinite(seeds), axis=1)
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
    return {
        "conditioned": conditioned,
        "in_domain": in_domain,
        "domain_candidate": domain_candidate,
        "inside": inside,
        "candidate": candidate,
        "active": active,
        "valid": polished.valid,
        "reason": polished.reason,
        "inside_wall": inside_wall,
        "duplicate": duplicate,
        "deduped": deduped,
        "seeds": all_seeds,
        "position": polished.position,
        "eigenvalues": eigenvalues,
        "determinant": jnp.linalg.det(polished.jet.hessian),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cells", type=int, default=250)
    parser.add_argument("--low", type=float, default=0.170)
    parser.add_argument("--high", type=float, default=0.174)
    parser.add_argument("--count", type=int, default=21)
    parser.add_argument("--warm-stop", type=float, default=0.16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True

    state = tie_probe.build("diverted", args.cells, args.cache)
    sampled = tie_probe.make_sampler(state)
    prior, warm_count = tie_probe.warm(state, sampled, -0.2, args.warm_stop, 0.01)
    geometry = state["geometry"]
    convention = state["convention"]
    policy = state["policy"]

    def traced(offset):
        varied = replace(
            state["jets"],
            exterior=state["jets"].exterior + offset * state["jet_slope"],
        )
        field = varied.field(state["reference"])
        return stages(field, geometry, convention, policy, prior)

    rows = []
    for value in np.linspace(args.low, args.high, args.count):
        traced_detail = jax.jit(traced)(jnp.asarray(value * state["scale"]))
        detail = {key: np.asarray(leaf) for key, leaf in traced_detail.items()}
        kept = np.flatnonzero(detail["deduped"])
        rows.append(
            {
                "offset": float(value),
                "candidate_cells": np.flatnonzero(detail["candidate"]).tolist(),
                "conditioned_fail": np.flatnonzero(
                    ~detail["conditioned"] & detail["inside"]
                ).tolist(),
                "kept": kept.tolist(),
                "conditioned_count": int(detail["conditioned"].sum()),
                "domain_candidate_count": int(detail["domain_candidate"].sum()),
                "positions": detail["position"][kept].tolist(),
                "determinants": detail["determinant"][kept].tolist(),
                "eigenvalues": detail["eigenvalues"][kept].tolist(),
                "seeds": detail["seeds"][kept].tolist(),
                "reasons": detail["reason"].tolist(),
                "inside_wall": detail["inside_wall"].tolist(),
                "valid": detail["valid"].tolist(),
                "duplicate": detail["duplicate"].tolist(),
            }
        )
    Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
    print(f"cells {len(geometry.centre)} warm {warm_count}", flush=True)
    for row in rows:
        print(
            f"offset {row['offset']:.6g} candidates {row['candidate_cells']} "
            f"kept {row['kept']} conditioned {row['conditioned_count']} "
            f"domain_candidates {row['domain_candidate_count']}",
            flush=True,
        )
        for root in row["positions"]:
            print(f"    pos {[round(v, 6) for v in root]}", flush=True)
        print(f"    det {[f'{d:.4e}' for d in row['determinants']]}", flush=True)
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()