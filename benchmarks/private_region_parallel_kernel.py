#!/usr/bin/env python3
"""Measure four formulations of the private-region cell mask.

On the analytic single-null flux the private region is the set of in-material
cells on the plasma side of the separatrix that the magnetic-axis connected
component cannot reach (``closed & ~connected & inside_material`` in the
production read).  This driver measures how the mask is built four ways:

  a  the production hex flood as executed: the saddle-aware hex minimum-label
     propagation that the read's ``axis_component`` runs.
  b  the raster-doubling ``flood_fill_core`` applied to the cell field
     embedded in a (height, radius) hex-lattice raster, pass count set from
     the mesh diameter (``nr + nz``).
  c  pointer jumping over the confined-cell adjacency seeded at the axis cell:
     ``ceil(log2(cells))`` fully vectorised pass-doubling rounds, no per-cell
     loop.
  d  the saddle-wedge level test: a cell is private when its flux lies on the
     open side of the admitted X-point level and its centroid lies on the
     private side of both separatrix-leg rays given by the saddle Hessian
     eigenvectors at the X-point.  O(cells), zero passes.

Membership identity against the production mask is the positive control per
rung: the differing-cell count is reported for every formulation.  Each mask
is timed as a jitted function vmapped over a batch of states, so the per-state
cost sits beside the production read's measured totals.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp

from benchmarks import limiter_read_resolution_audit as limiter_audit
from benchmarks import solovev_certificate as certificate
from nova.equilibrium import flux_surface_connectivity as fsc
import nova.equilibrium.connectivity_boundary as cbound
from nova.equilibrium.domain import PlasmaDomain
from nova.jax.config import configure_dtypes

configure_dtypes()

WALL_NODE_COUNT = 121
TIMING_BATCH = 16
TIMING_REPEATS = 5
DIVERTED = certificate.DIVERTED_CASE_NAME


@dataclass
class _RungBundle:
    requested: int
    realised: int
    operator: object
    topo: object
    psi_grid: np.ndarray
    production_private: np.ndarray
    boundary_flux: float
    axis_flux: float
    axis: np.ndarray
    x_point: np.ndarray
    x_point_flux: float
    inside: np.ndarray
    rings: np.ndarray
    link_admissible: np.ndarray
    seed: np.ndarray
    n_iter: int
    n_pass: int
    raster: dict
    raster_collisions: int
    legs: dict
    pass_counts: dict = field(default_factory=dict)


def _machine_field_and_state(requested_cells: int):
    """Return the machine, operator, flux and single production read state."""
    carrier, source, exact = certificate._case(DIVERTED)
    machine = limiter_audit._machine(
        DIVERTED, carrier, exact, -requested_cells, WALL_NODE_COUNT
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = limiter_audit._exact_flux(DIVERTED, exact, coordinates)
    operator = limiter_audit.oracle_fixture.forward_operator(source, machine)
    physical = np.asarray(analytic[: operator.physical_node_number])
    read = operator._fixed_design_read(jnp.asarray(physical, dtype=jnp.float64))
    masks, state, connected, _admitted = jax.block_until_ready(read)
    return machine, operator, physical, exact, state, connected, masks


def _axis_seed(confined: np.ndarray, coordinate: np.ndarray, axis: np.ndarray):
    distance2 = np.sum((coordinate - axis) ** 2, axis=-1)
    flat = int(np.argmin(np.where(confined.reshape(-1), distance2.reshape(-1), np.inf)))
    seed = np.zeros(confined.shape, dtype=bool).reshape(-1)
    seed[flat] = True
    return seed.reshape(confined.shape)


def _cell_raster_map(coordinate: np.ndarray, rings: np.ndarray) -> dict:
    """Place every cell on the (radius, height) raster at the mesh pitch.

    The unstructured plasma mesh is not a global affine hex lattice: every
    cell has six exact-pitch, sixty-degree neighbours, yet the point set
    drifts from any single (i, j) basis (residuals grow with hop distance
    from any chosen origin).  A structured raster doubling flood therefore
    cannot tile the mesh exactly.  The surrogate raster a 2-D
    ``flood_fill_core`` operates on is the bounding (R, Z) grid at the mesh
    pitch; each cell lands on the nearest site by rounding, and site
    collisions the non-lattice drift produces are counted so the
    approximation is visible in the differing-cell report.
    """
    pitches = np.linalg.norm(
        coordinate[rings[:, 1:]] - coordinate[rings[:, 0]][:, None], axis=-1
    )
    pitch = float(np.min(pitches[pitches > 1e-9]))
    r, z = coordinate[:, 0], coordinate[:, 1]
    r_min, z_min = float(r.min()), float(z.min())
    i = np.rint((r - r_min) / pitch).astype(np.int64)
    j = np.rint((z - z_min) / pitch).astype(np.int64)
    n_i = int(np.ceil((r.max() - r_min) / pitch)) + 2
    n_j = int(np.ceil((z.max() - z_min) / pitch)) + 2
    slot = np.ravel_multi_index((i, j), (n_i, n_j))
    _, counts = np.unique(slot, return_counts=True)
    return {
        "i": i,
        "j": j,
        "n_i": n_i,
        "n_j": n_j,
        "slot": slot,
        "pitch": pitch,
        "collisions": int(np.count_nonzero(counts > 1)),
    }


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _saddle_wedge(
    case_name: str,
    exact,
    coordinate: np.ndarray,
    x_point: np.ndarray,
    x_point_flux: float,
) -> dict:
    """Return oriented separatrix-leg rays and the private-wedge probe signs.

    The 2x2 Hessian at the admitted X-point is finite-differenced on the
    analytic poloidal flux; its eigenvectors are the separatrix-leg
    directions.  Each leg is oriented toward the private well (negative Z
    beat), and the wedge signs are read at a probe one millimetre below the
    X-point, strictly inside the wedge.
    """
    h = 1.0e-4

    def flux_at(r: float, z: float) -> float:
        return float(
            limiter_audit._exact_flux(
                case_name, exact, np.array([[r, z]], dtype=np.float64)
            )[0]
        )

    r0, z0 = float(x_point[0]), float(x_point[1])
    psi_00 = flux_at(r0, z0)
    hrr = (flux_at(r0 + h, z0) - 2.0 * psi_00 + flux_at(r0 - h, z0)) / h**2
    hzz = (flux_at(r0, z0 + h) - 2.0 * psi_00 + flux_at(r0, z0 - h)) / h**2
    hrz = (
        flux_at(r0 + h, z0 + h)
        - flux_at(r0 + h, z0 - h)
        - flux_at(r0 - h, z0 + h)
        + flux_at(r0 - h, z0 - h)
    ) / (4.0 * h**2)
    _hessian, eigenvectors = np.linalg.eigh(np.array([[hrr, hrz], [hrz, hzz]]))
    legs = np.stack(
        [
            direction if direction[1] < 0.0 else -direction
            for direction in eigenvectors.T
        ]
    )
    probe = np.array([r0, z0 - 1.0e-3])
    return {
        "legs": legs,
        "sides": np.asarray([_cross2(leg, probe - x_point) for leg in legs]),
        "x_point": x_point,
        "x_point_flux": x_point_flux,
    }


def _build_rung(requested_cells: int) -> _RungBundle:
    machine, operator, physical, exact, state, connected, masks = (
        _machine_field_and_state(requested_cells)
    )
    topo = operator._fixed_design_topology
    polarity = int(operator.polarity)
    psi_grid = np.asarray(topo.split_flux_map(jnp.asarray(physical))[0])
    inside = np.asarray(operator.inside_material, dtype=bool)
    boundary_flux = float(np.asarray(state.boundary_flux))
    axis_flux = float(np.asarray(state.axis_flux))
    axis = np.asarray(state.axis, dtype=np.float64)
    x_point = np.asarray(state.x_point, dtype=np.float64)
    x_point_flux = float(np.asarray(state.x_point_flux))
    closed = np.asarray(
        topo.psi_mask(polarity, jnp.asarray(psi_grid), jnp.asarray(boundary_flux)),
        dtype=bool,
    )
    confined = closed & inside
    rings = np.asarray(topo.connectivity_rings)
    shared_edges = np.asarray(topo.connectivity_shared_edges)
    coordinate = np.asarray(topo.connectivity_coordinate, dtype=np.float64)
    edge_gather = np.asarray(topo.connectivity_edge_gather)
    edge_weight = np.asarray(topo.connectivity_edge_weight, dtype=np.float64)
    edge_values = np.sum(edge_weight * psi_grid[edge_gather], axis=-1)

    direction = 1.0 if polarity >= 0 else -1.0
    inward_offset = cbound._PRE_SADDLE_OFFSET_FRACTION * (
        np.nanmax(np.where(confined, psi_grid, np.nan))
        - np.nanmin(np.where(confined, psi_grid, np.nan))
    )
    component_flux = boundary_flux + direction * inward_offset  # diverted read
    exact_link = np.asarray(
        fsc.hex_edge_admissibility(
            jnp.asarray(psi_grid),
            jnp.asarray(coordinate[:, 0]),
            jnp.asarray(coordinate[:, 1]),
            jnp.asarray(boundary_flux),
            jnp.asarray(axis_flux),
            jnp.asarray(shared_edges),
            edge_values=jnp.asarray(edge_values),
        )
    )
    inward_link = np.asarray(
        fsc.hex_edge_admissibility(
            jnp.asarray(psi_grid),
            jnp.asarray(coordinate[:, 0]),
            jnp.asarray(coordinate[:, 1]),
            jnp.asarray(component_flux),
            jnp.asarray(axis_flux),
            jnp.asarray(shared_edges),
            edge_values=jnp.asarray(edge_values),
        )
    )
    missing = np.zeros(rings.shape, dtype=bool)
    missing[:, 1:] = rings[:, 1:] == rings[:, :1]
    exact_link = exact_link & ~missing
    inward_link = inward_link & ~missing
    centre = coordinate[rings[:, 0]]
    edge_pitch = np.linalg.norm(coordinate[rings] - centre[:, None], axis=-1)
    edge_midpoint = np.mean(shared_edges, axis=-2)
    saddle_distance = np.linalg.norm(edge_midpoint - x_point, axis=-1)
    saddle_cut = bool(state.diverted)
    saddle_neighbourhood = saddle_cut & (saddle_distance <= 3.0 * edge_pitch)
    link_admissible = exact_link & (inward_link | ~saddle_neighbourhood)
    link_admissible = np.asarray(
        cbound._canonicalize_reciprocal_hex_edges(
            jnp.asarray(rings), jnp.asarray(link_admissible)
        )
    )
    seed = _axis_seed(confined, coordinate, axis)

    labels, steps = fsc.label_saddle_aware_hex_connected_components_with_steps(
        jnp.asarray(confined),
        jnp.asarray(rings),
        jnp.asarray(link_admissible),
        confined.size,
    )
    rebuilt = (np.asarray(labels) > 0) & ~np.asarray(
        fsc.private_flux_mask(jnp.asarray(labels), jnp.asarray(seed))
    )
    if not np.array_equal(rebuilt, np.asarray(connected, dtype=bool)):
        raise RuntimeError("reconstructed axis component disagrees with the read")

    production_private = np.asarray(masks.label == int(PlasmaDomain.PRIVATE_FLUX))
    raster = _cell_raster_map(coordinate, rings)
    legs = _saddle_wedge(DIVERTED, exact, coordinate, x_point, x_point_flux)
    return _RungBundle(
        requested=requested_cells,
        realised=len(machine.node),
        operator=operator,
        topo=topo,
        psi_grid=psi_grid,
        production_private=production_private,
        boundary_flux=boundary_flux,
        axis_flux=axis_flux,
        axis=axis,
        x_point=x_point,
        x_point_flux=x_point_flux,
        inside=inside,
        rings=rings,
        link_admissible=link_admissible,
        seed=seed,
        n_iter=confined.size,
        n_pass=int(np.ceil(np.log2(max(confined.size, 2)))),
        raster=raster,
        raster_collisions=int(raster["collisions"]),
        legs=legs,
        pass_counts={
            "production_hex": int(steps),
            "raster": int(raster["n_i"] + raster["n_j"]),
        },
    )


def _closed(bundle: _RungBundle, psi_grid: jax.Array) -> jax.Array:
    return bundle.topo.psi_mask(
        int(bundle.operator.polarity),
        psi_grid,
        jnp.asarray(bundle.boundary_flux),
    )


def _mask_a(bundle: _RungBundle, psi_grid: jax.Array) -> jax.Array:
    """(a) the hex flood the production read executes."""
    closed = _closed(bundle, psi_grid)
    confined = closed & jnp.asarray(bundle.inside)
    labels = fsc.label_saddle_aware_hex_connected_components(
        confined,
        jnp.asarray(bundle.rings),
        jnp.asarray(bundle.link_admissible),
        bundle.n_iter,
    )
    connectivity = (labels > 0) & ~fsc.private_flux_mask(
        labels, jnp.asarray(bundle.seed)
    )
    return confined & ~connectivity


def _mask_b(bundle: _RungBundle, psi_grid: jax.Array) -> jax.Array:
    """(b) raster-doubling flood on the pitch raster, mesh-diameter passes."""
    closed = _closed(bundle, psi_grid)
    confined = closed & jnp.asarray(bundle.inside)
    slot = jnp.asarray(bundle.raster["slot"])
    n_i, n_j = bundle.raster["n_i"], bundle.raster["n_j"]
    confined_flat = jnp.zeros(n_i * n_j, dtype=bool).at[slot].max(confined)
    seed_flat = jnp.zeros(n_i * n_j, dtype=bool).at[slot].max(jnp.asarray(bundle.seed))
    core = fsc.flood_fill_core(
        confined_flat.reshape(n_i, n_j), seed_flat.reshape(n_i, n_j), n_i + n_j
    )
    reached = jnp.asarray(core > 0, dtype=bool).reshape(-1)[slot]
    return confined & ~reached


def _mask_c(bundle: _RungBundle, psi_grid: jax.Array) -> jax.Array:
    """(c) pointer jumping with log2(cells) pass-doubling rounds.

    Every round hooks each ring onto the minimum root of its open slots and
    then shortcuts ``root[root[x]]``; the re-hook reads the shortcut roots so
    a component minimum advances two hops per round and ``log2(cells)``
    rounds collapse every component onto its minimum label, with no per-cell
    loop.
    """
    closed = _closed(bundle, psi_grid)
    confined = closed & jnp.asarray(bundle.inside)
    rings = jnp.asarray(bundle.rings)
    link_admissible = jnp.asarray(bundle.link_admissible)
    seed = jnp.asarray(bundle.seed)
    n_nodes = confined.size
    sentinel = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    ids = jnp.arange(n_nodes, dtype=jnp.int32) + 1
    centre_open = confined[rings[:, 0]]
    open_slot = jnp.concatenate(
        (
            centre_open[:, None],
            link_admissible[:, 1:] & centre_open[:, None] & confined[rings[:, 1:]],
        ),
        axis=1,
    )

    def round(_iteration, root):
        reach = jnp.where(open_slot, root[rings], sentinel)
        ring_minimum = jnp.min(reach, axis=1, keepdims=True)
        contributed = jnp.where(open_slot, ring_minimum, sentinel)
        hooked = root.at[rings].min(contributed)
        hooked = jnp.where(confined, hooked, 0)
        padded = jnp.concatenate((jnp.zeros(1, dtype=jnp.int32), hooked))
        return jnp.where(confined, padded[hooked], 0)

    root = jax.lax.fori_loop(0, bundle.n_pass, round, jnp.where(confined, ids, 0))
    axis_parent = jnp.min(jnp.where(seed & (root > 0), root, sentinel))
    return confined & (root != axis_parent)


def _mask_d(bundle: _RungBundle, psi_grid: jax.Array) -> jax.Array:
    """(d) saddle-wedge level test, O(cells) with no passes."""
    closed = _closed(bundle, psi_grid)
    inside = jnp.asarray(bundle.inside)
    legs = jnp.asarray(bundle.legs["legs"])
    sides = jnp.asarray(bundle.legs["sides"])
    x_point = jnp.asarray(bundle.legs["x_point"])
    offset = jnp.asarray(bundle.topo.connectivity_coordinate) - x_point
    cross = offset[:, 1][:, None] * legs[:, 0] - offset[:, 0][:, None] * legs[:, 1]
    side_sign = jnp.where(sides > 0.0, 1.0, -1.0)
    wedge = jnp.all(side_sign * cross >= 0.0, axis=1)
    polarity = int(bundle.operator.polarity)
    open_side = jnp.where(
        polarity >= 0,
        psi_grid >= jnp.asarray(bundle.legs["x_point_flux"]),
        psi_grid <= jnp.asarray(bundle.legs["x_point_flux"]),
    )
    return closed & inside & open_side & wedge


_FORMULATIONS = {
    "a": _mask_a,
    "b": _mask_b,
    "c": _mask_c,
    "d": _mask_d,
}


def _measure_timing(single: callable, batch: jax.Array) -> float:
    """Median per-state milliseconds of a jitted batch-vmapped mask function."""
    jitted = jax.jit(jax.vmap(single))
    jitted(batch)
    jax.block_until_ready(jitted(batch))
    timings = []
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        jax.block_until_ready(jitted(batch))
        timings.append((time.perf_counter() - start) / batch.shape[0] * 1e3)
    return float(np.median(timings))


def _run_rung(bundle: _RungBundle, timing: bool) -> dict:
    row: dict[str, object] = {
        "requested": bundle.requested,
        "realised": bundle.realised,
        "production_private": int(np.count_nonzero(bundle.production_private)),
        "raster_collisions": bundle.raster_collisions,
        "pass_counts": bundle.pass_counts,
    }
    batch = jnp.broadcast_to(
        jnp.asarray(bundle.psi_grid), (TIMING_BATCH,) + bundle.psi_grid.shape
    )
    for key, func in _FORMULATIONS.items():
        single = lambda state: func(bundle, state)  # noqa: E731
        mask = np.asarray(jax.jit(single)(jnp.asarray(bundle.psi_grid)))
        row.setdefault("private", {})[key] = int(np.count_nonzero(mask))
        differing = int(np.count_nonzero(mask != bundle.production_private))
        row.setdefault("differing_cells", {})[key] = differing
        print(f"  {key}: private {row['private'][key]} differing {differing}")
        if timing:
            milliseconds = _measure_timing(single, batch)
            row.setdefault("per_state_ms", {})[key] = milliseconds
    if timing:
        print(
            "  per-state ms:",
            " ".join(
                f"{key}={row['per_state_ms'][key]:.4f}"
                for key in sorted(row["per_state_ms"])
            ),
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, nargs="+", default=[500, 750, 1000, 2500])
    parser.add_argument(
        "--identity-only",
        action="store_true",
        help="skip the timing loop (login-node smoke)",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    results: dict[str, object] = {"rungs": []}
    for requested in args.cells:
        bundle = _build_rung(requested)
        print(f"\n== requested {requested} (realised {bundle.realised} cells) ==")
        if args.identity_only:
            bundle.pass_counts["pointer_jump"] = bundle.n_pass
            results["rungs"].append(_run_rung(bundle, timing=False))
        else:
            run = _run_rung(bundle, timing=True)
            run["per_state_pass_counts"] = {
                "a": bundle.pass_counts["production_hex"],
                "b": bundle.pass_counts["raster"],
                "c": bundle.n_pass,
                "d": 0,
            }
            results["rungs"].append(run)
        if args.output:
            with open(args.output, "w") as handle:
                json.dump(results, handle, indent=2)
            print("wrote", args.output)


if __name__ == "__main__":
    main()
