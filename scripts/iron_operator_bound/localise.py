"""Localise the affine interaction's energy excess on a small graded mesh.

The volume-weighted Galerkin interaction of an affine magnetisation must
satisfy ``x.T E x <= mu0 * (x.T H x)``.  This script splits the assembled
energy into its three contributing blocks (affine edge-pair flux, affine edge
against uniform volume, uniform volume against volume) and measures the
metric-weighted quotient of every block combination, so the block that
carries the excess is separated from the ones that do not.  Every block is
re-integrated over a range of quadrature orders, so a quadrature cause is
separated from a basis or weighting cause.

    python scripts/iron_operator_bound/localise.py --output <run-dir>

The mesh is the small octagonal shell of the linear spectral gate: the
uniform 40 mm mesh sits just below the bound and a 16/80 mm grading crosses it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import shapely

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402
from nova.biot import tiledassembly  # noqa: E402
from nova.biot.tiledassembly import (  # noqa: E402
    _linear_edge_mutual,
    _polygon_current_geometry,
    _polygon_current_response,
    linear_galerkin_cell_interaction,
    linear_magnetisation_geometry,
)

assert jax.config.jax_enable_x64


def context(polygons):
    """Return the geometry and packed volume sources every block shares."""
    return dict(
        geometry=linear_magnetisation_geometry(polygons),
        packed=_polygon_current_geometry(polygons),
    )


def edge_block(polygons, shared, plan, order):
    """Return the affine edge-pair flux energy at one quadrature order."""
    edge = shared["geometry"]["edge_coefficient"]
    mutual = _linear_edge_mutual(shared["geometry"]["atoms"], plan, order)
    return np.asarray((edge @ mutual) @ edge.T)


def cross_block(polygons, shared, plan, boundary_order):
    """Return the affine-edge against uniform-volume energy moments."""
    geometry = shared["geometry"]
    atoms = geometry["atoms"]
    edge = geometry["edge_coefficient"]
    volume = geometry["volume_coefficient"]
    nodes, weights = np.polynomial.legendre.leggauss(boundary_order)
    parameter, weights = (nodes + 1) / 2, weights / 2
    edge_volume = np.empty((2 * len(atoms), len(polygons)))
    for begin in range(0, len(atoms), 256):
        selected = atoms[begin : begin + 256]
        points = selected[:, :1] + parameter[None, :, None] * (
            selected[:, 1:] - selected[:, :1]
        )
        flux = _polygon_current_response(
            polygons,
            points.reshape(-1, 2),
            plan,
            packed=shared["packed"],
            flux_only=True,
        )[0]
        flux = flux.reshape(len(selected), boundary_order, len(polygons))
        moments = np.einsum(
            "eqc,q,qp,e->epc",
            flux,
            weights,
            np.column_stack((np.ones(boundary_order), parameter)),
            np.linalg.norm(selected[:, 1] - selected[:, 0], axis=1),
        )
        edge_volume[2 * begin : 2 * (begin + len(selected))] = moments.reshape(
            -1, len(polygons)
        )
    return np.asarray((edge @ edge_volume) @ volume.T)


def volume_block(polygons, shared, plan, volume_order):
    """Return the uniform volume-current energy at one section rule.

    The block is the owner's: :func:`nova.biot.tiledassembly.volume_mutual`
    integrates the section rule this study asks for, and this wrapper only
    weights it by the volume coefficient of the energy.
    """
    volume = shared["geometry"]["volume_coefficient"]
    mutual = tiledassembly.volume_mutual(
        polygons, plan, packed=shared["packed"], section_order=volume_order
    )
    return np.asarray((volume @ mutual) @ volume.T)


def quotient(gram, energy):
    """Return the extreme metric-weighted quotients and the top witness.

    The solve is the owner's: :func:`nova.biot.magnetisation.galerkin_block_spectrum`
    applies the block metric and the ``1/mu0`` normalisation, so this study and
    the assembler's spectrum cannot drift apart.  The block energies here are
    already in the metric's scale, so the call passes ``scale=False``; the
    metric it returns is the one the witness is weighed with below.
    """
    return iron.galerkin_block_spectrum(gram, energy, witness=True, scale=False)


def pair_matrix(energy, witness, blocks):
    """Return ``x_i.T E_ij x_j`` for every ordered block pair."""
    shaped = energy.reshape(blocks, 3, blocks, 3)
    vector = witness.reshape(blocks, 3)
    partial = (shaped * vector[None, None, :, :]).sum(axis=3)
    return (partial * vector[:, :, None]).sum(axis=1)


def localise(polygons, blocks, witness):
    """Return the top self cells and cell pairs of one witness."""
    count = len(polygons)
    total = np.zeros((6 * count, 6 * count))
    for block in blocks.values():
        total = total + block
    pairs = pair_matrix(total, witness, 2 * count)
    pairs = (pairs + pairs.T) / 2
    pairs = pairs.reshape(2, count, 2, count).sum(axis=(0, 2))
    diagonal = np.diag(pairs)
    bodies = [shapely.Polygon(p) for p in polygons]
    upper = np.triu(pairs, 1)
    rows = []
    for flat in np.argsort(-np.abs(upper).ravel())[:12]:
        first, second = np.unravel_index(flat, upper.shape)
        if second <= first:
            continue
        centres = np.asarray(
            [bodies[first].centroid.coords[0], bodies[second].centroid.coords[0]]
        )
        rows.append(
            dict(
                first=int(first),
                second=int(second),
                contribution=float(pairs[first, second] / iron._MU0),
                spacing_m=float(np.linalg.norm(centres[0] - centres[1])),
                areas_m2=[float(bodies[first].area), float(bodies[second].area)],
            )
        )
    top_cells = [
        dict(
            cell=int(cell),
            self_contribution=float(diagonal[cell] / iron._MU0),
            area_m2=float(bodies[cell].area),
            bounds_m=[float(v) for v in bodies[cell].bounds],
        )
        for cell in np.argsort(-np.abs(diagonal))[:8]
    ]
    return dict(pairs=rows, cells=top_cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fine", type=float, default=0.016)
    parser.add_argument("--coarse", type=float, default=0.08)
    parser.add_argument("--edge-orders", type=int, nargs="+", default=[24])
    parser.add_argument("--boundary-orders", type=int, nargs="+", default=[12])
    parser.add_argument("--volume-orders", type=int, nargs="+", default=[6])
    args = parser.parse_args()
    plan = iron._PLAN
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    field = iron.graded_cell_size(
        [0.28, 0.4], [-0.4, 0.4], fine=args.fine, coarse=args.coarse, cells_across=20
    )
    cells = iron._mesh(iron.gate_shell(), field)
    polygons = cells.polygons
    print(f"mesh {len(polygons)} cells", flush=True)
    shared = context(polygons)
    gram = shared["geometry"]["gram"]
    stamp = time.perf_counter()
    edges = {}
    for order in args.edge_orders:
        edges[order] = edge_block(polygons, shared, plan, order)
        print(f"block edge-{order} {time.perf_counter() - stamp:.1f}s", flush=True)
        stamp = time.perf_counter()
    crosses = {}
    for order in args.boundary_orders:
        crosses[order] = cross_block(polygons, shared, plan, order)
        print(f"block cross-{order} {time.perf_counter() - stamp:.1f}s", flush=True)
        stamp = time.perf_counter()
    volumes = {}
    for order in args.volume_orders:
        volumes[order] = volume_block(polygons, shared, plan, order)
        print(f"block volume-{order} {time.perf_counter() - stamp:.1f}s", flush=True)
        stamp = time.perf_counter()
    default_edge = edges[args.edge_orders[0]]
    default_cross = crosses[args.boundary_orders[0]]
    default_volume = volumes[args.volume_orders[0]]
    owner, blocks = linear_galerkin_cell_interaction(polygons, plan=plan)
    mirrored = default_edge + default_cross + default_cross.T + default_volume
    metric = iron.galerkin_block_spectrum(gram, mirrored)["metric"]
    record = dict(
        cells=len(polygons),
        owner_mirror_defect=float(
            np.linalg.norm(metric @ np.asarray(owner) - (mirrored + mirrored.T) / 2)
            / np.linalg.norm(mirrored)
        ),
        arms={},
    )
    for key_e, edge in edges.items():
        for key_c, cross in crosses.items():
            for key_v, volume in volumes.items():
                energy = edge + cross + cross.T + volume
                measured = quotient(gram, energy)
                label = f"edge-{key_e}-boundary-{key_c}-volume-{key_v}"
                record["arms"][label] = dict(
                    maximum=measured["maximum"], minimum=measured["minimum"]
                )
                print(
                    f"{label} {measured['maximum']:.12f} {measured['minimum']:.3e}",
                    flush=True,
                )
                if (
                    key_e == args.edge_orders[0]
                    and key_c == args.boundary_orders[0]
                    and key_v == args.volume_orders[0]
                ):
                    witness = measured["witness"]
                    parts = {
                        "edge": float(witness @ edge @ witness),
                        "cross": float(witness @ (cross + cross.T) @ witness),
                        "volume": float(witness @ volume @ witness),
                        "gram": float(witness @ metric @ witness),
                    }
                    record["witness-blocks"] = {
                        key: value / iron._MU0 for key, value in parts.items()
                    }
                    record["localisation"] = localise(
                        polygons,
                        dict(edge=edge, cross=cross + cross.T, volume=volume),
                        witness,
                    )
                    print(
                        "witness blocks " + json.dumps(record["witness-blocks"]),
                        flush=True,
                    )
                (output / "localise.json").write_text(
                    json.dumps(record, indent=1) + "\n"
                )
    (output / "localise.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
