"""Test whether a local saddle frame separates cell-graph private support."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import nova

from nova.equilibrium import topology
from scripts.prototypes.sub_millisecond import measure_wall_march as wall_march


def frame_connections(geometry, reading, sigma):
    matrix = wall_march.edge_connections(geometry, reading, sigma)
    cells, edges = geometry.neighbour.shape
    slot = jnp.arange(edges)
    following = jnp.where(
        slot[None] + 1 < geometry.vertex_count[:, None], slot[None] + 1, 0
    )
    end = jnp.take_along_axis(geometry.vertices, following[..., None], axis=1)
    middle = (geometry.vertices + end) / 2
    sector = topology._normal_form_sectors(middle, reading.saddle_form)
    axis_sector = jnp.argmax(
        topology._normal_form_sectors(reading.axis[None], reading.saddle_form)[0]
    )
    core_edge = sector[..., axis_sector]
    peer = jnp.maximum(geometry.neighbour, 0)
    peer_edge = jnp.maximum(geometry.neighbour_edge, 0)
    blocked = (
        (reading.normal_form_cells[:, None] & ~core_edge)
        | (reading.normal_form_cells[peer] & ~core_edge[peer, peer_edge])
    ) & (geometry.neighbour >= 0)
    row = jnp.broadcast_to(jnp.arange(cells)[:, None], (cells, edges))
    excluded = (
        jnp.zeros((cells, cells), dtype=jnp.int32)
        .at[row, peer]
        .add(blocked.astype(jnp.int32))
    )
    return matrix & (excluded == 0)


def connected_cells(geometry, reading, sigma, seed):
    matrix = frame_connections(geometry, reading, sigma).astype(jnp.float32)
    for _ in range(9):
        matrix = (matrix @ matrix > 0).astype(jnp.float32)
    return matrix[seed] > 0


def run(kind, cache, output):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    original = wall_march.connected_cells
    wall_march.connected_cells = connected_cells
    try:
        wall_march.run(kind, cache, output)
    finally:
        wall_march.connected_cells = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.kind, args.cache, args.output)


if __name__ == "__main__":
    main()
