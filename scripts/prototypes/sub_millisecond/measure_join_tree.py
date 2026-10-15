"""Measure a fixed-capacity device join sweep with two candidate levels."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import nova

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.sub_millisecond.measure_cuts import time_call
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def join_levels(values, neighbour, wall, seed):
    """Sort nodes once, fuse component labels, and return first axis events."""
    cells = len(values)
    order = jnp.argsort(values).astype(jnp.int32)
    slot = jnp.arange(cells, dtype=jnp.int32)
    peer = jnp.maximum(neighbour, 0)
    lesser = jnp.where(neighbour >= 0, values[peer] >= values[:, None], True)
    minimum = jnp.all(lesser, axis=1)

    def visit(carry, current):
        label, active, axis, basin, wall_flag, limited, diverted = carry
        active = active.at[current].set(True)
        axis = axis.at[current].set(current == seed)
        basin = basin.at[current].set(minimum[current] & (current != seed))
        wall_flag = wall_flag.at[current].set(wall[current])

        def join(edge, state):
            label, axis, basin, wall_flag, limited, diverted = state
            other = neighbour[current, edge]
            valid = (other >= 0) & active[jnp.maximum(other, 0)]
            current_root = label[current]
            other_root = label[jnp.maximum(other, 0)]
            distinct = valid & (current_root != other_root)
            crossing = (axis[current_root] & basin[other_root]) | (
                axis[other_root] & basin[current_root]
            )
            diverted = jnp.where(
                jnp.isnan(diverted) & distinct & crossing,
                values[current],
                diverted,
            )
            combined_axis = axis[current_root] | axis[other_root]
            combined_basin = basin[current_root] | basin[other_root]
            combined_wall = wall_flag[current_root] | wall_flag[other_root]
            label = jnp.where(distinct & (label == other_root), current_root, label)
            axis = axis.at[current_root].set(
                jnp.where(distinct, combined_axis, axis[current_root])
            )
            basin = basin.at[current_root].set(
                jnp.where(distinct, combined_basin, basin[current_root])
            )
            wall_flag = wall_flag.at[current_root].set(
                jnp.where(distinct, combined_wall, wall_flag[current_root])
            )
            limited = jnp.where(
                jnp.isnan(limited) & distinct & combined_axis & combined_wall,
                values[current],
                limited,
            )
            return label, axis, basin, wall_flag, limited, diverted

        label, axis, basin, wall_flag, limited, diverted = jax.lax.fori_loop(
            0,
            neighbour.shape[1],
            join,
            (label, axis, basin, wall_flag, limited, diverted),
        )
        axis_root = label[seed]
        limited = jnp.where(
            jnp.isnan(limited) & active[seed] & wall_flag[axis_root],
            values[current],
            limited,
        )
        return (label, active, axis, basin, wall_flag, limited, diverted), None

    initial = (
        slot,
        jnp.zeros(cells, bool),
        jnp.zeros(cells, bool),
        jnp.zeros(cells, bool),
        jnp.zeros(cells, bool),
        jnp.asarray(jnp.nan, values.dtype),
        jnp.asarray(jnp.nan, values.dtype),
    )
    final, _ = jax.lax.scan(visit, initial, order)
    return final[-2], final[-1]


def run(kind, cache, output):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, _, _ = certificate._case(name, clip_mode="exact")
    machine, _, _ = _machine(cache, carrier, 250)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic_field, _, _, _ = _analytic_inputs(kind)
    field = certificate_field(analytic_field, kind)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    reading = jax.jit(
        lambda: topology.read(field, geometry, convention, TopologyPolicy())
    )()
    assert bool(reading.valid) and bool(reading.qualified)
    centre = geometry.centre
    scalar = jax.vmap(field.value)(centre)
    outward = -convention.sigma * (scalar - reading.axis_flux)
    neighbour = geometry.neighbour
    wall = jnp.any(neighbour < 0, axis=1)
    seed = int(jnp.argmin(jnp.sum((centre - reading.axis) ** 2, axis=1)))
    def function(values):
        return join_levels(values, neighbour, wall, seed)
    (limited, diverted), wall_ms, compile_s = time_call(function, outward)
    paired = jax.jit(jax.vmap(function))(jnp.stack((outward, outward)))
    assert bool(jnp.allclose(paired[0][0], paired[0][1], equal_nan=True))
    assert bool(jnp.allclose(paired[1][0], paired[1][1], equal_nan=True))
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    slopes = jnp.linspace(-0.12, 0.12, 41)
    centre_z = centre[:, 1] - reading.axis[1]
    varied = outward[None] + slopes[:, None] * span * centre_z[None] / jnp.median(
        geometry.pitch
    )
    levels = jax.jit(jax.vmap(function))(varied)
    pairs = np.stack((np.asarray(levels[0]), np.asarray(levels[1])), axis=1)
    finite = np.all(np.isfinite(pairs), axis=1)
    classes = pairs[:, 0] < pairs[:, 1]
    crossing = np.flatnonzero(classes[1:] != classes[:-1])
    jumps = np.max(np.abs(np.diff(pairs, axis=0)), axis=0)
    result = {
        "kind": kind,
        "cells": len(machine.node),
        "wall_ms": wall_ms,
        "compile_s": compile_s,
        "vmap_width": 2,
        "limited_level": float(limited),
        "diverted_level": float(diverted),
        "reference_boundary_level": float(
            -convention.sigma * (reading.boundary_flux - reading.axis_flux)
        ),
        "sweep_finite_rows": int(finite.sum()),
        "sweep_class_changes": int(len(crossing)),
        "sweep_max_adjacent_change_over_span": (jumps / span).tolist(),
        "sweep_levels": pairs.tolist(),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "RESULT "
        + json.dumps({k: v for k, v in result.items() if k != "sweep_levels"}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.kind, args.cache, args.output)


if __name__ == "__main__":
    main()
