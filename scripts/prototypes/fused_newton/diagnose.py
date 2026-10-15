"""Compare fixed-shape saddle booking with the analytic exact-cut control."""

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
from nova.equilibrium import current, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.fused_newton.program import _saddle_moments
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.sub_millisecond.measure_bell_cut import bell_moments
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def run(cache, output):
    assert jax.config.jax_enable_x64 is True
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}",
        flush=True,
    )
    carrier, source, _ = certificate._case("diverted-single-null", clip_mode="exact")
    machine, _, _ = _machine(cache, carrier, 250)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic_field, _, _, _ = _analytic_inputs("diverted")
    field = certificate_field(analytic_field, "diverted")
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    reading = jax.jit(
        lambda: topology.read(field, geometry, convention, TopologyPolicy())
    )()
    assert bool(reading.valid) and bool(reading.qualified)
    samples = jax.vmap(field.value)(geometry.sample_points.reshape(-1, 2)).reshape(
        geometry.sample_points.shape[:-1]
    )
    profile = fixture.analytic_profile(source)
    exact = jax.jit(
        lambda psi: current.book(
            psi,
            reading,
            geometry,
            profile,
            current.Quadrature(sigma=convention.sigma),
        )
    )(samples)
    assert bool(exact.valid)
    exact_moments = np.asarray(
        jnp.stack(
            (exact.cell_current, exact.radial_moment, exact.vertical_moment), axis=-1
        )
    )
    saddle, valid = jax.jit(lambda: _saddle_moments(geometry, reading, profile))()
    ordinary, _ = jax.jit(
        lambda: bell_moments(geometry, reading, profile, convention.sigma, 4)
    )()
    selected = np.flatnonzero(
        np.asarray(reading.normal_form_cells) & (np.asarray(reading.membership) > 0)
    )
    rows = []
    for index in selected:
        rows.append(
            {
                "cell": int(index),
                "exact": exact_moments[index].tolist(),
                "saddle": np.asarray(saddle[index]).tolist(),
                "ordinary": np.asarray(ordinary[index]).tolist(),
                "membership": float(reading.membership[index]),
                "selected_fragments": np.asarray(
                    reading.fragment_selected[index]
                ).tolist(),
            }
        )
    result = {
        "cells": len(machine.node),
        "saddle_cells": len(selected),
        "saddle_valid": bool(valid),
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.output)


if __name__ == "__main__":
    main()
