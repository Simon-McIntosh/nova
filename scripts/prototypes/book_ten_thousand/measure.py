"""Diagnose cell moments on the analytic diverted carrier."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
import subprocess
import sys

import jax
import numpy as np
import nova

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures.measure import analytic_profile
from scripts.prototypes.support.geometry import analytic_moments, analytic_polygons
from tests.equilibrium.test_topology_read import (
    _analytic_inputs,
    _kernel_backed_field,
    _realised_hex_geometry,
)


def measure(cells: int, output: Path, fixture_only: bool = False) -> dict:
    """Compare the read, book, and exact support without a coupling build."""
    oracle, field, wall, _, saddle = _analytic_inputs("diverted")
    geometry = _realised_hex_geometry(wall, cells)
    print(f"GEOMETRY_READY {len(geometry.centre)}", flush=True)
    fixture = (
        _kernel_backed_field("diverted", oracle, field, geometry)
        if fixture_only
        else None
    )
    if fixture_only:
        print("REFERENCE_READY", flush=True)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy(normal_form_radius=0.054806712567833996)
    reading = jax.jit(topology.read)(field, geometry, convention, policy)
    jax.block_until_ready(reading)
    print(f"READ_READY valid={bool(reading.valid)}", flush=True)
    samples = jax.vmap(jax.vmap(field.value))(geometry.sample_points)
    _, source, exact = certificate._case("diverted-single-null", clip_mode="exact")
    profile = analytic_profile(source)
    booked = jax.jit(
        lambda psi, read, mesh: current.book(
            psi,
            read,
            mesh,
            profile,
            current.Quadrature(sigma=convention.sigma, segments=64),
        )
    )(samples, reading, geometry)
    jax.block_until_ready(booked)
    print(f"BOOK_READY valid={bool(booked.valid)}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    if fixture_only:
        reference = np.stack(tuple(map(np.asarray, fixture.moments)))
        booked_values = np.stack(tuple(map(np.asarray, booked[:3])))
        ratio = float(
            np.max(np.abs(booked_values - reference)) / np.max(np.abs(reference))
        )
        np.savez_compressed(
            output / f"diverted-{cells}-combined.npz",
            reference=reference,
            booked=booked_values,
            membership=np.asarray(reading.membership),
            fragment_area=np.asarray(reading.fragment_area),
            fragment_selected=np.asarray(reading.fragment_selected),
            read_axis=np.asarray(reading.axis),
            read_x_points=np.asarray(reading.x_points),
            read_x_point_valid=np.asarray(reading.x_point_valid),
            centre=np.asarray(geometry.centre),
        )
        print(f"COMBINED_RATIO={ratio:.15g}", flush=True)
        return {"relative_max": ratio}
    polygons = analytic_polygons(
        exact,
        [
            np.asarray(vertices)[: int(count)]
            for vertices, count in zip(
                geometry.vertices, geometry.vertex_count, strict=True
            )
        ],
        points=4097,
    )
    print("POLYGONS_READY", flush=True)
    truth = analytic_moments(source, polygons, np.asarray(geometry.centre))
    booked_values = np.stack(tuple(map(np.asarray, booked[:3])))
    truth_values = np.stack(tuple(map(np.asarray, truth)))
    full_area = np.asarray(geometry.full_area)
    exact_area = np.asarray([polygon.area for polygon in polygons])
    membership = np.asarray(reading.membership)
    difference = np.abs(booked_values - truth_values)
    scale = np.max(np.abs(truth_values))
    ratio = float(np.max(difference) / scale)
    cell_difference = np.max(difference, axis=0)
    ranked = np.argsort(cell_difference)[::-1]
    top = ranked[: min(20, len(ranked))]
    result = {
        "requested_cells": cells,
        "realised_cells": len(full_area),
        "booked_valid": bool(booked.valid),
        "read_valid": bool(reading.valid),
        "relative_max": ratio,
        "scale": float(scale),
        "top_cells": top.tolist(),
        "top_share_l1": float(cell_difference[top].sum() / cell_difference.sum()),
        "top_records": [
            {
                "cell": int(index),
                "centre": np.asarray(geometry.centre)[index].tolist(),
                "membership": float(membership[index]),
                "fragment_area_fraction": float(
                    np.asarray(reading.fragment_area)[index][
                        np.asarray(reading.fragment_selected)[index]
                    ].sum()
                    / full_area[index]
                ),
                "exact_area_fraction": float(exact_area[index] / full_area[index]),
                "booked_moments": booked_values[:, index].tolist(),
                "exact_moments": truth_values[:, index].tolist(),
                "normal_form": bool(np.asarray(reading.normal_form_cells)[index]),
            }
            for index in top
        ],
        "saddle": np.asarray(saddle).tolist(),
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"diverted-{cells}-cells.npz",
        booked=booked_values,
        exact=truth_values,
        membership=membership,
        fragment_area=np.asarray(reading.fragment_area),
        fragment_selected=np.asarray(reading.fragment_selected),
        exact_area=exact_area,
        full_area=full_area,
        centre=np.asarray(geometry.centre),
        vertices=np.asarray(geometry.vertices),
        vertex_count=np.asarray(geometry.vertex_count),
        normal_form=np.asarray(reading.normal_form_cells),
        wall=np.asarray(wall),
        axis=np.asarray(oracle.magnetic_axis),
        saddle=np.asarray(saddle),
        read_axis=np.asarray(reading.axis),
        read_x_points=np.asarray(reading.x_points),
        read_x_point_valid=np.asarray(reading.x_point_valid),
    )
    (output / f"diverted-{cells}.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs="+", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture-only", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"REVISION={revision} TREE={Path.cwd().resolve()} "
        f"COMMAND={sys.executable} {' '.join(sys.argv)}",
        flush=True,
    )
    print(f"NOVA_MODULE={nova.__file__} CWD={Path.cwd().resolve()}", flush=True)
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu"
    for cells in args.cells:
        measure(cells, args.out, fixture_only=args.fixture_only)


if __name__ == "__main__":
    main()
