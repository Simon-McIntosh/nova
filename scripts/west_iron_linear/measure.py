"""Measure and replay geometry-bound affine WEST iron interactions."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import time

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import numpy as np  # noqa: E402
from nova.biot import magnetisation as iron  # noqa: E402
from nova.biot import tiledassembly as tiled  # noqa: E402
from scripts.iron_graded_refinement.measure import west_inputs, provenance  # noqa: E402
from scripts.iron_linear_magnetisation.assembly_cost import Evaluators  # noqa: E402
from scripts.west_iron_galerkin_pair.measure import (  # noqa: E402
    SOURCE,
    source_field,
)

EVIDENCE = Path("/work/projects/imas_gpu/sophelio/evidence/iron-core-magnetisation")


def save(path, row):
    path.write_text(json.dumps(row, indent=2) + "\n")


def mesh_size(label, inputs):
    if label.startswith("uniform-"):
        return float(label.removeprefix("uniform-"))
    fine, coarse = {"graded-16-80": (0.016, 0.08), "graded-8-40": (0.008, 0.04)}[label]
    return iron.graded_cell_size(
        inputs["surfaces"]["radial"],
        inputs["surfaces"]["vertical"],
        fine=fine,
        coarse=coarse,
        cells_across=20,
    )


def constant_reference(label):
    if label.startswith("uniform-"):
        path = EVIDENCE / "west-62751/mesh-pair/measurement.json"
        key = label.removeprefix("uniform-")
        document = json.loads(path.read_text())
        row = document["meshes"][key]
    else:
        path = EVIDENCE / "graded-refinement/west-cpu/measurement.json"
        key = {"graded-16-80": "graded-fine", "graded-8-40": "graded-half"}[label]
        row = json.loads(path.read_text())["west"][key]
    return dict(
        receipt=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        key=key,
        cells=row["cells"],
        magnitude_T=row["magnitude_T"],
    )


def factor_capacity(cells, geometry, inputs):
    """Compile the actual quadrature shape and measure live factor storage."""
    n = 6 * len(cells.polygons)

    def shape(dims):
        return jax.ShapeDtypeStruct(dims, np.float64)

    law = tuple(
        shape(v.shape) for v in iron._fit_law(inputs["table_b"], inputs["table_mu"])
    )
    operands = (
        shape((n, n)),
        shape((n,)),
        *law,
        shape(geometry["basis"].shape),
        shape(geometry["basis"].shape),
        jax.ShapeDtypeStruct(geometry["owners"].shape, np.int64),
    )
    started = time.monotonic()
    executable = iron._linear_factor.lower(shape((n,)), operands).compile()
    stats = executable.memory_analysis()
    live = (
        stats.argument_size_in_bytes
        + stats.output_size_in_bytes
        + stats.temp_size_in_bytes
        - stats.alias_size_in_bytes
    )
    physical = (
        int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
        * 1024**2
    )
    row = dict(
        compile_seconds=time.monotonic() - started,
        argument_bytes=stats.argument_size_in_bytes,
        output_bytes=stats.output_size_in_bytes,
        temporary_bytes=stats.temp_size_in_bytes,
        compiled_live_bytes=live,
        physical_bytes=physical,
        reserve_bytes=int(0.1 * physical),
        fits=bool(live < 0.9 * physical),
    )
    del executable
    gc.collect()
    return row


def solve(inputs, size, interaction, external=None):
    """Measure projection and Newton separately; forbid hidden assembly."""

    def refuse(*args, **kwargs):
        raise AssertionError("persisted solve attempted interaction assembly")

    iron._galerkin_interaction = refuse
    residual = iron._linear_residual
    clock = {}

    def timed_residual(state, operands):
        if isinstance(state, jax.core.Tracer):
            return residual(state, operands)
        clock.setdefault("start", time.monotonic())
        value = residual(state, operands)
        jax.block_until_ready(value)
        clock["end"] = time.monotonic()
        return value

    iron._linear_residual = timed_residual
    started = time.monotonic()
    result = iron.solve_magnetisation(
        inputs["region"],
        external or inputs["external"],
        inputs["table_b"],
        inputs["table_mu"],
        cell_size=size,
        basis="linear",
        interaction=interaction,
        tolerance=1e-10,
        max_iterations=100,
    )
    wall = time.monotonic() - started
    iron._linear_residual = residual
    air = (external or inputs["external"])(inputs["probes"])
    total = air + result.field(inputs["probes"])
    return dict(
        solve_seconds=wall,
        newton_seconds=clock["end"] - clock["start"],
        preparation_seconds=clock["start"] - started,
        converged=bool(result.converged),
        iterations=result.iterations,
        termination=result.termination,
        residual_history=list(result.residual_history),
        regularisations=list(result.regularisation_history),
        mu_min=float(result.permeability_relative.min()),
        mu_max=float(result.permeability_relative.max()),
        field_T=total.tolist(),
        magnitude_T=np.linalg.norm(total, axis=1).tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--blob", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    jax.config.update("jax_enable_compilation_cache", False)
    stamp = provenance(Path(__file__).resolve(), args.revision, args.blob)
    assert jax.devices()[0].platform == "gpu"
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = west_inputs()
    size = mesh_size(args.mesh, inputs)
    cells = iron._mesh(inputs["region"], size)
    count = len(cells.polygons)
    path = args.output / f"{args.mesh}.json"
    operator_path = args.output / f"{args.mesh}-operator.npy"
    row = dict(
        **stamp,
        mesh=args.mesh,
        cells=count,
        unknowns=6 * count,
        operator_bytes=8 * (6 * count) ** 2,
        completed=False,
        device=jax.devices()[0].device_kind,
        x64=True,
        probes_m=inputs["probes"].tolist(),
        air_drift=inputs["air_drift"],
        constant=constant_reference(args.mesh),
    )
    assert row["constant"]["cells"] == count, "constant receipt names another mesh"
    if args.replay:
        original = json.loads(path.read_text())
        started = time.monotonic()
        interaction = iron.load_magnetisation_interaction(
            operator_path, region=inputs["region"], cell_size=size, basis="linear"
        )
        row["load_seconds"] = time.monotonic() - started
        row["full_drive"] = solve(inputs, size, interaction)
        row["bit_identical"] = (
            np.asarray(row["full_drive"]["field_T"], dtype=np.float64).tobytes()
            == np.asarray(original["full_drive"]["field_T"], dtype=np.float64).tobytes()
        )
        save(args.output / "replay.json", row)
        assert row["bit_identical"], "fresh-process probe fields changed"
        with SOURCE.open("rb") as stream:
            data = copy.deepcopy(pickle.load(stream))
        for key in data["currents"]:
            data["currents"][key] *= 0.5
        half_external = source_field(data)
        row["half_coil_drive"] = solve(inputs, size, interaction, half_external)
        row["plasma_unchanged"] = True
        # Load a real operator from the coarse mesh against this mesh.
        wrong = args.output / "uniform-0.08-operator.npy"
        try:
            iron.load_magnetisation_interaction(
                wrong, region=inputs["region"], cell_size=size, basis="linear"
            )
        except ValueError as error:
            assert "geometry digest" in str(error)
            row["negative_control"] = dict(operator=str(wrong), refusal=str(error))
        else:
            raise AssertionError("different-mesh interaction was accepted")
        row["assembly_calls"] = 0
        row["completed"] = True
        save(args.output / "replay.json", row)
        assert row["full_drive"]["converged"] and row["half_coil_drive"]["converged"]
        print(json.dumps(row), flush=True)
        return
    save(path, row)
    geometry = tiled.linear_magnetisation_geometry(cells.polygons)
    row["capacity"] = factor_capacity(cells, geometry, inputs)
    save(path, row)
    if not row["capacity"]["fits"]:
        row["skipped"] = (
            "compiled factor and interaction exceed card capacity with 10% reserve"
        )
        save(path, row)
        return
    # Assembly compile is measured independently of the Newton-shape compilation.
    timing = Evaluators()
    for name in ("_mutual_edge_evaluator", "_polygon_current_evaluator"):
        setattr(tiled, name, timing.factory(name, getattr(tiled, name)))
    started = time.monotonic()
    matrix, _ = tiled.linear_galerkin_cell_interaction(cells.polygons, plan=iron._PLAN)
    row["assembly_seconds"] = time.monotonic() - started
    row["programs"] = list(timing.programs.values())
    row["compile_seconds"] = sum(p["compile_seconds"] for p in row["programs"])
    row["execute_seconds"] = sum(p["execute_seconds"] for p in row["programs"])
    row["host_seconds"] = (
        row["assembly_seconds"] - row["compile_seconds"] - row["execute_seconds"]
    )
    row["seconds_per_cell_pair"] = row["assembly_seconds"] / count**2
    row["projected_11708_seconds"] = row["seconds_per_cell_pair"] * 11708**2
    save(path, row)
    interaction = iron.save_magnetisation_interaction(
        operator_path, matrix, region=inputs["region"], cell_size=size, basis="linear"
    )
    row["operator_path"] = str(operator_path)
    row["geometry_digest"] = interaction.geometry_digest
    save(path, row)
    row["full_drive"] = solve(inputs, size, interaction)
    row["completed"] = True
    save(path, row)
    print(json.dumps(row), flush=True)
    assert row["full_drive"]["converged"], row["full_drive"]["termination"]


if __name__ == "__main__":
    main()
