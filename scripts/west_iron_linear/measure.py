"""Measure and replay geometry-bound affine WEST iron interactions."""

from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
import gc
import hashlib
import json
from pathlib import Path
import pickle
import resource
import subprocess
import time
from unittest.mock import patch

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


def interaction_agreement(reference, candidate):
    """Compare every entry in bounded row chunks, including transferred operands."""
    if reference.shape != candidate.shape:
        raise AssertionError("interaction shapes differ")
    squared_error = squared_reference = maximum_error = maximum_reference = 0.0
    entries = 0
    for first in range(0, len(reference), 256):
        expected = np.asarray(reference[first : first + 256])
        actual = np.asarray(candidate[first : first + 256])
        assert np.isfinite(expected).all() and np.isfinite(actual).all()
        difference = actual - expected
        squared_error += float(np.vdot(difference, difference))
        squared_reference += float(np.vdot(expected, expected))
        maximum_error = max(maximum_error, float(np.abs(difference).max()))
        maximum_reference = max(maximum_reference, float(np.abs(expected).max()))
        entries += expected.size
    assert maximum_reference > 0, "interaction comparison has no nonzero control"
    row = dict(
        entries=entries,
        frobenius_relative=(squared_error / squared_reference) ** 0.5,
        maximum_entry_relative=maximum_error / maximum_reference,
    )
    assert max(row["frobenius_relative"], row["maximum_entry_relative"]) <= 1e-12, row
    return row


def probe_agreement(reference, candidate):
    """Require each probe vector to agree relative to its reference magnitude."""
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    assert reference.shape == candidate.shape == (3, 2)
    scale = np.linalg.norm(reference, axis=1)
    assert np.isfinite(candidate).all() and np.all(scale > 0)
    relative = np.linalg.norm(candidate - reference, axis=1) / scale
    assert np.all(relative <= 1e-12), relative
    return relative.tolist()


def refinement_verdict(reference, candidate):
    """A one-percent change at any probe leaves the refinement blocked."""
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    assert reference.shape == candidate.shape == (3,)
    assert np.isfinite(candidate).all() and np.all(reference > 0)
    change = (candidate - reference) / reference
    return dict(
        status="complete" if np.all(np.abs(change) < 0.01) else "blocked",
        relative_change=change.tolist(),
        maximum_change=float(np.abs(change).max()),
    )


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


def factor_capacity(cells, geometry, inputs, *, factorisation="combined"):
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
    programs = {}
    functions = (
        [("factor", iron._linear_factor, (shape((n,)), operands))]
        if factorisation == "combined"
        else [
            ("jacobian", iron._linear_jacobian, (shape((n,)), operands)),
            ("lu", iron._donated_lu, (shape((n, n)),)),
        ]
    )
    for name, function, arguments in functions:
        executable = function.lower(*arguments).compile()
        stats = executable.memory_analysis()
        live = (
            stats.argument_size_in_bytes
            + stats.output_size_in_bytes
            + stats.temp_size_in_bytes
            - stats.alias_size_in_bytes
        )
        # LU receives only the Jacobian; the interaction remains resident.
        retained = 8 * n**2 if name == "lu" else 0
        programs[name] = dict(
            argument_bytes=stats.argument_size_in_bytes,
            output_bytes=stats.output_size_in_bytes,
            temporary_bytes=stats.temp_size_in_bytes,
            alias_bytes=stats.alias_size_in_bytes,
            retained_interaction_bytes=retained,
            live_bytes=live + retained,
        )
        del executable
    live = max(program["live_bytes"] for program in programs.values())
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
        factorisation=factorisation,
        programs=programs,
        compiled_live_bytes=live,
        physical_bytes=physical,
        allocator_budget_bytes=jax.devices()[0].memory_stats()["bytes_limit"],
        reserve_bytes=int(0.1 * physical),
        fits=bool(
            live < min(0.9 * physical, jax.devices()[0].memory_stats()["bytes_limit"])
        ),
    )
    gc.collect()
    return row


def solve(
    inputs,
    size,
    interaction,
    external=None,
    *,
    factorisation="combined",
    observe_interaction=None,
):
    """Measure projection and Newton separately; forbid hidden assembly."""

    def refuse(*args, **kwargs):
        raise AssertionError("persisted solve attempted interaction assembly")

    residual = iron._linear_residual
    clock = {}
    preparation = {}

    def timed(name, function):
        def call(*args, **kwargs):
            began = time.monotonic()
            try:
                return function(*args, **kwargs)
            finally:
                preparation[name] = preparation.get(name, 0.0) + (
                    time.monotonic() - began
                )

        return call

    def timed_residual(state, operands):
        if isinstance(state, jax.core.Tracer):
            return residual(state, operands)
        if "start" not in clock:
            # Charge pending operand transfers to preparation, before Newton.
            jax.block_until_ready((state, operands))
            if observe_interaction is not None:
                observe_interaction(operands[0])
        clock.setdefault("start", time.monotonic())
        value = residual(state, operands)
        jax.block_until_ready(value)
        clock["end"] = time.monotonic()
        return value

    with ExitStack() as stack:
        for name, attribute in (
            ("law_fit_seconds", "_fit_law"),
            ("mesh_seconds", "_mesh"),
            ("linear_geometry_seconds", "linear_magnetisation_geometry"),
        ):
            stack.enter_context(
                patch.object(iron, attribute, timed(name, getattr(iron, attribute)))
            )
        stack.enter_context(
            patch.object(
                iron.MagnetisationInteraction,
                "validate",
                timed("interaction_validation_seconds", interaction.validate.__func__),
            )
        )
        stack.enter_context(patch.object(iron, "_galerkin_interaction", refuse))
        stack.enter_context(patch.object(iron, "_linear_residual", timed_residual))
        started = time.monotonic()
        result = iron.solve_magnetisation(
            inputs["region"],
            timed("external_field_seconds", external or inputs["external"]),
            inputs["table_b"],
            inputs["table_mu"],
            cell_size=size,
            basis="linear",
            interaction=interaction,
            tolerance=1e-10,
            max_iterations=100,
            linear_factorisation=factorisation,
        )
        wall = time.monotonic() - started
    preparation["projection_checks_and_device_setup_seconds"] = (
        clock["start"] - started - sum(preparation.values())
    )
    air = (external or inputs["external"])(inputs["probes"])
    total = air + result.field(inputs["probes"])
    return dict(
        solve_seconds=wall,
        newton_seconds=clock["end"] - clock["start"],
        preparation_seconds=clock["start"] - started,
        preparation_breakdown=preparation,
        preparation_breakdown_note=(
            "Disjoint host wall intervals inside solve_magnetisation. Linear geometry "
            "includes quadrature, basis, Gram blocks and bound-current incidence. "
            "The remainder includes external-field projection, matrix finiteness "
            "scan, operand construction and synchronized host-to-device transfers. "
            "Probe evaluation and persisted-operator loading are outside this split."
        ),
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
    parser.add_argument("--refinement", action="store_true")
    parser.add_argument("--resume-refinement", action="store_true")
    parser.add_argument("--compare-routes", action="store_true")
    replay_mode = parser.add_mutually_exclusive_group()
    replay_mode.add_argument("--replay", action="store_true")
    replay_mode.add_argument("--half-drive-only", action="store_true")
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
        module=str(Path(iron.__file__).resolve()),
        cwd=str(Path.cwd().resolve()),
    )
    assert row["constant"]["cells"] == count, "constant receipt names another mesh"
    if args.resume_refinement:
        assert args.refinement and args.mesh == "graded-8-40"
        original = json.loads(path.read_text())
        assert original["geometry_digest"] and original["operator_path"] == str(
            operator_path
        )
        archive = args.output / "graded-8-40-assembly.json"
        if not archive.exists():
            save(archive, original)
        assembly = json.loads(archive.read_text())
        row = original | row
        row["assembly_provenance"] = dict(
            revision=assembly["revision"],
            measure_py_blob=assembly["measure_py_blob"],
            receipt=str(archive),
            geometry_digest=assembly["geometry_digest"],
        )
        interaction = iron.load_magnetisation_interaction(
            operator_path, region=inputs["region"], cell_size=size, basis="linear"
        )
        assert interaction.geometry_digest == assembly["geometry_digest"]
        geometry = tiled.linear_magnetisation_geometry(cells.polygons)
        row["split_capacity"] = factor_capacity(
            cells, geometry, inputs, factorisation="split"
        )
        save(path, row)
        assert row["split_capacity"]["fits"], row["split_capacity"]
        row["full_drive"] = solve(inputs, size, interaction, factorisation="split")
        row["solve_device_memory_stats"] = jax.devices()[0].memory_stats()
        row["assembly_calls"] = 0
        reference_path = args.output.parent / "graded-16-80.json"
        row["refinement_reference"] = str(reference_path)
        reference = json.loads(reference_path.read_text())["full_drive"]["magnitude_T"]
        row["refinement"] = refinement_verdict(
            reference, row["full_drive"]["magnitude_T"]
        )
        row["completed"] = bool(row["full_drive"]["converged"])
        save(path, row)
        assert row["completed"], row["full_drive"]["termination"]
        print(json.dumps(row), flush=True)
        return
    if args.compare_routes:
        source = args.output.parent / f"{args.mesh}-operator.npy"
        interaction = iron.load_magnetisation_interaction(
            source, region=inputs["region"], cell_size=size, basis="linear"
        )
        row["operator_path"] = str(source)
        row["geometry_digest"] = interaction.geometry_digest
        save(path, row)
        for route in ("combined", "split"):

            def observe(matrix):
                row[f"{route}_interaction"] = interaction_agreement(
                    interaction.matrix, matrix
                )

            row[route] = solve(
                inputs,
                size,
                interaction,
                factorisation=route,
                observe_interaction=observe,
            )
            save(path, row)
            assert row[route]["converged"], row[route]["termination"]
        row["probe_relative_difference"] = probe_agreement(
            row["combined"]["field_T"], row["split"]["field_T"]
        )
        row["completed"] = True
        save(path, row)
        print(json.dumps(row), flush=True)
        return
    if args.replay or args.half_drive_only:
        started = time.monotonic()
        interaction = iron.load_magnetisation_interaction(
            operator_path, region=inputs["region"], cell_size=size, basis="linear"
        )
        row["load_seconds"] = time.monotonic() - started
        row["geometry_digest"] = interaction.geometry_digest
        row["operator_path"] = str(operator_path)
        if args.replay:
            original = json.loads(path.read_text())
            row["full_drive"] = solve(inputs, size, interaction)
            row["bit_identical"] = (
                np.asarray(row["full_drive"]["field_T"], dtype=np.float64).tobytes()
                == np.asarray(
                    original["full_drive"]["field_T"], dtype=np.float64
                ).tobytes()
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
        receipt = "half-drive.json" if args.half_drive_only else "replay.json"
        save(args.output / receipt, row)
        assert row["half_coil_drive"]["converged"]
        if args.replay:
            assert row["full_drive"]["converged"]
        print(json.dumps(row), flush=True)
        return
    save(path, row)
    geometry = tiled.linear_magnetisation_geometry(cells.polygons)
    row["capacity"] = factor_capacity(cells, geometry, inputs)
    if args.refinement:
        row["split_capacity"] = factor_capacity(
            cells, geometry, inputs, factorisation="split"
        )
    save(path, row)
    capacity = row["split_capacity"] if args.refinement else row["capacity"]
    if not capacity["fits"]:
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
    row["assembly_peak_host_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    row["assembly_device_memory_stats"] = jax.devices()[0].memory_stats()
    row["assembly_program_memory"] = {
        str(key): dict(
            argument_bytes=(
                stats := executable.memory_analysis()
            ).argument_size_in_bytes,
            output_bytes=stats.output_size_in_bytes,
            temporary_bytes=stats.temp_size_in_bytes,
            alias_bytes=stats.alias_size_in_bytes,
        )
        for key, executable in timing.executables.items()
    }
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
    row["full_drive"] = solve(
        inputs,
        size,
        interaction,
        factorisation="split" if args.refinement else "combined",
    )
    if args.refinement:
        reference_path = args.output.parent / "graded-16-80.json"
        row["refinement_reference"] = str(reference_path)
        reference = json.loads(reference_path.read_text())["full_drive"]["magnitude_T"]
        row["refinement"] = refinement_verdict(
            reference, row["full_drive"]["magnitude_T"]
        )
        row["solve_device_memory_stats"] = jax.devices()[0].memory_stats()
    row["completed"] = True
    save(path, row)
    print(json.dumps(row), flush=True)
    assert row["full_drive"]["converged"], row["full_drive"]["termination"]


if __name__ == "__main__":
    main()
