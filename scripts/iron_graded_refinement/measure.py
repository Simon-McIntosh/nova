"""Measure graded refinement of the piecewise-constant iron mesh.

The analytic mu_r = 1000 closed shell is solved on uniform and graded meshes,
and WEST 62751 is solved on one graded mesh at two refinement levels. Each arm
records its cell count, per-probe accuracy, and the operator assembly and
solve walls. The WEST source recipe, its coil and plasma quadrature, and the
machine description are imported from ``scripts/west_iron_galerkin_pair`` so
this measurement reads the same source as the mesh pair it refines.

    python scripts/iron_graded_refinement/measure.py --output <run-dir>

One receipt per arm lands as soon as that arm completes, so an allocation
that ends part-way loses one arm rather than the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shlex
import subprocess
import sys
import time

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402
from nova.imas.machine import StaticMachineDescription  # noqa: E402
from scripts.west_iron_galerkin_pair.measure import (  # noqa: E402
    DISTANCES,
    REFERENCE,
    SOURCE,
    SOURCE_SHA256,
    machine_source,
    save,
    source_field,
)
from tests.test_magnetisation import loop, shell  # noqa: E402

SHELL_MU = 1000.0
SHELL_PROBES = np.array([[2.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
SHELL_UNIFORM = (0.04, 0.02, 0.01, 0.008, 0.006)
SHELL_SURFACES = dict(radial=[0.9, 1.0], vertical=[-1.0, 1.0])
# Candidate gradings measured while choosing the arm the tests pin: how far the
# spacing may coarsen away from a surface, and how many cells a thin feature
# keeps across its thickness.
SHELL_SWEEP = (
    ("fine-2mm-coarse-6mm-across-20", 0.002, 0.006, 20.0),
    ("fine-2mm-coarse-8mm-across-20", 0.002, 0.008, 20.0),
    ("fine-2mm-coarse-10mm-across-20", 0.002, 0.01, 20.0),
)


def blob_sha(path: Path) -> str:
    """Return the git blob hash of a file, computed without a repository."""
    data = path.read_bytes()
    return hashlib.sha1(
        b"blob %d\x00" % len(data) + data, usedforsecurity=False
    ).hexdigest()


def provenance(script: Path, revision: str | None, blob: str | None) -> dict:
    """Pin the revision and blob this run executes, refusing a dirty tree.

    A job launched from an archive extracted into its scratch space has no
    repository to consult, so the launcher passes the revision and blob it
    read from the clean checkout and this function re-hashes the extracted
    script. A direct run inspects the checkout it sits in.
    """
    actual = blob_sha(script)
    root = script.parents[2]
    if blob is not None or revision is not None:
        if actual != blob:
            raise SystemExit(
                f"refusing to measure {script} at blob {actual}: "
                f"the launcher stamped blob {blob} at revision {revision}"
            )
        return dict(revision=revision, measure_py_blob=actual, tree=str(root))

    def git(*arguments):
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments], text=True
        ).strip()

    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise SystemExit("refusing to measure a dirty tree:\n" + dirty)
    revision = git("rev-parse", "HEAD")
    committed = git("rev-parse", f"{revision}:{script.relative_to(root)}")
    blob = actual
    if blob != committed:
        raise SystemExit(
            f"refusing to measure {script} at blob {blob}: revision {revision} "
            f"commits blob {committed}"
        )
    return dict(revision=revision, measure_py_blob=blob)


def install_assembly_cache():
    """Assemble each mesh's Galerkin operator once and time the cold build.

    Returns the production callable to restore and a cold-build lookup.
    """
    production = iron._galerkin_interaction
    timings = {}

    def cold(cells, plan=None):
        key = tuple(p.tobytes() for p in cells.polygons)
        if key not in timings:
            started = time.monotonic()
            production(cells, plan=plan or iron._PLAN)
            timings[key] = time.monotonic() - started
        return timings[key]

    cache = {}

    def installed(cells, plan=None):
        key = (tuple(p.tobytes() for p in cells.polygons), plan or iron._PLAN)
        if key not in cache:
            cache[key] = production(cells, plan=plan or iron._PLAN)
        return cache[key]

    return production, installed, cold


def mesh_fields(cells, region):
    polygons = [shapely.Polygon(p) for p in cells.polygons]
    return dict(
        cells=int(len(cells.centroids)),
        area_m2=float(cells.areas.sum()),
        area_union_error_m2=float(
            shapely.union_all(polygons).symmetric_difference(region).area
        ),
    )


def shell_arm(label, size, update):
    region = shell()
    cells = iron._mesh(region, size)
    row = dict(mesh=label, **mesh_fields(cells, region))
    row["assemble_seconds"] = cold_assembly(cells)
    spectrum = np.linalg.eigvalsh(
        np.asarray(iron._galerkin_interaction(cells)) / iron._MU0
    )
    row.update(spectrum_min=float(spectrum[0]), spectrum_max=float(spectrum[-1]))
    update(row)
    started = time.monotonic()
    result = iron.solve_magnetisation(
        region, loop, [0, 10], [SHELL_MU, SHELL_MU], cell_size=size
    )
    row["solve_seconds"] = time.monotonic() - started
    air = loop(SHELL_PROBES)
    shielding = np.linalg.norm(air, axis=1) / np.linalg.norm(
        air + result.field(SHELL_PROBES), axis=1
    )
    exact = ((2 * SHELL_MU + 1) * (SHELL_MU + 2) - 2 * (SHELL_MU - 1) ** 2 * 0.9**3) / (
        9 * SHELL_MU
    )
    row.update(
        converged=bool(result.converged),
        termination=result.termination,
        iterations=result.iterations,
        residual_history=list(result.residual_history),
        shielding=shielding.tolist(),
        shielding_exact=float(exact),
        relative_error=np.abs(shielding / exact - 1).tolist(),
        mesh_cells_match=int(result.cell_count) == row["cells"],
    )
    print("SHELL " + json.dumps(row), flush=True)
    update(row)
    return row


def west_arm(label, field, update, inputs):
    region = inputs["region"]
    cells = iron._mesh(region, field)
    row = dict(mesh=label, **mesh_fields(cells, region))
    row["assemble_seconds"] = cold_assembly(cells)
    update(row)
    started = time.monotonic()
    result = iron.solve_magnetisation(
        region,
        inputs["external"],
        inputs["table_b"],
        inputs["table_mu"],
        cell_size=field,
        tolerance=1e-10,
        max_iterations=100,
    )
    row["solve_seconds"] = time.monotonic() - started
    total = inputs["air"] + result.field(inputs["probes"])
    row.update(
        converged=bool(result.converged),
        termination=result.termination,
        iterations=result.iterations,
        residual_history=list(result.residual_history),
        mu_min=float(result.permeability_relative.min()),
        mu_max=float(result.permeability_relative.max()),
        field_T=total.tolist(),
        magnitude_T=np.linalg.norm(total, axis=1).tolist(),
        iron_to_air=(
            np.linalg.norm(total, axis=1) / np.linalg.norm(inputs["air"], axis=1)
        ).tolist(),
        mesh_cells_match=int(result.cell_count) == row["cells"],
    )
    print("WEST " + json.dumps(row), flush=True)
    update(row)
    np.save(OUT / "west" / label / "centroids.npy", cells.centroids)
    np.save(OUT / "west" / label / "M.npy", result.magnetisation)
    return row


def west_inputs():
    """Rebuild the WEST region, source field and material table from their pins."""
    assert (
        Path(iron.__file__)
        .resolve()
        .is_relative_to(Path(__file__).resolve().parents[2])
    )
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == SOURCE_SHA256, digest
    with SOURCE.open("rb") as stream:
        data = pickle.load(stream)
    uri, dd_version = machine_source()
    machine = StaticMachineDescription.read_iron_core(uri, dd_version=dd_version)
    assert len(machine.iron_segments) == 6
    table = machine.iron_segments[0].material
    table_b = np.asarray(table.b_field)
    table_mu = np.asarray(table.permeability_relative)
    region = shapely.union_all(
        [
            shapely.Polygon(np.column_stack((s.outline.r, s.outline.z)))
            for s in machine.iron_segments
        ]
    )
    assert region.is_valid
    external = source_field(data)
    probes = np.column_stack((data["edge"] + DISTANCES, np.zeros(3)))
    air = external(probes)
    reference = json.loads(REFERENCE.read_text())
    np.testing.assert_allclose(reference["probes_m"], probes, rtol=0, atol=0)
    np.testing.assert_array_equal(reference["table_b_T"], table_b.tolist())
    np.testing.assert_array_equal(reference["table_mu_relative"], table_mu.tolist())
    drift = float(np.max(np.abs(air / np.asarray(reference["air_field_T"]) - 1)))
    assert drift < 1e-12, drift
    # Each segment's bounds are the faces the grading resolves; an internal
    # face between touching segments only adds cells, it never coarsens one.
    radial = sorted(
        {
            float(value)
            for segment in machine.iron_segments
            for value in (np.min(segment.outline.r), np.max(segment.outline.r))
        }
    )
    vertical = sorted(
        {
            float(value)
            for segment in machine.iron_segments
            for value in (np.min(segment.outline.z), np.max(segment.outline.z))
        }
    )
    return dict(
        region=region,
        external=external,
        table_b=table_b,
        table_mu=table_mu,
        probes=probes,
        air=air,
        air_drift=drift,
        uri=uri,
        dd_version=dd_version,
        surfaces=dict(radial=radial, vertical=vertical),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fine", type=float, default=0.002)
    parser.add_argument("--coarse", type=float, default=0.004)
    parser.add_argument("--cells-across", type=float, default=20.0)
    parser.add_argument("--west-fine", type=float, default=0.008)
    parser.add_argument("--west-coarse", type=float, default=0.08)
    parser.add_argument("--arms", default="shell,graded,west")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--blob", default=None)
    args = parser.parse_args()
    global OUT, cold_assembly
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    arms = set(args.arms.split(","))
    script = Path(__file__).resolve()
    record = dict(
        **provenance(script, args.revision, args.blob),
        module=iron.__file__,
        cwd=str(Path.cwd().resolve()),
        device=jax.devices()[0].device_kind,
        command=shlex.join([sys.executable, *sys.argv]),
        grading=dict(
            shell=dict(
                fine=args.fine,
                coarse=args.coarse,
                cells_across=args.cells_across,
                surfaces=SHELL_SURFACES,
            ),
            west=dict(
                fine=args.west_fine,
                coarse=args.west_coarse,
                cells_across=args.cells_across,
            ),
            growth=2.0,
        ),
        shell={},
        sweep={},
        west={},
    )
    print(
        f"# revision: {record['revision']} "
        f"measure_py_blob: {record['measure_py_blob']} tree: {record['cwd']}",
        flush=True,
    )
    print(
        f"MODULE={iron.__file__} CWD={record['cwd']} DEVICE={jax.devices()}", flush=True
    )
    assert jax.config.jax_enable_x64, "the material solve requires float64"

    production, installed, cold_assembly = install_assembly_cache()

    if (OUT / "measurement.json").is_file():
        # An allocation that ended part-way leaves the arms it finished; adopt
        # them so a rerun of the same command spends its time on the rest.
        for section in ("shell", "sweep", "west"):
            for label, row in (
                json.loads((OUT / "measurement.json").read_text())
                .get(section, {})
                .items()
            ):
                if row.get("converged"):
                    record[section][label] = row

    def finished(section, label):
        return record[section].get(label, {}).get("converged", False)

    def shell_update(row, label=None):
        if label is not None:
            record["shell"][label] = row
            save(OUT / "shell" / label / "receipt.json", row)
        save(OUT / "measurement.json", record)

    def west_update(row, label=None):
        if label is not None:
            record["west"][label] = row
            save(OUT / "west" / label / "receipt.json", row)
        save(OUT / "measurement.json", record)

    def sweep_update(row, label=None):
        if label is not None:
            record["sweep"][label] = row
            save(OUT / "sweep" / label / "receipt.json", row)
        save(OUT / "measurement.json", record)

    iron._galerkin_interaction = installed
    try:
        if "shell" in arms:
            for size in SHELL_UNIFORM:
                label = f"uniform-{size:.4g}"
                if finished("shell", label):
                    print(f"SKIP shell {label}: receipt already converged", flush=True)
                    continue
                (OUT / "shell" / label).mkdir(parents=True, exist_ok=True)
                shell_arm(
                    label, size, lambda row, label=label: shell_update(row, label)
                )
        if "graded" in arms:
            # The pinned grading and its deliberately under-resolved doubling:
            # the pair the accuracy test asserts on.
            for label, scale in (("graded-double", 2.0), ("graded-fine", 1.0)):
                if finished("shell", label):
                    print(f"SKIP shell {label}: receipt already converged", flush=True)
                    continue
                field = iron.graded_cell_size(
                    SHELL_SURFACES["radial"],
                    SHELL_SURFACES["vertical"],
                    args.fine * scale,
                    args.coarse * scale,
                    cells_across=args.cells_across,
                )
                (OUT / "shell" / label).mkdir(parents=True, exist_ok=True)
                shell_arm(
                    label, field, lambda row, label=label: shell_update(row, label)
                )
                record["shell"][label]["grading_scale"] = scale
                save(OUT / "measurement.json", record)
        if "sweep" in arms:
            for label, fine, coarse, across in SHELL_SWEEP:
                if finished("sweep", label):
                    print(f"SKIP sweep {label}: receipt already converged", flush=True)
                    continue
                field = iron.graded_cell_size(
                    SHELL_SURFACES["radial"],
                    SHELL_SURFACES["vertical"],
                    fine,
                    coarse,
                    cells_across=across,
                )
                (OUT / "sweep" / label).mkdir(parents=True, exist_ok=True)
                shell_arm(
                    label,
                    field,
                    lambda row, label=label: sweep_update(row, label),
                )
                record["sweep"][label]["grading"] = dict(
                    fine=fine, coarse=coarse, cells_across=across
                )
                save(OUT / "measurement.json", record)
        if "west" in arms:
            inputs = west_inputs()
            record["west_inputs"] = dict(
                source=str(SOURCE),
                machine=inputs["uri"].removeprefix("imas:hdf5?path="),
                dd_version=inputs["dd_version"],
                surfaces=inputs["surfaces"],
                probes_m=inputs["probes"].tolist(),
                air_field_T=inputs["air"].tolist(),
                air_field_reproduction_max_relative=inputs["air_drift"],
            )
            save(OUT / "measurement.json", record)
            for label, scale in (("graded-fine", 1.0), ("graded-half", 0.5)):
                if finished("west", label):
                    print(f"SKIP west {label}: receipt already converged", flush=True)
                    continue
                field = iron.graded_cell_size(
                    inputs["surfaces"]["radial"],
                    inputs["surfaces"]["vertical"],
                    args.west_fine * scale,
                    args.west_coarse * scale,
                    cells_across=args.cells_across,
                )
                (OUT / "west" / label).mkdir(parents=True, exist_ok=True)
                west_arm(
                    label,
                    field,
                    lambda row, label=label: west_update(row, label),
                    inputs,
                )
                record["west"][label]["grading_scale"] = scale
                save(OUT / "measurement.json", record)
    finally:
        iron._galerkin_interaction = production

    if "graded-fine" in record["west"] and "graded-half" in record["west"]:
        coarse_field = np.asarray(record["west"]["graded-fine"]["magnitude_T"])
        fine_field = np.asarray(record["west"]["graded-half"]["magnitude_T"])
        record["west_level_change"] = dict(
            finer_over_coarser_minus_one=(fine_field / coarse_field - 1).tolist(),
            magnitude_T=fine_field.tolist(),
            cells=[
                record["west"]["graded-fine"]["cells"],
                record["west"]["graded-half"]["cells"],
            ],
        )
        print("WESTCHANGE " + json.dumps(record["west_level_change"]), flush=True)
    record["complete"] = True
    save(OUT / "measurement.json", record)
    print("SUMMARY " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
