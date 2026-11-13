"""Measure WEST 62751 iron on the Galerkin operator at two cell sizes.

Both meshes are solved from a demagnetised state at full drive through the
production material path, and the far field at the plan's probes is recorded
for each. The difference between the two is the discretisation error bar.

    python scripts/west_iron_galerkin_pair/measure.py --output <run-dir>

The run directory receives one receipt per mesh as it lands, so an allocation
that ends part-way loses one mesh rather than the pair.
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
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402
from nova.biot.greens import greens_bz_br  # noqa: E402
from nova.imas.machine import StaticMachineDescription  # noqa: E402

SOURCE = Path("/home/ITER/mcintos/.cache/nova-west-stray/hex/west-62751-hex.pkl")
SOURCE_SHA256 = "228e9c7a277e6d16f180cae4566eaadbb4e8df13ae2b49e78a33c53bd5918533"
EVIDENCE = Path(
    "/work/projects/imas_gpu/sophelio/evidence/iron-core-magnetisation/west-62751/mesh-pair"
)
REFERENCE = EVIDENCE / "inputs/west-62751-solve-inputs.json"
PRIOR_ROOTS = EVIDENCE / "inputs/galerkin-interaction-measurement.json"
DISTANCES = np.array([1.0, 2.0, 4.0])
MU0 = 4e-7 * np.pi


def machine_source():
    """Resolve the WEST machine description at its curation manifest's DD pin."""
    manifest_path = Path("/home/ITER/mcintos/Code/imas-efit/tests/data-manifest.json")
    manifest = json.loads(manifest_path.read_text())
    row = next(
        entry
        for entry in manifest["machine_descriptions"]
        if entry["key"] == "west-dd4-1-0"
    )
    source = manifest_path.parent.parent / row["target"]
    return f"imas:hdf5?path={source}", row["dd_version"]


def source_field(data):
    """Rebuild the prototype's coil quadrature and prescribed plasma currents."""
    radius, height, amps = [], [], []
    for coil in data["coils"]:
        offsets = (np.arange(6) + 0.0 + 0.5) / 6 - 0.5
        rr, zz = np.meshgrid(
            coil["r"] + offsets * coil["dr"], coil["z"] + offsets * coil["dz"]
        )
        radius.extend(rr.ravel())
        height.extend(zz.ravel())
        amps.extend(
            np.full(rr.size, data["currents"][coil["name"]] * coil["nturn"] / rr.size)
        )
    for polygon, current in zip(
        data["plasma_cells"], data["plasma_weight"], strict=True
    ):
        centre = shapely.Polygon(polygon).centroid
        radius.append(centre.x)
        height.append(centre.y)
        amps.append(current)
    radius, height, amps = map(np.asarray, (radius, height, amps))

    def evaluate(points):
        result = np.empty_like(points)
        for start in range(0, len(points), 256):
            stop = start + 256
            bz, br = greens_bz_br(
                points[start:stop, 0, None],
                points[start:stop, 1, None],
                radius[None],
                height[None],
            )
            assert np.isfinite(br).all() and np.isfinite(bz).all()
            result[start:stop] = np.column_stack((br @ amps, bz @ amps))
        return result

    return evaluate


def save(path, record):
    path.write_text(json.dumps(record, indent=2) + "\n")


def spectrum(gram, cells):
    """Return the symmetric spectrum of the volume-weighted G / mu0."""
    volume = np.tile(2 * np.pi * cells.areas * cells.centroids[:, 0], 2)
    weighted_energy = volume[:, None] * gram
    symmetry = float(
        np.linalg.norm(weighted_energy - weighted_energy.T)
        / np.linalg.norm(weighted_energy)
    )
    scale = jnp.asarray(np.sqrt(volume))
    weighted = jnp.asarray(gram) / iron._MU0 * scale[:, None] / scale[None, :]
    eigenvalues = np.asarray(jnp.linalg.eigvalsh((weighted + weighted.T) / 2))
    return dict(
        cells=int(len(cells.centroids)),
        weighted_symmetry=symmetry,
        minimum=float(eigenvalues[0]),
        maximum=float(eigenvalues[-1]),
    )


def memoise_interaction():
    """Return the production interaction, assembled once per (mesh, targets)."""
    production = iron._interaction
    cache = {}

    def interaction(cells, points, plan):
        key = (
            tuple(p.tobytes() for p in cells.polygons),
            None if points is None else np.asarray(points).tobytes(),
        )
        if key not in cache:
            cache[key] = production(cells, points, plan)
        return cache[key]

    return production, interaction


def measure_size(region, external, table_b, table_mu, size, air, probes, out):
    row = dict(cell_size_m=size, phase="starting")
    save(out / f"{size:.4g}/receipt.json", row)
    cells = iron._mesh(region, size)
    polygons = [shapely.Polygon(p) for p in cells.polygons]
    volume = 2 * np.pi * cells.centroids[:, 0] * cells.areas
    clearance = np.array([p.centroid.distance(p.boundary) for p in polygons])
    started = time.monotonic()
    gram = iron._interaction(cells, None, iron._PLAN)
    build_seconds = time.monotonic() - started
    row.update(
        phase="operator-complete",
        cells=int(len(cells.centroids)),
        area_m2=float(cells.areas.sum()),
        area_union_error_m2=float(
            shapely.union_all(polygons).symmetric_difference(region).area
        ),
        mean_area_over_h2=float(cells.areas.mean() / size**2),
        clearance_min_m=float(clearance.min()),
        clearance_median_m=float(np.median(clearance)),
        build_seconds=build_seconds,
        spectrum=spectrum(gram, cells),
    )
    print("OPERATOR " + json.dumps(row["spectrum"]), flush=True)
    save(out / f"{size:.4g}/receipt.json", row)
    started = time.monotonic()
    result = iron.solve_magnetisation(
        region,
        external,
        table_b,
        table_mu,
        cell_size=size,
        tolerance=1e-10,
        max_iterations=100,
    )
    total = air + result.field(probes)
    row.update(
        phase="nonlinear-complete",
        converged=result.converged,
        termination=result.termination,
        iterations=result.iterations,
        residual_history=list(result.residual_history),
        step_scales=list(result.step_scales),
        step_types=list(result.step_types),
        regularisation_history=list(result.regularisation_history),
        mu_min=float(result.permeability_relative.min()),
        mu_max=float(result.permeability_relative.max()),
        volume_fraction_above_mu_100=float(
            volume[result.permeability_relative > 100].sum() / volume.sum()
        ),
        volume_fraction_above_mu_1000=float(
            volume[result.permeability_relative > 1000].sum() / volume.sum()
        ),
        volume_m3=float(volume.sum()),
        maximum_M_A_per_m=float(np.linalg.norm(result.magnetisation, axis=1).max()),
        field_T=total.tolist(),
        magnitude_T=np.linalg.norm(total, axis=1).tolist(),
        iron_to_air=(
            np.linalg.norm(total, axis=1) / np.linalg.norm(air, axis=1)
        ).tolist(),
        solve_seconds=time.monotonic() - started,
        mesh_cells_match=int(result.cell_count) == len(cells.centroids),
    )
    print(
        "SOLVE "
        + json.dumps(
            {
                key: row[key]
                for key in (
                    "cell_size_m",
                    "cells",
                    "converged",
                    "termination",
                    "iterations",
                    "residual_history",
                    "mu_min",
                    "mu_max",
                    "magnitude_T",
                )
            }
        ),
        flush=True,
    )
    save(out / f"{size:.4g}/receipt.json", row)
    np.save(out / f"{size:.4g}/M.npy", result.magnetisation)
    np.save(out / f"{size:.4g}/mu.npy", result.permeability_relative)
    np.save(out / f"{size:.4g}/centroids.npy", cells.centroids)
    with (out / f"{size:.4g}/solution.pkl").open("wb") as stream:
        pickle.dump(result, stream)
    started = time.monotonic()
    flat = iron.solve_magnetisation(
        region,
        external,
        table_b,
        np.full_like(np.asarray(table_mu), 215.0),
        cell_size=size,
        tolerance=1e-10,
        max_iterations=100,
    )
    flat_total = air + flat.field(probes)
    row.update(
        flat_converged=flat.converged,
        flat_iterations=flat.iterations,
        flat_magnitude_T=np.linalg.norm(flat_total, axis=1).tolist(),
        flat_iron_to_air=(
            np.linalg.norm(flat_total, axis=1) / np.linalg.norm(air, axis=1)
        ).tolist(),
        table_minus_flat_T=(
            np.linalg.norm(total, axis=1) - np.linalg.norm(flat_total, axis=1)
        ).tolist(),
        table_minus_flat_over_air=(
            (np.linalg.norm(total, axis=1) - np.linalg.norm(flat_total, axis=1))
            / np.linalg.norm(air, axis=1)
        ).tolist(),
        flat_seconds=time.monotonic() - started,
    )
    print(
        "FLAT "
        + json.dumps(
            {
                key: row[key]
                for key in (
                    "cell_size_m",
                    "flat_converged",
                    "flat_magnitude_T",
                    "table_minus_flat_over_air",
                )
            }
        ),
        flush=True,
    )
    row["phase"] = "complete"
    save(out / f"{size:.4g}/receipt.json", row)
    return row


def code_provenance():
    """Return the revision and the blob hash of the code this run executes.

    A receipt that names only the tree HEAD can describe code that never ran:
    the measured script is a working-tree file, and an edit between the last
    commit and the launch is invisible in HEAD. So the tree must be clean, and
    the blob of this file must be the blob that HEAD commits.
    """
    root = Path(__file__).resolve().parents[2]
    script = Path(__file__).resolve()

    def git(*arguments):
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments], text=True
        ).strip()

    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise SystemExit(
            "refusing to measure a dirty tree; commit these paths first:\n" + dirty
        )
    revision = git("rev-parse", "HEAD")
    blob = git("hash-object", "--", str(script))
    committed = git("rev-parse", f"{revision}:{script.relative_to(root)}")
    if blob != committed:
        raise SystemExit(
            f"refusing to measure {script} at blob {blob}: revision {revision} "
            f"commits blob {committed}"
        )
    return revision, blob


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=float, nargs="+", default=[0.08, 0.04])
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    revision, blob = code_provenance()
    source_bytes = Path(iron.__file__).read_bytes()
    print(
        f"# revision: {revision} measure_py_blob: {blob} tree: {Path.cwd().resolve()} "
        "command: " + shlex.join([sys.executable, *sys.argv]),
        flush=True,
    )
    print(
        f"MODULE={iron.__file__} CWD={Path.cwd().resolve()} DEVICE={jax.devices()}",
        flush=True,
    )
    assert jax.config.jax_enable_x64, "the material solve requires float64"
    # The shared environment's editable install points at the main checkout;
    # a shadowing PYTHONPATH entry is what makes this tree the one measured.
    assert (
        Path(iron.__file__)
        .resolve()
        .is_relative_to(Path(__file__).resolve().parents[2])
    ), iron.__file__
    assert "P100" in jax.devices()[0].device_kind, jax.devices()[0].device_kind
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
    for segment in machine.iron_segments:
        np.testing.assert_array_equal(segment.material.b_field, table_b)
        np.testing.assert_array_equal(segment.material.permeability_relative, table_mu)
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
    air_drift = float(np.max(np.abs(air / np.asarray(reference["air_field_T"]) - 1)))
    assert air_drift < 1e-12, air_drift
    line_points = np.column_stack((data["radius"], np.zeros_like(data["radius"])))
    known = data["radius"] > data["edge"] + 0.3
    line_air = np.linalg.norm(external(line_points), axis=1)
    control = float(np.max(np.abs(line_air[known] / data["bmag"][0, known] - 1)))
    assert control < 0.002, control
    prior = json.loads(PRIOR_ROOTS.read_text())
    assert prior["probes_m"] == probes.tolist()
    roots = {name: prior["far_field"][name]["magnitude_T"] for name in ("A", "B")}
    record = dict(
        revision=revision,
        measure_py_blob=blob,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        module=iron.__file__,
        cwd=str(Path.cwd().resolve()),
        device=jax.devices()[0].device_kind,
        source=str(SOURCE),
        source_control_max_relative=control,
        machine=uri.removeprefix("imas:hdf5?path="),
        dd_version=dd_version,
        segments=len(machine.iron_segments),
        time_s=data["time"],
        probe_distance_m=DISTANCES.tolist(),
        probes_m=probes.tolist(),
        air_field_T=air.tolist(),
        air_magnitude_T=np.linalg.norm(air, axis=1).tolist(),
        air_field_reproduction_max_relative=air_drift,
        table_b_T=table_b.tolist(),
        table_mu_relative=table_mu.tolist(),
        law=(
            "Anhysteretic and single-valued: the model carries no remanent "
            "magnetisation and no minor-loop permeability."
        ),
        former_roots_at_probes_T=roots,
        former_roots_source=str(PRIOR_ROOTS),
        meshes={},
    )
    save(out / "measurement.json", record)
    production, interaction = memoise_interaction()
    iron._interaction = interaction
    try:
        for size in args.sizes:
            (out / f"{size:.4g}").mkdir(exist_ok=True)
            record["meshes"][f"{size:.4g}"] = measure_size(
                region, external, table_b, table_mu, size, air, probes, out
            )
            save(out / "measurement.json", record)
    finally:
        iron._interaction = production
    coarse, fine = (record["meshes"][f"{size:.4g}"] for size in args.sizes)
    assert coarse["converged"] and fine["converged"]
    coarse_field = np.asarray(coarse["magnitude_T"])
    fine_field = np.asarray(fine["magnitude_T"])
    difference = fine_field - coarse_field
    record["error_bar"] = dict(
        pair=[float(size) for size in args.sizes],
        difference_T=difference.tolist(),
        difference_over_coarse=(difference / coarse_field).tolist(),
        difference_over_air=(difference / np.linalg.norm(air, axis=1)).tolist(),
        cell_ratio=coarse["cells"] / fine["cells"],
        iterations=[coarse["iterations"], fine["iterations"]],
    )
    record["complete"] = True
    save(out / "measurement.json", record)
    print("ERRORBAR " + json.dumps(record["error_bar"]), flush=True)
    print("SUMMARY " + json.dumps(record), flush=True)
    assert Path(iron.__file__).read_bytes() == source_bytes


if __name__ == "__main__":
    main()
