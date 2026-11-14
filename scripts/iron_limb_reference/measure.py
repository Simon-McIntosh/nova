"""Measure surface refinement, annulus agreement and a missing-limb control.

Run as one CPU scheduler job. Each solved row is persisted before the next
resolution starts. All fields are poloidal components in tesla unless the
key explicitly says cylindrical, in which case the order is R, phi, Z.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import numpy as np  # noqa: E402
from numpy.polynomial.legendre import leggauss  # noqa: E402

from scripts.iron_limb_reference.surface import (  # noqa: E402
    COLUMN_RADIUS,
    MU0,
    SurfaceModel,
    circuit_panels,
)
from scripts.iron_limb_reference.drive import FilamentDrive  # noqa: E402


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def toroidal_field(model, charge, probes, samples=48, full=False):
    extent = 2 * np.pi if full else np.pi / 3
    angle = (np.arange(samples) + 0.5) / samples * extent - np.pi / 6
    radius = np.tile(probes[:, 0], samples)
    height = np.tile(probes[:, 1], samples)
    phi = np.repeat(angle, len(probes))
    points = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), height))
    field = model.iron_field(points, charge)
    cylindrical = np.column_stack(
        (
            field[:, 0] * np.cos(phi) + field[:, 1] * np.sin(phi),
            -field[:, 0] * np.sin(phi) + field[:, 1] * np.cos(phi),
            field[:, 2],
        )
    ).reshape(samples, len(probes), 3)
    return angle, cylindrical


def column_flux(
    model,
    charge,
    permeability,
    bore=0.0,
    radial_order=24,
    angular_order=12,
    vertical_order=4,
):
    """Volume-average axial flux over the same 0 < Z < .2 m slab as Nova."""
    node, weight = leggauss(radial_order)
    squared = bore**2 + (node + 1) / 2 * (COLUMN_RADIUS**2 - bore**2)
    radius = np.sqrt(squared)
    area_weights = weight / 2 * np.pi * (COLUMN_RADIUS**2 - bore**2)
    znode, zweight = leggauss(vertical_order)
    angle = (np.arange(angular_order) + 0.5) * (np.pi / 3) / angular_order
    rr, phi, zz = np.meshgrid(radius, angle, (znode + 1) * 0.1, indexing="ij")
    points = np.column_stack(
        (rr.ravel() * np.cos(phi.ravel()), rr.ravel() * np.sin(phi.ravel()), zz.ravel())
    )
    b = permeability * (model.external(points) + model.iron_field(points, charge))
    axial = b[:, 2].reshape(radial_order, angular_order, vertical_order)
    return float(
        np.sum(axial * area_weights[:, None, None] * (zweight / 2)[None, None, :])
        / angular_order
    )


def metrics_difference(coarse, fine, vessel_count):
    a, b = np.asarray(coarse["iron_average_T"]), np.asarray(fine["iron_average_T"])
    air = np.asarray(fine["air_T"])
    vessel = np.linalg.norm(a[:vessel_count] - b[:vessel_count])
    vessel /= np.linalg.norm(b[:vessel_count])
    outside = np.linalg.norm(
        a[vessel_count:] - b[vessel_count:], axis=1
    ) / np.linalg.norm((b + air)[vessel_count:], axis=1)
    flux = abs(coarse["column_flux_Wb"] / fine["column_flux_Wb"] - 1)
    return dict(
        column_flux_relative=flux,
        vessel_iron_relative=float(vessel),
        outside_total_relative=outside.tolist(),
        worst_relative=float(max(flux, vessel, outside.max())),
        passed=bool(max(flux, vessel, outside.max()) < 0.02),
    )


def solve_resolution(case, size, output, annulus=False, bore=0.0, missing=None):
    started = time.perf_counter()
    panels = circuit_panels(size, annulus=annulus, bore=bore, missing_limb=missing)
    label = "annulus" if annulus else "six-limb" if missing is None else "missing-limb"
    tag = f"{label}-{size:g}"
    print(f"BUILD {tag} panels={len(panels.vertices)}", flush=True)
    external = FilamentDrive(case["sources"])
    model = SurfaceModel(panels, external, rotations=6 if missing is None else 1)
    print(f"ASSEMBLED {tag} seconds={time.perf_counter() - started:.2f}", flush=True)
    probes = np.asarray(case["probes"])
    air_xyz = external(
        np.column_stack((probes[:, 0], np.zeros(len(probes)), probes[:, 1]))
    )
    air = air_xyz[:, [0, 2]]
    record = dict(
        size_m=size,
        panels=len(panels.vertices),
        annulus=annulus,
        bore_m=bore,
        missing_limb=missing,
        arc_segments_per_sector=24,
        boundary_volume_m3=float(
            4
            * np.sum(panels.areas * np.sum(panels.centres * panels.normals, axis=1))
            / (6 if missing is not None else 1)
        ),
        solid_angle_error_range=[
            float(model.conservation_error.min()),
            float(model.conservation_error.max()),
        ],
        normal_drive_quadrature=external.last_quadrature,
        rows=[],
    )
    for mu in [50.0, 200.0]:
        charge, residual = model.solve(mu)
        samples = 144 if missing is not None else 48
        angle, resolved = toroidal_field(
            model, charge, probes, samples, full=missing is not None
        )
        mean = resolved.mean(axis=0)[:, [0, 2]]
        flux = None if missing is not None else column_flux(model, charge, mu, bore)
        row = dict(
            mu=mu,
            residual=residual,
            neutrality=model.last_solve,
            column_flux_Wb=flux,
            iron_average_T=mean.tolist(),
            air_T=air.tolist(),
            angle_rad=angle.tolist(),
            iron_resolved_cylindrical_T=resolved.tolist(),
            iron_resolved_T=resolved[:, :, [0, 2]].tolist(),
        )
        if missing is None:
            # A second angular rule detects aliasing at the closest outside probes.
            _, doubled = toroidal_field(model, charge, probes, 96)
            delta = np.linalg.norm(doubled.mean(0)[:, [0, 2]] - mean, axis=1)
            scale = np.linalg.norm(mean + air, axis=1)
            row["angular_mean_relative"] = (delta / scale).tolist()
            if np.max(delta / scale) > 0.002:
                raise RuntimeError("toroidal quadrature is not resolved to 0.2 percent")
        else:
            outside = resolved[:, case["vessel_count"] :, :]
            diff = outside - np.roll(outside, samples // 6, axis=0)
            row["sixfold_defect_relative"] = float(
                np.linalg.norm(diff) / np.linalg.norm(outside)
            )
        record["rows"].append(row)
        np.save(output / f"{tag}-mu{mu:g}-charge.npy", charge)
        write_json(output / f"{tag}.json", record)
        print(f"SOLVED {tag} mu={mu:g} residual={residual:.3e} flux={flux}", flush=True)
    record["wall_seconds"] = time.perf_counter() - started
    write_json(output / f"{tag}.json", record)
    return record


def reproduce_prototype(case_path):
    """Read both saved states and recompute flux from their constitutive law."""
    result = []
    root = case_path.parent / "bricks"
    for tag in ["0.1333", "0.1-order1"]:
        old = json.loads((root / f"bricks-{tag}.json").read_text())
        centres = np.load(root / f"centres-{tag}.npy")
        size = old["size"]
        core = (np.abs(centres[:, 2] - size / 2) < 1e-9) & (
            np.hypot(centres[:, 0], centres[:, 1]) < COLUMN_RADIUS
        )
        rows = []
        for mu in [50.0, 200.0]:
            moment = np.load(root / f"moment-{tag}-mu{mu:g}.npy").reshape(-1, 3)
            flux = float(6 * size**2 * np.sum(MU0 * mu / (mu - 1) * moment[core, 2]))
            stored = next(r["column_flux_Wb"] for r in old["rows"] if r["mu"] == mu)
            if not np.isclose(flux, stored, rtol=2e-11):
                raise RuntimeError(
                    f"prototype state and stored flux disagree: {flux}, {stored}"
                )
            rows.append(dict(mu=mu, recomputed_flux_Wb=flux, recorded_flux_Wb=stored))
        result.append(
            dict(
                size_m=size,
                represented_column_area_m2=float(6 * size**2 * core.sum()),
                rows=rows,
            )
        )
    print("PROTOTYPE_REPRODUCTION " + json.dumps(result), flush=True)
    return result


def axisymmetric_column_flux(model, state, permeability, slab=0.2):
    """Integrate the projected linear field over the exactly clipped slab.

    Each solved magnetisation is a constant cell coefficient. Its linear
    constitutive flux is integrated with the exact 2*pi*R polygon volume;
    neither centroid selection nor an assumed integer number of cells enters.
    """
    import shapely

    material = shapely.box(0.0, 0.0, COLUMN_RADIUS, slab)
    weights = []
    for vertices in model.cells.polygons:
        clipped = shapely.Polygon(vertices).intersection(material)
        weights.append(
            0.0
            if clipped.is_empty
            else 2 * np.pi * clipped.area * clipped.centroid.x / slab
        )
    vertical = np.asarray(state).reshape(2, -1)[1]
    return float(np.dot(weights, MU0 * permeability / (permeability - 1) * vertical))


def axisymmetric(case, sizes, output):
    from scripts.prototypes.iron_limb_fill.axisymmetric import (
        Model,
        axisymmetric_drive,
        linear_regions,
    )

    region = linear_regions()[0]
    records = []
    probes = np.asarray(case["probes"])
    external = axisymmetric_drive(case["sources"])
    for size in sizes:
        model = Model(region, size, external, lambda c: np.ones(len(c)))
        rows = []
        for mu in [50.0, 200.0]:
            state = model.linear(mu)
            row = dict(
                mu=mu,
                column_flux_Wb=axisymmetric_column_flux(model, state, mu),
                iron_average_T=model.probe(state, probes).tolist(),
                air_T=external(probes).tolist(),
            )
            rows.append(row)
            print(
                f"AXISYMMETRIC size={size} mu={mu} flux={row['column_flux_Wb']}",
                flush=True,
            )
        records.append(dict(size_m=size, cells=len(model.cells.centroids), rows=rows))
        write_json(output / "axisymmetric.json", records)
        del model
        jax.clear_caches()
        if len(records) >= 2:
            changes = [
                metrics_difference(a, b, case["vessel_count"])
                for a, b in zip(records[-2]["rows"], records[-1]["rows"])
            ]
            records[-1]["refinement"] = changes
            write_json(output / "axisymmetric.json", records)
            print("AXISYMMETRIC_REFINEMENT " + json.dumps(changes), flush=True)
            if all(c["passed"] for c in changes):
                break
    return records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sizes",
        type=float,
        nargs="+",
        default=[0.5, 0.35, 0.25, 0.18, 0.13, 0.1, 0.075],
    )
    parser.add_argument(
        "--axisymmetric-sizes", type=float, nargs="+", default=[0.18, 0.13, 0.1, 0.075]
    )
    parser.add_argument("--sphere-receipt", type=Path, required=True)
    args = parser.parse_args()
    prerequisite = json.loads(args.sphere_receipt.read_text())
    from scripts.iron_limb_reference import surface

    if (
        not prerequisite.get("passed")
        or prerequisite.get("solver_sha256")
        != hashlib.sha256(Path(surface.__file__).read_bytes()).hexdigest()
    ):
        raise RuntimeError("sphere convergence must pass on the same surface solver")
    print("SPHERE_PREREQUISITE passed at both permeabilities", flush=True)
    assert jax.config.jax_enable_x64 is True
    args.output.mkdir(parents=True, exist_ok=True)
    case_bytes = args.case.read_bytes()
    case = json.loads(case_bytes)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = dict(
        revision=revision,
        module=__file__,
        cwd=str(Path.cwd()),
        host=socket.gethostname(),
        jax_devices=str(jax.devices()),
        case_sha256=hashlib.sha256(case_bytes).hexdigest(),
        probes=case["probes"],
        vessel_count=case["vessel_count"],
        prototype_reproduction=reproduce_prototype(args.case),
        reference=[],
        reference_differences=[],
        annulus=[],
        annulus_differences=[],
        axisymmetric=[],
        symmetry=[],
    )
    summary = args.output / "summary.json"
    write_json(summary, record)
    for size in args.sizes:
        row = solve_resolution(case, size, args.output)
        record["reference"].append(row)
        if len(record["reference"]) >= 2:
            prev = record["reference"][-2]
            changes = [
                metrics_difference(a, b, case["vessel_count"])
                for a, b in zip(prev["rows"], row["rows"])
            ]
            record["reference_differences"].append(
                dict(coarse_m=prev["size_m"], fine_m=size, changes=changes)
            )
            print(
                "REFINEMENT " + json.dumps(record["reference_differences"][-1]),
                flush=True,
            )
        write_json(summary, record)
        if len(record["reference"]) >= 3 and all(c["passed"] for c in changes):
            break
    record["axisymmetric"] = axisymmetric(case, args.axisymmetric_sizes, args.output)
    write_json(summary, record)
    # The complete annulus shares the production bore and the same 3D algorithm.
    for size in args.sizes:
        row = solve_resolution(case, size, args.output, annulus=True, bore=0.05)
        record["annulus"].append(row)
        comparison = [
            metrics_difference(a, b, case["vessel_count"])
            for a, b in zip(record["axisymmetric"][-1]["rows"], row["rows"])
        ]
        record["annulus_differences"].append(dict(size_m=size, changes=comparison))
        print(
            "ANNULUS_COMPARISON " + json.dumps(record["annulus_differences"][-1]),
            flush=True,
        )
        write_json(summary, record)
        if all(c["passed"] for c in comparison):
            break
    record["symmetry"] = solve_resolution(case, 0.5, args.output, missing=0)
    final_ref = record["reference_differences"][-1]["changes"]
    final_annulus = record["annulus_differences"][-1]["changes"]
    symmetry = [r["sixfold_defect_relative"] for r in record["symmetry"]["rows"]]
    record["passed"] = bool(
        all(c["passed"] for c in final_ref + final_annulus) and min(symmetry) > 0.001
    )
    write_json(summary, record)
    print(
        "GATE_SUMMARY "
        + json.dumps(
            dict(
                passed=record["passed"],
                reference_worst_relative=[c["worst_relative"] for c in final_ref],
                annulus_worst_relative=[c["worst_relative"] for c in final_annulus],
                missing_limb_sixfold_defect=symmetry,
            )
        ),
        flush=True,
    )
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
