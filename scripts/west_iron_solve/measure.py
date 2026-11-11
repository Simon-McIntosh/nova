"""Measure WEST iron through the production reader and material solve."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import pickle
import shlex
import subprocess
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np
import shapely

from nova.biot import magnetisation as material
from nova.biot.greens import greens_bz_br
from nova.biot.tiledassembly import TilePlan
from nova.imas.machine import StaticMachineDescription
from nova.jax.config import configure_dtypes

MU0 = 4e-7 * np.pi
PLAN = TilePlan(128, 128, 256, 16, 48)
MACHINE = Path("/home/ITER/mcintos/Code/imas-efit/tests/data/imas/WEST/machine/dd4-1-0")


def save(path, record):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_field(data):
    """Reuse the prototype's coil quadrature and prescribed plasma currents."""
    radius, height, amps = [], [], []
    for coil in data["coils"]:
        offsets = (np.arange(6) + 0.5) / 6 - 0.5
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


@contextmanager
def observe_interactions(out, row):
    """Memoise identical production operands without changing their values."""
    original = material._interaction
    cache = {}

    def interaction(cells, points, plan):
        digest = hashlib.sha256(points.tobytes())
        for polygon in cells.polygons:
            digest.update(polygon.tobytes())
        key = (digest.hexdigest(), plan)
        if key not in cache:
            started = perf_counter()
            print(
                f"BUILD targets={len(points)} cells={len(cells.polygons)}", flush=True
            )
            matrix = original(cells, points, plan)
            assert np.isfinite(matrix).all()
            cache[key] = matrix
            row.setdefault("interaction_seconds", []).append(perf_counter() - started)
            print(f"BUILT seconds={row['interaction_seconds'][-1]:.3f}", flush=True)
            if len(points) == len(cells.polygons) and np.array_equal(
                points, cells.centroids
            ):
                np.save(out / "gram.npy", matrix)
                row["gram_path"] = str(out / "gram.npy")
                save(out / "receipt.json", row)
        return cache[key]

    material._interaction = interaction
    try:
        yield
    finally:
        material._interaction = original
        cache.clear()


def picard(gram, external, table_b, table_mu):
    """Freeze secant permeability in each linear subproblem, without Newton."""
    operands = tuple(
        jnp.asarray(x)
        for x in (gram, external.T.reshape(-1), *material._fit_law(table_b, table_mu))
    )
    state = jnp.zeros(external.size, dtype=jnp.float64)
    reference = max(np.linalg.norm(external) / MU0, 1e-30)
    history = []
    for iteration in range(11):
        residual = material._residual(state, operands)
        relative = float(jnp.linalg.norm(residual)) / max(
            float(jnp.linalg.norm(state)), reference
        )
        history.append(relative)
        print(f"PICARD iteration={iteration} residual={relative:.12e}", flush=True)
        if relative <= 1e-8 or iteration == 10:
            break
        total = (operands[1] + operands[0] @ state).reshape(2, -1)
        coefficient = material._coefficient(
            jnp.linalg.norm(total, axis=0), *operands[2:]
        )
        c = jnp.tile(coefficient, 2)
        system = jnp.eye(external.size) - c[:, None] * operands[0]
        state = jla.solve(system, c * operands[1])
    return dict(
        residual_history=history,
        iterations=len(history) - 1,
        converged=history[-1] <= 1e-8,
        definition=(
            "Secant permeability frozen per linear subproblem, refreshed from "
            "the resulting total field; unrelaxed Picard, ten corrections."
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    device = jax.devices()[0]
    assert device.platform == "gpu" and "P100" in device.device_kind, device
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"# revision: {revision} tree: {Path.cwd().resolve()} command: "
        + shlex.join([sys.executable, *sys.argv]),
        flush=True,
    )
    print(
        f"MODULE={material.__file__} CWD={Path.cwd().resolve()} DEVICE={device}",
        flush=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with args.source.open("rb") as stream:
        data = pickle.load(stream)
    machine = StaticMachineDescription.read_iron_core(
        f"imas:hdf5?path={MACHINE}", dd_version="4.1.0"
    )
    assert len(machine.iron_segments) == 6
    table = machine.iron_segments[0].material
    b, mu = np.asarray(table.b_field), np.asarray(table.permeability_relative)
    for segment in machine.iron_segments:
        np.testing.assert_array_equal(segment.material.b_field, b)
        np.testing.assert_array_equal(segment.material.permeability_relative, mu)
    region = shapely.union_all(
        [
            shapely.Polygon(np.column_stack((s.outline.r, s.outline.z)))
            for s in machine.iron_segments
        ]
    )
    assert region.is_valid
    external = source_field(data)
    distances = np.array([1.0, 2.0, 4.0])
    probes = np.column_stack((data["edge"] + distances, np.zeros(3)))
    air = external(probes)
    line_points = np.column_stack((data["radius"], np.zeros_like(data["radius"])))
    line_air = external(line_points)
    known = data["radius"] > data["edge"] + 0.3
    control = float(
        np.max(
            np.abs(np.linalg.norm(line_air, axis=1)[known] / data["bmag"][0, known] - 1)
        )
    )
    assert control < 0.002, control
    common = dict(
        revision=revision,
        module=material.__file__,
        cwd=str(Path.cwd().resolve()),
        device=device.device_kind,
        source=str(args.source),
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        machine=str(MACHINE),
        dd_version="4.1.0",
        segments=len(machine.iron_segments),
        time_s=data["time"],
        source_control_max_relative=control,
        probe_distance_m=distances.tolist(),
        probes_m=probes.tolist(),
        air_field_T=air.tolist(),
        air_magnitude_T=np.linalg.norm(air, axis=1).tolist(),
        table_b_T=b.tolist(),
        table_mu_relative=mu.tolist(),
        law="Anhysteretic, single-valued; no remanence or minor-loop permeability.",
    )
    save(args.output / "inputs.json", common)
    rows = []
    for size in (0.08, 0.04):
        out = args.output / f"{size:.2f}"
        out.mkdir(exist_ok=True)
        row = dict(common, cell_size_m=size, phase="starting")
        save(out / "receipt.json", row)
        try:
            with observe_interactions(out, row):
                started = perf_counter()
                result = material.solve_magnetisation(
                    region,
                    external,
                    b,
                    mu,
                    cell_size=size,
                    max_iterations=10,
                    tile_plan=PLAN,
                )
                cells = result._mesh
                polygons = [shapely.Polygon(p) for p in cells.polygons]
                volume = 2 * np.pi * cells.centroids[:, 0] * cells.areas
                minimum_distance = min(
                    p.centroid.distance(p.boundary) for p in polygons
                )
                total = air + result.field(probes)
                row.update(
                    phase="nonlinear-complete",
                    cells=result.cell_count,
                    converged=result.converged,
                    termination=result.termination,
                    iterations=result.iterations,
                    residual_history=list(result.residual_history),
                    step_scales=list(result.step_scales),
                    mu_range=[
                        float(result.permeability_relative.min()),
                        float(result.permeability_relative.max()),
                    ],
                    fraction_volume_above_mu_100=float(
                        volume[result.permeability_relative > 100].sum() / volume.sum()
                    ),
                    area_m2=float(cells.areas.sum()),
                    area_union_error_m2=float(
                        shapely.union_all(polygons).symmetric_difference(region).area
                    ),
                    minimum_area_fraction=float(cells.areas.min() / size**2),
                    minimum_centroid_edge_distance_m=minimum_distance,
                    volume_m3=float(volume.sum()),
                    field_T=total.tolist(),
                    magnitude_T=np.linalg.norm(total, axis=1).tolist(),
                    iron_to_air=(
                        np.linalg.norm(total, axis=1) / np.linalg.norm(air, axis=1)
                    ).tolist(),
                    solve_seconds=perf_counter() - started,
                )
                save(out / "receipt.json", row)
                with (out / "solution.pkl").open("wb") as stream:
                    pickle.dump(result, stream)
                print(
                    "NEWTON "
                    + json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "cell_size_m",
                                "cells",
                                "converged",
                                "iterations",
                                "residual_history",
                                "mu_range",
                            )
                        }
                    ),
                    flush=True,
                )
                assert result.converged and result.iterations <= 10, row["termination"]
                assert (
                    mu.min() - 1e-8
                    <= result.permeability_relative.min()
                    <= result.permeability_relative.max()
                    <= mu.max() + 1e-8
                )
                flat = material.solve_magnetisation(
                    region,
                    external,
                    b,
                    np.full_like(mu, 215.0),
                    cell_size=size,
                    max_iterations=10,
                    tile_plan=PLAN,
                )
                flat_field = air + flat.field(probes)
                row.update(
                    flat_converged=flat.converged,
                    flat_residual_history=list(flat.residual_history),
                    flat_field_T=flat_field.tolist(),
                    flat_magnitude_T=np.linalg.norm(flat_field, axis=1).tolist(),
                    flat_iron_to_air=(
                        np.linalg.norm(flat_field, axis=1) / np.linalg.norm(air, axis=1)
                    ).tolist(),
                    table_minus_flat_magnitude_T=(
                        np.linalg.norm(total, axis=1)
                        - np.linalg.norm(flat_field, axis=1)
                    ).tolist(),
                )
                save(out / "receipt.json", row)
                assert flat.converged
                if size == 0.08:
                    row["picard"] = picard(
                        np.load(out / "gram.npy"), external(cells.centroids), b, mu
                    )
                    save(out / "receipt.json", row)
                    assert not row["picard"]["converged"], (
                        "Picard negative control unexpectedly converged"
                    )
                row["phase"] = "complete"
                save(out / "receipt.json", row)
                rows.append(row)
        except BaseException as error:
            row.update(phase="failed", error=f"{type(error).__name__}: {error}")
            save(out / "receipt.json", row)
            raise
        jax.clear_caches()
    coarse, fine = rows
    difference = np.asarray(fine["field_T"]) - np.asarray(coarse["field_T"])
    summary = dict(
        rows=rows,
        fine_minus_coarse_field_T=difference.tolist(),
        discretisation_error_vector_norm_T=np.linalg.norm(difference, axis=1).tolist(),
        fine_minus_coarse_magnitude_T=(
            np.asarray(fine["magnitude_T"]) - coarse["magnitude_T"]
        ).tolist(),
        discretisation_relative_to_fine=(
            np.linalg.norm(difference, axis=1) / fine["magnitude_T"]
        ).tolist(),
    )
    save(args.output / "summary.json", summary)
    print(
        "SUMMARY " + json.dumps({k: v for k, v in summary.items() if k != "rows"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
