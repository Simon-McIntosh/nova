"""Production material comparisons against the accepted six-limb surface mean."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402
from scripts.iron_limb_production.outline import measure as measure_outline  # noqa: E402
from scripts.prototypes.iron_limb_fill.axisymmetric import (  # noqa: E402
    ARM_Z,
    LIMB_R,
    axisymmetric_drive,
    limb_demagnetising,
    linear_fill,
    linear_regions,
)

EVIDENCE = Path("/work/projects/imas_gpu/sophelio/evidence/iron-limb-fill")
REFERENCE = EVIDENCE / "reference/six-limb-extended-20261114/six-limb-0.075.json"
SECTION_WIDTH = 0.8
SECTION_DEPTH = 1.2
GEOMETRY_FACTOR = SECTION_WIDTH / (SECTION_WIDTH + SECTION_DEPTH)


def reference_inputs():
    """Read the case and finest accepted reference, retaining nonzero controls."""
    case = json.loads((EVIDENCE / "case.json").read_text())
    reference = json.loads(REFERENCE.read_text())
    assert reference["size_m"] == 0.075 and not reference["annulus"]
    rows = {row["mu"]: row for row in reference["rows"]}
    for row in rows.values():
        field = np.asarray(row["iron_average_T"])
        assert field.shape == (len(case["probes"]), 2)
        assert np.isfinite(field).all() and np.linalg.norm(field[:16]) > 1e-3
        assert np.ptp(field[:16, 1]) > 1e-3
    return case, rows


def control_result(case, name, size, mu, factor=GEOMETRY_FACTOR):
    """Solve one control through the public production material interface."""
    flat, smear, _ = linear_regions()
    filled = name in ("geometry", "fitted", "plain")
    return iron.solve_magnetisation(
        smear if name == "smear" else flat,
        axisymmetric_drive(case["sources"]),
        [0, 100],
        [mu, mu],
        cell_size=size,
        fill_fraction=linear_fill if filled else 1.0,
        cross_field_factor=(
            (lambda c: limb_demagnetising(c, ARM_Z[0], LIMB_R[0], factor))
            if name in ("geometry", "fitted")
            else 0.0
        ),
        max_iterations=2,
    )


@contextmanager
def reuse_operators():
    """Cache the current mesh's unchanged assembly and drive across material arms."""
    original = iron._galerkin_interaction
    average = iron._average_external
    cache, drives = {}, {}

    def key(cells):
        return tuple(p.tobytes() for p in cells.polygons)

    def interaction(cells, plan):
        identity = (key(cells), plan)
        if identity not in cache:
            cache.clear()
            cache[identity] = original(cells, plan)
        return cache[identity]

    def external(cells, callback):
        identity = key(cells)
        if identity not in drives:
            drives.clear()
            drives[identity] = average(cells, callback)
        return drives[identity]

    iron._galerkin_interaction, iron._average_external = interaction, external
    try:
        yield
    finally:
        iron._galerkin_interaction, iron._average_external = original, average


def write(path, record):
    """Persist each completed measurement before starting the next arm."""
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def outside_probe_groups(probes):
    """Separate outside probes by their physical position in the control circuit."""
    points = np.asarray(probes)
    outside = np.arange(len(points)) >= 16
    midplane = points[:, 1] == 0
    groups = dict(
        midplane=np.flatnonzero(outside & midplane & (points[:, 0] > LIMB_R[1])),
        inner_midplane=np.flatnonzero(outside & midplane & (points[:, 0] < LIMB_R[0])),
        above_arm=np.flatnonzero(outside & (points[:, 1] > ARM_Z[1])),
        corner=np.flatnonzero(outside & ~midplane & (points[:, 1] <= ARM_Z[1])),
    )
    assert [len(v) for v in groups.values()] == [5, 1, 5, 1]
    assert sorted(np.concatenate(list(groups.values()))) == list(range(16, len(points)))
    return {name: indices.tolist() for name, indices in groups.items()}


def measure_control(record, output):
    case, references = reference_inputs()
    probes = np.asarray(case["probes"])
    air = axisymmetric_drive(case["sources"])(probes)
    record["reference_path"] = str(REFERENCE)
    record["reference_sha256"] = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    record["case_sha256"] = hashlib.sha256(
        (EVIDENCE / "case.json").read_bytes()
    ).hexdigest()
    record["control"] = []
    groups = record["probe_groups"] = outside_probe_groups(probes)
    fitted_factor = None
    for size in (0.05, 0.1):
        # Flat-outline arms share one assembly; the smear is solved last.
        with reuse_operators():
            for name in ("solid", "plain", "geometry", "fitted", "smear"):
                for mu in (50.0, 200.0):
                    reference = np.asarray(references[mu]["iron_average_T"])

                    def objective(factor):
                        errors = []
                        for permeability in (50.0, 200.0):
                            solution = control_result(
                                case, "fitted", size, permeability, factor
                            )
                            target = np.asarray(
                                references[permeability]["iron_average_T"]
                            )[:16]
                            errors.append(
                                np.linalg.norm(solution.field(probes[:16]) - target)
                                / np.linalg.norm(target)
                            )
                        return float(np.linalg.norm(errors))

                    if name == "fitted" and fitted_factor is None:
                        fit = minimize_scalar(
                            objective,
                            bounds=(0.0, 1.0),
                            method="bounded",
                            options={"xatol": 0.005},
                        )
                        assert fit.success
                        fitted_factor = float(fit.x)
                        record["fit"] = dict(
                            factor=fitted_factor,
                            size_m=size,
                            permeabilities=[50.0, 200.0],
                            objective="norm of both vessel relative rms errors",
                            error=float(fit.fun),
                        )
                    fitted = fitted_factor if name == "fitted" else None
                    started = perf_counter()
                    result = control_result(
                        case,
                        name,
                        size,
                        mu,
                        fitted if fitted is not None else GEOMETRY_FACTOR,
                    )
                    field = result.field(probes)
                    error = float(
                        np.linalg.norm(field[:16] - reference[:16])
                        / np.linalg.norm(reference[:16])
                    )
                    ratios = np.linalg.norm(air + field, axis=1) / np.linalg.norm(
                        air + reference, axis=1
                    )
                    row = dict(
                        model=name,
                        size_m=size,
                        mu=mu,
                        device=record["device"],
                        cells=result.cell_count,
                        cross_field_factor=fitted
                        if fitted is not None
                        else GEOMETRY_FACTOR
                        if name == "geometry"
                        else 0.0,
                        converged=result.converged,
                        iterations=result.iterations,
                        residual_history=result.residual_history,
                        vessel_relative_rms=error,
                        iron_T=field.tolist(),
                        **{
                            f"{name}_ratios": ratios[indices].tolist()
                            for name, indices in groups.items()
                        },
                        seconds=perf_counter() - started,
                    )
                    record["control"].append(row)
                    write(output / "summary.json", record)
                    print(
                        "CONTROL "
                        + json.dumps({k: v for k, v in row.items() if k != "iron_T"}),
                        flush=True,
                    )
                    assert result.converged
        jax.clear_caches()
    for row in record["control"]:
        if row["model"] == "geometry":
            smear = next(
                r
                for r in record["control"]
                if r["model"] == "smear"
                and r["size_m"] == row["size_m"]
                and r["mu"] == row["mu"]
            )
            row["smear_error_ratio"] = (
                row["vessel_relative_rms"] / smear["vessel_relative_rms"]
            )
            row["passed"] = (
                row["vessel_relative_rms"] < 0.1 and row["smear_error_ratio"] <= 1 / 3
            )
        elif row["model"] == "solid":
            row["passed"] = row["vessel_relative_rms"] > 1
    record["control_passed"] = all(r.get("passed", True) for r in record["control"])
    write(output / "summary.json", record)


def measure_west(record, output, sizes=(0.08, 0.04), max_iterations=10):
    """Solve WEST 62751 on the stored outline and on the true one read."""
    record["sizes_m"] = sorted(sizes, reverse=True)
    record["max_iterations"] = max_iterations
    measure_outline(record, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    device = jax.devices()[0]
    assert device.platform == "gpu", device
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip(), "production measurement requires a clean committed tree"
    assert Path(iron.__file__).resolve().is_relative_to(Path.cwd().resolve())
    record = dict(
        revision=revision,
        module=iron.__file__,
        cwd=str(Path.cwd().resolve()),
        device=device.device_kind,
        geometry_factor=GEOMETRY_FACTOR,
        section_width_m=SECTION_WIDTH,
        section_depth_m=SECTION_DEPTH,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "summary.json", record)
    print("PROVENANCE " + json.dumps(record), flush=True)
    measure_control(record, args.output)
    measure_west(record, args.output)
    record["passed"] = record["control_passed"] and record["west_passed"]
    write(args.output / "summary.json", record)
    print(
        f"SUMMARY passed={record['passed']} "
        f"control_passed={record['control_passed']} "
        f"west_passed={record['west_passed']}",
        flush=True,
    )
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
