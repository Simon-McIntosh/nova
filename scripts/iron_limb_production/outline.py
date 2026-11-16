"""Solve WEST 62751 on the stored and the true iron outline.

The true model is read once from the machine description
(:meth:`StaticMachineDescription.read_west_iron_outline`); its arm and limb
regions carry the fill fraction the stored section requires, with the
section-derived cross-field factor. The stored model is the stored tapered
segments with unit fill fraction and no cross-field factor.

Both are solved on the production material interface at the mesh pair this
evidence reports. The in-vessel ring, behind-the-limbs and above-the-arms
probes are compared with each model's own mesh-pair difference beside the
model difference.

Run as ``python -m scripts.iron_limb_production.outline --output <dir>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from pathlib import Path
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402
from nova.imas.machine import StaticMachineDescription  # noqa: E402
from scripts.west_iron_galerkin_pair.measure import (  # noqa: E402
    SOURCE,
    SOURCE_SHA256,
)
from scripts.west_iron_solve.measure import MACHINE, source_field  # noqa: E402

EVIDENCE = Path("/work/projects/imas_gpu/sophelio/evidence/iron-limb-fill")
DD_VERSION = "4.1.0"
SIZES = (0.08, 0.04)
# The superseded prototype flat arm, stated so a reader can see the section
# difference it carried: a 6.0 m long box at the 1.2035 m limb depth.
PROTOTYPE_ARM = dict(inner_radius=0.05, outer_radius=6.0, depth=1.2035)


def save(path, record):
    """Persist a completed measurement before the next arm starts."""
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def outline():
    """Read the true iron outline once from the machine description."""
    assert MACHINE.is_dir(), MACHINE
    return StaticMachineDescription.read_west_iron_outline(
        f"imas:hdf5?path={MACHINE}", dd_version=DD_VERSION
    )


def material(source):
    """Return the stored table's ``B`` and secant relative permeability."""
    table = source.description.iron_segments[0].material
    b_field = np.asarray(table.b_field)
    permeability = np.asarray(table.permeability_relative)
    for segment in source.description.iron_segments:
        np.testing.assert_array_equal(segment.material.b_field, b_field)
        np.testing.assert_array_equal(
            segment.material.permeability_relative, permeability
        )
    return b_field, permeability


def region(contours):
    """Return the union of closed coordinate contours as a solved region."""
    polygons = [shapely.Polygon(np.asarray(item, dtype=float)) for item in contours]
    union = shapely.union_all(polygons)
    assert union.is_valid
    return union


def models(source):
    """Return the two solved models: the stored outline and the true outline."""
    stored = [
        tuple(zip(contour.r, contour.z, strict=True)) for contour in source.stored
    ]
    return {
        "true": dict(
            region=region(source.outlines),
            fill_fraction=source.fill,
            cross_field_factor=source.cross_field,
        ),
        "stored": dict(
            region=region(stored),
            fill_fraction=1.0,
            cross_field_factor=0.0,
        ),
    }


def probe_groups():
    """Return the in-vessel, behind-the-limb and above-the-arm probe groups.

    The in-vessel ring sits at 0.9 of the 2.45 m by 0.675 m vessel-wall
    ellipse. Behind the limbs is outboard of the 6.0 m limb face across the
    limb's own height, and above the arms is above the upper arm top at
    Z = 3.6 m. Every outboard group is mirrored about the midplane.
    """
    angle = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ring = np.column_stack((2.45 + 0.675 * np.cos(angle), 0.675 * np.sin(angle)))
    outboard = np.array([2.5, 3.0, 3.5])
    above = np.array([3.7, 3.9, 4.2])
    limb = np.column_stack(
        (
            np.repeat([6.05, 6.5, 7.0], 2 * len(outboard)),
            np.tile(np.concatenate((outboard, -outboard)), 3),
        )
    )
    arm = np.column_stack(
        (
            np.repeat([0.6, 1.4, 2.2, 3.0], 2 * len(above)),
            np.tile(np.concatenate((above, -above)), 4),
        )
    )
    return dict(in_vessel=ring, behind_limbs=limb, above_arms=arm)


def section_accounting(source):
    """Return the arm section per radian of each geometry.

    The true arm reproduces the stored arm's section by construction; the
    superseded prototype flat arm carried the larger section that this node
    measures away from.
    """
    true_arm = sum(
        item.section_per_radian()
        for item in source.regions
        if item.kind == "arm"
    )
    stored_arm = sum(
        section.stored
        for section, item in zip(source.sections, source.regions, strict=True)
        if item.kind == "arm"
    )
    prototype = (
        PROTOTYPE_ARM["depth"]
        * (
            PROTOTYPE_ARM["outer_radius"] ** 2
            - PROTOTYPE_ARM["inner_radius"] ** 2
        )
        / 2
    )
    return dict(
        true_arm=float(true_arm),
        stored_arm=float(stored_arm),
        prototype_flat_arm=float(prototype),
        prototype_minus_stored=float(prototype - stored_arm),
    )


def solve(model, external, b_field, permeability, size, budget, tolerance):
    """Solve one model through the public production material interface."""
    started = perf_counter()
    result = iron.solve_magnetisation(
        model["region"],
        external,
        b_field,
        permeability,
        cell_size=size,
        fill_fraction=model["fill_fraction"],
        cross_field_factor=model["cross_field_factor"],
        max_iterations=budget,
        tolerance=tolerance,
    )
    return result, perf_counter() - started


def group_fields(field, groups):
    """Split one (points, 2) field into the named probe groups."""
    lengths = [len(points) for points in groups.values()]
    edges = np.cumsum(lengths)[:-1]
    parts = np.split(np.arange(len(field)), edges)
    return {name: field[index] for name, index in zip(groups, parts, strict=True)}


def comparisons(fields, sizes, groups):
    """Return per-group model differences beside the mesh-pair differences.

    A model difference no larger than the larger of the two models' mesh-pair
    differences is reported as not resolved.
    """
    rows = []
    for size in sizes:
        coarser = [other for other in sizes if other > size]
        if not coarser:
            continue
        for name in (*groups, "all"):
            def access(values, key=name):
                if key == "all":
                    return np.concatenate([values[key] for key in groups])
                return values[key]
            fine = {key: access(fields[key][size]) for key in fields}
            coarse = {key: access(fields[key][coarser[0]]) for key in fields}
            model = float(
                np.linalg.norm(fine["true"] - fine["stored"])
                / max(float(np.linalg.norm(fine["stored"])), 1e-30)
            )
            mesh = max(
                float(
                    np.linalg.norm(fine[key] - coarse[key])
                    / max(float(np.linalg.norm(fine[key])), 1e-30)
                )
                for key in fields
            )
            rows.append(
                dict(
                    group=name,
                    size_m=size,
                    mesh_pair_size_m=coarser[0],
                    model_difference=model,
                    mesh_pair_difference=mesh,
                    resolved=bool(model > mesh),
                )
            )
    return rows


def compare_only(directory):
    """Recompute one finished run's comparison rows from its stored fields."""
    directory = Path(directory)
    record = json.loads((directory / "summary.json").read_text())
    groups = {
        name: np.asarray(points) for name, points in record["probe_groups"].items()
    }
    sizes = record["sizes_m"]
    fields = {
        key: {
            size: group_fields(np.load(directory / f"west-{key}-{size:g}.npz")["field"], groups)
            for size in sizes
        }
        for key in ("true", "stored")
    }
    record["west_comparison"] = comparisons(fields, sizes, groups)
    (directory / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    return record["west_comparison"]


def fill_one_arm(record, output, source, external, b_field, permeability, groups, fields, points):
    """Evaluate the true model with fill fraction 1 at the coarsest mesh.

    This is the declared negative control: unit fill makes the flat model
    solid, so its in-vessel field sits nearer the stored solid arm. The
    displacement it moves and the true model's own mesh-pair difference are
    recorded side by side.
    """
    coarse = max(record["sizes_m"])
    model = dict(models(source)["true"], fill_fraction=1.0)
    solution, seconds = solve(
        model,
        external,
        b_field,
        permeability,
        coarse,
        record["max_iterations"],
        record["tolerance"],
    )
    field = solution.field(points)
    true = fields["true"][coarse]
    in_vessel = slice(0, len(groups["in_vessel"]))
    displacement = float(
        np.linalg.norm(field[in_vessel] - true[in_vessel])
        / max(float(np.linalg.norm(true[in_vessel])), 1e-30)
    )
    mesh = float(
        np.linalg.norm(true[in_vessel] - fields["true"][min(record["sizes_m"])][in_vessel])
        / max(float(np.linalg.norm(fields["true"][min(record["sizes_m"])][in_vessel])), 1e-30)
    ) if len(record["sizes_m"]) > 1 else float("nan")
    row = dict(
        size_m=coarse,
        fill_fraction=1.0,
        cells=solution.cell_count,
        converged=solution.converged,
        iterations=solution.iterations,
        residual_history=solution.residual_history,
        seconds=seconds,
        in_vessel_displacement=displacement,
        mesh_pair_difference=mesh,
        exceeds_mesh_pair=bool(displacement > mesh),
    )
    np.savez(output / "west-true-fill-one.npz", points=points, field=field)
    print("MUTATION " + json.dumps(row), flush=True)
    del solution
    jax.clear_caches()
    return row


def measure(record, output):
    """Solve both models at every size and record the probe comparisons."""
    assert SOURCE.is_file(), SOURCE
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    with SOURCE.open("rb") as stream:
        data = pickle.load(stream)
    external = source_field(data)
    source = outline()
    b_field, permeability = material(source)
    record["outline"] = dict(
        source=str(MACHINE),
        dd_version=DD_VERSION,
        gap_input=source.gap_input,
        limb_arm_gap_m=source.limb_arm_gap,
        cross_field_factor=source.cross_field_factor,
        limb_width_m=source.limb_width,
        limb_depth_m=source.limb_depth,
        regions=[item.name for item in source.regions],
        sections={
            section.region: dict(stored=section.stored, filled=section.filled)
            for section in source.sections
        },
        maximum_relative_section_error=source.maximum_relative_section_error(
            ("arm", "limb")
        ),
        arm_section_per_radian=section_accounting(source),
    )
    groups = probe_groups()
    record["probe_groups"] = {name: points.tolist() for name, points in groups.items()}
    offsets, start = {}, 0
    for name, points in groups.items():
        offsets[name] = start
        start += len(points)
    record["group_offsets"] = offsets
    record["west"] = []
    points = np.concatenate(list(groups.values()))
    fields = {key: {} for key in ("true", "stored")}
    for key, model in models(source).items():
        for size in record["sizes_m"]:
            solution, seconds = solve(
                model,
                external,
                b_field,
                permeability,
                size,
                record["max_iterations"],
                record["tolerance"],
            )
            field = solution.field(points)
            fields[key][size] = field
            row = dict(
                model=key,
                size_m=size,
                cells=solution.cell_count,
                converged=solution.converged,
                iterations=solution.iterations,
                residual_history=solution.residual_history,
                seconds=seconds,
                device=record["device"],
            )
            record["west"].append(row)
            save(output / "summary.json", record)
            np.savez(
                output / f"west-{key}-{size:g}.npz",
                magnetisation=solution.magnetisation,
                permeability_relative=solution.permeability_relative,
                fill_fraction=solution.fill_fraction,
                cross_field_factor=solution.cross_field_factor,
                centroids=solution._mesh.centroids,
                areas=solution._mesh.areas,
                points=points,
                field=field,
            )
            with (output / f"west-{key}-{size:g}-polygons.pkl").open("wb") as stream:
                pickle.dump(solution._mesh.polygons, stream)
            print("WEST " + json.dumps(row), flush=True)
            del solution
            jax.clear_caches()
    record["west_comparison"] = comparisons(
        {key: {size: group_fields(value, groups) for size, value in rows.items()}
         for key, rows in fields.items()},
        record["sizes_m"],
        groups,
    )
    record["west_passed"] = all(
        row["converged"]
        and row["iterations"] <= record["max_iterations"]
        and row["residual_history"][-1] <= record["tolerance"]
        for row in record["west"]
    )
    record["mutation"] = fill_one_arm(
        record, output, source, external, b_field, permeability, groups, fields, points
    )
    save(output / "summary.json", record)
    for row in record["west_comparison"]:
        print("COMPARE " + json.dumps(row, sort_keys=True), flush=True)


def fill_one_displacement(directory):
    """Return the unit-fill displacement of the in-vessel field.

    ``directory`` holds the measurement: the true model's fields at both
    sizes and the unit-fill arm at the coarse size. The displacement the
    declared mutation moves is compared with the true model's own mesh-pair
    difference at the same probes.
    """
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    sizes = summary["sizes_m"]
    coarse, fine = max(sizes), min(sizes)
    ring = len(summary["probe_groups"]["in_vessel"])
    index = slice(0, ring)
    true_coarse = np.load(directory / f"west-true-{coarse:g}.npz")["field"][index]
    true_fine = np.load(directory / f"west-true-{fine:g}.npz")["field"][index]
    mutation = np.load(directory / "west-true-fill-one.npz")["field"][index]
    displacement = float(
        np.linalg.norm(mutation - true_coarse)
        / max(float(np.linalg.norm(true_coarse)), 1e-30)
    )
    mesh = float(
        np.linalg.norm(true_coarse - true_fine)
        / max(float(np.linalg.norm(true_fine)), 1e-30)
    )
    return dict(
        displacement=displacement,
        mesh_pair_difference=mesh,
        exceeds=bool(displacement > mesh),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=float, default=list(SIZES))
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    device = jax.devices()[0]
    assert device.platform == "gpu", device
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    assert not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip(), "the outline measurement requires a clean committed tree"
    module = Path(iron.__file__).resolve()
    assert module.is_relative_to(Path.cwd().resolve()), module
    with SOURCE.open("rb") as stream:
        time = float(pickle.load(stream)["time"])
    record = dict(
        revision=revision,
        module=str(module),
        cwd=str(Path.cwd().resolve()),
        device=device.device_kind,
        purpose="WEST 62751 stored against true iron outline",
        sizes_m=sorted(args.sizes, reverse=True),
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        west_source_sha256=SOURCE_SHA256,
        west_source=str(SOURCE),
        west_time_s=time,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / "summary.json", record)
    print("PROVENANCE " + json.dumps(record), flush=True)
    measure(record, args.output)
    print(f"SUMMARY west_passed={record['west_passed']}", flush=True)
    raise SystemExit(0 if record["west_passed"] else 1)


if __name__ == "__main__":
    main()
