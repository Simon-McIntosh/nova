"""Locate the analytic limited boundary on every edge of the saved wall."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import jax
import numpy as np
from numpy.polynomial import Polynomial

from scripts.analytic_oracle_fixtures import measure as fixture
from tests import rotating_equilibrium_references as references


def flux(case, points):
    """Evaluate total poloidal flux in Wb, including the per-radian conversion."""
    points = np.asarray(points)
    return fixture.TOTAL_FLUX_FACTOR * case.flux(points[..., 0], points[..., 1])


def sample_wall(wall, spacing):
    """Sample every edge including its interior, with a maximum physical gap."""
    edges = np.roll(wall, -1, axis=0) - wall
    lengths = np.linalg.norm(edges, axis=1)
    counts = np.ceil(lengths / spacing).astype(int)
    starts = np.r_[0.0, np.cumsum(lengths)]
    points, arcs = [], []
    for start, edge, length, count, arc in zip(
        wall, edges, lengths, counts, starts[:-1], strict=True
    ):
        parameter = np.arange(count) / count
        points.append(start + parameter[:, None] * edge)
        arcs.append(arc + parameter * length)
    return np.concatenate(points), np.concatenate(arcs), float(max(lengths / counts))


def edge_extrema(case, wall):
    """Enumerate all stationary points of the static quartic on every edge.

    The restricted flux is psi_a - k_p (R(t)^2-R_0^2)^2/2 - k_z Z(t)^2;
    its derivative is cubic. Endpoints and every real interior root exhaust
    the extrema. Values are evaluated with case.flux, independently of reads.
    """
    assert case.rotation_parameter == 0.0
    candidates = list(wall)
    for start, end in zip(wall, np.roll(wall, -1, axis=0), strict=True):
        radius, height = [Polynomial([a, b - a]) for a, b in zip(start, end)]
        polynomial = (
            case.axis_flux
            - case.pressure_coefficient / 2 * (radius**2 - case.major_radius**2) ** 2
            - case.field_coefficient * height**2
        )
        for root in polynomial.deriv().roots():
            if abs(root.imag) < 1e-12 and 0 < root.real < 1:
                candidates.append(start + root.real * (end - start))
    candidates = np.asarray(candidates)
    return candidates[np.argmax(flux(case, candidates))], len(candidates)


def position(wall, point, case, designed_flux):
    """Project a point to the wall to report its edge, arc, and distance."""
    edges = np.roll(wall, -1, axis=0) - wall
    lengths = np.linalg.norm(edges, axis=1)
    fraction = np.clip(np.sum((point - wall) * edges, axis=1) / lengths**2, 0, 1)
    projected = wall + fraction[:, None] * edges
    distances = np.linalg.norm(projected - point, axis=1)
    edge = int(np.argmin(distances))
    value = float(flux(case, point))
    return dict(
        rz_m=point.tolist(),
        flux_wb=value,
        margin_over_designed_wb=value - designed_flux,
        edge=edge,
        edge_fraction=float(fraction[edge]),
        arc_m=float(np.sum(lengths[:edge]) + fraction[edge] * lengths[edge]),
        distance_to_wall_m=float(distances[edge]),
    )


def measure(inputs):
    case = references.reference_cases()["weak-rotation-reactor"].static_limit()
    generated_wall = fixture.limiter_contour(case, points=fixture.WALL_POINT_COUNT)
    report = dict(
        case=case.name, flux_convention="total poloidal flux = 2*pi*case.flux"
    )
    receipt_path = inputs / "limited-oracle.json"
    receipt = json.loads(receipt_path.read_text())
    report["recorded_clause_receipt"] = str(receipt_path)
    report["recorded_clause_sha256"] = hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    report["rows"] = []
    for cells in (550, 2000):
        path = inputs / f"limited-{cells}-oracle.npz"
        with np.load(path, allow_pickle=False) as archive:
            wall = archive["wall"]
            saved_oracle = archive["analytic_roots"][1]
            route = archive["solved_roots"][1]
        np.testing.assert_allclose(wall, generated_wall, rtol=0, atol=2e-13)
        designed = fixture.designed_contact(wall)
        designed_flux = float(flux(case, designed))
        extremum, candidate_count = edge_extrema(case, wall)
        points, arcs, gap = sample_wall(wall, 0.001)
        values = flux(case, points)
        fine, _, fine_gap = sample_wall(wall, 0.0005)
        maximum = float(flux(case, extremum))
        np.testing.assert_allclose(extremum, designed, rtol=0, atol=1e-10)
        assert np.ptp(values) > 1.0, "the wall scan must resolve nonconstant flux"
        assert abs(flux(case, case.magnetic_axis) - 2 * np.pi * case.axis_flux) < 1e-12
        assert abs(maximum) < 1e-11
        assert maximum - max(values) < 1e-6
        assert maximum - max(flux(case, fine)) < 1e-6
        entries = {
            name: position(wall, point, case, designed_flux)
            for name, point in dict(
                extremum=extremum,
                designed_contact=designed,
                saved_oracle=saved_oracle,
                route=route,
            ).items()
        }
        assert all(entry["distance_to_wall_m"] < 1e-10 for entry in entries.values())
        np.testing.assert_allclose(
            saved_oracle, wall[np.argmax(flux(case, wall))], atol=1e-12
        )
        assert entries["saved_oracle"]["flux_wb"] < -0.049
        perimeter = float(
            np.linalg.norm(np.roll(wall, -1, axis=0) - wall, axis=1).sum()
        )
        for entry in entries.values():
            entry["arc_from_contact_m"] = (
                entry["arc_m"] - entries["extremum"]["arc_m"] + perimeter / 2
            ) % perimeter - perimeter / 2
        row = dict(
            cells=cells,
            source=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            wall_vertices=len(wall),
            generated_wall_max_difference_m=float(abs(wall - generated_wall).max()),
            perimeter_m=perimeter,
            samples=len(points),
            maximum_spacing_m=gap,
            refined_samples=len(fine),
            refined_spacing_m=fine_gap,
            stationary_candidates_including_vertices=candidate_count,
            scan_min_flux_wb=float(min(values)),
            scan_max_flux_wb=float(max(values)),
            refined_max_flux_wb=float(max(flux(case, fine))),
            axis_flux_wb=float(2 * np.pi * case.axis_flux),
            analytic_route_gap_from_saved_oracle_wb=float(
                flux(case, route) - flux(case, saved_oracle)
            ),
            **entries,
        )
        recorded = next(
            item["oracle"] for item in receipt["rows"] if item["cells"] == cells
        )
        np.testing.assert_allclose(
            recorded["boundary_point"], saved_oracle, rtol=0, atol=1e-12
        )
        assert 9.8e-4 < recorded["boundary_ratio"] < 1e-3
        row["recorded_boundary_ratio"] = recorded["boundary_ratio"]
        row["recorded_boundary_error_wb"] = recorded["boundary_error"]
        row["analytic_route_error_fraction_of_correct_span"] = abs(
            entries["route"]["flux_wb"] - maximum
        ) / (row["axis_flux_wb"] - maximum)
        report["rows"].append(row)
        print("ROW " + json.dumps(row, sort_keys=True), flush=True)
    return report, arcs, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    started = perf_counter()
    tree = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True
    ).strip()
    print(
        f"REVISION={revision} TREE={tree} COMMAND={sys.executable} "
        "-m scripts.prototypes.bank_limited_contact.measure " + " ".join(sys.argv[1:]),
        flush=True,
    )
    print(
        f"MODULE={fixture.__file__} REFERENCE={references.__file__} "
        f"CWD={Path.cwd().resolve()}",
        flush=True,
    )
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "cpu"
    report, arcs, values = measure(args.inputs)
    report["revision"] = revision
    if args.figure:
        from scripts.prototypes.bank_limited_contact.plot import render

        render(report, arcs, values, args.figure, args.preview)
    report["elapsed_seconds_after_imports"] = perf_counter() - started
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "CONCLUSION analytic boundary is designed_contact at the outboard midplane; "
        "saved oracle selected a vertex; compare against 0 Wb.",
        flush=True,
    )
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
