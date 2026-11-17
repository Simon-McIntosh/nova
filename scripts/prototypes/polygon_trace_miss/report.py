"""Verify recorded derivative arms against the unchanged physical trace budget."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from nova.biot.greens import MU0
from scripts.prototypes.bank_hessian_diagnosis.measure import ROOT, STATES, reference
from scripts.prototypes.bank_hessian_diagnosis.report import density, jet, trace

assert jax.config.jax_enable_x64 is True


def describe(name, value):
    print("ARRAY", name, "shape", value.shape, "finite", bool(np.isfinite(value).all()))
    assert np.isfinite(value).all()


def read_array(path, name):
    with np.load(path) as archive:
        value = archive[name]
    describe(str(path) + ":" + name, value)
    return value


def entry_counts(entries):
    return {(e["caller"], e["stable_poles"], e["tracer"]): e["count"] for e in entries}


def replace_height_column(hessian, height_column):
    """Assemble the independently measured columns of a linear target JVP."""
    assert hessian.shape[-2:] == (2, 2)
    assert height_column.shape == hessian.shape[:-1]
    combined = hessian.copy()
    combined[:, :, :, 1] = height_column
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate-refusal", action="store_true")
    parser.add_argument("--small-step-refusal", action="store_true")
    args = parser.parse_args()
    census = json.loads((args.run / "census/receipt.json").read_text())
    assert census["completed"]
    candidate = census["reproduction"][1]
    assert candidate["arm"] == "accepted-head-bank"
    if args.candidate_refusal:
        maximum = max(row["relative"] for row in candidate["rows"])
        print("CANDIDATE_TRACE", maximum, "limit", 1e-8, flush=True)
        assert maximum <= 1e-8, (
            f"candidate trace relative {maximum} exceeds unchanged limit 1e-8"
        )
        return
    small = json.loads((args.run / "stable-corner/receipt.json").read_text())
    assert small["completed"]
    assert [row["count"] for row in small["states"]] == [64, 64, 64]
    if args.small_step_refusal:
        maximum = max(row["trace"]["relative"] for row in small["states"])
        print("SMALL_STEP_TRACE", maximum, "limit", 1e-8, flush=True)
        assert maximum <= 1e-8, (
            f"small-step trace relative {maximum} exceeds unchanged limit 1e-8"
        )
        return
    height = json.loads((args.run / "height-stencil/receipt.json").read_text())
    calibration = json.loads(
        (args.run / "height-stencil/calibration-independent.json").read_text()
    )
    assert height["completed"] and height["count"] == 64
    assert height["per_cell"] and height["corner_mask"] and height["stable_poles"]
    assert height["states"] == list(STATES)
    assert calibration["passed"] and calibration["budget"] == 1e-8
    assert max(calibration["relative"]) <= 1e-8
    assert min(calibration["perturbation_relative"]) > 1e-8
    gradient = read_array(args.run / "stable-corner/plasma-partial.npz", "gradient")
    hessian = read_array(args.run / "stable-corner/plasma-partial.npz", "hessian")
    points = read_array(args.run / "stable-corner/plasma-partial.npz", "points")
    height_column = read_array(
        args.run / "height-stencil/height-partial.npz", "height_derivative"
    )
    height_points = read_array(args.run / "height-stencil/height-partial.npz", "points")
    assert gradient.shape == (3, 64, 2) and hessian.shape == (3, 64, 2, 2)
    assert points.shape == (64, 2) and height_column.shape == (3, 64, 2)
    np.testing.assert_array_equal(points, height_points)
    combined = replace_height_column(hessian, height_column)
    describe("combined_hessian", combined)
    prior_root = (
        Path.home()
        / ".config/reckon/crew/runs"
        / "r-20261114T211231918544-polygon-tangent-repair"
    )
    stop = json.loads((prior_root / "accepted-trace-stop.json").read_text())
    prior_path = prior_root / "accepted-head-bank/plasma-partial.npz"
    assert (
        hashlib.file_digest(prior_path.open("rb"), "sha256").hexdigest()
        == stop["head"]["archive_sha256"]
    )
    assert candidate["sha256"] == stop["head"]["archive_sha256"]
    prior_gradient = read_array(prior_path, "gradient")
    prior_hessian = read_array(prior_path, "hessian")
    np.testing.assert_array_equal(points, read_array(prior_path, "points"))
    geometry = dict(np.load(ROOT / "h200/geometry.npz"))
    polygons = tuple(
        p[:n]
        for p, n in zip(geometry["vertices"], geometry["vertex_count"], strict=True)
    )
    moments = np.stack(
        [np.load(ROOT / f"diagnosis/accuracy-{s}.npz")["moments"] for s in STATES]
    )
    packed = reference().coupling(polygons)
    densities, owners = density(points, polygons, packed, moments)
    forcing = -2 * np.pi * MU0 * points[None, :, 0] * densities
    for name, value in [
        ("moments", moments),
        ("authored_density", densities),
        ("owners", owners),
        ("forcing", forcing),
    ]:
        describe(name, value)
    corner = np.array(census["counts"]["corner_cells_per_point"]) > 0
    describe("corner_target_mask", corner)
    rows = []
    for index, state in enumerate(STATES):
        reproduced = trace(
            jet(prior_gradient[index], prior_hessian[index]), points, forcing[index]
        )
        np.testing.assert_allclose(
            reproduced["relative"],
            candidate["rows"][index]["relative"],
            rtol=1e-13,
            atol=1e-15,
        )
        intermediate = trace(
            jet(gradient[index], hessian[index]), points, forcing[index]
        )
        np.testing.assert_allclose(
            intermediate["relative"],
            small["states"][index]["trace"]["relative"],
            rtol=1e-13,
            atol=1e-15,
        )
        final = trace(jet(gradient[index], combined[index]), points, forcing[index])
        residual = np.array(reproduced["residual"])
        ordinary_fraction = float(np.sum(residual[~corner] ** 2) / np.sum(residual**2))
        radial = hessian[index, :, 0, 0] - prior_hessian[index, :, 0, 0]
        vertical = hessian[index, :, 1, 1] - prior_hessian[index, :, 1, 1]
        stencil = combined[index, :, 1, 1] - hessian[index, :, 1, 1]
        prefactor = (
            -(gradient[index, :, 0] - prior_gradient[index, :, 0]) / points[:, 0]
        )
        reconstructed = residual + radial + vertical + stencil + prefactor
        for name, value in [
            ("candidate_residual", residual),
            ("radial_correction", radial),
            ("small_height_correction", vertical),
            ("height_stencil_correction", stencil),
            ("prefactor_correction", prefactor),
            ("reconstructed_residual", reconstructed),
        ]:
            describe(f"{state}:{name}", value)
        np.testing.assert_allclose(
            reconstructed, final["residual"], rtol=1e-6, atol=1e-13
        )
        row = dict(
            state=state,
            count=64,
            candidate_relative=reproduced["relative"],
            stable_small_step_relative=intermediate["relative"],
            stable_sixth_order_relative=final["relative"],
            ordinary_only_residual_fraction=ordinary_fraction,
            radial_correction_norm=float(np.linalg.norm(radial)),
            small_height_correction_norm=float(np.linalg.norm(vertical)),
            height_stencil_correction_norm=float(np.linalg.norm(stencil)),
            prefactor_correction_norm=float(np.linalg.norm(prefactor)),
        )
        rows.append(row)
        print("RESULT", json.dumps(row), flush=True)
        assert ordinary_fraction < 1e-12
        assert final["relative"] <= 1e-8
    probe = json.loads((args.run / "candidate/receipt.json").read_text())
    assert probe["completed"]
    assert entry_counts(probe["expression_entries"]) == {
        ("evaluate", "False", "JVPTracer"): 2,
        ("tangent_expression", "True", "JVPTracer"): 1,
        ("evaluate", "False", "ArrayImpl"): 4,
    }
    assert entry_counts(small["expression_entries"]) == {
        ("evaluate", "False", "JVPTracer"): 4,
        ("tangent_expression", "True", "JVPTracer"): 8,
        ("tangent_expression", "True", "ArrayImpl"): 16,
        ("evaluate", "False", "ArrayImpl"): 4,
    }
    height_entries = height["expression_entries"]
    assert 0 < height_entries["height_jvp"] <= 4
    assert height_entries["sample"] == 6 * height_entries["height_jvp"]
    probe_gradient = read_array(args.run / "candidate/plasma-partial.npz", "gradient")
    probe_hessian = read_array(args.run / "candidate/plasma-partial.npz", "hessian")
    np.testing.assert_allclose(
        probe_gradient, prior_gradient[:, :1], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        probe_hessian, prior_hessian[:, :1], rtol=1e-12, atol=1e-12
    )
    result = dict(
        completed=True,
        rows=rows,
        branch_counts=census["counts"],
        candidate_expression_entries=probe["expression_entries"],
        small_step_expression_entries=small["expression_entries"],
        height_expression_entries=height_entries,
        height_offsets_m=height["steps"],
        height_calibration=calibration,
        small_step_passed=max(row["stable_small_step_relative"] for row in rows)
        <= 1e-8,
        primal_gradient_maximum_difference=float(
            np.max(abs(gradient - prior_gradient))
        ),
        primal_gradient_relative_change=float(
            np.linalg.norm(gradient - prior_gradient) / np.linalg.norm(prior_gradient)
        ),
        primal_gradient_bit_mismatches=int(
            np.count_nonzero(gradient.view(np.uint64) != prior_gradient.view(np.uint64))
        ),
    )
    assert result["primal_gradient_relative_change"] <= 1e-12
    (args.run / "summary.json").write_text(json.dumps(result, indent=2))
    np.savez(
        args.run / "combined-derivatives.npz",
        gradient=gradient,
        hessian=combined,
        points=points,
        forcing=forcing,
    )
    print("DIAGNOSIS_COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
