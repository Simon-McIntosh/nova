"""Read completed trace receipts and refuse a candidate above the physical bound."""

import argparse
import json
from pathlib import Path

import numpy as np


def read_array(path, name):
    with np.load(path) as archive:
        value = archive[name]
    print(
        "ARRAY",
        str(path),
        name,
        "shape",
        value.shape,
        "finite",
        bool(np.isfinite(value).all()),
    )
    assert np.isfinite(value).all()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate-refusal", action="store_true")
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
    diagnostic = json.loads((args.run / "stable-corner/receipt.json").read_text())
    path = args.run / "stable-corner/plasma-partial.npz"
    gradient = read_array(path, "gradient")
    hessian = read_array(path, "hessian")
    points = read_array(path, "points")
    assert gradient.shape == (3, 64, 2)
    assert hessian.shape == (3, 64, 2, 2)
    assert points.shape == (64, 2)
    assert diagnostic["completed"]
    assert [r["count"] for r in diagnostic["states"]] == [64, 64, 64]
    corner = np.array(census["counts"]["corner_cells_per_point"]) > 0
    rows = []
    for old, stable in zip(candidate["rows"], diagnostic["states"], strict=True):
        assert old["state"] == stable["state"]
        residual = np.array(old["residual"])
        ordinary_energy = np.linalg.norm(residual[~corner]) ** 2
        full_energy = np.linalg.norm(residual) ** 2
        row = dict(
            state=old["state"],
            candidate_relative=old["relative"],
            diagnostic_relative=stable["trace"]["relative"],
            ordinary_only_residual_fraction=ordinary_energy / full_energy,
            candidate_corner_residual_norm=float(np.linalg.norm(residual[corner])),
            candidate_ordinary_residual_norm=float(np.linalg.norm(residual[~corner])),
        )
        rows.append(row)
        print("RESULT", json.dumps(row), flush=True)
        assert stable["trace"]["relative"] <= 1e-8
        assert ordinary_energy / full_energy < 1e-12
    result = dict(
        completed=True,
        rows=rows,
        branch_counts=census["counts"],
        diagnostic_expression_entries=diagnostic["expression_entries"],
    )
    probe = json.loads((args.run / "candidate/receipt.json").read_text())
    assert probe["completed"]
    result["candidate_expression_entries"] = probe["expression_entries"]
    probe_path = args.run / "candidate/plasma-partial.npz"
    probe_gradient = read_array(probe_path, "gradient")
    probe_hessian = read_array(probe_path, "hessian")
    original = (
        Path.home()
        / ".config/reckon/crew/runs"
        / "r-20261114T211231918544-polygon-tangent-repair"
        / "accepted-head-bank/plasma-partial.npz"
    )
    saved_gradient = read_array(original, "gradient")
    saved_hessian = read_array(original, "hessian")
    result["probe_maximum_hessian_difference"] = float(
        np.max(abs(probe_hessian - saved_hessian[:, :1]))
    )
    result["probe_maximum_gradient_difference"] = float(
        np.max(abs(probe_gradient - saved_gradient[:, :1]))
    )
    np.testing.assert_allclose(
        probe_hessian, saved_hessian[:, :1], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        probe_gradient, saved_gradient[:, :1], rtol=1e-12, atol=1e-12
    )
    result["primal_gradient_maximum_difference"] = float(
        np.max(abs(gradient - saved_gradient))
    )
    np.testing.assert_array_equal(
        gradient.view(np.uint64), saved_gradient.view(np.uint64)
    )
    (args.run / "summary.json").write_text(json.dumps(result, indent=2))
    print("DIAGNOSIS_COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
