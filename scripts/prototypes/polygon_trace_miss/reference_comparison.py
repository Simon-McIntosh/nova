"""Attribute a retained diagnostic prefix against existing independent Hessians."""

import argparse
import json
from pathlib import Path

import numpy as np


def describe(name, value):
    print(
        "ARRAY",
        name,
        "shape",
        value.shape,
        "finite",
        bool(np.isfinite(value).all()),
        "dtype",
        value.dtype,
    )
    assert np.isfinite(value).all()


def trace_metric(gradient, hessian, points, forcing):
    terms = np.stack(
        (
            hessian[:, :, 0, 0],
            hessian[:, :, 1, 1],
            -gradient[:, :, 0] / points[None, :, 0],
        ),
        axis=-1,
    )
    residual = terms.sum(axis=-1) - forcing
    scale = np.maximum(
        np.sqrt(np.sum(forcing**2, axis=1)), np.sqrt(np.sum(terms**2, axis=(1, 2)))
    )
    return np.sqrt(np.sum(residual**2, axis=1)) / scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = Path.home() / ".cache/nova-hessian-diagnosis"
    selected = [
        row["node"]
        for row in json.loads((root / "split.json").read_text())["nodes"]
        if row["interior"]
    ]
    assert len(selected) == 64
    with np.load(args.run / "stable-corner/plasma-partial.npz") as archive:
        gradient = archive["gradient"].astype(np.longdouble)
        hessian = archive["hessian"].astype(np.longdouble)
        points = archive["points"].astype(np.longdouble)
    n = len(points)
    reference = []
    for node in selected[:n]:
        with np.load(root / f"node-{node}.npz") as archive:
            assert int(archive["node"]) == node
            reference.append(
                {key: archive[key] for key in ("gradient", "hessian", "bound")}
            )
    reference_gradient = np.stack([row["gradient"] for row in reference], axis=1)
    reference_hessian = np.stack([row["hessian"] for row in reference], axis=1)
    reference_bound = np.stack([row["bound"] for row in reference], axis=1)
    prior = (
        Path.home()
        / ".config/reckon/crew/runs"
        / "r-20261114T211231918544-polygon-tangent-repair"
        / "accepted-head-bank/plasma-partial.npz"
    )
    with np.load(prior) as archive:
        prior_gradient = archive["gradient"][:, :n].astype(np.longdouble)
        prior_hessian = archive["hessian"][:, :n].astype(np.longdouble)
        np.testing.assert_array_equal(points, archive["points"][:n])
    census = json.loads((args.run / "census/receipt.json").read_text())
    prior_residual = np.array(
        [row["residual"][:n] for row in census["reproduction"][1]["rows"]],
        dtype=np.longdouble,
    )
    forcing = (
        prior_hessian[:, :, 0, 0]
        + prior_hessian[:, :, 1, 1]
        - prior_gradient[:, :, 0] / points[None, :, 0]
        - prior_residual
    )
    error = hessian - reference_hessian
    for name, value in [
        ("gradient", gradient),
        ("hessian", hessian),
        ("points", points),
        ("reference_gradient", reference_gradient),
        ("reference_hessian", reference_hessian),
        ("reference_bound", reference_bound),
        ("prior_gradient", prior_gradient),
        ("prior_hessian", prior_hessian),
        ("prior_residual", prior_residual),
        ("forcing", forcing),
        ("hessian_error", error),
    ]:
        describe(name, value)
    reference_trace = trace_metric(
        reference_gradient, reference_hessian, points, forcing
    )
    diagnostic_trace = trace_metric(gradient, hessian, points, forcing)
    describe("reference_trace_relative", reference_trace)
    describe("diagnostic_trace_relative", diagnostic_trace)
    assert max(reference_trace) < 1e-8
    rows = []
    for index, state in enumerate((220, 333, 336)):
        row = dict(
            state=state,
            count=n,
            reference_trace_relative=float(reference_trace[index]),
            diagnostic_trace_relative=float(diagnostic_trace[index]),
        )
        for label, entry in [("radial", (0, 0)), ("height", (1, 1))]:
            delta = error[index, :, entry[0], entry[1]]
            truth = reference_hessian[index, :, entry[0], entry[1]]
            bound = reference_bound[index, :, entry[0], entry[1]]
            row[label + "_error_norm"] = float(np.linalg.norm(delta))
            row[label + "_relative_error"] = float(
                np.linalg.norm(delta) / np.linalg.norm(truth)
            )
            row[label + "_reference_bound_norm"] = float(np.linalg.norm(bound))
        rows.append(row)
        print("REFERENCE_COMPARISON", json.dumps(row), flush=True)
    result = dict(completed=n == 64, count=n, rows=rows)
    (args.run / f"reference-comparison-{n}.json").write_text(
        json.dumps(result, indent=2)
    )
    print("REFERENCE_COMPARISON_COMPLETE", n, flush=True)


if __name__ == "__main__":
    main()
