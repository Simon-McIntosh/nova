"""Measure the integrated polygon Jacobian against retained independent jets."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from nova.biot import polygonanalytic as kernel
from nova.biot.greens import MU0
from nova.equilibrium.topology import BiotMomentCoupling
from scripts.prototypes.bank_hessian_diagnosis.measure import ROOT, STATES, reference
from scripts.prototypes.bank_hessian_diagnosis.report import density, jet, trace
from scripts.prototypes.polygon_trace_miss.measure import PointCheckpoint, evaluate

assert jax.config.jax_enable_x64 is True


def inputs():
    geometry = dict(np.load(ROOT / "h200/geometry.npz"))
    polygons = tuple(
        p[:n]
        for p, n in zip(geometry["vertices"], geometry["vertex_count"], strict=True)
    )
    split = json.loads((ROOT / "split.json").read_text())
    indices = np.array([row["node"] for row in split["nodes"] if row["interior"]])
    assert indices.shape == (64,)
    archives = [dict(np.load(ROOT / f"diagnosis/accuracy-{s}.npz")) for s in STATES]
    points = archives[0]["points"][indices]
    for archive in archives[1:]:
        np.testing.assert_array_equal(archive["points"][indices], points)
    moments = np.stack([a["moments"] for a in archives])
    assert points.shape == (64, 2) and moments.shape == (3, 558, 3)
    assert np.isfinite(points).all() and np.isfinite(moments).all()
    return polygons, indices, points, moments


def summarise(output):
    polygons, indices, points, moments = inputs()
    data = dict(np.load(output / "plasma-partial.npz"))
    np.testing.assert_array_equal(data["points"], points)
    gradient, hessian = data["gradient"], data["hessian"]
    assert gradient.shape == (3, 64, 2) and hessian.shape == (3, 64, 2, 2)
    assert np.isfinite(gradient).all() and np.isfinite(hessian).all()
    assert np.linalg.norm(gradient) > 0 and np.linalg.norm(hessian) > 0
    retained = [dict(np.load(ROOT / f"node-{i}.npz")) for i in indices]
    known = np.stack([row["hessian"] for row in retained], axis=1)
    bounds = np.stack([row["bound"] for row in retained], axis=1)
    densities, owners = density(
        points, polygons, reference().coupling(polygons), moments
    )
    assert owners.shape == (64,) and np.all(owners == 1)
    forcing = -2 * np.pi * MU0 * points[None, :, 0] * densities
    rows = []
    for index, state in enumerate(STATES):
        components = {}
        for name, a, b in (
            ("psi_RR", 0, 0),
            ("psi_RZ", 0, 1),
            ("psi_ZR", 1, 0),
            ("psi_ZZ", 1, 1),
        ):
            scale = np.linalg.norm(known[index, :, a, b])
            assert scale > 0
            components[name] = dict(
                relative=float(
                    np.linalg.norm(hessian[index, :, a, b] - known[index, :, a, b])
                    / scale
                ),
                reference_bound=float(np.linalg.norm(bounds[index, :, a, b]) / scale),
            )
        record = dict(
            state=state,
            trace=trace(jet(gradient[index], hessian[index]), points, forcing[index]),
            reference_trace=trace(
                jet(gradient[index], known[index]), points, forcing[index]
            ),
            components=components,
        )
        rows.append(record)
        print("INTEGRATED_STATE", json.dumps(record), flush=True)
    receipt = dict(
        completed=True,
        count=64,
        states=rows,
        kernel_sha256=hashlib.sha256(Path(kernel.__file__).read_bytes()).hexdigest(),
    )
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summarise", action="store_true")
    parser.add_argument("--point-budget", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print("MEASUREMENT_MODULE", kernel.__file__, flush=True)
    print("MEASUREMENT_CWD", Path.cwd().resolve(), flush=True)
    print("DEVICES", jax.devices(), "x64", jax.config.jax_enable_x64, flush=True)
    if args.summarise:
        summarise(args.output)
        return
    polygons, _, points, moments = inputs()
    coupling = BiotMomentCoupling.from_polygons(polygons)
    try:
        evaluate(
            coupling,
            points,
            moments,
            args.output,
            "plasma",
            resume=True,
            point_budget=args.point_budget,
        )
    except PointCheckpoint as checkpoint:
        print("CHECKPOINT", checkpoint, flush=True)


if __name__ == "__main__":
    main()
