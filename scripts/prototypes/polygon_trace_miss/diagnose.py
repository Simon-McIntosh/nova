"""Measure corner-route coverage and a shared stable field-derivative expression."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from collections import Counter
import hashlib
import inspect
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
jax.config.update("jax_enable_compilation_cache", False)


def array_record(name, value):
    """Print full shape and finiteness for every inspected numerical array."""
    value = np.asarray(value)
    row = dict(
        name=name, shape=list(value.shape), finite=bool(np.isfinite(value).all())
    )
    print("ARRAY", json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("census", "candidate", "stable-corner"), required=True
    )
    parser.add_argument("--point-budget", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print("MEASUREMENT_MODULE", kernel.__file__, flush=True)
    print("MEASUREMENT_CWD", Path.cwd().resolve(), flush=True)
    print("DEVICES", jax.devices(), "x64", jax.config.jax_enable_x64, flush=True)
    geometry = dict(np.load(ROOT / "h200/geometry.npz"))
    polygons = tuple(
        p[:n]
        for p, n in zip(geometry["vertices"], geometry["vertex_count"], strict=True)
    )
    split = json.loads((ROOT / "split.json").read_text())
    indices = np.array([row["node"] for row in split["nodes"] if row["interior"]])
    assert len(indices) == 64
    archives = [dict(np.load(ROOT / f"diagnosis/accuracy-{s}.npz")) for s in STATES]
    points = archives[0]["points"][indices]
    moments = np.stack([a["moments"] for a in archives])
    coupling = BiotMomentCoupling.from_polygons(polygons)
    arrays = [array_record("points", points), array_record("moments", moments)]
    edge = np.asarray(coupling.edge)
    nearest = np.minimum(
        np.min(abs(edge[:, 1, None, :] - points[None, :, 1, None]), axis=0),
        np.min(abs(edge[:, 3, None, :] - points[None, :, 1, None]), axis=0),
    )
    near = nearest <= 1e-7
    arrays += [
        array_record("packed_edges", edge),
        array_record("nearest_corner_height", nearest),
        array_record("near_corner_mask", near),
    ]
    counts = dict(
        points=len(points),
        cells=len(polygons),
        selected_pairs=int(near.sum()),
        total_pairs=int(near.size),
        points_with_corner=int(near.any(axis=1).sum()),
        corner_cells_per_point=near.sum(axis=1).tolist(),
    )
    print("BRANCH_COUNTS", json.dumps(counts), flush=True)
    packed = reference().coupling(polygons)
    densities, owners = density(points, polygons, packed, moments)
    forcing = -2 * np.pi * MU0 * points[None, :, 0] * densities
    arrays += [
        array_record("authored_density", densities),
        array_record("owners", owners),
        array_record("forcing", forcing),
    ]
    prior_root = (
        Path.home()
        / ".config/reckon/crew/runs/r-20261114T211231918544-polygon-tangent-repair"
    )
    reproduction = []
    for label in ("accepted-base-bank", "accepted-head-bank"):
        path = prior_root / label / "plasma-partial.npz"
        data = dict(np.load(path))
        arrays += [array_record(f"{label}.{k}", v) for k, v in data.items()]
        np.testing.assert_array_equal(data["points"], points)
        assert np.linalg.norm(data["hessian"]) > 0
        rows = [
            dict(
                state=s,
                **trace(
                    jet(data["gradient"][i], data["hessian"][i]), points, forcing[i]
                ),
            )
            for i, s in enumerate(STATES)
        ]
        reproduction.append(
            dict(
                arm=label,
                sha256=hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
                rows=rows,
            )
        )
        print(
            "REPRODUCTION",
            label,
            json.dumps([dict(state=r["state"], relative=r["relative"]) for r in rows]),
            flush=True,
        )
    control = json.loads((prior_root / "head-eager/receipt.json").read_text())
    assert max(r["trace"]["relative"] for r in control["states"]) < 1e-8
    assert max(r["relative"] for r in reproduction[-1]["rows"]) > 0.07
    result = dict(
        mode=args.mode,
        arrays=arrays,
        counts=counts,
        reproduction=reproduction,
        completed=False,
    )
    if args.mode == "census":
        result["completed"] = True
        (args.output / "receipt.json").write_text(json.dumps(result, indent=2))
        print("MEASUREMENT_COMPLETE census", flush=True)
        return
    # Count Python expression entries during the actual field Jacobian trace.
    entries = Counter()
    original = kernel._packed_analytic_moments_primal

    def counted(*positional, **keywords):
        caller = inspect.currentframe().f_back.f_code.co_name
        tracer = type(positional[1]).__name__
        entries[(caller, str(keywords.get("stable_poles", False)), tracer)] += 1
        return original(*positional, **keywords)

    kernel._packed_analytic_moments_primal = counted
    selected = points[: args.point_budget] if args.mode == "candidate" else points
    try:
        gradient, hessian = evaluate(
            coupling,
            selected,
            moments,
            args.output,
            "plasma",
            resume=True,
            point_budget=args.point_budget if args.mode == "stable-corner" else 0,
        )
    except PointCheckpoint as checkpoint:
        print("MEASUREMENT_CHECKPOINT", str(checkpoint), flush=True)
        print(
            "EXPRESSION_ENTRIES",
            json.dumps(
                [
                    dict(caller=k[0], stable_poles=k[1], tracer=k[2], count=v)
                    for k, v in entries.items()
                ]
            ),
            flush=True,
        )
        return
    result["arrays"] += [
        array_record("diagnostic.gradient", gradient),
        array_record("diagnostic.hessian", hessian),
    ]
    result["expression_entries"] = [
        dict(caller=k[0], stable_poles=k[1], tracer=k[2], count=v)
        for k, v in entries.items()
    ]
    print("EXPRESSION_ENTRIES", json.dumps(result["expression_entries"]), flush=True)
    result["states"] = [
        dict(
            state=s,
            count=len(selected),
            trace=trace(
                jet(gradient[i], hessian[i]), selected, forcing[i, : len(selected)]
            ),
        )
        for i, s in enumerate(STATES)
    ]
    for row in result["states"]:
        print(
            "TRACE_SUMMARY",
            json.dumps(
                dict(
                    state=row["state"],
                    count=row["count"],
                    relative=row["trace"]["relative"],
                )
            ),
            flush=True,
        )
    result["completed"] = True
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2))
    print("MEASUREMENT_COMPLETE", args.mode, flush=True)


if __name__ == "__main__":
    main()
