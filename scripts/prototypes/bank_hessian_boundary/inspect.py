"""Partition retained Hessian samples by distance from every source edge."""

import json
from pathlib import Path

import jax
import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64

ROOT = Path.home() / ".cache/nova-hessian-boundary"
STATES = (220, 333, 336)


def load():
    geometry = dict(np.load(ROOT / "h200/geometry.npz"))
    archives = [dict(np.load(ROOT / f"diagnosis/accuracy-{s}.npz")) for s in STATES]
    reference = dict(np.load(ROOT / "measurement/qualified-reference.npz"))
    polygons = tuple(
        p[:n]
        for p, n in zip(geometry["vertices"], geometry["vertex_count"], strict=True)
    )
    points = archives[0]["points"][reference["indices"]]
    for archive in archives:
        np.testing.assert_array_equal(archive["points"][reference["indices"]], points)
    starts = np.concatenate(polygons)
    ends = np.concatenate([np.roll(p, -1, axis=0) for p in polygons])
    delta = ends - starts
    offset = points[:, None] - starts[None]
    fraction = np.clip(
        np.sum(offset * delta, axis=-1) / np.sum(delta**2, axis=-1), 0, 1
    )
    distances = np.linalg.norm(offset - fraction[..., None] * delta, axis=-1)
    clearance = distances.min(axis=1)
    interior = clearance > reference["steps"].max()
    return polygons, archives, reference, points, clearance, interior


def relative(delta, scale):
    return np.sqrt(np.sum(delta**2, axis=0) / np.sum(scale**2, axis=0))


def main():
    polygons, archives, ref, points, clearance, interior = load()
    corner_distance = np.linalg.norm(
        points[:, None] - np.concatenate(polygons), axis=-1
    ).min(axis=1)
    assert interior.any() and (~interior).any()
    records = []
    for s, state in enumerate(STATES):
        truth = ref["plasma"][s] + ref["coils"][s]
        bounds = ref["plasma_bound"][s] + ref["coil_bound"][s]
        routes = {
            name: archives[s][key][ref["indices"], 4:6]
            for name, key in (("bank", "actual"), ("direct", "expected"))
        }
        item = dict(
            state=state,
            interior_count=int(interior.sum()),
            boundary_count=int((~interior).sum()),
        )
        for subset, mask in (
            ("all", np.ones(len(points), bool)),
            ("interior", interior),
            ("boundary", ~interior),
        ):
            item[subset] = {
                name: relative(value[mask] - truth[mask], truth[mask]).tolist()
                for name, value in routes.items()
            }
            item[subset]["reference_bound"] = relative(
                bounds[mask], truth[mask]
            ).tolist()
        mutation = truth[interior].copy()
        mutation[:, 1] *= np.longdouble("1.000001")
        control_error = float(relative(mutation - truth[interior], truth[interior])[1])
        threshold = max(1e-8, 4 * float(item["interior"]["reference_bound"][1]))
        item["negative_control"] = dict(
            error=control_error, threshold=threshold, detected=control_error > threshold
        )
        records.append(item)
        print("SPLIT", json.dumps(item, default=float), flush=True)
    result = dict(
        max_step_m=float(ref["steps"].max()),
        near_corner_count=int((corner_distance < 1e-7).sum()),
        states=records,
        nodes=[
            dict(
                node=int(i),
                point=p.tolist(),
                clearance_m=float(d),
                interior=bool(m),
                corner_distance_m=float(c),
            )
            for i, p, d, m, c in zip(
                ref["indices"],
                points,
                clearance,
                interior,
                corner_distance,
                strict=True,
            )
        ],
    )
    (ROOT / "split.json").write_text(json.dumps(result, indent=2, default=float))
    assert all(row["negative_control"]["detected"] for row in records)


if __name__ == "__main__":
    main()
