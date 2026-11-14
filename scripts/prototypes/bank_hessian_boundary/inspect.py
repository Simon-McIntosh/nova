"""Partition retained Hessian samples by distance from every source edge."""

import json
from pathlib import Path

import jax
import numpy as np

from nova.jax.config import configure_dtypes
from scripts.prototypes.bank_hessian_boundary.geometry import clearance

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
    distance = clearance(points, polygons)
    assert np.all(np.isfinite(distance))
    interior = distance > reference["steps"].max()
    return polygons, archives, reference, points, distance, interior


def relative(delta, scale):
    return np.sqrt(np.sum(delta**2, axis=0) / np.sum(scale**2, axis=0))


def route_residuals(archive, indices, mask, truth, *, mutate=False):
    """Compare the retained measured routes with the independent Hessians.

    The perturbation enters the measured curvature before the same comparison
    used by the unperturbed arm. The independent estimate is never mutated.
    """
    result = {}
    for name, key in (("bank", "actual"), ("direct", "expected")):
        measured = np.array(archive[key][indices[mask], 4:6], copy=True)
        if mutate:
            measured[:, 1] *= np.longdouble("1.000001")
        result[name] = measured - truth[mask]
    return result


def route_metrics(archive, indices, mask, truth):
    return {
        name: relative(residual, truth[mask]).tolist()
        for name, residual in route_residuals(archive, indices, mask, truth).items()
    }


def comparison_control(baseline, measured, truth, threshold):
    # Compare vectors before reducing: a changed residual can retain its norm.
    changes = {
        name: float(relative(measured[name] - np.asarray(baseline[name]), truth)[1])
        for name in ("bank", "direct")
    }
    return dict(
        changes=changes,
        threshold=threshold,
        accepted=all(x <= threshold for x in changes.values()),
        detected=all(x > threshold for x in changes.values()),
    )


def main():
    polygons, archives, ref, points, clearance, interior = load()
    corner_distance = np.linalg.norm(
        points[:, None] - np.concatenate(polygons), axis=-1
    ).min(axis=1)
    assert interior.any() and (~interior).any()
    records = []
    prior = json.loads((ROOT / "measurement/receipt.json").read_text())
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
        item["reproduced_plasma"] = {
            name: relative(value - truth, ref["plasma"][s]).tolist()
            for name, value in routes.items()
        }
        for name in routes:
            np.testing.assert_allclose(
                item["reproduced_plasma"][name],
                prior["states"][s][f"{name}_relative"],
                rtol=1e-8,
                atol=1e-12,
            )
        for subset, mask in (
            ("all", np.ones(len(points), bool)),
            ("interior", interior),
            ("boundary", ~interior),
        ):
            item[subset] = route_metrics(archives[s], ref["indices"], mask, truth)
            item[subset]["reference_bound"] = relative(
                bounds[mask], truth[mask]
            ).tolist()
        threshold = max(1e-8, 4 * float(item["interior"]["reference_bound"][1]))
        residuals = route_residuals(archives[s], ref["indices"], interior, truth)
        item["interior_residuals"] = {name: r.tolist() for name, r in residuals.items()}
        mutated = route_residuals(
            archives[s], ref["indices"], interior, truth, mutate=True
        )
        item["negative_control"] = comparison_control(
            residuals, mutated, truth[interior], threshold
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
