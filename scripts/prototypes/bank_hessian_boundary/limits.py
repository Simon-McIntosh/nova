"""Measure cell-contained derivatives approaching polygon corners from each side."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import time

import numpy as np
import shapely
from shapely.geometry import Polygon

import scripts.prototypes
from scripts.prototypes.bank_hessian_boundary.inspect import ROOT, load

scripts.prototypes.__path__.append(str(ROOT / "source/scripts/prototypes"))
from scripts.prototypes.bank_hessian_reference.probe import coupling, gradient  # noqa: E402
from scripts.prototypes.bank_hessian_reference.measure import extrapolate  # noqa: E402
from scripts.prototypes.bank_hessian_reference.report import reference  # noqa: E402


def clearance(point, polygons):
    start = np.concatenate(polygons)
    delta = np.concatenate([np.roll(p, -1, axis=0) - p for p in polygons])
    offset = point - start
    fraction = np.clip(
        np.sum(offset * delta, axis=-1) / np.sum(delta**2, axis=-1), 0, 1
    )
    return np.linalg.norm(offset - fraction[:, None] * delta, axis=-1).min()


def membership(points, polygons):
    return np.stack(
        [shapely.contains_xy(Polygon(p), points[:, 0], points[:, 1]) for p in polygons],
        axis=-1,
    )


def evaluate(task):
    local, node, point, polygons, moments, smooth, full, all_polygons, source_cells = (
        task
    )
    started = time.monotonic()
    packed = coupling(polygons)
    point = np.asarray(point, dtype=np.longdouble)
    gap = clearance(point, all_polygons)
    corner = min(np.linalg.norm(p - point, axis=-1).min() for p in all_polygons) < 1e-7
    distances = (
        1e-3 if corner or node == -1 else min(float(gap) / 8, 1e-3)
    ) * 0.1 ** np.arange(5)
    coeff = np.einsum(
        "cij,scj->sci", packed["inverse_second_moment"], moments[:, :, 1:]
    )
    areas = np.array([Polygon(p).area for p in polygons])
    values, errors, traces, forcing, owners, stencils = [], [], [], [], [], []
    for distance in distances:
        side_values, side_errors, side_trace, side_force, side_owners = (
            [],
            [],
            [],
            [],
            [],
        )
        for side in (-1, 1):
            q = point + np.array([0, side * distance], dtype=np.longdouble)
            h = min(distance / 16, float(clearance(q, all_polygons)) / 4)
            assert h > 0
            shifts = []
            for axis in (0, 1):
                for level in range(4):
                    unit = np.eye(2, dtype=np.longdouble)[axis] * h / 2**level
                    shifts.extend((q + unit, q - unit))
            shifts = np.array(shifts)
            inside = membership(np.asarray(shifts, float), all_polygons)
            all_at = membership(np.asarray(q[None], float), all_polygons)[0]
            at = all_at[source_cells]
            assert np.all(inside == all_at), f"stencil crossed source edge at {node}"
            g = np.concatenate(
                [
                    gradient(shifts[i : i + 2], packed, moments)
                    for i in range(0, len(shifts), 2)
                ],
                axis=1,
            )
            hessians = []
            for axis in (0, 1):
                raw = np.array(
                    [
                        (g[:, axis * 8 + 2 * k] - g[:, axis * 8 + 2 * k + 1])
                        / (2 * h / 2**k)
                        for k in range(4)
                    ]
                )
                levels = extrapolate(raw)
                hessians.append(
                    (levels[-1][-1], 4 * np.abs(levels[-1][-1] - levels[-2][-1]))
                )
            hz, error = hessians[1]
            hr = hessians[0][0]
            gcentre = gradient(q[None], packed, moments)[:, 0]
            density = np.sum(
                np.where(
                    at,
                    (moments[:, :, 0] + np.sum(coeff * (q - packed["centre"]), axis=-1))
                    / areas,
                    0,
                ),
                axis=-1,
            )
            side_values.append(hz + smooth)
            side_errors.append(error)
            side_trace.append(hr[:, 0] + hz[:, 1] - gcentre[:, 0] / q[0])
            side_force.append(-2 * np.pi * (4e-7 * np.pi) * q[0] * density)
            side_owners.append(np.flatnonzero(all_at).tolist())
            stencils.append(
                dict(
                    distance_m=float(distance),
                    side=side,
                    step_m=float(h),
                    clearance_m=float(clearance(q, all_polygons)),
                    contained=True,
                )
            )
        values.append(side_values)
        errors.append(side_errors)
        traces.append(side_trace)
        forcing.append(side_force)
        owners.append(side_owners)
    result = dict(
        node=int(node),
        local=int(local),
        corner=bool(corner),
        full_field=full,
        source_cells=source_cells.tolist(),
        distance_m=distances.tolist(),
        values=np.asarray(values).tolist(),
        error=np.asarray(errors).tolist(),
        trace=np.asarray(traces).tolist(),
        forcing=np.asarray(forcing).tolist(),
        owners=owners,
        stencils=stencils,
    )
    (ROOT / "nodes" / f"node-{node}.json").write_text(json.dumps(result, default=float))
    print(
        "LIMIT",
        node,
        "corner",
        corner,
        "seconds",
        time.monotonic() - started,
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--nodes", type=int, nargs="*")
    args = parser.parse_args()
    (ROOT / "nodes").mkdir(exist_ok=True)
    polygons, archives, ref, points, _, interior = load()
    moments = np.stack([a["moments"] for a in archives])
    with (
        np.load(ROOT / "measurement/plasma-sweep.npz") as sweep,
        np.load(ROOT / "measurement/corner-correction-refined.npz") as correction,
    ):
        smooth, bound, _, _ = reference(sweep["values"] - correction["near_values"])
    smooth += ref["coils"]
    active = np.any(moments != 0, axis=(0, 2))
    tasks = []
    for local in np.flatnonzero(~interior):
        node = ref["indices"][local]
        if args.nodes and node not in args.nodes:
            continue
        near = (
            np.array(
                [
                    np.linalg.norm(p - points[local], axis=-1).min() < 1e-7
                    for p in polygons
                ]
            )
            & active
        )
        if not near.any():
            # Inactive source corners need no singular correction; nearby open
            # edges use the complete field and cell-contained stencils.
            near = active.copy()
            remainder = ref["coils"][:, local]
            full = True
        else:
            remainder = smooth[:, local]
            full = False
        tasks.append(
            (
                local,
                node,
                points[local],
                tuple(p for p, keep in zip(polygons, near, strict=True) if keep),
                moments[:, near],
                remainder,
                full,
                polygons,
                np.flatnonzero(near),
            )
        )
    if not args.nodes:
        square = np.array([[5.0, -0.2], [5.4, -0.2], [5.4, 0.2], [5.0, 0.2]])
        uniform = np.zeros((3, 1, 3))
        uniform[:, 0, 0] = (1000, 1500, 2000)
        tasks.append(
            (
                -1,
                -1,
                np.array([5.2, 0.2]),
                (square,),
                uniform,
                np.zeros((3, 2)),
                False,
                (square,),
                np.array([0]),
            )
        )
    print("TASKS", len(tasks), "full", sum(t[6] for t in tasks), flush=True)
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for future in as_completed([pool.submit(evaluate, task) for task in tasks]):
            future.result()
    print("LIMITS_COMPLETE", len(tasks), flush=True)


if __name__ == "__main__":
    main()
