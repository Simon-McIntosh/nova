"""Continuous sublevel cuts from a saddle and exact endpoint enclosures."""

from __future__ import annotations

import pathlib

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.optimize import root
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import split

from scripts.prototypes.efit_maximin_certificate.bounds import (
    bound_polyline,
    enclosing_target_cut,
)
from scripts.prototypes.efit_maximin_certificate.certificate import (
    load_record,
    spline_of,
    vessel_ring,
)


def saddle_cut(spline, ring, axis, target, level, seed=(0.575, 1.123)):
    """Trace both downhill branches and certify the wall-to-wall barrier."""

    def gradient(point):
        return np.asarray(
            [spline.ev(*point, dx=1, dy=0), spline.ev(*point, dx=0, dy=1)],
            dtype=float,
        ).reshape(2)

    stationary = root(gradient, seed, tol=1e-13)
    saddle = stationary.x
    if np.linalg.norm(gradient(saddle)) > 1e-12:
        raise ArithmeticError("saddle stationarity failed")
    hessian = np.asarray(
        [
            [spline.ev(*saddle, dx=2, dy=0), spline.ev(*saddle, dx=1, dy=1)],
            [spline.ev(*saddle, dx=1, dy=1), spline.ev(*saddle, dx=0, dy=2)],
        ],
        dtype=float,
    ).reshape(2, 2)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    if not eigenvalues[0] < 0 < eigenvalues[1]:
        raise ArithmeticError("stationary point is not a saddle")
    downhill = eigenvectors[:, 0]
    domain = MplPath(ring)
    branches = []
    for direction in (-1, 1):
        point = saddle + direction * 1e-4 * downhill
        chain = [saddle.copy(), point.copy()]
        for _ in range(10000):
            vector = gradient(point)
            norm = np.linalg.norm(vector)
            if norm < 1e-15:
                raise ArithmeticError("downhill trace stopped inside vessel")
            point = point - 0.001 * vector / norm
            chain.append(point.copy())
            if not domain.contains_point(point):
                break
        else:
            raise ArithmeticError("downhill trace did not reach the wall")
        branches.append(np.asarray(chain))
    points = np.vstack((branches[0][::-1], branches[1][1:]))
    bound = bound_polyline(spline, 1, points)
    if not bound["maximum_upper"] < level:
        raise ArithmeticError("downhill cut rises to the tested level")
    parts = list(split(Polygon(ring), LineString(points)).geoms)
    axis_members = [
        index for index, part in enumerate(parts) if part.covers(Point(axis))
    ]
    target_members = [
        index for index, part in enumerate(parts) if part.covers(Point(target))
    ]
    if len(parts) != 2 or len(axis_members) != 1 or len(target_members) != 1:
        raise ArithmeticError(
            f"wall-to-wall cut separation is undecided: parts={len(parts)}, "
            f"axis={axis_members}, target={target_members}"
        )
    if axis_members == target_members:
        raise ArithmeticError("axis and target remain on the same side of the cut")
    return {
        "points": points.tolist(),
        "bound": bound,
        "saddle": saddle.tolist(),
        "saddle_value": float(spline.ev(*saddle)),
        "separates": True,
        "axis_part": axis_members[0],
        "target_part": target_members[0],
    }


def make_cuts(spec, output: pathlib.Path):
    record = load_record(pathlib.Path(spec["source"]))
    spline = spline_of(record)
    if spec["case"].startswith("mastu"):
        cut = saddle_cut(
            spline,
            vessel_ring(record),
            spec["axis"],
            spec["target"],
            spec["nova_level"],
        )
        level = spec["nova_level"]
    else:
        cut = enclosing_target_cut(
            spline,
            int(record.polarity),
            spec["target"],
            spec["axis"],
            spec["efit_level"],
        )
        level = spec["efit_level"]
    if cut is None:
        raise ArithmeticError("no strict endpoint cut was certified")
    rows = [
        f"# exact_level {level:.17g}",
        f"# max_on_curve_upper {cut['bound']['maximum_upper']:.17g}",
        f"# pieces {cut['bound']['pieces']} "
        f"arithmetic_pad {cut['bound']['arithmetic_pad']}",
        "# R Z polarity*psi",
    ]
    for r, z in cut["points"]:
        rows.append(
            f"{r:.17g} {z:.17g} {record.polarity * spline.ev(r, z).item():.17g}"
        )
    output.write_text("\n".join(rows) + "\n")
    return {key: value for key, value in cut.items() if key != "points"}
