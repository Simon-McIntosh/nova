"""Maximin certificates on the imas-efit record's own spline.

Each imas-efit ``BOUNDARY_LEVEL_TRACE 2`` record carries a rectilinear psi grid
and the wall ring it was traced against.  This module rebuilds the record's own
not-a-knot bicubic spline with scipy (the same object the axis-adjudication
rebuild check uses) and evaluates one question on it per row.

The maximin from the axis to a target is the largest level ``L`` for which
``{polarity*psi >= L}`` inside the vessel still connects the two points: any
path at that level attains it, and no higher level can.  A claimed escape level
with an attaining path is a real escape level; a claimed level with no such path
is beaten by the true maximin, which is itself an attaining path to the same
target.

Nothing here evaluates a field with either field-swap adapter: the field is the
record's own scipy spline on its own rgrid/zgrid.  The search runs on a fine
square sampling of that spline restricted to the vessel ring, and the maximin is
read off a descending-level union-find, which is exact for the sampled field.
The attaining path is a breadth-first path inside the maximin's superlevel set,
so its minimum polarity*psi equals the maximin by construction.
"""

from __future__ import annotations

import json
import pathlib
from collections import deque
from dataclasses import dataclass, asdict

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.interpolate import RectBivariateSpline

REPORT = pathlib.Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/imas-efit/"
    "continuous-boundary-level/cbl-saddle-admission"
)


@dataclass
class Row:
    case: str
    source: str
    polarity: int
    verdict: int
    nova_label: str
    efit_label: str
    axis: list
    target: list
    nova_level: float
    efit_level: float
    f_axis: float
    f_target: float
    maximin_axis_to_target: float | None
    attaining_path: list
    path_min: float
    path_min_at: list
    path_length: float
    path_samples: int
    blocking_cut_exists: bool
    blocking_cut_points: list
    cut_at_nova_level: bool
    grid_step: float
    note: str = ""


def load_record(path: pathlib.Path):
    from scripts.prototypes.efit_axis_adjudication.records import parse_record

    return parse_record(path)


def spline_of(record):
    return RectBivariateSpline(record.rgrid, record.zgrid, record.psi, kx=3, ky=3, s=0)


def vessel_ring(record) -> np.ndarray:
    if record.polygon_r.size >= 3:
        return np.column_stack((record.polygon_r, record.polygon_z))
    return np.column_stack((record.wall_r[:, 0], record.wall_z[:, 0]))


def sample_field(spline, ring, polarity, n):
    rmin, zmin = ring.min(axis=0)
    rmax, zmax = ring.max(axis=0)
    rs = np.linspace(rmin, rmax, n)
    zs = np.linspace(zmin, zmax, n)
    R, Z = np.meshgrid(rs, zs, indexing="ij")
    pts = np.column_stack((R.ravel(), Z.ravel()))
    values = polarity * spline.ev(pts[:, 0], pts[:, 1])
    inside = MplPath(ring).contains_points(pts)
    return pts, values, inside, n


class Discrete:
    """A 2-D sampled field restricted to the vessel, with 4-connected adjacency."""

    def __init__(self, pts, values, inside, n):
        self.pts = pts
        self.values = values
        self.inside = inside
        self.n = n
        self.cell = np.flatnonzero(inside)
        self.row = self.cell // n
        self.col = self.cell % n
        self.lookup = np.full(len(pts), -1, dtype=np.int64)
        self.lookup[self.cell] = np.arange(len(self.cell))
        # neighbour position array: (n_inside, 4)
        self.nbr = np.full((len(self.cell), 4), -1, dtype=np.int64)
        for k, (di, dj) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
            a, b = self.row + di, self.col + dj
            ok = (a >= 0) & (a < n) & (b >= 0) & (b < n)
            flat = a * n + b
            flat = np.where(ok, flat, 0)
            self.nbr[ok, k] = self.lookup[flat[ok]]

    def nearest(self, point):
        d = np.linalg.norm(self.pts[self.cell] - np.asarray(point), axis=1)
        return int(np.argmin(d))


def maximin(discrete: Discrete, ax: int, tg: int):
    """Kruskal: the level at which the axis and target components first merge."""
    parent = np.arange(len(discrete.cell))
    side = np.zeros(len(discrete.cell), dtype=np.int8)
    side[ax] = 1
    side[tg] = 2
    active = np.zeros(len(discrete.cell), dtype=bool)

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    order = np.argsort(-discrete.values[discrete.cell], kind="stable")
    for k in order:
        active[k] = True
        rk = find(k)
        for nb in discrete.nbr[k]:
            if nb < 0 or not active[nb]:
                continue
            rn = find(nb)
            if rn == rk:
                continue
            side[rk] |= side[rn]
            parent[rn] = rk
            if side[rk] == 3:
                return float(discrete.values[discrete.cell[k]])
    return None


def attaining_path(discrete: Discrete, ax: int, tg: int, level: float):
    """Breadth-first path from the axis to the target inside {f >= level}."""
    ok = discrete.values[discrete.cell] >= level - 1e-15
    if not (ok[ax] and ok[tg]):
        return None
    parent = np.full(len(discrete.cell), -1, dtype=np.int64)
    seen = np.zeros(len(discrete.cell), dtype=bool)
    seen[ax] = True
    dq = deque([ax])
    while dq:
        cur = dq.popleft()
        if cur == tg:
            break
        for nb in discrete.nbr[cur]:
            if nb >= 0 and ok[nb] and not seen[nb]:
                seen[nb] = True
                parent[nb] = cur
                dq.append(nb)
    if not seen[tg]:
        return None
    chain = [tg]
    while chain[-1] != ax:
        chain.append(int(parent[chain[-1]]))
    chain.reverse()
    return np.asarray(chain, dtype=np.int64)


def blocking_cut(discrete: Discrete, ax: int, tg: int, level: float):
    """Interface between the axis component and the rest at ``polarity*psi >= level``.

    When the axis and target sit in different components of the superlevel set,
    no path at that level connects them; the cut is the set of below-level nodes
    adjacent to the axis's own component, a closed barrier on which
    polarity*psi stays below the level.
    """
    ok = discrete.values[discrete.cell] >= level - 1e-15
    parent = np.arange(len(discrete.cell))

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    for k in np.flatnonzero(ok):
        for nb in discrete.nbr[k]:
            if nb >= 0 and ok[nb]:
                ra, rb = find(int(k)), find(nb)
                if ra != rb:
                    parent[rb] = ra
    ra = find(ax)
    if ok[ax] and ok[tg] and find(tg) == ra:
        return False, []
    cut_cells = []
    for k in np.flatnonzero(ok):
        if find(int(k)) != ra:
            continue
        for nb in discrete.nbr[k]:
            if nb >= 0 and not ok[nb]:
                cut_cells.append(int(nb))
    cells = np.unique(np.asarray(cut_cells, dtype=np.int64)) if cut_cells else np.zeros(0, int)
    pts = discrete.pts[discrete.cell[cells]]
    return True, [[float(a), float(b)] for a, b in pts[:4000]]


def certify(case, source, path, axis, target, nova_level, efit_level,
            nova_label, efit_label, n=800):
    record = load_record(path)
    spline = spline_of(record)
    polarity = int(record.polarity)
    ring = vessel_ring(record)
    pts, values, inside, n = sample_field(spline, ring, polarity, n)
    discrete = Discrete(pts, values, inside, n)
    ax = discrete.nearest(axis)
    tg = discrete.nearest(target)
    f_axis = float(polarity * spline.ev(*axis))
    f_target = float(polarity * spline.ev(*target))
    step = float(np.hypot(*(ring.max(axis=0) - ring.min(axis=0)) / (n - 1)))

    level = maximin(discrete, ax, tg)
    samples = []
    pmin = float("nan")
    pmin_at = []
    plen = 0.0
    if level is not None:
        nodes = attaining_path(discrete, ax, tg, level)
        if nodes is not None and len(nodes) > 1:
            xy = discrete.pts[discrete.cell[nodes]]
            vals = discrete.values[discrete.cell[nodes]]
            seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            plen = float(seg.sum())
            k = int(np.argmin(vals))
            pmin = float(vals[k])
            pmin_at = [float(xy[k, 0]), float(xy[k, 1])]
            cum = np.concatenate(([0.0], np.cumsum(seg)))
            spacing = max(plen / 200.0, step)
            t = np.arange(0.0, plen + 1e-12, spacing)
            rr = np.interp(t, cum, xy[:, 0])
            zz = np.interp(t, cum, xy[:, 1])
            at = np.clip(np.searchsorted(cum, t), 0, len(nodes) - 1)
            for a, b in zip(rr, zz):
                samples.append([float(a), float(b), float(polarity * spline.ev(a, b))])

    cut_exists, cut_pts = blocking_cut(discrete, ax, tg, efit_level)
    cut_nova, _ = blocking_cut(discrete, ax, tg, nova_level)
    return Row(
        case=case, source=source, polarity=polarity, verdict=int(record.verdict),
        nova_label=nova_label, efit_label=efit_label,
        axis=[float(axis[0]), float(axis[1])],
        target=[float(target[0]), float(target[1])],
        nova_level=float(nova_level), efit_level=float(efit_level),
        f_axis=f_axis, f_target=f_target,
        maximin_axis_to_target=(float(level) if level is not None else None),
        attaining_path=samples, path_min=pmin, path_min_at=pmin_at,
        path_length=plen, path_samples=len(samples),
        blocking_cut_exists=bool(cut_exists), blocking_cut_points=cut_pts,
        cut_at_nova_level=bool(cut_nova),
        grid_step=step,
    )


def write_row(row: Row, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = row.case.replace(":", "-")
    path_txt = out_dir / f"{stem}-attaining-path.txt"
    lines = [
        f"# {row.case} attaining path on the record's own scipy spline",
        f"# axis {row.axis} f_axis {row.f_axis:.10g}",
        f"# target {row.target} f_target {row.f_target:.10g}",
        f"# nova_level {row.nova_level:.10g} efit_level {row.efit_level:.10g}",
        f"# maximin {row.maximin_axis_to_target} path_min {row.path_min:.10g} "
        f"at {row.path_min_at}",
        f"# grid_step {row.grid_step:.6g} path_length {row.path_length:.6g} "
        f"samples {row.path_samples}",
        "# R Z polarity*psi",
    ]
    for a, b, v in row.attaining_path:
        lines.append(f"{a:.8f} {b:.8f} {v:.10g}")
    path_txt.write_text("\n".join(lines) + "\n")
    cut_txt = out_dir / f"{stem}-blocking-cut.txt"
    clines = [
        f"# {row.case} blocking cut at efit_level {row.efit_level:.10g}",
        f"# exists {row.blocking_cut_exists} points {len(row.blocking_cut_points)}",
        "# R Z   (below-level nodes adjacent to the axis component)",
    ]
    for a, b in row.blocking_cut_points:
        clines.append(f"{a:.8f} {b:.8f}")
    cut_txt.write_text("\n".join(clines) + "\n")
    return path_txt, cut_txt