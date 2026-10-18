"""Read imas-efit BOUNDARY_LEVEL_TRACE 2 records and build the spline field.

The record layout follows ``read_boundary_trace_record`` in the imas-efit
source tree (src/EFIT/boundary_level_trace.f90).  The same fields are read
here so the analysis shares no code with either nova field-swap adapter.

Nothing in this module evaluates a field with adapter code: the field for
every record is the record's own not-a-knot bicubic spline of its 65x65
grid, built with scipy.
"""

from __future__ import annotations

from dataclasses import dataclass
import pathlib

import numpy as np
from scipy.interpolate import RectBivariateSpline


@dataclass
class TraceRecord:
    path: pathlib.Path
    slice: int
    iteration: int
    verdict: int
    rgrid: np.ndarray
    zgrid: np.ndarray
    wall_r: np.ndarray
    wall_z: np.ndarray
    wall_index: np.ndarray
    cell_r: np.ndarray
    cell_z: np.ndarray
    polygon_r: np.ndarray
    polygon_z: np.ndarray
    psi: np.ndarray
    saddle_source_node: int
    wall_source_node: int
    wall_axis_connected: bool
    handover_reverted: bool
    polarity: int
    sweep_armed: bool
    axis_psi: float
    axis_r: float
    axis_z: float
    saddle_contested: float
    saddle_refined: float
    wall_level: float
    fed_back_psi: float

    @property
    def label(self) -> str:
        return "diverted" if self.verdict == 2 else "limited"

    @property
    def wall_polygon(self) -> np.ndarray:
        return np.column_stack((self.polygon_r, self.polygon_z))

    @property
    def saddle_source_position(self) -> np.ndarray | None:
        node = self.saddle_source_node
        n_r, n_z = len(self.rgrid), len(self.zgrid)
        if not 1 <= node <= n_r * n_z:
            return None
        ir, iz = divmod(node - 1, n_z)
        return np.array([self.rgrid[ir], self.zgrid[iz]])

    def spline(self) -> RectBivariateSpline:
        """Not-a-knot bicubic interpolant of psi on (rgrid, zgrid)."""
        return RectBivariateSpline(self.rgrid, self.zgrid, self.psi, kx=3, ky=3, s=0)


def parse_record(path: pathlib.Path | str) -> TraceRecord:
    path = pathlib.Path(path)
    lines = path.read_text().splitlines()
    header = lines[0].strip()
    if header != "BOUNDARY_LEVEL_TRACE 2":
        raise ValueError(f"unsupported boundary trace header: {header!r} in {path}")

    counts = [int(v) for v in lines[1].split()]
    if len(counts) != 9:
        raise ValueError(f"expected 9 counts on line 2 of {path}: {counts}")
    _slice, iteration, verdict, nw, nh, nwall, nlim, ncell, npoly = counts

    meta = lines[2].split()
    saddle_source_node = int(meta[0])
    wall_source_node = int(meta[1])
    wall_axis_connected = meta[3] in ("T", "t", ".TRUE.")
    handover_reverted = meta[4] in ("T", "t", ".TRUE.")
    polarity = int(meta[5])
    sweep_armed = meta[6] in ("T", "t", ".TRUE.")

    scalars = [float(v) for v in lines[3].split()]
    (
        axis_psi,
        axis_r,
        axis_z,
        saddle_contested,
        saddle_refined,
        wall_level,
        fed_back_psi,
    ) = scalars

    body = " ".join(lines[4:]).split()
    values = np.array([float(v) for v in body], dtype=float)
    expected = nw + nh + nwall * 4 + nlim * 2 + ncell * 4 + npoly * 2 + nw * nh
    if values.size != expected:
        raise ValueError(f"token count {values.size} != expected {expected} in {path}")

    cursor = 0

    def take(n: int) -> np.ndarray:
        nonlocal cursor
        chunk = values[cursor : cursor + n]
        cursor += n
        return chunk

    rgrid = take(nw)
    zgrid = take(nh)
    wall = take(nwall * 4).reshape(nwall, 4)
    wall_r = wall[:, :2]
    wall_z = wall[:, 2:]
    wall_index = take(max(nlim, 0) * 2).reshape(nlim, 2) if nlim else np.zeros((0, 2))
    cells = take(ncell * 4).reshape(ncell, 4)
    cell_r = cells[:, :2]
    cell_z = cells[:, 2:]
    poly = take(npoly * 2).reshape(npoly, 2)

    psi = take(nw * nh).reshape(nw, nh)

    return TraceRecord(
        path=path,
        slice=_slice,
        iteration=iteration,
        verdict=verdict,
        rgrid=rgrid,
        zgrid=zgrid,
        wall_r=wall_r,
        wall_z=wall_z,
        wall_index=wall_index,
        cell_r=cell_r,
        cell_z=cell_z,
        polygon_r=poly[:, 0],
        polygon_z=poly[:, 1],
        psi=psi,
        saddle_source_node=saddle_source_node,
        wall_source_node=wall_source_node,
        wall_axis_connected=wall_axis_connected,
        handover_reverted=handover_reverted,
        polarity=polarity,
        sweep_armed=sweep_armed,
        axis_psi=axis_psi,
        axis_r=axis_r,
        axis_z=axis_z,
        saddle_contested=saddle_contested,
        saddle_refined=saddle_refined,
        wall_level=wall_level,
        fed_back_psi=fed_back_psi,
    )