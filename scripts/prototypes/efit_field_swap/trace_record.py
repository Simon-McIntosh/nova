"""Reader and writer for the imas-efit ``BOUNDARY_LEVEL_TRACE 2`` record.

The field order and layout mirror ``read_boundary_trace_record`` and
``write_boundary_trace_record`` in imas-efit's
``src/EFIT/boundary_level_trace.f90``: a text header naming the version, a
dimension line, a discrete-status line, a real status line, then the r grid,
the z grid, the wall segments, the limiter cell indices, the decision cells,
the wall polygon and the row-major psi grid. Fortran writes the real lines with
``ES26.17E3`` and reads them list-directed, so any whitespace-separated spelling
of 17 significant digits round-trips bit-exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER = "BOUNDARY_LEVEL_TRACE 2"


def _format_es(value: float) -> str:
    """One real in the Fortran ``ES26.17E3`` spelling (17 significant digits)."""
    mantissa, exponent = f"{value: .17E}".split("E")
    sign, digits = exponent[0], exponent[1:].zfill(3)
    return f"{mantissa}E{sign}{digits}"


def _format_row(values: np.ndarray) -> str:
    return " ".join(_format_es(float(value)) for value in np.ravel(values))


def _scan(handle, count: int) -> list[float]:
    """Read ``count`` whitespace-separated reals, spanning lines as needed."""
    out: list[float] = []
    while len(out) < count:
        line = handle.readline()
        if not line:
            raise ValueError("unexpected end of trace record")
        out.extend(float(token) for token in line.split())
    if len(out) != count:
        raise ValueError("trace record line carries too many values")
    return out


def _int_row(handle, count: int) -> list[int]:
    line = handle.readline()
    if not line:
        raise ValueError("unexpected end of trace record")
    return [int(token) for token in line.split()[:count]]


@dataclass(frozen=True)
class TraceRecord:
    """One boundary-level trace record as the Fortran writer lays it out."""

    slice: int
    iteration: int
    verdict: int
    saddle_source_node: int
    wall_source_node: int
    wall_source_segment: int
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

    @property
    def diverted(self) -> bool:
        """The record's own limited-or-diverted label: verdict 2 is diverted."""
        return self.verdict == 2

    @property
    def wall_ring(self) -> np.ndarray:
        """The wall polygon as ordered RZ vertices.

        ``wall_r``/``wall_z`` store the trace wall as consecutive segments
        (``nwall`` rows of two endpoints). Some crossing traces store only the
        polygon, which serves as the wall for the sampled comparison.
        """
        if self.wall_r.size:
            vertices = np.stack((self.wall_r, self.wall_z), axis=-1).reshape(-1, 2)
        elif self.polygon_r.size >= 3:
            vertices = np.column_stack((self.polygon_r, self.polygon_z))
        else:
            raise ValueError("trace has neither wall segments nor a wall polygon")
        keep = np.ones(len(vertices), dtype=bool)
        keep[1:] = np.any(vertices[1:] != vertices[:-1], axis=1)
        ring = vertices[keep]
        if not np.array_equal(ring[-1], ring[0]):
            ring = np.vstack((ring, ring[0]))
        return ring


def read_trace_record(path: str | Path) -> TraceRecord:
    """Parse one version-2 boundary trace record."""
    path = Path(path)
    with path.open() as handle:
        header = handle.readline().strip()
        if header != HEADER:
            raise ValueError(f"unsupported boundary trace header: {header!r}")
        line = [int(token) for token in handle.readline().split()]
        slice_, iteration, verdict, nw, nh, nwall, nlim, ncell, npoly = line
        flags = handle.readline().split()
        saddle_source_node = int(flags[0])
        wall_source_node = int(flags[1])
        wall_source_segment = int(flags[2])
        wall_axis_connected = flags[3] == "T"
        handover_reverted = flags[4] == "T"
        polarity = int(flags[5])
        sweep_armed = flags[6] == "T"
        status = _scan(handle, 7)
        axis_psi, axis_r, axis_z = status[0], status[1], status[2]
        saddle_contested = status[3]
        saddle_refined, wall_level, fed_back_psi = status[4], status[5], status[6]
        rgrid = np.asarray(_scan(handle, nw))
        zgrid = np.asarray(_scan(handle, nh))
        wall_r = np.empty((nwall, 2))
        wall_z = np.empty((nwall, 2))
        for index in range(nwall):
            row = _scan(handle, 4)
            wall_r[index] = row[0:2]
            wall_z[index] = row[2:4]
        wall_index = np.asarray(
            [_int_row(handle, 2) for _ in range(nlim)], dtype=np.int32
        ).reshape(nlim, 2)
        cell_r = np.empty((ncell, 2))
        cell_z = np.empty((ncell, 2))
        for index in range(ncell):
            row = _scan(handle, 4)
            cell_r[index] = row[0:2]
            cell_z[index] = row[2:4]
        polygon_r = np.empty(npoly)
        polygon_z = np.empty(npoly)
        for index in range(npoly):
            row = _scan(handle, 2)
            polygon_r[index] = row[0]
            polygon_z[index] = row[1]
        psi = np.asarray(_scan(handle, nw * nh)).reshape(nw, nh)
    return TraceRecord(
        slice=slice_,
        iteration=iteration,
        verdict=verdict,
        saddle_source_node=saddle_source_node,
        wall_source_node=wall_source_node,
        wall_source_segment=wall_source_segment,
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
        rgrid=rgrid,
        zgrid=zgrid,
        wall_r=wall_r,
        wall_z=wall_z,
        wall_index=wall_index,
        cell_r=cell_r,
        cell_z=cell_z,
        polygon_r=polygon_r,
        polygon_z=polygon_z,
        psi=psi,
    )


def write_trace_record(path: str | Path, record: TraceRecord) -> None:
    """Write one version-2 boundary trace record in the Fortran layout."""
    nw, nh = int(record.rgrid.size), int(record.zgrid.size)
    nwall = int(record.wall_r.shape[0])
    nlim = int(record.wall_index.shape[0])
    ncell = int(record.cell_r.shape[0])
    npoly = int(record.polygon_r.size)
    lines = [HEADER]
    lines.append(
        " ".join(
            str(value)
            for value in (
                record.slice,
                record.iteration,
                record.verdict,
                nw,
                nh,
                nwall,
                nlim,
                ncell,
                npoly,
            )
        )
    )
    lines.append(
        " ".join(
            (
                str(record.saddle_source_node),
                str(record.wall_source_node),
                str(record.wall_source_segment),
                "T" if record.wall_axis_connected else "F",
                "T" if record.handover_reverted else "F",
                str(record.polarity),
                "T" if record.sweep_armed else "F",
            )
        )
    )
    lines.append(
        _format_row(
            np.asarray(
                (
                    record.axis_psi,
                    record.axis_r,
                    record.axis_z,
                    record.saddle_contested,
                    record.saddle_refined,
                    record.wall_level,
                    record.fed_back_psi,
                )
            )
        )
    )
    lines.extend(_format_es(float(value)) for value in record.rgrid)
    lines.extend(_format_es(float(value)) for value in record.zgrid)
    for index in range(nwall):
        lines.append(
            _format_row(
                np.asarray(
                    (
                        record.wall_r[index, 0],
                        record.wall_r[index, 1],
                        record.wall_z[index, 0],
                        record.wall_z[index, 1],
                    )
                )
            )
        )
    for index in range(nlim):
        lines.append(
            f"{int(record.wall_index[index, 0])} {int(record.wall_index[index, 1])}"
        )
    for index in range(ncell):
        lines.append(
            _format_row(
                np.asarray(
                    (
                        record.cell_r[index, 0],
                        record.cell_r[index, 1],
                        record.cell_z[index, 0],
                        record.cell_z[index, 1],
                    )
                )
            )
        )
    for index in range(npoly):
        lines.append(
            _format_row(np.asarray((record.polygon_r[index], record.polygon_z[index])))
        )
    for i in range(nw):
        lines.extend(_format_es(float(record.psi[i, j])) for j in range(nh))
    Path(path).write_text("\n".join(lines) + "\n")
