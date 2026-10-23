"""Read imas-efit boundary traces and build the spline field.

The field-swap trace reader owns the versioned record layout.

Nothing in this module evaluates a field with adapter code: the field for
every record is the record's own not-a-knot bicubic spline of its 65x65
grid, built with scipy.
"""

from __future__ import annotations

from dataclasses import dataclass
import pathlib

import numpy as np
from scipy.interpolate import RectBivariateSpline

from scripts.prototypes.efit_field_swap.trace_record import read_trace_record


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
        if node is None or not 1 <= node <= n_r * n_z:
            return None
        ir, iz = divmod(node - 1, n_z)
        return np.array([self.rgrid[ir], self.zgrid[iz]])

    def spline(self) -> RectBivariateSpline:
        """Not-a-knot bicubic interpolant of psi on (rgrid, zgrid)."""
        return RectBivariateSpline(self.rgrid, self.zgrid, self.psi, kx=3, ky=3, s=0)


def parse_record(path: pathlib.Path | str) -> TraceRecord:
    """Adapt the shared versioned trace reader to this analysis interface."""
    source = read_trace_record(path)
    return TraceRecord(
        path=pathlib.Path(path),
        slice=source.slice,
        iteration=source.iteration,
        verdict=source.verdict,
        rgrid=source.rgrid,
        zgrid=source.zgrid,
        wall_r=source.wall_r,
        wall_z=source.wall_z,
        wall_index=source.wall_index,
        cell_r=source.cell_r,
        cell_z=source.cell_z,
        polygon_r=source.polygon_r,
        polygon_z=source.polygon_z,
        psi=source.psi,
        saddle_source_node=source.saddle_source_node,
        wall_source_node=source.wall_source_node,
        wall_axis_connected=source.wall_axis_connected,
        handover_reverted=source.handover_reverted,
        polarity=source.polarity,
        sweep_armed=source.sweep_armed,
        axis_psi=source.axis_psi,
        axis_r=source.axis_r,
        axis_z=source.axis_z,
        saddle_contested=source.saddle_contested,
        saddle_refined=source.saddle_refined,
        wall_level=source.wall_level,
        fed_back_psi=source.fed_back_psi,
    )
