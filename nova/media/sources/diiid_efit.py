"""Read one DIII-D EFIT reconstruction from its IMAS entry.

The entry is opened through IMAS-Python with ``autoconvert=False``, so leaves
arrive in the Data Dictionary version they were written in rather than being
silently migrated.

Two conventions differ from the MAST level-1 store and are handled here:

*The flux is already total.* An equilibrium IDS stores ``profiles_2d.psi`` in
Wb, so no ``2 pi`` factor is applied -- applying one would double-count the
convention that :mod:`nova.media.sources.mast_efit` has to introduce.

*The grid travels with each slice.* ``profiles_2d.grid.dim1`` and ``dim2`` are
per-slice arrays, and the map is indexed ``(dim1, dim2)`` = ``(R, Z)``. A
media frame wants ``(Z, R)``, so the map is transposed once at the read; the
axes are checked for uniformity first, because a non-uniform stored axis would
make the transpose the least of the problems.

The DIII-D coordinate convention is unresolved in this repository, so these
frames carry EFIT's own psi and its own flux-function gradients without any
sign or factor applied. They are drawn against each other, never mixed with a
Nova-convention map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nova.equilibrium.wall_mask import WallUnit, wall_units_from_ids
from nova.media.sources.frame import EquilibriumFrame, MachineGeometry, Pulse

NETCDF_SOURCE = Path("/home/ITER/tribolp/Public/imasdb/DIII-D/200000.nc")


def _uniform_axis(stored: Any, name: str) -> np.ndarray:
    """Return an endpoint-preserving uniform axis, or refuse a ragged one."""
    axis = np.asarray(stored, dtype=float)
    if axis.ndim != 1 or axis.size < 2 or not np.all(np.diff(axis) > 0.0):
        raise ValueError(f"{name} must be a one-dimensional increasing axis")
    expected = np.linspace(axis[0], axis[-1], axis.size)
    tolerance = 1.0e-6 * max(1.0, abs(float(axis[-1])))
    if float(np.max(np.abs(axis - expected))) > tolerance:
        raise ValueError(f"{name} is not uniform to {tolerance:.3g} m")
    return expected


def _points(nodes: Any) -> np.ndarray:
    """Return finite ``(r, z)`` pairs from a structure array of nodes."""
    pairs = [
        [float(nodes[index].r), float(nodes[index].z)] for index in range(len(nodes))
    ]
    array = np.asarray(pairs, dtype=float).reshape(-1, 2)
    return array[np.all(np.isfinite(array), axis=1)]


def _outline(node: Any) -> np.ndarray:
    """Return one stored outline as finite ``(r, z)`` pairs."""
    pairs = np.column_stack(
        (np.asarray(node.r, dtype=float), np.asarray(node.z, dtype=float))
    )
    return pairs[np.all(np.isfinite(pairs), axis=1)]


def _element_outline(element: Any) -> np.ndarray:
    """Return a stored element outline, expanding a rectangle exactly."""
    geometry = element.geometry
    if len(getattr(geometry.outline, "r", ())):
        return _outline(geometry.outline)
    rectangle = geometry.rectangle
    half_width = 0.5 * float(rectangle.width)
    half_height = 0.5 * float(rectangle.height)
    centre_r = float(rectangle.r)
    centre_z = float(rectangle.z)
    return np.asarray(
        [
            [centre_r - half_width, centre_z - half_height],
            [centre_r + half_width, centre_z - half_height],
            [centre_r + half_width, centre_z + half_height],
            [centre_r - half_width, centre_z + half_height],
        ]
    )


def read_geometry(entry: Any) -> MachineGeometry:
    """Return the limiter outline and every active-coil element outline."""
    wall = entry.get("wall", 0, autoconvert=False)
    active = entry.get("pf_active", 0, autoconvert=False)
    units = wall_units_from_ids(wall)
    limiter = units[0].vertices
    coils = tuple(
        _element_outline(active.coil[coil].element[element])
        for coil in range(len(active.coil))
        for element in range(len(active.coil[coil].element))
    )
    geometry = MachineGeometry(limiter=limiter, coils=coils)
    object.__setattr__(geometry, "wall_units", units)
    return geometry


def read_wall_units(entry: Any) -> tuple[WallUnit, ...]:
    """Return every typed limiter unit from the entry's wall description."""

    return wall_units_from_ids(entry.get("wall", 0, autoconvert=False))


def read_frame(equilibrium: Any, index: int) -> EquilibriumFrame:
    """Return one equilibrium time slice as a machine-neutral record."""
    slice_ = equilibrium.time_slice[index]
    profiles = slice_.profiles_2d[0]
    radius = _uniform_axis(profiles.grid.dim1, "profiles_2d/grid/dim1")
    height = _uniform_axis(profiles.grid.dim2, "profiles_2d/grid/dim2")
    stored = np.asarray(profiles.psi, dtype=float)
    if stored.shape != (radius.size, height.size):
        raise ValueError(
            f"profiles_2d/psi must be shaped (dim1, dim2) = "
            f"{(radius.size, height.size)}, got {stored.shape}"
        )
    boundary = slice_.boundary_separatrix
    axis = slice_.global_quantities.magnetic_axis
    one_dimensional = slice_.profiles_1d
    flux = np.asarray(one_dimensional.psi, dtype=float)
    axis_flux, boundary_flux = float(flux[0]), float(flux[-1])
    span = boundary_flux - axis_flux
    return EquilibriumFrame(
        time=float(np.asarray(equilibrium.time, dtype=float)[index]),
        radius=radius,
        height=height,
        flux=stored.T,
        flux_axis=axis_flux,
        flux_boundary=boundary_flux,
        psi_norm=(flux - axis_flux) / span if span else np.zeros_like(flux),
        p_prime=np.asarray(one_dimensional.dpressure_dpsi, dtype=float),
        ff_prime=np.asarray(one_dimensional.f_df_dpsi, dtype=float),
        boundary=_outline(boundary.outline),
        magnetic_axis=np.asarray([float(axis.r), float(axis.z)]),
        x_points=_points(boundary.x_point),
        strike_points=_points(boundary.strike_point),
        plasma_current=float(slice_.global_quantities.ip),
    )


def read_pulse(
    pulse: str | int = 200000,
    source: Path | str = NETCDF_SOURCE,
    stride: int = 4,
    times: Sequence[float] | None = None,
) -> Pulse:
    """Return a strided selection of one DIII-D pulse's equilibrium slices.

    ``stride`` exists because the entry holds 340 slices: a ten-second
    animation of all of them spends 29 ms on each, which is below what a
    viewer resolves, so the default takes every fourth.
    """
    import imas

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"no DIII-D IMAS entry at {path}")
    with imas.DBEntry(str(path), "r") as entry:
        equilibrium = entry.get("equilibrium", 0, autoconvert=False)
        homogeneous = int(equilibrium.ids_properties.homogeneous_time)
        stored_time = np.asarray(equilibrium.time, dtype=float)
        if times is None:
            selected = np.arange(0, stored_time.size, max(int(stride), 1))
        else:
            selected = np.unique(
                [int(np.argmin(np.abs(stored_time - float(time)))) for time in times]
            )
        frames = tuple(read_frame(equilibrium, int(index)) for index in selected)
        geometry = read_geometry(entry)
        version = str(equilibrium.ids_properties.version_put.data_dictionary).strip()
    return Pulse(
        machine="DIII-D",
        identifier=str(pulse),
        geometry=geometry,
        frames=frames,
        provenance={
            "source": str(path),
            "ids": "equilibrium",
            "dd_version": version,
            "homogeneous_time": homogeneous,
            "flux_unit": "Wb",
            "total_flux_factor": 1.0,
            "convention": "EFIT stored psi, no sign or factor applied",
            "stored_slice_count": int(stored_time.size),
            "selected_slice_count": len(frames),
            "stride": int(stride),
            "section_count": len(geometry.coils),
        },
    )
