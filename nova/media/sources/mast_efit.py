"""Read one MAST EFIT reconstruction from the level-1 shot store.

The store's ``efm`` group is EFIT's own output, so three of its conventions
have to be handled at the read rather than downstream:

*The flux map is stored on a wider profile axis than the grid.* ``psirz`` has
one column per ``profile_r`` point, of which only the ``gridr`` columns carry
finite values. The finite columns are selected and checked against ``gridr``,
because taking the first 65 columns instead would shift the map in R without
raising.

*The stored axes are not uniform at float64 precision.* They are float32 and
reconstructing a uniform axis is required, but the endpoints must be
preserved -- so the axis is rebuilt with :func:`numpy.linspace` between the
stored ends only after confirming the stored spacing is uniform to float32.

*The flux is per radian.* It is multiplied by
:data:`nova.equilibrium.convention.TOTAL_FLUX_FACTOR` once, here, so every
record downstream is Nova's total poloidal flux in Wb.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nova.catalog.mast_geometry import shaped_section_vertices
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.imas.mast_vacuum_cohort import SHOT_STORE
from nova.media.sources.frame import EquilibriumFrame, MachineGeometry, Pulse

_SECTION_COLUMNS = (
    "fcoil_r",
    "fcoil_z",
    "fcoil_width",
    "fcoil_height",
    "fcoil_ang1",
    "fcoil_ang2",
)


def _uniform_axis(stored: np.ndarray, name: str) -> np.ndarray:
    """Return an endpoint-preserving uniform axis, or refuse a ragged one.

    The comparison is against the reconstructed axis at the STORED precision,
    not against the spacing. Differencing float32 coordinates amplifies their
    representation error by the ratio of coordinate to spacing -- here about a
    factor of sixty -- so a spacing test rejects an axis that is uniform to
    every bit the archive actually holds.
    """
    axis = np.asarray(stored)
    if axis.ndim != 1 or axis.size < 2 or not np.all(np.diff(axis) > 0.0):
        raise ValueError(f"{name} must be a one-dimensional increasing axis")
    expected = np.linspace(axis[0], axis[-1], axis.size, dtype=axis.dtype)
    tolerance = 16.0 * np.finfo(axis.dtype).eps * max(1.0, abs(float(axis[-1])))
    deviation = float(np.max(np.abs(axis - expected)))
    if deviation > tolerance:
        raise ValueError(
            f"{name} is non-uniform at {axis.dtype} precision: "
            f"{deviation:.6g} exceeds {tolerance:.6g}"
        )
    return np.linspace(float(axis[0]), float(axis[-1]), axis.size)


def _finite_pairs(radius: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Return the finite ``(r, z)`` pairs of a padded stored polyline."""
    pairs = np.column_stack(
        (np.asarray(radius, dtype=float), np.asarray(height, dtype=float))
    )
    return pairs[np.all(np.isfinite(pairs), axis=1)]


def _flux_map(group: Any, index: int, radius: np.ndarray) -> np.ndarray:
    """Return one slice's flux map on the grid axes, in total Wb."""
    raw = np.asarray(group["psirz"][index], dtype=np.float64)
    columns = np.flatnonzero(np.all(np.isfinite(raw), axis=0))
    if columns.size != radius.size:
        raise ValueError(
            f"psirz carries {columns.size} finite columns for "
            f"{radius.size} radial coordinates"
        )
    stored_profile = np.asarray(group["profile_r"], dtype=np.float64)[columns]
    if not np.allclose(stored_profile, radius, rtol=2.0e-7, atol=1.0e-8):
        raise ValueError("the finite psirz columns do not match the stored gridr")
    return TOTAL_FLUX_FACTOR * raw[:, columns]


def read_geometry(group: Any) -> MachineGeometry:
    """Return the wall polyline and every stored conductor section outline."""
    limiter = _finite_pairs(group["limiterr"], group["limiterz"])
    sections = {name: np.asarray(group[name], dtype=float) for name in _SECTION_COLUMNS}
    coils = tuple(
        shaped_section_vertices(*(sections[name][index] for name in _SECTION_COLUMNS))
        for index in range(sections["fcoil_r"].size)
    )
    return MachineGeometry(limiter=limiter, coils=coils)


def read_frame(group: Any, index: int) -> EquilibriumFrame:
    """Return one time slice as a machine-neutral record."""
    radius = _uniform_axis(np.asarray(group["gridr"]), "efm/gridr")
    height = _uniform_axis(np.asarray(group["gridz"]), "efm/gridz")
    x_points = np.array(
        [
            [float(group["xpoint1_rc"][index]), float(group["xpoint1_zc"][index])],
            [float(group["xpoint2_rc"][index]), float(group["xpoint2_zc"][index])],
        ]
    )
    return EquilibriumFrame(
        time=float(np.asarray(group["time"])[index]),
        radius=radius,
        height=height,
        flux=_flux_map(group, index, radius),
        flux_axis=TOTAL_FLUX_FACTOR * float(group["psi_axis"][index]),
        flux_boundary=TOTAL_FLUX_FACTOR * float(group["psi_boundary"][index]),
        psi_norm=np.asarray(group["psi_norm"], dtype=float),
        p_prime=np.asarray(group["pprime"][index], dtype=float),
        ff_prime=np.asarray(group["ffprime"][index], dtype=float),
        boundary=_finite_pairs(group["lcfs_r"][index], group["lcfs_z"][index]),
        magnetic_axis=np.array(
            [
                float(group["magnetic_axis_r"][index]),
                float(group["magnetic_axis_z"][index]),
            ]
        ),
        x_points=x_points,
        plasma_current=float(group["plasma_current_c"][index]),
    )


def read_pulse(
    shot: int,
    store: Path | str = SHOT_STORE,
    times: Sequence[float] | None = None,
    require_plasma: float = 1.0e4,
) -> Pulse:
    """Return every usable slice of one MAST shot.

    ``require_plasma`` drops slices whose reconstructed plasma current is
    below the given magnitude in amperes. The early and late slices of a shot
    carry a reconstruction with no plasma in it, and an animation that opens
    on those spends its first seconds on an empty vessel.
    """
    import zarr

    source = Path(store) / f"{shot}.zarr"
    if not source.is_dir():
        raise FileNotFoundError(f"no MAST level-1 shot store at {source}")
    group = zarr.open_group(str(source), mode="r")["efm"]
    stored_time = np.asarray(group["time"], dtype=float)
    if times is None:
        selected = np.arange(stored_time.size)
    else:
        selected = np.unique(
            [int(np.argmin(np.abs(stored_time - float(time)))) for time in times]
        )
    current = np.asarray(group["plasma_current_c"], dtype=float)
    kept = [
        index
        for index in selected
        if np.isfinite(current[index]) and abs(current[index]) >= require_plasma
    ]
    if not kept:
        raise ValueError(
            f"MAST {shot} has no slice carrying at least {require_plasma:g} A"
        )
    frames = tuple(read_frame(group, int(index)) for index in kept)
    geometry = read_geometry(group)
    return Pulse(
        machine="MAST",
        identifier=str(shot),
        geometry=geometry,
        frames=frames,
        provenance={
            "source": str(source),
            "group": "efm",
            "stored_flux_unit": "Wb/rad",
            "flux_unit": "Wb",
            "total_flux_factor": float(TOTAL_FLUX_FACTOR),
            "stored_slice_count": int(stored_time.size),
            "selected_slice_count": len(kept),
            "plasma_current_floor_a": float(require_plasma),
            "section_count": len(geometry.coils),
        },
    )
