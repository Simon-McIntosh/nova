"""Read Nova's own labelled solves from a steering-frame session.

A session records each solve as nested flux surfaces with their absolute flux,
the topology points, the divertor legs and the TORAX face profiles. It carries
no 2-D map -- these sessions are rasterless -- so a frame arrives as a
:class:`~nova.media.sources.frame.SurfaceFrame` and is drawn as curves at
known flux rather than contoured. A session may also carry no boundary
polyline, in which case the outermost nested surface supplies it; see
:func:`_boundary`.

Frames are admitted on ``branch_guard_ok``, which is the per-frame validity
flag this schema actually carries; there is no ``converged`` variable in it.
The session's own attributes record COCOS 17 and that the flux-function
gradients come from EFIT, so no convention factor is applied here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nova.media.sources.frame import SurfaceFrame

SESSION_ROOT = Path("/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29")


def _finite_pairs(radius: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Return the finite ``(r, z)`` pairs of a padded stored polyline."""
    pairs = np.column_stack(
        (np.asarray(radius, dtype=float), np.asarray(height, dtype=float))
    )
    return pairs[np.all(np.isfinite(pairs), axis=1)]


def _boundary(surfaces: tuple[np.ndarray, ...], psi_norm: np.ndarray) -> np.ndarray:
    """Return the surface at normalised flux one, which is the LCFS.

    Some sessions store no boundary polyline: their ``lcfs_r`` carries a
    zero-length vertex dimension and ``n_boundary_coords`` is zero for every
    frame. The outermost nested surface sits at normalised flux one and is the
    last closed surface by definition, so it is the boundary rather than a
    substitute for it. The choice is recorded in the provenance so a figure
    caption can say which curve it drew.
    """
    if psi_norm.size != len(surfaces):
        raise ValueError("each surface needs one normalised-flux value")
    return surfaces[int(np.argmax(psi_norm))]


def _frame(dataset, index: int) -> SurfaceFrame:
    """Return one session slice as a rasterless record."""
    take = dataset.isel(time=index)

    def value(name: str) -> np.ndarray:
        return np.asarray(take[name].values, dtype=float)

    surface_r = value("flux_surface_r")
    surface_z = value("flux_surface_z")
    surfaces = tuple(
        _finite_pairs(surface_r[surface], surface_z[surface])
        for surface in range(surface_r.shape[0])
    )
    leg_r = value("divertor_leg_r")
    leg_z = value("divertor_leg_z")
    leg_finite = np.asarray(take["divertor_leg_finite"].values, dtype=bool)
    legs = tuple(
        _finite_pairs(leg_r[leg], leg_z[leg])
        for leg in range(leg_r.shape[0])
        if leg_finite[leg]
    )
    stored_boundary = _finite_pairs(value("lcfs_r"), value("lcfs_z"))
    surface_psi_norm = np.asarray(dataset["flux_surface_psi_norm"].values, dtype=float)
    return SurfaceFrame(
        time=float(np.asarray(take["time"].values, dtype=float)),
        surface_flux=value("flux_surface_psi"),
        surfaces=surfaces,
        boundary=(
            stored_boundary
            if stored_boundary.size
            else _boundary(surfaces, surface_psi_norm)
        ),
        magnetic_axis=np.array(
            [float(value("magnetic_axis_r")), float(value("magnetic_axis_z"))]
        ),
        x_points=_finite_pairs(value("x_point_r"), value("x_point_z")),
        legs=legs,
        strike_points=_finite_pairs(value("strike_points_r"), value("strike_points_z")),
        psi_norm=value("rho_face_norm"),
        p_prime=value("p_prime_face"),
        ff_prime=value("ff_prime_face"),
        guarded=bool(np.asarray(take["branch_guard_ok"].values)),
    )


def read_labels(
    shot: int,
    dirname: Path | str = SESSION_ROOT,
    guarded_only: bool = True,
) -> tuple[tuple[SurfaceFrame, ...], dict]:
    """Return one shot's label frames and the session's provenance.

    ``guarded_only`` keeps the frames whose branch guard passed. An unguarded
    frame is a solve whose branch selection is not trusted, so drawing one
    beside a reconstruction would compare a reference against a solve its own
    author flagged.
    """
    from nova.equilibrium.steering_frames import read_session

    dataset = read_session(filename=str(shot), dirname=str(dirname))
    count = int(dataset.sizes["time"])
    guard = np.asarray(dataset["branch_guard_ok"].values, dtype=bool)
    selected = np.flatnonzero(guard) if guarded_only else np.arange(count)
    if selected.size == 0:
        raise ValueError(f"shot {shot} has no branch-guarded label frame")
    frames = tuple(_frame(dataset, int(index)) for index in selected)
    return frames, {
        "session": str(Path(dirname)),
        "shot": int(shot),
        "cocos": int(dataset.attrs.get("cocos", 0)),
        "p_prime_source": str(dataset.attrs.get("p_prime_source", "")),
        "flux_unit": "Wb",
        "raster": "absent; the session records nested surfaces, not a map",
        "stored_frame_count": count,
        "guarded_frame_count": int(guard.sum()),
        "selected_frame_count": len(frames),
        "admission": "branch_guard_ok" if guarded_only else "every stored frame",
        "surface_count": len(frames[0].surfaces),
        "boundary_source": (
            "stored lcfs polyline"
            if np.any(np.asarray(dataset["n_boundary_coords"].values))
            else "outermost nested surface at normalised flux one"
        ),
    }
