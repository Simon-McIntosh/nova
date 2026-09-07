"""Read Nova's own labelled solves from a steering-frame session.

A session records each solve as nested flux surfaces with their absolute flux,
the topology points, the divertor legs and the TORAX face profiles. It carries
no 2-D map -- these sessions are rasterless -- so a frame arrives as a
:class:`~nova.media.sources.frame.SurfaceFrame` and is drawn as curves at
known flux rather than contoured.

The boundary is the stored LCFS polyline and nothing else. The writer persists
``lcfs_r``/``lcfs_z`` as the authored boundary -- the raster-derived separatrix
when one exists, else the outermost traced surface closed onto itself -- so the
reader takes the stored polyline as authoritative and refuses a frame that
carries none. There is no fallback to the outermost nested surface: a session
that mixes stored and unstored frames must fail rather than silently draw two
boundary sources inside one corpus.

Frames are admitted on ``branch_guard_ok``, the per-frame validity flag this
schema carries; there is no ``converged`` variable in the frame, by design --
the conditioning and guard flags live per slice in the ``.npz`` companion
beside each session file, which :func:`read_labels` reads and aligns by time.

That companion carries the fact a caption depends on: ``conditioned`` marks a
slice that was pinned to the reference's own current centroid. Such a slice is
not a free solve, so it is excluded by default rather than drawn as one --
measured on shot 27079, 23 of 114 slices are conditioned and 81 are both
guarded and free.

The session's own attributes record COCOS 17 and that the flux-function
gradients come from EFIT, so no convention factor is applied here.
"""

from __future__ import annotations

from dataclasses import replace
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
    if stored_boundary.size == 0:
        raise ValueError(
            "the session's slice has no stored boundary polyline: the label "
            "reader takes lcfs_r/lcfs_z as the authoritative boundary and "
            "refuses a frame without one rather than substituting the "
            "outermost nested surface"
        )
    return SurfaceFrame(
        time=float(np.asarray(take["time"].values, dtype=float)),
        surface_flux=value("flux_surface_psi"),
        surfaces=surfaces,
        boundary=stored_boundary,
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
        diverted=(
            bool(np.asarray(take["diverted"].values))
            if "diverted" in dataset.data_vars
            else None
        ),
    )


def _conditioning(shot: int, dirname: Path, time: np.ndarray) -> np.ndarray:
    """Return the per-slice conditioned flag, aligned to the frame times.

    Alignment is by time equality rather than by the companion's ``row``
    column: that column is the upstream source index -- it starts at 11 on
    shot 27079 -- so using it to index frames would silently shift the flags.
    The companion's own time vector is bit-identical to the session's, so an
    exact comparison is the right check and a mismatch is a refusal.
    """
    companion = Path(dirname) / f"{shot}.npz"
    if not companion.exists():
        return np.zeros(time.size, dtype=bool)
    with np.load(companion, allow_pickle=True) as loaded:
        if "conditioned" not in loaded.files:
            return np.zeros(time.size, dtype=bool)
        companion_time = np.asarray(loaded["time"], dtype=float)
        conditioned = np.asarray(loaded["conditioned"], dtype=bool)
    if companion_time.size != time.size or not np.array_equal(companion_time, time):
        raise ValueError(
            f"the {shot}.npz companion does not share the session's time base"
        )
    return conditioned


def read_labels(
    shot: int,
    dirname: Path | str = SESSION_ROOT,
    guarded_only: bool = True,
    free_only: bool = True,
) -> tuple[tuple[SurfaceFrame, ...], dict]:
    """Return one shot's label frames and the session's provenance.

    ``guarded_only`` keeps the frames whose branch guard passed. An unguarded
    frame is a solve whose branch selection is not trusted, so drawing one
    beside a reconstruction would compare a reference against a solve its own
    author flagged.

    ``free_only`` drops the conditioned slices. Keep it on for any figure that
    presents the solve as Nova's own: a conditioned slice was handed the
    reference current centroid, so including one silently credits the solve
    with a position it was given.
    """
    from nova.equilibrium.steering_frames import read_session

    dataset = read_session(filename=str(shot), dirname=str(dirname))
    count = int(dataset.sizes["time"])
    guard = np.asarray(dataset["branch_guard_ok"].values, dtype=bool)
    time = np.asarray(dataset["time"].values, dtype=float)
    conditioned = _conditioning(shot, Path(dirname), time)
    admitted = np.ones(count, dtype=bool)
    if guarded_only:
        admitted &= guard
    if free_only:
        admitted &= ~conditioned
    selected = np.flatnonzero(admitted)
    if selected.size == 0:
        raise ValueError(f"shot {shot} has no admissible label frame")
    frames = tuple(
        replace(_frame(dataset, int(index)), conditioned=bool(conditioned[index]))
        for index in selected
    )
    return frames, {
        "session": str(Path(dirname)),
        "shot": int(shot),
        "cocos": int(dataset.attrs.get("cocos", 0)),
        "p_prime_source": str(dataset.attrs.get("p_prime_source", "")),
        "flux_unit": "Wb",
        "raster": "absent; the session records nested surfaces, not a map",
        "stored_frame_count": count,
        "guarded_frame_count": int(guard.sum()),
        "conditioned_frame_count": int(conditioned.sum()),
        "free_guarded_frame_count": int((guard & ~conditioned).sum()),
        "selected_frame_count": len(frames),
        "admission": ", ".join(
            filter(
                None,
                (
                    "branch_guard_ok" if guarded_only else "",
                    "not conditioned on the reference centroid" if free_only else "",
                ),
            )
        )
        or "every stored frame",
        "conditioning_source": f"{shot}.npz companion, aligned by time",
        "surface_count": len(frames[0].surfaces),
        "boundary_source": (
            "stored lcfs polyline; a frame without one is refused, never "
            "substituted from the nested surfaces"
        ),
    }
