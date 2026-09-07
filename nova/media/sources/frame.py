"""The machine-neutral records a media figure draws.

A frame carries what one time slice needs and nothing about where it came
from. Two rules make the records safe to mix across archives:

*One flux convention.* ``flux``, ``flux_axis`` and ``flux_boundary`` are
Nova's total poloidal flux in Wb, per :mod:`nova.equilibrium.convention` --
never flux per radian. An adapter converts at the read and records the factor
it applied in :attr:`Pulse.provenance`, so a figure comparing two sources
cannot be showing one of each.

*Absence is NaN, never zero.* A slice with one X-point stores the absent
second as a non-finite pair rather than the origin, which a painter would
otherwise draw as a null on the machine axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class EquilibriumFrame:
    """One time slice of a reconstructed or solved equilibrium."""

    time: float
    radius: np.ndarray
    height: np.ndarray
    flux: np.ndarray
    flux_axis: float
    flux_boundary: float
    psi_norm: np.ndarray
    p_prime: np.ndarray
    ff_prime: np.ndarray
    boundary: np.ndarray
    magnetic_axis: np.ndarray
    x_points: np.ndarray
    strike_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    plasma_current: float = float("nan")

    def __post_init__(self) -> None:
        """Reject a frame whose map does not match its own axes.

        A silently transposed map is the failure this catches: it contours
        without error and draws a plausible, wrong machine.
        """
        expected = (self.height.size, self.radius.size)
        if self.flux.shape != expected:
            raise ValueError(
                f"flux must be shaped (height, radius) = {expected}, "
                f"got {self.flux.shape}"
            )


@dataclass(frozen=True)
class SurfaceFrame:
    """One slice of a solve recorded as flux surfaces rather than a map.

    A rasterless record carries the nested surfaces themselves, each with the
    absolute flux it sits at, so it can be drawn against another source's map
    at shared levels without a 2-D field ever existing. It is a different
    record from :class:`EquilibriumFrame` on purpose: code that needs a map
    cannot accidentally be handed surfaces and contour them.
    """

    time: float
    surface_flux: np.ndarray
    surfaces: tuple[np.ndarray, ...]
    boundary: np.ndarray
    magnetic_axis: np.ndarray
    x_points: np.ndarray
    legs: tuple[np.ndarray, ...] = ()
    strike_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    psi_norm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    p_prime: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ff_prime: np.ndarray = field(default_factory=lambda: np.zeros(0))
    guarded: bool = True
    diverted: bool | None = None
    """Whether the solve classified this slice as diverted rather than limited.

    A LIMITED boundary must touch its limiter by definition, so the class is
    what makes a boundary-to-wall standoff checkable: the same gap is correct
    for a diverted slice and a defect for a limited one."""
    conditioned: bool = False
    """Whether this solve was pinned to the reference's own current centroid.

    A conditioned slice is not a free solve: it was given the reference
    centroid, so showing one beside that reference overstates how much the
    solve found on its own. A figure drawing one must say so."""

    def __post_init__(self) -> None:
        """Reject a frame whose surfaces and their flux values disagree."""
        if len(self.surfaces) != self.surface_flux.size:
            raise ValueError(
                f"{len(self.surfaces)} surfaces carry "
                f"{self.surface_flux.size} flux values"
            )


@dataclass(frozen=True)
class MachineGeometry:
    """Static poloidal geometry: the wall and the conductor sections."""

    limiter: np.ndarray
    coils: tuple[np.ndarray, ...] = ()

    def bounds(self) -> tuple[float, float, float, float]:
        """Return ``(r_min, r_max, z_min, z_max)`` over wall and conductors.

        The conductors are included because a flux map is only shown to cover
        them if the panel does; a bound taken from the wall alone would crop
        the coils a figure exists to show.
        """
        stacked = [np.asarray(self.limiter, dtype=float).reshape(-1, 2)]
        stacked += [np.asarray(coil, dtype=float).reshape(-1, 2) for coil in self.coils]
        points = np.vstack([block for block in stacked if block.size])
        finite = points[np.all(np.isfinite(points), axis=1)]
        if finite.size == 0:
            raise ValueError("the machine geometry carries no finite coordinate")
        return (
            float(np.min(finite[:, 0])),
            float(np.max(finite[:, 0])),
            float(np.min(finite[:, 1])),
            float(np.max(finite[:, 1])),
        )


@dataclass(frozen=True)
class Pulse:
    """Every frame of one pulse, with the geometry they share."""

    machine: str
    identifier: str
    geometry: MachineGeometry
    frames: tuple[EquilibriumFrame, ...]
    provenance: dict = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the frame count."""
        return len(self.frames)

    def extent(self, pad: float = 0.04) -> tuple[float, float, float, float]:
        """Return panel bounds enclosing the machine, widened by ``pad``.

        ``pad`` is a fraction of the larger span rather than of each axis, so
        the widening is isotropic and cannot distort the equal-aspect panel.
        The inboard edge is clamped at the machine axis, since a panel showing
        negative major radius would place empty space where the solenoid is.
        """
        r_min, r_max, z_min, z_max = self.geometry.bounds()
        reach = pad * max(r_max - r_min, z_max - z_min)
        return (
            max(0.0, r_min - reach),
            r_max + reach,
            z_min - reach,
            z_max + reach,
        )

    def times(self) -> np.ndarray:
        """Return every frame's time, in seconds."""
        return np.asarray([frame.time for frame in self.frames], dtype=float)
