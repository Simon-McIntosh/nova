"""Does the as-built winding PATH explain what the smeared coil misses?

A poloidal field coil ships as a uniform current density over a gross
rectangular outline, and the machine description sits above that continuum on
every self term.  A companion study asked whether CONCENTRATING the current
into discrete conductors inside the same outline accounts for the offset, and
found that it does not on its own.  This study changes the other two things the
continuum asserts rather than measures: the number of turns, and the path they
follow.

Both are available as built.  The ITER machine description carries the
conductor CENTRELINES of all six poloidal field coils and all six central
solenoid modules as sequences of arc and line elements in wrapped cylindrical
coordinates -- one conductor per coil, at one turn, so the winding is resolved
GEOMETRICALLY rather than smeared.  The path includes the feeders that run out
to the coil terminals, which are conductor but are not winding, so it has to be
separated before anything is counted or coupled.

Three things follow, and they are the three the study measures:

    turns       unwrap the toroidal angle along the winding pack and divide the
                swept angle by two pi.  This is a COUNT, not a fit: it needs no
                model and no inductance.
    path        the pack is a helix with joggles and layer transitions, not a
                stack of coaxial rings.  Flux linkage along the real path is
                A.dl summed over it, which the axisymmetric operator cannot
                express at all.
    section     the conductor has a finite cross-section.  This study leaves it
                declared rather than resolved, and says so in every number.

The self term of a filament diverges, so a filamentary winding cannot supply
its own diagonal.  The split used here is by TURN and is stated wherever a
number is quoted:

    turn i against turn j, i != j    from the as-built 3-D path, A.dl
    turn i against itself            from an axisymmetric ring at that turn's
                                     own mean radius and height, carrying the
                                     true cable-space disc section

So the PATH and the TURN COUNT are as built, and only the per-turn self term is
declared.  That is the honest reading of the result: it is a measurement of what
the winding geometry buys, with the conductor section held as an input.

The continuum side has to be CONVERGED before it can be compared.  The shipped
mesh overshoots the converged value by more than the whole effect under study,
so the continuum reference is rebuilt at a fine mesh and at the MEASURED turn
count, which separates "the model has the wrong number of turns" from "the
model has the wrong winding".

Stages, each writing its own JSON beside the figures:

    python benchmarks/wound_coil_inductance.py turns
    python benchmarks/wound_coil_inductance.py wound
    python benchmarks/wound_coil_inductance.py converge
    python benchmarks/wound_coil_inductance.py figures

The turn count is seconds on a login node.  The rest wants a compute node, and a
CPU one: the tiled operator reaches the axisymmetric polygon section only, so
there is no device route to the arc and line kernels a filamentary winding is
built from, and the host assembly is fast enough that there is nothing to gain
by looking for one -- a hundred million filament pairs is under a minute.  What
does cost is building the winding: the swept volume of a five-thousand-element
path is a few minutes, so it is built once and reused across a sweep.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
import math
import pathlib
import time

import numpy as np

from benchmarks.cs_winding_inductance import (
    MACHINE_DESCRIPTION,
    MODULES,
    Conductor,
    continuum_coilset,
    reduced_inductance,
)

FIGURES = (
    pathlib.Path(__file__).resolve().parents[1] / "docs" / "figures" / "wound-coil"
)

# The as-built centrelines.  Neither pulse is catalogued in the machine
# description summary; the description is carried by the IDS comment
# ("Poloidal Field Coils - conductor centerlines", "Central Solenoid Modules -
# conductor centerlines").  Both were written by a development build of the data
# dictionary whose version stamp is a git-describe suffix, so the reader has to
# pin 3.39.0 AND accept the unknown stamp -- accepting it alone silently
# converts to the latest dictionary.
CENTRELINE = {"PF": (111005, 1), "CS": (111004, 1)}
DD_VERSION = "3.39.0"

# The IDS names its coils in full.  This is the map to the short names the rest
# of the machine description, the model and this study all use.
SHORT_NAME = {
    "Poloidal Field Coil 1": "PF1",
    "Poloidal Field Coil 2": "PF2",
    "Poloidal Field Coil 3": "PF3",
    "Poloidal Field Coil 4": "PF4",
    "Poloidal Field Coil 5": "PF5",
    "Poloidal Field Coil 6": "PF6",
    "Central Solenoid Module 1 Upper": "CS1U",
    "Central Solenoid Module 2 Upper": "CS2U",
    "Central Solenoid Module 3 Upper": "CS3U",
    "Central Solenoid Module 1 Lower": "CS1L",
    "Central Solenoid Module 2 Lower": "CS2L",
    "Central Solenoid Module 3 Lower": "CS3L",
}


@dataclass(frozen=True)
class Outline:
    """A coil's gross rectangular outline and its nominal turn count.

    Used for two things that must not be confused: separating the winding pack
    from the feeders (a geometric test, needing only a box), and providing the
    nominal count the measured one is compared against.
    """

    name: str
    radius: float
    height: float
    width: float
    thickness: float
    nturn: float

    def encloses(self, radius, height, inflate: float) -> np.ndarray:
        """Return which (radius, height) lie in the outline inflated by a margin."""
        return (np.abs(radius - self.radius) <= 0.5 * self.width + inflate) & (
            np.abs(height - self.height) <= 0.5 * self.thickness + inflate
        )

    @property
    def outer_radius(self) -> float:
        """Return the outline's largest major radius."""
        return self.radius + 0.5 * self.width


# The model geometry, from the poloidal field coil table nova ships.  The six
# solenoid modules are quoted in BOTH revisions of that table, because the
# revisions disagree by 150 mm in module elevation and the as-built pack is
# centred on one of them -- which revision encloses the pack is a measurement
# this study reports rather than a convention it picks.
POLOIDAL_FIELD = (
    Outline("PF1", 3.9431, 7.5641, 0.9590, 0.9841, 248.64),
    Outline("PF2", 8.2851, 6.5298, 0.5801, 0.7146, 115.20),
    Outline("PF3", 11.9919, 3.2652, 0.6963, 0.9538, 185.92),
    Outline("PF4", 11.9630, -2.2336, 0.6382, 0.9538, 169.92),
    Outline("PF5", 8.3908, -6.7369, 0.8125, 0.9538, 216.80),
    Outline("PF6", 4.3340, -7.4765, 1.5590, 1.1075, 459.36),
)

SOLENOID = {
    "wide": (
        Outline("CS3U", 1.6870, 5.4640, 0.7400, 2.093, 554),
        Outline("CS2U", 1.6870, 3.2780, 0.7400, 2.093, 554),
        Outline("CS1U", 1.6870, 1.0920, 0.7400, 2.093, 554),
        Outline("CS1L", 1.6870, -1.0720, 0.7400, 2.093, 554),
        Outline("CS2L", 1.6870, -3.2580, 0.7400, 2.093, 554),
        Outline("CS3L", 1.6870, -5.4440, 0.7400, 2.093, 554),
    ),
    "narrow": (
        Outline("CS3U", 1.722, 5.313, 0.719, 2.075, 554),
        Outline("CS2U", 1.722, 3.188, 0.719, 2.075, 554),
        Outline("CS1U", 1.722, 1.063, 0.719, 2.075, 554),
        Outline("CS1L", 1.722, -1.063, 0.719, 2.075, 554),
        Outline("CS2L", 1.722, -3.188, 0.719, 2.075, 554),
        Outline("CS3L", 1.722, -5.313, 0.719, 2.075, 554),
    ),
}

# The three candidate turn counts for the upper poloidal field coil, which the
# as-built centreline arbitrates with no model in the loop.  The last is what
# the coil's inductance gap to the machine description is worth at the measured
# dL/dN, so agreement there would mean the gap IS a turn count.
PF1_CANDIDATES = {
    "machine description": 248.6,
    "design description": 248.64,
    "inductance gap": 249.36,
}


@dataclass
class Path:
    """One coil's as-built conductor centreline, and where its winding is.

    ``kind`` is the element type per the data dictionary: 1 a straight line
    between its two end points, 2 a circular arc through its intermediate one.
    ``start``, ``middle`` and ``end`` are ``(n, 3)`` cylindrical ``(r, phi, z)``
    -- middle is meaningless for a line and is not read for one.
    """

    name: str
    kind: np.ndarray
    start: np.ndarray
    middle: np.ndarray
    end: np.ndarray
    turns: float
    resolution: int = 8
    pack: np.ndarray = field(init=False, default=None)
    _angle: list = field(init=False, repr=False, default=None)
    _segments: list = field(init=False, repr=False, default=None)

    def __len__(self):
        """Return the element count."""
        return len(self.kind)

    @property
    def is_arc(self) -> np.ndarray:
        """Return which elements are arcs."""
        return self.kind == 2

    def cartesian(self, cylindrical: np.ndarray) -> np.ndarray:
        """Return an ``(n, 3)`` cylindrical point array as ``(x, y, z)``."""
        radius, phi, height = cylindrical.T
        return np.c_[radius * np.cos(phi), radius * np.sin(phi), height]

    @property
    def segments(self) -> list:
        """Return every element as the geometry object the field kernels take.

        An element carries three stored points, and three points do not fix the
        toroidal angle it sweeps: the angles are stored WRAPPED, so recovering
        the sweep means unwrapping, and unwrapping a step of more than half a
        turn is not merely inaccurate, it silently loses a whole turn.  Some of
        these arcs run to 200 degrees between stored points, which is close
        enough to the limit that a three-point count cannot be trusted -- and
        the stored intermediate point does not bisect the arc, so the two halves
        are not even equal.

        Three points on a circle fix it exactly, so the fix is to resolve the
        arc GEOMETRICALLY and read the sweep off its own circle, leaving nothing
        to argue about.  The object that does it is the same one that hands the
        element to the field kernels, so the count and the coupling are taken
        from one geometry rather than two.
        """
        if self._segments is None:
            self._segments = [
                _segment(
                    self.cartesian(self.start[element : element + 1])[0],
                    self.cartesian(self.middle[element : element + 1])[0],
                    self.cartesian(self.end[element : element + 1])[0],
                    self.is_arc[element],
                )
                for element in range(len(self))
            ]
        return self._segments

    def elements(self, samples: int = 0) -> list[np.ndarray]:
        """Return each element resolved as an ``(n, 3)`` cartesian point run."""
        return [_resolve(segment, samples) for segment in self.segments]

    @property
    def resolved_angle(self) -> list[np.ndarray]:
        """Return the unwrapped toroidal angle sampled along every element [rad]."""
        if self._angle is None:
            self._angle = [
                np.unwrap(np.arctan2(run[:, 1], run[:, 0]))
                for run in self.elements(self.resolution)
            ]
        return self._angle

    @property
    def element_sweep(self) -> np.ndarray:
        """Return the toroidal angle each element sweeps, in turns."""
        return np.array(
            [(angle[-1] - angle[0]) / (2 * np.pi) for angle in self.resolved_angle]
        )

    @property
    def step(self) -> float:
        """Return the largest angular step between resolved samples [rad].

        The guard on the count: a step approaching half a turn is a step whose
        unwrapping cannot be verified from the data, so the resolution is only
        adequate while this stays comfortably below it.
        """
        return float(
            max(np.max(np.abs(np.diff(angle))) for angle in self.resolved_angle)
        )

    def swept(self, index=None) -> float:
        """Return the toroidal angle swept along a contiguous run, in turns.

        Consecutive elements share an end point, so a run's sweep is the sum of
        its elements' -- there is no joint left to unwrap across.
        """
        sweep = self.element_sweep
        return float(sweep.sum() if index is None else sweep[np.asarray(index)].sum())

    def winding_box(self, minimum_sweep: float) -> np.ndarray:
        """Return the poloidal extent of the elements that WIND, as ``(2, 2)``.

        The pack has to be bounded before it can be separated from the feeders,
        and bounding it by the model's gross outline does not work: the as-built
        packs carry an outer cross-over layer at a much coarser pitch than the
        winding, which reaches beyond the outline by up to 0.44 m and is real
        conductor.  An outline box clips it and loses most of a turn.

        So the bound is taken from the path itself, from the elements that sweep
        toroidally.  A feeder run is radial and axial travel at essentially
        fixed angle -- it sweeps a few thousandths of a turn over its whole
        length -- so a sweep threshold excludes the feeders from the box without
        reference to any coil geometry, and the box it leaves is the winding's
        own.
        """
        winds = np.abs(self.element_sweep) > minimum_sweep
        points = np.r_[self.start[winds][:, [0, 2]], self.end[winds][:, [0, 2]]]
        return np.c_[points.min(axis=0), points.max(axis=0)]

    def classify(self, minimum_sweep: float, inflate: float) -> dict:
        """Split the path into winding pack and feeders, and say how.

        An element is pack when BOTH its end points lie inside the winding box
        grown by one jacket width, and the run between the first and last such
        element is then filled -- which keeps the joggles and layer transitions
        that leave the box briefly and come back.

        A feeder is AXIAL while it stays within the winding's own major radius
        and RADIAL once it runs outboard of it, which is the order the terminals
        are reached in and separates the two legs of the run.
        """
        box = self.winding_box(minimum_sweep)
        inside = np.ones(len(self), dtype=bool)
        for point in (self.start, self.end):
            for axis, column in enumerate((0, 2)):
                inside &= point[:, column] >= box[axis, 0] - inflate
                inside &= point[:, column] <= box[axis, 1] + inflate
        pack = inside.copy()
        first, last = np.flatnonzero(pack)[[0, -1]]
        pack[first : last + 1] = True
        self.pack = pack
        outboard = np.maximum(self.start[:, 0], self.end[:, 0]) > box[0, 1] + inflate
        return {
            "elements": len(self),
            "pack": int(pack.sum()),
            "pack_lines": int((pack & ~self.is_arc).sum()),
            "pack_arcs": int((pack & self.is_arc).sum()),
            "filled": int(pack.sum() - inside.sum()),
            "feeder_axial": int((~pack & ~outboard).sum()),
            "feeder_radial": int((~pack & outboard).sum()),
            "first_pack": int(first),
            "last_pack": int(last),
            "winding_box": box.tolist(),
            "path_radius_max": float(self.start[:, 0].max()),
        }


def _segment(start, middle, end, is_arc: bool):
    """Return one path element as a nova geometry segment.

    A filament element takes no section, so a line's stored normal is unused by
    the field it produces -- but the constructor orthogonalises against it and
    divides by the result, so a normal parallel to the line is a division by
    zero.  It is built here from whichever coordinate axis is least aligned with
    the line, which cannot be parallel to it.
    """
    from nova.geometry.polyline import Arc, Line

    if is_arc:
        return Arc(np.stack([start, middle, end]))
    axis = end - start
    axis /= np.linalg.norm(axis)
    normal = np.cross(axis, np.identity(3)[np.argmin(np.abs(axis))])
    return Line(np.stack([start, end]), normal)


def _resolve(segment, samples: int) -> np.ndarray:
    """Return a segment as an ``(n, 3)`` point run, arcs on their own circle."""
    if segment.name == "line":
        return np.linspace(segment.start_point, segment.end_point, max(samples, 2))
    return segment.sample(max(samples, 3))


def _stations(segment, intervals: int) -> tuple[np.ndarray, np.ndarray]:
    """Return midpoint quadrature stations along a segment, with exact ``dl``.

    A flux linkage is ``A.dl`` along the conductor, so the quadrature has to get
    both the station and the step right.  Sampling an arc and joining the
    samples by chords gets neither: a chord is shorter than the arc it spans by
    one part in ``24/dtheta**2``, which at a handful of samples per element is
    two parts in a thousand of the whole linkage -- four times the effect under
    study.

    So the arc is sampled at twice the interval count and the ODD stations are
    taken.  Those are the true arc midpoints, and a circular arc's tangent at
    its midpoint is parallel to its chord, so the step is the chord rescaled to
    the arc length it stands for.  What is left is second order in the interval
    with no first-order length defect.
    """
    run = _resolve(segment, 2 * intervals + 1)
    station = run[1::2]
    chord = run[2::2] - run[:-1:2]
    length = np.linalg.norm(chord, axis=1)
    if segment.name == "line":
        return station, chord
    return station, chord * (segment.length / intervals / length)[:, np.newaxis]


def _split_parameter(segment, sweep: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Return where along a segment the cumulative sweep crosses each boundary.

    Returned as a fraction of the segment, from a fine resample of its own
    geometry.  The toroidal angle is not linear in the segment parameter for
    either kind of element -- an arc off the machine axis and a straight chord
    both sweep unevenly -- so the crossing is interpolated from the resolved
    angle rather than assumed proportional.
    """
    run = _resolve(segment, 256)
    angle = np.unwrap(np.arctan2(run[:, 1], run[:, 0]))
    cumulative = sweep[0] + (angle - angle[0]) / (2 * np.pi)
    fraction = np.linspace(0.0, 1.0, len(run))
    if cumulative[-1] < cumulative[0]:
        cumulative, fraction = cumulative[::-1], fraction[::-1]
    return np.interp(boundary, cumulative, fraction)


def _subsegment(segment, first: float, last: float):
    """Return the piece of a segment between two fractions of it."""
    if segment.name == "line":
        start, end = segment.start_point, segment.end_point
        return _segment(
            start + first * (end - start), None, start + last * (end - start), False
        )
    theta = segment.theta[0] + np.array([first, 0.5 * (first + last), last]) * (
        segment.theta[-1] - segment.theta[0]
    )
    points = segment.center[np.newaxis, :] + segment.radius * (
        np.cos(theta)[:, np.newaxis] * -segment.arc_axes[np.newaxis, 1]
        + np.sin(theta)[:, np.newaxis] * segment.arc_axes[np.newaxis, 0]
    )
    return _segment(points[0], points[1], points[2], True)


def read_paths(part: str) -> list[Path]:
    """Return every coil centreline in one machine-description pulse.

    The pulse stamp is a development build of the data dictionary, so the read
    pins the release it derives from AND accepts the unrecognised suffix.
    Accepting the suffix alone converts the whole IDS to the latest dictionary,
    which changes field semantics silently.
    """
    import imas

    pulse, run = CENTRELINE[part]
    uri = (
        "imas:hdf5?path=/work/imas/shared/imasdb/"
        f"ITER_MACHINE_DESCRIPTION/3/{pulse}/{run}"
    )
    ids = imas.DBEntry(uri, "r", dd_version=DD_VERSION).get(
        "coils_non_axisymmetric", ignore_unknown_dd_version=True
    )
    paths = []
    for coil in ids.coil:
        elements = coil.conductor[0].elements
        paths.append(
            Path(
                name=SHORT_NAME[str(coil.name)],
                kind=np.asarray(elements.types),
                start=_points(elements.start_points),
                middle=_points(elements.intermediate_points),
                end=_points(elements.end_points),
                turns=float(coil.turns),
            )
        )
    return paths


def _points(node) -> np.ndarray:
    """Return an IDS cylindrical point structure as an ``(n, 3)`` array."""
    return np.c_[
        np.asarray(node.r, dtype=float),
        np.asarray(node.phi, dtype=float),
        np.asarray(node.z, dtype=float),
    ]


def section_radius(part: str) -> dict[str, float]:
    """Return each conductor's cross-section radius, measured from the IDS [m].

    The section is stored in POLAR form, which is not what the field names say:
    ``delta_r`` is a radius held constant round the outline, ``delta_phi`` steps
    the polar ANGLE about the section centre through eight stations and a closing
    repeat, and ``delta_z`` is the out-of-plane offset that tilts the disc into
    the plane the path's own start tangent defines.  Read as the cylindrical
    triple the names suggest, the toroidal angle would be taken for a length and
    the answer comes out in metres of major radius.

    The radius is measured rather than asserted because the two pulses disagree
    about what it means: the field coils carry the cable-space RADIUS and the
    solenoid modules carry half of it.
    """
    import imas

    pulse, run = CENTRELINE[part]
    uri = (
        "imas:hdf5?path=/work/imas/shared/imasdb/"
        f"ITER_MACHINE_DESCRIPTION/3/{pulse}/{run}"
    )
    ids = imas.DBEntry(uri, "r", dd_version=DD_VERSION).get(
        "coils_non_axisymmetric", ignore_unknown_dd_version=True
    )
    measured = {}
    for coil in ids.coil:
        section = coil.conductor[0].cross_section
        radius = np.hypot(
            np.asarray(section.delta_r, dtype=float),
            np.asarray(section.delta_z, dtype=float),
        )
        measured[SHORT_NAME[str(coil.name)]] = float(radius.mean())
    return measured


def outlines(revision: str) -> dict[str, Outline]:
    """Return the model outline of every coil, at one solenoid table revision."""
    table = list(POLOIDAL_FIELD) + list(SOLENOID[revision])
    return {outline.name: outline for outline in table}


@dataclass
class Pack:
    """One coil's winding pack, cut into turns.

    The cut is what makes the self term separable.  A filament's own field
    diverges on itself, so a wound filament cannot supply its own diagonal, and
    the split used here replaces each TURN's self term with a finite-section
    ring while taking every turn-against-turn term from the as-built path.  That
    split is only well posed if a source element belongs to exactly one turn:
    otherwise a target sits ON a source element of a different turn and the
    retained term is the divergence the split exists to remove.

    So the pack is cut AT the turn boundaries -- at the points where its
    cumulative sweep passes a whole turn -- and every element after the cut lies
    within one turn.  The cut is exact, on each element's own circle, so no
    geometry is lost to it.
    """

    name: str
    segments: list
    turn: np.ndarray
    sweep: np.ndarray
    nturn: float

    def __len__(self):
        """Return the element count."""
        return len(self.segments)

    @property
    def turn_count(self) -> int:
        """Return the number of turns the pack is cut into."""
        return int(self.turn.max()) + 1

    @property
    def turn_weight(self) -> np.ndarray:
        """Return how much of a whole turn each turn actually sweeps.

        A pack does not end on a whole turn, so one turn at each end is partial.
        A ring standing in for a partial turn has to carry that fraction of an
        ampere-turn or the stack would be wound tighter than the coil: the
        excess on the field coil is 0.07 of a turn, worth more in inductance
        than the whole effect under study.  Every full turn weighs one and the
        weights sum to the MEASURED count, so the ring stack and the as-built
        path carry the same winding.
        """
        return np.bincount(self.turn, weights=np.abs(self.sweep))

    def stations(self, intervals: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the quadrature stations, their steps, and their turn index."""
        station, step, turn = [], [], []
        for segment, index in zip(self.segments, self.turn):
            points, delta = _stations(segment, intervals)
            station.append(points)
            step.append(delta)
            turn.append(np.full(len(points), index))
        return np.vstack(station), np.vstack(step), np.concatenate(turn)

    def turn_centre(self) -> tuple[np.ndarray, np.ndarray]:
        """Return each turn's mean major radius and elevation.

        Length-weighted over the turn's own elements, because that is the ring
        whose self-inductance stands in for it: an unweighted mean over elements
        would let a short joggle count as much as the arc it interrupts.  The
        element's own position is taken at its MIDPOINT along itself, never at
        the midpoint of its chord -- an element here can run most of the way
        round the machine, and the chord midpoint of such an arc is out by most
        of the major radius.
        """
        radius = np.zeros(self.turn_count)
        height = np.zeros(self.turn_count)
        weight = np.zeros(self.turn_count)
        for segment, index in zip(self.segments, self.turn):
            middle = _resolve(segment, 3)[1]
            radius[index] += segment.length * np.hypot(*middle[:2])
            height[index] += segment.length * middle[2]
            weight[index] += segment.length
        return radius / weight, height / weight


def _sweep(segment) -> float:
    """Return the toroidal angle one segment sweeps, in turns."""
    run = _resolve(segment, 256)
    angle = np.unwrap(np.arctan2(run[:, 1], run[:, 0]))
    return float(angle[-1] - angle[0]) / (2 * np.pi)


def cut_turns(path: Path) -> Pack:
    """Return the winding pack of a classified path, cut at its turn boundaries.

    Cutting and labelling are done in one pass over the pack: a piece's turn is
    the whole turn its own midpoint falls in, which after an exact cut is
    unambiguous because no piece straddles a boundary.  The winding may run
    either way about the machine, so the labels are taken from the SIGNED sweep
    and shifted to start at zero.
    """
    pack = np.flatnonzero(path.pack)
    cumulative = np.r_[0.0, np.cumsum(path.element_sweep[pack])]
    segments, middle, spread = [], [], []
    for position, element in enumerate(pack):
        span = cumulative[position : position + 2]
        crossing = np.arange(math.floor(min(span)) + 1.0, math.ceil(max(span)))
        crossing = crossing[(crossing > min(span)) & (crossing < max(span))]
        segment = path.segments[element]
        if len(crossing) == 0:
            segments.append(segment)
            middle.append(0.5 * span.sum())
            spread.append(abs(span[1] - span[0]))
            continue
        fraction = np.r_[0.0, _split_parameter(segment, span, crossing), 1.0]
        edge = np.r_[span[0], crossing, span[1]]
        for piece in range(len(fraction) - 1):
            if abs(fraction[piece + 1] - fraction[piece]) < 1e-9:
                continue
            segments.append(_subsegment(segment, fraction[piece], fraction[piece + 1]))
            middle.append(0.5 * (edge[piece] + edge[piece + 1]))
            spread.append(abs(edge[piece + 1] - edge[piece]))
    turn = np.floor(np.array(middle)).astype(int)
    turn -= turn.min()
    spread = np.array(spread)
    return Pack(
        path.name, segments, _merge_stub(turn, spread), spread, path.swept(pack)
    )


def _merge_stub(turn: np.ndarray, spread: np.ndarray) -> np.ndarray:
    """Return the turn labels with an end stub folded into its neighbour.

    A pack does not end on a whole turn, so cutting at whole turns leaves a stub
    at one or both ends -- a few hundredths of a turn of conductor labelled as a
    turn of its own.  Standing a whole ring in for it would credit the coil with
    most of a turn's self-inductance it does not have, which for a solenoid
    module is a fiftieth of the gap under study, so a stub shorter than half a
    turn is folded into the turn beside it.  What is left is one partial turn at
    each end, and the ring standing in for it overstates that turn's self by the
    fraction it falls short -- a few parts in a hundred of one turn out of
    several hundred.
    """
    for end in (0, turn.max()):
        if spread[turn == end].sum() < 0.5 and turn.max() > 0:
            turn[turn == end] = end - 1 if end else 1
    return turn - turn.min()


# The cable-space DIAMETER standing in for each conductor's section, in metres.
# Declared, not measured: the field coils' stored section is the 35 mm cable
# space exactly, but the solenoid modules' is stored at half the documented
# 32.6 mm, so one of the two pulses cannot be taken at face value and neither is
# taken at face value here.  The self term goes as the logarithm of this, so the
# study reports what it is worth rather than resting on it.
CABLE = {"PF": 0.035, "CS": 0.0326}


def cable_diameter(name: str) -> float:
    """Return the cable-space diameter standing in for a coil's conductor."""
    return CABLE[name[:2]]


def wound_coilset(packs, field_attrs=("Ax", "Ay", "Az")):
    """Return a coilset whose every coil is its as-built filamentary path.

    One frame per coil and one subframe element per cut path piece, so the
    coupling can be taken at element granularity and reduced afterwards -- which
    is what lets a turn be held out of its own field without a separate solve
    for every turn.
    """
    from nova.frame.coilset import CoilSet
    from nova.geometry.polygeom import Polygon
    from nova.geometry.polyline import PolyLine

    coilset = CoilSet(field_attrs=list(field_attrs), dwinding=0)
    for pack in packs:
        polyline = PolyLine(minimum_arc_nodes=3, filament=True)
        polyline.segments = list(pack.segments)
        coilset.winding.insert(
            polyline=polyline,
            cross_section=Polygon({"c": (0, 0, cable_diameter(pack.name))}),
            nturn=1,
            Ic=1,
            name=pack.name,
            part=pack.name[:2],
            delta=0,
        )
    return coilset


def ring_matrix(packs, scale: float = 1.0, nodes: int = 24) -> np.ndarray:
    """Return the turn-by-turn inductance of one coaxial ring per as-built turn.

    Every turn becomes a circular ring at its own measured major radius and
    elevation.  Two things come out of the one matrix: its DIAGONAL is the
    per-turn self term the wound filament cannot supply, and its full block sum
    is the coaxial-ring stack -- the same turns at the same places with the
    helix, the joggles and the layer transitions taken out, which is the rung
    that isolates what the PATH is worth.

    The two parts are computed by DIFFERENT kernels, and that is a statement
    about the physics rather than a saving.  Two non-overlapping circular areas
    have a geometric mean distance exactly equal to the distance between their
    centres, so a filament at each turn's centre reproduces what a finite
    circular section would give for the mutual between two turns -- and the
    off-diagonal is taken from the cheap axisymmetric filament kernel with no
    approximation the section could repair.  It is only ON the diagonal, where
    the two areas are the same area, that the identity fails and the section is
    the whole answer, so that is where the finite-section element is spent.

    The finite-section kernel costs milliseconds a pair against microseconds for
    the filament, and a winding is ALL near field, so the whole matrix through it
    is hours.  The diagonal alone is seconds, and it is reached through the one
    thing a ring's self-inductance depends on: its radius.  The section and the
    elevation do not enter -- a single ring in free space does not know where it
    is -- so the self term is sampled on a radius grid spanning the pack and
    interpolated onto the turns.  ``validate_rings`` checks the whole split,
    diagonal included, against the full finite-section matrix on one coil.
    """
    from nova.biot.greens import greens_psi

    centre = [pack.turn_centre() for pack in packs]
    radius = np.concatenate([entry[0] for entry in centre])
    height = np.concatenate([entry[1] for entry in centre])
    weight = np.concatenate([pack.turn_weight for pack in packs])
    matrix = np.array(
        [
            greens_psi(radius, height, radius[source], height[source])
            for source in range(len(radius))
        ]
    )
    offset = 0
    for pack in packs:
        rows = slice(offset, offset + pack.turn_count)
        index = np.arange(rows.start, rows.stop)
        matrix[index, index] = ring_self(
            radius[rows], scale * cable_diameter(pack.name), nodes
        )
        offset += pack.turn_count
    return matrix * np.outer(weight, weight)


def ring_self(radius, diameter: float, nodes: int) -> np.ndarray:
    """Return each ring's finite-section self-inductance, by radius [H].

    A ring's self-inductance depends on its radius and its section and on
    nothing else, so it is sampled on a radius grid through the shipped element
    and interpolated onto the turns.  Over a pack the function is a radius times
    a logarithm of it, whose curvature is small enough that the interpolation
    residual is eleven orders below the effect under study -- and it is checked
    against direct evaluation rather than argued, by ``validate_rings``.
    """
    grid = np.linspace(radius.min(), radius.max(), nodes)
    if np.ptp(radius) == 0.0:
        grid = radius[:1]
    sample = np.diag(ring_block(grid, np.zeros_like(grid), diameter))
    return (
        np.interp(radius, grid, sample)
        if len(grid) > 1
        else np.full_like(radius, sample[0])
    )


def ring_block(radius, height, diameter: float) -> np.ndarray:
    """Return the finite-section inductance of a set of coaxial disc rings [H]."""
    from nova.biot.circle import Circle
    from nova.frame.coilset import CoilSet
    from nova.frame.turn import Turn

    coilset = CoilSet(dcoil=0.25)
    Turn(*coilset.frames, turn="disc").insert(
        radius,
        height,
        diameter,
        1.0,
        nturn=1,
        name="ring",
        part="ring",
        active=True,
        delta=0,
        segment="circle",
    )
    biot = Circle(
        coilset.subframe, coilset.subframe, turns=[True, True], reduce=[False, False]
    )
    return np.asarray(biot.compute("Psi")[0])


def validate_rings(pack, scale: float = 1.0, nodes: int = 24) -> dict:
    """Return how far the split kernel is from the finite-section matrix.

    The claim under test is the geometric-mean-distance identity: that a
    filament ring stands in for a finite circular section wherever the two
    sections do not overlap.  It is tested on one coil by building the whole
    finite-section matrix -- which is why this is a separate stage and not part
    of every run.
    """
    radius, height = pack.turn_centre()
    weight = pack.turn_weight
    section = ring_block(radius, height, scale * cable_diameter(pack.name)) * np.outer(
        weight, weight
    )
    split = ring_matrix([pack], scale, nodes)
    off = ~np.identity(len(section), dtype=bool)
    return {
        "coil": pack.name,
        "turns": int(len(section)),
        "section_stack": float(section.sum()),
        "split_stack": float(split.sum()),
        "stack_relative": float(abs(split.sum() - section.sum()) / abs(section.sum())),
        "worst_off_diagonal_relative": float(
            np.max(np.abs(split[off] - section[off])) / np.max(np.abs(section[off]))
        ),
        "nearest_pair_relative": float(
            abs(split[0, 1] - section[0, 1]) / abs(section[0, 1])
        ),
        "diagonal_relative": float(
            np.max(np.abs(np.diag(split) - np.diag(section)))
            / np.max(np.abs(np.diag(section)))
        ),
    }


def wound_linkage(coilset, packs, intervals: int, slab: int) -> dict:
    """Return the flux linkage of the as-built winding, turn selfs held out.

    The linkage of coil ``a`` per ampere in coil ``b`` is ``A.dl`` along a's
    conductor, summed, with A taken from b's.  For two different coils that is
    the whole story and the filament is excellent: the coils are metres apart
    and the section cannot matter at that range.

    Within ONE coil it is not, because a filament's own field diverges on itself.
    The pairs held out are those whose target and source lie in the SAME turn,
    which is exactly the set a ring self-inductance stands in for -- and holding
    them out by turn rather than by distance is what makes the seam clean: a
    ring's self term covers the whole of one turn against itself and nothing
    beyond it.

    What is held out is reported, not discarded.  It is the filament's own
    divergent estimate of the same quantity, so its size next to the ring that
    replaces it says how much of the answer rests on the replacement.
    """
    from nova.biot.biotframe import Target
    from nova.biot.solve import Solve
    from scipy.constants import mu_0

    names = [pack.name for pack in packs]
    station, step, turn, coil = [], [], [], []
    for index, pack in enumerate(packs):
        points, delta, label = pack.stations(intervals)
        station.append(points)
        step.append(delta)
        turn.append(label)
        coil.append(np.full(len(points), index))
    station, step = np.vstack(station), np.vstack(step)
    turn, coil = np.concatenate(turn), np.concatenate(coil)

    source_turn = np.concatenate([pack.turn for pack in packs])
    source_coil = np.concatenate(
        [np.full(len(pack), index) for index, pack in enumerate(packs)]
    )
    cross = np.zeros((len(packs), len(packs)))
    same = np.zeros(len(packs))
    singular = 0
    start = time.perf_counter()
    for first in range(0, len(station), slab):
        rows = slice(first, min(first + slab, len(station)))
        data = Solve(
            coilset.subframe,
            Target(
                {axis: station[rows, index] for index, axis in enumerate("xyz")},
                label="Station",
            ),
            reduce=[False, False],
            turns=[True, False],
            attrs=["Ax", "Ay", "Az"],
            name="wound",
        ).data
        potential = np.stack(
            [np.asarray(data[attr], dtype=float) for attr in ("Ax", "Ay", "Az")],
            axis=-1,
        )
        linkage = np.einsum("tsc,tc->ts", potential, step[rows])
        finite = np.isfinite(linkage)
        singular += int((~finite).sum())
        linkage = np.where(finite, linkage, 0.0)
        for target_index in range(len(packs)):
            target_rows = coil[rows] == target_index
            if not target_rows.any():
                continue
            block = linkage[target_rows]
            for source_index in range(len(packs)):
                columns = source_coil == source_index
                patch = block[:, columns]
                if target_index != source_index:
                    cross[target_index, source_index] += patch.sum()
                    continue
                shared = (
                    turn[rows][target_rows][:, np.newaxis]
                    == source_turn[columns][np.newaxis, :]
                )
                same[target_index] += patch[shared].sum()
                cross[target_index, source_index] += patch[~shared].sum()
    return {
        "coils": names,
        "cross_turn": (mu_0 * cross).tolist(),
        "same_turn": (mu_0 * same).tolist(),
        "stations": int(len(station)),
        "elements": int(len(source_turn)),
        "pairs": int(len(station) * len(source_turn)),
        "intervals": intervals,
        "singular_pairs": singular,
        "seconds": time.perf_counter() - start,
    }


def wound_inductance(
    packs,
    intervals: int,
    slab: int,
    scale: float = 1.0,
    nodes: int = 24,
    coilset=None,
    linkage=None,
) -> dict:
    """Return the wound coil's inductance matrix and the rungs it decomposes into.

    ``coilset`` and ``linkage`` are accepted already built so a sweep can vary
    ONE input at a time.  The filament path does not move when the declared
    section changes, and neither does its linkage, so a section sweep that
    rebuilt them would spend most of its wall clock re-deriving an answer it
    already had.
    """
    rings = ring_matrix(packs, scale, nodes)
    size = [pack.turn_count for pack in packs]
    edge = np.r_[0, np.cumsum(size)]
    block = [slice(edge[index], edge[index + 1]) for index in range(len(packs))]
    stack = np.array(
        [[rings[rows, columns].sum() for columns in block] for rows in block]
    )
    self_term = np.array([np.diag(rings)[rows].sum() for rows in block])

    if linkage is None:
        coilset = wound_coilset(packs) if coilset is None else coilset
        linkage = wound_linkage(coilset, packs, intervals, slab)
    wound = np.array(linkage["cross_turn"]) + np.diag(self_term)
    return linkage | {
        "turn_count": size,
        "measured_turns": [pack.nturn for pack in packs],
        "ring_stack": stack.tolist(),
        "ring_self": self_term.tolist(),
        "wound": wound.tolist(),
        "cable": [cable_diameter(pack.name) * scale for pack in packs],
        "reciprocity": float(np.max(np.abs(wound - wound.T)) / np.max(np.abs(wound))),
    }


def continuum_reference(modules, deltas, measured, boxes) -> dict:
    """Return the smeared model at its own turn count and at the measured one.

    Two things have to be separated before a winding can be credited with
    anything.  The shipped mesh is not converged -- it overshoots by more than
    the whole effect under study -- so the continuum is rebuilt over a mesh
    sequence and the finest is the reference.  And the continuum asserts a turn
    count, so it is rebuilt at the MEASURED count as well: whatever that moves
    is a turn count and not a winding.
    """
    result = {"mesh": {}, "measured_turns": list(measured)}
    for delta in deltas:
        result["mesh"][f"{delta:g}"] = reduced_inductance(
            continuum_coilset(modules, delta)
        ).tolist()
    rebuilt = [replace(module, nturn=count) for module, count in zip(modules, measured)]
    result["measured"] = reduced_inductance(
        continuum_coilset(rebuilt, min(deltas))
    ).tolist()
    relocated = [
        replace(module, **outline)
        for module, outline in zip(rebuilt, as_built_outline(rebuilt, boxes))
    ]
    result["as_built"] = reduced_inductance(
        continuum_coilset(relocated, min(deltas))
    ).tolist()
    result["as_built_outline"] = [module.__dict__ for module in relocated]
    result["nominal_turns"] = [module.nturn for module in modules]
    return result


def as_built_outline(modules, boxes) -> list[dict]:
    """Return the gross outline the as-built winding actually occupies.

    The model's outline is a separate assertion from the model's winding, and the
    two have to be separated or one is read for the other.  Where the as-built
    pack sits at a different radius from the outline the model carries, that
    alone moves the self-inductance by more than any winding effect -- a solenoid
    module's inductance goes as the square of its radius, so the 26 mm the
    shipped outline revision differs from the pack by is worth three per cent.

    The as-built outline is the winding box grown by one conductor: the box
    bounds the CENTRELINE, and a gross outline bounds the conductor.
    """
    outline = []
    for module, box in zip(modules, boxes):
        box = np.asarray(box)
        cable = cable_diameter(module.name)
        outline.append(
            {
                "radius": float(box[0].mean()),
                "height": float(box[1].mean()),
                "width": float(np.ptp(box[0]) + cable),
                "thickness": float(np.ptp(box[1]) + cable),
            }
        )
    return outline


def attribute(payload: dict) -> dict:
    """Return what each step of the ladder buys, and what is left over.

    The ladder is ordered so every step changes exactly one thing, which is what
    makes the differences attributable rather than merely observed:

        continuum       uniform current density over the outline the model
                        carries, at the turn count the model carries
        turn count      the same continuum at the MEASURED turn count
        outline         the same continuum over the outline the as-built pack
                        actually occupies -- still smeared, still coaxial
        ring stack      one finite-section ring per as-built turn, at its own
                        radius and elevation -- the turns are now discrete and
                        where the hardware puts them, but still coaxial
        wound path      the as-built three-dimensional path, per-turn self terms
                        from the same rings -- the helix, the joggles and the
                        layer transitions

    The outline rung is not a formality.  The model asserts an outline as well as
    a winding, and for the solenoid modules the revision the shipped gate uses
    sits 26 mm out in radius and 150 mm out in elevation from where the conductor
    is -- worth several per cent of the self term, which is two orders above
    anything the winding does.  Without that rung the winding would be credited
    or blamed for the outline.

    What the machine description still holds over the last rung is unexplained by
    turn count, by outline, by winding position or by winding path, and is what
    the conductor SECTION and the mesh have left to account for.
    """
    machine = np.array(payload["machine_description"])
    mesh = payload["continuum"]["mesh"]
    continuum = np.array(mesh[min(mesh, key=float)])
    rungs = {
        "continuum": continuum,
        "turn count": np.array(payload["continuum"]["measured"]),
        "outline": np.array(payload["continuum"]["as_built"]),
        "ring stack": np.array(payload["wound"]["ring_stack"]),
        "wound path": np.array(payload["wound"]["wound"]),
    }
    order = list(rungs)
    return {
        "rungs": {name: value.tolist() for name, value in rungs.items()},
        "gap": {name: (machine - value).tolist() for name, value in rungs.items()},
        "step": {
            order[index]: (rungs[order[index]] - rungs[order[index - 1]]).tolist()
            for index in range(1, len(order))
        },
        "shipped_mesh": (np.array(mesh[max(mesh, key=float)]) - continuum).tolist(),
    }


def count_turns(paths, revision: str, minimum_sweep: float, inflate: float) -> dict:
    """Return the measured turn count of every coil, with the feeders removed.

    Both counts are reported.  The whole path is what a reader of the IDS gets
    if they do not separate the feeders, and the difference between the two is
    the feeder contribution -- which is what makes the separation auditable
    instead of assumed.
    """
    table = outlines(revision)
    result = {}
    for path in paths:
        split = path.classify(minimum_sweep, inflate)
        pack = np.flatnonzero(path.pack)
        outline = table[path.name]
        box = np.array(split["winding_box"])
        result[path.name] = split | {
            "turns_full_path": path.swept(),
            "turns_pack": path.swept(pack),
            "step": path.step,
            "step_margin": (np.pi - path.step) / np.pi,
            "nominal": outline.nturn,
            "outline": [
                outline.radius,
                outline.height,
                outline.width,
                outline.thickness,
            ],
            "outline_excess": float(
                np.max(
                    np.abs(
                        box
                        - np.array(
                            [
                                [
                                    outline.radius - 0.5 * outline.width,
                                    outline.radius + 0.5 * outline.width,
                                ],
                                [
                                    outline.height - 0.5 * outline.thickness,
                                    outline.height + 0.5 * outline.thickness,
                                ],
                            ]
                        )
                    )
                )
            ),
        }
    return result


def _write(name: str, payload: dict) -> pathlib.Path:
    """Write a stage result beside the figures and return its path."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    path.write_text(json.dumps(payload, indent=1))
    return path


def _read(name: str) -> dict:
    """Return a stage result written earlier."""
    return json.loads((FIGURES / name).read_text())


def load_packs(names, args) -> tuple[list, dict]:
    """Return the cut winding packs of the named coils, and the turn table."""
    paths = {path.name: path for path in read_paths("PF") + read_paths("CS")}
    for path in paths.values():
        path.resolution = args.resolution
    wanted = [paths[name] for name in names]
    table = count_turns(wanted, args.revision, args.minimum_sweep, args.inflate)
    return [cut_turns(path) for path in wanted], table


def stage_turns(args) -> None:
    """Count the turns of every coil from its as-built centreline."""
    paths = read_paths("PF") + read_paths("CS")
    for path in paths:
        path.resolution = args.resolution
    payload = {
        "resolution": args.resolution,
        "minimum_sweep": args.minimum_sweep,
        "inflate": args.inflate,
        "revision": args.revision,
        "section_radius": section_radius("PF") | section_radius("CS"),
        "cable": CABLE,
        "candidates": PF1_CANDIDATES,
        "turns": count_turns(paths, args.revision, args.minimum_sweep, args.inflate),
        "sensitivity": {
            f"{sweep:g}": {
                name: entry["turns_pack"]
                for name, entry in count_turns(
                    paths, args.revision, sweep, args.inflate
                ).items()
            }
            for sweep in args.sweep_scan
        },
    }
    print(f"wrote {_write('turns.json', payload)}")
    report_turns(payload)


def report_turns(payload: dict) -> None:
    """Print the turn table, and arbitrate the field coil's three candidates."""
    print(
        f"\n{'coil':<6}{'elements':>9}{'pack':>7}{'feeder':>7}"
        f"{'full path':>11}{'pack':>10}{'nominal':>10}{'pack-nominal':>14}"
        f"{'box excess':>12}"
    )
    for name, entry in payload["turns"].items():
        print(
            f"{name:<6}{entry['elements']:>9}{entry['pack']:>7}"
            f"{entry['elements'] - entry['pack']:>7}"
            f"{entry['turns_full_path']:>11.3f}{entry['turns_pack']:>10.3f}"
            f"{entry['nominal']:>10.2f}"
            f"{entry['turns_pack'] - entry['nominal']:>+14.3f}"
            f"{1e3 * entry['outline_excess']:>11.0f}mm"
        )
    solenoid = [
        entry["turns_pack"] - entry["nominal"]
        for name, entry in payload["turns"].items()
        if name.startswith("CS")
    ]
    print(
        "\nthe method's own error bar, from the six solenoid modules whose turn"
        "\ncount is not in question: worst |measured - 554| ="
        f" {max(np.abs(solenoid)):.3f} turns\n"
    )
    measured = payload["turns"]["PF1"]["turns_pack"]
    print(f"PF1 measured {measured:.3f} turns over the pack, against its candidates:")
    for label, candidate in payload["candidates"].items():
        print(f"  {label:<22}{candidate:>9.3f}   residual {measured - candidate:+.3f}")
    print("\nturn count against the winding-box sweep threshold [turns]:")
    for sweep, entry in payload["sensitivity"].items():
        row = "".join(f"{value:>10.3f}" for value in entry.values())
        print(f"  {float(sweep):>5.2f}{row}")
    print("        " + "".join(f"{name:>10}" for name in payload["turns"]))
    print("\nsection radius stored in the IDS [mm]:")
    for name, radius in payload["section_radius"].items():
        print(f"  {name:<6}{1e3 * radius:>8.3f}")


def stage_wound(args) -> None:
    """Solve the wound filament coil and place it on the attribution ladder."""
    packs, table = load_packs([module.name for module in MODULES], args)
    payload = {
        "coils": [module.name for module in MODULES],
        "turns": table,
        "machine_description": MACHINE_DESCRIPTION.tolist(),
        "conductor": Conductor().__dict__,
        "cable": {pack.name: cable_diameter(pack.name) for pack in packs},
        "continuum": continuum_reference(
            MODULES,
            args.deltas,
            [pack.nturn for pack in packs],
            [table[pack.name]["winding_box"] for pack in packs],
        ),
        "wound": wound_inductance(packs, args.intervals, args.slab),
    }
    payload["attribution"] = attribute(payload)
    print(f"wrote {_write(f'wound{args.label}.json', payload)}")
    report_wound(payload)


def report_wound(payload: dict) -> None:
    """Print the ladder, the mutuals, and what the winding leaves unexplained."""
    names = payload["coils"]
    machine = np.array(payload["machine_description"])
    attribution = payload["attribution"]
    wound = payload["wound"]
    print(
        f"\nself-inductance ladder [H], and what the machine description still"
        f" holds over each rung\n\n{'rung':<12}"
        + "".join(f"{name:>26}" for name in names)
    )
    print(f"{'':<12}" + "".join(f"{'self      gap to machine':>26}" for _ in names))
    for rung, matrix in attribution["rungs"].items():
        row = f"{rung:<12}"
        for index in range(len(names)):
            row += (
                f"{matrix[index][index]:>14.6f}"
                f"{machine[index, index] - matrix[index][index]:>+12.2e}"
            )
        print(row)
    print(f"\n{'what each step buys [H]':<12}")
    for step, matrix in attribution["step"].items():
        row = f"{step:<12}"
        for index in range(len(names)):
            row += f"{matrix[index][index]:>+14.2e}"
        print(row)
    print(
        f"{'shipped mesh':<12}"
        + "".join(
            f"{attribution['shipped_mesh'][index][index]:>+14.2e}"
            for index in range(len(names))
        )
        + "   (the mesh error the ladder had to be converged past)"
    )
    print("\nmutual inductance [H]: wound against the machine description")
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            print(
                f"  {names[first]:<5}-{names[second]:<6}"
                f"wound {wound['wound'][first][second]:.6f}"
                f"   ring stack {wound['ring_stack'][first][second]:.6f}"
                f"   continuum {attribution['rungs']['turn count'][first][second]:.6f}"
                f"   machine {machine[first, second]:.6f}"
                f"   gap {machine[first, second] - wound['wound'][first][second]:+.2e}"
            )
    print(
        "\nthe self term is DECLARED, not measured.  Per coil, the ring standing in"
        "\nfor each turn against the filament estimate it replaces [H]:"
    )
    for index, name in enumerate(names):
        print(
            f"  {name:<6}rings {wound['ring_self'][index]:.6f}"
            f"   filament same-turn {wound['same_turn'][index]:.6f}"
            f"   turns {wound['turn_count'][index]}"
            f"   cable {1e3 * wound['cable'][index]:.1f} mm"
        )
    print(
        f"\n{wound['stations']} stations x {wound['elements']} elements ="
        f" {wound['pairs']} pairs in {wound['seconds']:.1f} s"
        f"  ({wound['singular_pairs']} singular)"
        f"   reciprocity {wound['reciprocity']:.1e}"
    )


def stage_converge(args) -> None:
    """Sweep the quadrature and the declared section, the two remaining inputs.

    Two things the wound answer rests on that are not the winding: how finely the
    conductor is sampled along itself, and how wide the conductor is declared to
    be.  The first has to converge or the number is a discretisation; the second
    cannot be converged at all -- it is an input -- so what it is worth is
    reported as a slope instead.
    """
    packs, table = load_packs([module.name for module in MODULES], args)
    coilset = wound_coilset(packs)
    quadrature = {
        str(intervals): wound_inductance(packs, intervals, args.slab, coilset=coilset)
        for intervals in args.interval_scan
    }
    reference = quadrature[str(args.intervals)]
    payload = {
        "coils": [module.name for module in MODULES],
        "machine_description": MACHINE_DESCRIPTION.tolist(),
        "quadrature": quadrature,
        "section": {
            f"{scale:g}": wound_inductance(
                packs, args.intervals, args.slab, scale, linkage=reference
            )
            for scale in args.section_scan
        },
        "kernel_split": validate_rings(packs[0]),
    }
    print(f"wrote {_write('converge.json', payload)}")
    report_converge(payload)


def report_converge(payload: dict) -> None:
    """Print the quadrature convergence and the section slope."""
    names = payload["coils"]
    print(f"\n{'intervals':<11}{'stations':>10}" + "".join(f"{n:>16}" for n in names))
    for intervals, entry in payload["quadrature"].items():
        wound = np.array(entry["wound"])
        print(
            f"{intervals:<11}{entry['stations']:>10}"
            + "".join(f"{wound[i, i]:>16.6f}" for i in range(len(names)))
        )
    print(
        f"\n{'cable scale':<11}{'cable [mm]':>12}" + "".join(f"{n:>16}" for n in names)
    )
    for scale, entry in payload["section"].items():
        wound = np.array(entry["wound"])
        print(
            f"{scale:<11}{1e3 * entry['cable'][0]:>12.1f}"
            + "".join(f"{wound[i, i]:>16.6f}" for i in range(len(names)))
        )
    scales = sorted(payload["section"], key=float)
    low, high = payload["section"][scales[0]], payload["section"][scales[-1]]
    print(
        "\nthe self term goes as the logarithm of the cable space, so what the"
        "\ndeclared section is worth is a slope, dL/dln(a) [H per e-fold]:"
    )
    for index, name in enumerate(names):
        slope = (
            np.array(high["wound"])[index, index] - np.array(low["wound"])[index, index]
        ) / math.log(float(scales[-1]) / float(scales[0]))
        print(f"  {name:<6}{slope:>+12.3e}")


def stage_figures(args) -> None:
    """Draw the winding, the turn count and the attribution ladder."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    turns = _read("turns.json")
    paths = read_paths("PF") + read_paths("CS")
    for path in paths:
        path.resolution = turns["resolution"]
    count_turns(paths, turns["revision"], turns["minimum_sweep"], turns["inflate"])

    draw = [path for path in paths if path.name in args.draw]
    figure, axes = plt.subplots(
        2, len(draw), figsize=(4.4 * len(draw), 8.4), squeeze=False
    )
    for column, path in enumerate(draw):
        entry = turns["turns"][path.name]
        radius, height, width, thickness = entry["outline"]
        box = np.array(entry["winding_box"])
        runs = path.elements(16)
        for row, projection in enumerate(("rz", "plan")):
            axis = axes[row, column]
            for element, run in enumerate(runs):
                pack = path.pack[element]
                if projection == "rz":
                    trace = (np.hypot(run[:, 0], run[:, 1]), run[:, 2])
                else:
                    trace = (run[:, 0], run[:, 1])
                axis.plot(
                    *trace,
                    color="C0" if pack else "0.6",
                    lw=0.6 if pack else 1.1,
                    zorder=2 if pack else 1,
                )
            axis.set_aspect("equal")
            axis.set_rasterization_zorder(3)
            if projection == "rz":
                axis.add_patch(
                    plt.Rectangle(
                        (radius - 0.5 * width, height - 0.5 * thickness),
                        width,
                        thickness,
                        fill=False,
                        edgecolor="C3",
                        lw=1.0,
                        label="model outline",
                    )
                )
                axis.add_patch(
                    plt.Rectangle(
                        (box[0, 0], box[1, 0]),
                        np.ptp(box[0]),
                        np.ptp(box[1]),
                        fill=False,
                        edgecolor="C2",
                        ls="--",
                        lw=1.0,
                        label="winding box",
                    )
                )
                axis.set_xlim(box[0, 0] - 0.35, box[0, 1] + 0.35)
                axis.set_ylim(box[1, 0] - 0.35, box[1, 1] + 0.35)
                axis.set_xlabel("radius [m]")
                axis.set_ylabel("elevation [m]")
                axis.legend(fontsize=7, loc="lower left")
                axis.set_title(
                    f"{path.name}: {entry['pack']} pack of {entry['elements']} elements"
                    f"\n{entry['turns_pack']:.3f} turns"
                    f" (nominal {entry['nominal']:g}, whole path"
                    f" {entry['turns_full_path']:.3f})",
                    fontsize=9,
                )
            else:
                axis.set_xlabel("x [m]")
                axis.set_ylabel("y [m]")
                axis.set_title(
                    f"plan view, feeders grey to r = {entry['path_radius_max']:.1f} m",
                    fontsize=9,
                )
    figure.suptitle(
        "the as-built conductor centreline: winding pack in colour, feeders in grey"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(FIGURES / "winding-path.svg", dpi=200)
    plt.close(figure)

    # The turn count, as a residual on the nominal count -- the quantity is a few
    # tenths of a turn out of hundreds, so the count itself shows nothing.
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), width_ratios=[2, 1])
    names = list(turns["turns"])
    residual = [
        turns["turns"][name]["turns_pack"] - turns["turns"][name]["nominal"]
        for name in names
    ]
    whole = [
        turns["turns"][name]["turns_full_path"] - turns["turns"][name]["nominal"]
        for name in names
    ]
    position = np.arange(len(names))
    axes[0].bar(position - 0.2, residual, 0.4, color="C0", label="winding pack")
    axes[0].bar(position + 0.2, whole, 0.4, color="0.7", label="whole path")
    axes[0].axhline(0.0, color="C3", lw=1.0)
    solenoid = [value for name, value in zip(names, residual) if name.startswith("CS")]
    band = max(np.abs(solenoid))
    axes[0].axhspan(
        -band,
        band,
        color="C2",
        alpha=0.15,
        label=f"solenoid scatter, {band:.3f} turns",
    )
    axes[0].set_xticks(position)
    axes[0].set_xticklabels(names, rotation=45, ha="right")
    axes[0].set_ylabel("measured - nominal [turns]")
    axes[0].set_title(
        "as-built turn count against the number the model carries\n"
        "the six solenoid modules set the method's own error bar"
    )
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)

    # Candidates as horizontal lines rather than markers: the quantity is a
    # value on one axis, and a scatter over a meaningless abscissa invites the
    # reader to look for a trend across candidates that does not exist.
    measured = turns["turns"]["PF1"]["turns_pack"]
    axes[1].axhline(measured, color="C0", lw=2.0, label=f"as built {measured:.3f}")
    axes[1].axhspan(measured - band, measured + band, color="C0", alpha=0.18)
    # Labels alternate below and above the line they belong to: two of the three
    # candidates sit four hundredths of a turn apart and would otherwise collide.
    ranked = sorted(turns["candidates"].items(), key=lambda item: item[1])
    for position, (label, candidate) in enumerate(ranked):
        axes[1].axhline(candidate, color="C3", lw=1.0, ls="--")
        axes[1].annotate(
            f"{label}  {candidate:g}   ({measured - candidate:+.3f})",
            (0.03, candidate),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            xytext=(0, -11 if position % 2 == 0 else 4),
            fontsize=8,
            color="C3",
        )
    reach = list(turns["candidates"].values()) + [measured]
    span = max(reach) - min(reach)
    axes[1].set_ylim(min(reach) - 0.25 * span, max(reach) + 0.25 * span)
    axes[1].set_xticks([])
    axes[1].set_ylabel("turns")
    axes[1].set_title("PF1: which candidate\nthe centreline supports", fontsize=10)
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(FIGURES / "turn-count.svg")
    plt.close(figure)

    # The unlabelled run by name, never the first of a glob: a labelled variant
    # sorts ahead of it and the ladder would silently be drawn from a sweep point.
    if not (FIGURES / "wound.json").exists():
        print(f"wrote figures to {FIGURES} (no wound stage yet)")
        return
    payload = _read("wound.json")
    coils = payload["coils"]
    machine = np.array(payload["machine_description"])
    attribution = payload["attribution"]
    rungs = list(attribution["rungs"])
    # Two rows, because for the solenoid modules one step is two orders larger
    # than the rest: the as-built pack sits 200 mm above the outline the shipped
    # gate carries, and on a scale that holds THAT the winding steps are a flat
    # line.  The lower row rebases on the as-built outline, so it shows only what
    # the WINDING buys once the outline is agreed.
    figure, axes = plt.subplots(
        2, len(coils), figsize=(4.6 * len(coils), 8.6), squeeze=False
    )
    for row, anchor in enumerate(
        ("continuum", "outline" if "outline" in rungs else None)
    ):
        if anchor is None:
            continue
        shown = rungs if row == 0 else rungs[rungs.index(anchor) :]
        for index, name in enumerate(coils):
            axis = axes[row, index]
            base = attribution["rungs"][anchor][index][index]
            offset = [attribution["rungs"][rung][index][index] - base for rung in shown]
            gap = machine[index, index] - base
            axis.axhline(0.0, color="C7", ls=":", label=f"{anchor} rung")
            axis.axhline(gap, color="C3", ls="--", label="machine description")
            axis.plot(range(len(shown)), offset, "o-", color="C0", label="ladder")
            for position, rung in enumerate(shown[1:], start=1):
                axis.annotate(
                    f"{attribution['step'][rung][index][index]:+.2e}",
                    (position - 0.5, 0.5 * (offset[position] + offset[position - 1])),
                    fontsize=7,
                    ha="center",
                    color="C0",
                )
            axis.annotate(
                f"unexplained {gap - offset[-1]:+.2e} H",
                (0.02, 0.06),
                xycoords="axes fraction",
                fontsize=8,
                color="C3",
            )
            axis.set_xticks(range(len(shown)))
            axis.set_xticklabels(shown, rotation=25, ha="right")
            axis.set_ylabel(f"offset from the {anchor} rung [H]")
            axis.set_title(
                f"{name}: {'the whole ladder' if row == 0 else 'the winding alone'}",
                fontsize=10,
            )
            axis.grid(alpha=0.3)
            axis.legend(fontsize=8)
    figure.suptitle(
        "does the as-built turn count, outline and winding path close the gap"
        " to the machine description?"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(FIGURES / "ladder.svg")
    plt.close(figure)
    print(f"wrote figures to {FIGURES}")


def parse_args(argv=None):
    """Return the parsed command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolution",
        type=int,
        default=8,
        help="samples per element when resolving an arc's own toroidal sweep",
    )
    parser.add_argument(
        "--minimum-sweep",
        type=float,
        default=0.3,
        help="turns an element must sweep to bound the winding box",
    )
    parser.add_argument(
        "--inflate",
        type=float,
        default=0.049,
        help="margin on the winding box, one jacket width [m]",
    )
    parser.add_argument(
        "--revision", default="wide", choices=tuple(SOLENOID), help="solenoid table"
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    count = stages.add_parser("turns")
    count.add_argument(
        "--sweep-scan", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4]
    )
    count.set_defaults(run=stage_turns)

    solve = stages.add_parser("wound")
    solve.add_argument("--intervals", type=int, default=4)
    solve.add_argument("--slab", type=int, default=4000)
    solve.add_argument(
        "--deltas", type=float, nargs="+", default=[0.25, 0.12, 0.08, 0.06]
    )
    solve.add_argument("--label", default="")
    solve.set_defaults(run=stage_wound)

    converge = stages.add_parser("converge")
    converge.add_argument("--intervals", type=int, default=4)
    converge.add_argument("--slab", type=int, default=4000)
    converge.add_argument("--interval-scan", type=int, nargs="+", default=[1, 2, 4, 8])
    converge.add_argument(
        "--section-scan", type=float, nargs="+", default=[0.5, 1.0, 2.0]
    )
    converge.set_defaults(run=stage_converge)

    figures = stages.add_parser("figures")
    figures.add_argument("--draw", nargs="+", default=["PF1", "CS3U"])
    figures.set_defaults(run=stage_figures)
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    arguments.run(arguments)
