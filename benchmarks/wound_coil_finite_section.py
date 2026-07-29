"""Does the conductor's true SECTION close what the as-built winding path leaves?

A companion study (:mod:`benchmarks.wound_coil_inductance`) took the winding of
the upper poloidal field coil as built -- its turn count measured from the
conductor centreline and its three-dimensional path integrated as ``A.dl`` -- and
left one input DECLARED rather than resolved: the conductor's cross-section.  Each
turn's self term was supplied by an axisymmetric ring at that turn's own mean
radius carrying a disc of the cable space, because a filament's self term
diverges and a filamentary winding cannot supply its own diagonal.

That declaration is what this study replaces.  Every turn's self term is taken
from the turn's OWN path, swept with the conductor's own section: the arcs through
the closed-form polygon-section arc (:func:`nova.biot.polygonarc.polygon_arc_greens`)
and the straight joggles through its prism peer
(:func:`nova.biot.polybeam.polygon_beam_greens`).  Nothing else about the ladder
moves -- the turn count, the path and the turn-against-turn coupling are the
companion's, unchanged -- so the difference is attributable to the section and to
nothing else.

Three things shape the computation, and the third is what the study had to
discover.

    which pairs need it   Two non-overlapping circular sections have a geometric
                          mean distance exactly equal to the distance between
                          their centres, so a filament reproduces what a finite
                          section would give for any pair of turns that do not
                          overlap -- and the pack's turn-to-turn clearance is
                          larger than the cable space everywhere.  The finite
                          section is spent on the WITHIN-TURN block alone, which
                          is five source elements against twenty stations rather
                          than the whole matrix.

    where the target is   An inductance is the flux linked by the current, which
                          is the MEAN of the flux over the section the current
                          occupies -- not its value on the centreline.  On the
                          coincident term the two differ by the whole gap between
                          a section's mean logarithmic distance from its own
                          centre and its geometric mean distance from itself: for
                          a disc, ``mu0/(8 pi)`` per metre of conductor, which
                          over this coil's six kilometres of cable is 3.1e-04 H
                          -- four tenths of the residual under study.  So the
                          centreline sum carries an explicit section-mean
                          correction, computed from the section's own corners by
                          an axisymmetric kernel that shares nothing with the
                          swept elements, and gated against it.

    which section         The frame's section of record is the swept volume's
                          poloidal FOOTPRINT, and the two thickened elements read
                          it as their cross-section.  For an axisymmetric coil
                          they are the same polygon, because every element holds
                          its ``(r, z)`` fixed as it sweeps.  A winding pack does
                          not: each element advances across the pack as it goes
                          round, so its footprint is a smear reaching 170 mm where
                          the section is 35 mm -- 244 corners instead of 32 and 7.6
                          times the area the current occupies, on an element three
                          quarters of the pack's length looks like.  A self term is
                          the logarithm of the section's geometric mean distance, so
                          that is a first-order error on it.  Two joggles in a
                          hundred move so far that the footprint is not even
                          connected.  So the closed forms are driven directly, with
                          the section handed over as the section.  The two routes
                          are run against each other on the coaxial ring, which is
                          the one path where the frame's column IS the section.

The cooling channel is carried the way the frame carries any hollow section, by
superposition: the cable space at current density ``+j`` and the central channel
as a core at ``-j``, one circuit, so the annulus between them carries ``j`` and the
channel cancels.  Its contribution is reported on its own.

One more thing the ladder was missing turns out to matter more than the section
does, and it is not a model choice at all.  The as-built centreline runs past the
winding pack and out to the coil's terminals, and that run is separated before
anything is counted because it does not WIND -- but it is thirty-six metres of
conductor carrying the coil's own current in series with the pack, so the
inductance a terminal sees includes it and no rung either study built has ever had
it in.  The ``feeder`` stage measures it.

Stages, each writing its own JSON beside the figures::

    python benchmarks/wound_coil_finite_section.py gate
    python benchmarks/wound_coil_finite_section.py self
    python benchmarks/wound_coil_finite_section.py seam
    python benchmarks/wound_coil_finite_section.py feeder
    python benchmarks/wound_coil_finite_section.py ladder
    python benchmarks/wound_coil_finite_section.py figures

Every stage is minutes on a login node once the frame is out of the loop: the
direct route pays for the section's corners and for nothing else.  It is host
numpy throughout, for the reason the companion sets out -- the tiled operator
reaches the axisymmetric polygon section only, so there is no device route to the
swept arc and prism kernels a resolved winding is built from.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
import pathlib
import time

import numpy as np
from scipy.constants import mu_0

from benchmarks.wound_coil_inductance import (
    CABLE,
    FIGURES,
    _stations,
    cable_diameter,
    count_turns,
    cut_turns,
    read_paths,
    ring_block,
)

CHANNEL = 0.010
"""The central cooling spiral's diameter, in metres.

The strands wrap a spiral that carries no current, so the conducting region of
the cable space is the annulus outside it.  Declared, like the cable space
itself: the centreline pulse carries a section radius and no channel.
"""

CORNERS = (16, 32, 64)
"""The corner counts a round section is resolved at.

A closed-form section reduction costs one evaluation per corner and a disc has as
many as its generator is asked for, so the resolution is a cost knob with an
accuracy attached and neither is assumed.  Sixty-four is what the shipped disc
generator produces at its default sixteen quadrant segments.
"""


# ---------------------------------------------------------------------------
# The section, and the two polygons a hollow one is made of.


@dataclass(frozen=True)
class Section:
    """A conductor cross-section: a cable space, optionally about a channel.

    ``corners`` is the count on EACH boundary, so a hollow section carries twice
    that many and costs twice as much to integrate.
    """

    diameter: float
    channel: float = 0.0
    corners: int = 64

    @property
    def hollow(self) -> bool:
        """Return whether the section has a channel to cancel."""
        return self.channel > 0.0

    @property
    def label(self) -> str:
        """Return the name this section is reported under."""
        kind = "annulus" if self.hollow else "disc"
        return f"{kind}-{self.corners}"

    def descriptor(self, radius: float = 0.0, height: float = 0.0) -> dict:
        """Return the frame descriptor for this section, centred where asked."""
        quadrant = max(self.corners // 4, 1)
        if not self.hollow:
            return {"disc": (radius, height, self.diameter, self.diameter, quadrant)}
        return {
            "skin": (
                radius,
                height,
                self.diameter,
                1.0 - self.channel / self.diameter,
                quadrant,
            )
        }

    def boundaries(self, radius: float, height: float) -> list[np.ndarray]:
        """Return the section's outer boundary and every interior one, as corners.

        Corner loops in ``(r, z)``, closing repeat dropped, outer first.  These are
        the polygons the superposition is built from: the outer at ``+j`` and each
        interior as a core at ``-j``.
        """
        from nova.geometry.polygeom import Polygon

        poly = Polygon(self.descriptor(radius, height)).poly
        loops = [np.asarray(poly.exterior.coords, dtype=float)[:-1]]
        loops += [np.asarray(ring.coords, dtype=float)[:-1] for ring in poly.interiors]
        return loops

    def area(self, radius: float = 0.0, height: float = 0.0) -> float:
        """Return the conducting area, the outer boundary less its holes."""
        loops = self.boundaries(radius, height)
        return _shoelace(loops[0]) - sum(_shoelace(loop) for loop in loops[1:])


def _shoelace(vertices: np.ndarray) -> float:
    """Return the area a closed corner list encloses."""
    rolled = np.roll(vertices, -1, axis=0)
    return 0.5 * abs(
        float(np.sum(vertices[:, 0] * rolled[:, 1] - rolled[:, 0] * vertices[:, 1]))
    )


def _centroid(loops: list[np.ndarray]) -> np.ndarray:
    """Return the area centroid of an outer boundary less its holes."""
    from nova.biot.greens import section_centroid

    total, moment = 0.0, np.zeros(2)
    for index, loop in enumerate(loops):
        area = _shoelace(loop) * (1.0 if index == 0 else -1.0)
        total += area
        moment += area * np.asarray(section_centroid(loop), dtype=float)
    return moment / total


# ---------------------------------------------------------------------------
# The section-mean correction, and the axisymmetric reference it comes with.


def ring_flux(section: Section, radius: float, height: float = 0.0) -> dict:
    """Return a coaxial ring's self flux on its centreline and as a section mean.

    Both from the same closed-form axisymmetric polygon-section kernel, which
    shares no code with the swept elements this study measures, so the pair is an
    independent reference for them.  A hollow section is carried by the same
    superposition the frame uses -- outer boundary less core, in the source
    spread AND in the target average -- so the reference exists for the annulus,
    where no shipped axisymmetric element does.

    ``centreline`` is the flux at the section's own centroid, which is what a
    sum of ``A.dl`` along the conductor's centreline returns.  ``mean`` is the
    flux averaged over the section the current occupies, which is what an
    inductance is.  Their difference is the correction, and it is a property of
    the SECTION rather than of the ring: divided by the conductor's length it is
    the same number at any radius, which ``gate`` reports.
    """
    from nova.biot.polygonanalytic import polygon_analytic_greens
    from nova.biot.sectionaverage import averaged_greens

    loops = section.boundaries(radius, height)
    sign = np.array([1.0] + [-1.0] * (len(loops) - 1))
    area = sign * np.array([_shoelace(loop) for loop in loops])
    centre = _centroid(loops)
    total = area.sum()
    centreline = sum(
        weight * float(polygon_analytic_greens(centre[:1], centre[1:], loop)[0][0])
        for weight, loop in zip(area, loops)
    )
    mean = 0.0
    for column, source in enumerate(loops):
        averaged = averaged_greens(loops, source)[0]
        mean += area[column] * float(area @ np.asarray(averaged, dtype=float))
    return {
        "centreline": centreline / total,
        "mean": mean / total**2,
        "area": float(total),
        "length": 2 * np.pi * radius,
    }


CORRECTION_RADIUS = 0.1
"""How finely the section-mean correction is resolved in ring radius, in metres.

The correction is a two-dimensional property of the section; the ring it is
measured on enters only through terms of order the squared aspect ratio, and the
gate reports what that comes to -- five parts in a hundred thousand over two
DECADES of radius, so a tenth of a metre is four orders finer than anything the
answer can see.  Coarsening it is what lets one evaluation serve a whole layer of
the pack, where the double section integral it rests on is otherwise the most
expensive thing in a per-turn loop.
"""


def section_mean_correction(section: Section, radius: float) -> float:
    """Return the centreline-minus-mean self flux, per metre of conductor [H/m].

    The quantity a centreline ``A.dl`` sum has to give back.  It is the section's
    own mean logarithmic distance from its centre against its geometric mean
    distance from itself, which is a two-dimensional property of the section.
    """
    return _correction(section, round(radius / CORRECTION_RADIUS))


@lru_cache(maxsize=None)
def _correction(section: Section, band: int) -> float:
    """Return the correction for one section in one band of ring radius."""
    flux = ring_flux(section, band * CORRECTION_RADIUS)
    return (flux["centreline"] - flux["mean"]) / flux["length"]


# ---------------------------------------------------------------------------
# The swept winding, and the linkage along its own centreline.


def swept_coilset(segments, section: Section, name="turn"):
    """Return a coilset carrying one winding of that section along those segments.

    ``filament=False`` is what routes an arc to :class:`nova.biot.polybow.PolyBow`
    and a straight segment to :class:`nova.biot.polybeam.PolyBeam` instead of to
    the filament pair, and it is also what lets a hollow section reach the frame
    as a linked pair of solid ones.
    """
    from nova.frame.coilset import CoilSet
    from nova.geometry.polygeom import Polygon
    from nova.geometry.polyline import PolyLine

    coilset = CoilSet(field_attrs=["Ax", "Ay", "Az"], dwinding=0)
    polyline = PolyLine(minimum_arc_nodes=3, filament=False)
    polyline.segments = list(segments)
    coilset.winding.insert(
        polyline=polyline,
        cross_section=Polygon(section.descriptor()),
        nturn=1,
        Ic=1,
        name=name,
        part=name,
        delta=0,
    )
    return coilset


def section_health(coilset, section: Section) -> dict:
    """Return how far each source's section column is from the section asked for.

    The column is a POLOIDAL FOOTPRINT: the union of the swept section at a
    segment's two end planes, projected.  For a coaxial arc the two coincide in
    projection and the union should return the section unchanged -- and mostly it
    does, exactly.  It does not always: a section straddling ``z = 0`` comes back
    from the union with corners dropped and its centroid moved by a twentieth of
    its own width, which costs the element a thousandth of the self term it is
    reading the section for.  Nothing in the frame reports it, so it is measured
    here on every source before any number is taken from one, as a corner count, a
    centroid offset in units of the section's own radius, and an area ratio.
    """
    from nova.biot.polybow import section_corners

    counts, skew, ratio = [], [], []
    for poly in np.asarray(coilset.subframe["poly"]):
        corners = section_corners(poly)
        span = np.ptp(corners, axis=0)
        counts.append(int(len(corners)))
        # a round section's corners are symmetric about its centre, so the mean of
        # the corners and the area centroid coincide; they part company only where
        # the union has dropped corners from one side of it
        skew.append(
            float(
                np.hypot(*(corners.mean(axis=0) - _area_centroid(corners)))
                / max(span.max(), np.finfo(float).tiny)
            )
        )
        ratio.append(float(_shoelace(corners) / (0.25 * np.pi * span[0] * span[1])))
    return {
        "wanted_corners": int(section.corners),
        "rows": int(len(counts)),
        "short_rows": int(sum(count < section.corners for count in counts)),
        "min_corners": int(min(counts)),
        "worst_corner_skew": float(max(skew)),
        "worst_area_ratio": float(max(abs(value - 1.0) for value in ratio)),
    }


def _area_centroid(vertices: np.ndarray) -> np.ndarray:
    """Return a corner loop's area centroid."""
    from nova.biot.greens import section_centroid

    return np.asarray(section_centroid(vertices), dtype=float)


def centreline_linkage(coilset, segments, intervals: int) -> np.ndarray:
    """Return ``mu0 A.dl`` per (station, source element) along the centreline [H].

    The source's ``factor`` column is applied here rather than by the frame's own
    reduction, because the reduction is what the element-granularity solve turns
    off: a hollow section's core carries ``-1`` and the pair only sums once that
    column is read.
    """
    from nova.biot.biotframe import Target
    from nova.biot.solve import Solve

    station, step = [], []
    for segment in segments:
        points, delta = _stations(segment, intervals)
        station.append(points)
        step.append(delta)
    station, step = np.vstack(station), np.vstack(step)
    data = Solve(
        coilset.subframe,
        Target(
            {axis: station[:, index] for index, axis in enumerate("xyz")},
            label="Station",
        ),
        reduce=[False, False],
        turns=[True, False],
        attrs=["Ax", "Ay", "Az"],
        name="swept",
    ).data
    potential = np.stack(
        [np.asarray(data[attr], dtype=float) for attr in ("Ax", "Ay", "Az")], axis=-1
    )
    factor = np.asarray(coilset.subframe["factor"], dtype=float)
    return mu_0 * np.einsum("tsc,tc->ts", potential, step) * factor[np.newaxis]


# ---------------------------------------------------------------------------
# The swept section along a path that MOVES, which the frame column cannot carry.


def _arc_frame(segment) -> tuple:
    """Return a circular arc's own frame, and the sweep it covers in it.

    Built from the arc's own three points rather than from its stored axes, so the
    frame is right-handed about the direction the current actually flows and the
    sweep runs from zero to a positive angle -- which is the convention the closed
    form takes its limits in.  The section a swept conductor presents is a disc in
    THIS frame's own ``(r, z)`` plane whatever the arc's orientation, because the
    plane current crosses is the plane perpendicular to an azimuthal path.
    """
    centre = np.asarray(segment.center, dtype=float)
    run = np.asarray(segment.sample(3), dtype=float)
    radial = (run[0] - centre) / np.linalg.norm(run[0] - centre)
    # from the two chords of the arc's own three points rather than from a chord
    # against the radius: their cross product is best conditioned in the middle of
    # the sweep range and does not collapse as the sweep approaches a half turn
    axis = np.cross(run[1] - run[0], run[2] - run[1])
    axis /= np.linalg.norm(axis)
    axes = np.stack([radial, np.cross(axis, radial), axis])
    return centre, axes, float(segment.radius), float(segment.central_angle)


def _line_frame(segment) -> tuple:
    """Return a straight segment's own frame, its axis on the local vertical."""
    start = np.asarray(segment.start_point, dtype=float)
    end = np.asarray(segment.end_point, dtype=float)
    axis = (end - start) / np.linalg.norm(end - start)
    across = np.cross(axis, np.identity(3)[int(np.argmin(np.abs(axis)))])
    across /= np.linalg.norm(across)
    return (
        start,
        np.stack([across, np.cross(axis, across), axis]),
        float(segment.length),
    )


def _local(points: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Return global points resolved on a local frame's own axes."""
    return (np.asarray(points, dtype=float) - origin) @ axes.T


def _signed_area(loops: list[np.ndarray]) -> np.ndarray:
    """Return each boundary's area, the outer positive and every hole negative."""
    return np.array(
        [
            (1.0 if index == 0 else -1.0) * _shoelace(loop)
            for index, loop in enumerate(loops)
        ]
    )


def segment_potential(segment, section: Section, station: np.ndarray) -> np.ndarray:
    """Return what one swept segment puts at each station, per member [T m/A].

    Shape ``(members, stations, 3)`` in global cartesian, per ampere of TOTAL
    conductor current at uniform density over the CONDUCTING section -- the
    potential itself in SI, not the frame's convention that leaves ``mu0`` off it.
    A hollow section is carried as the outer boundary less its core, each already
    weighted by the density the annulus sets, so the members sum to the annulus and
    the second one is what the cooling channel takes off.

    The closed forms are called directly rather than through the frame, and that is
    a correction rather than a shortcut.  The frame's section of record is a
    poloidal FOOTPRINT -- the union of the swept section over a whole segment --
    and for an axisymmetric coil, whose every element holds ``(r, z)`` fixed as it
    sweeps, the footprint IS the section.  A winding pack's path does not hold them
    fixed: every element advances across the pack as it goes round, so its
    footprint is a smear reaching 170 mm where the section is 35 mm.  Measured on
    this coil's own pack, three quarters of its length sits on elements whose ends
    move further than their own section is wide, the worst column reaches the kernel
    with 244 corners instead of 32, and it carries 7.6 times the area the current
    occupies.  A self term is the logarithm of the section's own geometric mean
    distance, so a section several times too large is a first-order error on it,
    not a rounding.  Two of the pack's joggles in a hundred move so far that the
    footprint is not even connected -- the column comes back as a pair of disjoint
    discs, which the frame's own section reader raises on.

    So the section is handed over AS the section: for an arc, the conductor's own
    disc at the arc's radius in the arc's own frame, which is exactly what the
    frame column reduces to for the coaxial ring the gate checks both routes on.
    """
    from nova.biot.polybeam import polygon_beam_greens
    from nova.biot.polygonarc import polygon_arc_greens

    station = np.asarray(station, dtype=float)
    if segment.name == "line":
        origin, axes, length = _line_frame(segment)
        local = _local(station, origin, axes)
        loops = section.boundaries(0.0, 0.0)
        weight = _signed_area(loops)
        member = np.stack(
            [
                np.asarray(
                    polygon_beam_greens(
                        local[:, 0],
                        local[:, 1],
                        local[:, 2],
                        loop,
                        np.zeros(len(station)),
                        np.full(len(station), length),
                    )[0],
                    dtype=float,
                )
                for loop in loops
            ]
        )
        member *= mu_0 * (weight / weight.sum())[:, np.newaxis]
        return member[:, :, np.newaxis] * axes[2][np.newaxis, np.newaxis, :]

    centre, axes, arc_radius, sweep = _arc_frame(segment)
    local = _local(station, centre, axes)
    azimuth = np.arctan2(local[:, 1], local[:, 0])
    loops = section.boundaries(arc_radius, 0.0)
    weight = _signed_area(loops)
    rows = np.stack(
        [
            np.stack(
                polygon_arc_greens(
                    np.hypot(local[:, 0], local[:, 1]),
                    local[:, 2],
                    azimuth,
                    loop,
                    0.0,
                    sweep,
                )[:2]
            )
            for loop in loops
        ]
    )
    rows *= (weight / weight.sum())[:, np.newaxis, np.newaxis]
    cosine, sine = np.cos(azimuth), np.sin(azimuth)
    local_vector = np.stack(
        [
            rows[:, 0] * cosine - rows[:, 1] * sine,
            rows[:, 0] * sine + rows[:, 1] * cosine,
            np.zeros_like(rows[:, 0]),
        ],
        axis=-1,
    )
    return local_vector @ axes


def path_self(segments, section: Section, intervals: int, radius: float) -> dict:
    """Return one run of conductor's self-inductance from its own swept section.

    ``A.dl`` along the run's own centreline against the run's own resolved
    conductor, less the section-mean correction the centreline sum owes.  Every
    station sees every element of the run including the one it sits on, which is
    what the finite section exists to make finite.
    """
    station, step = [], []
    for segment in segments:
        points, delta = _stations(segment, intervals)
        station.append(points)
        step.append(delta)
    station, step = np.vstack(station), np.vstack(step)
    members = len(section.boundaries(0.0, 0.0))
    linkage = np.zeros(members)
    for segment in segments:
        linkage += np.einsum(
            "mnc,nc->m", segment_potential(segment, section, station), step
        )
    length = float(sum(segment.length for segment in segments))
    correction = section_mean_correction(section, radius) * length
    return {
        "centreline": float(linkage.sum()),
        "correction": correction,
        "self": float(linkage.sum()) - correction,
        "core": float(linkage[1:].sum()),
        "length": length,
        "sources": int(len(segments)),
        "stations": int(len(station)),
    }


def framed_self(segments, section: Section, intervals: int, radius: float) -> dict:
    """Return one run of conductor's self-inductance from its own swept section.

    The centreline sum less the section-mean correction, with the correction taken
    at the run's own mean radius.  ``core`` is what the cooling channel takes off:
    the columns the superposition carries at negative density, which is a
    contribution rather than an error and is reported on its own.
    """
    coilset = swept_coilset(segments, section)
    pairwise = centreline_linkage(coilset, segments, intervals)
    length = float(sum(segment.length for segment in segments))
    factor = np.asarray(coilset.subframe["factor"], dtype=float)
    core = float(pairwise[:, factor < 0].sum())
    correction = section_mean_correction(section, radius) * length
    return {
        "centreline": float(pairwise.sum()),
        "correction": correction,
        "self": float(pairwise.sum()) - correction,
        "core": core,
        "length": length,
        "sources": int(pairwise.shape[1]),
        "stations": int(pairwise.shape[0]),
        "health": section_health(coilset, section),
    }


# ---------------------------------------------------------------------------
# The gate: a coaxial ring, where an independent kernel knows the answer.


def ring_segments(radius: float, height: float, count: int) -> list:
    """Return a coaxial ring cut into ``count`` equal arcs."""
    from benchmarks.wound_coil_inductance import _segment

    theta = np.linspace(0.0, 2 * np.pi, 2 * count + 1)
    point = np.c_[
        radius * np.cos(theta), radius * np.sin(theta), height * np.ones_like(theta)
    ]
    return [
        _segment(point[2 * i], point[2 * i + 1], point[2 * i + 2], True)
        for i in range(count)
    ]


def _by_angle(loop: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Return a corner loop ordered by its polar angle about a centre.

    Which corner of the inner boundary faces which corner of the outer is what a
    partition into cells between them depends on, and it cannot be read off the
    stored order: a section with a hole comes back from a boolean difference with
    its exterior counterclockwise and its interior CLOCKWISE, so pairing the two
    by index builds a crossed quadrilateral whose cells overlap -- the partition
    then covers 18 per cent more area than the annulus it stands for and reads as a
    superposition failure it is not.  Both boundaries are generated at the same
    angular stations about the same centre, so ordering each by angle pairs them.
    """
    return loop[np.argsort(np.arctan2(loop[:, 1] - centre[1], loop[:, 0] - centre[0]))]


def annulus_partition(loops: list[np.ndarray]) -> list[np.ndarray]:
    """Return a partition of an annulus into solid quadrilateral cells.

    Between corresponding corners of the two boundaries, so every cell is a solid
    polygon and the sum has no cancellation between members.  This is what makes
    the agreement a statement about the SUPERPOSITION rather than about the
    kernel, which both routes share.
    """
    centre = _centroid(loops)
    outer, core = (_by_angle(loop, centre) for loop in loops)
    return [
        np.array(
            [outer[i], outer[(i + 1) % len(outer)], core[(i + 1) % len(core)], core[i]]
        )
        for i in range(len(outer))
    ]


def partition_flux(loops: list[np.ndarray], radius: float, height: float) -> float:
    """Return an annulus ring's centreline self flux, cell by cell [H].

    The annulus integrated DIRECTLY, at its own uniform density, with the target
    on the section centroid.  No member of this sum is negative, so it cannot
    hide a superposition that fails to cancel.
    """
    from nova.biot.polygonanalytic import polygon_analytic_greens

    centre = _centroid(loops)
    total, area = 0.0, 0.0
    for cell in annulus_partition(loops):
        weight = _shoelace(cell)
        total += weight * float(
            polygon_analytic_greens(centre[:1], centre[1:], cell)[0][0]
        )
        area += weight
    return total / area


def stage_gate(args) -> None:
    """Check the swept route, the section-mean correction and the channel pair."""
    diameter = CABLE["PF"]
    payload = {"diameter": diameter, "channel": CHANNEL, "radius": args.radius}

    # The correction is a section property, so it must not move with the ring it
    # is measured on.  Reported over two decades of aspect ratio.
    payload["correction"] = {}
    for section in _gate_sections(diameter):
        record = {}
        for radius in args.radius_scan:
            flux = ring_flux(section, radius, args.height)
            record[f"{radius:g}"] = {
                "per_metre": (flux["centreline"] - flux["mean"]) / flux["length"],
                "centreline": flux["centreline"],
                "mean": flux["mean"],
                "aspect": diameter / (2 * radius),
            }
        payload["correction"][section.label] = record

    # The solid disc has a shipped axisymmetric finite-section element, so the
    # reference kernel is itself referenced before anything is built on it.
    solid = Section(diameter, 0.0, 64)
    shipped = float(
        ring_block(np.array([args.radius]), np.array([args.height]), diameter)[0, 0]
    )
    payload["axisymmetric"] = {
        "shipped_ring": shipped,
        "analytic_mean": ring_flux(solid, args.radius, args.height)["mean"],
    }

    # The channel superposition, against the annulus integrated cell by cell.
    payload["hollow_pair"] = {}
    for corners in CORNERS:
        section = Section(diameter, CHANNEL, corners)
        loops = section.boundaries(args.radius, args.height)
        pair = ring_flux(section, args.radius, args.height)
        payload["hollow_pair"][section.label] = {
            "pair_centreline": pair["centreline"],
            "partition_centreline": partition_flux(loops, args.radius, args.height),
            "cells": len(annulus_partition(loops)),
            "pair_area": pair["area"],
            "partition_area": sum(_shoelace(cell) for cell in annulus_partition(loops)),
            "annulus_area": np.pi / 4 * (diameter**2 - CHANNEL**2),
        }

    # And the swept route itself, on the ring the reference knows.
    payload["swept"] = {}
    for section in _gate_sections(diameter):
        record = {}
        reference = ring_flux(section, args.radius, args.height)
        for count in args.arc_scan:
            segments = ring_segments(args.radius, args.height, count)
            for intervals in args.interval_scan:
                start = time.perf_counter()
                direct = path_self(segments, section, intervals, args.radius)
                middle = time.perf_counter()
                framed = framed_self(segments, section, intervals, args.radius)
                record[f"{count}-{intervals}"] = direct | {
                    "reference": reference["mean"],
                    "relative": (direct["self"] - reference["mean"])
                    / reference["mean"],
                    "framed": framed["self"],
                    "framed_relative": (framed["self"] - reference["mean"])
                    / reference["mean"],
                    "seconds": middle - start,
                    "framed_seconds": time.perf_counter() - middle,
                }
        payload["swept"][section.label] = record

    # The elevation the footprint union is taken at, which is not a free choice:
    # a section straddling z = 0 comes back from the union with corners dropped.
    payload["footprint"] = {}
    for height in args.height_scan:
        for corners in CORNERS:
            section = Section(diameter, 0.0, corners)
            coilset = swept_coilset(ring_segments(args.radius, height, 2), section)
            payload["footprint"][f"{height:g}-{corners}"] = section_health(
                coilset, section
            )

    # The frame's own current split, which is what makes the pair sum.
    section = Section(diameter, CHANNEL, 64)
    coilset = swept_coilset(ring_segments(args.radius, args.height, 2), section)
    area = np.asarray(coilset.subframe["area"], dtype=float)
    factor = np.asarray(coilset.subframe["factor"], dtype=float)
    footprint = np.array(
        [poly.poly.area for poly in np.asarray(coilset.subframe["poly"])]
    )
    payload["current_split"] = {
        "rows": int(len(area)),
        "segments": int(len(area) // 2),
        "area_column": area.tolist(),
        "factor": factor.tolist(),
        "footprint": footprint.tolist(),
        "current_per_segment": float(
            (factor * footprint / area).sum() / (len(area) // 2)
        ),
    }
    print(f"wrote {_write('gate.json', payload)}")
    report_gate(payload)


def _gate_sections(diameter: float) -> list[Section]:
    """Return the sections the gate runs, solid and hollow, at every resolution."""
    return [Section(diameter, 0.0, corners) for corners in CORNERS] + [
        Section(diameter, CHANNEL, corners) for corners in CORNERS
    ]


def report_gate(payload: dict) -> None:
    """Print the gate: the correction, the reference, the pair and the route."""
    print(
        "\nthe section-mean correction is a property of the SECTION, so it must not"
        "\nmove with the ring it is measured on [H per metre of conductor]:\n"
    )
    radii = list(next(iter(payload["correction"].values())))
    print(f"{'section':<14}" + "".join(f"{f'R = {r} m':>18}" for r in radii))
    for label, record in payload["correction"].items():
        print(
            f"{label:<14}" + "".join(f"{record[r]['per_metre']:>18.10e}" for r in radii)
        )
    print(
        f"\n{'disc, analytic':<14}"
        f"{mu_0 / (8 * np.pi):>18.10e}   (mu0/8pi, the uniform-density disc)"
    )
    axisymmetric = payload["axisymmetric"]
    print(
        f"\nthe reference kernel against the shipped axisymmetric element, on a"
        f" disc:\n  shipped {axisymmetric['shipped_ring']:.9e} H"
        f"   analytic mean {axisymmetric['analytic_mean']:.9e} H"
        f"   relative"
        f" {abs(axisymmetric['analytic_mean'] / axisymmetric['shipped_ring'] - 1):.2e}"
    )
    print(
        "\nthe cooling channel as a linked pair, against the annulus integrated"
        "\ncell by cell at its own density:\n"
    )
    print(
        f"{'section':<14}{'pair [H]':>18}{'cells [H]':>18}{'relative':>12}{'cells':>8}"
    )
    for label, record in payload["hollow_pair"].items():
        pair, cells = record["pair_centreline"], record["partition_centreline"]
        print(
            f"{label:<14}{pair:>18.10e}{cells:>18.10e}"
            f"{abs(pair / cells - 1):>12.1e}{record['cells']:>8}"
        )
    print(
        "\nthe swept route on the ring the reference knows: centreline sum, the"
        "\nsection-mean correction it gives back, and what is left [H].  The two"
        "\nroutes are the closed forms driven DIRECTLY and the same closed forms"
        "\nreached through the frame, which agree here because a coaxial ring is the"
        "\none path whose swept footprint IS its section\n"
    )
    print(
        f"{'section':<14}{'arcs-intervals':>16}{'centreline':>16}{'correction':>14}"
        f"{'self':>16}{'reference':>16}{'direct':>11}{'framed':>11}{'s':>7}{'s':>7}"
    )
    for label, record in payload["swept"].items():
        for key, entry in record.items():
            print(
                f"{label:<14}{key:>16}{entry['centreline']:>16.9e}"
                f"{entry['correction']:>14.6e}{entry['self']:>16.9e}"
                f"{entry['reference']:>16.9e}{entry['relative']:>+11.1e}"
                f"{entry['framed_relative']:>+11.1e}{entry['seconds']:>7.2f}"
                f"{entry['framed_seconds']:>7.2f}"
            )
    print(
        "\nthe section column the swept element reads is a poloidal FOOTPRINT, and"
        "\nthe union that forms it is not unconditional -- corners are dropped where"
        "\nthe section straddles the machine midplane:\n"
    )
    print(
        f"{'elevation-corners':<20}{'rows':>6}{'short':>7}{'fewest':>8}"
        f"{'corner skew':>14}{'area ratio - 1':>17}"
    )
    for key, entry in payload["footprint"].items():
        print(
            f"{key:<20}{entry['rows']:>6}{entry['short_rows']:>7}"
            f"{entry['min_corners']:>8}{entry['worst_corner_skew']:>14.2e}"
            f"{entry['worst_area_ratio']:>17.2e}"
        )
    split = payload["current_split"]
    print(
        f"\nthe frame's current split, {split['rows']} rows over"
        f" {split['segments']} segments:"
        f"\n  area column {['%.6e' % value for value in split['area_column']]}"
        f"\n  factor      {split['factor']}"
        f"\n  footprint   {['%.6e' % value for value in split['footprint']]}"
        f"\n  sum of factor * footprint / area per segment ="
        f" {split['current_per_segment']:.12f}   (one ampere)"
    )


# ---------------------------------------------------------------------------
# The pack: every turn's self term from its own swept section.


def load_pack(name: str, args):
    """Return one coil's winding pack, cut at its turn boundaries."""
    path = {entry.name: entry for entry in read_paths(name[:2])}[name]
    path.resolution = args.resolution
    table = count_turns([path], args.revision, args.minimum_sweep, args.inflate)
    return cut_turns(path), table[name]


def turn_slices(pack) -> list[np.ndarray]:
    """Return the element indices of every turn, in path order."""
    return [np.flatnonzero(pack.turn == index) for index in range(pack.turn_count)]


def pack_self(
    pack, section: Section, intervals: int, turns=None, progress=None
) -> dict:
    """Return every turn's self-inductance from its own swept section.

    One coilset per turn rather than one for the pack: a swept element's cost is
    per pair, and a turn's own stations against the whole pack's elements would be
    two hundred and fifty times the work for an answer the geometric mean distance
    identity already gives as a filament.
    """
    radius, _ = pack.turn_centre()
    index = list(range(pack.turn_count)) if turns is None else list(turns)
    record, start = {}, time.perf_counter()
    for position, turn in enumerate(index):
        elements = np.flatnonzero(pack.turn == turn)
        segments = [pack.segments[element] for element in elements]
        record[int(turn)] = path_self(
            segments, section, intervals, float(radius[turn])
        ) | {"elements": int(len(segments))}
        if progress and (position + 1) % progress == 0:
            done = position + 1
            elapsed = time.perf_counter() - start
            print(
                f"  {done}/{len(index)} turns, {elapsed:.0f} s,"
                f" {elapsed / done:.2f} s/turn",
                flush=True,
            )
    return {
        "section": section.label,
        "diameter": section.diameter,
        "channel": section.channel,
        "corners": section.corners,
        "intervals": intervals,
        "turns": record,
        "total": sum(entry["self"] for entry in record.values()),
        "core": sum(entry["core"] for entry in record.values()),
        "centreline": sum(entry["centreline"] for entry in record.values()),
        "correction": sum(entry["correction"] for entry in record.values()),
        "length": sum(entry["length"] for entry in record.values()),
        "seconds": time.perf_counter() - start,
    }


def stage_self(args) -> None:
    """Sweep the pack's per-turn self term over the section and the quadrature."""
    pack, table = load_pack(args.coil, args)
    diameter = cable_diameter(pack.name)
    turns = None if args.turn_scan == 0 else _spread(pack.turn_count, args.turn_scan)
    payload = {
        "coil": pack.name,
        "turn_table": table,
        "turn_count": pack.turn_count,
        "measured_turns": pack.nturn,
        "diameter": diameter,
        "channel": CHANNEL,
        "subset": None if turns is None else [int(turn) for turn in turns],
        "runs": {},
    }
    for corners in args.corner_scan:
        for channel in args.channel_scan:
            section = Section(diameter, channel * CHANNEL, corners)
            for intervals in args.interval_scan:
                # the turn coverage is part of the key, so a subset sweep and a
                # whole-pack run merge into one ladder without overwriting each
                # other: they are different measurements of different things
                key = f"{section.label}-{intervals}"
                if turns is not None:
                    key += f"-of{len(turns)}"
                print(f"\n{key}", flush=True)
                payload["runs"][key] = pack_self(
                    pack, section, intervals, turns, args.progress
                )
                print(f"wrote {_write(args.output, payload)}", flush=True)
    report_self(payload)


def _spread(count: int, wanted: int) -> np.ndarray:
    """Return ``wanted`` turn indices spread evenly through a pack."""
    return np.unique(np.linspace(0, count - 1, min(wanted, count)).round().astype(int))


def report_self(payload: dict) -> None:
    """Print the per-turn self sweep, and what the channel is worth."""
    print(
        f"\n{payload['coil']}: {payload['turn_count']} turns,"
        f" {payload['measured_turns']:.3f} measured, cable"
        f" {1e3 * payload['diameter']:.1f} mm"
        + ("" if payload["subset"] is None else f", subset of {len(payload['subset'])}")
    )
    print(
        f"\n{'run':<22}{'centreline':>16}{'correction':>14}{'self [H]':>16}"
        f"{'channel':>14}{'length [m]':>12}{'s':>8}"
    )
    for key, run in payload["runs"].items():
        print(
            f"{key:<22}{run['centreline']:>16.9e}{run['correction']:>14.6e}"
            f"{run['total']:>16.9e}{run['core']:>+14.3e}{run['length']:>12.1f}"
            f"{run['seconds']:>8.0f}"
        )
    solid = {
        key.replace("annulus", "disc"): run["total"]
        for key, run in payload["runs"].items()
        if key.startswith("disc")
    }
    channel = {
        key: run["total"] - solid[key.replace("annulus", "disc")]
        for key, run in payload["runs"].items()
        if key.startswith("annulus") and key.replace("annulus", "disc") in solid
    }
    if channel:
        print(
            "\nwhat the cooling channel is worth: the annulus against the solid"
            "\ncable space at the same resolution [H]\n"
        )
        for key, value in channel.items():
            print(f"  {key:<22}{value:>+14.3e}")


# ---------------------------------------------------------------------------
# The seam: what the turn cut leaves for the filament to get wrong.


def stage_seam(args) -> None:
    """Measure the one pair set the turn split hands to the filament in error.

    The split holds out the pairs whose target and source lie in the same turn,
    and a ring -- or now a swept section -- stands in for them.  What it cannot
    hold out is the CONTIGUITY across a turn boundary: the conductor does not stop
    there, so a station just inside one turn sits a fraction of a station's spacing
    from an element of the next, where the two sections are separated along the
    conductor rather than across it and the geometric mean distance identity the
    filament rests on does not apply.

    So the boundary pairs are recomputed with the section resolved and differenced
    against the filament they were taken as.  Both routes see the same stations,
    the same elements and the same steps, so the difference is the seam and
    nothing else.  It is the same seam in the companion study, which is why the
    two ladders remain comparable whatever it comes to.
    """
    pack, _ = load_pack(args.coil, args)
    diameter = cable_diameter(pack.name)
    section = Section(diameter, args.channel * CHANNEL, args.corners)
    radius, _ = pack.turn_centre()
    boundaries = _spread(pack.turn_count - 1, args.boundary_scan)
    record, start = {}, time.perf_counter()
    for boundary in boundaries:
        pair = np.flatnonzero((pack.turn == boundary) | (pack.turn == boundary + 1))
        segments = [pack.segments[element] for element in pair]
        turn = pack.turn[pair]
        station, step, label = [], [], []
        for segment, index in zip(segments, turn):
            points, delta = _stations(segment, args.intervals)
            station.append(points)
            step.append(delta)
            label.append(np.full(len(points), index))
        station, step = np.vstack(station), np.vstack(step)
        label = np.concatenate(label)
        filament = centreline_linkage(
            _filament_coilset(segments, diameter), segments, args.intervals
        )
        swept_cross, pairs = 0.0, 0
        for column, (segment, index) in enumerate(zip(segments, turn)):
            cross = label != index
            potential = segment_potential(segment, section, station[cross])
            swept_cross += float(np.einsum("mnc,nc->m", potential, step[cross]).sum())
            pairs += int(cross.sum())
        record[int(boundary)] = {
            "swept": swept_cross,
            "filament": float(filament[_cross_turn(label, turn, filament)].sum()),
            "pairs": pairs,
            "elements": int(len(segments)),
            "radius": float(radius[boundary]),
        }
    payload = {
        "coil": pack.name,
        "section": section.label,
        "intervals": args.intervals,
        "boundaries": [int(value) for value in boundaries],
        "turn_count": pack.turn_count,
        "record": record,
        "seconds": time.perf_counter() - start,
    }
    difference = np.array(
        [entry["swept"] - entry["filament"] for entry in record.values()]
    )
    payload["mean_per_boundary"] = float(difference.mean())
    payload["pack_estimate"] = float(difference.mean() * (pack.turn_count - 1))
    print(f"wrote {_write('seam.json', payload)}")
    report_seam(payload)


# ---------------------------------------------------------------------------
# The conductor the coupling leaves out: the run to the terminals.


def filament_linkage(target: tuple, source: tuple, slab: int) -> float:
    """Return the flux one filament run links from another, ``mu0 A.dl`` [H].

    Both runs as their own quadrature stations and exact steps, summed in slabs so
    the pair matrix stays bounded.  A pair at zero separation -- which the diagonal
    of a run against itself has -- is dropped rather than accumulated, so the same
    call returns a finite number for a self block whose true value a filament cannot
    supply.  That block is reported only as the divergent estimate it is.
    """
    station, step = target
    source_station, source_step = source
    total = 0.0
    for first in range(0, len(station), slab):
        rows = slice(first, min(first + slab, len(station)))
        separation = station[rows, np.newaxis, :] - source_station[np.newaxis, :, :]
        distance = np.linalg.norm(separation, axis=-1)
        kernel = np.divide(
            1.0, distance, out=np.zeros_like(distance), where=distance > 0
        )
        total += float(np.einsum("ts,tc,sc->", kernel, step[rows], source_step))
    return mu_0 / (4 * np.pi) * total


def stage_feeder(args) -> None:
    """Measure the feeder run, which is conductor and is in neither ladder rung.

    The as-built centreline runs past the winding pack and out to the coil's
    terminals.  That run is separated before anything is counted, because it does
    not WIND -- it sweeps seven hundredths of a turn over its whole length -- and
    counting it as winding would put the turn count wrong.  But it is thirty-six
    metres of conductor carrying the coil's own current in series with the pack, so
    the coil's terminal inductance includes it and neither the companion's ladder
    nor this study's has ever had it in.

    Three terms, and they are reported separately because they answer different
    questions.  ``mutual`` is the pack against the feeder, taken BOTH ways so
    reciprocity is a check rather than an assumption; a radial or axial run carries
    no toroidal ``dl``, so what couples is only the fraction of a turn the run
    sweeps, and how far out it sweeps it decides how much.  ``self`` is the feeder
    against itself, from its own swept section, which is the only term that needs
    one -- its two legs run side by side and a filament cannot supply that.  The
    total the coil's terminals would see is the pack plus twice the mutual plus the
    feeder's own self.

    The mutual is taken by direct Biot-Savart over both runs' own quadrature
    stations rather than through the frame, because the frame cannot carry a feeder
    at all: it builds every segment's swept volume whatever the profile, and a run
    that travels radially or axially without sweeping has a degenerate poloidal
    footprint that the section reader raises on.  A filament sum needs no volume,
    and it is what the term wants anyway -- the pack and the feeder are separate
    conductor with no shared section, and where they meet the feeder's ``dl`` is
    across the pack's own potential rather than along it.
    """
    path = {entry.name: entry for entry in read_paths(args.coil[:2])}[args.coil]
    path.resolution = args.resolution
    table = count_turns([path], args.revision, args.minimum_sweep, args.inflate)
    pack = cut_turns(path)
    feeder = [path.segments[index] for index in np.flatnonzero(~path.pack)]
    diameter = cable_diameter(path.name)
    section = Section(diameter, args.channel * CHANNEL, args.corners)

    start = time.perf_counter()
    runs = []
    for segments in (pack.segments, feeder):
        station, step = [], []
        for segment in segments:
            points, delta = _stations(segment, args.intervals)
            station.append(points)
            step.append(delta)
        runs.append((np.vstack(station), np.vstack(step)))
    block = np.array(
        [
            [filament_linkage(target, source, args.slab) for source in runs]
            for target in runs
        ]
    )
    mutual_seconds = time.perf_counter() - start

    start = time.perf_counter()
    radius = float(
        np.mean([np.hypot(*_stations(segment, 1)[0][0][:2]) for segment in feeder])
    )
    feeder_self = path_self(feeder, section, args.intervals, radius)
    payload = {
        "coil": path.name,
        "section": section.label,
        "intervals": args.intervals,
        "feeder_elements": int(len(feeder)),
        "feeder_length": float(sum(segment.length for segment in feeder)),
        "feeder_sweep": float(path.swept(np.flatnonzero(~path.pack))),
        "pack_length": float(sum(segment.length for segment in pack.segments)),
        "mean_radius": radius,
        "path_radius_max": table[path.name]["path_radius_max"],
        "filament_block": block.tolist(),
        "mutual": 0.5 * float(block[0, 1] + block[1, 0]),
        "reciprocity": float(
            abs(block[0, 1] - block[1, 0]) / max(abs(block[0, 1]), 1e-30)
        ),
        "feeder_self": feeder_self,
        "filament_feeder_self": float(block[1, 1]),
        "mutual_seconds": mutual_seconds,
        "self_seconds": time.perf_counter() - start,
    }
    payload["terminal_step"] = 2 * payload["mutual"] + feeder_self["self"]
    print(f"wrote {_write('feeder.json', payload)}")
    report_feeder(payload)


def report_feeder(payload: dict) -> None:
    """Print what the feeder run adds to the coil's terminal inductance."""
    print(
        f"\n{payload['coil']}: the feeder run, {payload['feeder_elements']} elements,"
        f" {payload['feeder_length']:.2f} m of conductor sweeping"
        f" {payload['feeder_sweep']:.4f} turns out to"
        f" r = {payload['path_radius_max']:.1f} m"
        f"\nagainst a winding pack of {payload['pack_length']:.1f} m\n"
    )
    print(f"  {'pack against feeder, filament':<44}{payload['mutual']:>+14.3e} H")
    print(f"  {'reciprocity of that block':<44}{payload['reciprocity']:>14.1e}")
    print(
        f"  {'feeder against itself, section resolved':<44}"
        f"{payload['feeder_self']['self']:>+14.3e} H"
    )
    print(
        f"  {'the same term as a filament, which diverges':<44}"
        f"{payload['filament_feeder_self']:>+14.3e} H"
    )
    print(
        f"  {'what the terminals would see, 2 M + L':<44}"
        f"{payload['terminal_step']:>+14.3e} H"
    )


def _cross_turn(
    station_turn: np.ndarray, element_turn: np.ndarray, pairwise: np.ndarray
) -> np.ndarray:
    """Return which pairs cross a turn boundary, whatever the section costs in rows.

    A hollow section reaches the frame as a linked pair, so it carries TWO source
    rows per path element -- the outer boundary and the core -- where a solid one
    and a filament carry one.  Both members belong to the same element and so to
    the same turn, and the frame lays the cores out after the whole outer block, so
    the label array repeats by the row-to-element ratio.
    """
    repeat = pairwise.shape[1] // len(element_turn)
    labels = np.tile(element_turn, repeat)
    return station_turn[:, np.newaxis] != labels[np.newaxis, :]


def _filament_coilset(segments, diameter: float):
    """Return the filament coilset the companion study's cross-turn term uses."""
    from nova.frame.coilset import CoilSet
    from nova.geometry.polygeom import Polygon
    from nova.geometry.polyline import PolyLine

    coilset = CoilSet(field_attrs=["Ax", "Ay", "Az"], dwinding=0)
    polyline = PolyLine(minimum_arc_nodes=3, filament=True)
    polyline.segments = list(segments)
    coilset.winding.insert(
        polyline=polyline,
        cross_section=Polygon({"c": (0, 0, diameter)}),
        nturn=1,
        Ic=1,
        name="wire",
        part="wire",
        delta=0,
    )
    return coilset


def report_seam(payload: dict) -> None:
    """Print what the contiguity across a turn boundary is worth."""
    record = payload["record"]
    print(
        f"\n{payload['coil']}: the turn-boundary seam, section"
        f" {payload['section']}, {len(record)} of"
        f" {payload['turn_count'] - 1} boundaries\n"
    )
    print(
        f"{'boundary':>9}{'swept [H]':>16}{'filament [H]':>16}"
        f"{'difference':>14}{'pairs':>8}"
    )
    for boundary, entry in record.items():
        print(
            f"{boundary:>9}{entry['swept']:>16.8e}{entry['filament']:>16.8e}"
            f"{entry['swept'] - entry['filament']:>+14.2e}{entry['pairs']:>8}"
        )
    print(
        f"\nmean per boundary {payload['mean_per_boundary']:+.3e} H,"
        f" over {payload['turn_count'] - 1} boundaries"
        f" {payload['pack_estimate']:+.3e} H"
    )


# ---------------------------------------------------------------------------
# The ladder, extended by one rung.


def stage_ladder(args) -> None:
    """Put the resolved section on the companion study's attribution ladder.

    Exactly one thing changes from the last rung the companion reached: the
    per-turn self term, which was an axisymmetric ring carrying a declared disc
    and is now the turn's own path swept with its own section.  Everything else --
    the turn count, the outline, the path, and every turn-against-turn term -- is
    the companion's number, read from its own JSON rather than recomputed, so the
    step is the section and nothing else.
    """
    wound = _read(args.wound)
    section = _merge_sections(args.section)
    coils = wound["coils"]
    index = coils.index(section["coil"])
    machine = np.array(wound["machine_description"])
    cross = np.array(wound["wound"]["cross_turn"])
    ring = np.array(wound["wound"]["ring_self"])
    rungs = {
        name: np.array(matrix)[index, index]
        for name, matrix in wound["attribution"]["rungs"].items()
    }
    # the feeder run, which is conductor in series with the pack and is in no rung
    # either study has built: its own self plus twice its mutual with the pack
    terminal = 0.0
    if (FIGURES / "feeder.json").exists():
        terminal = _read("feeder.json")["terminal_step"]
    runs = {}
    for key, run in section["runs"].items():
        # a subset run measures a section against another section on the same
        # turns; it is not a rung, because the turns it leaves out carry
        # inductance the cross-turn term it would be added to already includes
        whole = len(run["turns"]) == section["turn_count"]
        resolved = cross[index, index] + run["total"] if whole else None
        runs[key] = {
            "self": run["total"],
            "ring_self": float(ring[index]) if whole else None,
            "step": run["total"] - float(ring[index]) if whole else None,
            "rung": resolved,
            "gap": float(machine[index, index]) - resolved if whole else None,
            "with_feeder": resolved + terminal if whole else None,
            "feeder_gap": (
                float(machine[index, index]) - resolved - terminal if whole else None
            ),
            "channel": run["core"],
            "centreline": run["centreline"],
            "correction": run["correction"],
            "length": run["length"],
            "turns_covered": len(run["turns"]),
        }
    payload = {
        "coil": section["coil"],
        "machine": float(machine[index, index]),
        "cross_turn": float(cross[index, index]),
        "rungs": {name: float(value) for name, value in rungs.items()},
        "gap": {
            name: float(machine[index, index] - value) for name, value in rungs.items()
        },
        "resolved": runs,
        "terminal_step": terminal,
        "subset": section["subset"],
        "turn_count": section["turn_count"],
    }
    if (FIGURES / "seam.json").exists():
        seam = _read("seam.json")
        payload["seam"] = {
            "mean_per_boundary": seam["mean_per_boundary"],
            "pack_estimate": seam["pack_estimate"],
            "boundaries": len(seam["record"]),
            "section": seam["section"],
        }
    print(f"wrote {_write('ladder.json', payload)}")
    report_ladder(payload)


def _merge_sections(names) -> dict:
    """Return several pack runs as one payload, one entry per section and mesh.

    A run is one section at one quadrature over one set of turns, and the runs are
    independent, so a sweep is a set of processes rather than a loop -- which is
    what lets the whole sweep land inside one allocation.  Merging them here keeps
    the ladder a single reading of a single set of numbers.
    """
    merged = None
    for name in [names] if isinstance(names, str) else names:
        payload = _read(name)
        if merged is None:
            merged = payload | {"runs": {}, "files": []}
        if payload["coil"] != merged["coil"]:
            raise ValueError(f"{name} is {payload['coil']}, not {merged['coil']}")
        merged["runs"] |= payload["runs"]
        merged["files"].append(name)
    return merged


def _section_pairs(resolved: dict) -> list[tuple[str, str, str]]:
    """Return the run pairs that isolate one input each, and what they isolate.

    A comparison is only attributable if the two runs share everything but the one
    thing under test, so the pairs are built by name rather than chosen: the same
    shape at two resolutions, the same resolution with and without the channel, and
    the same section at two quadratures.  Only pairs both of whose runs exist are
    returned, so a sweep that lost a run to a wall clock reports fewer rows rather
    than a wrong one.
    """
    pairs = []
    for key in resolved:
        shape, corners, intervals, *rest = key.split("-")
        suffix = "-".join(rest)
        tail = f"-{suffix}" if suffix else ""
        finer = f"{shape}-{2 * int(corners)}-{intervals}{tail}"
        if finer in resolved:
            name = f"{shape}, {corners} to {2 * int(corners)} corners"
            pairs.append((name, key, finer))
        hollow = f"annulus-{corners}-{intervals}{tail}"
        if shape == "disc" and hollow in resolved:
            name = f"the cooling channel at {corners} corners"
            pairs.append((name, key, hollow))
        mesh = f"{shape}-{corners}-{2 * int(intervals)}{tail}"
        if mesh in resolved:
            name = f"{shape}-{corners}, {intervals} to {2 * int(intervals)} intervals"
            pairs.append((name, key, mesh))
    return pairs


def report_ladder(payload: dict) -> None:
    """Print the ladder with the resolved section on the end of it."""
    machine = payload["machine"]
    print(
        f"\n{payload['coil']} self-inductance ladder [H], and what the machine"
        f" description still holds over each rung\n\n{'rung':<26}{'self [H]':>14}"
        f"{'gap to machine':>18}"
    )
    for name, value in payload["rungs"].items():
        print(f"{name:<26}{value:>14.6f}{machine - value:>+18.2e}")
    for key, run in payload["resolved"].items():
        if run["rung"] is None:
            continue
        print(f"{'wound section ' + key:<26}{run['rung']:>14.6f}{run['gap']:>+18.2e}")
    if payload["terminal_step"]:
        print(
            f"\nand the conductor no rung has ever had in -- the feeder run out to"
            f" the coil's\nterminals, in series with the pack, worth"
            f" {payload['terminal_step']:+.3e} H:\n"
        )
        for key, run in payload["resolved"].items():
            if run["with_feeder"] is None:
                continue
            print(
                f"{'+ feeders, ' + key:<26}{run['with_feeder']:>14.6f}"
                f"{run['feeder_gap']:>+18.2e}"
            )
    print(
        f"\nwhat the section is worth against the ring that stood in for it [H]\n"
        f"\n{'run':<22}{'turns':>7}{'ring self':>16}{'swept self':>16}"
        f"{'step':>13}{'channel':>13}"
    )
    for key, run in payload["resolved"].items():
        ring = "" if run["ring_self"] is None else f"{run['ring_self']:.9f}"
        step = "" if run["step"] is None else f"{run['step']:+.2e}"
        print(
            f"{key:<22}{run['turns_covered']:>7}{ring:>16}{run['self']:>16.9f}"
            f"{step:>13}{run['channel']:>+13.2e}"
        )
    print(
        "\nwhat resolving the section finer is worth, and what the channel takes"
        "\noff, both on the same turns [H]\n"
    )
    print(f"{'comparison':<44}{'difference':>14}{'relative':>12}")
    for name, first, second in _section_pairs(payload["resolved"]):
        low = payload["resolved"][first]["self"]
        high = payload["resolved"][second]["self"]
        print(f"{name:<44}{high - low:>+14.3e}{(high - low) / low:>+12.2e}")
    if "seam" in payload:
        seam = payload["seam"]
        print(
            f"\nthe turn-boundary seam, common to both rungs, measured over"
            f" {seam['boundaries']} boundaries at {seam['section']}:"
            f" {seam['pack_estimate']:+.2e} H over the pack"
        )


# ---------------------------------------------------------------------------
# Figures.


def _pack_footprint(args, diameter: float) -> dict:
    """Return one pack element's frame section column against its true section.

    Drawn from the longest arc of the pack, which is the element the smear is
    largest on, together with the pack-wide distribution of how far an element's
    own ends move in the poloidal plane -- the quantity that decides whether a
    swept footprint can stand in for a cross-section at all.
    """
    from nova.geometry.polygeom import Polygon
    from nova.geometry.section import poloidal_footprint
    from nova.geometry.volume import Sweep

    from benchmarks.wound_coil_inductance import _resolve

    pack, _ = load_pack(args.coil, args)
    step, length = [], []
    for segment in pack.segments:
        run = _resolve(segment, 3)
        ends = np.array(
            [[np.hypot(*point[:2]), point[2]] for point in (run[0], run[-1])]
        )
        step.append(float(np.linalg.norm(ends[1] - ends[0])))
        length.append(float(segment.length))
    section = Section(diameter, 0.0, 32)
    corners = Polygon(section.descriptor()).points
    drawn = pack.segments[int(np.argmax(np.asarray(step)))]
    poly = poloidal_footprint(Sweep(corners, drawn.path).section_loops)
    smear = np.asarray(poly.exterior.coords, dtype=float)[:-1]
    centreline = _resolve(drawn, 64)
    _, _, arc_radius, _ = _arc_frame(drawn)
    middle = centreline[len(centreline) // 2]
    return {
        "footprint": smear,
        "section": section.boundaries(float(np.hypot(*middle[:2])), float(middle[2]))[
            0
        ],
        "centreline": np.stack(
            [np.hypot(centreline[:, 0], centreline[:, 1]), centreline[:, 2]], axis=-1
        ),
        "ratio": _shoelace(smear) / section.area(),
        "arc_radius": arc_radius,
        "steps": step,
        "lengths": length,
    }


def stage_figures(args) -> None:
    """Draw the section, the two gates, and the extended ladder."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gate = _read("gate.json")
    diameter, channel = gate["diameter"], gate["channel"]

    # --- the conductor: what section the study means, and what the frame offers.
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.6))
    for corners, colour in zip(CORNERS, ("C0", "C1", "C2")):
        loops = Section(diameter, channel, corners).boundaries(0.0, 0.0)
        for order, loop in enumerate(loops):
            closed = np.r_[loop, loop[:1]]
            axes[0].plot(
                1e3 * closed[:, 0],
                1e3 * closed[:, 1],
                color=colour,
                lw=1.2,
                label=f"{corners} corners a boundary" if order == 0 else None,
            )
    theta = np.linspace(0, 2 * np.pi, 361)
    for scale, style in ((diameter, "-"), (channel, "--")):
        axes[0].plot(
            1e3 * 0.5 * scale * np.cos(theta),
            1e3 * 0.5 * scale * np.sin(theta),
            color="0.4",
            ls=style,
            lw=0.8,
            zorder=0,
        )
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("radial offset from the conductor centre [mm]")
    axes[0].set_ylabel("vertical offset [mm]")
    axes[0].set_title(
        f"the cable space, {1e3 * diameter:.0f} mm, about its"
        f" {1e3 * channel:.0f} mm cooling\nchannel: outer at +j, core at -j,"
        " one circuit",
        fontsize=9,
    )
    exact = np.pi / 4 * (diameter**2 - channel**2)
    deficit = {
        corners: 100 * (1 - Section(diameter, channel, corners).area() / exact)
        for corners in CORNERS
    }
    axes[0].annotate(
        "conducting area against the true annulus:\n"
        + "\n".join(
            f"  {corners:>3} corners   {value:+.2f} %"
            for corners, value in deficit.items()
        ),
        (0.03, 0.04),
        xycoords="axes fraction",
        fontsize=7,
        color="0.3",
    )
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].grid(alpha=0.3)

    footprint = _pack_footprint(args, diameter)
    section = np.r_[footprint["section"], footprint["section"][:1]]
    smear = np.r_[footprint["footprint"], footprint["footprint"][:1]]
    axes[1].plot(
        smear[:, 0],
        smear[:, 1],
        color="C3",
        lw=1.0,
        label=f"frame column, {len(footprint['footprint'])} corners",
    )
    axes[1].plot(
        section[:, 0],
        section[:, 1],
        color="C0",
        lw=1.6,
        label=f"the section, {len(footprint['section'])} corners",
    )
    axes[1].plot(
        footprint["centreline"][:, 0],
        footprint["centreline"][:, 1],
        color="0.5",
        ls=":",
        lw=1.0,
        label="the element's own centreline",
    )
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("radius [m]")
    axes[1].set_ylabel("elevation [m]")
    axes[1].set_title(
        f"one pack arc: the frame's section of record is the swept\nvolume's"
        f" FOOTPRINT, {footprint['ratio']:.1f} times the area the current occupies",
        fontsize=9,
    )
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    step = np.asarray(footprint["steps"])
    length = np.asarray(footprint["lengths"])
    order = np.argsort(step)
    axes[2].plot(
        1e3 * step[order],
        100 * np.cumsum(length[order]) / length.sum(),
        color="C0",
        lw=1.4,
    )
    axes[2].axvline(
        1e3 * diameter,
        color="C3",
        ls="--",
        lw=1.0,
        label=f"the section, {1e3 * diameter:.0f} mm",
    )
    crossing = 100 * length[step <= diameter].sum() / length.sum()
    axes[2].annotate(
        f"{100 - crossing:.0f} per cent of the pack sits on elements\n"
        f"that move further than their own section is wide",
        (0.04, 0.72),
        xycoords="axes fraction",
        fontsize=8,
        color="C3",
    )
    axes[2].set_xscale("log")
    axes[2].set_xlabel("how far an element's own ends move in (r, z) [mm]")
    axes[2].set_ylabel("cumulative share of the pack's length [%]")
    axes[2].set_title(
        "why the footprint is not the section: a winding\nelement advances across"
        " the pack as it sweeps",
        fontsize=9,
    )
    axes[2].legend(fontsize=7, loc="lower right")
    axes[2].grid(alpha=0.3, which="both")
    figure.suptitle(
        "the conductor this study resolves, and why its section had to be handed to"
        " the kernel directly"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(FIGURES / "section-conductor.svg")
    plt.close(figure)

    # --- the two gates: the correction, and the route against the reference.
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    radii = sorted(next(iter(gate["correction"].values())), key=float)
    for label, record in gate["correction"].items():
        axes[0].plot(
            [float(value) for value in radii],
            [record[value]["per_metre"] for value in radii],
            "o-",
            lw=1.0,
            ms=4,
            label=label,
        )
    axes[0].axhline(
        mu_0 / (8 * np.pi),
        color="C3",
        ls="--",
        lw=1.0,
        label=r"$\mu_0/8\pi$, exact disc",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("ring radius the correction is measured on [m]")
    axes[0].set_ylabel("centreline less section mean [H per metre]")
    axes[0].set_title(
        "the section-mean correction a centreline sum owes:\na property of the"
        " SECTION, not of the ring",
        fontsize=9,
    )
    axes[0].annotate(
        "the three resolutions of each section coincide on this scale:\n"
        "the correction is set by the section's SHAPE, not by how\n"
        "finely its boundary is cut",
        (0.03, 0.60),
        xycoords="axes fraction",
        fontsize=7,
        color="0.3",
    )
    axes[0].legend(fontsize=7, ncol=2, loc="center right")
    axes[0].grid(alpha=0.3)

    for route, marker, colour in (
        ("relative", "o", "C0"),
        ("framed_relative", "s", "C1"),
    ):
        for order, (label, record) in enumerate(gate["swept"].items()):
            corners = int(label.split("-")[-1])
            axes[1].plot(
                [corners] * len(record),
                [max(abs(entry[route]), 1e-13) for entry in record.values()],
                marker,
                color=colour,
                ms=5,
                alpha=0.8,
                label=(
                    ("driven directly" if route == "relative" else "through the frame")
                    if order == 0
                    else None
                ),
            )
    axes[1].set_yscale("log")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(list(CORNERS))
    axes[1].set_xticklabels([str(value) for value in CORNERS])
    axes[1].set_xlabel("corners a boundary is resolved at")
    axes[1].set_ylabel("relative error against the axisymmetric reference")
    axes[1].set_title(
        "both routes on a coaxial ring, the one path whose\nfootprint IS its"
        " section, against an independent kernel",
        fontsize=9,
    )
    floor = max(
        abs(entry["relative"])
        for record in gate["swept"].values()
        for entry in record.values()
    )
    axes[1].annotate(
        f"the floor is not the kernels but the section-mean correction's own radius\n"
        f"banding at {CORRECTION_RADIUS:g} m: {floor:.0e} relative, which over this"
        f" coil's whole self term is\n"
        f"{floor * 7.1e-03:.0e} H.  Evaluated at the ring's exact radius both routes"
        f" reach 1e-10.",
        (0.03, 0.06),
        xycoords="axes fraction",
        fontsize=7,
        color="0.3",
    )
    axes[1].set_ylim(1e-9, 1e-6)
    axes[1].legend(fontsize=7, loc="upper left")
    axes[1].grid(alpha=0.3, which="both")
    figure.suptitle("what the swept self term rests on, measured rather than argued")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(FIGURES / "section-gate.svg")
    plt.close(figure)

    if not (FIGURES / "ladder.json").exists():
        print(f"wrote figures to {FIGURES} (no ladder stage yet)")
        return
    ladder = _read("ladder.json")
    whole = {
        key: run for key, run in ladder["resolved"].items() if run["rung"] is not None
    }
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), width_ratios=[1.4, 1])
    names = list(ladder["rungs"]) + [f"wound section\n{key}" for key in whole]
    values = list(ladder["rungs"].values()) + [run["rung"] for run in whole.values()]
    if ladder["terminal_step"]:
        names += [f"+ feeders\n{key}" for key in whole]
        values += [run["with_feeder"] for run in whole.values()]
    base = ladder["rungs"]["continuum"]
    axes[0].axhline(0.0, color="C7", ls=":", lw=1.0, label="continuum rung")
    axes[0].axhline(
        ladder["machine"] - base,
        color="C3",
        ls="--",
        lw=1.2,
        label="machine description",
    )
    axes[0].plot(
        range(len(names)),
        [value - base for value in values],
        "o-",
        color="C0",
        label="ladder",
    )
    for position in range(1, len(names)):
        axes[0].annotate(
            f"{values[position] - values[position - 1]:+.1e}",
            (position - 0.5, 0.5 * (values[position] + values[position - 1]) - base),
            fontsize=7,
            ha="center",
            color="C0",
        )
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    axes[0].set_ylabel("offset from the continuum rung [H]")
    axes[0].set_title(
        f"{ladder['coil']}: does resolving the conductor section close what the"
        f"\nas-built winding path leaves?  (and what does?)",
        fontsize=9,
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    keys = list(whole)
    residual = [whole[key]["gap"] for key in keys]
    position = np.arange(len(keys))
    width = 0.38 if ladder["terminal_step"] else 0.55
    axes[1].axhline(
        ladder["gap"]["wound path"],
        color="C7",
        ls=":",
        lw=1.2,
        label=f"declared ring  {ladder['gap']['wound path']:+.2e}",
    )
    axes[1].axhline(0.0, color="C3", ls="--", lw=1.0, label="machine description")
    bars = [
        (
            position - 0.5 * width * bool(ladder["terminal_step"]),
            residual,
            "C0",
            "section resolved",
        )
    ]
    if ladder["terminal_step"]:
        bars.append(
            (
                position + 0.5 * width,
                [whole[key]["feeder_gap"] for key in keys],
                "C2",
                "and the feeders in",
            )
        )
    for centre, values_, colour, label in bars:
        axes[1].bar(centre, values_, width, color=colour, label=label)
        for at, value in zip(centre, values_):
            axes[1].annotate(
                f"{value:+.2e}",
                (at, value),
                textcoords="offset points",
                xytext=(0, 4 if value >= 0 else -11),
                ha="center",
                fontsize=7,
            )
    axes[1].set_xticks(position)
    axes[1].set_xticklabels(keys, rotation=25, ha="right", fontsize=7)
    axes[1].set_ylabel("machine description less the rung [H]")
    axes[1].set_title(
        "what is left: the section moves it by a per cent,\nthe feeder run by half of"
        " it",
        fontsize=9,
    )
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES / "section-ladder.svg")
    plt.close(figure)
    print(f"wrote figures to {FIGURES}")


# ---------------------------------------------------------------------------


def _write(name: str, payload: dict) -> pathlib.Path:
    """Write a stage result beside the figures and return its path."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    path.write_text(json.dumps(payload, indent=1))
    return path


def _read(name: str) -> dict:
    """Return a stage result written earlier."""
    return json.loads((FIGURES / name).read_text())


def parse_args(argv=None):
    """Return the parsed command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--minimum-sweep", type=float, default=0.3)
    parser.add_argument("--inflate", type=float, default=0.049)
    parser.add_argument("--revision", default="wide")
    stages = parser.add_subparsers(dest="stage", required=True)

    gate = stages.add_parser("gate")
    gate.add_argument("--radius", type=float, default=3.9431)
    gate.add_argument("--height", type=float, default=7.5641)
    gate.add_argument(
        "--radius-scan", type=float, nargs="+", default=[1.0, 3.9431, 40.0]
    )
    gate.add_argument("--height-scan", type=float, nargs="+", default=[0.0, 7.5641])
    gate.add_argument("--arc-scan", type=int, nargs="+", default=[2, 4])
    gate.add_argument("--interval-scan", type=int, nargs="+", default=[4])
    gate.set_defaults(run=stage_gate)

    solve = stages.add_parser("self")
    solve.add_argument("--coil", default="PF1")
    solve.add_argument("--corner-scan", type=int, nargs="+", default=list(CORNERS))
    solve.add_argument("--channel-scan", type=int, nargs="+", default=[0, 1])
    solve.add_argument("--interval-scan", type=int, nargs="+", default=[4])
    solve.add_argument("--turn-scan", type=int, default=0)
    solve.add_argument("--progress", type=int, default=25)
    solve.add_argument("--output", default="section.json")
    solve.set_defaults(run=stage_self)

    seam = stages.add_parser("seam")
    seam.add_argument("--coil", default="PF1")
    seam.add_argument("--corners", type=int, default=32)
    seam.add_argument("--channel", type=int, default=1)
    seam.add_argument("--intervals", type=int, default=4)
    seam.add_argument("--boundary-scan", type=int, default=12)
    seam.set_defaults(run=stage_seam)

    feeder = stages.add_parser("feeder")
    feeder.add_argument("--coil", default="PF1")
    feeder.add_argument("--corners", type=int, default=32)
    feeder.add_argument("--channel", type=int, default=1)
    feeder.add_argument("--intervals", type=int, default=4)
    feeder.add_argument("--slab", type=int, default=4000)
    feeder.set_defaults(run=stage_feeder)

    ladder = stages.add_parser("ladder")
    ladder.add_argument("--wound", default="wound.json")
    ladder.add_argument("--section", nargs="+", default=["section.json"])
    ladder.set_defaults(run=stage_ladder)

    figures = stages.add_parser("figures")
    figures.add_argument("--coil", default="PF1")
    figures.set_defaults(run=stage_figures)
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    arguments.run(arguments)
