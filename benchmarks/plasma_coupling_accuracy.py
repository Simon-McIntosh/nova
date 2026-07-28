"""What the banded plasma-coupling default costs against exact everywhere.

``segment="circle"`` (:mod:`nova.biot.circle`) evaluates each source cell's own
finite section inside a band of the section's radii and the point filament outside
it, and averages the target cell's own section over the pairs closest in. Exact
everywhere -- the closed form of :mod:`nova.biot.polygonanalytic` on every pair,
which is what :mod:`nova.biot.polysection` ships -- is the reference. So what this
measures is the SEAM: what the shipped element gives up by banding the finite
section rather than paying for it on all of a plasma grid's pairs.

Why the diagonal is reported apart
----------------------------------
An all-to-all plasma-plasma interaction matrix puts a target inside its own source
cell on every diagonal entry. For a point filament that is a coincident target,
log-singular, and no answer at all; for a finite section it is an ordinary interior
point where the flux and both field components are bounded and smooth. The
diagonal is also the one place the shipped element evaluates a DOUBLE integral --
the source section's flux averaged over the target cell's own section -- where the
exact-everywhere reference evaluates a single integral at the cell centre, so the
two differ there by design and by the size of the target-side average. It is
reported separately rather than averaged into the bulk.

The reference
-------------
The closed form is the reference here, and it is not taken on faith. It is
independently checked over this grid's own pair population against the boundary
quadrature at a rule far above the shipped one (64 x 96 against 16 x 48), which
converges spectrally off the section boundary; :mod:`tests.test_biotpolygonanalytic`
records that oracle as converged to about 1e-11 everywhere. The check is run
across the whole contour-distance range, with the shipped rule measured against
the same oracle on the same pairs -- so where the two candidate evaluations
disagree, the measurement says which one moved.

Normalisation
-------------
Every error is reported two ways, because neither alone is honest for all three
components. LOCAL relative error, ``|point - exact| / |exact|``, is what a claim
like "the self-flux term is 7% wrong" means, but it is meaningless where the exact
value passes through zero -- which is exactly what ``B_R`` does on the diagonal of
a section symmetric about its own centroid. COLUMN-PEAK relative error,
``|point - exact| / max_target |exact|``, is finite everywhere and is the scale the
assembled operator's action actually sees. Both are tabulated; where they tell
different stories the text says so.

    python benchmarks/plasma_coupling_accuracy.py <output.json> [figure.png]
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

from nova.biot.bandedcoupling import NEAR_LIMIT, NEAR_RULE, contour_distance
from nova.biot.biotframe import Target
from nova.biot.greens import greens_psi, second_moments, section_centroid
from nova.biot.polygon import polygon_greens
from nova.biot.polygonanalytic import polygon_analytic_greens
from nova.biot.solve import Solve
from nova.frame.coilset import CoilSet

COMPONENTS = ("psi", "B_R", "B_Z")

WALL = {"e": [6.2, 0.5, 3.0, 5.0]}
"""Elliptical first wall ``[r0, z0, width, height]`` [m] -- ITER-scale."""

CELLS = 300
"""Requested plasma cell count; the tiling delivers the nearest it can fit."""

PLASMA_CURRENT = 15.0e6
"""Total plasma current [A] the integrated statement is scaled to."""

ORACLE_RULE = dict(n_panels=64, n_nodes=96)
"""Boundary quadrature rule serving as the independent reference check."""

ORACLE_SAMPLE = 2000
"""Pairs drawn at random from beyond half a section radius for the oracle check.

Every pair INSIDE half a radius is taken, not sampled: there are a few dozen on a
real grid, they are the ones the two evaluations can disagree on, and sampling
them would be sampling away the whole question.
"""

CLOSE_CONTOUR = 0.5
"""Contour distance, in section radii, below which every pair goes to the oracle."""

QUARTER_CONTOUR = 0.25
"""Contour distance, in section radii, inside which the shipped rule is known to
lose digits -- the band whose gain from the closed form is being sized."""

REFINEMENT_LADDER = ((16, 48), (24, 64), (32, 80), (48, 96), (64, 96), (96, 128))
"""Boundary quadrature rules, in increasing order, for the refinement check.

Whether the closed form or the shipped rule is the one in error near the contour is
settled by refining the rule: a quadrature that is not yet converged moves TOWARDS
the true value as its node count rises, so if the ladder walks onto the closed form
the closed form is what it is converging to.
"""

SWEEP_CELLS = (150, 300, 600, 1200)
"""Grid resolutions the self-term error is re-measured at.

The self term's error is a ratio of two logarithms of the cell size, so it varies
only slowly with resolution -- but "only slowly" has to be measured rather than
asserted, because a single grid cannot tell a robust figure from a coincidence.
"""

RING_EDGES = (1.37, 2.32)
"""Centre-separation cuts, in lattice pitch, between first / second-third / bulk.

A hexagonal tiling's shells sit at 1, sqrt(3), 2, sqrt(7), ... times the pitch, so
the cuts are the midpoints between the first and second shells and between the
third and fourth. Cells clipped by the wall have centroids closer together than
the pitch; they fall into the first-neighbour class, which is where they belong.
"""

SEED = 20260726
"""Fixed sampling seed, so every number here is reproducible from the file alone."""


# --- the grid ---------------------------------------------------------------


def plasma_grid(cells: int = CELLS):
    """Return a real hexagonally-tiled plasma grid inside an elliptical wall.

    Returns the coilset (the point-filament matrices come from it, through the
    shipped solve path) alongside the per-cell geometry the closed form needs:
    centres, section polygons, areas and section labels. The tiling produces both
    regular hexagons and cells clipped by the wall, and the two behave differently
    -- a clipped cell has no symmetry to cancel its odd moments, and its centroid
    can sit far closer to its own boundary than a hexagon's does.
    """
    coilset = CoilSet(dplasma=-cells, tplasma="hex")
    coilset.firstwall.insert(WALL, Ic=PLASMA_CURRENT)
    subframe = coilset.subframe
    vertices = []
    for poly in np.asarray(subframe["poly"]):
        points = np.asarray(poly.points, dtype=np.float64)[:, [0, 2]]
        if len(points) > 1 and np.allclose(points[0], points[-1]):
            points = points[:-1]  # drop the repeated closing vertex
        vertices.append(points)
    return coilset, dict(
        centre_r=np.asarray(subframe.x, dtype=np.float64),
        centre_z=np.asarray(subframe.z, dtype=np.float64),
        filament_r=np.asarray(subframe.rms, dtype=np.float64),
        width=np.asarray(subframe.dx, dtype=np.float64),
        height=np.asarray(subframe.dz, dtype=np.float64),
        area=np.asarray(subframe.area, dtype=np.float64),
        section=np.asarray(subframe.section),
        vertices=vertices,
    )


def section_radii(vertices: list[np.ndarray]) -> np.ndarray:
    """Return each section's bounding radius about its area centroid [m]."""
    return np.array(
        [np.max(np.hypot(*(np.asarray(v) - section_centroid(v)).T)) for v in vertices]
    )


def lattice_pitch(grid: dict) -> float:
    """Return the tiling's centre-to-centre pitch [m], from the regular cells only.

    The median nearest-neighbour separation over cells the wall did not clip. Taking
    the minimum over all cells instead would return a clipped sliver's centroid
    spacing, which is not the lattice pitch and would misclassify every ring.
    """
    keep = np.flatnonzero(grid["section"] == "hexagon")
    r, z = grid["centre_r"][keep], grid["centre_z"][keep]
    separation = np.hypot(r[:, None] - r[None, :], z[:, None] - z[None, :])
    np.fill_diagonal(separation, np.inf)
    return float(np.median(np.min(separation, axis=1)))


# --- the two couplings ------------------------------------------------------


def element_coupling(coilset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the shipped plasma-plasma matrices, per ampere.

    Assembled through :class:`nova.biot.solve.Solve` exactly as
    :meth:`nova.biot.plasmagrid.PlasmaGrid.solve` does -- targets are the plasma
    cell centres, sources are the same cells -- rather than reimplemented here, so
    what is measured is the code path a solve takes. The ``<attr>_`` variables are
    the plasma-source block before any turn weighting or reduction, which is the
    per-ampere coupling the comparison needs.
    """
    target = Target({attr: coilset.aloc["plasma", attr] for attr in ["x", "z", "poly"]})
    data = Solve(
        coilset.subframe,
        target,
        reduce=[True, False],
        attrs=["Br", "Bz", "Psi"],
        name="point",
    ).data
    return tuple(np.asarray(data[f"{attr}_"].data) for attr in ("Psi", "Br", "Bz"))


def exact_coupling(grid: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the closed-form finite-section matrices, per ampere.

    One column per source cell: the whole target set against that cell's own
    polygon, which is what :meth:`nova.biot.polysection.PolySection._coupling`
    does. Uniform current density over the true section, so the diagonal is an
    ordinary interior evaluation with no offset, floor or cap anywhere in it.
    """
    target_r, target_z = grid["centre_r"], grid["centre_z"]
    shape = (target_r.size, len(grid["vertices"]))
    psi, br, bz = (np.empty(shape) for _ in range(3))
    for column, vertices in enumerate(grid["vertices"]):
        psi[:, column], br[:, column], bz[:, column] = polygon_analytic_greens(
            target_r, target_z, vertices
        )
    return psi, br, bz


def sparse_quadrature(
    grid: dict, mask: np.ndarray, **rule
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the boundary quadrature on the masked pairs only, NaN elsewhere.

    The converged rule costs three orders more per pair than the closed form, so
    the reference check and the near-band comparison both run on a pair subset. NaN
    outside the mask rather than zero, so an accidental read of an unevaluated pair
    propagates instead of quietly contributing a wrong number.
    """
    values = [np.full(mask.shape, np.nan) for _ in range(3)]
    for column, vertices in enumerate(grid["vertices"]):
        rows = np.flatnonzero(mask[:, column])
        if rows.size == 0:
            continue
        computed = polygon_greens(
            grid["centre_r"][rows], grid["centre_z"][rows], vertices, **rule
        )
        for store, got in zip(values, computed):
            store[rows, column] = got
    return tuple(values)


# --- pair classification ----------------------------------------------------


def pair_geometry(grid: dict) -> dict:
    """Return the per-pair contour distance, centre separation and class masks."""
    radius = section_radii(grid["vertices"])
    target_r, target_z = grid["centre_r"], grid["centre_z"]
    contour = np.empty((target_r.size, len(grid["vertices"])))
    for column, vertices in enumerate(grid["vertices"]):
        contour[:, column] = contour_distance(target_r, target_z, vertices)
    scaled = contour / radius[None, :]
    separation = np.hypot(
        target_r[:, None] - target_r[None, :], target_z[:, None] - target_z[None, :]
    )
    pitch = lattice_pitch(grid)
    shell = separation / pitch
    diagonal = np.eye(target_r.size, dtype=bool)
    first = ~diagonal & (shell < RING_EDGES[0])
    middle = (shell >= RING_EDGES[0]) & (shell < RING_EDGES[1])
    bulk = shell >= RING_EDGES[1]
    return dict(
        radius=radius,
        contour=contour,
        scaled=scaled,
        separation=separation,
        pitch=pitch,
        shell=shell,
        classes={
            "diagonal": diagonal,
            "first ring": first,
            "second-third ring": middle,
            "bulk": bulk,
        },
    )


def statistics(error: np.ndarray, mask: np.ndarray) -> dict:
    """Return the error distribution over the masked pairs.

    A median and a maximum alone hide a heavy tail, and the recorded claim this
    reproduces is a median-and-maximum pair, so the deciles that carry the tail are
    reported with them.
    """
    sample = error[mask]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return dict(count=0)
    return dict(
        count=int(sample.size),
        median=float(np.median(sample)),
        p90=float(np.percentile(sample, 90)),
        p99=float(np.percentile(sample, 99)),
        maximum=float(np.max(sample)),
        mean=float(np.mean(sample)),
    )


def error_tables(point, exact, geometry) -> dict:
    """Return per-component, per-pair-class point-kernel error, both normalisations."""
    tables = {}
    for name, got, want in zip(COMPONENTS, point, exact):
        peak = np.max(np.abs(want), axis=0)
        deviation = np.abs(got - want)
        local = np.divide(
            deviation,
            np.abs(want),
            out=np.full_like(deviation, np.nan),
            where=np.abs(want) > 0.0,
        )
        column = deviation / peak[None, :]
        tables[name] = {
            label: dict(
                local=statistics(local, mask),
                column_peak=statistics(column, mask),
            )
            for label, mask in geometry["classes"].items()
        }
        tables[name]["scale"] = dict(
            column_peak_median=float(np.median(peak)),
            matrix_peak=float(np.max(np.abs(want))),
        )
    return tables


# --- what the point kernel does on the diagonal -----------------------------


def diagonal_detail(point, exact, grid, geometry) -> dict:
    """Return the SIGNED self-term error, split by whether the wall clipped the cell.

    Sign matters here and a magnitude hides it: a self term uniformly low by a fixed
    fraction is a rescaled diagonal, which a solve partly absorbs, whereas a self
    term of the wrong sign is a different operator. The two cell populations are
    separated because they fail differently -- a regular hexagon is symmetric about
    its own centroid and a clipped one is not, and the point model has no way to
    know the difference.
    """
    regular = grid["section"] == "hexagon"
    report = {}
    for name, got, want in zip(COMPONENTS, point, exact):
        got, want = np.diag(got), np.diag(want)
        peak = np.max(np.abs(np.asarray(exact[COMPONENTS.index(name)])), axis=0)
        signed = np.divide(
            got - want,
            np.abs(want),
            out=np.full(want.shape, np.nan),
            where=np.abs(want) > 0.0,
        )
        column = np.abs(got - want) / peak
        entry = dict(
            exact_median=float(np.median(want)),
            point_median=float(np.median(got)),
            signed_median=float(np.nanmedian(signed)),
            signed_extreme=float(signed[np.nanargmax(np.abs(signed))]),
            fraction_low=float(np.mean(signed < 0.0)),
            sign_flips=int(np.sum(np.sign(got) != np.sign(want))),
        )
        for label, mask in (("regular", regular), ("wall-clipped", ~regular)):
            entry[label] = dict(
                cells=int(mask.sum()),
                local_median=float(np.nanmedian(np.abs(signed[mask]))),
                local_maximum=float(np.nanmax(np.abs(signed[mask]))),
                column_peak_median=float(np.median(column[mask])),
                column_peak_maximum=float(np.max(column[mask])),
            )
        worst = int(np.argmax(column))
        entry["worst_cell"] = dict(
            section=str(grid["section"][worst]),
            section_radius=float(geometry["radius"][worst]),
            contour_over_radius=float(geometry["scaled"][worst, worst]),
            exact=float(want[worst]),
            point=float(got[worst]),
            column_peak=float(peak[worst]),
        )
        report[name] = entry
    return report


def resolution_sweep(cell_counts=SWEEP_CELLS) -> dict:
    """Return the self term AND the integrated effect at several grid resolutions.

    Both are needed, because they move in opposite directions. The self term is a
    ratio of two coupling values at one cell, and the target-side average the shipped
    diagonal carries scales with the cell while the reference's single integral at the
    cell centre does not, so the per-pair figure moves with resolution. The integrated
    effect weights that same term by the cell's own current, which falls as the area,
    so it can fall while the per-pair figure rises. Quoting only one of the two
    would let the reader draw the opposite conclusion from the other.
    """
    sweep = {}
    for cells in cell_counts:
        coilset, grid = plasma_grid(cells)
        point = element_coupling(coilset)
        exact = exact_coupling(grid)
        regular = grid["section"] == "hexagon"
        radius = section_radii(grid["vertices"])
        entry = dict(
            cells=int(grid["centre_r"].size),
            regular=int(regular.sum()),
            median_section_radius=float(np.median(radius[regular])),
        )
        current = current_profile(grid)
        for name, got, want in zip(COMPONENTS, point, exact):
            diagonal_got, diagonal_want = np.diag(got), np.diag(want)
            peak = np.max(np.abs(want), axis=0)
            signed = np.divide(
                diagonal_got - diagonal_want,
                np.abs(diagonal_want),
                out=np.full(diagonal_want.shape, np.nan),
                where=np.abs(diagonal_want) > 0.0,
            )
            column = np.abs(diagonal_got - diagonal_want) / peak
            difference = (got - want) @ current
            swing = float(np.ptp(want @ current))
            entry[name] = dict(
                signed_median=float(np.nanmedian(signed)),
                signed_median_regular=float(np.nanmedian(signed[regular])),
                magnitude_maximum=float(np.nanmax(np.abs(signed))),
                column_peak_median=float(np.median(column)),
                column_peak_maximum=float(np.max(column)),
                swing=swing,
                integrated_median_of_swing=float(np.median(np.abs(difference)) / swing),
                integrated_max_of_swing=float(np.max(np.abs(difference)) / swing),
            )
        entry["energy_relative"] = float(
            abs(current @ (point[0] - exact[0]) @ current)
            / abs(current @ exact[0] @ current)
        )
        sweep[str(cells)] = entry
    return sweep


# --- the reference check ----------------------------------------------------


def reference_check(grid, exact, geometry) -> dict:
    """Check the closed form against a far-above-shipped boundary quadrature.

    Both candidate evaluations are measured against the SAME oracle on the SAME
    pairs, which is what makes the check decide anything: away from the contour all
    three must agree to round-off, and where the shipped rule departs the oracle
    says whether the closed form went with it or stayed put.
    """
    scaled = geometry["scaled"]
    close = scaled < CLOSE_CONTOUR
    rest = np.flatnonzero(~close.ravel())
    drawn = np.random.default_rng(SEED).choice(
        rest, size=min(ORACLE_SAMPLE, rest.size), replace=False
    )
    mask = close.copy()
    mask.ravel()[drawn] = True
    oracle = sparse_quadrature(grid, mask, **ORACLE_RULE)
    shipped = sparse_quadrature(grid, mask, n_panels=NEAR_RULE[0], n_nodes=NEAR_RULE[1])
    bands = (
        ("inside a quarter radius", scaled < QUARTER_CONTOUR),
        ("quarter to half a radius", (scaled >= QUARTER_CONTOUR) & close),
        ("half a radius and beyond", ~close),
    )
    report = {}
    for name, reference, want, alternative in zip(COMPONENTS, oracle, exact, shipped):
        peak = np.max(np.abs(want), axis=0)[None, :]
        closed_error = np.abs(want - reference) / peak
        shipped_error = np.abs(alternative - reference) / peak
        report[name] = {
            label: dict(
                closed_form=statistics(closed_error, band & mask),
                shipped_rule=statistics(shipped_error, band & mask),
            )
            for label, band in bands
        }
    report["pairs_checked"] = int(mask.sum())
    report["pairs_inside_half_radius"] = int(close.sum())
    return report


def refinement_check(grid, exact, geometry) -> dict:
    """Return how the quadrature moves, near the contour, as its rule is refined.

    Run on the pairs inside half a section radius, where the shipped rule and the
    closed form disagree. If the disagreement were the closed form's, refining the
    quadrature would leave it where it is; the ladder instead has to walk onto the
    closed form for the closed form to be the reference. Reported as worst deviation
    from the closed form per rule, per component, as a fraction of column peak.
    """
    mask = geometry["scaled"] < CLOSE_CONTOUR
    report = dict(
        rules=[list(rule) for rule in REFINEMENT_LADDER], pairs=int(mask.sum())
    )
    for name in COMPONENTS:
        report[name] = []
    for rule in REFINEMENT_LADDER:
        computed = sparse_quadrature(grid, mask, n_panels=rule[0], n_nodes=rule[1])
        for name, got, want in zip(COMPONENTS, computed, exact):
            peak = np.max(np.abs(want), axis=0)[None, :]
            report[name].append(float(np.max((np.abs(got - want) / peak)[mask])))
    return report


# --- the near band ----------------------------------------------------------


def near_band_gain(grid, exact, geometry) -> dict:
    """Return what the closed form buys the near band, and on what share of pairs.

    The near band is the pairs inside :data:`nova.biot.bandedcoupling.NEAR_LIMIT`
    section radii of the contour; the shipped near rule and the closed form are
    compared there directly, banded by contour distance. The population fractions
    matter as much as the deviations: a gain that applies to a thousandth of a
    percent of the matrix is a robustness argument, not an accuracy one.
    """
    scaled = geometry["scaled"]
    near = scaled < NEAR_LIMIT
    shipped = sparse_quadrature(grid, near, n_panels=NEAR_RULE[0], n_nodes=NEAR_RULE[1])
    bands = (
        ("inside a quarter radius", scaled < QUARTER_CONTOUR),
        ("quarter to half a radius", (scaled >= QUARTER_CONTOUR) & (scaled < 0.5)),
        ("half to one radius", (scaled >= 0.5) & (scaled < 1.0)),
        ("one radius to the near seam", (scaled >= 1.0) & near),
    )
    report = {}
    for name, want, alternative in zip(COMPONENTS, exact, shipped):
        peak = np.max(np.abs(want), axis=0)[None, :]
        deviation = np.abs(alternative - want) / peak
        report[name] = {
            label: statistics(deviation, band & near) for label, band in bands
        }
    total = scaled.size
    diagonal = geometry["classes"]["diagonal"]
    report["population"] = dict(
        pairs=int(total),
        near_band_fraction=float(near.mean()),
        inside_quarter_radius_fraction=float((scaled < QUARTER_CONTOUR).mean()),
        inside_quarter_radius_pairs=int((scaled < QUARTER_CONTOUR).sum()),
        inside_quarter_radius_on_diagonal=int(
            (scaled < QUARTER_CONTOUR)[diagonal].sum()
        ),
        inside_half_radius_fraction=float((scaled < CLOSE_CONTOUR).mean()),
        diagonal_scaled_contour_median=float(np.median(scaled[diagonal])),
        diagonal_scaled_contour_minimum=float(np.min(scaled[diagonal])),
        clipped_cells=int(np.sum(grid["section"] != "hexagon")),
        regular_cells=int(np.sum(grid["section"] == "hexagon")),
    )
    return report


# --- does it matter to a solve ---------------------------------------------


def current_profile(grid: dict) -> np.ndarray:
    """Return a parabolic plasma current distribution over the cells [A].

    ``j ~ (1 - rho^2)`` on the wall's own normalised elliptical radius, times cell
    area, scaled to :data:`PLASMA_CURRENT`. Peaked on axis and vanishing at the
    boundary, which is what makes the self and first-neighbour terms carry the
    operator's action -- a flat profile would understate them.
    """
    r0, z0, width, height = WALL["e"]
    rho2 = ((grid["centre_r"] - r0) / (width / 2)) ** 2 + (
        (grid["centre_z"] - z0) / (height / 2)
    ) ** 2
    density = np.clip(1.0 - rho2, 0.0, None)
    current = density * grid["area"]
    return PLASMA_CURRENT * current / current.sum()


def integrated_effect(point, exact, grid) -> dict:
    """Return the difference the two models make to quantities a solve compares.

    Pairwise error does not say whether a reconstruction would notice. These are
    the assembled operator's ACTION on a realistic current profile: the poloidal
    flux and both field components at every cell from the whole plasma current
    distribution, and the magnetic energy the flux matrix carries -- one scalar
    that weights every pair by the currents that actually flow through it.
    """
    current = current_profile(grid)
    report = dict(
        plasma_current=float(current.sum()),
        peak_cell_current=float(np.max(current)),
    )
    for name, got, want in zip(COMPONENTS, point, exact):
        point_field = got @ current
        exact_field = want @ current
        difference = point_field - exact_field
        swing = float(np.ptp(exact_field))
        report[name] = dict(
            exact_range=[float(np.min(exact_field)), float(np.max(exact_field))],
            swing=swing,
            median_absolute=float(np.median(np.abs(difference))),
            max_absolute=float(np.max(np.abs(difference))),
            median_of_swing=float(np.median(np.abs(difference)) / swing),
            max_of_swing=float(np.max(np.abs(difference)) / swing),
            median_relative=float(np.median(np.abs(difference / exact_field))),
            max_relative=float(np.max(np.abs(difference / exact_field))),
            difference=difference,
        )
    point_energy = 0.5 * float(current @ point[0] @ current)
    exact_energy = 0.5 * float(current @ exact[0] @ current)
    report["energy"] = dict(
        point=point_energy,
        exact=exact_energy,
        relative=abs(point_energy - exact_energy) / abs(exact_energy),
    )
    report["current"] = current
    return report


def second_moment_scale(grid: dict) -> dict:
    """Return the section's own second-moment correction size, as a sanity check.

    Spreading a filament over a section changes its coupling in leading order by
    the section's second moments times the curvature of the ring Green's function,
    which for a full ring is set by the major radius. It puts a scale on the bulk
    error -- ``(a / R0)^2`` -- that the measured bulk figures should sit near, and
    it is why the bulk error does not fall off with distance.
    """
    radius = section_radii(grid["vertices"])
    keep = np.flatnonzero(grid["section"] == "hexagon")
    moments = np.array([second_moments(grid["vertices"][i])[:2] for i in keep])
    return dict(
        median_section_radius=float(np.median(radius)),
        median_radius_over_major=float(np.median(radius / grid["centre_r"])),
        aspect_squared=float(np.median((radius / grid["centre_r"]) ** 2)),
        median_second_moment_over_major_squared=float(
            np.median(moments.sum(axis=1) / grid["centre_r"][keep] ** 2)
        ),
    )


# --- the figure -------------------------------------------------------------


def resolution_inset(axis, sweep):
    """Draw the resolution trend of the self term and of the integrated effect."""
    inset = axis.inset_axes((0.32, 0.54, 0.38, 0.32))
    inset.set_facecolor("white")
    inset.patch.set_alpha(0.96)
    inset.set_zorder(5)
    radius = [entry["median_section_radius"] * 1e3 for entry in sweep.values()]
    curves = (
        ("psi self term", "psi", "signed_median", "C0", "o-"),
        ("B_Z integrated", "B_Z", "integrated_median_of_swing", "C2", "s--"),
        ("psi integrated", "psi", "integrated_median_of_swing", "C0", "^--"),
    )
    for label, name, key, colour, style in curves:
        values = [100 * abs(entry[name][key]) for entry in sweep.values()]
        inset.plot(radius, values, style, lw=1.3, ms=4, color=colour, label=label)
    inset.set_yscale("log")
    inset.set_xlabel("cell radius  [mm]", fontsize=7)
    inset.set_ylabel("error  [%]", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.grid(alpha=0.25)
    inset.legend(fontsize=5.5, loc="lower right")
    inset.set_title("against grid resolution", fontsize=7, pad=2)


def figure(
    path, point, exact, geometry, grid, near, reference, refinement, integrated, sweep
):
    """Write the six-panel evidence figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colour = dict(zip(COMPONENTS, ("C0", "C1", "C2")))
    fig, axes = plt.subplots(2, 3, figsize=(19.0, 10.5))
    rng = np.random.default_rng(SEED)
    regular = grid["section"] == "hexagon"

    # (a) point-kernel error against contour distance, per component
    axis = axes[0, 0]
    scaled = geometry["scaled"]
    diagonal = geometry["classes"]["diagonal"]
    thin = rng.random(scaled.shape) < min(1.0, 40000 / scaled.size)
    for name, got, want in zip(COMPONENTS, point, exact):
        peak = np.max(np.abs(want), axis=0)[None, :]
        error = np.abs(got - want) / peak
        show = thin & ~diagonal
        axis.plot(
            scaled[show],
            error[show],
            ".",
            ms=1.6,
            alpha=0.22,
            color=colour[name],
            label=f"{name}, off-diagonal",
        )
        axis.plot(
            np.diag(scaled),
            np.diag(error),
            "o",
            ms=6,
            mfc="none",
            mew=1.4,
            color=colour[name],
            label=f"{name}, DIAGONAL",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_ylim(1e-13, 3e2)
    axis.set_xlabel("target distance to source cell contour  [source section radii]")
    axis.set_ylabel("shipped-element error  [fraction of column peak]")
    axis.set_title("(a) what the band costs, per pair")
    axis.axhline(0.072, ls="--", lw=1, color="0.3")
    axis.text(
        0.97,
        0.075,
        "7.2% recorded self-flux median  ",
        ha="right",
        transform=axis.get_yaxis_transform(),
        fontsize=8,
        color="0.2",
        va="bottom",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, loc="lower left", ncols=2, markerscale=2.5)

    # (b) the diagonal's own distribution
    axis = axes[0, 1]
    for name, got, want in zip(COMPONENTS, point, exact):
        peak = np.max(np.abs(want), axis=0)
        deviation = np.abs(np.diag(got) - np.diag(want))
        local = deviation / np.abs(np.diag(want))
        column = deviation / peak
        for values, style, tag in (
            (local, "-", "local"),
            (column, "--", "column peak"),
        ):
            finite = np.sort(values[np.isfinite(values) & (values > 0)])
            if finite.size == 0:
                continue
            axis.plot(
                finite,
                np.linspace(0, 1, finite.size),
                style,
                lw=1.7,
                color=colour[name],
                label=f"{name} / {tag}",
            )
    axis.set_xscale("log")
    axis.set_xlim(1e-15, 1e4)
    axis.set_xlabel("shipped-element error on the self term")
    axis.set_ylabel("cumulative fraction of cells")
    axis.set_title("(b) the diagonal, cell by cell")
    for value, label, shift in ((0.072, "7.2%", 0.02), (0.18, "18%", 0.10)):
        axis.axvline(value, ls="--", lw=1, color="0.3")
        axis.text(
            value,
            shift,
            f" {label} recorded",
            fontsize=7.5,
            color="0.2",
            rotation=90,
            va="bottom",
        )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, loc="upper left")
    resolution_inset(axis, sweep)

    # (c) the reference check: both evaluations against the same oracle
    axis = axes[0, 2]
    bands = list(reference[COMPONENTS[0]])
    offset = np.arange(len(bands))
    slot = 0.13
    for order, (key, hatch, tag) in enumerate(
        (("closed_form", None, "closed form"), ("shipped_rule", "//", "16x48 rule"))
    ):
        for index, name in enumerate(COMPONENTS):
            values = [
                reference[name][band][key].get("maximum", np.nan) for band in bands
            ]
            axis.bar(
                offset + (3 * order + index - 2.5) * slot,
                values,
                slot,
                color=colour[name],
                hatch=hatch,
                edgecolor="k",
                lw=0.4,
                label=f"{name}, {tag}",
            )
    axis.set_yscale("log")
    axis.set_xticks(offset)
    axis.set_xticklabels([band.replace(" a ", "\na ") for band in bands], fontsize=8)
    axis.set_ylabel("worst deviation from the 64x96 oracle\n[fraction of column peak]")
    axis.set_title("(c) the reference, checked on this grid's pairs")
    axis.axhline(1e-10, ls="--", lw=1, color="0.3")
    axis.text(
        0.5,
        1.5e-10,
        "1e-10 acceptance gate",
        ha="center",
        transform=axis.get_yaxis_transform(),
        fontsize=8,
        color="0.2",
    )
    axis.grid(alpha=0.25, axis="y")
    axis.legend(fontsize=6.5, ncols=2, loc="upper center")

    # (d) refining the quadrature walks it onto the closed form
    axis = axes[1, 0]
    nodes = [rule[0] * rule[1] for rule in refinement["rules"]]
    for name in COMPONENTS:
        axis.plot(
            nodes, refinement[name], "o-", lw=1.6, ms=5, color=colour[name], label=name
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("boundary quadrature nodes per pair")
    axis.set_ylabel("worst deviation from the closed form\n[fraction of column peak]")
    axis.set_title(
        f"(d) which one is in error near the contour\n"
        f"({refinement['pairs']} pairs inside half a section radius)"
    )
    axis.axvline(NEAR_RULE[0] * NEAR_RULE[1], ls="--", lw=1, color="0.3")
    axis.text(
        NEAR_RULE[0] * NEAR_RULE[1],
        0.03,
        " shipped\n rule",
        fontsize=8,
        color="0.2",
        transform=axis.get_xaxis_transform(),
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    # (e), (f) the integrated effect on a realistic current profile
    for axis, name, unit in ((axes[1, 1], "psi", "Wb"), (axes[1, 2], "B_Z", "T")):
        difference = integrated[name]["difference"]
        swing = integrated[name]["swing"]
        limit = 100 * np.max(np.abs(difference)) / swing
        scatter = axis.scatter(
            grid["centre_r"],
            grid["centre_z"],
            c=100 * difference / swing,
            s=28,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            edgecolors=np.where(regular, "none", "k"),
            linewidths=0.4,
        )
        bar = fig.colorbar(scatter, ax=axis)
        bar.set_label(f"shipped - exact everywhere  [% of the {name} swing]")
        axis.set_aspect("equal")
        axis.set_xlabel("R  [m]")
        axis.set_ylabel("Z  [m]")
        median = 100 * integrated[name]["median_of_swing"]
        axis.set_title(
            f"({'e' if name == 'psi' else 'f'}) {name} at each cell from the whole"
            f" plasma current\n{integrated['plasma_current'] / 1e6:.0f} MA parabolic"
            f" profile, swing {swing:.3g} {unit}, median |error| {median:.2g}%"
        )
        axis.text(
            0.02,
            0.02,
            "black rim: wall-clipped cell",
            transform=axis.transAxes,
            fontsize=7,
            color="0.3",
        )

    fig.suptitle(
        "Banded plasma coupling against the closed-form finite section everywhere, "
        f"{grid['centre_r'].size}-cell hexagonal grid "
        f"({near['population']['regular_cells']} regular, "
        f"{near['population']['clipped_cells']} wall-clipped) at R0 = 6.2 m",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


# --- reporting --------------------------------------------------------------


def _percent(value):
    """Return a value as a percentage string, or a dash when absent."""
    return "--" if value is None or not np.isfinite(value) else f"{100 * value:.3g}%"


def report(
    tables,
    detail,
    near,
    reference,
    refinement,
    integrated,
    moment,
    geometry,
    grid,
    sweep,
):
    """Print every headline number the tables above hold."""
    print(
        f"\ncells {grid['centre_r'].size}"
        f"  regular {near['population']['regular_cells']}"
        f"  wall-clipped {near['population']['clipped_cells']}"
        f"  pairs {near['population']['pairs']}"
    )
    print(
        f"lattice pitch {geometry['pitch']:.6f} m"
        f"  median section radius {moment['median_section_radius']:.6f} m"
        f"  (a/R0)^2 {moment['aspect_squared']:.3e}"
    )

    header = f"{'component':10s}" + "".join(
        f"{label:>26s}" for label in tables["psi"] if label != "scale"
    )
    for scale in ("local", "column_peak"):
        print(f"\nshipped-element relative error, {scale.upper()} normalisation")
        print(header)
        for name in COMPONENTS:
            row = f"{name:10s}"
            for label, entry in tables[name].items():
                if label == "scale":
                    continue
                stats = entry[scale]
                pair = _percent(stats.get("median")) + " / "
                pair += _percent(stats.get("maximum"))
                row += f"{pair:>26s}"
            print(row)

    print("\nthe self term, signed, split by whether the wall clipped the cell")
    for name in COMPONENTS:
        entry = detail[name]
        print(
            f"  {name:4s} exact median {entry['exact_median']:+.4e}"
            f"  point median {entry['point_median']:+.4e}"
            f"  signed median {entry['signed_median']:+.4g}"
            f"  low on {entry['fraction_low']:.1%} of cells"
            f"  sign flips {entry['sign_flips']}"
        )
        for label in ("regular", "wall-clipped"):
            cell = entry[label]
            print(
                f"       {label:13s} n={cell['cells']:4d}"
                f"  local {_percent(cell['local_median'])}"
                f" / {_percent(cell['local_maximum'])}"
                f"  column peak {_percent(cell['column_peak_median'])}"
                f" / {_percent(cell['column_peak_maximum'])}"
            )
        worst = entry["worst_cell"]
        print(
            f"       worst cell: {worst['section']},"
            f" radius {worst['section_radius']:.5f} m,"
            f" contour {worst['contour_over_radius']:.4f} radii,"
            f" exact {worst['exact']:+.4e} point {worst['point']:+.4e}"
        )

    print("\nagainst grid resolution: the self term, then the integrated effect")
    for cells, entry in sweep.items():
        print(
            f"  {entry['cells']:5d} cells,"
            f" a = {entry['median_section_radius'] * 1e3:6.2f} mm"
            f"  |  self term signed: psi {_percent(entry['psi']['signed_median'])}"
            f"  B_Z {_percent(entry['B_Z']['signed_median'])}"
            f"  |  column peak: B_R {_percent(entry['B_R']['column_peak_maximum'])}"
        )
        print(
            "        integrated, median / max of swing:"
            f"  psi {_percent(entry['psi']['integrated_median_of_swing'])}"
            f" / {_percent(entry['psi']['integrated_max_of_swing'])}"
            f"  B_Z {_percent(entry['B_Z']['integrated_median_of_swing'])}"
            f" / {_percent(entry['B_Z']['integrated_max_of_swing'])}"
            f"  energy {_percent(entry['energy_relative'])}"
        )

    print("\nrefining the quadrature near the contour, deviation from the closed form")
    print(f"  {refinement['pairs']} pairs inside half a section radius")
    for name in COMPONENTS:
        ladder = "  ".join(f"{value:.2e}" for value in refinement[name])
        print(f"  {name:4s} " + ladder)
    print(
        "  rules " + "  ".join(f"{rule[0]}x{rule[1]}" for rule in refinement["rules"])
    )

    print("\nthe near band and its population")
    population = near["population"]
    print(
        f"  near band (< {NEAR_LIMIT} radii of contour):"
        f" {population['near_band_fraction']:.4%} of pairs"
    )
    print(
        f"  inside a quarter radius: {population['inside_quarter_radius_pairs']} pairs"
        f" = {population['inside_quarter_radius_fraction']:.4%},"
        f" {population['inside_quarter_radius_on_diagonal']} of them diagonal"
    )
    print(
        "  diagonal contour distance: median"
        f" {population['diagonal_scaled_contour_median']:.4f}"
        f" radii, minimum {population['diagonal_scaled_contour_minimum']:.4f}"
    )
    for name in COMPONENTS:
        for band, stats in near[name].items():
            if stats.get("count", 0) == 0:
                continue
            print(
                f"  {name:4s} {band:28s} n={stats['count']:6d}"
                f"  median {stats['median']:.2e}  worst {stats['maximum']:.2e}"
            )

    print("\nthe reference check, both evaluations against the 64x96 oracle")
    print(
        f"  pairs checked {reference['pairs_checked']}"
        f" (all {reference['pairs_inside_half_radius']} inside half a radius)"
    )
    for name in COMPONENTS:
        for band, entry in reference[name].items():
            closed, shipped = entry["closed_form"], entry["shipped_rule"]
            if closed.get("count", 0) == 0:
                continue
            print(
                f"  {name:4s} {band:26s} n={closed['count']:5d}"
                f"  closed form {closed['median']:.2e} / {closed['maximum']:.2e}"
                f"  16x48 rule {shipped['median']:.2e} / {shipped['maximum']:.2e}"
                "   (median / worst)"
            )

    print("\nthe integrated effect on a realistic current profile")
    print(
        f"  {integrated['plasma_current'] / 1e6:.1f} MA parabolic, peak cell"
        f" {integrated['peak_cell_current'] / 1e3:.1f} kA"
    )
    for name in COMPONENTS:
        entry = integrated[name]
        print(
            f"  {name:4s} swing {entry['swing']:.4g}"
            f"  |difference| median {entry['median_absolute']:.3e}"
            f" max {entry['max_absolute']:.3e}"
            f"  of swing {_percent(entry['median_of_swing'])}"
            f" / {_percent(entry['max_of_swing'])}"
            f"  local {_percent(entry['median_relative'])}"
            f" / {_percent(entry['max_relative'])}"
        )
    energy = integrated["energy"]
    print(
        f"  magnetic energy {energy['point']:.6g} vs {energy['exact']:.6g} J"
        f"  -> {_percent(energy['relative'])}"
    )


def jsonable(value):
    """Return a JSON-serialisable copy, dropping the per-cell arrays."""
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def main(output=None, figure_path=None, cells=CELLS):
    """Measure, report and record the whole comparison."""
    coilset, grid = plasma_grid(cells)
    point = element_coupling(coilset)
    exact = exact_coupling(grid)
    # beyond the element's own section band the shipped path is the bare ring at the
    # section's rms radius -- checked against the ring formula there, so the
    # comparison below is against the matrix a solve builds and not a rescaling of it
    geometry = pair_geometry(grid)
    far = geometry["scaled"] > 20.0
    bare = greens_psi(
        np.repeat(grid["centre_r"][:, None], grid["filament_r"].size, axis=1)[far],
        np.repeat(grid["centre_z"][:, None], grid["filament_r"].size, axis=1)[far],
        np.repeat(grid["filament_r"][None, :], grid["centre_r"].size, axis=0)[far],
        np.repeat(grid["centre_z"][None, :], grid["centre_r"].size, axis=0)[far],
    )
    residual = np.max(np.abs(point[0][far] / bare - 1.0))
    assert residual < 1e-9, f"far field is not the bare ring: {residual:.3e}"

    tables = error_tables(point, exact, geometry)
    detail = diagonal_detail(point, exact, grid, geometry)
    near = near_band_gain(grid, exact, geometry)
    reference = reference_check(grid, exact, geometry)
    refinement = refinement_check(grid, exact, geometry)
    integrated = integrated_effect(point, exact, grid)
    moment = second_moment_scale(grid)
    sweep = resolution_sweep()
    report(
        tables,
        detail,
        near,
        reference,
        refinement,
        integrated,
        moment,
        geometry,
        grid,
        sweep,
    )

    if figure_path is not None:
        written = figure(
            figure_path,
            point,
            exact,
            geometry,
            grid,
            near,
            reference,
            refinement,
            integrated,
            sweep,
        )
        print(f"\nfigure {written}")

    record = dict(
        grid=dict(
            cells=int(grid["centre_r"].size),
            regular=near["population"]["regular_cells"],
            clipped=near["population"]["clipped_cells"],
            wall=WALL["e"],
            pitch=geometry["pitch"],
        ),
        moment=moment,
        error=tables,
        diagonal=detail,
        resolution=sweep,
        near_band=near,
        reference=reference,
        refinement=refinement,
        integrated={
            key: {name: item for name, item in value.items() if name != "difference"}
            if isinstance(value, dict)
            else value
            for key, value in integrated.items()
            if key != "current"
        },
        far_field_residual=float(residual),
    )
    if output is not None:
        pathlib.Path(output).write_text(json.dumps(jsonable(record), indent=2))
        print(f"wrote {output}")
    return record


if __name__ == "__main__":
    main(*sys.argv[1:3])
