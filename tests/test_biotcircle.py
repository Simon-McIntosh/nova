"""What the circular-filament element returns where a filament cannot answer.

The element's own contract, at the level a solve sees it: the geometry it is handed
is the geometry it uses, an all-to-all matrix has a defined diagonal, that diagonal
is the double integral over the source and target sections, and the two bands hand
over to the point filament at seams whose size is measured here rather than assumed.
"""

import numpy as np
import pytest

from nova.biot.biotframe import Target
from nova.biot.circle import Circle
from nova.biot.greens import greens_bz_br, greens_psi, section_centroid
from nova.biot.polygonanalytic import polygon_analytic_greens
from nova.biot.sectionaverage import ORDER, averaged_greens
from nova.frame.coilset import CoilSet


def sections(frame):
    """Return each element's ``(n, 2)`` r-z section polygon."""
    vertices = []
    for poly in np.asarray(frame["poly"]):
        points = np.asarray(poly.points, dtype=np.float64)[:, [0, 2]]
        if len(points) > 1 and np.allclose(points[0], points[-1]):
            points = points[:-1]
        vertices.append(points)
    return vertices


@pytest.fixture(name="coilset")
def fixture_coilset():
    """Return a discretised coil pair: rectangular turns, several radii and heights."""
    coilset = CoilSet(dcoil=0.25)
    coilset.coil.insert(
        3.9431, 7.5641, 0.9590, 0.9841, nturn=248.64, name="PF1", part="PF"
    )
    coilset.coil.insert(1.722, 5.313, 0.719, 2.075, nturn=554, name="CS3U", part="CS")
    return coilset


@pytest.fixture(name="cells")
def fixture_cells():
    """Return a hexagonally tiled plasma grid inside an elliptical wall."""
    coilset = CoilSet(dplasma=-40, tplasma="hex")
    coilset.firstwall.insert({"e": [6.2, 0.5, 3.0, 5.0]}, Ic=15e6)
    return coilset


def test_the_element_uses_the_geometry_it_was_handed(coilset):
    """No offset, blend or floor anywhere between the frame and the kernel.

    The source ring sits at the frame's own root-mean-square radius and the target at
    its own coordinates, bit for bit, on every pair -- the coincident ones included,
    which are the pairs a self-coupling model expressed in the geometry would have to
    move apart by some fraction of the turn's own width. Any offset, blend or floor
    introduced for any reason fails here, because it would have to change one of these
    four arrays.
    """
    frame = coilset.subframe
    biot = Circle(frame, frame, reduce=[False, False])
    assert np.array_equal(biot.rs, np.tile(np.asarray(frame.rms), (len(frame), 1)))
    assert np.array_equal(biot.zs, np.tile(np.asarray(frame.z), (len(frame), 1)))
    assert np.array_equal(biot.z, np.tile(np.asarray(frame.z), (len(frame), 1)).T)
    assert np.array_equal(biot.r, np.tile(np.asarray(frame.x), (len(frame), 1)).T)


@pytest.mark.parametrize("case", ["coilset", "cells"])
def test_an_all_to_all_matrix_has_no_divergent_entry(case, request):
    """Every entry of a source-is-target matrix is finite, the diagonal included.

    A point filament asked for its own coincident value returns the divergence, which
    is the right answer for a filament; the diagonal of an all-to-all operator is
    exactly that configuration on every entry, so the element cannot be allowed to
    return it. The check runs on both populations a real build produces -- rectangular
    coil turns and a hexagonal plasma tiling clipped by a wall -- and on all three
    components, since the two field rows divide by the modulus complement the flux
    does not.
    """
    frame = request.getfixturevalue(case).subframe
    biot = Circle(frame, frame, reduce=[False, False])
    for name in ("Psi", "Br", "Bz", "Aphi"):
        value = np.asarray(getattr(biot, name))
        assert np.all(np.isfinite(value)), name
    # bounded, not merely finite: the self term is the largest entry of its own
    # column and stays within a factor of the mutual terms beside it, where a
    # coincident filament would put a spike there of whatever size the arithmetic
    # happened to stop at
    psi = np.asarray(biot.Psi)
    column_peak = np.max(psi - np.diag(np.diag(psi)), axis=0)
    assert np.all(np.diag(psi) > column_peak)
    assert np.all(np.diag(psi) < 3.0 * column_peak)


def test_the_diagonal_is_the_double_integral_over_its_own_section(coilset):
    """Each diagonal entry against the section average built independently.

    :func:`nova.biot.sectionaverage.averaged_greens` is the readable statement of the
    quantity; the element batches the same nodes into one kernel call per source
    column alongside its point targets, so the two are separate arrangements of one
    definition and this is what stops them drifting apart.
    """
    frame = coilset.subframe
    biot = Circle(frame, frame, reduce=[False, False])
    vertices = sections(frame)
    for index, section in enumerate(vertices):
        want = averaged_greens([section], section)
        scale = (want[0][0], want[0][0] / 1.0, want[0][0])
        for row, name in enumerate(("Psi", "Br", "Bz")):
            value = np.asarray((biot.Psi, biot.Br, biot.Bz)[row])[index, index]
            # B_R passes through zero on a section symmetric about its own centroid,
            # so it is held against the flux scale rather than against itself
            assert value == pytest.approx(
                want[row][0], rel=1e-13, abs=1e-13 * abs(scale[row])
            ), name


def test_the_diagonal_is_not_the_single_integral_at_the_target_point(coilset):
    """The two candidate diagonals are measurably different, and this is which one.

    Both are exact evaluations of a well-posed integral, so neither is "wrong" in
    isolation -- what decides between them is which quantity the operator wants, and
    :func:`tests.test_biotsectionaverage.test_the_self_term_converges_onto_the_published_ring_inductance`
    is the measurement that decides it. Here the gap between them is simply pinned, so
    a silent reversion to a point evaluation cannot pass.
    """
    frame = coilset.subframe
    biot = Circle(frame, frame, reduce=[False, False])
    vertices = sections(frame)
    centre = np.array([section_centroid(section) for section in vertices])
    point = np.array(
        [
            polygon_analytic_greens(centre[index, :1], centre[index, 1:], section)[0][0]
            for index, section in enumerate(vertices)
        ]
    )
    ratio = point / np.diag(np.asarray(biot.Psi))
    assert np.min(ratio) > 1.05  # measured 1.066 lowest, 1.071 highest
    assert np.max(ratio) < 1.10


def test_a_vanishing_section_reproduces_the_point_filament(monkeypatch):
    """The finite-section branch and the filament branch meet in the limit joining them.

    A section driven to nothing is a filament at its own centroid, so the two
    treatments have to agree there -- and they fold the permeability and the
    ``2 pi r`` of the total flux differently on the way, so agreeing is what pins the
    unit and normalisation conventions between them rather than an inspection of the
    two expressions. A factor of ``2 pi`` or of ``mu_0`` misplaced in either fails
    this by ten orders.

    The target is held at a FIXED distance while the section shrinks, which is the
    limit that converges: at a standoff held in section radii instead, the ratio of
    the two is scale-invariant and approaches nothing. Holding the target fixed drives
    the pair out of ``section_band``, where the element returns the filament exactly
    and the comparison becomes vacuous, so the band is opened for the duration --
    which is also what makes the deviation below the SECTION's own second-moment term
    rather than a band artefact.
    """
    monkeypatch.setattr(Circle, "section_band", 1.0e9)
    point_r, point_z = np.array([5.05]), np.array([0.52])
    previous = None
    for width in (2e-2, 1e-2, 5e-3):
        source = CoilSet(dcoil=-1)
        source.coil.insert(5.0, 0.5, width, width, section="rectangle", nturn=1)
        frame = source.subframe
        target = Target({"x": point_r, "z": point_z})
        biot = Circle(frame, target, reduce=[False, False])
        radius = float(np.asarray(frame.rms)[0])
        height = float(np.asarray(frame.z)[0])
        want_psi = greens_psi(point_r, point_z, radius, height)
        want_bz, want_br = greens_bz_br(point_r, point_z, radius, height)
        deviation = max(
            abs(np.asarray(biot.Psi)[0, 0] / want_psi[0] - 1.0),
            abs(np.asarray(biot.Bz)[0, 0] / want_bz[0] - 1.0),
            abs(np.asarray(biot.Br)[0, 0] / want_br[0] - 1.0),
        )
        # the section's second moment against the ring Green's function's curvature,
        # so a quarter of the section halves the deviation four times over
        assert deviation < 2.0 * (width / 0.05) ** 2  # measured 0.29 of the bound
        if previous is not None:
            assert deviation < 0.35 * previous
        previous = deviation


def test_the_averaged_block_is_reciprocal(coilset):
    """Where both directions of a pair are averaged, the matrix is symmetric.

    Green's reciprocity is a property of the DOUBLE integral, not of a point
    evaluation, so the symmetry of the averaged pairs is an independent check that
    both directions are evaluating the same thing. The band is measured in the
    SOURCE section's radii, so it is only symmetric between elements of equal size --
    which is why the check is restricted to pairs whose sections have the same
    bounding radius, rather than asserted over the whole matrix.
    """
    frame = coilset.subframe
    biot = Circle(frame, frame, reduce=[False, False])
    vertices = sections(frame)
    centre = np.array([section_centroid(section) for section in vertices])
    radius = np.array(
        [
            float(np.max(np.hypot(*(section - centre[index]).T)))
            for index, section in enumerate(vertices)
        ]
    )
    separation = np.hypot(
        centre[:, 0][:, None] - centre[:, 0][None, :],
        centre[:, 1][:, None] - centre[:, 1][None, :],
    )
    equal = np.isclose(radius[:, None], radius[None, :], rtol=1e-09)
    both = equal & (separation < Circle.average_band * radius[None, :])
    assert both.sum() > len(frame)  # off-diagonal pairs, not the diagonal alone
    psi = np.asarray(biot.Psi)
    # the exact double integral is symmetric; the rule that evaluates it is not
    # symmetric between two sections, so this is the quadrature's own residual
    assert np.max(np.abs(psi[both] - psi.T[both]) / np.abs(psi[both])) < 1e-06


def test_the_band_seams_are_the_size_the_contract_states():
    """What each band hands over at its own edge, measured on a real coil turn.

    Neither seam closes to zero, and the reason is in the physics rather than in the
    placing: for a FULL ring the finite-section correction is set by the major radius,
    not by the distance to the target, so a bare filament does not converge onto a
    section at any standoff and both deviations flatten onto a floor of order
    ``(a/R0)^2``. The bands are placed where the term they carry has come down onto
    that floor, and these are the numbers that says so.
    """
    major, height, width = 3.9431, 7.5641, 0.2398
    vertices = np.array(
        [
            [major - width / 2, height - width / 2],
            [major + width / 2, height - width / 2],
            [major + width / 2, height + width / 2],
            [major - width / 2, height + width / 2],
        ]
    )
    centre = section_centroid(vertices)
    radius = float(np.max(np.hypot(*(vertices - centre).T)))
    self_flux = averaged_greens([vertices], vertices)[0][0]

    def at(standoff):
        """Return ``(filament, single, double)`` psi at a radial standoff."""
        target = np.array([centre[0] + standoff * radius])
        level = np.array([centre[1]])
        shifted = vertices + np.array([standoff * radius, 0.0])
        return (
            greens_psi(target, level, centre[0], centre[1])[0],
            polygon_analytic_greens(target, level, vertices)[0][0],
            averaged_greens([shifted], vertices)[0][0],
        )

    filament, single, double = at(Circle.section_band)
    assert abs(filament - single) / self_flux < 1e-03  # measured 3.0e-04
    filament, single, double = at(Circle.average_band)
    seam = abs(single - double) / self_flux
    assert seam < 3e-03  # measured 8.6e-04
    # two-sided: the term the band carries is still growing inside it, so the band
    # cannot be narrowed to a cheaper width without giving up more than it hands over
    _, single, double = at(0.5 * Circle.average_band)
    assert abs(single - double) / self_flux > 3.0 * seam  # measured 5.6x
    # and the floor both of them are heading for, twenty radii out
    filament, single, double = at(20.0)
    assert abs(filament - double) / self_flux < 1e-04  # measured 6.6e-05


def test_the_average_band_reaches_the_pairs_whose_sections_can_meet():
    """Two slender sections just inside two radii, at the shipped width and narrower.

    Two sections of equal bounding radius stop being able to TOUCH at exactly two
    radii of separation, and the configuration that makes the width load-bearing is a
    pair of SLENDER sections side by side: their bounding radius is set by the long
    direction while the gap between them is set by the short one, so they nearly
    overlap at a separation a narrower band has already handed to the point target.
    Two undiscretised ITER CS sections are exactly that pair, at 1.94 radii.

    The band is pinned from both sides here. At the shipped width the reduced mutual
    lands on the uniform-current double integral over both sections; narrowed to 1.5
    radii it misses it by four orders more, which is what stops the value being
    lowered for cost. Widening further buys nothing -- the deviation at four radii is
    the same 8.8e-06 H as at two.
    """
    coilset = CoilSet(dcoil=-1)
    coilset.coil.insert(1.722, 5.313, 0.719, 2.075, nturn=554, name="CS3U", part="CS")
    coilset.coil.insert(1.722, 3.188, 0.719, 2.075, nturn=554, name="CS2U", part="CS")
    frame = coilset.subframe
    upper, lower = sections(frame)
    centre = np.array([section_centroid(upper), section_centroid(lower)])
    radius = float(np.max(np.hypot(*(upper - centre[0]).T)))
    separation = float(np.hypot(*(centre[0] - centre[1]))) / radius
    assert 1.5 < separation < Circle.average_band  # measured 1.94 radii

    want = 554.0**2 * averaged_greens([lower], upper)[0][0]

    def mutual():
        """Return the reduced turn-weighted mutual flux between the two sections."""
        biot = Circle(frame, frame, turns=[True, True], reduce=[True, True])
        return float(np.asarray(biot.compute("Psi")[0])[0, 1])

    assert abs(mutual() - want) < 1e-04  # measured 8.8e-06 H
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Circle, "average_band", 1.5)
        assert abs(mutual() - want) > 1e-02  # measured 3.4e-02 H


def test_a_tiling_of_sub_sections_sums_to_the_whole_section_integral(coilset):
    """The discretisation is free once both bands cover the coil, and this is why.

    A pair sum over a tiling IS the area integral split up: summing the double
    integral over every ordered pair of sub-sections of one section reconstructs the
    double integral over the whole of it, exactly -- and only because the current is
    taken UNIFORM over the section, which is what lets the sub-sections tile it with
    no gaps and no weighting. That identity is what makes the reduced inductance of a
    discretised coil independent of ``dcoil``, and it holds only where every pair
    takes the double integral, so the bands are opened here and what is left is the
    quadrature. A subdivided real conductor tiles nothing, and this identity would
    not hold for one.

    It is also the check that ties the element to the readable statement of the
    quantity on something other than the diagonal: 64 sub-elements interacting
    through the shipped bands, reduced, against one call over the undivided section.
    """
    frame = coilset.subframe
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Circle, "section_band", 1.0e9)
        patch.setattr(Circle, "average_band", 1.0e9)
        biot = Circle(frame, frame, turns=[True, True], reduce=[True, True])
        got = np.diag(np.asarray(biot.compute("Psi")[0]))
    whole = [
        (3.9431, 7.5641, 0.9590, 0.9841, 248.64),
        (1.722, 5.313, 0.719, 2.075, 554.0),
    ]
    for index, (major, height, width, depth, nturn) in enumerate(whole):
        vertices = np.array(
            [
                [major - width / 2, height - depth / 2],
                [major + width / 2, height - depth / 2],
                [major + width / 2, height + depth / 2],
                [major - width / 2, height + depth / 2],
            ]
        )
        want = nturn**2 * averaged_greens([vertices], vertices, 2 * ORDER)[0][0]
        # measured 4.5e-07 and 3.7e-06 relative; the sum over compact sub-sections is
        # the more accurate of the two, since the undivided section carries aspect 2.9
        assert got[index] == pytest.approx(want, rel=1e-05, abs=0)


def test_the_flux_and_the_vector_potential_stay_one_quantity(cells):
    """``Psi`` is ``2 pi mu_0 r Aphi`` on the finite-section branch as on the filament.

    The element's flux row is the primitive now and the vector potential is derived
    from it, where the filament expression had it the other way round; a solve reads
    both, so the identity between them has to hold on every pair and not only where
    the filament runs.
    """
    frame = cells.subframe
    biot = Circle(frame, frame, reduce=[False, False])
    want = 2.0 * np.pi * Circle.mu_0 * np.asarray(biot.r) * np.asarray(biot.Aphi)
    assert np.asarray(biot.Psi) == pytest.approx(want, rel=1e-14, abs=0)
