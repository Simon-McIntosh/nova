"""Contract for the polygon-section thick-filament coupling element.

The element couples a toroidal conductor of arbitrary polygonal cross-section.
Two independent oracles pin it: for a rectangular section it must reproduce
:class:`nova.biot.cylinder.Cylinder` (the closed-form finite-area kernel), and
far from the section it must reproduce the point-filament loop. Between the two
it must stay finite and smooth *through* the conductor, which is exactly where
the point kernel is log-singular and wrong — the reason a plasma cell wants a
thick-filament kernel at all.
"""

import numpy as np
import pytest

from nova.biot.greens import greens_bz_br, greens_psi
from nova.biot.polysection import PolySection
from nova.frame.coilset import CoilSet


def rectangle(r0=1.0, z0=0.0, width=0.06, height=0.04):
    """Return the vertices of a rectangular section, counter-clockwise."""
    return np.array(
        [
            [r0 - width / 2, z0 - height / 2],
            [r0 + width / 2, z0 - height / 2],
            [r0 + width / 2, z0 + height / 2],
            [r0 - width / 2, z0 + height / 2],
        ]
    )


def hexagon(r0=1.0, z0=0.0, radius=0.03):
    """Return the vertices of a regular hexagon section, flat-top."""
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)])


# --- the two oracles --------------------------------------------------------


def test_a_rectangular_section_reproduces_the_closed_form_kernel():
    """Against the exact rectangle kernel, marching through the conductor.

    Targets that land exactly on a section edge are excluded: there the boundary
    integral is evaluated at its own singularity and the field normal to a
    current sheet is genuinely discontinuous, so no finite value is the right
    one. That case is pinned separately below.
    """
    from nova.biot.greens import cylinder_greens

    width, height = 0.06, 0.04
    vertices = rectangle(width=width, height=height)
    target_r = np.linspace(0.93, 1.07, 29)
    target_z = np.full(target_r.size, 0.005)
    off_edge = ~np.isclose(np.abs(target_r - 1.0), width / 2, atol=1e-12)

    reference = cylinder_greens(target_r, target_z, 1.0, 0.0, width, height)
    computed = PolySection.section_greens(target_r, target_z, vertices)
    for got, expected in zip(computed, reference):
        scale = np.max(np.abs(expected))
        np.testing.assert_allclose(
            got[off_edge], expected[off_edge], rtol=1e-6, atol=1e-8 * scale
        )


def test_a_target_on_a_section_edge_stays_finite():
    """On the current sheet itself the field is bounded, if not exact.

    The flux is still accurate; the field component normal to the edge carries
    the sheet's discontinuity, so it is held to a loose bound rather than to the
    closed-form value.
    """
    from nova.biot.greens import cylinder_greens

    width, height = 0.06, 0.04
    vertices = rectangle(width=width, height=height)
    target_r = np.array([1.0 - width / 2, 1.0 + width / 2])
    target_z = np.full(2, 0.005)

    psi_ref, br_ref, bz_ref = cylinder_greens(
        target_r, target_z, 1.0, 0.0, width, height
    )
    psi, br, bz = PolySection.section_greens(target_r, target_z, vertices)
    for component in (psi, br, bz):
        assert np.all(np.isfinite(component))
    np.testing.assert_allclose(psi, psi_ref, rtol=1e-6)
    np.testing.assert_allclose(br, br_ref, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(bz, bz_ref, rtol=5e-3, atol=1e-9)


def test_the_far_field_reproduces_the_point_filament():
    """Beyond a few section sizes the thick filament is a point loop."""
    vertices = hexagon(radius=0.03)
    angle = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    target_r = 1.0 + 0.5 * np.cos(angle)
    target_z = 0.5 * np.sin(angle)

    psi, br, bz = PolySection.section_greens(target_r, target_z, vertices)
    point_psi = greens_psi(target_r, target_z, 1.0, 0.0)
    point_bz, point_br = greens_bz_br(target_r, target_z, 1.0, 0.0)
    np.testing.assert_allclose(psi, point_psi, rtol=2e-3)
    np.testing.assert_allclose(br, point_br, rtol=5e-3, atol=1e-9)
    np.testing.assert_allclose(bz, point_bz, rtol=5e-3, atol=1e-9)


def test_the_flux_stays_finite_and_smooth_through_the_conductor():
    """The point kernel diverges at the source; the thick filament does not."""
    vertices = hexagon(radius=0.03)
    target_r = np.linspace(0.985, 1.015, 31)
    target_z = np.zeros(target_r.size)

    psi, _br, _bz = PolySection.section_greens(target_r, target_z, vertices)
    assert np.all(np.isfinite(psi))
    # smooth: no interior spike, so the curvature stays bounded and the peak is
    # interior rather than at a sampled singularity
    curvature = np.abs(np.diff(psi, 2))
    assert np.max(curvature) < 0.05 * np.max(np.abs(psi))
    singular = greens_psi(np.array([1.0]), np.array([0.0]), 1.0, 0.0)
    assert np.max(psi) < float(singular[0])


# --- the near/far blend -----------------------------------------------------


def test_the_blend_is_continuous_across_the_standoff_band():
    """Flux does not step where the element switches kernels.

    Both kernels are evaluated at the SAME radius, on the band edge, so the
    comparison isolates the switch rather than the radial variation of the flux.
    A finite band is a scoped-study setting (the shipped default is exact
    everywhere), so one is configured explicitly here.
    """
    from nova.biot.polygon import polygon_greens

    vertices = hexagon(radius=0.03)
    standoff = 3.0
    edge = np.array([1.0 + standoff * PolySection.section_radius(vertices)])
    height = np.zeros(1)
    with PolySection.configured(standoff=standoff):
        exact = polygon_greens(edge, height, vertices)[0]
        point = greens_psi(edge, height, 1.0, 0.0)
    assert abs(float(exact[0]) - float(point[0])) < 1e-3 * abs(float(point[0]))


def test_the_default_band_is_unbounded_and_exact_everywhere():
    """The shipped default routes every pair through the exact kernel.

    A finite standoff is a scoped-study setting: configuring one excludes far
    targets from the near band and blends them to the point form, which agrees
    closely far out but is not the exact path.
    """
    vertices = hexagon(radius=0.03)
    far_r = np.array([1.9])
    far_z = np.array([0.8])
    assert PolySection.standoff is None
    assert PolySection.near_band(far_r, far_z, vertices).all()
    exact = PolySection.section_greens(far_r, far_z, vertices)[0]
    with PolySection.configured(standoff=3.0):
        assert not PolySection.near_band(far_r, far_z, vertices).any()
        blended = PolySection.section_greens(far_r, far_z, vertices)[0]
    # far out the two agree closely, but the exact path is not the point form
    np.testing.assert_allclose(exact, blended, rtol=1e-3)
    assert float(exact[0]) != float(blended[0])


def test_the_configuration_is_restored_after_use():
    """A scoped configuration never leaks into the next solve."""
    before = (PolySection.standoff, PolySection.quadrature)
    with PolySection.configured(standoff=None, quadrature=(4, 12)):
        assert PolySection.standoff is None
        assert PolySection.quadrature == (4, 12)
    assert (PolySection.standoff, PolySection.quadrature) == before


def test_the_quadrature_override_reaches_the_kernel():
    """The override changes the result, so it is genuinely being applied."""
    from nova.biot.greens import cylinder_greens

    width, height = 0.06, 0.04
    vertices = rectangle(width=width, height=height)
    target_r = np.linspace(0.955, 1.045, 11)
    target_z = np.full(target_r.size, 0.005)
    reference = cylinder_greens(target_r, target_z, 1.0, 0.0, width, height)[2]

    default = PolySection.section_greens(target_r, target_z, vertices)[2]
    with PolySection.configured(quadrature=(2, 6)):
        coarse = PolySection.section_greens(target_r, target_z, vertices)[2]
    scale = np.max(np.abs(reference))
    assert np.max(np.abs(default - reference)) / scale < 1e-6
    assert np.max(np.abs(coarse - reference)) / scale > 1e-4


def test_a_finite_band_is_a_small_fraction_of_a_grid():
    """A configured blend is what keeps the exact kernel affordable on a grid."""
    vertices = hexagon(radius=0.03)
    radius, height = np.meshgrid(np.linspace(0.3, 1.7, 45), np.linspace(-1.1, 1.1, 45))
    with PolySection.configured(standoff=3.0):
        near = PolySection.near_band(radius.ravel(), height.ravel(), vertices)
    assert near.mean() < 0.05


# --- the coilset wiring -----------------------------------------------------


def test_a_hexagonal_plasma_cell_keeps_point_coupling_for_now():
    """Plasma cells stay on the point-filament ring pending the operator review.

    The polygon-section element exists (and is pinned above) but is not yet
    the plasma default: the near/far standoff needed to make it affordable
    demands a principled cutoff and a parallel/batchable operator-assembly
    design, which is under review. Until then the interaction matrix keeps
    the point kernel everywhere rather than a blend with an ad-hoc band.
    """
    coilset = CoilSet(dplasma=-40)
    coilset.firstwall.insert({"e": [1.0, 0, 0.3, 0.4]}, Ic=1e6)
    segment = set(np.asarray(coilset.subframe.segment).tolist())
    assert segment == {"circle"}


def test_the_plasma_grid_defaults_to_hexagonal_cells():
    """The default plasma mesh is hexagonal without asking for it."""
    coilset = CoilSet(dplasma=-40)
    coilset.firstwall.insert({"e": [1.0, 0, 0.3, 0.4]}, Ic=1e6)
    section = set(np.asarray(coilset.subframe.section).tolist())
    # interior cells are hexagons; cells clipped by the wall stay polygons
    assert "hexagon" in section
    assert section <= {"hexagon", "polygon"}


@pytest.mark.parametrize("orientation", [1, -1])
def test_the_section_orientation_does_not_change_the_field(orientation):
    """Vertices wound either way describe the same conductor."""
    vertices = hexagon()[::orientation]
    target_r = np.array([1.4, 1.5, 0.7])
    target_z = np.array([0.1, 0.2, -0.3])
    psi, br, bz = PolySection.section_greens(target_r, target_z, vertices)
    point_psi = greens_psi(target_r, target_z, 1.0, 0.0)
    point_bz, point_br = greens_bz_br(target_r, target_z, 1.0, 0.0)
    np.testing.assert_allclose(psi, point_psi, rtol=5e-3)
    np.testing.assert_allclose(br, point_br, rtol=1e-2, atol=1e-9)
    np.testing.assert_allclose(bz, point_bz, rtol=1e-2, atol=1e-9)
