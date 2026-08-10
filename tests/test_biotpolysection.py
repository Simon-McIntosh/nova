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
from dataclasses import FrozenInstanceError
from shapely.geometry import Polygon

from nova.biot.biotframe import Source, Target
from nova.biot.greens import greens_bz_br, greens_psi
from nova.biot.polysection import PolySection, PolySectionPolicy, TiledPolySection
from nova.biot.polygonanalytic import polygon_analytic_greens
from nova.biot.sectionaverage import section_triangles
from nova.biot.solve import Solve
from nova.biot.target import TargetQuadraturePolicy
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
    policy = PolySectionPolicy(
        arrangement="standoff",
        exact_kernel="quadrature",
        standoff=standoff,
    )
    exact = polygon_greens(edge, height, vertices)[0]
    point = greens_psi(edge, height, 1.0, 0.0)
    assert PolySection.near_band(edge, height, vertices, policy).tolist() == [False]
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
    default_policy = PolySectionPolicy()
    assert default_policy.arrangement == "exact"
    assert PolySection.near_band(far_r, far_z, vertices, default_policy).all()
    exact = PolySection.section_greens(far_r, far_z, vertices)[0]
    study_policy = PolySectionPolicy(arrangement="standoff", standoff=3.0)
    assert not PolySection.near_band(far_r, far_z, vertices, study_policy).any()
    blended = PolySection.section_greens(far_r, far_z, vertices, study_policy)[0]
    # far out the two agree closely, but the exact path is not the point form
    np.testing.assert_allclose(exact, blended, rtol=1e-3)
    assert float(exact[0]) != float(blended[0])


def test_route_policies_are_immutable_and_cache_distinct():
    """One study route cannot mutate the default or collide with its cache key."""
    default = PolySectionPolicy()
    study = PolySectionPolicy(
        arrangement="standoff",
        exact_kernel="quadrature",
        standoff=3.0,
        quadrature=(4, 12),
    )
    assert default == PolySectionPolicy()
    assert default.key != study.key
    with pytest.raises(FrozenInstanceError):
        study.standoff = 5.0


def test_filament_policy_rejects_unused_exact_settings():
    """One point-filament value has one cache identity, with no inert quadrature."""
    with pytest.raises(ValueError, match="does not accept an exact kernel"):
        PolySectionPolicy(
            arrangement="filament",
            exact_kernel="quadrature",
            quadrature=(-1, 0),
        )


def test_policy_numeric_domains_have_one_canonical_cache_key():
    """Equivalent scalar spellings cannot split one route into multiple keys."""
    policies = [
        PolySectionPolicy(arrangement="standoff", standoff=value)
        for value in (3, 3.0, np.int64(3))
    ]
    assert {policy.key for policy in policies} == {policies[0].key}
    assert all(type(policy.standoff) is float for policy in policies)
    with pytest.raises(ValueError, match="finite distance"):
        PolySectionPolicy(arrangement="standoff", standoff=True)
    with pytest.raises(ValueError, match="positive integers"):
        PolySectionPolicy(exact_kernel="quadrature", quadrature=(True, 4))
    with pytest.raises(ValueError, match="positive integer"):
        TargetQuadraturePolicy(order=True)


def test_accelerator_policy_requires_the_exact_axisymmetric_ring_lane():
    """Backend and geometry eligibility form one canonical executable identity."""
    with pytest.raises(ValueError, match="requires 'axisymmetric_ring'"):
        PolySectionPolicy(backend="jax")
    with pytest.raises(ValueError, match="only exact routing"):
        PolySectionPolicy(
            arrangement="banded",
            backend="jax",
            device_eligibility="axisymmetric_ring",
        )
    with pytest.raises(ValueError, match="compiled quadrature"):
        PolySectionPolicy(backend="jax", device_eligibility="axisymmetric_ring")
    policy = PolySectionPolicy(
        exact_kernel="quadrature",
        quadrature=(2, 4),
        backend="jax",
        device_eligibility="axisymmetric_ring",
    )
    assert PolySectionPolicy.resolve(policy.key) == policy


def test_coilset_factories_reject_routes_outside_the_cache_identity():
    """Per-insert and post-construction mutations cannot bypass the stored route."""
    banded = PolySectionPolicy(arrangement="banded")
    coilset = CoilSet()
    with pytest.raises(ValueError, match="fixed by its CoilSet constructor"):
        coilset.coil.insert(
            3.0,
            0.0,
            0.2,
            0.2,
            nturn=1,
            polysection_policy=banded,
        )
    coilset.firstwall.polysection_policy = banded.key
    with pytest.raises(ValueError, match="fixed by its CoilSet constructor"):
        coilset.firstwall.insert({"circle": [3.0, 0.0, 0.5]})


def test_the_quadrature_override_reaches_the_kernel():
    """The override changes the result, so it is genuinely being applied.

    It is a knob on the boundary-quadrature route alone, which is no longer the
    default, so the route is configured explicitly here. The closed form has no
    ``(n_panels, n_nodes)`` to override -- its residual node count is fixed by its
    own acceptance gate -- and it ignores this setting, which is asserted too.
    """
    from nova.biot.greens import cylinder_greens

    width, height = 0.06, 0.04
    vertices = rectangle(width=width, height=height)
    target_r = np.linspace(0.955, 1.045, 11)
    target_z = np.full(target_r.size, 0.005)
    reference = cylinder_greens(target_r, target_z, 1.0, 0.0, width, height)[2]

    default_policy = PolySectionPolicy(exact_kernel="quadrature")
    coarse_policy = PolySectionPolicy(exact_kernel="quadrature", quadrature=(2, 6))
    default = PolySection.section_greens(target_r, target_z, vertices, default_policy)[
        2
    ]
    coarse = PolySection.section_greens(target_r, target_z, vertices, coarse_policy)[2]
    scale = np.max(np.abs(reference))
    assert np.max(np.abs(default - reference)) / scale < 1e-6
    assert np.max(np.abs(coarse - reference)) / scale > 1e-4

    # A meaningless quadrature cannot be smuggled into a closed-form identity.
    closed = PolySection.section_greens(target_r, target_z, vertices)[2]
    with pytest.raises(ValueError, match="does not accept"):
        PolySectionPolicy(quadrature=(2, 6))
    assert np.max(np.abs(closed - reference)) / scale < 1e-6


def test_a_finite_band_is_a_small_fraction_of_a_grid():
    """A configured blend is what keeps the exact kernel affordable on a grid."""
    vertices = hexagon(radius=0.03)
    radius, height = np.meshgrid(np.linspace(0.3, 1.7, 45), np.linspace(-1.1, 1.1, 45))
    policy = PolySectionPolicy(arrangement="standoff", standoff=3.0)
    near = PolySection.near_band(radius.ravel(), height.ravel(), vertices, policy)
    assert near.mean() < 0.05


def test_standoff_dispatch_evaluates_exact_and_filament_masks_separately(monkeypatch):
    """Neither kernel sees targets belonging exclusively to the other route."""
    vertices = hexagon(radius=0.03)
    target_r = np.array([1.0, 1.8])
    target_z = np.array([0.0, 0.4])
    calls = []

    def exact(near_r, near_z, section, policy):
        calls.append((near_r.copy(), near_z.copy(), section.copy(), policy))
        assert near_r.tolist() == [1.0]
        return tuple(np.ones_like(near_r) for _ in range(3))

    monkeypatch.setattr(PolySection, "exact_greens", staticmethod(exact))
    policy = PolySectionPolicy(arrangement="standoff", standoff=2.0)
    psi, br, bz = PolySection.section_greens(target_r, target_z, vertices, policy)
    assert len(calls) == 1
    assert (psi[0], br[0], bz[0]) == (1.0, 1.0, 1.0)
    assert np.all(np.isfinite([psi[1], br[1], bz[1]]))


def test_vector_potential_masks_the_magnetic_axis_division():
    """A target on the axis has finite zero Aphi without hiding other rows."""
    coilset = CoilSet(dcoil=-1)
    coilset.coil.insert(
        {"hexagon": [1.0, 0.0, 0.08, 0.08]},
        nturn=1.0,
        segment="polysection",
    )
    element = PolySection(
        Source(coilset.subframe),
        Target({"x": [0.0, 1.4], "z": [0.0, 0.2]}),
        turns=False,
        reduce=False,
    )
    assert element.Aphi[0, 0] == 0.0
    assert np.all(np.isfinite(element.Aphi))
    assert element.Aphi[1, 0] != 0.0


@pytest.mark.slow
def test_tiled_quadrature_partitions_axis_rows_onto_the_finite_host_limit():
    """The traced gradient never divides an exact-axis target by its radius."""
    coilset = CoilSet(dcoil=-1)
    coilset.coil.insert(
        {"hexagon": [1.0, 0.0, 0.08, 0.08]},
        nturn=1.0,
        segment="polysection",
    )
    source = Source(coilset.subframe)
    target_data = {"x": [0.0, 1.4], "z": [0.2, -0.1]}
    host_policy = PolySectionPolicy(exact_kernel="quadrature", quadrature=(2, 4))
    tiled_policy = PolySectionPolicy(
        exact_kernel="quadrature",
        quadrature=(2, 4),
        backend="jax",
        device_eligibility="axisymmetric_ring",
    )
    host = PolySection(
        source,
        Target(target_data),
        turns=False,
        reduce=False,
        policy=host_policy,
    )
    tiled = TiledPolySection(
        Source(coilset.subframe),
        Target(target_data),
        turns=False,
        reduce=False,
        policy=tiled_policy,
    )
    for attribute in ("Psi", "Br", "Bz", "Aphi"):
        got = getattr(tiled, attribute)
        expected = getattr(host, attribute)
        assert np.all(np.isfinite(got))
        np.testing.assert_allclose(got, expected, rtol=2e-12, atol=1e-15)
    assert tiled.Aphi[0, 0] == 0.0


def test_hollow_source_integrates_only_positive_material_triangles():
    """A source void carries no current and is not filled by its exterior."""
    material = Polygon(
        [(2.7, -0.3), (3.3, -0.3), (3.3, 0.3), (2.7, 0.3)],
        holes=[[(2.9, -0.1), (3.1, -0.1), (3.1, 0.1), (2.9, 0.1)]],
    )
    coilset = CoilSet(dcoil=0)
    coilset.coil.insert(
        material,
        nturn=1.0,
        name="Hollow",
        ifttt=False,
        segment="polysection",
    )
    target_r = np.array([2.82, 3.0, 3.4])
    target_z = np.array([0.0, 0.0, 0.2])
    element = PolySection(
        Source(coilset.subframe),
        Target({"x": target_r, "z": target_z}),
        turns=False,
        reduce=False,
    )

    source_material = coilset.subframe.poly[0].poly
    assert len(source_material.interiors) == 1
    triangles, area = section_triangles(source_material)
    reference = [np.zeros(len(target_r)) for _ in range(3)]
    for vertices, weight in zip(triangles, area / area.sum()):
        for row, value in enumerate(
            polygon_analytic_greens(target_r, target_z, vertices)
        ):
            reference[row] += weight * value
    for got, expected in zip((element.Psi, element.Br, element.Bz), reference):
        np.testing.assert_allclose(got[:, 0], expected, rtol=3e-13, atol=1e-15)


@pytest.mark.slow
def test_tiled_product_adapter_matches_host_and_preserves_hollow_material():
    """The explicit JAX route packs material pieces and restores authored columns."""
    material = Polygon(
        [(2.7, -0.3), (3.3, -0.3), (3.3, 0.3), (2.7, 0.3)],
        holes=[[(2.9, -0.1), (3.1, -0.1), (3.1, 0.1), (2.9, 0.1)]],
    )
    coilset = CoilSet(dcoil=0)
    coilset.coil.insert(
        material,
        nturn=1.0,
        name="Hollow",
        ifttt=False,
        segment="polysection",
    )
    source = Source(coilset.subframe)
    target = Target({"x": [2.82, 3.0, 3.4], "z": [0.0, 0.0, 0.2]})
    host_policy = PolySectionPolicy(exact_kernel="quadrature", quadrature=(2, 4))
    tiled_policy = PolySectionPolicy(
        exact_kernel="quadrature",
        quadrature=(2, 4),
        backend="jax",
        device_eligibility="axisymmetric_ring",
    )
    host = PolySection(source, target, turns=False, reduce=False, policy=host_policy)
    tiled = TiledPolySection(
        Source(coilset.subframe),
        Target({"x": [2.82, 3.0, 3.4], "z": [0.0, 0.0, 0.2]}),
        turns=False,
        reduce=False,
        policy=tiled_policy,
    )
    assert tiled.coordinate_axes.shape == (0, 3, 3)
    for got, expected in zip(
        (tiled.Psi, tiled.Br, tiled.Bz), (host.Psi, host.Br, host.Bz)
    ):
        np.testing.assert_allclose(got, expected, rtol=2e-12, atol=1e-15)


@pytest.mark.slow
def test_solve_tiled_ring_matches_host_after_turns_and_electrical_links():
    """Adapter dispatch precedes one global authored-source electrical reduction."""
    host_policy = PolySectionPolicy(exact_kernel="quadrature", quadrature=(2, 4))
    tiled_policy = PolySectionPolicy(
        exact_kernel="quadrature",
        quadrature=(2, 4),
        backend="jax",
        device_eligibility="axisymmetric_ring",
    )

    def source(policy):
        return Source(
            {
                "x": [1.0, 1.12],
                "y": [0.0, 0.0],
                "z": [0.0, 0.03],
                "segment": ["polysection", "polysection"],
                "polysection_policy": [policy.key, policy.key],
                "poly": [
                    Polygon([(0.96, -0.04), (1.04, -0.04), (1.04, 0.04), (0.96, 0.04)]),
                    Polygon([(1.08, -0.01), (1.16, -0.01), (1.16, 0.07), (1.08, 0.07)]),
                ],
                "frame": ["head", "dependent"],
                "nturn": [2.0, 3.0],
                "plasma": [False, False],
                "link": ["", "head"],
                "factor": [1.0, -0.5],
            },
            index=["head", "dependent"],
        )

    def solve(policy):
        return Solve(
            source(policy),
            Target({"x": [1.3, 1.5], "z": [0.1, -0.2]}),
            attrs=["Psi"],
            turns=[True, False],
            reduce=[True, False],
        )

    host = solve(host_policy)
    tiled = solve(tiled_policy)
    assert tiled.source_batches[0].policy == tiled_policy
    assert tiled.data.source.values.tolist() == ["head"]
    np.testing.assert_allclose(tiled.data.Psi, host.data.Psi, rtol=2e-12, atol=1e-15)


# --- the coilset wiring -----------------------------------------------------


def test_a_hexagonal_plasma_cell_is_coupled_as_the_finite_section_it_is():
    """Plasma cells couple through their own polygon, not through a point ring.

    A point filament is log-singular at its own location, and an all-to-all
    plasma matrix puts a target inside its own source cell on every diagonal
    entry -- the one configuration the point model cannot represent and the
    finite section handles as an ordinary interior point. The default is the
    section because the exact treatment stopped being expensive: the closed form
    costs a few hundred microseconds a pair against the boundary quadrature's
    858, the build is paid once per geometry and cached, and the banded scheme
    keeps the far field at filament cost.
    """
    coilset = CoilSet(dplasma=-40)
    coilset.firstwall.insert({"e": [1.0, 0, 0.3, 0.4]}, Ic=1e6)
    segment = set(np.asarray(coilset.subframe.segment).tolist())
    assert segment == {"polysection"}


def test_a_rectangular_plasma_mesh_routes_only_complete_cells_to_cylinder():
    """The rectangle shortcut does not claim wall-clipped polygon cells.

    Complete axis-aligned cells keep their closed-form finite-area kernel. Cells
    clipped against the curved wall carry their actual polygon through the exact
    polygon-section lane.
    """
    coilset = CoilSet(dplasma=-40)
    coilset.firstwall.insert({"e": [1.0, 0, 0.3, 0.4]}, Ic=1e6, turn="rectangle")
    segment = set(np.asarray(coilset.subframe.segment).tolist())
    assert segment == {"cylinder", "polysection"}


def test_every_real_plasma_cell_evaluates_in_closed_form_including_the_clipped_ones():
    """A real grid's cells are not the tidy sections the acceptance gate uses.

    Measured on a 179-cell grid: the wall-clipped cells carry three to twelve
    vertices, and clipping leaves edges as short as 1.6e-10 m beside edges of
    0.12 m -- coincident corners, in effect, a ratio of nine orders. The closed
    form has to stay finite through all of them, because the shipped default now
    routes every one of these cells through it and a single non-finite entry
    poisons the whole operator.

    The disagreement with the boundary quadrature is measured the other way
    round, and it is the quadrature that is wrong: on this grid the worst
    off-diagonal pair is a neighbouring cell centre sitting 0.001 contour radii
    outside its neighbour's boundary, where the shipped ``(16, 48)`` rule is
    2.9e-03 out on B_Z. Refining it to 1024 panels closes that to 2.1e-12 OF THE
    CLOSED FORM's value, so the closed form is what the quadrature converges to.
    """
    from nova.biot.polygonanalytic import polygon_analytic_greens

    coilset = CoilSet(dplasma=-60)
    coilset.firstwall.insert({"e": [6.2, 0, 2.0, 3.0]}, Ic=1e6)
    subframe = coilset.subframe
    sections = []
    for poly in np.asarray(subframe["poly"]):
        points = np.asarray(poly.points, dtype=float)[:, [0, 2]]
        if len(points) > 1 and np.allclose(points[0], points[-1]):
            points = points[:-1]
        sections.append(points)
    assert len(sections) > 40
    assert max(len(points) for points in sections) > 6  # clipped cells are present
    edge = np.concatenate(
        [np.hypot(*(np.roll(points, -1, axis=0) - points).T) for points in sections]
    )
    assert edge.min() < 1e-9 * edge.max()  # clipping leaves near-coincident corners

    target_r = np.asarray(subframe.x, dtype=float)
    target_z = np.asarray(subframe.z, dtype=float)
    for points in sections:
        for component in polygon_analytic_greens(target_r, target_z, points):
            assert np.all(np.isfinite(component))


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


# --- the banded scheme, opt-in ----------------------------------------------


def plasma_cell(r0=6.2, z0=0.0, radius=0.06):
    """Return a hexagonal plasma cell at a tokamak major radius."""
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)])


def test_the_shipped_default_is_the_closed_form_everywhere_and_not_banded():
    """Neither binning reduction is on by default, and the exact kernel is closed.

    No pair is approximated: there is no standoff band handing a far pair to a
    point filament, and no three-band split handing it to a reduced rule. Every
    pair goes through the closed-form reduction, which the measured cost makes
    affordable -- 171 µs/pair against the 858 the boundary quadrature spent for
    one to two orders less accuracy.
    """
    from nova.biot.polygonanalytic import polygon_analytic_greens

    policy = PolySectionPolicy()
    assert policy.arrangement == "exact"
    assert policy.standoff is None
    assert policy.exact_kernel == "closed_form"
    assert (policy.backend, policy.precision, policy.device_eligibility) == (
        "numpy",
        "float64",
        "host",
    )
    vertices = plasma_cell()
    target_r = np.array([6.2, 7.4, 8.9])
    target_z = np.array([0.5, -0.9, 1.4])
    for got, expected in zip(
        PolySection.section_greens(target_r, target_z, vertices),
        polygon_analytic_greens(target_r, target_z, vertices),
    ):
        np.testing.assert_array_equal(got, expected)


def test_the_banded_scheme_is_reached_through_an_instance_policy():
    """A banded instance routes every pair through the band dispatch, and only it."""
    from nova.biot.bandedcoupling import banded_greens

    vertices = plasma_cell()
    angle = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    target_r = 6.2 + np.geomspace(0.1, 2.0, 12) * np.cos(angle)
    target_z = np.geomspace(0.1, 2.0, 12) * np.sin(angle)

    exact = PolySection.section_greens(target_r, target_z, vertices)
    policy = PolySectionPolicy(arrangement="banded")
    banded = PolySection.section_greens(target_r, target_z, vertices, policy)
    for got, expected in zip(
        banded,
        banded_greens(target_r, target_z, vertices, closed_form=True),
    ):
        np.testing.assert_array_equal(got, expected)
    # it is a different path, not a no-op rename of the exact one
    assert any(not np.array_equal(one, other) for one, other in zip(banded, exact))


def test_the_banded_scheme_holds_every_component_against_the_exact_lane():
    """Through the element's own entry point, the two lanes agree to the bound."""
    vertices = plasma_cell()
    angle = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    reach = np.geomspace(0.07, 2.4, 16)[:, None]
    target_r = (6.2 + reach * np.cos(angle)).ravel()
    target_z = (reach * np.sin(angle)).ravel()

    exact = PolySection.section_greens(target_r, target_z, vertices)
    banded = PolySection.section_greens(
        target_r,
        target_z,
        vertices,
        PolySectionPolicy(arrangement="banded"),
    )
    for got, expected in zip(banded, exact):
        scale = np.max(np.abs(expected))
        assert np.max(np.abs(got - expected)) / scale <= 1e-6


def test_banded_and_exact_instances_coexist_without_shared_state():
    """An opt-in batch cannot alter the policy of any following batch."""
    exact = PolySectionPolicy()
    banded = PolySectionPolicy(arrangement="banded")
    assert exact.arrangement == "exact"
    assert banded.arrangement == "banded"
    assert PolySectionPolicy() == exact


# --- the closed form as the exact kernel, opt-in ------------------------------


def test_the_closed_form_is_reached_through_the_scoped_configuration():
    """Turning it on takes every exact evaluation through the reduction instead.

    Bit-identity to the closed form, and a difference from the quadrature: the
    same physics through a different evaluation. Which of the two is nearer the
    truth, and by how much where, is measured in
    :mod:`tests.test_biotbandedcoupling` -- here the contract is only that the
    configuration selects it.
    """
    from nova.biot.polygon import polygon_greens
    from nova.biot.polygonanalytic import polygon_analytic_greens

    vertices = plasma_cell()
    angle = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    target_r = 6.2 + np.geomspace(0.02, 2.0, 12) * np.cos(angle)
    target_z = np.geomspace(0.02, 2.0, 12) * np.sin(angle)

    closed = PolySection.section_greens(
        target_r, target_z, vertices, PolySectionPolicy()
    )
    for got, expected in zip(
        closed, polygon_analytic_greens(target_r, target_z, vertices)
    ):
        np.testing.assert_array_equal(got, expected)
    # a different evaluation of the same physics, not a rename of the quadrature
    quadrature = polygon_greens(target_r, target_z, vertices)
    assert any(not np.array_equal(one, other) for one, other in zip(closed, quadrature))
    for name, one, other in zip(("psi", "br", "bz"), closed, quadrature):
        scale = float(np.max(np.abs(other)))
        assert np.max(np.abs(one - other)) / scale <= 1e-3, name


def test_the_closed_form_serves_the_near_band_of_the_banded_scheme():
    """The two knobs compose: one bins the pairs, the other evaluates the exact ones.

    Where it lands is the near band, which is bit-identical to whichever exact
    kernel is configured -- so this is the only place in the banded scheme where
    the accuracy gain can appear, and it appears there in full.
    """
    from nova.biot.bandedcoupling import band, banded_greens
    from nova.biot.polygonanalytic import polygon_analytic_greens

    vertices = plasma_cell()
    angle = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    reach = np.geomspace(0.07, 1.5, 10)[:, None]
    target_r = (6.2 + reach * np.cos(angle)).ravel()
    target_z = (reach * np.sin(angle)).ravel()

    scheme = PolySection.section_greens(
        target_r,
        target_z,
        vertices,
        PolySectionPolicy(arrangement="banded"),
    )
    for got, expected in zip(
        scheme, banded_greens(target_r, target_z, vertices, closed_form=True)
    ):
        np.testing.assert_array_equal(got, expected)

    near = band(target_r, target_z, vertices) == 0
    assert near.any()
    reference = polygon_analytic_greens(target_r[near], target_z[near], vertices)
    for got, expected in zip(scheme, reference):
        np.testing.assert_array_equal(got[near], expected)


def test_the_closed_form_also_serves_a_standoff_band():
    """It replaces the exact kernel wherever the exact kernel is used, not only far.

    The standoff arrangement keeps a point filament outside its band, so the two
    routes must differ inside the band and agree bit for bit outside it -- which is
    what says the choice is about the exact treatment alone.
    """
    from nova.biot.polygonanalytic import polygon_analytic_greens

    vertices = plasma_cell()
    # the band is 3 section radii = 0.18 m about the centroid: 0.04 and 0.11 m out
    # are inside it, the third target is far outside
    target_r = np.array([6.24, 6.31, 8.9])
    target_z = np.array([0.01, 0.0, 1.4])
    quadrature_policy = PolySectionPolicy(
        arrangement="standoff",
        exact_kernel="quadrature",
        standoff=3.0,
    )
    closed_policy = PolySectionPolicy(arrangement="standoff", standoff=3.0)
    inside = PolySection.near_band(target_r, target_z, vertices, quadrature_policy)
    quadrature = PolySection.section_greens(
        target_r, target_z, vertices, quadrature_policy
    )
    closed = PolySection.section_greens(target_r, target_z, vertices, closed_policy)
    assert inside.tolist() == [True, True, False]
    reference = polygon_analytic_greens(target_r[inside], target_z[inside], vertices)
    for got, expected in zip(closed, reference):
        np.testing.assert_array_equal(got[inside], expected)
    for got, expected in zip(closed, quadrature):
        np.testing.assert_array_equal(got[~inside], expected[~inside])


def test_the_point_far_field_sits_at_the_section_area_centroid():
    """A standoff blend places its filament at the area centroid, not the vertex mean.

    The two coincide only for a section whose corners pair up. On one where they do
    not, the vertex mean carries a first-moment error, which is a dipole the far
    field has no way to absorb.
    """
    from nova.biot.greens import section_centroid

    vertices = np.array([[6.15, -0.03], [6.24, -0.03], [6.26, 0.02], [6.18, 0.04]])
    centre = section_centroid(vertices)
    assert not np.allclose(centre, vertices.mean(axis=0), atol=1e-6)
    far_r = np.array([7.6])
    far_z = np.array([1.1])
    policy = PolySectionPolicy(arrangement="standoff", standoff=3.0)
    psi = PolySection.section_greens(far_r, far_z, vertices, policy)[0]
    np.testing.assert_array_equal(psi, greens_psi(far_r, far_z, centre[0], centre[1]))
