"""Contract for the exact polygon-section coupling element.

The element couples a toroidal conductor of arbitrary polygonal cross-section.
For a rectangular section it reproduces :class:`nova.biot.cylinder.Cylinder`;
for every authored polygon it integrates the finite section without a reachable
point-filament substitution.
"""

import warnings

import numpy as np
import pytest
from dataclasses import FrozenInstanceError
from shapely.geometry import MultiPolygon, Polygon

from nova.biot.biotframe import Source, Target
from nova.biot.greens import greens_bz_br, greens_psi, second_moments
from nova.biot.plasmagrid import PlasmaGrid
from nova.biot.plasmawall import PlasmaWall
from nova.biot.polysection import PolySection, PolySectionPolicy, TiledPolySection
from nova.biot.polygonanalytic import (
    polygon_analytic_field_moments,
    polygon_analytic_flux_moments,
    polygon_analytic_greens,
)
from nova.biot.sectionaverage import section_triangles
from nova.biot.solve import Solve
from nova.biot.target import TargetQuadraturePolicy
from nova.equilibrium.stencil_mesh import MomentGeometry, StencilMesh
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


def test_a_hexagon_authored_machine_couples_the_exact_section_moments():
    """The coupling consumes the machine's exact regular-hexagon inertia."""
    coilset = CoilSet(dplasma=-40)
    coilset.firstwall.insert({"e": [1.0, 0, 0.3, 0.4]}, Ic=1e6)
    authored = np.asarray(coilset.subframe["section"], dtype=str)
    cell = int(np.flatnonzero(authored == "hexagon")[0])
    element = PolySection(coilset.subframe, coilset.subframe, reduce=[False, False])
    components = element._section_components[cell]
    assert len(components) == 1
    vertices, weight = components[0]
    assert weight == 1.0

    edge = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
    np.testing.assert_allclose(edge, edge[0], rtol=2e-14)
    pitch = np.sqrt(3.0) * float(edge.mean())
    radial, vertical, cross = second_moments(vertices)
    expected = 5.0 / 72.0 * pitch**2
    np.testing.assert_allclose([radial, vertical], expected, rtol=2e-14)
    assert abs(cross) <= 8.0 * np.spacing(expected)


def hex_mesh_source():
    """Return three neighbouring polygon-section plasma cells."""
    centres = np.array([[2.94, -0.04], [3.06, -0.04], [3.0, 0.065]])
    sections = [hexagon(r, z, radius=0.065) for r, z in centres]
    return Source(
        {
            "x": centres[:, 0],
            "y": np.zeros(len(centres)),
            "z": centres[:, 1],
            "segment": ["polysection"] * len(centres),
            "poly": [Polygon(vertices) for vertices in sections],
            "frame": [f"Coil{index}" for index in range(len(centres))],
            "nturn": np.ones(len(centres)),
            "plasma": np.ones(len(centres), dtype=bool),
            "link": [""] * len(centres),
        }
    )


MOMENT_ATTRIBUTES = (
    "Psi",
    "PsiR",
    "PsiZ",
    "Br",
    "BrR",
    "BrZ",
    "Bz",
    "BzR",
    "BzZ",
)


def test_hex_mesh_moment_attributes_match_direct_polygon_evaluation():
    """Every assembled row is the direct fixed-centroid polygon block."""
    target_r = np.array([2.72, 3.0, 3.31, 4.2])
    target_z = np.array([-0.16, 0.01, 0.22, -0.5])
    solve = Solve(
        hex_mesh_source(),
        Target({"x": target_r, "z": target_z}),
        attrs=list(MOMENT_ATTRIBUTES),
        turns=[False, False],
        reduce=[False, False],
    )

    expected = {attribute: [] for attribute in MOMENT_ATTRIBUTES}
    for polygon in hex_mesh_source()["poly"]:
        vertices = np.asarray(polygon.exterior.coords, dtype=float)[:-1, :2]
        flux = polygon_analytic_flux_moments(target_r, target_z, vertices)
        radial, vertical = polygon_analytic_field_moments(target_r, target_z, vertices)
        for names, rows in zip(
            (("Psi", "PsiR", "PsiZ"), ("Br", "BrR", "BrZ"), ("Bz", "BzR", "BzZ")),
            (flux, radial, vertical),
            strict=True,
        ):
            for name, row in zip(names, rows, strict=True):
                expected[name].append(row)

    for attribute in MOMENT_ATTRIBUTES:
        np.testing.assert_array_equal(
            solve.data[attribute], np.column_stack(expected[attribute])
        )


def test_adding_moment_attributes_keeps_uniform_assembly_bitwise_identical():
    """Requesting companion rows cannot reassociate established uniform rows."""
    target = {"x": [2.72, 3.0, 3.31, 4.2], "z": [-0.16, 0.01, 0.22, -0.5]}
    uniform = Solve(
        hex_mesh_source(),
        Target(target),
        attrs=["Psi", "Br", "Bz"],
        turns=[False, False],
        reduce=[False, False],
    )
    expanded = Solve(
        hex_mesh_source(),
        Target(target),
        attrs=list(MOMENT_ATTRIBUTES),
        turns=[False, False],
        reduce=[False, False],
    )
    for attribute in ("Psi", "Br", "Bz"):
        np.testing.assert_array_equal(expanded.data[attribute], uniform.data[attribute])


def test_plasma_assemblies_request_kernel_axis_companions():
    """Grid and wall defaults expose the rows their solve contractions consume."""
    assert PlasmaGrid.__dataclass_fields__["attrs"].default_factory() == [
        "Br",
        "BrR",
        "BrZ",
        "Bz",
        "BzR",
        "BzZ",
        "Psi",
        "PsiR",
        "PsiZ",
    ]
    assert PlasmaWall.__dataclass_fields__["attrs"].default_factory() == [
        "Psi",
        "PsiR",
        "PsiZ",
    ]


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


def test_policy_exposes_only_exact_section_kernels():
    """Approximate production routing cannot be restored through policy data."""
    default = PolySectionPolicy()
    reference = PolySectionPolicy(exact_kernel="quadrature", quadrature=(4, 12))
    assert default.exact_kernel == "closed_form"
    assert default.key != reference.key
    with pytest.raises(FrozenInstanceError):
        reference.quadrature = (2, 4)
    for retired in ("standoff", "banded", "filament"):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            PolySectionPolicy.resolve({"arrangement": retired})


def test_policy_numeric_domains_have_one_canonical_cache_key():
    """Invalid scalar spellings cannot create ambiguous exact-kernel keys."""
    with pytest.raises(ValueError, match="positive integers"):
        PolySectionPolicy(exact_kernel="quadrature", quadrature=(True, 4))
    with pytest.raises(ValueError, match="positive integer"):
        TargetQuadraturePolicy(order=True)


def test_accelerator_policy_requires_the_exact_axisymmetric_ring_lane():
    """Backend and geometry eligibility form one canonical executable identity."""
    with pytest.raises(ValueError, match="requires 'axisymmetric_ring'"):
        PolySectionPolicy(backend="jax")
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
    reference = PolySectionPolicy(exact_kernel="quadrature")
    coilset = CoilSet()
    with pytest.raises(ValueError, match="fixed by its CoilSet constructor"):
        coilset.coil.insert(
            3.0,
            0.0,
            0.2,
            0.2,
            nturn=1,
            polysection_policy=reference,
        )
    coilset.firstwall.polysection_policy = reference.key
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
    material = [
        Polygon(
            [(0.92, -0.08), (1.08, -0.08), (1.11, 0.01), (1.02, 0.10), (0.91, 0.05)]
        ),
        MultiPolygon(
            [
                Polygon(
                    [(1.16, -0.05), (1.30, -0.05), (1.30, 0.08), (1.16, 0.08)],
                    holes=[[(1.20, -0.01), (1.24, -0.01), (1.24, 0.03), (1.20, 0.03)]],
                ),
                Polygon([(1.33, 0.00), (1.38, 0.00), (1.38, 0.05), (1.33, 0.05)]),
            ]
        ),
    ]
    target_data = {
        "x": [0.0, 0.0, 1.35, 1.10],
        "y": [0.0, 0.0, 0.20, -0.30],
        "z": [0.0, 0.31, -0.12, 0.24],
    }
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
                "x": [1.0, 1.24],
                "y": [0.0, 0.0],
                "z": [0.0, 0.02],
                "segment": ["polysection", "polysection"],
                "polysection_policy": [policy.key, policy.key],
                "poly": material,
                "frame": ["head", "dependent"],
                "nturn": [2.0, 3.0],
                "plasma": [False, False],
                "link": ["", "head"],
                "factor": [1.0, -0.25],
            },
            index=["head", "dependent"],
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        host = PolySection(
            source(host_policy),
            Target(target_data),
            turns=False,
            reduce=False,
            policy=host_policy,
        )
        tiled = TiledPolySection(
            source(tiled_policy),
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
        np.testing.assert_array_equal(tiled.Psi[:2, :], 0.0)
        np.testing.assert_array_equal(tiled.Br[:2, :], 0.0)
        np.testing.assert_array_equal(tiled.Aphi[:2, :], 0.0)
        assert np.all(np.isfinite(tiled.Bz[:2]))
        np.testing.assert_allclose(
            tiled.Bz[:2],
            [
                [6.255885856967584e-07, 5.040164419323532e-07],
                [5.446909684052329e-07, 4.6482555846759426e-07],
            ],
            rtol=0.0,
            atol=1.2e-16,
        )

        def solve(policy):
            return Solve(
                source(policy),
                Target(target_data),
                attrs=["Psi", "Br", "Bz"],
                turns=[True, False],
                reduce=[True, False],
            )

        host_solve = solve(host_policy)
        tiled_solve = solve(tiled_policy)
        for attribute in ("Psi", "Br", "Bz"):
            expected = getattr(host_solve.data, attribute)
            got = getattr(tiled_solve.data, attribute)
            assert np.all(np.isfinite(got))
            np.testing.assert_allclose(got, expected, rtol=2e-12, atol=1e-15)
    np.testing.assert_array_equal(tiled_solve.data.Psi[:2], 0.0)
    np.testing.assert_array_equal(tiled_solve.data.Br[:2], 0.0)
    np.testing.assert_allclose(
        tiled_solve.data.Bz[:2, 0],
        [8.731648399442519e-07, 7.4076276795977e-07],
        rtol=0.0,
        atol=1.2e-16,
    )


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


def test_preclip_samples_come_from_the_verified_authored_generator():
    """Sampling geometry remains a fixed hexagon when material support is clipped."""
    coilset = CoilSet(dplasma=-40)
    coilset.firstwall.insert({"e": [1.0, 0, 0.317, 0.413]}, Ic=1e6)
    plasma = np.asarray(coilset.subframe["plasma"], dtype=bool)
    target = Target(
        {
            "x": np.asarray(coilset.subframe["x"])[plasma],
            "z": np.asarray(coilset.subframe["z"])[plasma],
            "poly": np.asarray(coilset.subframe["poly"], dtype=object)[plasma],
        }
    )

    vertices = coilset.plasmagrid._preclip_sampling_vertices(target)
    centres = np.c_[np.asarray(target.x), np.asarray(target.z)]
    assert vertices.shape == (len(centres), 6, 2)
    assert coilset.plasmagrid.sampling_identity_deviation <= (
        coilset.plasmagrid.sampling_identity_bound
    )
    coordinate_scale = float(np.max(np.abs(centres)))
    coordinate_roundoff = 2.0 * abs(float(np.spacing(coordinate_scale)))
    np.testing.assert_allclose(
        vertices.mean(axis=1), centres, atol=coordinate_roundoff, rtol=0
    )
    radii = np.linalg.norm(vertices - centres[:, None, :], axis=2)
    np.testing.assert_allclose(radii, radii[0, 0], atol=2e-16, rtol=0)

    material_counts = [
        len(np.asarray(polygon.poly.exterior.coords)) - 1 for polygon in target.poly
    ]
    assert any(count != 6 for count in material_counts)
    assert not np.any(np.asarray(target["section"], dtype=str) == "hexagon")
    authored = np.asarray(coilset.plasmagrid.aloc["plasma", "section"], dtype=str)
    assert np.any(authored == "hexagon")

    material = tuple(
        np.asarray(polygon.poly.exterior.coords, dtype=float)[:-1, :2]
        for polygon in target.poly
    )
    sample_tolerance = coilset.plasmagrid._support_roundoff_bound(target)
    coordinates, cell_nodes = coilset.plasmagrid._index_sampling_vertices(
        vertices, sample_tolerance
    )
    full = np.flatnonzero(authored == "hexagon")
    central = full[np.argmin(np.linalg.norm(centres[full] - centres.mean(0), axis=1))]
    distance = np.linalg.norm(centres - centres[central], axis=1)
    neighbours = np.argsort(distance)[1:7]
    mesh = StencilMesh(
        coordinate=centres,
        stencil=np.asarray([[central, *neighbours]], dtype=np.intp),
        area=np.ones(len(centres)),
    )
    geometry = MomentGeometry.from_cells(
        mesh,
        material,
        sampling_vertices=vertices,
    )
    np.testing.assert_array_equal(coordinates, geometry.sample_node_coordinates)
    np.testing.assert_array_equal(cell_nodes, geometry.cell_sample_nodes)


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


def plasma_cell(r0=6.2, z0=0.0, radius=0.06):
    """Return a hexagonal plasma cell at a tokamak major radius."""
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)])


def test_the_shipped_default_is_the_closed_form_everywhere():
    """Every target-source pair goes through the exact closed-form reduction."""
    from nova.biot.polygonanalytic import polygon_analytic_greens

    policy = PolySectionPolicy()
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


def test_the_closed_form_is_reached_through_the_production_configuration():
    """The default takes every exact evaluation through the reduction.

    Bit-identity to the closed form, and a difference from the quadrature: the
    same physics through a different evaluation. Which of the two is nearer the
    truth, and by how much where, is measured in
    :mod:`tests.test_biotbandedcoupling`.
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
    for name, one, other in zip(("psi", "br", "bz"), closed, quadrature, strict=True):
        scale = float(np.max(np.abs(other)))
        assert np.max(np.abs(one - other)) / scale <= 1e-3, name
