r"""Differential contract of the neighbour-ring mesh the receipts are read on.

:class:`~nova.equilibrium.conservation.FluxLattice` differentiates a flux map
by central differences, which needs a tensor-product raster of one cell size.
:class:`~nova.equilibrium.stencil_mesh.StencilMesh` differentiates it by a
least-squares quadratic on centre-first neighbour rings, which needs nothing
but centroids and rings and therefore also works on the offset, wall-trimmed
hexagonal tiling the package ships. This module pins that the second is a
faithful stand-in for the first.

Three properties carry that claim, and each is measured rather than asserted:

exactness
    A quadratic is in the fitted space, so gradient and :math:`\Delta^\star`
    are recovered on one to round-off. This is what makes the fit a
    derivative operator rather than a smoother.

order
    On a smooth non-quadratic field both converge at second order in the
    pitch, the same order the central differences carry, so a receipt does
    not become resolution-limited by moving to the ring fit.

agreement
    Reading the SAME raster nodes through both meshes returns the same
    Grad-Shafranov residual to four digits and the same force residual to a
    percent, so the two are measuring one quantity two ways.

The fourth property is the one that differs, and it is pinned as a
difference rather than smoothed over. Central differences commute, so
:math:`\nabla \cdot B` and :math:`\nabla \cdot J` — identically zero for a
flux-function representation — cancel term by term and land at round-off. A
least-squares fit does not commute, so on a ring mesh they land at the
truncation floor of the second fit and FALL at second order under refinement.
They remain four to five decades below any physical residual, which is what a
receipt needs, but their floor is a property of the mesh.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.conservation import (
        FluxLattice,
        conservation_ledger,
        delta_star,
        poloidal_field,
    )
    from nova.equilibrium.domain import DomainMasks, PlasmaDomain
    from nova.equilibrium.observation import clipped_support_quadrature
    from nova.equilibrium.source import DomainProfile, ForwardSource
    from nova.equilibrium.stencil_mesh import (
        MomentGeometry,
        PROFILE_DENSITY_POWERS,
        RING_CONDITION_LIMIT,
        StencilMesh,
    )
    from nova.geometry.hexstencil import hex_stencil
    from nova.jax.config import configure_dtypes

#: Machine-scale origin the tilings are laid out around, so the fit is
#: exercised at the major radius a tokamak mesh actually sits at rather than
#: near unity where a badly scaled design matrix would still look healthy.
ORIGIN = (6.2, 0.0)
#: Physical extent every refinement level spans, so halving the pitch refines
#: the mesh instead of shrinking the domain the error is measured over.
EXTENT = 3.2

#: A quadratic lies in the fitted space, so only round-off separates the
#: fitted derivative from the analytic one over a metre-scale patch.
EXACT_TOLERANCE = 1.0e-9
#: Error ratio a second-order operator shows per halving of the pitch. The
#: floor is set below four to leave room for the sup-norm wandering between
#: cells as the mesh changes.
SECOND_ORDER_RATIO = 3.2
#: Agreement between the two meshes reading one raster. The elliptic operator
#: is exact on both for a quadratic flux, so the Grad-Shafranov residual they
#: report agrees to four digits; on a flux carrying every higher derivative
#: they separate by their own truncation, measured at 1.5e-04 of the field.
#: The force residual differs by more because the pressure and diamagnetic
#: gradients it differences are read through the clipped normalised flux.
ELLIPTIC_AGREEMENT = 1.0e-4
TRUNCATION_AGREEMENT = 5.0e-4
FORCE_AGREEMENT = 0.05
#: Largest relative divergence receipt a ring mesh may report. The measured
#: value is 4e-05 at the coarsest raster below and falls at second order.
DIVERGENCE_CEILING = 1.0e-3


def regular_hexagon_vertices(centre: np.ndarray, pitch: float) -> np.ndarray:
    """Return vertices of a hexagonal cell whose opposite sides are ``pitch`` apart."""
    angle = np.arange(6) * np.pi / 3.0 + np.pi / 6.0
    circumradius = pitch / np.sqrt(3.0)
    return centre + circumradius * np.c_[np.cos(angle), np.sin(angle)]


def shared_hexagon_vertices(mesh: StencilMesh, pitch: float):
    """Return one shared-node pool and the cell-to-node gather indices."""
    nodes: list[np.ndarray] = []
    lookup: dict[tuple[float, float], int] = {}
    cell_node = np.empty((mesh.node_count, 6), dtype=np.intp)
    for cell, centre in enumerate(mesh.coordinate):
        for corner, coordinate in enumerate(regular_hexagon_vertices(centre, pitch)):
            key = tuple(np.round(coordinate, decimals=12))
            if key not in lookup:
                lookup[key] = len(nodes)
                nodes.append(coordinate)
            cell_node[cell, corner] = lookup[key]
    return np.asarray(nodes), cell_node


def hex_tiling(shape: tuple[int, int], pitch: float) -> np.ndarray:
    """Return the centroids of a regular hexagonal tiling in C order.

    Rows are offset by half a pitch and separated by ``sqrt(3)/2`` of one, so
    the six cells :func:`~nova.geometry.hexstencil.hex_stencil` names in axial
    index coordinates are the six that physically touch.
    """
    column, row = np.indices(shape)
    radius = ORIGIN[0] + pitch * (column - shape[0] / 2 + 0.5 * row)
    height = ORIGIN[1] + pitch * np.sqrt(3.0) / 2.0 * (row - shape[1] / 2)
    return np.c_[radius.ravel(), height.ravel()]


def hex_mesh(pitch: float) -> StencilMesh:
    """Return a regular hexagonal mesh of one pitch over the fixed extent."""
    count = int(round(EXTENT / pitch)) + 1
    coordinate = hex_tiling((count, count), pitch)
    return StencilMesh(
        coordinate=coordinate,
        stencil=hex_stencil((count, count)),
        area=np.full(len(coordinate), np.sqrt(3.0) / 2.0 * pitch**2),
    )


def smooth_field(coordinate: np.ndarray) -> np.ndarray:
    """Return a manufactured field carrying every second derivative."""
    radius, height = coordinate[:, 0], coordinate[:, 1]
    return np.exp(0.21 * radius) * np.cos(0.9 * height) + 0.13 * radius * height


def smooth_operators(coordinate: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return the analytic gradient and elliptic operator of that field."""
    radius, height = coordinate[:, 0], coordinate[:, 1]
    grow = np.exp(0.21 * radius)
    radial = 0.21 * grow * np.cos(0.9 * height) + 0.13 * height
    vertical = -0.9 * grow * np.sin(0.9 * height) + 0.13 * radius
    elliptic = (0.21**2 - 0.9**2) * grow * np.cos(0.9 * height) - radial / radius
    return radial, vertical, elliptic


def sup_error(mesh: StencilMesh, value, reference) -> float:
    """Return the sup-norm error over the cells that carry a derivative."""
    inside = np.asarray(mesh.interior(1))
    return float(np.max(np.abs(np.asarray(value) - reference)[inside]))


def test_shared_hexagon_node_evaluations_amortise_to_three_per_cell():
    """Two shared vertices plus one centroid are needed asymptotically per cell."""
    mesh = hex_mesh(0.08)
    node_coordinate, _cell_node = shared_hexagon_vertices(mesh, 0.08)
    evaluations_per_cell = 1.0 + len(node_coordinate) / mesh.node_count
    assert 3.0 <= evaluations_per_cell < 3.11


def test_moment_geometry_shared_nodes_reproduce_a_quadratic_flux_map():
    """The fixed shared-node evaluator is exact on the fitted polynomial space."""
    configure_dtypes()
    pitch = 0.16
    mesh = hex_mesh(pitch)
    cells = [regular_hexagon_vertices(centre, pitch) for centre in mesh.coordinate]
    geometry = MomentGeometry.from_cells(mesh, cells)

    def flux(coordinate):
        radius, height = coordinate[..., 0], coordinate[..., 1]
        return (
            1.3 + 0.2 * radius - 0.4 * height + 0.7 * radius**2 + 0.3 * radius * height
        )

    expected = flux(geometry.atomic_mesh.node_coordinates)
    actual = geometry.shared_node_flux(flux(mesh.coordinate))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=8.0e-14)
    assert geometry.second_moment.shape == (mesh.node_count, 3)
    assert len(geometry.polygons) == mesh.node_count


def test_moment_geometry_is_static_across_jitted_flux_updates():
    """One compiled evaluator accepts changing cell flux without rebuilding geometry."""
    configure_dtypes()
    pitch = 0.32
    mesh = hex_mesh(pitch)
    geometry = MomentGeometry.from_cells(
        mesh,
        [regular_hexagon_vertices(centre, pitch) for centre in mesh.coordinate],
    )
    traces = 0

    def evaluate(flux):
        nonlocal traces
        traces += 1
        return geometry.shared_node_flux(flux)

    compiled = jax.jit(evaluate)
    first = compiled(jnp.ones(mesh.node_count))
    second = compiled(jnp.arange(mesh.node_count, dtype=jnp.float64))
    jax.block_until_ready((first, second))

    assert traces == 1
    assert first.shape == (len(geometry.atomic_mesh.node_coordinates),)


def boundary_support_problem(profile=None):
    """Return one mesh whose outer cells have own-node support evaluation."""
    configure_dtypes()
    pitch = 0.32
    mesh = hex_mesh(pitch)
    cells = [regular_hexagon_vertices(centre, pitch) for centre in mesh.coordinate]
    geometry = MomentGeometry.from_cells(mesh, cells)
    atomic = geometry.atomic_mesh
    stencil = mesh.current_moment_stencil(
        support_centre=np.arange(mesh.node_count),
        sampling_node_coordinate=geometry.sample_node_coordinates,
        sampling_cell_node=geometry.cell_sample_nodes[:, :6],
    )
    if profile is None:
        profile = DomainProfile(
            p_prime=lambda psi: -(1.0 + 0.2 * psi),
            ff_prime=lambda psi: jnp.zeros_like(psi),
        )

    def flux(coordinate):
        radius, height = coordinate[..., 0], coordinate[..., 1]
        return (
            0.35
            + 0.04 * radius
            - 0.03 * height
            + 0.01 * radius**2
            + 0.005 * radius * height
        )

    centroid_flux = jnp.asarray(flux(mesh.coordinate))
    atomic_flux = jnp.asarray(flux(atomic.node_coordinates))
    sample_flux = jnp.asarray(flux(geometry.sample_node_coordinates))
    centroid_density = profile.current_density(
        jnp.asarray(mesh.coordinate[:, 0]), centroid_flux
    )
    shared_density = profile.current_density(
        jnp.asarray(atomic.node_coordinates[:, 0]), atomic_flux
    )
    ring = np.setdiff1d(np.arange(mesh.node_count), mesh.centre)
    return {
        "mesh": mesh,
        "geometry": geometry,
        "stencil": stencil,
        "profile": profile,
        "centroid_flux": centroid_flux,
        "atomic_flux": atomic_flux,
        "sample_flux": sample_flux,
        "centroid_density": centroid_density,
        "shared_density": shared_density,
        "flux": flux,
        "ring": ring,
    }


def evaluate_boundary_support(problem, support):
    """Apply the production support callable to one fixed flux field."""
    stencil = problem["stencil"]
    return stencil.support_flux_moments(
        problem["profile"],
        problem["centroid_flux"],
        problem["sample_flux"],
        support,
    )


def test_own_node_field_varies_exactly_across_clipped_quadrature_points():
    """The seven samples recover a quadratic flux and both derivatives in-cell."""
    problem = boundary_support_problem()
    atomic = problem["geometry"].atomic_mesh
    signed = 6.28 - atomic.node_coordinates[:, 0] - 0.08 * atomic.node_coordinates[:, 1]
    support = atomic.traced_clip(jnp.asarray(signed))
    points, _weights = clipped_support_quadrature(
        support, jnp.ones(problem["mesh"].node_count, dtype=bool)
    )
    value, radial, vertical = problem["stencil"].sample_flux_field(
        problem["centroid_flux"], problem["sample_flux"], points
    )
    query = np.asarray(points)
    expected_value = problem["flux"](query)
    expected_radial = 0.04 + 0.02 * query[..., 0] + 0.005 * query[..., 1]
    expected_vertical = -0.03 + 0.005 * query[..., 0]
    np.testing.assert_allclose(value, expected_value, rtol=2.0e-13, atol=2.0e-13)
    np.testing.assert_allclose(radial, expected_radial, rtol=2.0e-13, atol=2.0e-13)
    np.testing.assert_allclose(vertical, expected_vertical, rtol=2.0e-13, atol=2.0e-13)


def adaptive_polygon_integral(polygon, function) -> float:
    """Integrate a scalar over one convex polygon with adaptive triangles."""
    total = 0.0
    anchor = polygon[0]
    for first, second in zip(polygon[1:-1], polygon[2:], strict=True):
        edge = np.column_stack((first - anchor, second - anchor))
        determinant = abs(float(np.linalg.det(edge)))
        value, _error = integrate.dblquad(
            lambda vertical, radial: (
                function(anchor + edge @ np.asarray([radial, vertical])) * determinant
            ),
            0.0,
            1.0,
            0.0,
            lambda radial: 1.0 - radial,
            epsabs=2.0e-13,
            epsrel=2.0e-13,
        )
        total += value
    return total


def test_complete_supports_use_the_own_node_profile_to_oracle_accuracy():
    """Complete supports use the unified profile within its accuracy gates."""
    problem = boundary_support_problem()
    atomic = problem["geometry"].atomic_mesh
    support = atomic.traced_clip(jnp.ones(len(atomic.node_coordinates)))
    attributed = evaluate_boundary_support(problem, support)
    complete = problem["mesh"].centre
    cells = complete[[0, len(complete) // 2, -1]]
    expected = []
    for cell in cells:
        polygon = np.asarray(problem["geometry"].polygons[int(cell)])
        centre = np.asarray(problem["mesh"].coordinate[int(cell)])

        def density(point):
            return float(
                problem["profile"].current_density(point[0], problem["flux"](point))
            )

        expected.append(
            [
                adaptive_polygon_integral(
                    polygon,
                    lambda point, component=component: (
                        density(point)
                        * (
                            1.0
                            if component == 0
                            else point[component - 1] - centre[component - 1]
                        )
                    ),
                )
                for component in range(3)
            ]
        )
    expected = np.asarray(expected).T
    actual = np.asarray(jnp.stack(attributed))[:, cells]
    denominator = np.sum(np.abs(expected[0]))
    assert np.sum(np.abs(actual[0] - expected[0])) / denominator <= 2.5e-4
    assert abs(np.sum(actual[0]) / np.sum(expected[0]) - 1.0) <= 5.0e-5
    np.testing.assert_allclose(actual[1:], expected[1:], rtol=2.0e-10, atol=2.0e-12)
    assert np.all(np.isfinite(np.asarray(attributed.cell_current)[problem["ring"]]))


def test_boundary_profile_density_matches_adaptive_polygon_quadrature():
    """A ring cell's fixed profile polynomial matches adaptive quadrature."""
    problem = boundary_support_problem()
    atomic = problem["geometry"].atomic_mesh
    support = atomic.traced_clip(jnp.ones(len(atomic.node_coordinates)))
    attributed = evaluate_boundary_support(problem, support)
    cell = int(problem["ring"][len(problem["ring"]) // 2])
    polygon = np.asarray(problem["geometry"].polygons[cell])
    centre = np.asarray(problem["mesh"].coordinate[cell])
    scale = np.max(
        np.abs(np.asarray(problem["geometry"].sampling_vertices[cell]) - centre),
        axis=0,
    )

    def local_basis(point):
        local = (point - centre) / scale
        return np.asarray(
            [local[0] ** p * local[1] ** q for p, q in PROFILE_DENSITY_POWERS]
        )

    stencil = problem["stencil"]
    assert stencil.ring_profile_weight.shape[1:] == (
        len(PROFILE_DENSITY_POWERS),
        672,
    )
    assert np.max(stencil.ring_profile_condition) < 1.4e4
    ring_slot = int(np.flatnonzero(stencil.ring_centre == cell)[0])
    pool = np.concatenate(
        [np.asarray(problem["centroid_flux"]), np.asarray(problem["sample_flux"])]
    )
    gathered = pool[stencil.ring_gather_index[ring_slot]]
    flux_coefficient = stencil.ring_flux_weight[ring_slot] @ gathered
    profile_flux = stencil.ring_profile_flux_design[ring_slot] @ flux_coefficient
    profile_density = np.asarray(
        problem["profile"].current_density(
            stencil.ring_profile_point[ring_slot, :, 0], profile_flux
        )
    )
    coefficient = stencil.ring_profile_weight[ring_slot] @ profile_density
    expected = np.asarray(
        [
            adaptive_polygon_integral(
                polygon,
                lambda point, component=component: (
                    (coefficient @ local_basis(point))
                    * (
                        1.0
                        if component == 0
                        else point[component - 1] - centre[component - 1]
                    )
                ),
            )
            for component in range(3)
        ]
    )
    actual = np.asarray(
        [
            attributed.cell_current[cell],
            attributed.radial_moment[cell],
            attributed.vertical_moment[cell],
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-13)


def test_boundary_support_error_converges_with_missing_area():
    """Moment error falls in proportion to the omitted ring-cell area."""
    problem = boundary_support_problem()
    atomic = problem["geometry"].atomic_mesh
    cell = int(problem["ring"][len(problem["ring"]) // 2])
    polygon = np.asarray(problem["geometry"].polygons[cell])
    centre = problem["mesh"].coordinate[cell]
    half_width = 0.5 * np.ptp(polygon, axis=0)

    class EdgeVanishingCubic:
        def current_density(self, radius, height):
            radial = (radius - centre[0]) / half_width[0]
            vertical = (height - centre[1]) / half_width[1]
            return (
                1.0
                - radial
                + 0.2 * vertical
                + 0.1 * radial**2
                - 0.2 * radial * vertical
                - 0.1 * radial**3
            )

    profile = EdgeVanishingCubic()
    centroid_flux = jnp.asarray(problem["mesh"].coordinate[:, 1])
    sample_flux = jnp.asarray(problem["geometry"].sample_node_coordinates[:, 1])
    full_support = atomic.traced_clip(jnp.ones(len(atomic.node_coordinates)))

    def evaluate(support):
        moments = problem["stencil"].support_flux_moments(
            profile,
            centroid_flux,
            sample_flux,
            support,
        )
        return np.asarray(jnp.stack(moments))[:, cell]

    full = evaluate(full_support)
    width = float(np.ptp(polygon[:, 0]))
    maximum = float(np.max(polygon[:, 0]))
    observed_missing = []
    for missing_fraction in (2.8e-4, 2.0e-4, 1.0e-4, 3.0e-5):
        cut = maximum - width * missing_fraction
        support = atomic.traced_clip(cut - atomic.node_coordinates[:, 0])
        value = evaluate(support)
        observed = float(1.0 - support.area[cell] / support.full_area[cell])
        observed_missing.append(observed)
        error = float(np.max(np.abs(value - full) / np.maximum(np.abs(full), 1.0e-30)))
        assert error <= 4.1 * observed
    assert np.all(np.diff(observed_missing) < 0.0)


def test_boundary_support_is_c1_at_full_fill_without_smoothing():
    """Both-sided support JVPs agree at the exact full-cell transition."""
    profile = DomainProfile(
        p_prime=lambda psi: -((1.0 - psi) ** 2),
        ff_prime=lambda psi: jnp.zeros_like(psi),
    )
    problem = boundary_support_problem(profile)
    atomic = problem["geometry"].atomic_mesh
    cell = int(problem["ring"][len(problem["ring"]) // 2])
    centre = problem["mesh"].coordinate[cell]
    half_width = 0.5 * np.ptp(problem["geometry"].polygons[cell], axis=0)
    atomic_u = (atomic.node_coordinates[:, 0] - centre[0]) / half_width[0]
    sample_u = (
        problem["geometry"].sample_node_coordinates[:, 0] - centre[0]
    ) / half_width[0]
    centroid_u = (problem["mesh"].coordinate[:, 0] - centre[0]) / half_width[0]

    def composed(cut):
        support = atomic.traced_clip(cut - jnp.asarray(atomic_u))
        centroid_flux = 1.0 - (cut - jnp.asarray(centroid_u))
        sample_flux = 1.0 - (cut - jnp.asarray(sample_u))
        moments = problem["stencil"].support_flux_moments(
            profile,
            centroid_flux,
            sample_flux,
            support,
        )
        return jnp.stack(moments)[:, cell]

    displacement = 2.0e-7
    left_value, left_derivative = jax.jvp(
        composed, (jnp.asarray(1.0 - displacement),), (jnp.asarray(1.0),)
    )
    right_value, right_derivative = jax.jvp(
        composed, (jnp.asarray(1.0 + displacement),), (jnp.asarray(1.0),)
    )
    assert np.all(np.isfinite(np.asarray([left_value, right_value])))
    np.testing.assert_allclose(
        left_derivative, right_derivative, rtol=2.0e-6, atol=2.0e-9
    )


def test_banked_unified_representation_satisfies_coverage_split_gates():
    """The bank separates representation accuracy from support coverage."""
    artifact = (
        Path(__file__).parents[1]
        / "scripts/ring_attribution/results/ring-attribution-results.json"
    )
    report = json.loads(artifact.read_text())
    gates = report["gates"]
    assert gates["all_cells_own_node_attributed"]
    assert gates["interior_current_weighted_l1"]
    assert gates["ring_current_weighted_l1"]
    assert gates["support_reference_total_current"]
    assert gates["topology_zero_exact"]
    assert gates["support_reference_field_sup"]

    decomposition = report["two_reference_decomposition"]
    representation = decomposition["attributed_vs_support_geometry"]
    coverage = decomposition["support_coverage_vs_analytic"]
    tracked = decomposition["attributed_vs_analytic_tracked"]
    errors = report["priority_ordered_errors"]["net_current"]
    assert errors["current_weighted_interior_l1"] == 4.740850588173696e-05
    assert errors["current_weighted_ring_l1"] == 0.0016004849848483096
    assert representation["signed_total_difference_a"] == -621.3295577503741
    assert report["population"]["topology_zero_lower_leg_supports"] == 17
    assert report["error_components"]["zero_current_leakage_a"] == 0.0
    assert representation["total_current_relative_error"] <= 5.0e-5
    assert representation["all_target_sup_wb"] < 0.826
    assert (
        abs(
            representation["signed_total_difference_a"]
            + coverage["signed_total_difference_a"]
            - tracked["signed_total_difference_a"]
        )
        < 1.0e-6
    )


# --------------------------------------------------------------------------
# the fitted operators
# --------------------------------------------------------------------------
def test_a_hexagonal_ring_resolves_the_full_quadratic():
    """Both second derivatives are recovered, not just their sum.

    Restricted to six equally spaced ring points the quadratic basis spans the
    angular modes up to :math:`\\cos 2\\theta`, and the centre supplies the
    sixth degree of freedom, so a hexagonal neighbourhood determines
    :math:`\\partial^2/\\partial R^2` and :math:`\\partial^2/\\partial Z^2`
    separately. A fit that could only see the Laplacian would fail this on the
    anisotropic term.
    """
    configure_dtypes()
    mesh = hex_mesh(0.2)
    radius, height = mesh.coordinate[:, 0], mesh.coordinate[:, 1]
    field = 3.1 - 0.7 * radius + 1.3 * height + 0.5 * radius**2 - 0.9 * radius * height
    radial, vertical = mesh.gradient(field)
    assert sup_error(mesh, radial, -0.7 + radius - 0.9 * height) < EXACT_TOLERANCE
    assert sup_error(mesh, vertical, 1.3 - 0.9 * radius) < EXACT_TOLERANCE
    elliptic = 1.0 - (-0.7 + radius - 0.9 * height) / radius
    assert sup_error(mesh, mesh.delta_star(field), elliptic) < EXACT_TOLERANCE


def test_the_fitted_operators_converge_at_second_order():
    """Halving the pitch quarters the error of both fitted operators."""
    configure_dtypes()
    error = {}
    for pitch in (0.32, 0.16, 0.08):
        mesh = hex_mesh(pitch)
        field = smooth_field(mesh.coordinate)
        radial, vertical, elliptic = smooth_operators(mesh.coordinate)
        fitted_radial, fitted_vertical = mesh.gradient(field)
        error[pitch] = (
            max(
                sup_error(mesh, fitted_radial, radial),
                sup_error(mesh, fitted_vertical, vertical),
            ),
            sup_error(mesh, mesh.delta_star(field), elliptic),
        )
    for coarse, fine in ((0.32, 0.16), (0.16, 0.08)):
        for index, name in enumerate(("gradient", "elliptic")):
            ratio = error[coarse][index] / error[fine][index]
            assert ratio > SECOND_ORDER_RATIO, (name, coarse, ratio)


def test_the_ring_fit_holds_its_conditioning_at_machine_scale():
    """A displaced centroid does not spoil the normalised fit.

    The ring is centred and scaled before the design matrix is formed, so the
    conditioning follows the SHAPE of the neighbourhood and not the major
    radius it sits at. A quarter-pitch displacement is far more than first-wall
    clipping produces and still leaves a wide margin.
    """
    configure_dtypes()
    mesh = hex_mesh(0.16)
    assert mesh.ring_condition.max() < 6.0
    generator = np.random.default_rng(7)
    displaced = StencilMesh(
        coordinate=mesh.coordinate
        + 0.25 * 0.16 * generator.uniform(-1.0, 1.0, mesh.coordinate.shape),
        stencil=mesh.stencil,
        area=mesh.area,
    )
    assert displaced.ring_condition.max() < 0.1 * RING_CONDITION_LIMIT


def test_a_ring_that_cannot_determine_a_quadratic_is_rejected():
    """A collinear neighbourhood raises rather than returning a least-norm fit."""
    configure_dtypes()
    mesh = hex_mesh(0.16)
    collinear = mesh.coordinate.copy()
    collinear[mesh.stencil[0]] = np.c_[
        ORIGIN[0] + 0.1 * np.arange(mesh.stencil.shape[1]),
        np.full(mesh.stencil.shape[1], ORIGIN[1]),
    ]
    with pytest.raises(ValueError, match="span both coordinate axes"):
        StencilMesh(coordinate=collinear, stencil=mesh.stencil, area=mesh.area)


def test_a_cell_may_centre_at_most_one_ring():
    """A repeated centre would scatter two derivatives onto one cell."""
    configure_dtypes()
    mesh = hex_mesh(0.16)
    repeated = np.r_[mesh.stencil, mesh.stencil[:1]]
    with pytest.raises(ValueError, match="centre at most one ring"):
        StencilMesh(coordinate=mesh.coordinate, stencil=repeated, area=mesh.area)


def test_cells_without_a_ring_carry_no_derivative():
    """The hull of the tessellation is reported as zero and never checked.

    A cell the mesh cannot differentiate must not reach a receipt carrying a
    one-sided or invented value, which is the unstructured counterpart of
    trimming a lattice border.
    """
    configure_dtypes()
    mesh = hex_mesh(0.16)
    field = smooth_field(mesh.coordinate)
    radial, _ = mesh.gradient(field)
    outside = np.ones(mesh.node_count, dtype=bool)
    outside[mesh.centre] = False
    assert outside.any()
    assert np.all(np.asarray(radial)[outside] == 0.0)
    assert not np.any(np.asarray(mesh.interior(1))[outside])
    # each erosion step drops one ring, so the checked set shrinks strictly
    counts = [int(np.sum(np.asarray(mesh.interior(margin)))) for margin in (1, 2, 3)]
    assert counts[0] > counts[1] > counts[2] > 0


# --------------------------------------------------------------------------
# the same nodes read through both meshes
# --------------------------------------------------------------------------
def raster_problem(nodes: int, *, quadratic: bool = True):
    """Return one flux map and source labelled on a raster, read two ways.

    The stencil mesh is given the lattice's own nodes and the hexagonal rings
    of its index space, so the two meshes carry identical geometry and differ
    only in how a derivative is formed on it. The quadratic flux is in the
    fitted space and in the central-difference kernel alike, so it separates
    exactness from truncation; the exponential one carries every higher
    derivative and measures the truncation itself.
    """
    lattice = FluxLattice(
        np.linspace(0.55, 1.45, nodes), np.linspace(-0.9, 0.5, int(1.6 * nodes))
    )
    mesh = StencilMesh(
        coordinate=lattice.coordinate,
        stencil=hex_stencil(lattice.shape),
        area=lattice.cell_area,
    )
    radius = lattice.node_radius
    height = lattice.coordinate[:, 1]
    if quadratic:
        flux = 0.7 * (radius - 1.0) ** 2 + 0.4 * height**2 + 0.03 * radius * height
    else:
        flux = 0.7 * np.exp(0.9 * radius) * np.cos(1.4 * height) + 0.2 * height**2
    source = ForwardSource(
        core=DomainProfile(
            p_prime=lambda psi_norm: 1.7e4 * (1.0 - psi_norm),
            ff_prime=lambda psi_norm: 0.4 * (1.0 - psi_norm**2),
        ),
        boundary_pressure=1.0e3,
        boundary_field_function=2.0,
    )
    label = np.full(lattice.node_count, PlasmaDomain.COMMON_SOL, dtype=np.int32)
    label[((radius - 1.0) / 0.25) ** 2 + (height / 0.4) ** 2 < 1.0] = PlasmaDomain.CORE
    masks = DomainMasks(
        label=jnp.asarray(label),
        psi_norm=jnp.asarray(
            np.clip((flux - flux.min()) / (flux.max() - flux.min()), 0.0, 1.0)
        ),
    )
    return lattice, mesh, jnp.asarray(flux), source, masks


def relative_spread(checked, left, right) -> float:
    """Return the sup difference of two fields against the sup of the first."""
    left, right = np.asarray(left), np.asarray(right)
    return float(np.max(np.abs(left - right)[checked])) / float(
        np.max(np.abs(left)[checked])
    )


@pytest.mark.parametrize(
    ("quadratic", "tolerance"),
    [(True, EXACT_TOLERANCE), (False, TRUNCATION_AGREEMENT)],
)
def test_the_two_meshes_read_one_flux_map_alike(quadratic, tolerance):
    """Elliptic operator and poloidal field agree on the shared nodes.

    On a quadratic both operators are exact, so only round-off separates them;
    on a field carrying every higher derivative they separate by their own
    truncation, which is what the looser tolerance measures.
    """
    configure_dtypes()
    lattice, mesh, flux, _, _ = raster_problem(45, quadratic=quadratic)
    checked = np.asarray(lattice.interior()) & np.asarray(mesh.interior())
    spread = [
        relative_spread(checked, delta_star(lattice, flux), delta_star(mesh, flux))
    ]
    raster_field = poloidal_field(lattice, flux)
    ring_field = poloidal_field(mesh, flux)
    spread += [
        relative_spread(checked, left, right)
        for left, right in zip(raster_field, ring_field)
    ]
    assert max(spread) < tolerance, spread


def test_the_conservation_receipts_agree_between_the_meshes():
    """The physical residuals do not depend on which mesh formed them."""
    configure_dtypes()
    lattice, mesh, flux, source, masks = raster_problem(45)
    span = jnp.asarray(1.0)
    raster = conservation_ledger(lattice, flux, source, masks, span)
    ring = conservation_ledger(mesh, flux, source, masks, span)
    elliptic = float(ring.relative_grad_shafranov) / float(
        raster.relative_grad_shafranov
    )
    assert abs(elliptic - 1.0) < ELLIPTIC_AGREEMENT
    assert (
        abs(float(ring.relative_force) / float(raster.relative_force) - 1.0)
        < FORCE_AGREEMENT
    )
    # the ring erosion trims a different shell than the four-neighbour one, so
    # the checked sets differ in size while both stay well inside the border
    assert 0.9 < float(ring.checked_cells) / float(raster.checked_cells) < 1.0


def test_the_divergence_floor_is_truncation_and_falls_with_the_mesh():
    """The identically-zero receipts converge instead of sitting at round-off.

    Central differences commute, so the raster cancels these term by term and
    reports round-off. The ring fit does not commute, so it reports the
    truncation of the second fit — which must therefore FALL under refinement,
    and must stay far below any physical residual for the receipt to keep its
    meaning.
    """
    configure_dtypes()
    floor = {}
    for nodes in (45, 91):
        lattice, mesh, flux, source, masks = raster_problem(nodes, quadratic=False)
        span = jnp.asarray(1.0)
        raster = conservation_ledger(lattice, flux, source, masks, span)
        ring = conservation_ledger(mesh, flux, source, masks, span)
        assert float(raster.relative_divergence_b) < 1.0e-12
        assert float(raster.relative_divergence_j) < 1.0e-12
        floor[nodes] = (
            float(ring.relative_divergence_b),
            float(ring.relative_divergence_j),
        )
        assert max(floor[nodes]) < DIVERGENCE_CEILING
        assert max(floor[nodes]) < 1.0e-2 * float(ring.relative_grad_shafranov)
    # the step halves between the two rasters; the magnetic divergence follows
    # the flux map and falls at second order, while the current divergence is
    # limited by the clipped normalised flux the field function is read on
    assert floor[45][0] / floor[91][0] > 3.0, floor
    assert floor[45][1] / floor[91][1] > 1.5, floor
