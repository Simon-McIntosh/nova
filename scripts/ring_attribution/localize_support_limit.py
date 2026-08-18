"""Localise discontinuity in the support-dependent density projection."""

from pathlib import Path
import runpy

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.separatrix_clip import (
    POLYNOMIAL_POWERS,
    padded_polynomial_current_moments,
)
from nova.equilibrium.stencil_mesh import _quadratic_flux_design
from nova.jax.config import configure_dtypes


configure_dtypes()


ROOT = Path(__file__).resolve().parents[2]
TEST = runpy.run_path(str(ROOT / "tests/test_equilibrium_stencil_mesh.py"))
boundary_support_problem = TEST["boundary_support_problem"]
evaluate_boundary_support = TEST["evaluate_boundary_support"]

BARYCENTRIC = jnp.asarray(
    [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.059715871789770, 0.470142064105115, 0.470142064105115],
        [0.470142064105115, 0.059715871789770, 0.470142064105115],
        [0.470142064105115, 0.470142064105115, 0.059715871789770],
        [0.797426985353087, 0.101286507323456, 0.101286507323456],
        [0.101286507323456, 0.797426985353087, 0.101286507323456],
        [0.101286507323456, 0.101286507323456, 0.797426985353087],
    ],
    dtype=jnp.float64,
)
RULE_WEIGHT = jnp.asarray(
    [
        0.225,
        0.132394152788506,
        0.132394152788506,
        0.132394152788506,
        0.125939180544827,
        0.125939180544827,
        0.125939180544827,
    ],
    dtype=jnp.float64,
)


def fan_components(
    profile,
    vertices,
    count,
    moment_centre,
    sampling_centre,
    coordinate_scale,
    flux_coefficient,
):
    """Return direct moments and projection state for one padded support."""
    capacity = vertices.shape[0]
    slot = jnp.arange(capacity)
    valid = slot < count
    following_slot = jnp.where(slot + 1 < count, slot + 1, 0)
    following = vertices[following_slot]
    relative_vertex = vertices - sampling_centre
    relative_following = following - sampling_centre
    cross = (
        relative_vertex[:, 0] * relative_following[:, 1]
        - relative_following[:, 0] * relative_vertex[:, 1]
    )
    cross = jnp.where(valid, cross, 0.0)
    area_twice = jnp.sum(cross)
    fan_centre = sampling_centre + jnp.sum(
        (relative_vertex + relative_following) * cross[:, None], axis=0
    ) / (3.0 * area_twice)
    triangles = jnp.stack(
        [jnp.broadcast_to(fan_centre, vertices.shape), vertices, following], axis=1
    )
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    triangle_area = 0.5 * jnp.abs(
        first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    )
    triangle_area = jnp.where(valid, triangle_area, 0.0)
    points = jnp.einsum("qa,tad->tqd", BARYCENTRIC, triangles)
    weight = triangle_area[:, None] * RULE_WEIGHT[None]
    points = points.reshape(capacity * len(RULE_WEIGHT), 2)
    weight = weight.reshape(capacity * len(RULE_WEIGHT))
    points = jnp.where(weight[:, None] > 0.0, points, sampling_centre)
    local = (points - sampling_centre) / coordinate_scale
    design = _quadratic_flux_design(local)
    flux = design @ flux_coefficient
    density = profile.current_density(points[:, 0], flux)
    displacement = points - moment_centre
    direct = jnp.asarray(
        [
            jnp.sum(weight * density),
            jnp.sum(weight * density * displacement[:, 0]),
            jnp.sum(weight * density * displacement[:, 1]),
        ]
    )
    affine = design[:, :3]
    weighted = affine * weight[:, None]
    normal = affine.T @ weighted
    projection = jnp.linalg.solve(normal, weighted.T)
    coefficient = projection @ density
    return direct, normal, projection, coefficient, points, weight


def relative_sup(value, reference):
    """Return the largest componentwise relative difference."""
    return jnp.max(jnp.abs(value - reference) / jnp.maximum(jnp.abs(reference), 1e-30))


def fixed_density_moments(
    problem, support, cell, ring_slot, centroid_flux, sample_flux
):
    """Integrate one clip-independent quadratic fitted at fixed sample nodes."""
    stencil = problem["stencil"]
    profile = problem["profile"]
    centroid_density = profile.current_density(
        jnp.asarray(problem["mesh"].coordinate[:, 0]), centroid_flux
    )
    sample_density = profile.current_density(
        jnp.asarray(problem["geometry"].sample_node_coordinates[:, 0]), sample_flux
    )
    pool = jnp.concatenate([centroid_density, sample_density])
    gathered = pool[jnp.asarray(stencil.ring_gather_index[ring_slot])]
    coefficient = jnp.asarray(stencil.ring_flux_weight[ring_slot]) @ gathered
    polynomial = jnp.pad(coefficient, (0, len(POLYNOMIAL_POWERS) - len(coefficient)))[
        None
    ]
    current, first = padded_polynomial_current_moments(
        support.support_vertices[cell][None],
        support.vertex_count[cell][None],
        jnp.asarray(stencil.ring_sampling_centre[ring_slot])[None],
        jnp.asarray(stencil.ring_coordinate_scale[ring_slot])[None],
        polynomial,
    )
    moment_centre = support.centroids[cell]
    sampling_centre = jnp.asarray(stencil.ring_sampling_centre[ring_slot])
    first = first + current[:, None] * (sampling_centre - moment_centre)[None]
    return jnp.asarray([current[0], first[0, 0], first[0, 1]]), coefficient


def fixed_profile_moments(
    problem, support, cell, ring_slot, centroid_flux, sample_flux
):
    """Project the profile once on the full pre-clip cell and integrate it."""
    stencil = problem["stencil"]
    profile = problem["profile"]
    pool = jnp.concatenate([centroid_flux, sample_flux])
    gathered = pool[jnp.asarray(stencil.ring_gather_index[ring_slot])]
    flux_coefficient = jnp.asarray(stencil.ring_flux_weight[ring_slot]) @ gathered
    capacity = support.support_vertices.shape[1]
    authored = jnp.asarray(problem["geometry"].sampling_vertices[cell])
    full_vertices = jnp.pad(authored, ((0, capacity - len(authored)), (0, 0)))
    sampling_centre = jnp.asarray(stencil.ring_sampling_centre[ring_slot])
    coordinate_scale = jnp.asarray(stencil.ring_coordinate_scale[ring_slot])
    _direct, _normal, _projection, _coefficient, points, weight = fan_components(
        profile,
        full_vertices,
        jnp.asarray(len(authored)),
        jnp.asarray(support.centroids[cell]),
        sampling_centre,
        coordinate_scale,
        flux_coefficient,
    )
    local = (points - sampling_centre) / coordinate_scale
    basis = jnp.stack(
        [local[:, 0] ** p * local[:, 1] ** q for p, q in POLYNOMIAL_POWERS],
        axis=1,
    )
    weighted = basis * weight[:, None]
    projection = jnp.linalg.solve(basis.T @ weighted, weighted.T)
    flux = _quadratic_flux_design(local) @ flux_coefficient
    density = profile.current_density(points[:, 0], flux)
    coefficient = projection @ density
    current, first = padded_polynomial_current_moments(
        support.support_vertices[cell][None],
        support.vertex_count[cell][None],
        sampling_centre[None],
        coordinate_scale[None],
        coefficient[None],
    )
    moment_centre = support.centroids[cell]
    first = first + current[:, None] * (sampling_centre - moment_centre)[None]
    return jnp.asarray([current[0], first[0, 0], first[0, 1]]), coefficient


def fixed_field_ablation():
    """Compare projected output with identity fan moments near full fill."""
    problem = boundary_support_problem()
    atomic = problem["geometry"].atomic_mesh
    stencil = problem["stencil"]
    cell = int(problem["ring"][len(problem["ring"]) // 2])
    ring_slot = int(np.flatnonzero(stencil.ring_centre == cell)[0])
    polygon = np.asarray(problem["geometry"].polygons[cell])
    sampling_centre = jnp.asarray(stencil.ring_sampling_centre[ring_slot])
    coordinate_scale = jnp.asarray(stencil.ring_coordinate_scale[ring_slot])
    pool = jnp.concatenate([problem["centroid_flux"], problem["sample_flux"]])
    gathered = pool[jnp.asarray(stencil.ring_gather_index[ring_slot])]
    coefficient = jnp.asarray(stencil.ring_flux_weight[ring_slot]) @ gathered

    def identity(support):
        direct, normal, projection, projected_coefficient, _points, _weight = (
            fan_components(
                problem["profile"],
                support.support_vertices[cell],
                support.vertex_count[cell],
                jnp.asarray(support.centroids[cell]),
                sampling_centre,
                coordinate_scale,
                coefficient,
            )
        )
        return direct, normal, projection, projected_coefficient

    full_support = atomic.traced_clip(jnp.ones(len(atomic.node_coordinates)))
    full_production = jnp.stack(evaluate_boundary_support(problem, full_support))[
        :, cell
    ]
    full_identity, full_normal, full_projection, full_coefficient = identity(
        full_support
    )
    full_fixed, _full_fixed_coefficient = fixed_density_moments(
        problem,
        full_support,
        cell,
        ring_slot,
        problem["centroid_flux"],
        problem["sample_flux"],
    )
    full_profile, _full_profile_coefficient = fixed_profile_moments(
        problem,
        full_support,
        cell,
        ring_slot,
        problem["centroid_flux"],
        problem["sample_flux"],
    )
    print("fixed-field identity ablation")
    print("full projected-minus-identity", np.asarray(full_production - full_identity))
    width = float(np.ptp(polygon[:, 0]))
    maximum = float(np.max(polygon[:, 0]))
    for nominal in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6):
        support = atomic.traced_clip(
            maximum - width * nominal - atomic.node_coordinates[:, 0]
        )
        production = jnp.stack(evaluate_boundary_support(problem, support))[:, cell]
        direct, normal, projection, projected_coefficient = identity(support)
        fixed, _fixed_coefficient = fixed_density_moments(
            problem,
            support,
            cell,
            ring_slot,
            problem["centroid_flux"],
            problem["sample_flux"],
        )
        fixed_profile, _fixed_profile_coefficient = fixed_profile_moments(
            problem,
            support,
            cell,
            ring_slot,
            problem["centroid_flux"],
            problem["sample_flux"],
        )
        observed = 1.0 - support.area[cell] / support.full_area[cell]
        profile_error = relative_sup(fixed_profile, full_profile)
        identity_delta = jnp.max(jnp.abs(production - direct))
        projection_delta = jnp.max(jnp.abs(projection - full_projection))
        coefficient_delta = jnp.max(jnp.abs(projected_coefficient - full_coefficient))
        print(
            "missing",
            f"nominal={nominal:.1e}",
            f"observed={float(observed):.9e}",
            f"production_error={float(relative_sup(production, full_production)):.9e}",
            f"identity_error={float(relative_sup(direct, full_identity)):.9e}",
            f"fixed_density_error={float(relative_sup(fixed, full_fixed)):.9e}",
            f"fixed_profile_error={float(profile_error):.9e}",
            f"projected_minus_identity={float(identity_delta):.9e}",
            f"normal_delta={float(jnp.max(jnp.abs(normal - full_normal))):.9e}",
            f"projection_delta={float(projection_delta):.9e}",
            f"coefficient_delta={float(coefficient_delta):.9e}",
            f"flux_coefficient_delta={0.0:.1e}",
        )


def moving_field_jvp_ablation():
    """Compare projection and identity JVPs across the full-fill point."""
    profile_type = TEST["DomainProfile"]
    profile = profile_type(
        p_prime=lambda psi: -((1.0 - psi) ** 2),
        ff_prime=lambda psi: jnp.zeros_like(psi),
    )
    problem = boundary_support_problem(profile)
    atomic = problem["geometry"].atomic_mesh
    stencil = problem["stencil"]
    cell = int(problem["ring"][len(problem["ring"]) // 2])
    ring_slot = int(np.flatnonzero(stencil.ring_centre == cell)[0])
    centre = problem["mesh"].coordinate[cell]
    half_width = 0.5 * np.ptp(problem["geometry"].polygons[cell], axis=0)
    atomic_u = (atomic.node_coordinates[:, 0] - centre[0]) / half_width[0]
    sample_u = (
        problem["geometry"].sample_node_coordinates[:, 0] - centre[0]
    ) / half_width[0]
    centroid_u = (problem["mesh"].coordinate[:, 0] - centre[0]) / half_width[0]
    sampling_centre = jnp.asarray(stencil.ring_sampling_centre[ring_slot])
    coordinate_scale = jnp.asarray(stencil.ring_coordinate_scale[ring_slot])

    def components(cut):
        support = atomic.traced_clip(cut - jnp.asarray(atomic_u))
        centroid_flux = jnp.clip(1.0 - (cut - jnp.asarray(centroid_u)), 0.0, 1.0)
        sample_flux = jnp.clip(1.0 - (cut - jnp.asarray(sample_u)), 0.0, 1.0)
        pool = jnp.concatenate([centroid_flux, sample_flux])
        gathered = pool[jnp.asarray(stencil.ring_gather_index[ring_slot])]
        flux_coefficient = jnp.asarray(stencil.ring_flux_weight[ring_slot]) @ gathered
        direct, normal, projection, coefficient, _points, _weight = fan_components(
            profile,
            support.support_vertices[cell],
            support.vertex_count[cell],
            jnp.asarray(support.centroids[cell]),
            sampling_centre,
            coordinate_scale,
            flux_coefficient,
        )
        fixed, density_coefficient = fixed_density_moments(
            problem, support, cell, ring_slot, centroid_flux, sample_flux
        )
        profile_fixed, profile_coefficient = fixed_profile_moments(
            problem, support, cell, ring_slot, centroid_flux, sample_flux
        )
        raw_centroid_flux = 1.0 - (cut - jnp.asarray(centroid_u))
        raw_sample_flux = 1.0 - (cut - jnp.asarray(sample_u))
        raw_profile_fixed, raw_profile_coefficient = fixed_profile_moments(
            problem,
            support,
            cell,
            ring_slot,
            raw_centroid_flux,
            raw_sample_flux,
        )
        return (
            direct,
            flux_coefficient,
            normal,
            projection,
            coefficient,
            fixed,
            density_coefficient,
            profile_fixed,
            profile_coefficient,
            raw_profile_fixed,
            raw_profile_coefficient,
        )

    displacement = 2e-7
    print("moving-field JVP ablation")
    names = (
        "identity_moments",
        "flux_interpolant",
        "normal",
        "projection",
        "projected_density_coefficient",
        "fixed_density_moments",
        "fixed_density_coefficient",
        "fixed_profile_moments",
        "fixed_profile_coefficient",
        "raw_fixed_profile_moments",
        "raw_fixed_profile_coefficient",
    )
    for component, name in enumerate(names):

        def function(cut):
            return components(cut)[component]

        left_value, left_derivative = jax.jvp(
            function, (jnp.asarray(1.0 - displacement),), (jnp.asarray(1.0),)
        )
        right_value, right_derivative = jax.jvp(
            function, (jnp.asarray(1.0 + displacement),), (jnp.asarray(1.0),)
        )
        output = [
            name,
            "value_delta",
            float(jnp.max(jnp.abs(left_value - right_value))),
            "jvp_delta",
            float(jnp.max(jnp.abs(left_derivative - right_derivative))),
        ]
        if name in (
            "identity_moments",
            "fixed_density_moments",
            "fixed_profile_moments",
            "raw_fixed_profile_moments",
        ):
            output.extend(
                [
                    "left_jvp",
                    np.asarray(left_derivative),
                    "right_jvp",
                    np.asarray(right_derivative),
                ]
            )
        print(*output)


if __name__ == "__main__":
    fixed_field_ablation()
    moving_field_jvp_ablation()
