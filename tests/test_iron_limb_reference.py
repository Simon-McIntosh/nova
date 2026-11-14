"""Independent field identities and surface boundary-condition checks."""

import os

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from scripts.iron_limb_reference.surface import (
    MU0,
    Panels,
    SurfaceModel,
    circuit_panels,
    triangle_field,
)

from scripts.prototypes.iron_limb_fill.bricks import brick_tensor


def test_dipole_and_cube_demagnetisation():
    half = np.array([0.1, 0.15, 0.2])
    far = np.array([[3.0, 2.0, 1.5]])
    radius = np.linalg.norm(far)
    unit = far[0] / radius
    dipole = (
        8
        * half.prod()
        / (4 * np.pi * radius**3)
        * (3 * np.outer(unit, unit) - np.eye(3))
    )
    np.testing.assert_allclose(brick_tensor(far, half)[0], dipole, rtol=5e-3, atol=1e-9)
    cube = brick_tensor(np.zeros((1, 3)), np.full(3, 0.1))[0]
    np.testing.assert_allclose(cube, -np.eye(3) / 3, atol=1e-12)


def test_triangle_surface_integral_matches_brick_inside_and_outside():
    half = np.array([0.4, 0.3, 0.2])
    points = np.array(
        [[0.0, 0.0, 0.0], [0.1, -0.08, 0.03], [0.9, -0.4, 0.5], [3.0, 2.0, 1.0]]
    )
    response = np.zeros((len(points), 3, 3))
    for axis in range(3):
        first, second = (axis + 1) % 3, (axis + 2) % 3
        for side in (-1.0, 1.0):
            quad = np.zeros((4, 3))
            quad[:, axis] = side * half[axis]
            quad[:, first] = np.array([-1, 1, 1, -1]) * half[first]
            quad[:, second] = np.array([-1, -1, 1, 1]) * half[second]
            panels = Panels(quad[np.array([[0, 1, 2], [0, 2, 3]])])
            for p in range(2):
                for i, point in enumerate(points):
                    response[i, :, axis] += side * triangle_field(
                        point,
                        panels.vertices[p],
                        panels.normals[p],
                        panels.areas[p],
                        panels.lengths[p],
                        panels.outward[p],
                    )
    np.testing.assert_allclose(response, brick_tensor(points, half), atol=2e-14)


@pytest.fixture(scope="module")
def sphere_convergence():
    from scripts.iron_limb_reference.sphere import sphere_study

    return sphere_study(output=os.environ.get("NOVA_IRON_SPHERE_RECEIPT"))


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("permeability", [50.0, 200.0])
def test_sphere_permeability_converges(sphere_convergence, permeability):
    comparison = next(
        c for c in sphere_convergence["convergence"] if c["mu"] == permeability
    )
    assert len(comparison["errors"]) == 3
    assert np.all(np.diff(comparison["errors"]) < 0), comparison
    assert comparison["errors"][-1] < 0.01, comparison
    for row in sphere_convergence["rows"]:
        assert max(row["neutrality"]["body_charge_relative"], default=0.0) < 1e-12
        assert row["solid_angle_error"] < 1e-10


@pytest.mark.parametrize("annulus", [False, True])
def test_circuit_boundary_preserves_volume_and_outward_normals(annulus):
    panels = circuit_panels(0.7, annulus=annulus, arc_segments=8)
    volume = 4 * np.sum(panels.areas * np.sum(panels.centres * panels.normals, axis=1))
    column_volume = np.pi * 0.664**2 * 4.8
    if annulus:
        expected = (
            column_volume + np.pi * 6.0**2 * 2.4 + np.pi * (6.0**2 - 4.8**2) * 4.8
        )
    else:
        hub = 6 * 0.4**2 / np.tan(np.pi / 6)
        upper = hub + 6 * (6.0 - 0.4 / np.tan(np.pi / 6)) * 0.8
        expected = column_volume + 2.4 * upper + 6 * 0.8 * 1.2 * 4.8
    assert volume == pytest.approx(expected, rel=0.003)


def test_invalid_permeability_is_refused():
    vertices = np.array(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
    )
    triangles = vertices[ConvexHull(vertices).simplices]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    reverse = np.sum(normals * triangles.mean(axis=1), axis=1) < 0
    triangles[reverse] = triangles[reverse, ::-1]
    model = SurfaceModel(
        Panels(triangles), lambda p: np.ones_like(p), rotations=1, reflect=False
    )
    with pytest.raises(ValueError, match="permeability"):
        model.solve(1.0)


def test_open_surface_is_refused_by_solid_angle_identity():
    panels = Panels(np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]))
    with pytest.raises(ValueError, match="not closed"):
        SurfaceModel(panels, lambda p: np.ones_like(p), rotations=1, reflect=False)


def test_each_body_charge_is_deflated():
    vertices = np.array(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
    )
    triangles = vertices[ConvexHull(vertices).simplices]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    reverse = np.sum(normals * triangles.mean(axis=1), axis=1) < 0
    triangles[reverse] = triangles[reverse, ::-1]
    panels = Panels(np.concatenate((triangles, triangles + [5.0, 0.0, 0.0])))
    model = SurfaceModel(
        panels,
        lambda p: np.tile([0.0, 0.0, MU0], (len(p), 1)),
        rotations=1,
        reflect=False,
    )
    assert len(model.constraints) == 2
    # Excite the forbidden monopole independently on each closed body.
    for index, members in enumerate(model.constraints):
        model.rhs[members] += (index + 1) * 0.25
    charge, residual = model.solve(200.0)
    for members in model.constraints:
        assert abs(np.dot(panels.areas[members], charge[members])) < 1e-11
    assert residual < 1e-9
    assert model.last_solve["removed_rhs_relative"] > 0.1
    np.testing.assert_allclose(
        panels.areas @ model.interaction, 0.5 * panels.areas, atol=2e-14
    )


@pytest.mark.parametrize("spacing", [0.25, 0.18, 0.11])
def test_column_flux_clips_the_slab_exactly(spacing):
    from types import SimpleNamespace

    from scripts.iron_limb_reference.measure import axisymmetric_column_flux

    vertical = np.r_[np.arange(-0.3, 0.5, spacing), 0.5]
    cells = tuple(
        np.array([[0.05, low], [0.664, low], [0.664, high], [0.05, high]])
        for low, high in zip(vertical[:-1], vertical[1:])
    )
    model = SimpleNamespace(cells=SimpleNamespace(polygons=cells))
    mu, field = 200.0, 0.7
    state = np.r_[
        np.zeros(len(cells)), np.full(len(cells), field * (mu - 1) / (MU0 * mu))
    ]
    flux = axisymmetric_column_flux(model, state, mu)
    assert flux == pytest.approx(np.pi * (0.664**2 - 0.05**2) * field, rel=2e-14)


def test_near_filament_panel_drive_matches_area_integral():
    from numpy.polynomial.legendre import leggauss

    from scripts.iron_limb_reference.drive import FilamentDrive

    panels = Panels(
        np.array([[[0.664, -0.05, -0.1], [0.664, 0.05, -0.1], [0.664, 0.05, 0.1]]])
    )
    external = FilamentDrive(dict(r=[0.7], z=[0.03], amps=[1.0]))
    average = external.normal_average(panels, order=64)[0]
    node, weight = leggauss(48)
    node, weight = (node + 1) / 2, weight / 2
    u, v = np.meshgrid(node, node, indexing="ij")
    a, b, c = panels.vertices[0]
    points = a + u[..., None] * (b - a) + ((1 - u) * v)[..., None] * (c - a)
    normal = (external(points.reshape(-1, 3)) @ panels.normals[0]).reshape(u.shape)
    independent = 2 * np.sum(normal * (1 - u) * weight[:, None] * weight[None])
    assert average == pytest.approx(independent, rel=2e-8, abs=1e-14)
    centroid = external(panels.centres)[0] @ panels.normals[0]
    assert abs(centroid - average) > 0.05 * abs(average)


def test_slab_field_integral_preserves_constant_flux():
    from types import SimpleNamespace

    from scripts.iron_limb_reference.measure import checked_column_flux

    model = SimpleNamespace(
        external=lambda p: np.tile([0.0, 0.0, 0.02], (len(p), 1)),
        iron_field=lambda p, q: np.zeros_like(p),
    )
    flux, quadrature = checked_column_flux(model, None, 50.0, bore=0.05)
    expected = np.pi * (0.664**2 - 0.05**2)
    assert flux == pytest.approx(expected, rel=2e-14)
    assert quadrature["relative_change"] < 1e-13
