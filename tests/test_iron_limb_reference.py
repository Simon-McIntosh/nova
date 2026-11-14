"""Independent field identities and surface boundary-condition checks."""

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


def test_sphere_permeability_boundary_condition():
    # A triangulated sphere supplies a closed boundary independent of the circuit mesh.
    golden = (1 + np.sqrt(5)) / 2
    vertices = np.array(
        [(0, a, b * golden) for a in (-1, 1) for b in (-1, 1)]
        + [(a, b * golden, 0) for a in (-1, 1) for b in (-1, 1)]
        + [(b * golden, 0, a) for a in (-1, 1) for b in (-1, 1)]
    )
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    triangles = vertices[ConvexHull(vertices).simplices]
    for _ in range(3):
        refined = []
        for a, b, c in triangles:
            ab, bc, ca = a + b, b + c, c + a
            ab, bc, ca = (v / np.linalg.norm(v) for v in (ab, bc, ca))
            refined.extend(([a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]))
        triangles = np.array(refined)
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    reverse = np.sum(normals * triangles.mean(axis=1), axis=1) < 0
    triangles[reverse] = triangles[reverse, ::-1]

    def external(p):
        return np.tile([0.0, 0.0, MU0], (len(p), 1))

    model = SurfaceModel(Panels(triangles), external, rotations=1, reflect=False)
    charge, residual = model.solve(50.0)
    field = model.iron_field(np.zeros((1, 3)), charge)[0] / MU0 + [0.0, 0.0, 1.0]
    np.testing.assert_allclose(field, [0.0, 0.0, 3 / 52], atol=0.003)
    assert residual < 1e-9


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
