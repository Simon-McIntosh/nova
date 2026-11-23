"""Pin the brick evaluator's field scale against an independent point-dipole sum.

The closed-form brick tensor must reduce to the analytic field of a point
dipole as the brick shrinks against its distance, and the six-rotation
composition must reproduce a point-dipole sum over the same bricks. A
composition that emits eleven images instead of six — rotations for all six
sectors plus a mirror for five of them — overstates the vertical field by a
factor 1.32, which is the scale question this file guards.
"""

from __future__ import annotations

import numpy as np

from scripts.prototypes.iron_limb_fill.bricks import (
    SECTORS,
    brick_tensor,
    images,
    response,
)

MU0 = 4e-7 * np.pi
SIZE = 0.1
LIMB_MOMENT = np.array([0.0, 0.0, -1.46408762e5])


def _dipole_tensor(offset: np.ndarray, volume: float) -> np.ndarray:
    """Return H = T M for a point dipole of moment volume * M at offset."""
    radius = np.linalg.norm(offset, axis=-1)
    unit = offset / radius[..., None]
    return (
        volume
        / (4 * np.pi * radius**3)[..., None, None]
        * (3 * unit[..., :, None] * unit[..., None, :] - np.eye(3))
    )


def _identity():
    return [(np.eye(3), np.eye(3))]


def _rotations():
    """The six rotational images alone, for a sector spanning both halves."""
    frames = []
    for k in range(SECTORS):
        angle = 2 * np.pi * k / SECTORS
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        frames.append((rotation, rotation @ np.diag([1, 1, 1.0])))
    return frames


def _twelve():
    """The rotation and up-down mirror images, for a one-sided sector."""
    frames = []
    for k in range(SECTORS):
        angle = 2 * np.pi * k / SECTORS
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        for mirror in (1.0, -1.0):
            frames.append(
                (
                    rotation @ np.diag([1, 1, mirror]),
                    rotation @ np.diag([mirror, mirror, 1.0]),
                )
            )
    return frames


def _brick(offset, half):
    return brick_tensor(offset, half)


def _dipole(offset, half):
    return _dipole_tensor(offset, 8 * half.prod())


def _compose(tensor_of, target, centres, half, magnetisation, frames):
    """Sum frame @ T(frame^T target - centre) @ frame^T @ moment_map @ M."""
    total = np.zeros(3)
    for frame, moment_map in frames:
        offset = (target[None, :] @ frame) - centres
        total += np.einsum(
            "ab,nbc,cd,d->a",
            frame,
            tensor_of(offset, half),
            frame.T @ moment_map,
            magnetisation,
        )
    return MU0 * total


def _limb_slab():
    """A uniformly magnetised limb as a 0.1 m brick grid, both halves."""
    x = np.arange(4.3 + SIZE / 2, 5.5, SIZE)
    y = np.arange(-0.4 + SIZE / 2, 0.4, SIZE)
    z = np.arange(-2.4 + SIZE / 2, 2.4, SIZE)
    grid = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([axis.ravel() for axis in grid]), np.full(3, SIZE / 2)


def test_one_brick_falls_to_the_analytic_dipole_at_far_distance():
    half = np.full(3, 0.05)
    edge, volume = 2 * half[0], 8 * half.prod()
    relative = []
    for edges in (10.0, 30.0, 100.0):
        offset = np.array([[edges * edge, 0.0, 0.0]])
        brick = brick_tensor(offset, half)[0]
        analytic = _dipole_tensor(offset, volume)[0]
        relative.append(np.linalg.norm(brick - analytic) / np.linalg.norm(analytic))
    assert relative[0] < 1e-4, relative
    for previous, current, distance in zip(relative, relative[1:], (30.0, 100.0)):
        assert current <= previous * (10.0 / distance) ** 2, relative


def test_one_sector_without_images_matches_the_point_dipole():
    centres, half = _limb_slab()
    target = np.array([8.4667, 0.0, 0.6])
    brick = _compose(_brick, target, centres, half, LIMB_MOMENT, _identity())
    point = _compose(_dipole, target, centres, half, LIMB_MOMENT, _identity())
    assert abs(brick[2] - point[2]) <= 1e-5 * abs(point[2]), (brick, point)


def test_one_sector_with_six_rotations_matches_the_point_dipole():
    centres, half = _limb_slab()
    target = np.array([8.4667, 0.0, 0.6])
    brick = _compose(_brick, target, centres, half, LIMB_MOMENT, _rotations())
    point = _compose(_dipole, target, centres, half, LIMB_MOMENT, _rotations())
    assert abs(brick[2] - point[2]) <= 1e-5 * abs(point[2]), (brick, point)
    np.testing.assert_allclose(brick[2], 1.2194067e-3, rtol=1e-5)


def test_the_shipped_evaluator_matches_its_twelve_image_point_sum():
    centres, half = _limb_slab()
    target = np.array([8.4667, 0.0, 0.6])
    assert len(list(images())) == 2 * SECTORS
    moment = np.tile(LIMB_MOMENT, (len(centres), 1)).ravel()
    field = MU0 * (response(target[None, :], centres, half) @ moment).reshape(-1, 3)[0]
    point = _compose(_dipole, target, centres, half, LIMB_MOMENT, _twelve())
    assert abs(field[2] - point[2]) <= 1e-5 * abs(point[2]), (field, point)


def test_a_mirror_over_a_both_halves_sector_doubles_the_field():
    centres, half = _limb_slab()
    target = np.array([8.4667, 0.0, 0.6])
    six = _compose(_dipole, target, centres, half, LIMB_MOMENT, _rotations())
    twelve = _compose(_dipole, target, centres, half, LIMB_MOMENT, _twelve())
    assert abs(twelve[2] / six[2] - 2.0) < 1e-6, (six, twelve)
