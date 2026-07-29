"""Toroidal averages of 3-D magnetic-field operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nova.biot.biotframe import Target
from nova.biot.solve import Solve


@dataclass(frozen=True)
class ToroidalAverage:
    """Mean cylindrical field and the non-axisymmetric residual magnitude."""

    radial: np.ndarray
    toroidal: np.ndarray
    vertical: np.ndarray
    residual_rms: np.ndarray


def toroidal_points(
    radius: np.ndarray, height: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Return flattened points and their toroidal angles."""
    radius, height = np.broadcast_arrays(
        np.asarray(radius, dtype=float), np.asarray(height, dtype=float)
    )
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    points = np.stack(
        [
            radius[..., None] * np.cos(angle),
            radius[..., None] * np.sin(angle),
            np.broadcast_to(height[..., None], radius.shape + (count,)),
        ],
        axis=-1,
    )
    return points.reshape(-1, 3), angle, radius.shape


def frame_magnetic_field(
    source,
    points: np.ndarray,
    *,
    turns: bool = True,
    reduce: bool = True,
) -> np.ndarray:
    """Cartesian magnetic field at points, per source ampere."""
    points = np.asarray(points, dtype=float)
    target = Target(
        {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]}, available=[]
    )
    solved = Solve(
        source,
        target,
        turns=[turns, False],
        reduce=[reduce, False],
        attrs=["Bx", "By", "Bz"],
    )
    return np.stack(
        [np.asarray(solved.data[name]) for name in ("Bx", "By", "Bz")], axis=-1
    )


def average_toroidal_field(
    evaluator,
    radius: np.ndarray,
    height: np.ndarray,
    *,
    count: int = 64,
) -> ToroidalAverage:
    """Average a Cartesian field evaluator on toroidal rings.

    The evaluator may return additional axes between point and Cartesian
    component, such as one column per source circuit.
    """
    points, angle, base_shape = toroidal_points(radius, height, count)
    field = np.asarray(evaluator(points))
    if field.shape[0] != len(points) or field.shape[-1] != 3:
        raise ValueError("field evaluator must return shape (point, ..., 3)")
    extra_shape = field.shape[1:-1]
    field = field.reshape(base_shape + (count,) + extra_shape + (3,))
    angle_shape = (1,) * len(base_shape) + (count,) + (1,) * len(extra_shape)
    cosine = np.cos(angle).reshape(angle_shape)
    sine = np.sin(angle).reshape(angle_shape)
    radial = field[..., 0] * cosine + field[..., 1] * sine
    toroidal = -field[..., 0] * sine + field[..., 1] * cosine
    vertical = field[..., 2]
    sample_axis = len(base_shape)
    means = np.stack(
        [
            radial.mean(axis=sample_axis),
            toroidal.mean(axis=sample_axis),
            vertical.mean(axis=sample_axis),
        ],
        axis=-1,
    )
    cylindrical = np.stack([radial, toroidal, vertical], axis=-1)
    expanded = np.expand_dims(means, axis=sample_axis)
    residual_rms = np.sqrt(
        np.mean(np.sum((cylindrical - expanded) ** 2, axis=-1), axis=sample_axis)
    )
    return ToroidalAverage(
        radial=means[..., 0],
        toroidal=means[..., 1],
        vertical=means[..., 2],
        residual_rms=residual_rms,
    )


def frame_toroidal_average(
    source,
    radius: np.ndarray,
    height: np.ndarray,
    *,
    count: int = 64,
    turns: bool = True,
    reduce: bool = True,
) -> ToroidalAverage:
    """Toroidally averaged field operator for a source frame."""

    def evaluator(points: np.ndarray) -> np.ndarray:
        return frame_magnetic_field(source, points, turns=turns, reduce=reduce)

    return average_toroidal_field(evaluator, radius, height, count=count)
