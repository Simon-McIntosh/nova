"""Biot-Savart calculation for line segments."""

from dataclasses import dataclass, field

from functools import cached_property
from typing import ClassVar

import numpy as np

from nova.biot.matrix import Matrix


def _asinh_ratio(magnitude, radius):
    """Return ``asinh(magnitude / radius)`` without forming a huge ratio."""
    result = np.empty_like(magnitude)
    direct = magnitude <= radius
    result[direct] = np.arcsinh(magnitude[direct] / radius[direct])
    result[~direct] = np.log(
        magnitude[~direct] + np.hypot(magnitude[~direct], radius[~direct])
    ) - np.log(radius[~direct])
    return result


@dataclass
class Line(Matrix):
    """
    Extend Biot base class.

    Compute interaction for 3d line elements.

    """

    axisymmetric: ClassVar[bool] = False
    name: ClassVar[str] = "line"  # element name

    attrs: dict[str, str] = field(default_factory=lambda: {"dl": "dl"})

    def __post_init__(self):
        """Validate that every source represents a finite, nonzero segment."""
        super().__post_init__()
        start = np.column_stack(
            [
                np.asarray(self.source[f"{coordinate}1"], dtype=np.float64)
                for coordinate in "xyz"
            ]
        )
        end = np.column_stack(
            [
                np.asarray(self.source[f"{coordinate}2"], dtype=np.float64)
                for coordinate in "xyz"
            ]
        )
        length = np.linalg.norm(end - start, axis=1)
        if not np.all(np.isfinite(length) & (length > 0.0)):
            raise ValueError("line segments require a finite positive length")

    @cached_property
    def phi(self):
        """Return global target toroidal angle."""
        return np.arctan2(self.target("y"), self.target("x"))

    @property
    def u2(self):
        """Return stacked u2 coefficient."""
        return np.stack([self("source", "x2") - self("target", "x") for _ in range(2)])

    @property
    def v2(self):
        """Return stacked v2 coefficient."""
        return np.stack([self("source", "y2") - self("target", "y") for _ in range(2)])

    @cached_property
    def wi(self):
        """Return stacked wi coefficient."""
        return np.stack(
            [self("source", f"z{i}") - self("target", "z") for i in range(1, 3)]
        )

    @cached_property
    def a2(self):
        """Return stacked a2 coefficient."""
        return np.sqrt(self.u2**2 + self.v2**2)

    @cached_property
    def ri(self):
        """Return stacked ri coefficient."""
        return np.sqrt(self.a2**2 + self.wi**2)

    @property
    def _Ax_hat(self):
        return np.zeros((2,) + self.shape)

    @property
    def _Ay_hat(self):
        return np.zeros((2,) + self.shape)

    @property
    def _Az_hat(self):
        """Return the stable definite local vector-potential coefficient."""
        radius = self.a2[0]
        first, second = self.wi
        same_sign = ((first > 0.0) & (second > 0.0)) | ((first < 0.0) & (second < 0.0))
        exterior_axis = (radius == 0.0) & same_sign
        singular_axis = (radius == 0.0) & ~same_sign
        same_sign_off_axis = (radius != 0.0) & same_sign
        crossing_off_axis = (radius != 0.0) & ~same_sign

        difference = np.empty(self.shape, dtype=np.float64)
        scale = np.maximum.reduce([radius, np.abs(first), np.abs(second)])
        normalized_radius = radius / scale
        normalized_first = first / scale
        normalized_second = second / scale
        normalized_first_distance = np.hypot(normalized_radius, normalized_first)
        normalized_second_distance = np.hypot(normalized_radius, normalized_second)

        argument = np.zeros(self.shape, dtype=np.float64)
        argument[same_sign_off_axis] = (
            normalized_second[same_sign_off_axis] - normalized_first[same_sign_off_axis]
        ) / (
            normalized_first_distance[same_sign_off_axis]
            + normalized_second_distance[same_sign_off_axis]
        )
        direct = same_sign_off_axis & (np.abs(argument) <= 0.5)
        rationalized = same_sign_off_axis & ~direct
        difference[direct] = 2.0 * np.arctanh(argument[direct])

        first_magnitude = np.abs(normalized_first[rationalized])
        second_magnitude = np.abs(normalized_second[rationalized])
        distance_sum = (
            normalized_first_distance[rationalized]
            + normalized_second_distance[rationalized]
        )
        magnitude_difference = (second_magnitude - first_magnitude) * (
            1.0 + (second_magnitude + first_magnitude) / distance_sum
        )
        difference[rationalized] = np.sign(first[rationalized]) * np.log1p(
            magnitude_difference
            / (normalized_first_distance[rationalized] + first_magnitude)
        )
        difference[crossing_off_axis] = np.sign(
            normalized_second[crossing_off_axis] - normalized_first[crossing_off_axis]
        ) * (
            _asinh_ratio(
                np.abs(normalized_second[crossing_off_axis]),
                normalized_radius[crossing_off_axis],
            )
            + _asinh_ratio(
                np.abs(normalized_first[crossing_off_axis]),
                normalized_radius[crossing_off_axis],
            )
        )
        difference[exterior_axis] = np.sign(first[exterior_axis]) * (
            np.log(np.abs(normalized_second[exterior_axis]))
            - np.log(np.abs(normalized_first[exterior_axis]))
        )
        difference[singular_axis] = np.inf
        return np.stack([np.zeros(self.shape), difference])

    @property
    def _Bx_hat(self):
        """Return the stable definite local x-field coefficient."""
        return np.stack([np.zeros(self.shape), self.v2[0] * self._field_coefficient])

    @property
    def _By_hat(self):
        """Return the stable definite local y-field coefficient."""
        return np.stack([np.zeros(self.shape), -self.u2[0] * self._field_coefficient])

    @cached_property
    def _field_coefficient(self):
        """Return the endpoint field bracket with its exterior-axis limit."""
        radius = self.a2[0]
        first, second = self.wi
        same_sign = ((first > 0.0) & (second > 0.0)) | ((first < 0.0) & (second < 0.0))
        singular_axis = (radius == 0.0) & ~same_sign
        literal = ~same_sign & ~singular_axis

        scale = np.maximum.reduce([radius, np.abs(first), np.abs(second)])
        normalized_radius = radius / scale
        normalized_first = first / scale
        normalized_second = second / scale
        first_distance = np.hypot(normalized_radius, normalized_first)
        second_distance = np.hypot(normalized_radius, normalized_second)

        coefficient = np.empty(self.shape, dtype=np.float64)
        dimensionless = np.empty(self.shape, dtype=np.float64)
        dimensionless[same_sign] = (
            (normalized_second[same_sign] - normalized_first[same_sign])
            * (normalized_second[same_sign] + normalized_first[same_sign])
            / (
                first_distance[same_sign]
                * second_distance[same_sign]
                * (
                    normalized_second[same_sign] * first_distance[same_sign]
                    + normalized_first[same_sign] * second_distance[same_sign]
                )
            )
        )
        dimensionless[literal] = (
            (
                normalized_second[literal] / second_distance[literal]
                - normalized_first[literal] / first_distance[literal]
            )
            / normalized_radius[literal]
            / normalized_radius[literal]
        )
        live = ~singular_axis
        coefficient[live] = dimensionless[live] / scale[live] / scale[live]
        coefficient[singular_axis] = np.nan
        return coefficient

    @property
    def _Bz_hat(self):
        return np.zeros((2,) + self.shape)

    def _intergrate(self, data):
        """Return intergral quantity."""
        return 1 / (4 * np.pi) * (data[1] - data[0])


if __name__ == "__main__":
    from nova.frame.coilset import CoilSet

    radius = 3.945
    height = 2
    segment_number = 51

    theta = np.linspace(0, 2 * np.pi, 1 + 2 * segment_number)
    points = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), height * np.ones_like(theta)],
        axis=-1,
    )

    coilset = CoilSet(field_attrs=["Br", "Ay"])
    coilset.winding.insert(
        points, {"c": (0, 0, 0.5)}, minimum_arc_nodes=len(points) + 1
    )
    coilset.grid.solve(2500, [1, 0.9 * radius, 0, 4])

    coilset.saloc["Ic"] = 5.3e5
    levels = coilset.grid.plot("ay", nulls=False, colors="C0")
    axes = coilset.grid.axes

    print(coilset.grid.br.max(), coilset.grid.br.min())

    circle_coilset = CoilSet(field_attrs=["Br", "Bz", "Aphi", "Ay"])
    circle_coilset.coil.insert({"c": (radius, height, 0.05)})
    circle_coilset.grid.solve(2500, [1, 0.9 * radius, 0, 4])
    circle_coilset.saloc["Ic"] = 5.3e5
    circle_coilset.grid.plot("ay", nulls=False, colors="C1", axes=axes, linestyles="--")

    print(circle_coilset.grid.br.max(), circle_coilset.grid.br.min())
