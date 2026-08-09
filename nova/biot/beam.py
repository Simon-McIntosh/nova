"""Biot-Savart calculation for line segments."""

from dataclasses import dataclass, field

from functools import cached_property
from typing import ClassVar

import numpy as np

from nova.biot.matrix import Matrix
from nova.biot.polybow import section_corners
from nova.geometry.section import is_axis_aligned_rectangle


def _bounded_asinh_product(scale, numerator, denominator):
    """Return ``scale * asinh(numerator / denominator)`` at its zero limit."""
    scale, numerator, denominator = np.broadcast_arrays(scale, numerator, denominator)
    result = np.zeros(scale.shape, dtype=np.result_type(scale, numerator, denominator))
    active = (scale != 0.0) & (denominator != 0.0)
    result[active] = scale[active] * np.arcsinh(numerator[active] / denominator[active])
    return result


def _bounded_atan_product(scale, numerator, denominator):
    """Return ``scale * atan(numerator / denominator)`` at its zero limit."""
    scale, numerator, denominator = np.broadcast_arrays(scale, numerator, denominator)
    result = np.zeros(scale.shape, dtype=np.result_type(scale, numerator, denominator))
    active = (scale != 0.0) & (denominator != 0.0)
    result[active] = scale[active] * np.arctan(numerator[active] / denominator[active])
    return result


@dataclass
class Beam(Matrix):
    """
    Extend Biot base class.

    Compute interaction for 3d line elements with a finite cross-section.

    """

    axisymmetric: ClassVar[bool] = False
    name: ClassVar[str] = "beam"  # element name

    attrs: dict[str, str] = field(default_factory=lambda: {"dl": "dl"})

    def __post_init__(self):
        """Load intergration constants."""
        super().__post_init__()
        area = np.asarray(self.source("area"), dtype=np.float64)
        width = np.asarray(self.source("width"), dtype=np.float64)
        height = np.asarray(self.source("height"), dtype=np.float64)
        for name, values in (("area", area), ("width", width), ("height", height)):
            if not np.all(np.isfinite(values) & (values > 0.0)):
                raise ValueError(f"beam sources require finite positive {name}")
        if "poly" in self.source.columns:
            for poly in np.asarray(self.source["poly"], dtype=object):
                if not is_axis_aligned_rectangle(section_corners(poly)):
                    raise ValueError("beam sources require an axis-aligned rectangle")
        self.xs = np.stack(
            [
                self("source", "x2") + delta / 2 * self.source("width")
                for delta in [-1, 1]
            ],
        )
        self.ys = np.stack(
            [
                self("source", "y2") + delta / 2 * self.source("height")
                for delta in [-1, 1]
            ],
        )
        self.zs = np.stack([self("source", "z1"), self("source", "z2")])
        self.x = self("target", "x")
        self.y = self("target", "y")
        self.z = self("target", "z")

    @property
    def ui(self):
        """Return ui coefficent."""
        return (self.xs - self.x[np.newaxis])[:, np.newaxis, np.newaxis]

    @property
    def vj(self):
        """Return vi coefficent."""
        return (self.ys - self.y[np.newaxis])[np.newaxis, :, np.newaxis]

    @property
    def wk(self):
        """Return wi coefficent."""
        return (self.zs - self.z[np.newaxis])[np.newaxis, np.newaxis]

    @cached_property
    def alpha(self):
        """Return alpha_ijk coefficent."""
        return np.sqrt(self.ui**2 + self.vj**2)

    @cached_property
    def beta(self):
        """Return beta_ijk coefficent."""
        return np.sqrt(self.vj**2 + self.wk**2)

    @cached_property
    def gamma(self):
        """Return gamma_ijk coefficent."""
        return np.sqrt(self.wk**2 + self.ui**2)

    @cached_property
    def distance(self):
        """Return the distance from every target to every source corner."""
        return np.sqrt(self.ui**2 + self.vj**2 + self.wk**2)

    @cached_property
    def theta(self) -> dict[str, np.ndarray]:
        """Return theta coefficents 1-6."""
        return dict(
            zip(
                np.arange(1, 7),
                [
                    self.wk / self.alpha,
                    self.ui / self.beta,
                    self.vj / self.gamma,
                    self.vj * self.wk / (self.ui * self.distance),
                    self.wk * self.ui / (self.vj * self.distance),
                    self.ui * self.vj / (self.wk * self.distance),
                ],
            )
        )

    @cached_property
    def phi(self):
        """Return global target toroidal angle."""
        return np.arctan2(self.target("y"), self.target("x"))

    @property
    def _Ax_hat(self):
        return np.zeros((2, 2, 2) + self.shape)

    @property
    def _Ay_hat(self):
        return np.zeros((2, 2, 2) + self.shape)

    @property
    def _Az_hat(self):
        """Return stacked local z-coord vector potential intergration coefficents."""
        return (
            _bounded_asinh_product(self.ui * self.vj, self.wk, self.alpha)
            + _bounded_asinh_product(self.vj * self.wk, self.ui, self.beta)
            + _bounded_asinh_product(self.wk * self.ui, self.vj, self.gamma)
            - 0.5
            * (
                _bounded_atan_product(
                    self.ui**2, self.vj * self.wk, self.ui * self.distance
                )
                + _bounded_atan_product(
                    self.vj**2, self.wk * self.ui, self.vj * self.distance
                )
                + _bounded_atan_product(
                    self.wk**2, self.ui * self.vj, self.wk * self.distance
                )
            )
        )

    @property
    def _Bx_hat(self):
        """Return stacked local x-coord magnetic field intergration coefficents."""
        return (
            -_bounded_asinh_product(self.ui, self.wk, self.alpha)
            - _bounded_asinh_product(self.wk, self.ui, self.beta)
            + _bounded_atan_product(self.vj, self.wk * self.ui, self.vj * self.distance)
        )

    @property
    def _By_hat(self):
        """Return stacked local y-coord magnetic field intergration coefficents."""
        return (
            _bounded_asinh_product(self.vj, self.wk, self.alpha)
            + _bounded_asinh_product(self.wk, self.vj, self.gamma)
            - _bounded_atan_product(self.ui, self.vj * self.wk, self.ui * self.distance)
        )

    @property
    def _Bz_hat(self):
        return np.zeros((2, 2, 2) + self.shape)

    @cached_property
    def _sign(self):
        """Return intergrator sign."""
        i = j = k = np.arange(1, 3)
        return (-1) ** (
            i[:, np.newaxis, np.newaxis]
            + j[np.newaxis, :, np.newaxis]
            + k[np.newaxis, np.newaxis]
        )

    def _intergrate(self, data):
        """Return intergral quantity."""
        return (
            1
            / (4 * np.pi * self.source("area"))
            * np.einsum("ijk,ijk...", self._sign, data)
        )


if __name__ == "__main__":
    from nova.frame.coilset import CoilSet

    radius = 3.945
    height = 2
    segment_number = 3

    attr = "ay"
    factor = 0.3
    Ic = 5.3e5

    outer_width = 0.05
    inner_width = 0.04

    theta = np.linspace(0, 2 * np.pi, 1 + 3 * segment_number)
    points = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), height * np.ones_like(theta)],
        axis=-1,
    )

    coilset = CoilSet(field_attrs=["Ay", "Bx", "By", "Bz", "Br"])
    for i in range(segment_number):
        coilset.winding.insert(
            points[3 * i : 1 + 3 * (i + 1)],
            {"skin": (0, 0, outer_width, 1 - inner_width / outer_width)},
            nturn=1,
            minimum_arc_nodes=4,
            Ic=1,
            filament=False,
        )

    coilset.plot()

    coilset.point.solve(np.array([radius, height]))

    coilset.grid.solve(2500, factor)

    coilset.saloc["Ic"] = Ic

    levels = coilset.grid.plot(attr, colors="C0", levels=61)

    add = CoilSet()
    add += coilset

    axes = coilset.grid.axes

    cylinder = CoilSet(field_attrs=["Ay", "Bx", "By", "Bz", "Br"])
    cylinder.coil.insert({"rect": (radius, height, outer_width, outer_width)})
    cylinder.coil.insert({"rect": (radius, height, inner_width, inner_width)})
    # cylinder.linkframe(cylinder.frame.index, -1)

    Ashell = outer_width**2 - inner_width**2
    Jc = Ic / Ashell
    cylinder.grid.solve(2500, factor)
    cylinder.saloc["Ic"] = Jc * outer_width**2, -Jc * inner_width**2

    levels = cylinder.grid.plot(
        attr, levels=levels, colors="C1", axes=axes, linestyles="--"
    )

    # cylinder.plot()
