"""Cerfon--Freidberg lower-single-null Solov'ev equilibrium.

The dimensionless construction follows A. J. Cerfon and J. P. Freidberg,
Physics of Plasmas 17, 032502 (2010), equations 26--28.  The twelve
homogeneous basis functions are combined with the Solov'ev particular
solution by solving the paper's twelve boundary constraints in one linear
system.  Physical coordinates use ``R = R0 x`` and ``Z = R0 y``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, root


Array = np.ndarray


def _basis_values(x: Array, y: Array) -> Array:
    """Return the twelve homogeneous solutions in equations 8 and 27."""
    log_x = np.log(x)
    x2 = x**2
    x4 = x2**2
    x6 = x2**3
    y2 = y**2
    y4 = y2**2
    y6 = y2**3
    return np.stack(
        (
            np.ones_like(x),
            x2,
            y2 - x2 * log_x,
            x4 - 4.0 * x2 * y2,
            2.0 * y4 - 9.0 * y2 * x2 + 3.0 * x4 * log_x - 12.0 * x2 * y2 * log_x,
            x6 - 12.0 * x4 * y2 + 8.0 * x2 * y4,
            8.0 * y6
            - 140.0 * y4 * x2
            + 75.0 * y2 * x4
            - 15.0 * x6 * log_x
            + 180.0 * x4 * y2 * log_x
            - 120.0 * x2 * y4 * log_x,
            y,
            y * x2,
            y**3 - 3.0 * y * x2 * log_x,
            3.0 * y * x4 - 4.0 * y**3 * x2,
            8.0 * y**5
            - 45.0 * y * x4
            - 80.0 * y**3 * x2 * log_x
            + 60.0 * y * x4 * log_x,
        ),
        axis=-1,
    )


def _basis_dx(x: Array, y: Array) -> Array:
    """Return the radial derivative of every homogeneous basis function."""
    log_x = np.log(x)
    x2 = x**2
    x3 = x**3
    x4 = x2**2
    x5 = x**5
    y2 = y**2
    y3 = y**3
    y4 = y2**2
    return np.stack(
        (
            np.zeros_like(x),
            2.0 * x,
            -(2.0 * x * log_x + x),
            4.0 * x3 - 8.0 * x * y2,
            3.0 * x3 - 30.0 * x * y2 + 12.0 * x3 * log_x - 24.0 * x * y2 * log_x,
            6.0 * x5 - 48.0 * x3 * y2 + 16.0 * x * y4,
            -5.0
            * x
            * (
                3.0 * x4
                - 96.0 * x2 * y2
                + 80.0 * y4
                + 6.0 * (3.0 * x4 - 24.0 * x2 * y2 + 8.0 * y4) * log_x
            ),
            np.zeros_like(x),
            2.0 * x * y,
            -3.0 * x * y * (1.0 + 2.0 * log_x),
            12.0 * x3 * y - 8.0 * x * y3,
            40.0 * x * y * (-3.0 * x2 - 2.0 * y2 + (6.0 * x2 - 4.0 * y2) * log_x),
        ),
        axis=-1,
    )


def _basis_dy(x: Array, y: Array) -> Array:
    """Return the vertical derivative of every homogeneous basis function."""
    log_x = np.log(x)
    x2 = x**2
    x4 = x2**2
    y2 = y**2
    y3 = y**3
    y4 = y2**2
    return np.stack(
        (
            np.zeros_like(y),
            np.zeros_like(y),
            2.0 * y,
            -8.0 * x2 * y,
            2.0 * y * (-9.0 * x2 + 4.0 * y2 - 12.0 * x2 * log_x),
            -24.0 * x4 * y + 32.0 * x2 * y3,
            2.0
            * y
            * (
                75.0 * x4
                - 280.0 * x2 * y2
                + 24.0 * y4
                + 60.0 * (3.0 * x4 - 4.0 * x2 * y2) * log_x
            ),
            np.ones_like(y),
            x2,
            3.0 * (y2 - x2 * log_x),
            3.0 * (x4 - 4.0 * x2 * y2),
            5.0 * (-9.0 * x4 + 8.0 * y4 + 12.0 * (x4 - 4.0 * x2 * y2) * log_x),
        ),
        axis=-1,
    )


def _basis_dxx(x: Array, y: Array) -> Array:
    """Return the second radial derivative of every basis function."""
    log_x = np.log(x)
    x2 = x**2
    x4 = x2**2
    y2 = y**2
    y3 = y**3
    y4 = y2**2
    return np.stack(
        (
            np.zeros_like(x),
            2.0 * np.ones_like(x),
            -2.0 * log_x - 3.0,
            12.0 * x2 - 8.0 * y2,
            3.0 * (7.0 * x2 - 18.0 * y2 + 4.0 * (3.0 * x2 - 2.0 * y2) * log_x),
            2.0 * (15.0 * x4 - 72.0 * x2 * y2 + 8.0 * y4),
            -5.0
            * (
                33.0 * x4
                - 432.0 * x2 * y2
                + 128.0 * y4
                + 6.0 * (15.0 * x4 - 72.0 * x2 * y2 + 8.0 * y4) * log_x
            ),
            np.zeros_like(x),
            2.0 * y,
            -3.0 * y * (3.0 + 2.0 * log_x),
            36.0 * x2 * y - 8.0 * y3,
            -40.0 * y * (3.0 * x2 + 6.0 * y2 - 18.0 * x2 * log_x + 4.0 * y2 * log_x),
        ),
        axis=-1,
    )


def _basis_dyy(x: Array, y: Array) -> Array:
    """Return the second vertical derivative of every basis function."""
    log_x = np.log(x)
    x2 = x**2
    x4 = x2**2
    y2 = y**2
    y4 = y2**2
    return np.stack(
        (
            np.zeros_like(y),
            np.zeros_like(y),
            2.0 * np.ones_like(y),
            -8.0 * x2,
            -6.0 * (3.0 * x2 - 4.0 * y2 + 4.0 * x2 * log_x),
            -24.0 * x2 * (x2 - 4.0 * y2),
            30.0
            * (
                5.0 * x4
                - 56.0 * x2 * y2
                + 8.0 * y4
                + 12.0 * (x4 - 4.0 * x2 * y2) * log_x
            ),
            np.zeros_like(y),
            np.zeros_like(y),
            6.0 * y,
            -24.0 * x2 * y,
            160.0 * y * (y2 - 3.0 * x2 * log_x),
        ),
        axis=-1,
    )


def _basis_dxy(x: Array, y: Array) -> Array:
    """Return the mixed derivative of every homogeneous basis function."""
    log_x = np.log(x)
    x2 = x**2
    x3 = x**3
    y2 = y**2
    y3 = y**3
    return np.stack(
        (
            np.zeros_like(x),
            np.zeros_like(x),
            np.zeros_like(x),
            -16.0 * x * y,
            -12.0 * x * y * (5.0 + 4.0 * log_x),
            -96.0 * x3 * y + 64.0 * x * y3,
            x
            * y
            * (960.0 * x2 + 1440.0 * x2 * log_x - 1600.0 * y2 - 960.0 * y2 * log_x),
            np.zeros_like(x),
            2.0 * x,
            -3.0 * x * (1.0 + 2.0 * log_x),
            12.0 * x3 - 24.0 * x * y2,
            40.0 * x * (-3.0 * x2 - 6.0 * y2 + 6.0 * x2 * log_x - 12.0 * y2 * log_x),
        ),
        axis=-1,
    )


def _particular(x: Array, source_parameter: float) -> Array:
    x2 = x**2
    x4 = x2**2
    return x4 / 8.0 + source_parameter * (0.5 * x2 * np.log(x) - x4 / 8.0)


def _particular_dx(x: Array, source_parameter: float) -> Array:
    x2 = x**2
    return (
        0.5
        * x
        * (
            source_parameter
            + x2
            - source_parameter * x2
            + 2.0 * source_parameter * np.log(x)
        )
    )


def _particular_dxx(x: Array, source_parameter: float) -> Array:
    x2 = x**2
    return 1.5 * (
        source_parameter + x2 - source_parameter * x2
    ) + source_parameter * np.log(x)


@dataclass(frozen=True)
class CerfonFreidbergSingleNull:
    """Exact lower-single-null Solov'ev solution in physical coordinates."""

    major_radius: float = 1.70
    inverse_aspect_ratio: float = 0.32
    elongation: float = 1.7
    triangularity: float = 0.33
    source_parameter: float = -0.155
    flux_scale_per_radian_wb: float = -1.0 / (2.0 * np.pi)
    coefficients: Array = field(init=False, repr=False, compare=False)
    magnetic_axis: Array = field(init=False, repr=False, compare=False)
    x_point: Array = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.major_radius <= 0.0:
            raise ValueError("major radius must be positive")
        if not 0.0 < self.inverse_aspect_ratio < 1.0:
            raise ValueError("inverse aspect ratio must lie between zero and one")
        if self.elongation <= 0.0:
            raise ValueError("elongation must be positive")
        if not -1.0 < self.triangularity < 1.0:
            raise ValueError("triangularity must lie between minus one and one")

        coefficients = self._solve_coefficients()
        x_point = self.major_radius * np.array(
            (
                1.0 - 1.1 * self.triangularity * self.inverse_aspect_ratio,
                -1.1 * self.elongation * self.inverse_aspect_ratio,
            ),
            dtype=np.float64,
        )
        axis_result = root(
            lambda point: self._dimensionless_gradient(
                np.asarray(point[0]), np.asarray(point[1]), coefficients
            ),
            np.array((1.0, 0.0), dtype=np.float64),
            method="hybr",
        )
        if not axis_result.success:
            raise RuntimeError(
                f"analytic magnetic-axis solve failed: {axis_result.message}"
            )
        magnetic_axis = self.major_radius * np.asarray(axis_result.x, dtype=np.float64)
        for _ in range(3):
            normalized_axis = magnetic_axis / self.major_radius
            physical_gradient = (
                self.flux_scale_per_radian_wb
                / self.major_radius
                * self._dimensionless_gradient(
                    normalized_axis[0], normalized_axis[1], coefficients
                )
            )
            physical_hessian = self.hessian_with(coefficients, magnetic_axis)
            magnetic_axis = magnetic_axis - np.linalg.solve(
                physical_hessian, physical_gradient
            )
        axis_eigenvalues = np.linalg.eigvalsh(
            self.hessian_with(coefficients, magnetic_axis)
        )
        if not np.all(axis_eigenvalues < 0.0):
            raise RuntimeError("analytic magnetic-axis root is not a flux maximum")

        coefficients.setflags(write=False)
        magnetic_axis.setflags(write=False)
        x_point.setflags(write=False)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "magnetic_axis", magnetic_axis)
        object.__setattr__(self, "x_point", x_point)

    @property
    def minor_radius(self) -> float:
        """Return the requested geometric minor radius."""
        return self.major_radius * self.inverse_aspect_ratio

    @property
    def upper_point(self) -> Array:
        """Return the requested smooth upper shaping point."""
        return self.major_radius * np.array(
            (
                1.0 - self.triangularity * self.inverse_aspect_ratio,
                self.elongation * self.inverse_aspect_ratio,
            ),
            dtype=np.float64,
        )

    @property
    def inner_equatorial_point(self) -> Array:
        return np.array((self.major_radius - self.minor_radius, 0.0))

    @property
    def outer_equatorial_point(self) -> Array:
        return np.array((self.major_radius + self.minor_radius, 0.0))

    @property
    def axis_flux(self) -> float:
        return float(self.flux(self.magnetic_axis[None, :])[0])

    @property
    def pressure_coefficient(self) -> float:
        """Return the static Solov'ev pressure coefficient used by the source."""
        return -(
            self.flux_scale_per_radian_wb
            * (1.0 - self.source_parameter)
            / (4.0 * self.major_radius**4)
        )

    @property
    def field_coefficient(self) -> float:
        """Return the static Solov'ev field coefficient used by the source."""
        return -(
            self.flux_scale_per_radian_wb
            * self.source_parameter
            / (2.0 * self.major_radius**2)
        )

    def _solve_coefficients(self) -> Array:
        epsilon = self.inverse_aspect_ratio
        kappa = self.elongation
        delta = self.triangularity
        shaping_angle = np.arcsin(delta)
        curvature_outer = -((1.0 + shaping_angle) ** 2) / (epsilon * kappa**2)
        curvature_inner = ((1.0 - shaping_angle) ** 2) / (epsilon * kappa**2)
        curvature_upper = -kappa / (epsilon * np.cos(shaping_angle) ** 2)

        outer = np.array((1.0 + epsilon, 0.0))
        inner = np.array((1.0 - epsilon, 0.0))
        upper = np.array((1.0 - delta * epsilon, kappa * epsilon))
        x_point = np.array((1.0 - 1.1 * delta * epsilon, -1.1 * kappa * epsilon))

        rows: list[Array] = []
        right_hand_side: list[float] = []

        def append(
            point: Array, basis: Callable[[Array, Array], Array], particular: float
        ) -> None:
            x, y = np.asarray(point, dtype=np.float64)
            rows.append(np.asarray(basis(x, y), dtype=np.float64))
            right_hand_side.append(-float(particular))

        append(outer, _basis_values, _particular(outer[0], self.source_parameter))
        append(inner, _basis_values, _particular(inner[0], self.source_parameter))
        append(upper, _basis_values, _particular(upper[0], self.source_parameter))
        append(x_point, _basis_values, _particular(x_point[0], self.source_parameter))
        append(outer, _basis_dy, 0.0)
        append(inner, _basis_dy, 0.0)
        append(upper, _basis_dx, _particular_dx(upper[0], self.source_parameter))
        append(x_point, _basis_dx, _particular_dx(x_point[0], self.source_parameter))
        append(x_point, _basis_dy, 0.0)
        append(
            outer,
            lambda x, y: _basis_dyy(x, y) + curvature_outer * _basis_dx(x, y),
            curvature_outer * _particular_dx(outer[0], self.source_parameter),
        )
        append(
            inner,
            lambda x, y: _basis_dyy(x, y) + curvature_inner * _basis_dx(x, y),
            curvature_inner * _particular_dx(inner[0], self.source_parameter),
        )
        append(
            upper,
            lambda x, y: _basis_dxx(x, y) + curvature_upper * _basis_dy(x, y),
            _particular_dxx(upper[0], self.source_parameter),
        )
        return np.linalg.solve(np.stack(rows), np.asarray(right_hand_side))

    def _dimensionless_flux(
        self, x: Array, y: Array, coefficients: Array | None = None
    ) -> Array:
        coefficients = self.coefficients if coefficients is None else coefficients
        return (
            _particular(x, self.source_parameter) + _basis_values(x, y) @ coefficients
        )

    def _dimensionless_gradient(
        self, x: Array, y: Array, coefficients: Array | None = None
    ) -> Array:
        coefficients = self.coefficients if coefficients is None else coefficients
        radial = (
            _particular_dx(x, self.source_parameter) + _basis_dx(x, y) @ coefficients
        )
        vertical = _basis_dy(x, y) @ coefficients
        return np.stack((radial, vertical), axis=-1)

    def flux(self, points: Array) -> Array:
        """Return poloidal flux per radian at physical ``(R, Z)`` points."""
        points = np.asarray(points, dtype=np.float64)
        x = points[..., 0] / self.major_radius
        y = points[..., 1] / self.major_radius
        return self.flux_scale_per_radian_wb * self._dimensionless_flux(x, y)

    def gradient(self, points: Array) -> Array:
        """Return ``(dpsi/dR, dpsi/dZ)`` in physical coordinates."""
        points = np.asarray(points, dtype=np.float64)
        x = points[..., 0] / self.major_radius
        y = points[..., 1] / self.major_radius
        return (
            self.flux_scale_per_radian_wb
            / self.major_radius
            * self._dimensionless_gradient(x, y)
        )

    def hessian_with(self, coefficients: Array, points: Array) -> Array:
        points = np.asarray(points, dtype=np.float64)
        x = points[..., 0] / self.major_radius
        y = points[..., 1] / self.major_radius
        radial = (
            _particular_dxx(x, self.source_parameter) + _basis_dxx(x, y) @ coefficients
        )
        vertical = _basis_dyy(x, y) @ coefficients
        mixed = _basis_dxy(x, y) @ coefficients
        result = np.empty(points.shape[:-1] + (2, 2), dtype=np.float64)
        result[..., 0, 0] = radial
        result[..., 0, 1] = mixed
        result[..., 1, 0] = mixed
        result[..., 1, 1] = vertical
        return self.flux_scale_per_radian_wb / self.major_radius**2 * result

    def hessian(self, points: Array) -> Array:
        """Return the physical Hessian of the poloidal flux."""
        return self.hessian_with(self.coefficients, points)

    def grad_shafranov_source(self, points: Array) -> Array:
        """Return the exact strong-form ``DeltaStar(psi)`` source."""
        points = np.asarray(points, dtype=np.float64)
        x = points[..., 0] / self.major_radius
        return (
            self.flux_scale_per_radian_wb
            / self.major_radius**2
            * (self.source_parameter + (1.0 - self.source_parameter) * x**2)
        )

    def grad_shafranov_residual(self, points: Array) -> Array:
        """Return the analytic strong-form residual in physical coordinates."""
        points = np.asarray(points, dtype=np.float64)
        gradient = self.gradient(points)
        hessian = self.hessian(points)
        operator = (
            hessian[..., 0, 0] - gradient[..., 0] / points[..., 0] + hessian[..., 1, 1]
        )
        return operator - self.grad_shafranov_source(points)

    def separatrix(self, count: int = 721) -> Array:
        """Sample the closed core separatrix, including the prescribed X-point."""
        if count < 33:
            raise ValueError("separatrix sampling requires at least 33 points")
        axis = self.magnetic_axis / self.major_radius
        x_point = self.x_point / self.major_radius
        x_direction = x_point - axis
        x_angle = float(np.arctan2(x_direction[1], x_direction[0]))
        angles = x_angle + np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        radii = np.linspace(0.0, 2.5, 1001)
        boundary: list[Array] = []
        axis_value = float(self._dimensionless_flux(axis[0], axis[1]))
        if axis_value >= 0.0:
            raise RuntimeError("analytic axis does not lie inside the zero-flux lobe")

        for angle in angles:
            direction = np.array((np.cos(angle), np.sin(angle)))
            ray = axis + radii[:, None] * direction
            positive_radius = ray[:, 0] > 0.05
            values = np.full_like(radii, np.nan)
            values[positive_radius] = self._dimensionless_flux(
                ray[positive_radius, 0], ray[positive_radius, 1]
            )
            crossing = np.flatnonzero(
                np.isfinite(values[:-1])
                & np.isfinite(values[1:])
                & (values[:-1] < 0.0)
                & (values[1:] >= 0.0)
            )
            if len(crossing):
                lower = radii[crossing[0]]
                upper = radii[crossing[0] + 1]
                distance = brentq(
                    lambda value: float(
                        self._dimensionless_flux(*(axis + value * direction))
                    ),
                    lower,
                    upper,
                    xtol=1.0e-13,
                    rtol=1.0e-13,
                )
                boundary.append(axis + distance * direction)
                continue
            angle_error = abs(
                np.arctan2(np.sin(angle - x_angle), np.cos(angle - x_angle))
            )
            if angle_error < np.pi / count:
                boundary.append(x_point)
                continue
            raise RuntimeError(
                f"zero-flux lobe is not star-shaped at angle {angle:.6f}"
            )
        boundary[0] = x_point
        return self.major_radius * np.asarray(boundary, dtype=np.float64)


def cerfon_freidberg_single_null() -> CerfonFreidbergSingleNull:
    """Return the conventional-aspect-ratio lower-single-null reference."""
    return CerfonFreidbergSingleNull()
