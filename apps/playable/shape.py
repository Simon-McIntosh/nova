"""Commanded plasma shape and the key map that steps it.

A keyframe action is a change of one control parameter; the forward solve
re-solves from the previous equilibrium as a warm start.  The control
vocabulary is the pulse-design bounding-box set: bulk radius and height
through the geometric axis, elongation and the two triangularities through
the turning points, the X-point location through the null row, and the inner
and outer wall gaps through the gap rows.  Each key names one of those
controls and moves it by a stated signed step, and the commanded control-point
set is re-derived from the stepped parameters, so a key's pressed value is
exactly the control-point change it names.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

#: One key per enumerated control; each key applies a stated signed step.
#: The reverse direction of every control is also bound, so an operator can
#: correct an overshoot, but the control vocabulary is the signed sizes below.
STEPS: dict[str, float] = {
    "bulk_r": 0.02,  # geometric axis radius step, m
    "bulk_z": 0.01,  # geometric axis height step, m
    "elongation": 0.05,  # vertical stretch of the turning points
    "triangularity_upper": 0.02,  # upper point radial offset
    "triangularity_lower": 0.02,  # lower point radial offset
    "x_point_r": 0.02,  # X-point radius step, m
    "x_point_z": -0.01,  # X-point height step, m (down toward the divertor)
    "inner_gap": 0.005,  # inner point distance from the inner wall, m
    "outer_gap": 0.005,  # outer point distance from the outer wall, m
}


#: Control name to the :class:`PlasmaShape` field it steps.
PARAMETER_FIELD: dict[str, str] = {
    "bulk_r": "axis_r",
    "bulk_z": "axis_z",
    "elongation": "elongation",
    "triangularity_upper": "triangularity_upper",
    "triangularity_lower": "triangularity_lower",
    "x_point_r": "x_point_r",
    "x_point_z": "x_point_z",
    "inner_gap": "inner_gap",
    "outer_gap": "outer_gap",
}


def export_keys() -> tuple[str, ...]:
    """Return the one key press per control the gate names."""
    return tuple(STEPS)


def keymap() -> dict[str, tuple[str, float]]:
    """Return every bound key symbol and the (parameter, signed step) it names.

    The ``+`` symbol of each control applies the stated signed step — most
    controls step upward, the X-point height steps downward toward the
    divertor — and the ``-`` symbol applies its reverse, so an overshoot is
    always correctable: ``bulk_r+`` names ``("bulk_r", +0.02)``, ``x_point_z+``
    names ``("x_point_z", -0.01)``.
    """
    bindings: dict[str, tuple[str, float]] = {}
    for parameter, step in STEPS.items():
        bindings[f"{parameter}+"] = (parameter, step)
        bindings[f"{parameter}-"] = (parameter, -step)
    return bindings


@dataclass(frozen=True)
class PlasmaShape:
    """One commanded plasma shape in the pulse-design control vocabulary."""

    axis_r: float = 1.0
    axis_z: float = 0.0
    minor_radius: float = 0.2
    elongation: float = 1.3
    triangularity_upper: float = 0.05
    triangularity_lower: float = 0.1
    x_point_r: float = 0.95
    x_point_z: float = -0.31
    inner_gap: float = 0.02
    outer_gap: float = 0.02

    def apply(self, parameter: str, delta: float) -> "PlasmaShape":
        """Return the shape moved by ``delta`` along one named control."""
        if parameter not in STEPS:
            raise KeyError(f"unknown control {parameter!r}; choose from {tuple(STEPS)}")
        field = PARAMETER_FIELD[parameter]
        return replace(self, **{field: getattr(self, field) + delta})

    @property
    def x_point(self) -> np.ndarray:
        """Return the commanded X-point as a (2,) radius-height pair."""
        return np.array([self.x_point_r, self.x_point_z])

    @property
    def control_point_names(self) -> tuple[str, ...]:
        """Return the ordered semantic names of ``control_points``."""
        return ("outer", "upper", "inner", "lower", "x_point")

    def control_points(self) -> np.ndarray:
        """Return the commanded control-point set as a (2, N) array.

        The outer and inner points bound the small radius at the midplane,
        offset by the wall gaps; the upper and lower points carry the
        elongation and triangularities; the X-point rides last, NaN-free when
        a finite location is commanded (the profile is diverted) and dropped
        when absent, matching the pulse-design ``ControlPoints`` convention
        that a point equal to the origin is not a constraint.
        """
        radius, height = self.axis_r, self.axis_z
        half = self.minor_radius
        upper = np.array(
            [radius - half * self.triangularity_upper, height + half * self.elongation]
        )
        lower = np.array(
            [radius - half * self.triangularity_lower, height - half * self.elongation]
        )
        # A commanded wall gap moves the point away from its wall: the inner
        # point rides outboard of its base radius and the outer point inboard.
        inner = np.array([radius - half + self.inner_gap, height])
        outer = np.array([radius + half - self.outer_gap, height])
        points = np.c_[outer, upper, inner, lower]
        if np.all(np.isfinite(self.x_point)):
            points = np.c_[points, self.x_point]
        return points


def point_delta(
    shape: PlasmaShape, parameter: str, delta: float
) -> dict[str, np.ndarray]:
    """Return the exact commanded control-point change a key action names.

    The returned mapping names each affected control point and its (2,)
    displacement, mirroring ``control_points`` (outer, upper, inner, lower,
    x_point).  This is the quantity a key press commands and the value the
    gate asserts the session's commanded set takes on.
    """
    stepped = shape.apply(parameter, delta)
    before = {
        name: point
        for name, point in zip(shape.control_point_names, shape.control_points().T)
    }
    after = {
        name: point
        for name, point in zip(stepped.control_point_names, stepped.control_points().T)
    }
    return {name: after[name] - before[name] for name in before if name in after}


def key_help() -> str:
    """Return the operator help text listing every bound key and its step."""
    lines = ["<b>key map</b> — focus the poloidal view, then press:"]
    for key, (parameter, step) in sorted(keymap().items()):
        lines.append(f"{key}: {parameter} {step:+.2g}")
    return "<br>".join(lines)
