"""Biot-Savart calculation for line segments."""
from dataclasses import dataclass

# from functools import cached_property
from typing import ClassVar

import numpy as np

from nova.biot.matrix import Matrix


@dataclass
class Line(Matrix):
    """
    Extend Biot base class.

    Compute interaction for 3d line elements.

    """

    name: ClassVar[str] = "arc"  # element name


if __name__ == "__main__":
    from nova.frame.coilset import CoilSet

    radius = 3.945
    height = 2
    segment_number = 13

    theta = np.linspace(0, 2 * np.pi, 1 + 2 * segment_number)
    points = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), height * np.ones_like(theta)],
        axis=-1,
    )

    coilset = CoilSet(field_attrs=["Br"])
    coilset.winding.insert(
        points, {"c": (0, 0, 0.5)}, minimum_arc_nodes=len(points) + 1
    )

    coilset.grid.solve(2500, [1, 0.9 * radius, 0, 4])
