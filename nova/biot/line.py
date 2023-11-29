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

    from nova.geometry.polyline import PolyLine

    coilset = CoilSet(available=["vtk"], delta=-1)

    theta = np.linspace(0, 6 * np.pi, 200)
    path = np.c_[2 * np.cos(theta), np.linspace(0, 3, len(theta)), 3 * np.sin(theta)]

    coilset.winding.insert({"e": [0, 0, 0.25, 0.2]}, path, part="pf")

    coilset.frame.vtkplot()

    polyline = PolyLine(path)
