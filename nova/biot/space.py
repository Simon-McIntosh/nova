"""Coordinate system transform methods for BiotFrame."""
from dataclasses import dataclass, field

import numpy as np

import nova.frame.metamethod as metamethod
from nova.frame.framelink import FrameLink
from nova.graphics.plot import Plot3D


@dataclass
class Space(metamethod.Space, Plot3D):
    """Coordinate system transform methods."""

    name = "space"

    frame: FrameLink = field(repr=False)
    coordinate_axes: np.ndarray = field(init=False, repr=False)
    coordinate_origin: np.ndarray = field(init=False, repr=False)

    def initialize(self):
        """Build local coordinate axes."""
        self.coordinate_axes = np.zeros((len(self.frame), 3, 3))
        self.coordinate_axes[..., 0] = self.normal
        self.coordinate_axes[..., 1] = np.cross(self.axis, self.normal)
        self.coordinate_axes[..., 2] = self.axis
        self.coordinate_axes /= np.linalg.norm(self.coordinate_axes, axis=1)[
            :, np.newaxis
        ]
        self.origin = np.c_[
            self.frame.aloc["x"], self.frame.aloc["y"], self.frame.aloc["z"]
        ]

    @property
    def axis(self):
        """Return source element axis."""
        return np.c_[
            self.frame.aloc["ax"], self.frame.aloc["ay"], self.frame.aloc["az"]
        ]

    @property
    def normal(self):
        """Return source element normal."""
        return np.c_[
            self.frame.aloc["nx"], self.frame.aloc["ny"], self.frame.aloc["nz"]
        ]

    def to_local(self, points: np.ndarray):
        """Return point array (n, 3) mapped to local coordinate system."""
        return np.einsum("ij,ijk->ik", points, self.coordinate_axes)

    def to_global(self, points: np.ndarray):
        """Return point array (n, 3) mapped to global coordinate system."""
        return np.einsum("ij,ikj->ik", points, self.coordinate_axes)
