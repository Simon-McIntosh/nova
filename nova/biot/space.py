"""Coordinate system transform methods for BiotFrame."""
from dataclasses import dataclass, field

import numpy as np

import nova.frame.metamethod as metamethod
from nova.frame.framelink import FrameLink


@dataclass
class Space(metamethod.Space):
    """Coordinate system transform methods."""

    name = "space"

    frame: FrameLink = field(repr=False)
    axes: np.ndarray = field(init=False, repr=False)
    origin: np.ndarray = field(init=False, repr=False)

    def initialize(self):
        """Build local coordinate axes."""
        self.axes = np.zeros((len(self.frame), 3, 3))
        self.axes[..., 0] = self.normal
        self.axes[..., 1] = np.cross(self.axis, self.normal)
        self.axes[..., 2] = self.axis
        self.axes /= np.linalg.norm(self.axes, axis=1)[:, np.newaxis]
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
        return np.einsum("ij,ijk->ik", points, self.axes)

    def to_global(self, points: np.ndarray):
        """Return point array (n, 3) mapped to global coordinate system."""
        return np.einsum("ij,ikj->ik", points, self.axes)
