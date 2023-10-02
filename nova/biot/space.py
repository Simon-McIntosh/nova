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
    transform: np.ndarray = field(init=False, repr=False)

    def initialize(self):
        """Build global to local coordinate mapping transfrom."""
        self.transform = np.zeros((len(self.frame), 3, 3))
        self.transform[..., 0] = self.normal
        self.transform[..., 1] = np.cross(self.axis, self.normal)
        self.transform[..., 2] = self.axis
        self.transform /= np.linalg.norm(self.transform, axis=1)[:, np.newaxis]

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
